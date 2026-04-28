from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

import flashinfer
from flashinfer import SfLayout
from flashinfer.prefill import trtllm_fmha_v2_prefill
import flashinfer.gemm.gemm_base as gemm_base


def _event_ms(fn, *, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "samples_ms": samples,
    }


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    return ((448 * 6) / x.float().abs().nan_to_num().max()).reshape(1).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-len", type=int, default=512)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--max-group-size", type=int, default=16)
    parser.add_argument("--qk-tactic", type=int, default=2)
    parser.add_argument("--pv-tactic", type=int, default=1)
    parser.add_argument("--include-bf16-production", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if args.head_dim % 16 != 0:
        raise ValueError("head_dim must be divisible by 16 for NVFP4 block scales.")
    if args.kv_len % 16 != 0:
        raise ValueError("kv_len must be divisible by 16 for NVFP4 block scales.")

    torch.manual_seed(0)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16

    key = torch.randn((args.kv_len, args.head_dim), device=device, dtype=dtype)
    value = torch.randn((args.kv_len, args.head_dim), device=device, dtype=dtype)
    value_t = value.T.contiguous()
    key_global = _global_scale(key)
    value_global = _global_scale(value_t)
    key_fp4, key_scale = flashinfer.nvfp4_quantize(
        key,
        key_global,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    value_fp4_t, value_scale_t = flashinfer.nvfp4_quantize(
        value_t,
        value_global,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )

    prob_global = torch.tensor([448.0 * 6.0], device=device, dtype=torch.float32)
    pv_alpha = 1.0 / (prob_global * value_global)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    bf16_workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    seq_lens = torch.tensor([args.kv_len], device=device, dtype=torch.int32)
    cum_seq_lens_q = torch.tensor([0, args.q_len], device=device, dtype=torch.int32)
    cum_seq_lens_kv = torch.tensor([0, args.kv_len], device=device, dtype=torch.int32)
    runner = gemm_base.get_cutlass_fp4_gemm_module(
        *torch.cuda.get_device_capability(device)
    ).cutlass_fp4_gemm_runner()

    def run_fp4_mm(
        a: torch.Tensor,
        b_t: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale_t: torch.Tensor,
        alpha: torch.Tensor,
        out: torch.Tensor,
        tactic: int,
    ) -> None:
        runner(
            inputs=[
                a,
                b_t,
                a_scale,
                b_scale_t,
                alpha,
                out.dtype,
                out,
                16,
                True,
                workspace,
            ],
            tactic=tactic,
        )

    def run_one_attention(
        q: torch.Tensor,
        q_global: torch.Tensor,
        qk_alpha: torch.Tensor,
        logits: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        q_fp4, q_scale = flashinfer.nvfp4_quantize(
            q,
            q_global,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
        )
        run_fp4_mm(
            q_fp4,
            key_fp4.T,
            q_scale,
            key_scale.T,
            qk_alpha,
            logits,
            args.qk_tactic,
        )
        prob_fp4, prob_scale = flashinfer.nvfp4_softmax_quantize(
            logits,
            prob_global,
        )
        run_fp4_mm(
            prob_fp4,
            value_fp4_t.T,
            prob_scale,
            value_scale_t.T,
            pv_alpha,
            out,
            args.pv_tactic,
        )

    results: dict[str, object] = {
        "q_len": args.q_len,
        "kv_len": args.kv_len,
        "head_dim": args.head_dim,
        "qk_tactic": args.qk_tactic,
        "pv_tactic": args.pv_tactic,
        "groups": {},
    }

    group = 1
    while group <= args.max_group_size:
        q_grouped = torch.randn(
            (args.q_len * group, args.head_dim),
            device=device,
            dtype=dtype,
        )
        q_grouped_global = _global_scale(q_grouped)
        qk_grouped_alpha = (1.0 / (q_grouped_global * key_global)) / math.sqrt(
            args.head_dim
        )
        logits_grouped = torch.empty(
            (args.q_len * group, args.kv_len),
            device=device,
            dtype=dtype,
        )
        out_grouped = torch.empty(
            (args.q_len * group, args.head_dim),
            device=device,
            dtype=dtype,
        )

        separate_inputs = []
        for _ in range(group):
            q = torch.randn((args.q_len, args.head_dim), device=device, dtype=dtype)
            q_global = _global_scale(q)
            qk_alpha = (1.0 / (q_global * key_global)) / math.sqrt(args.head_dim)
            logits = torch.empty((args.q_len, args.kv_len), device=device, dtype=dtype)
            out = torch.empty((args.q_len, args.head_dim), device=device, dtype=dtype)
            separate_inputs.append((q, q_global, qk_alpha, logits, out))

        q_bf16 = torch.randn(
            (args.q_len, group, args.head_dim),
            device=device,
            dtype=dtype,
        )
        out_bf16 = torch.empty_like(q_bf16)
        key_bf16 = key.reshape(args.kv_len, 1, args.head_dim).contiguous()
        value_bf16 = value.reshape(args.kv_len, 1, args.head_dim).contiguous()

        def run_grouped() -> None:
            run_one_attention(
                q_grouped,
                q_grouped_global,
                qk_grouped_alpha,
                logits_grouped,
                out_grouped,
            )

        def run_separate() -> None:
            for q, q_global, qk_alpha, logits, out in separate_inputs:
                run_one_attention(q, q_global, qk_alpha, logits, out)

        def run_bf16_production() -> None:
            trtllm_fmha_v2_prefill(
                (q_bf16, key_bf16, value_bf16),
                "SEPARATE_Q_K_V",
                workspace_buffer=bf16_workspace,
                seq_lens=seq_lens,
                max_q_len=args.q_len,
                max_kv_len=args.kv_len,
                bmm1_scale=1.0 / math.sqrt(args.head_dim),
                bmm2_scale=1.0,
                batch_size=1,
                cum_seq_lens_q=cum_seq_lens_q,
                cum_seq_lens_kv=cum_seq_lens_kv,
                out=out_bf16,
                out_dtype=dtype,
                mask_mode="padding",
            )

        grouped_samples = _event_ms(
            run_grouped,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        separate_samples = _event_ms(
            run_separate,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        group_result = {
            "grouped": _summary(grouped_samples),
            "separate": _summary(separate_samples),
            "grouped_speedup": min(separate_samples) / min(grouped_samples),
        }
        if args.include_bf16_production:
            bf16_samples = _event_ms(
                run_bf16_production,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            group_result["bf16_production"] = _summary(bf16_samples)
            group_result["grouped_vs_bf16"] = min(bf16_samples) / min(grouped_samples)
        results["groups"][str(group)] = group_result
        group *= 2

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.include_bf16_production:
        print("\nmarkdown")
        print(
            "| group | fused FP4 min ms | fused FP4 grouped min ms | BF16 production min ms | grouped vs FP4 | grouped vs BF16 |"
        )
        print("|---:|---:|---:|---:|---:|---:|")
        for group, result in results["groups"].items():
            separate = result["separate"]["min_ms"]
            grouped = result["grouped"]["min_ms"]
            bf16 = result["bf16_production"]["min_ms"]
            print(
                f"| {group} | {separate:.4f} | {grouped:.4f} | {bf16:.4f} | "
                f"{separate / grouped:.2f}x | {bf16 / grouped:.2f}x |"
            )


if __name__ == "__main__":
    main()
