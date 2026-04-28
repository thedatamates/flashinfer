from __future__ import annotations

import argparse
import json
import statistics

import torch

import flashinfer
from flashinfer import SfLayout
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
    parser.add_argument("--pv-tactic", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if args.head_dim % 16 != 0:
        raise ValueError("head_dim must be divisible by 16 for NVFP4 block scales.")
    if args.kv_len % 16 != 0:
        raise ValueError("kv_len must be divisible by 16 for NVFP4 block scales.")

    torch.manual_seed(0)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16

    value_t = torch.randn(
        (args.head_dim, args.kv_len),
        device=device,
        dtype=dtype,
    )
    value_global = _global_scale(value_t)
    value_fp4_t, value_scale_t = flashinfer.nvfp4_quantize(
        value_t,
        value_global,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    prob_global = torch.tensor([448.0 * 6.0], device=device, dtype=torch.float32)
    pv_alpha = 1.0 / (prob_global * value_global)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    runner = gemm_base.get_cutlass_fp4_gemm_module(
        *torch.cuda.get_device_capability(device)
    ).cutlass_fp4_gemm_runner()

    def run_fp4_pv(
        prob_fp4: torch.Tensor,
        prob_scale: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        runner(
            inputs=[
                prob_fp4,
                value_fp4_t.T,
                prob_scale,
                value_scale_t.T,
                pv_alpha,
                out.dtype,
                out,
                16,
                True,
                workspace,
            ],
            tactic=args.pv_tactic,
        )

    results: dict[str, object] = {
        "q_len": args.q_len,
        "kv_len": args.kv_len,
        "head_dim": args.head_dim,
        "pv_tactic": args.pv_tactic,
        "groups": {},
    }

    group = 1
    while group <= args.max_group_size:
        grouped_logits = torch.randn(
            (args.q_len * group, args.kv_len),
            device=device,
            dtype=dtype,
        )
        grouped_prob_fp4, grouped_prob_scale = flashinfer.nvfp4_softmax_quantize(
            grouped_logits,
            prob_global,
        )
        grouped_out = torch.empty(
            (args.q_len * group, args.head_dim),
            device=device,
            dtype=dtype,
        )

        separate_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _ in range(group):
            logits = torch.randn((args.q_len, args.kv_len), device=device, dtype=dtype)
            prob_fp4, prob_scale = flashinfer.nvfp4_softmax_quantize(
                logits,
                prob_global,
            )
            out = torch.empty((args.q_len, args.head_dim), device=device, dtype=dtype)
            separate_inputs.append((prob_fp4, prob_scale, out))

        def run_grouped() -> None:
            run_fp4_pv(grouped_prob_fp4, grouped_prob_scale, grouped_out)

        def run_separate() -> None:
            for prob_fp4, prob_scale, out in separate_inputs:
                run_fp4_pv(prob_fp4, prob_scale, out)

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
        flops = 2 * args.q_len * group * args.kv_len * args.head_dim
        results["groups"][str(group)] = {
            "grouped": _summary(grouped_samples),
            "separate": _summary(separate_samples),
            "grouped_tflops_min_ms": flops / (min(grouped_samples) / 1000) / 1e12,
            "separate_tflops_min_ms": flops / (min(separate_samples) / 1000) / 1e12,
            "grouped_speedup": min(separate_samples) / min(grouped_samples),
        }
        group *= 2

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
