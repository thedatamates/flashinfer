from __future__ import annotations

import argparse
import json
import math
import statistics

import torch
import torch.nn.functional as F

import flashinfer
from flashinfer import SfLayout
from flashinfer.fp4_quantization import _select_nvfp4_softmax_quant_threads
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


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    # Matches the scale convention used by FlashInfer's NVFP4 GEMM examples.
    return ((448 * 6) / x.float().abs().nan_to_num().max()).reshape(1).float()


def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = _global_scale(x)
    packed, block_scale = _quantize_with_scale(x, scale)
    return packed, block_scale, scale


def _quantize_with_scale(
    x: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    packed, block_scale = flashinfer.nvfp4_quantize(
        x,
        scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    return packed, block_scale


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048, help="Query rows.")
    parser.add_argument("--n", type=int, default=8192, help="KV rows.")
    parser.add_argument("--d", type=int, default=512, help="Head dimension.")
    parser.add_argument("--backend", type=str, default="cutlass", choices=["cutlass"])
    parser.add_argument("--qk-tactic", type=int, default=2)
    parser.add_argument("--pv-tactic", type=int, default=1)
    parser.add_argument(
        "--softmax-quant-threads",
        type=int,
        default=0,
        help="Threads per row for fused softmax quantization. 0 uses shape-dependent dispatch.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--skip-two-stage",
        action="store_true",
        help="Only benchmark QK. Useful for very large logits.",
    )
    args = parser.parse_args()

    if args.d % 16 != 0:
        raise ValueError("d must be divisible by 16 for NVFP4 block scales.")
    if args.n % 16 != 0:
        raise ValueError("n must be divisible by 16 for the prototype PV quantization.")

    torch.manual_seed(0)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16

    q = torch.randn((args.m, args.d), dtype=dtype, device=device)
    k = torch.randn((args.n, args.d), dtype=dtype, device=device)
    v_t = torch.randn((args.d, args.n), dtype=dtype, device=device)

    q4, q_scale, q_global = _quantize(q)
    k4, k_scale, k_global = _quantize(k)
    v4_t, v_scale_t, v_global = _quantize(v_t)
    qk_alpha = 1.0 / (q_global * k_global)
    attn_qk_alpha = qk_alpha / math.sqrt(args.d)
    softmax_quant_threads = args.softmax_quant_threads
    if softmax_quant_threads == 0:
        softmax_quant_threads = _select_nvfp4_softmax_quant_threads(args.m, args.n)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    runner = gemm_base.get_cutlass_fp4_gemm_module(
        *torch.cuda.get_device_capability(device)
    ).cutlass_fp4_gemm_runner()

    def run_fp4_mm(
        a4: torch.Tensor,
        b4_t: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale_t: torch.Tensor,
        alpha: torch.Tensor,
        out: torch.Tensor,
        tactic: int,
    ) -> None:
        runner(
            inputs=[
                a4,
                b4_t,
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

    logits_fp4 = torch.empty((args.m, args.n), dtype=dtype, device=device)
    logits_bf16 = torch.empty_like(logits_fp4)

    def run_fp4_qk() -> None:
        run_fp4_mm(
            q4,
            k4.T,
            q_scale,
            k_scale.T,
            qk_alpha,
            logits_fp4,
            args.qk_tactic,
        )

    def run_bf16_qk() -> None:
        torch.mm(q, k.T, out=logits_bf16)

    qk_fp4_samples = _event_ms(run_fp4_qk, warmup=args.warmup, repeat=args.repeat)
    qk_bf16_samples = _event_ms(run_bf16_qk, warmup=args.warmup, repeat=args.repeat)

    torch.cuda.synchronize()
    ref = q @ k.T
    err = (logits_fp4.float() - ref.float()).abs()
    ref_abs = ref.float().abs().clamp_min(1e-6)

    result: dict[str, object] = {
        "backend": args.backend,
        "m": args.m,
        "n": args.n,
        "d": args.d,
        "qk_tactic": args.qk_tactic,
        "pv_tactic": args.pv_tactic,
        "softmax_quant_threads": softmax_quant_threads,
        "qk_fp4": _summary(qk_fp4_samples),
        "qk_bf16_torch": _summary(qk_bf16_samples),
        "qk_fp4_tflops_min_ms": (2 * args.m * args.n * args.d)
        / (min(qk_fp4_samples) / 1000)
        / 1e12,
        "qk_bf16_tflops_min_ms": (2 * args.m * args.n * args.d)
        / (min(qk_bf16_samples) / 1000)
        / 1e12,
        "qk_mean_abs_error": float(err.mean()),
        "qk_max_abs_error": float(err.max()),
        "qk_mean_relative_error": float((err / ref_abs).mean()),
        "qk_cosine_similarity": float(
            F.cosine_similarity(ref.reshape(-1), logits_fp4.reshape(-1), dim=0)
        ),
    }

    if not args.skip_two_stage:
        logits = torch.empty_like(logits_fp4)
        probs = torch.empty_like(logits_fp4)
        out_fp4 = torch.empty((args.m, args.d), dtype=dtype, device=device)
        out_fp4_fused = torch.empty_like(out_fp4)
        out_bf16 = torch.empty_like(out_fp4)
        p_global = torch.tensor([448.0 * 6.0], dtype=torch.float32, device=device)
        pv_alpha = 1.0 / (p_global * v_global)
        p4_holder: list[tuple[torch.Tensor, torch.Tensor] | None] = [None]
        p4_fused_holder: list[tuple[torch.Tensor, torch.Tensor] | None] = [None]

        def run_softmax() -> None:
            torch.softmax(logits_fp4, dim=-1, out=probs)

        def run_p_quantize() -> None:
            p4_holder[0] = _quantize_with_scale(probs, p_global)

        def run_fused_softmax_quantize() -> None:
            p4_fused_holder[0] = flashinfer.nvfp4_softmax_quantize(
                logits_fp4,
                p_global,
                num_threads=softmax_quant_threads,
            )

        def run_fp4_pv() -> None:
            p4, p_scale = p4_holder[0]  # type: ignore[misc]
            run_fp4_mm(
                p4,
                v4_t.T,
                p_scale,
                v_scale_t.T,
                pv_alpha,
                out_fp4,
                args.pv_tactic,
            )

        def run_fp4_pv_fused() -> None:
            p4, p_scale = p4_fused_holder[0]  # type: ignore[misc]
            run_fp4_mm(
                p4,
                v4_t.T,
                p_scale,
                v_scale_t.T,
                pv_alpha,
                out_fp4_fused,
                args.pv_tactic,
            )

        def run_two_stage() -> None:
            run_fp4_mm(
                q4,
                k4.T,
                q_scale,
                k_scale.T,
                attn_qk_alpha,
                logits,
                args.qk_tactic,
            )
            torch.softmax(logits, dim=-1, out=probs)
            p4, p_scale = _quantize_with_scale(probs, p_global)
            run_fp4_mm(
                p4,
                v4_t.T,
                p_scale,
                v_scale_t.T,
                pv_alpha,
                out_fp4,
                args.pv_tactic,
            )

        def run_two_stage_fused_softmax_quant() -> None:
            run_fp4_mm(
                q4,
                k4.T,
                q_scale,
                k_scale.T,
                attn_qk_alpha,
                logits,
                args.qk_tactic,
            )
            p4, p_scale = flashinfer.nvfp4_softmax_quantize(
                logits,
                p_global,
                num_threads=softmax_quant_threads,
            )
            run_fp4_mm(
                p4,
                v4_t.T,
                p_scale,
                v_scale_t.T,
                pv_alpha,
                out_fp4_fused,
                args.pv_tactic,
            )

        def run_bf16_attention() -> None:
            torch.mm(torch.softmax((q @ k.T) / math.sqrt(args.d), dim=-1), v_t.T, out=out_bf16)

        run_fp4_mm(
            q4,
            k4.T,
            q_scale,
            k_scale.T,
            attn_qk_alpha,
            logits_fp4,
            args.qk_tactic,
        )
        run_softmax()
        run_p_quantize()
        run_fused_softmax_quantize()
        torch.cuda.synchronize()

        softmax_samples = _event_ms(run_softmax, warmup=args.warmup, repeat=args.repeat)
        p_quant_samples = _event_ms(run_p_quantize, warmup=args.warmup, repeat=args.repeat)
        fused_softmax_quant_samples = _event_ms(
            run_fused_softmax_quantize,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        run_p_quantize()
        run_fused_softmax_quantize()
        torch.cuda.synchronize()
        pv_fp4_samples = _event_ms(run_fp4_pv, warmup=args.warmup, repeat=args.repeat)
        pv_fp4_fused_samples = _event_ms(
            run_fp4_pv_fused,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        two_stage_samples = _event_ms(run_two_stage, warmup=args.warmup, repeat=args.repeat)
        two_stage_fused_samples = _event_ms(
            run_two_stage_fused_softmax_quant,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        bf16_attention_samples = _event_ms(
            run_bf16_attention,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        run_two_stage()
        run_two_stage_fused_softmax_quant()
        run_bf16_attention()
        torch.cuda.synchronize()
        attn_err = (out_fp4.float() - out_bf16.float()).abs()
        attn_fused_err = (out_fp4_fused.float() - out_bf16.float()).abs()
        attn_ref_abs = out_bf16.float().abs().clamp_min(1e-6)

        result.update(
            {
                "softmax": _summary(softmax_samples),
                "p_quantize": _summary(p_quant_samples),
                "softmax_p_quantize_fused": _summary(fused_softmax_quant_samples),
                "pv_fp4": _summary(pv_fp4_samples),
                "pv_fp4_after_fused_p": _summary(pv_fp4_fused_samples),
                "two_stage_fp4_unfused": _summary(two_stage_samples),
                "two_stage_fp4_fused_softmax_quant": _summary(two_stage_fused_samples),
                "bf16_attention_torch_unfused": _summary(bf16_attention_samples),
                "pv_fp4_tflops_min_ms": (2 * args.m * args.n * args.d)
                / (min(pv_fp4_samples) / 1000)
                / 1e12,
                "attention_cosine_similarity": float(
                    F.cosine_similarity(
                        out_bf16.reshape(-1),
                        out_fp4.reshape(-1),
                        dim=0,
                    )
                ),
                "attention_mean_abs_error": float(attn_err.mean()),
                "attention_max_abs_error": float(attn_err.max()),
                "attention_mean_relative_error": float((attn_err / attn_ref_abs).mean()),
                "attention_fused_p_cosine_similarity": float(
                    F.cosine_similarity(
                        out_bf16.reshape(-1),
                        out_fp4_fused.reshape(-1),
                        dim=0,
                    )
                ),
                "attention_fused_p_mean_abs_error": float(attn_fused_err.mean()),
                "attention_fused_p_max_abs_error": float(attn_fused_err.max()),
                "attention_fused_p_mean_relative_error": float(
                    (attn_fused_err / attn_ref_abs).mean()
                ),
            }
        )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
