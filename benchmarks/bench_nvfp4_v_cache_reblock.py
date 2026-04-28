from __future__ import annotations

import argparse
import json
import math
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


def _gemm_global_scale(x: torch.Tensor) -> torch.Tensor:
    return ((448 * 6) / x.float().abs().nan_to_num().max()).reshape(1).float()


def _kv_global_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.float().abs().nan_to_num().max() / 448).reshape(1).float()


def _quantize_cutlass_layout(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return flashinfer.nvfp4_quantize(
        x,
        global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the cost of reblocking vLLM-style NVFP4 V cache scales into "
            "the K-major scale layout required by native block-scaled FP4 PV GEMM."
        )
    )
    parser.add_argument("--m", type=int, default=2048, help="Rows in P / query tile batch.")
    parser.add_argument("--n", type=int, default=512, help="Head dimension.")
    parser.add_argument("--k", type=int, default=8192, help="KV sequence length.")
    parser.add_argument("--tactic", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if args.n % 16 != 0 or args.k % 16 != 0:
        raise ValueError("n and k must be divisible by 16 for NVFP4 block scales.")

    torch.manual_seed(0)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16

    p = torch.softmax(torch.randn((args.m, args.k), dtype=dtype, device=device), dim=-1)
    v = torch.randn((args.k, args.n), dtype=dtype, device=device)

    p_global = torch.tensor([448.0 * 6.0], dtype=torch.float32, device=device)
    p4, p_scale = _quantize_cutlass_layout(p, p_global)

    # This is the vLLM KV-cache scale orientation: rows are tokens and scale
    # groups are contiguous head-dim blocks.
    v_cache_global = _kv_global_scale(v).to(device)
    v_cache4, v_cache_scale = flashinfer.nvfp4_kv_quantize(v, v_cache_global)

    # This is the layout native FP4 PV GEMM needs for B=V^T: rows are output
    # columns and scale groups are contiguous sequence/K blocks.
    v_t = torch.empty((args.n, args.k), dtype=dtype, device=device)
    v4_t: torch.Tensor | None = None
    v_scale_t: torch.Tensor | None = None
    v_native_global = _gemm_global_scale(v.T.contiguous()).to(device)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    runner = gemm_base.get_cutlass_fp4_gemm_module(
        *torch.cuda.get_device_capability(device)
    ).cutlass_fp4_gemm_runner()
    out_fp4 = torch.empty((args.m, args.n), dtype=dtype, device=device)
    out_bf16 = torch.empty_like(out_fp4)

    def run_bf16_pv() -> None:
        torch.mm(p, v, out=out_bf16)

    def run_dequant_transpose() -> None:
        dense_v = flashinfer.nvfp4_kv_dequantize(
            v_cache4, v_cache_scale, v_cache_global, output_dtype=dtype
        )
        v_t.copy_(dense_v.T)

    def run_quantize_reblocked_v() -> None:
        nonlocal v4_t, v_scale_t
        v4_t, v_scale_t = _quantize_cutlass_layout(v_t, v_native_global)

    def run_reblock_pipeline() -> None:
        run_dequant_transpose()
        run_quantize_reblocked_v()

    run_reblock_pipeline()
    assert v4_t is not None and v_scale_t is not None
    pv_alpha = 1.0 / (p_global * v_native_global)

    def run_fp4_pv_reblocked() -> None:
        assert v4_t is not None and v_scale_t is not None
        runner(
            inputs=[
                p4,
                v4_t.T,
                p_scale,
                v_scale_t.T,
                pv_alpha,
                out_fp4.dtype,
                out_fp4,
                16,
                True,
                workspace,
            ],
            tactic=args.tactic,
        )

    def run_full_reblock_plus_fp4_pv() -> None:
        run_reblock_pipeline()
        run_fp4_pv_reblocked()

    bf16_samples = _event_ms(run_bf16_pv, warmup=args.warmup, repeat=args.repeat)
    dequant_transpose_samples = _event_ms(
        run_dequant_transpose, warmup=args.warmup, repeat=args.repeat
    )
    reblock_samples = _event_ms(run_reblock_pipeline, warmup=args.warmup, repeat=args.repeat)
    fp4_pv_samples = _event_ms(run_fp4_pv_reblocked, warmup=args.warmup, repeat=args.repeat)
    full_samples = _event_ms(
        run_full_reblock_plus_fp4_pv, warmup=args.warmup, repeat=args.repeat
    )

    torch.cuda.synchronize()
    ref = p @ v
    err = (out_fp4.float() - ref.float()).abs()
    result = {
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "tactic": args.tactic,
        "bf16_pv": _summary(bf16_samples),
        "dequant_transpose_v_cache": _summary(dequant_transpose_samples),
        "dequant_transpose_plus_requant_v": _summary(reblock_samples),
        "fp4_pv_after_reblock": _summary(fp4_pv_samples),
        "full_reblock_plus_fp4_pv": _summary(full_samples),
        "bf16_pv_tflops_min_ms": (2 * args.m * args.n * args.k)
        / (min(bf16_samples) / 1000)
        / 1e12,
        "fp4_pv_tflops_min_ms": (2 * args.m * args.n * args.k)
        / (min(fp4_pv_samples) / 1000)
        / 1e12,
        "full_reblock_plus_fp4_pv_effective_tflops_min_ms": (
            2 * args.m * args.n * args.k
        )
        / (min(full_samples) / 1000)
        / 1e12,
        "fp4_vs_bf16_pv_speedup": min(bf16_samples) / min(fp4_pv_samples),
        "full_reblock_vs_bf16_pv_speedup": min(bf16_samples) / min(full_samples),
        "mean_abs_error": float(err.mean()),
        "max_abs_error": float(err.max()),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(ref.reshape(-1), out_fp4.reshape(-1), dim=0)
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
