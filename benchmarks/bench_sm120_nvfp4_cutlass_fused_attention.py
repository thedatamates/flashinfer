from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

import flashinfer
from flashinfer import SfLayout
import flashinfer.gemm.gemm_base as gemm_base

from bench_sm120_nvfp4_ref_attention import (
    E2M1_VALUES,
    GROUP,
    HEAD_DIM,
    PROB_GLOBAL_SCALE,
    Q_LEN,
    fp32_to_nvfp4_rowmajor,
    make_inputs,
    nvfp4_rowmajor_to_fp32,
)


def global_scale(x: torch.Tensor) -> torch.Tensor:
    return ((448.0 * 6.0) / x.float().abs().nan_to_num().max()).reshape(1).float()


def quantize_cutlass(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = global_scale(x)
    packed, block_scale = flashinfer.nvfp4_quantize(
        x,
        scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    return packed, block_scale, scale


def make_shape_inputs(
    device: torch.device,
    *,
    q_len: int,
    group: int,
    kv_len: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    q = (torch.randn((q_len, group, head_dim), device=device, generator=gen) / 4).to(
        torch.bfloat16
    )
    k_bf16 = (torch.randn((kv_len, head_dim), device=device, generator=gen) / 4).to(
        torch.bfloat16
    )
    v_bf16 = (torch.randn((kv_len, head_dim), device=device, generator=gen) / 4).to(
        torch.bfloat16
    )
    k, k_scales = fp32_to_nvfp4_rowmajor(k_bf16)
    v, v_scales = fp32_to_nvfp4_rowmajor(v_bf16)
    return q.contiguous(), k, v, k_scales, v_scales


def event_ms(fn, *, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "min_ms": min(samples),
        "mean_ms": sum(samples) / len(samples),
        "max_ms": max(samples),
    }


def build_extension(head_dim: int = HEAD_DIM):
    root = Path(__file__).resolve().parents[1]
    cutlass_root = Path(os.environ.get("CUTLASS_ROOT", root / "3rdparty" / "cutlass"))
    d256_tile_policy = os.environ.get("SM120_D256_TILE_POLICY", "64x128x128")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.2")
    extra_cuda_cflags = [
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-lineinfo",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "-gencode=arch=compute_120f,code=sm_120f",
        "-DENABLE_BF16",
        "-DENABLE_FP4",
        "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
    ]
    if maxrregcount := os.environ.get("SM120_NVFP4_MAXRREGCOUNT"):
        extra_cuda_cflags.append(f"-maxrregcount={maxrregcount}")
    if head_dim == 128:
        source_name = "sm120_nvfp4_cutlass_fused_attention_d128.cu"
        extension_name = "sm120_nvfp4_cutlass_fused_attention_d128_ext"
    elif head_dim == 256:
        d256_min_blocks = int(os.environ.get("SM120_D256_MIN_BLOCKS_PER_SM", "1"))
        d256_logits_row_skew = int(os.environ.get("SM120_D256_LOGITS_ROW_SKEW", "4"))
        try:
            d256_tile_m, d256_tile_n, d256_tile_k = (
                int(part) for part in d256_tile_policy.split("x")
            )
        except ValueError as exc:
            raise ValueError(
                "SM120_D256_TILE_POLICY must look like 64x128x128"
            ) from exc
        extra_cuda_cflags.extend(
            [
                f"-DSM120_D256_TILE_M={d256_tile_m}",
                f"-DSM120_D256_TILE_N={d256_tile_n}",
                f"-DSM120_D256_TILE_K={d256_tile_k}",
                f"-DSM120_D256_MIN_BLOCKS_PER_SM={d256_min_blocks}",
                f"-DSM120_D256_LOGITS_ROW_SKEW={d256_logits_row_skew}",
            ]
        )
        source_name = "sm120_nvfp4_cutlass_fused_attention_d256.cu"
        extension_name = (
            "sm120_nvfp4_cutlass_fused_attention_d256_"
            f"{d256_tile_m}_{d256_tile_n}_{d256_tile_k}_mb{d256_min_blocks}"
            f"_lskew{d256_logits_row_skew}"
        )
    elif head_dim == 512:
        d512_mma_owns_softmax = int(os.environ.get("SM120_D512_MMA_OWNS_SOFTMAX", "1"))
        d512_logits_row_skew = int(os.environ.get("SM120_D512_LOGITS_ROW_SKEW", "4"))
        d512_softmax_threads = int(
            os.environ.get("SM120_D512_SOFTMAX_THREADS_PER_ROW", "2")
        )
        d512_min_blocks = int(os.environ.get("SM120_D512_MIN_BLOCKS_PER_SM", "1"))
        extra_cuda_cflags.extend(
            [
                f"-DSM120_D512_MMA_OWNS_SOFTMAX={d512_mma_owns_softmax}",
                f"-DSM120_D512_LOGITS_ROW_SKEW={d512_logits_row_skew}",
                f"-DSM120_D512_SOFTMAX_THREADS_PER_ROW={d512_softmax_threads}",
                f"-DSM120_D512_MIN_BLOCKS_PER_SM={d512_min_blocks}",
            ]
        )
        source_name = "sm120_nvfp4_cutlass_fused_attention.cu"
        extension_name = (
            "sm120_nvfp4_cutlass_fused_attention_ext"
            f"_mmaown{d512_mma_owns_softmax}"
            f"_lskew{d512_logits_row_skew}"
            f"_sth{d512_softmax_threads}"
            f"_mb{d512_min_blocks}"
        )
    else:
        raise ValueError("head_dim must be one of {128, 256, 512}")
    return load(
        name=extension_name,
        sources=[
            str(root / "benchmarks" / source_name),
            str(root / "benchmarks" / "sm120_nvfp4_cutlass_runner_bf16_inst.cu"),
        ],
        extra_include_paths=[
            str(root / "include"),
            str(cutlass_root / "include"),
            str(cutlass_root / "tools" / "util" / "include"),
        ],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=["-O3", "-std=c++17", "-DNDEBUG"],
        verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--runner-check", action="store_true")
    parser.add_argument("--full-runner-bench", action="store_true")
    parser.add_argument("--smem-atom-check", action="store_true")
    parser.add_argument("--smem-atom-check-only", action="store_true")
    parser.add_argument("--smem-atom-block-check-only", action="store_true")
    parser.add_argument("--pv-smem-atom-tile-check-only", action="store_true")
    parser.add_argument("--sm120-qkv-online-check-only", action="store_true")
    parser.add_argument("--sm120-qkv-online-full-grid-bench", action="store_true")
    parser.add_argument("--sm120-qkv-online-splitkv-full-grid-bench", action="store_true")
    parser.add_argument("--sm120-qkv-online-splitkv-reuse2-full-grid-bench", action="store_true")
    parser.add_argument("--sm120-qkv-online-splitkv-reuse4-full-grid-bench", action="store_true")
    parser.add_argument("--sm120-d256-manual-compact-splitkv-full-grid-bench", action="store_true")
    parser.add_argument("--sm120-d256-manual-compact-online-splitkv-full-grid-bench", action="store_true")
    parser.add_argument("--sm120-d256-manual-compact-register-o-splitkv-full-grid-bench", action="store_true")
    parser.add_argument("--sm120-d256-manual-qk-check-only", action="store_true")
    parser.add_argument("--sm120-d256-manual-qk-smem-check-only", action="store_true")
    parser.add_argument("--sm120-d256-manual-pv-check-only", action="store_true")
    parser.add_argument("--sm120-d256-manual-pv-smem-check-only", action="store_true")
    parser.add_argument("--sm120-d256-manual-fused-tile-check-only", action="store_true")
    parser.add_argument("--sm120-d256-manual-fused-tile-smem-check-only", action="store_true")
    parser.add_argument("--sm120-d256-manual-fused-tile-compact-check-only", action="store_true")
    parser.add_argument("--sm120-role-schedule-check-only", action="store_true")
    parser.add_argument("--online-kv-tiles", type=int, default=2)
    parser.add_argument("--split-kv-len", type=int, default=1024)
    parser.add_argument("--smem-atom-data-mode", type=int, default=0)
    parser.add_argument("--smem-atom-scale-mode", type=int, default=0)
    parser.add_argument("--smem-atom-ones", action="store_true")
    parser.add_argument("--smem-atom-unit-scales", action="store_true")
    parser.add_argument("--smem-atom-q-code", type=int, default=-1)
    parser.add_argument("--smem-atom-k-code", type=int, default=-1)
    parser.add_argument("--q-len", type=int, default=Q_LEN)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--group", type=int, default=GROUP)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    default_shape = (
        args.q_len == Q_LEN
        and args.group == GROUP
        and args.kv_len == 32768
        and args.head_dim == HEAD_DIM
    )
    if default_shape:
        q, k, v, k_scales, v_scales = make_inputs(device)
    else:
        q, k, v, k_scales, v_scales = make_shape_inputs(
            device,
            q_len=args.q_len,
            group=args.group,
            kv_len=args.kv_len,
            head_dim=args.head_dim,
        )
    ext = build_extension(args.head_dim)
    metadata = dict(ext.cutlass_sm120_blockscaled_collective_metadata())

    def compare(name: str, actual: torch.Tensor, ref: torch.Tensor) -> dict[str, object]:
        delta = (actual.float() - ref.float()).abs()
        cos = torch.sum(actual.float() * ref.float()) / torch.clamp(
            torch.linalg.vector_norm(actual.float()) * torch.linalg.vector_norm(ref.float()),
            min=1.0e-20,
        )
        return {
            f"{name}_finite": bool(torch.isfinite(actual).all().item()),
            f"{name}_mean_abs": float(delta.mean().item()),
            f"{name}_max_abs": float(delta.max().item()),
            f"{name}_cosine": float(cos.item()),
        }

    if args.sm120_role_schedule_check_only:
        marker = torch.empty(16, device=device, dtype=torch.int32)
        ext.sm120_nvfp4_role_schedule_smoke(marker)
        torch.cuda.synchronize()
        values = marker.cpu().tolist()
        role_schedule = dict(metadata["sm120_role_schedule"])
        print(
            {
                "sm120_role_schedule_smoke": True,
                "role_warp_counts": {
                    "softmax0": values[0],
                    "softmax1": values[1],
                    "correction": values[2],
                    "mma": values[3],
                    "load": values[4],
                    "epilogue": values[5],
                    "empty": values[6],
                },
                "total_warps": values[7],
                "mma_threads": values[8],
                "total_threads": values[9],
                "mma_warp_begin": values[10],
                "load_warp": values[11],
                "epilogue_warp": values[12],
                "metadata_role_schedule": role_schedule,
                "storage_bytes": metadata["sm120_qkv_load_collective_storage_bytes"],
                "storage_margin_bytes": metadata[
                    "sm120_qkv_load_collective_storage_margin_bytes"
                ],
                "bf16_logits_128x128_bytes": metadata[
                    "bf16_logits_128x128_bytes"
                ],
                "independent_double_buffered_logits_128x128_margin_bytes": metadata[
                    "independent_double_buffered_logits_128x128_margin_bytes"
                ],
                "independent_double_buffered_logits_64x128_margin_bytes": metadata[
                    "independent_double_buffered_logits_64x128_margin_bytes"
                ],
            }
        )
        return

    if args.sm120_d256_manual_qk_check_only:
        if args.head_dim != 256:
            raise ValueError("--sm120-d256-manual-qk-check-only requires --head-dim 256")
        if args.q_len * args.group < 64:
            raise ValueError("manual D256 QK check requires at least 64 flattened Q rows")
        if args.kv_len < 128:
            raise ValueError("manual D256 QK check requires at least 128 KV rows")
        q_rows = args.q_len * args.group
        q_rowmajor, q_rowmajor_scales = fp32_to_nvfp4_rowmajor(
            q.reshape(q_rows, args.head_dim)
        )
        q_ref_f32 = nvfp4_rowmajor_to_fp32(q_rowmajor, q_rowmajor_scales)
        k_ref_f32 = nvfp4_rowmajor_to_fp32(k, k_scales)
        out_tile = torch.empty((64, 128), device=device, dtype=torch.float32)
        ext.sm120_d256_manual_qk_m64n128(
            q_rowmajor,
            q_rowmajor_scales,
            k,
            k_scales,
            out_tile,
            0,
            0,
        )
        torch.cuda.synchronize()
        ref = torch.matmul(q_ref_f32[:64].float(), k_ref_f32[:128].T.float())
        result = {
            "sm120_d256_manual_qk_m64n128": True,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "group": args.group,
        }
        result.update(compare("manual_qk_m64n128_vs_ref", out_tile, ref))
        result["bench_manual_qk_m64n128"] = event_ms(
            lambda: ext.sm120_d256_manual_qk_m64n128(
                q_rowmajor,
                q_rowmajor_scales,
                k,
                k_scales,
                out_tile,
                0,
                0,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(json.dumps(result))
        return

    if args.sm120_d256_manual_qk_smem_check_only:
        if args.head_dim != 256:
            raise ValueError("--sm120-d256-manual-qk-smem-check-only requires --head-dim 256")
        if args.q_len * args.group < 64:
            raise ValueError("manual D256 QK smem check requires at least 64 flattened Q rows")
        if args.kv_len < 128:
            raise ValueError("manual D256 QK smem check requires at least 128 KV rows")
        q_rows = args.q_len * args.group
        q_rowmajor, q_rowmajor_scales = fp32_to_nvfp4_rowmajor(
            q.reshape(q_rows, args.head_dim)
        )
        q_ref_f32 = nvfp4_rowmajor_to_fp32(q_rowmajor, q_rowmajor_scales)
        k_ref_f32 = nvfp4_rowmajor_to_fp32(k, k_scales)
        out_tile = torch.empty((64, 128), device=device, dtype=torch.float32)
        ext.sm120_d256_manual_qk_smem_m64n128(
            q_rowmajor,
            q_rowmajor_scales,
            k,
            k_scales,
            out_tile,
            0,
            0,
        )
        torch.cuda.synchronize()
        ref = torch.matmul(q_ref_f32[:64].float(), k_ref_f32[:128].T.float())
        result = {
            "sm120_d256_manual_qk_smem_m64n128": True,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "group": args.group,
        }
        result.update(compare("manual_qk_smem_m64n128_vs_ref", out_tile, ref))
        result["bench_manual_qk_smem_m64n128"] = event_ms(
            lambda: ext.sm120_d256_manual_qk_smem_m64n128(
                q_rowmajor,
                q_rowmajor_scales,
                k,
                k_scales,
                out_tile,
                0,
                0,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(json.dumps(result))
        return

    if args.sm120_d256_manual_pv_check_only:
        if args.head_dim != 256:
            raise ValueError("--sm120-d256-manual-pv-check-only requires --head-dim 256")
        if args.q_len * args.group < 64:
            raise ValueError("manual D256 PV check requires at least 64 flattened P rows")
        if args.kv_len < 128:
            raise ValueError("manual D256 PV check requires at least 128 KV rows")
        p_ref = torch.rand((64, args.kv_len), device=device, dtype=torch.float32)
        p_scaled = p_ref * PROB_GLOBAL_SCALE
        p_packed, p_scales = fp32_to_nvfp4_rowmajor(p_scaled)
        p_dequant_scaled = nvfp4_rowmajor_to_fp32(p_packed, p_scales)
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
        out_tile = torch.empty((64, 128), device=device, dtype=torch.float32)
        ext.sm120_d256_manual_pv_m64n128k128(
            p_packed,
            p_scales,
            v_pv,
            v_pv_scales,
            out_tile,
            0,
            0,
            0,
            0,
        )
        torch.cuda.synchronize()
        ref = torch.matmul(
            p_dequant_scaled[:64, :128].float(),
            v_pv_ref_f32[:128, :128].float().T,
        )
        result = {
            "sm120_d256_manual_pv_m64n128k128": True,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "group": args.group,
        }
        result.update(compare("manual_pv_m64n128k128_vs_ref", out_tile, ref))
        result["bench_manual_pv_m64n128k128"] = event_ms(
            lambda: ext.sm120_d256_manual_pv_m64n128k128(
                p_packed,
                p_scales,
                v_pv,
                v_pv_scales,
                out_tile,
                0,
                0,
                0,
                0,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(json.dumps(result))
        return

    if args.sm120_d256_manual_pv_smem_check_only:
        if args.head_dim != 256:
            raise ValueError("--sm120-d256-manual-pv-smem-check-only requires --head-dim 256")
        if args.q_len * args.group < 64:
            raise ValueError("manual D256 PV smem check requires at least 64 flattened P rows")
        if args.kv_len < 128:
            raise ValueError("manual D256 PV smem check requires at least 128 KV rows")
        p_ref = torch.rand((64, args.kv_len), device=device, dtype=torch.float32)
        p_scaled = p_ref * PROB_GLOBAL_SCALE
        p_packed, p_scales = fp32_to_nvfp4_rowmajor(p_scaled)
        p_dequant_scaled = nvfp4_rowmajor_to_fp32(p_packed, p_scales)
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
        out_tile = torch.empty((64, 128), device=device, dtype=torch.float32)
        ext.sm120_d256_manual_pv_smem_m64n128k128(
            p_packed,
            p_scales,
            v_pv,
            v_pv_scales,
            out_tile,
            0,
            0,
            0,
            0,
        )
        torch.cuda.synchronize()
        ref = torch.matmul(
            p_dequant_scaled[:64, :128].float(),
            v_pv_ref_f32[:128, :128].float().T,
        )
        result = {
            "sm120_d256_manual_pv_smem_m64n128k128": True,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "group": args.group,
        }
        result.update(compare("manual_pv_smem_m64n128k128_vs_ref", out_tile, ref))
        result["bench_manual_pv_smem_m64n128k128"] = event_ms(
            lambda: ext.sm120_d256_manual_pv_smem_m64n128k128(
                p_packed,
                p_scales,
                v_pv,
                v_pv_scales,
                out_tile,
                0,
                0,
                0,
                0,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(json.dumps(result))
        return

    if args.sm120_d256_manual_fused_tile_check_only:
        if args.head_dim != 256:
            raise ValueError(
                "--sm120-d256-manual-fused-tile-check-only requires --head-dim 256"
            )
        if args.q_len * args.group < 64:
            raise ValueError("manual D256 fused tile check requires at least 64 Q rows")
        if args.kv_len < 128:
            raise ValueError("manual D256 fused tile check requires at least 128 KV rows")
        q_rows = args.q_len * args.group
        q_rowmajor, q_rowmajor_scales = fp32_to_nvfp4_rowmajor(
            q.reshape(q_rows, args.head_dim)
        )
        q_ref_f32 = nvfp4_rowmajor_to_fp32(q_rowmajor, q_rowmajor_scales)
        k_ref_f32 = nvfp4_rowmajor_to_fp32(k, k_scales)
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
        out_tile = torch.empty((64, 128), device=device, dtype=torch.float32)
        ext.sm120_d256_manual_fused_tile_m64n128(
            q_rowmajor,
            q_rowmajor_scales,
            k,
            k_scales,
            v_pv,
            v_pv_scales,
            out_tile,
            0,
            0,
            0,
        )
        torch.cuda.synchronize()
        scores = (
            torch.matmul(q_ref_f32[:64].float(), k_ref_f32[:128].float().T)
            / math.sqrt(args.head_dim)
        )
        probs = torch.softmax(scores, dim=-1)
        ref_exact_p = torch.matmul(
            probs.float(), v_pv_ref_f32[:128, :128].float().T
        )
        p_packed, p_scales = fp32_to_nvfp4_rowmajor(probs * PROB_GLOBAL_SCALE)
        p_quant = nvfp4_rowmajor_to_fp32(p_packed, p_scales) / PROB_GLOBAL_SCALE
        ref_quant_p = torch.matmul(
            p_quant.float(), v_pv_ref_f32[:128, :128].float().T
        )
        result = {
            "sm120_d256_manual_fused_tile_m64n128": True,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "group": args.group,
        }
        result.update(
            compare("manual_fused_tile_m64n128_vs_quant_p_ref", out_tile, ref_quant_p)
        )
        result.update(
            compare("manual_fused_tile_m64n128_vs_exact_p_ref", out_tile, ref_exact_p)
        )
        result["bench_manual_fused_tile_m64n128"] = event_ms(
            lambda: ext.sm120_d256_manual_fused_tile_m64n128(
                q_rowmajor,
                q_rowmajor_scales,
                k,
                k_scales,
                v_pv,
                v_pv_scales,
                out_tile,
                0,
                0,
                0,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(json.dumps(result))
        return

    if args.sm120_d256_manual_fused_tile_smem_check_only:
        if args.head_dim != 256:
            raise ValueError(
                "--sm120-d256-manual-fused-tile-smem-check-only requires --head-dim 256"
            )
        if args.q_len * args.group < 64:
            raise ValueError(
                "manual D256 fused tile smem check requires at least 64 Q rows"
            )
        if args.kv_len < 128:
            raise ValueError(
                "manual D256 fused tile smem check requires at least 128 KV rows"
            )
        q_rows = args.q_len * args.group
        q_rowmajor, q_rowmajor_scales = fp32_to_nvfp4_rowmajor(
            q.reshape(q_rows, args.head_dim)
        )
        q_ref_f32 = nvfp4_rowmajor_to_fp32(q_rowmajor, q_rowmajor_scales)
        k_ref_f32 = nvfp4_rowmajor_to_fp32(k, k_scales)
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
        out_tile = torch.empty((64, 128), device=device, dtype=torch.float32)
        ext.sm120_d256_manual_fused_tile_smem_m64n128(
            q_rowmajor,
            q_rowmajor_scales,
            k,
            k_scales,
            v_pv,
            v_pv_scales,
            out_tile,
            0,
            0,
            0,
        )
        torch.cuda.synchronize()
        scores = (
            torch.matmul(q_ref_f32[:64].float(), k_ref_f32[:128].float().T)
            / math.sqrt(args.head_dim)
        )
        probs = torch.softmax(scores, dim=-1)
        ref_exact_p = torch.matmul(
            probs.float(), v_pv_ref_f32[:128, :128].float().T
        )
        p_packed, p_scales = fp32_to_nvfp4_rowmajor(probs * PROB_GLOBAL_SCALE)
        p_quant = nvfp4_rowmajor_to_fp32(p_packed, p_scales) / PROB_GLOBAL_SCALE
        ref_quant_p = torch.matmul(
            p_quant.float(), v_pv_ref_f32[:128, :128].float().T
        )
        result = {
            "sm120_d256_manual_fused_tile_smem_m64n128": True,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "group": args.group,
        }
        result.update(
            compare(
                "manual_fused_tile_smem_m64n128_vs_quant_p_ref",
                out_tile,
                ref_quant_p,
            )
        )
        result.update(
            compare(
                "manual_fused_tile_smem_m64n128_vs_exact_p_ref",
                out_tile,
                ref_exact_p,
            )
        )
        result["bench_manual_fused_tile_smem_m64n128"] = event_ms(
            lambda: ext.sm120_d256_manual_fused_tile_smem_m64n128(
                q_rowmajor,
                q_rowmajor_scales,
                k,
                k_scales,
                v_pv,
                v_pv_scales,
                out_tile,
                0,
                0,
                0,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(json.dumps(result))
        return

    if args.sm120_d256_manual_fused_tile_compact_check_only:
        if args.head_dim != 256:
            raise ValueError(
                "--sm120-d256-manual-fused-tile-compact-check-only requires --head-dim 256"
            )
        if args.q_len * args.group < 64:
            raise ValueError(
                "manual D256 compact fused tile check requires at least 64 Q rows"
            )
        if args.kv_len < 128:
            raise ValueError(
                "manual D256 compact fused tile check requires at least 128 KV rows"
            )
        q_rows = args.q_len * args.group
        q_rowmajor, q_rowmajor_scales = fp32_to_nvfp4_rowmajor(
            q.reshape(q_rows, args.head_dim)
        )
        q_ref_f32 = nvfp4_rowmajor_to_fp32(q_rowmajor, q_rowmajor_scales)
        k_ref_f32 = nvfp4_rowmajor_to_fp32(k, k_scales)
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
        out_tile = torch.empty((64, 128), device=device, dtype=torch.float32)
        ext.sm120_d256_manual_fused_tile_compact_m64n128(
            q_rowmajor,
            q_rowmajor_scales,
            k,
            k_scales,
            v_pv,
            v_pv_scales,
            out_tile,
            0,
            0,
            0,
        )
        torch.cuda.synchronize()
        scores = (
            torch.matmul(q_ref_f32[:64].float(), k_ref_f32[:128].float().T)
            / math.sqrt(args.head_dim)
        ).to(torch.bfloat16).float()
        probs = torch.softmax(scores, dim=-1)
        ref_exact_p = torch.matmul(
            probs.float(), v_pv_ref_f32[:128, :128].float().T
        )
        p_packed, p_scales = fp32_to_nvfp4_rowmajor(probs * PROB_GLOBAL_SCALE)
        p_quant = nvfp4_rowmajor_to_fp32(p_packed, p_scales) / PROB_GLOBAL_SCALE
        ref_quant_p = torch.matmul(
            p_quant.float(), v_pv_ref_f32[:128, :128].float().T
        )
        result = {
            "sm120_d256_manual_fused_tile_compact_m64n128": True,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "group": args.group,
        }
        result.update(
            compare(
                "manual_fused_tile_compact_m64n128_vs_quant_p_ref",
                out_tile,
                ref_quant_p,
            )
        )
        result.update(
            compare(
                "manual_fused_tile_compact_m64n128_vs_exact_p_ref",
                out_tile,
                ref_exact_p,
            )
        )
        result["bench_manual_fused_tile_compact_m64n128"] = event_ms(
            lambda: ext.sm120_d256_manual_fused_tile_compact_m64n128(
                q_rowmajor,
                q_rowmajor_scales,
                k,
                k_scales,
                v_pv,
                v_pv_scales,
                out_tile,
                0,
                0,
                0,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(json.dumps(result))
        return

    if (
        args.sm120_qkv_online_splitkv_full_grid_bench
        or args.sm120_qkv_online_splitkv_reuse2_full_grid_bench
        or args.sm120_qkv_online_splitkv_reuse4_full_grid_bench
    ):
        if args.head_dim not in (128, 256, 512):
            raise ValueError(
                "current fused specialization requires --head-dim in "
                "{128,256,512}"
            )
        tile_n = int(metadata["tile_n"])
        if args.kv_len % tile_n != 0:
            raise ValueError(f"--kv-len must be a multiple of {tile_n}")
        if args.split_kv_len <= 0 or args.split_kv_len % tile_n != 0:
            raise ValueError(
                f"--split-kv-len must be a positive multiple of {tile_n}"
            )
        q_rows = args.q_len * args.group
        k_ref_f32 = nvfp4_rowmajor_to_fp32(k, k_scales)
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        q_cutlass, q_cutlass_scales, q_cutlass_global = quantize_cutlass(
            q.reshape(q_rows, args.head_dim)
        )
        k_cutlass, k_cutlass_scales, k_cutlass_global = quantize_cutlass(
            k_ref_f32.to(torch.bfloat16)
        )
        v_pv_cutlass, v_pv_cutlass_scales, v_pv_cutlass_global = quantize_cutlass(
            v_ref_f32.T.contiguous().to(torch.bfloat16)
        )
        qk_alpha = 1.0 / (q_cutlass_global * k_cutlass_global)
        pv_alpha = 1.0 / v_pv_cutlass_global
        workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
        num_splits = (args.kv_len + args.split_kv_len - 1) // args.split_kv_len
        split_kv_tiles = args.split_kv_len // tile_n
        out = torch.empty((q_rows, args.head_dim), device=device, dtype=torch.bfloat16)
        partial = torch.empty(
            (num_splits, q_rows, args.head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        split_m = torch.empty((num_splits, q_rows), device=device, dtype=torch.float32)
        split_l = torch.empty((num_splits, q_rows), device=device, dtype=torch.float32)
        splitkv_fn = (
            ext.sm120_nvfp4_qkv_online_register_q_splitkv_reuse4_full_grid
            if args.sm120_qkv_online_splitkv_reuse4_full_grid_bench
            else (
                ext.sm120_nvfp4_qkv_online_register_q_splitkv_reuse2_full_grid
                if args.sm120_qkv_online_splitkv_reuse2_full_grid_bench
                else ext.sm120_nvfp4_qkv_online_register_q_splitkv_full_grid
            )
        )
        splitkv_name = (
            "sm120_qkv_online_register_q_splitkv_reuse4_full_grid"
            if args.sm120_qkv_online_splitkv_reuse4_full_grid_bench
            else (
                "sm120_qkv_online_register_q_splitkv_reuse2_full_grid"
                if args.sm120_qkv_online_splitkv_reuse2_full_grid_bench
                else "sm120_qkv_online_register_q_splitkv_full_grid"
            )
        )
        splitkv_fn(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            v_pv_cutlass,
            v_pv_cutlass_scales,
            partial,
            split_m,
            split_l,
            out,
            workspace,
            float(qk_alpha.item()),
            float(pv_alpha.item()),
            split_kv_tiles,
        )
        torch.cuda.synchronize()

        tactic = min(2, int(metadata["runner_tactic_count"]) - 1)
        qk_ref = torch.empty(
            (128, args.kv_len), device=device, dtype=torch.bfloat16
        )
        ext.cutlass_runner_fp4_gemm(
            q_cutlass[:128].contiguous(),
            k_cutlass.contiguous(),
            q_cutlass_scales[:128].contiguous(),
            k_cutlass_scales.contiguous(),
            qk_alpha,
            qk_ref,
            workspace,
            tactic,
        )
        torch.cuda.synchronize()
        probs = torch.softmax(qk_ref.float() / math.sqrt(args.head_dim), dim=-1)
        exact_ref = torch.matmul(probs.float(), v_ref_f32[:, :128].float())
        result = {
            splitkv_name: True,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "group": args.group,
            "output_shape": list(out.shape),
            "partial_shape": list(partial.shape),
            "splits": num_splits,
            "split_kv_len": args.split_kv_len,
            "storage_bytes": metadata["sm120_qkv_load_collective_storage_bytes"],
            "storage_margin_bytes": metadata[
                "sm120_qkv_load_collective_storage_margin_bytes"
            ],
        }
        result.update(compare("splitkv_full_grid_first_tile_vs_exact", out[:128, :128], exact_ref))
        result[f"bench_{splitkv_name}"] = event_ms(
            lambda: splitkv_fn(
                q_cutlass,
                q_cutlass_scales,
                k_cutlass,
                k_cutlass_scales,
                v_pv_cutlass,
                v_pv_cutlass_scales,
                partial,
                split_m,
                split_l,
                out,
                workspace,
                float(qk_alpha.item()),
                float(pv_alpha.item()),
                split_kv_tiles,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(json.dumps(result))
        return

    if args.sm120_d256_manual_compact_register_o_splitkv_full_grid_bench:
        if args.head_dim != 256:
            raise ValueError(
                "--sm120-d256-manual-compact-register-o-splitkv-full-grid-bench requires --head-dim 256"
            )
        if args.kv_len % 128 != 0:
            raise ValueError("--kv-len must be a multiple of 128")
        if args.split_kv_len <= 0 or args.split_kv_len % 128 != 0:
            raise ValueError("--split-kv-len must be a positive multiple of 128")
        q_rows = args.q_len * args.group
        q_rowmajor, q_rowmajor_scales = fp32_to_nvfp4_rowmajor(
            q.reshape(q_rows, args.head_dim)
        )
        q_ref_f32 = nvfp4_rowmajor_to_fp32(q_rowmajor, q_rowmajor_scales)
        k_ref_f32 = nvfp4_rowmajor_to_fp32(k, k_scales)
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        split_kv_tiles = args.split_kv_len // 128
        num_splits = (args.kv_len // 128 + split_kv_tiles - 1) // split_kv_tiles
        out = torch.empty((q_rows, args.head_dim), device=device, dtype=torch.bfloat16)
        partial = torch.empty(
            (num_splits, q_rows, args.head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        split_m = torch.empty((num_splits, q_rows), device=device, dtype=torch.float32)
        split_l = torch.empty((num_splits, q_rows), device=device, dtype=torch.float32)
        ext.sm120_d256_manual_compact_register_o_splitkv_full_grid(
            q_rowmajor,
            q_rowmajor_scales,
            k,
            k_scales,
            v_pv,
            v_pv_scales,
            partial,
            split_m,
            split_l,
            out,
            split_kv_tiles,
        )
        torch.cuda.synchronize()
        scores = (
            torch.matmul(q_ref_f32[:64].float(), k_ref_f32.float().T)
            / math.sqrt(args.head_dim)
        ).to(torch.bfloat16).float()
        probs = torch.softmax(scores, dim=-1)
        exact_ref = torch.matmul(probs.float(), v_ref_f32[:, :128].float())
        result = {
            "sm120_d256_manual_compact_register_o_splitkv_full_grid": True,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "group": args.group,
            "output_shape": list(out.shape),
            "partial_shape": list(partial.shape),
            "splits": num_splits,
            "split_kv_len": args.split_kv_len,
            "storage_bytes": 49408,
        }
        result.update(
            compare(
                "manual_compact_register_o_splitkv_first_tile_vs_exact",
                out[:64, :128],
                exact_ref,
            )
        )
        result["bench_sm120_d256_manual_compact_register_o_splitkv_full_grid"] = event_ms(
            lambda: ext.sm120_d256_manual_compact_register_o_splitkv_full_grid(
                q_rowmajor,
                q_rowmajor_scales,
                k,
                k_scales,
                v_pv,
                v_pv_scales,
                partial,
                split_m,
                split_l,
                out,
                split_kv_tiles,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(json.dumps(result))
        return

    if args.sm120_d256_manual_compact_online_splitkv_full_grid_bench:
        if args.head_dim != 256:
            raise ValueError(
                "--sm120-d256-manual-compact-online-splitkv-full-grid-bench requires --head-dim 256"
            )
        if args.kv_len % 128 != 0:
            raise ValueError("--kv-len must be a multiple of 128")
        if args.split_kv_len <= 0 or args.split_kv_len % 128 != 0:
            raise ValueError("--split-kv-len must be a positive multiple of 128")
        q_rows = args.q_len * args.group
        q_rowmajor, q_rowmajor_scales = fp32_to_nvfp4_rowmajor(
            q.reshape(q_rows, args.head_dim)
        )
        q_ref_f32 = nvfp4_rowmajor_to_fp32(q_rowmajor, q_rowmajor_scales)
        k_ref_f32 = nvfp4_rowmajor_to_fp32(k, k_scales)
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        split_kv_tiles = args.split_kv_len // 128
        num_splits = (args.kv_len // 128 + split_kv_tiles - 1) // split_kv_tiles
        out = torch.empty((q_rows, args.head_dim), device=device, dtype=torch.bfloat16)
        partial = torch.empty(
            (num_splits, q_rows, args.head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        split_m = torch.empty((num_splits, q_rows), device=device, dtype=torch.float32)
        split_l = torch.empty((num_splits, q_rows), device=device, dtype=torch.float32)
        ext.sm120_d256_manual_compact_online_splitkv_full_grid(
            q_rowmajor,
            q_rowmajor_scales,
            k,
            k_scales,
            v_pv,
            v_pv_scales,
            partial,
            split_m,
            split_l,
            out,
            split_kv_tiles,
        )
        torch.cuda.synchronize()
        scores = (
            torch.matmul(q_ref_f32[:64].float(), k_ref_f32.float().T)
            / math.sqrt(args.head_dim)
        ).to(torch.bfloat16).float()
        probs = torch.softmax(scores, dim=-1)
        exact_ref = torch.matmul(probs.float(), v_ref_f32[:, :128].float())
        result = {
            "sm120_d256_manual_compact_online_splitkv_full_grid": True,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "group": args.group,
            "output_shape": list(out.shape),
            "partial_shape": list(partial.shape),
            "splits": num_splits,
            "split_kv_len": args.split_kv_len,
            "storage_bytes": 65792,
        }
        result.update(
            compare(
                "manual_compact_online_splitkv_first_tile_vs_exact",
                out[:64, :128],
                exact_ref,
            )
        )
        result["bench_sm120_d256_manual_compact_online_splitkv_full_grid"] = event_ms(
            lambda: ext.sm120_d256_manual_compact_online_splitkv_full_grid(
                q_rowmajor,
                q_rowmajor_scales,
                k,
                k_scales,
                v_pv,
                v_pv_scales,
                partial,
                split_m,
                split_l,
                out,
                split_kv_tiles,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(json.dumps(result))
        return

    if args.sm120_d256_manual_compact_splitkv_full_grid_bench:
        if args.head_dim != 256:
            raise ValueError(
                "--sm120-d256-manual-compact-splitkv-full-grid-bench requires --head-dim 256"
            )
        if args.kv_len % 128 != 0:
            raise ValueError("--kv-len must be a multiple of 128")
        q_rows = args.q_len * args.group
        q_rowmajor, q_rowmajor_scales = fp32_to_nvfp4_rowmajor(
            q.reshape(q_rows, args.head_dim)
        )
        q_ref_f32 = nvfp4_rowmajor_to_fp32(q_rowmajor, q_rowmajor_scales)
        k_ref_f32 = nvfp4_rowmajor_to_fp32(k, k_scales)
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        num_splits = args.kv_len // 128
        out = torch.empty((q_rows, args.head_dim), device=device, dtype=torch.bfloat16)
        partial = torch.empty(
            (num_splits, q_rows, args.head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        split_m = torch.empty((num_splits, q_rows), device=device, dtype=torch.float32)
        split_l = torch.empty((num_splits, q_rows), device=device, dtype=torch.float32)
        ext.sm120_d256_manual_compact_splitkv_full_grid(
            q_rowmajor,
            q_rowmajor_scales,
            k,
            k_scales,
            v_pv,
            v_pv_scales,
            partial,
            split_m,
            split_l,
            out,
        )
        torch.cuda.synchronize()
        scores = (
            torch.matmul(q_ref_f32[:64].float(), k_ref_f32.float().T)
            / math.sqrt(args.head_dim)
        ).to(torch.bfloat16).float()
        probs = torch.softmax(scores, dim=-1)
        exact_ref = torch.matmul(probs.float(), v_ref_f32[:, :128].float())
        result = {
            "sm120_d256_manual_compact_splitkv_full_grid": True,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "group": args.group,
            "output_shape": list(out.shape),
            "partial_shape": list(partial.shape),
            "splits": num_splits,
            "split_kv_len": 128,
            "storage_bytes": 48640,
        }
        result.update(
            compare(
                "manual_compact_splitkv_first_tile_vs_exact",
                out[:64, :128],
                exact_ref,
            )
        )
        result["bench_sm120_d256_manual_compact_splitkv_full_grid"] = event_ms(
            lambda: ext.sm120_d256_manual_compact_splitkv_full_grid(
                q_rowmajor,
                q_rowmajor_scales,
                k,
                k_scales,
                v_pv,
                v_pv_scales,
                partial,
                split_m,
                split_l,
                out,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(json.dumps(result))
        return

    if not default_shape:
        raise ValueError(
            "non-default shapes are currently supported only by the split-KV "
            "Shape B benchmark path"
        )

    q_expected, q_scales_expected = fp32_to_nvfp4_rowmajor(
        q.reshape(Q_LEN * GROUP, HEAD_DIM)
    )
    q_actual = torch.empty_like(q_expected)
    q_scales_actual = torch.empty_like(q_scales_expected)
    ext.quantize_q_rowmajor(q, q_actual, q_scales_actual)
    torch.cuda.synchronize()

    q_expected_f32 = nvfp4_rowmajor_to_fp32(q_expected, q_scales_expected)
    q_actual_f32 = nvfp4_rowmajor_to_fp32(q_actual, q_scales_actual)
    q_dequant_delta = (q_actual_f32 - q_expected_f32).abs()
    k_ref_f32 = nvfp4_rowmajor_to_fp32(k, k_scales)

    if args.smem_atom_unit_scales:
        q_scales_actual.fill_(0x38)
        k_scales.fill_(0x38)
        q_actual_f32 = nvfp4_rowmajor_to_fp32(q_actual, q_scales_actual)
        k_ref_f32 = nvfp4_rowmajor_to_fp32(k, k_scales)
    if args.smem_atom_q_code >= 0 or args.smem_atom_k_code >= 0:
        if args.smem_atom_q_code not in range(16):
            raise ValueError("--smem-atom-q-code must be in [0, 15]")
        if args.smem_atom_k_code not in range(16):
            raise ValueError("--smem-atom-k-code must be in [0, 15]")
        q_byte = args.smem_atom_q_code | (args.smem_atom_q_code << 4)
        k_byte = args.smem_atom_k_code | (args.smem_atom_k_code << 4)
        q_actual.fill_(q_byte)
        q_scales_actual.fill_(0x38)
        k.fill_(k_byte)
        k_scales.fill_(0x38)
        q_actual_f32 = torch.full_like(q_actual_f32, E2M1_VALUES[args.smem_atom_q_code].item())
        k_ref_f32 = torch.full_like(k_ref_f32, E2M1_VALUES[args.smem_atom_k_code].item())
    if args.smem_atom_ones:
        q_actual.fill_(0x22)
        q_scales_actual.fill_(0x38)
        k.fill_(0x22)
        k_scales.fill_(0x38)
        q_actual_f32 = torch.ones_like(q_actual_f32)
        k_ref_f32 = torch.ones_like(k_ref_f32)

    def compare(name: str, actual: torch.Tensor, ref: torch.Tensor) -> dict[str, object]:
        delta = (actual.float() - ref.float()).abs()
        cos = torch.sum(actual.float() * ref.float()) / torch.clamp(
            torch.linalg.vector_norm(actual.float()) * torch.linalg.vector_norm(ref.float()),
            min=1.0e-20,
        )
        return {
            f"{name}_finite": bool(torch.isfinite(actual).all().item()),
            f"{name}_mean_abs": float(delta.mean().item()),
            f"{name}_max_abs": float(delta.max().item()),
            f"{name}_cosine": float(cos.item()),
        }

    def cutlass_qk_inputs():
        q_cutlass, q_cutlass_scales, q_cutlass_global = quantize_cutlass(
            q.reshape(Q_LEN * GROUP, HEAD_DIM)
        )
        k_cutlass, k_cutlass_scales, k_cutlass_global = quantize_cutlass(
            k_ref_f32.to(torch.bfloat16)
        )
        qk_alpha = 1.0 / (q_cutlass_global * k_cutlass_global)
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        tactic = min(2, int(metadata["runner_tactic_count"]) - 1)
        return (
            q_cutlass,
            q_cutlass_scales,
            q_cutlass_global,
            k_cutlass,
            k_cutlass_scales,
            k_cutlass_global,
            qk_alpha,
            workspace,
            tactic,
        )

    def cutlass_v_inputs():
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv_cutlass, v_pv_cutlass_scales, v_pv_cutlass_global = quantize_cutlass(
            v_ref_f32.T.contiguous().to(torch.bfloat16)
        )
        return v_ref_f32, v_pv_cutlass, v_pv_cutlass_scales, v_pv_cutlass_global

    def qk_runner_ref(q_cutlass, q_cutlass_scales, k_cutlass, k_cutlass_scales, qk_alpha, workspace, tactic, q_rows=128, kv_rows=128):
        qk_ref = torch.empty((q_rows, kv_rows), device=device, dtype=torch.bfloat16)
        ext.cutlass_runner_fp4_gemm(
            q_cutlass[:q_rows].contiguous(),
            k_cutlass[:kv_rows].contiguous(),
            q_cutlass_scales[:q_rows].contiguous(),
            k_cutlass_scales[:kv_rows].contiguous(),
            qk_alpha,
            qk_ref,
            workspace,
            tactic,
        )
        torch.cuda.synchronize()
        return qk_ref

    if args.sm120_qkv_online_check_only:
        (
            q_cutlass,
            q_cutlass_scales,
            _,
            k_cutlass,
            k_cutlass_scales,
            _,
            qk_alpha,
            workspace,
            tactic,
        ) = cutlass_qk_inputs()
        v_ref_f32, v_pv_cutlass, v_pv_cutlass_scales, v_pv_cutlass_global = cutlass_v_inputs()
        out_group = torch.empty((128, 128), device=device, dtype=torch.bfloat16)
        out_group_idx = 0
        kv_tiles = args.online_kv_tiles
        pv_alpha = 1.0 / v_pv_cutlass_global
        ext.sm120_nvfp4_qkv_online_register_q_stage(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            v_pv_cutlass,
            v_pv_cutlass_scales,
            out_group,
            workspace,
            float(qk_alpha.item()),
            float(pv_alpha.item()),
            0,
            0,
            kv_tiles,
            out_group_idx,
        )
        torch.cuda.synchronize()
        qk_ref = qk_runner_ref(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            qk_alpha,
            workspace,
            tactic,
            q_rows=128,
            kv_rows=128 * kv_tiles,
        )
        probs = torch.softmax(qk_ref.float() / (HEAD_DIM**0.5), dim=-1)
        group_start = out_group_idx * 128
        group_stop = group_start + 128
        exact_ref = torch.matmul(
            probs.float(),
            v_ref_f32[: 128 * kv_tiles, group_start:group_stop].float(),
        )
        result = {
            "sm120_qkv_online_register_q_stage": True,
            "kv_tiles": kv_tiles,
            "storage_bytes": metadata["sm120_qkv_load_collective_storage_bytes"],
            "storage_margin_bytes": metadata[
                "sm120_qkv_load_collective_storage_margin_bytes"
            ],
        }
        result.update(compare("online_register_q_vs_exact", out_group, exact_ref))
        if args.bench:
            result["bench_sm120_qkv_online_register_q_stage"] = event_ms(
                lambda: ext.sm120_nvfp4_qkv_online_register_q_stage(
                    q_cutlass,
                    q_cutlass_scales,
                    k_cutlass,
                    k_cutlass_scales,
                    v_pv_cutlass,
                    v_pv_cutlass_scales,
                    out_group,
                    workspace,
                    float(qk_alpha.item()),
                    float(pv_alpha.item()),
                    0,
                    0,
                    kv_tiles,
                    out_group_idx,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        print(result)
        return

    if args.sm120_qkv_online_full_grid_bench:
        (
            q_cutlass,
            q_cutlass_scales,
            _,
            k_cutlass,
            k_cutlass_scales,
            _,
            qk_alpha,
            workspace,
            tactic,
        ) = cutlass_qk_inputs()
        v_ref_f32, v_pv_cutlass, v_pv_cutlass_scales, v_pv_cutlass_global = cutlass_v_inputs()
        out = torch.empty((Q_LEN * GROUP, HEAD_DIM), device=device, dtype=torch.bfloat16)
        pv_alpha = 1.0 / v_pv_cutlass_global
        ext.sm120_nvfp4_qkv_online_register_q_full_grid(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            v_pv_cutlass,
            v_pv_cutlass_scales,
            out,
            workspace,
            float(qk_alpha.item()),
            float(pv_alpha.item()),
        )
        torch.cuda.synchronize()
        qk_ref = qk_runner_ref(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            qk_alpha,
            workspace,
            tactic,
            q_rows=128,
            kv_rows=k_cutlass.shape[0],
        )
        probs = torch.softmax(qk_ref.float() / (HEAD_DIM**0.5), dim=-1)
        exact_ref = torch.matmul(probs.float(), v_ref_f32[:, :128].float())
        result = {
            "sm120_qkv_online_register_q_full_grid": True,
            "output_shape": tuple(out.shape),
            "storage_bytes": metadata["sm120_qkv_load_collective_storage_bytes"],
            "storage_margin_bytes": metadata[
                "sm120_qkv_load_collective_storage_margin_bytes"
            ],
        }
        result.update(compare("full_grid_first_tile_vs_exact", out[:128, :128], exact_ref))
        result["bench_sm120_qkv_online_register_q_full_grid"] = event_ms(
            lambda: ext.sm120_nvfp4_qkv_online_register_q_full_grid(
                q_cutlass,
                q_cutlass_scales,
                k_cutlass,
                k_cutlass_scales,
                v_pv_cutlass,
                v_pv_cutlass_scales,
                out,
                workspace,
                float(qk_alpha.item()),
                float(pv_alpha.item()),
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(result)
        return

    if args.smem_atom_block_check_only:
        qk = torch.empty((128, k.shape[0]), device=device, dtype=torch.float32)
        ext.qk_cutlass_smem_atom_block(q_actual, q_scales_actual, k, k_scales, qk)
        torch.cuda.synchronize()
        ref = torch.matmul(q_actual_f32[:128].float(), k_ref_f32.float().T)
        result = compare("qk_smem_atom_block", qk, ref)
        if args.bench:
            result["bench_qk_smem_atom_block_128x32768"] = event_ms(
                lambda: ext.qk_cutlass_smem_atom_block(
                    q_actual, q_scales_actual, k, k_scales, qk
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        print(result)
        return

    if args.pv_smem_atom_tile_check_only:
        p_ref = torch.rand((128, k.shape[0]), device=device, dtype=torch.float32)
        p_scaled = p_ref * PROB_GLOBAL_SCALE
        p_packed, p_scales = fp32_to_nvfp4_rowmajor(p_scaled)
        p_dequant_scaled = nvfp4_rowmajor_to_fp32(p_packed, p_scales)
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
        out = torch.empty((128, 128), device=device, dtype=torch.float32)
        ext.pv_cutlass_smem_atom_tile(p_packed, p_scales, v_pv, v_pv_scales, out, 0, 0)
        torch.cuda.synchronize()
        ref = torch.matmul(
            p_dequant_scaled[:, :256].float(),
            v_pv_ref_f32[:128, :256].float().T,
        )
        result = compare("pv_smem_atom_tile", out, ref)
        if args.bench:
            result["bench_pv_smem_atom_tile_128x128x256"] = event_ms(
                lambda: ext.pv_cutlass_smem_atom_tile(
                    p_packed, p_scales, v_pv, v_pv_scales, out, 0, 0
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        print(result)
        return

    qk_smem_atom_result = {}
    if args.smem_atom_check or args.smem_atom_check_only:
        qk = torch.empty((128, 128), device=device, dtype=torch.float32)
        ext.qk_cutlass_smem_atom_tile(
            q_actual,
            q_scales_actual,
            k,
            k_scales,
            qk,
            args.smem_atom_data_mode,
            args.smem_atom_scale_mode,
        )
        torch.cuda.synchronize()
        ref = torch.matmul(q_actual_f32[:128].float(), k_ref_f32[:128].float().T)
        qk_smem_atom_result = compare("qk_smem_atom", qk, ref)
        qk_smem_atom_result.update(
            {
                "qk_smem_atom_data_mode": args.smem_atom_data_mode,
                "qk_smem_atom_scale_mode": args.smem_atom_scale_mode,
                "qk_smem_atom_min": float(qk.min().item()),
                "qk_smem_atom_max": float(qk.max().item()),
                "qk_smem_atom_mean": float(qk.mean().item()),
                "qk_smem_atom_ref_mean": float(ref.mean().item()),
            }
        )
        if args.bench:
            qk_smem_atom_result["bench_qk_smem_atom_tile_128x128"] = event_ms(
                lambda: ext.qk_cutlass_smem_atom_tile(
                    q_actual,
                    q_scales_actual,
                    k,
                    k_scales,
                    qk,
                    args.smem_atom_data_mode,
                    args.smem_atom_scale_mode,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        if args.smem_atom_check_only:
            print(qk_smem_atom_result)
            return

    runner_result = {}
    if args.runner_check or args.full_runner_bench or args.bench:
        (
            q_cutlass,
            q_cutlass_scales,
            q_cutlass_global,
            k_cutlass,
            k_cutlass_scales,
            k_cutlass_global,
            qk_alpha,
            workspace,
            tactic,
        ) = cutlass_qk_inputs()
        runner = gemm_base.get_cutlass_fp4_gemm_module(
            *torch.cuda.get_device_capability(device)
        ).cutlass_fp4_gemm_runner()
        qk_official = torch.empty((128, k_cutlass.shape[0]), device=device, dtype=torch.bfloat16)
        qk_ext = torch.empty_like(qk_official)
        runner(
            inputs=[
                q_cutlass[:128].contiguous(),
                k_cutlass.T,
                q_cutlass_scales[:128].contiguous(),
                k_cutlass_scales.T,
                qk_alpha,
                qk_official.dtype,
                qk_official,
                16,
                True,
                workspace,
            ],
            tactic=tactic,
        )
        ext.cutlass_runner_fp4_gemm(
            q_cutlass[:128].contiguous(),
            k_cutlass.contiguous(),
            q_cutlass_scales[:128].contiguous(),
            k_cutlass_scales.contiguous(),
            qk_alpha,
            qk_ext,
            workspace,
            tactic,
        )
        torch.cuda.synchronize()
        runner_result = {"qk_runner_tactic": tactic}
        runner_result.update(compare("qk_runner_vs_official", qk_ext.float(), qk_official.float()))

        p_ref = torch.softmax(qk_ext.float() / (HEAD_DIM**0.5), dim=-1)
        p_cutlass, p_cutlass_scales, p_cutlass_global = quantize_cutlass(
            p_ref.to(torch.bfloat16)
        )
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_cutlass, v_cutlass_scales, v_cutlass_global = quantize_cutlass(
            v_ref_f32.T.contiguous().to(torch.bfloat16)
        )
        pv_alpha = 1.0 / (p_cutlass_global * v_cutlass_global)
        pv_official = torch.empty(
            (p_cutlass.shape[0], v_cutlass.shape[0]), device=device, dtype=torch.bfloat16
        )
        pv_ext = torch.empty_like(pv_official)
        runner(
            inputs=[
                p_cutlass.contiguous(),
                v_cutlass.T,
                p_cutlass_scales.contiguous(),
                v_cutlass_scales.T,
                pv_alpha,
                pv_official.dtype,
                pv_official,
                16,
                True,
                workspace,
            ],
            tactic=tactic,
        )
        ext.cutlass_runner_fp4_gemm(
            p_cutlass.contiguous(),
            v_cutlass.contiguous(),
            p_cutlass_scales.contiguous(),
            v_cutlass_scales.contiguous(),
            pv_alpha,
            pv_ext,
            workspace,
            tactic,
        )
        torch.cuda.synchronize()
        runner_result.update(compare("pv_runner_vs_official", pv_ext.float(), pv_official.float()))
        pv_exact = torch.matmul(p_ref.float(), v_ref_f32.float())
        runner_result.update(compare("pv_runner_vs_exact", pv_ext.float(), pv_exact))
        if args.bench:
            runner_result["bench_cutlass_runner_qk_128x32768"] = event_ms(
                lambda: ext.cutlass_runner_fp4_gemm(
                    q_cutlass[:128].contiguous(),
                    k_cutlass.contiguous(),
                    q_cutlass_scales[:128].contiguous(),
                    k_cutlass_scales.contiguous(),
                    qk_alpha,
                    qk_ext,
                    workspace,
                    tactic,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
            runner_result["bench_cutlass_runner_pv_128x512_k32768"] = event_ms(
                lambda: ext.cutlass_runner_fp4_gemm(
                    p_cutlass.contiguous(),
                    v_cutlass.contiguous(),
                    p_cutlass_scales.contiguous(),
                    v_cutlass_scales.contiguous(),
                    pv_alpha,
                    pv_ext,
                    workspace,
                    tactic,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        if args.full_runner_bench:
            qk_full = torch.empty(
                (q_cutlass.shape[0], k_cutlass.shape[0]), device=device, dtype=torch.bfloat16
            )
            full_qk_bench = event_ms(
                lambda: ext.cutlass_runner_fp4_gemm(
                    q_cutlass.contiguous(),
                    k_cutlass.contiguous(),
                    q_cutlass_scales.contiguous(),
                    k_cutlass_scales.contiguous(),
                    qk_alpha,
                    qk_full,
                    workspace,
                    tactic,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
            ext.cutlass_runner_fp4_gemm(
                q_cutlass.contiguous(),
                k_cutlass.contiguous(),
                q_cutlass_scales.contiguous(),
                k_cutlass_scales.contiguous(),
                qk_alpha,
                qk_full,
                workspace,
                tactic,
            )
            torch.cuda.synchronize()
            p_full = torch.softmax(qk_full.float() / (HEAD_DIM**0.5), dim=-1)
            p_full_cutlass, p_full_scales, p_full_global = quantize_cutlass(
                p_full.to(torch.bfloat16)
            )
            pv_full_alpha = 1.0 / (p_full_global * v_cutlass_global)
            pv_full = torch.empty(
                (p_full_cutlass.shape[0], v_cutlass.shape[0]),
                device=device,
                dtype=torch.bfloat16,
            )
            full_pv_bench = event_ms(
                lambda: ext.cutlass_runner_fp4_gemm(
                    p_full_cutlass.contiguous(),
                    v_cutlass.contiguous(),
                    p_full_scales.contiguous(),
                    v_cutlass_scales.contiguous(),
                    pv_full_alpha,
                    pv_full,
                    workspace,
                    tactic,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
            runner_result.update(
                {
                    "full_runner_qk_shape": tuple(qk_full.shape),
                    "full_runner_pv_shape": tuple(pv_full.shape),
                    "bench_cutlass_runner_qk_4096x32768": full_qk_bench,
                    "bench_cutlass_runner_pv_4096x512_k32768": full_pv_bench,
                }
            )
        if args.runner_check or args.full_runner_bench:
            print(runner_result)
            return

    result = {
        "cutlass_collective_metadata": metadata,
        "q_quant_scale_match": bool(torch.equal(q_scales_actual, q_scales_expected)),
        "q_quant_packed_match": bool(torch.equal(q_actual, q_expected)),
        "q_quant_dequant_delta_mean": float(q_dequant_delta.mean().item()),
        "q_quant_dequant_delta_max": float(q_dequant_delta.max().item()),
    }
    result.update(qk_smem_atom_result)
    result.update(runner_result)
    print(result)


if __name__ == "__main__":
    main()
