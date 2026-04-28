from __future__ import annotations

import argparse
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


def build_extension():
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.2")
    extra_cuda_cflags = [
        "-std=c++17",
        "-O3",
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
    return load(
        name="sm120_nvfp4_cutlass_fused_attention_ext",
        sources=[
            str(root / "benchmarks" / "sm120_nvfp4_cutlass_fused_attention.cu"),
            str(root / "benchmarks" / "sm120_nvfp4_cutlass_runner_bf16_inst.cu"),
        ],
        extra_include_paths=[
            str(root / "include"),
            str(root / "3rdparty" / "cutlass" / "include"),
            str(root / "3rdparty" / "cutlass" / "tools" / "util" / "include"),
        ],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--runner-check", action="store_true")
    parser.add_argument("--full-runner-bench", action="store_true")
    parser.add_argument(
        "--collective-debug-mode",
        type=int,
        default=0,
        help=(
            "0=full collective smoke, 1=pipeline init only, 2=TMA load only, "
            "3=launch/smem only, 4=TMA descriptor prefetch only, "
            "5=pipeline init without prefetch"
        ),
    )
    parser.add_argument("--collective-check", action="store_true")
    parser.add_argument("--collective-check-only", action="store_true")
    parser.add_argument("--smem-atom-check", action="store_true")
    parser.add_argument("--smem-atom-check-only", action="store_true")
    parser.add_argument("--smem-atom-block-check-only", action="store_true")
    parser.add_argument("--pv-smem-atom-tile-check-only", action="store_true")
    parser.add_argument("--fused-smem-one-kv-tile-check-only", action="store_true")
    parser.add_argument("--fused-smem-online-col-check-only", action="store_true")
    parser.add_argument("--fused-smem-online-full-width-check-only", action="store_true")
    parser.add_argument("--fused-smem-online-split-full-width-check-only", action="store_true")
    parser.add_argument("--fused-smem-online-split-k128-full-width-check-only", action="store_true")
    parser.add_argument("--fused-tma-pv-online-split-k128-full-width-check-only", action="store_true")
    parser.add_argument("--fused-tma-qk-tma-pv-online-split-k128-full-width-check-only", action="store_true")
    parser.add_argument("--fused-tma-qk-pv-128-check-only", action="store_true")
    parser.add_argument("--fused-tma-qk-tma-pv-128-check-only", action="store_true")
    parser.add_argument("--persistent-owner-layout-check-only", action="store_true")
    parser.add_argument("--persistent-owner-qk-check-only", action="store_true")
    parser.add_argument("--persistent-owner-softmax-check-only", action="store_true")
    parser.add_argument("--persistent-owner-pv-group-check-only", action="store_true")
    parser.add_argument("--persistent-owner-full-tile-check-only", action="store_true")
    parser.add_argument("--persistent-owner-group-online-check-only", action="store_true")
    parser.add_argument("--persistent-owner-group-online-register-check-only", action="store_true")
    parser.add_argument(
        "--persistent-owner-group-online-register-q-check-only",
        action="store_true",
    )
    parser.add_argument("--persistent-owner-online-kv-tiles", type=int, default=2)
    parser.add_argument("--fused-smem-online-kv-tiles", type=int, default=4)
    parser.add_argument("--fused-smem-online-splits", type=int, default=8)
    parser.add_argument("--fused-smem-online-q-tiles", type=int, default=1)
    parser.add_argument("--fused-split-kernel-only", action="store_true")
    parser.add_argument("--smem-atom-data-mode", type=int, default=0)
    parser.add_argument("--smem-atom-scale-mode", type=int, default=0)
    parser.add_argument("--smem-atom-ones", action="store_true")
    parser.add_argument("--smem-atom-unit-scales", action="store_true")
    parser.add_argument("--smem-atom-q-code", type=int, default=-1)
    parser.add_argument("--smem-atom-k-code", type=int, default=-1)
    parser.add_argument("--smem-atom-code-sweep", action="store_true")
    parser.add_argument("--smem-atom-pattern-sweep", action="store_true")
    parser.add_argument("--smem-atom-k-map-probe", action="store_true")
    parser.add_argument("--smem-atom-b-nk-map-probe", action="store_true")
    parser.add_argument("--smem-atom-a-mk-map-probe", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    q, k, v, k_scales, v_scales = make_inputs(device)
    ext = build_extension()
    metadata = dict(ext.cutlass_sm120_blockscaled_collective_metadata())
    if args.persistent_owner_layout_check_only:
        marker = torch.empty(8, device=device, dtype=torch.float32)
        ext.persistent_mainloop_owner_layout_smoke(marker)
        torch.cuda.synchronize()
        marker_values = marker.cpu().tolist()
        result = {
            "persistent_owner_layout_smoke": True,
            "marker": marker_values,
            "sm120_optin_smem_bytes": metadata["sm120_optin_smem_bytes"],
            "phased_storage_bytes": int(marker_values[1]),
            "independent_qk_pv_fits_sm120": metadata[
                "persistent_independent_qk_pv_fits_sm120"
            ],
            "independent_qk_pv_smem_margin_bytes": metadata[
                "persistent_independent_qk_pv_smem_margin_bytes"
            ],
        }
        print(result)
        return

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
    if args.persistent_owner_qk_check_only:
        q_cutlass, q_cutlass_scales, q_cutlass_global = quantize_cutlass(
            q.reshape(Q_LEN * GROUP, HEAD_DIM)
        )
        k_cutlass, k_cutlass_scales, k_cutlass_global = quantize_cutlass(
            k_ref_f32.to(torch.bfloat16)
        )
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        qk_owner = torch.empty((128, 128), device=device, dtype=torch.float32)
        ext.persistent_mainloop_owner_qk_stage(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            qk_owner,
            workspace,
            0,
            0,
        )
        torch.cuda.synchronize()
        qk_official = torch.empty(
            (128, k_cutlass.shape[0]), device=device, dtype=torch.bfloat16
        )
        qk_alpha = 1.0 / (q_cutlass_global * k_cutlass_global)
        tactic = min(2, int(metadata["runner_tactic_count"]) - 1)
        ext.cutlass_runner_fp4_gemm(
            q_cutlass[:128].contiguous(),
            k_cutlass.contiguous(),
            q_cutlass_scales[:128].contiguous(),
            k_cutlass_scales.contiguous(),
            qk_alpha,
            qk_official,
            workspace,
            tactic,
        )
        torch.cuda.synchronize()
        qk_owner_scaled = qk_owner * qk_alpha
        qk_ref = qk_official[:, :128].float()
        delta = (qk_owner_scaled - qk_ref).abs()
        cos = torch.sum(qk_owner_scaled * qk_ref) / torch.clamp(
            torch.linalg.vector_norm(qk_owner_scaled) * torch.linalg.vector_norm(qk_ref),
            min=1.0e-20,
        )
        result = {
            "persistent_owner_qk_stage": True,
            "finite": bool(torch.isfinite(qk_owner_scaled).all().item()),
            "mean_abs": float(delta.mean().item()),
            "max_abs": float(delta.max().item()),
            "cosine": float(cos.item()),
        }
        print(result)
        return
    if args.persistent_owner_softmax_check_only:
        q_cutlass, q_cutlass_scales, q_cutlass_global = quantize_cutlass(
            q.reshape(Q_LEN * GROUP, HEAD_DIM)
        )
        k_cutlass, k_cutlass_scales, k_cutlass_global = quantize_cutlass(
            k_ref_f32.to(torch.bfloat16)
        )
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        p_packed = torch.empty((128, 64), device=device, dtype=torch.uint8)
        p_scales = torch.empty((128, 8), device=device, dtype=torch.uint8)
        row_m = torch.empty((128,), device=device, dtype=torch.float32)
        row_l = torch.empty((128,), device=device, dtype=torch.float32)
        qk_alpha = 1.0 / (q_cutlass_global * k_cutlass_global)
        ext.persistent_mainloop_owner_softmax_stage(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            p_packed,
            p_scales,
            row_m,
            row_l,
            workspace,
            float(qk_alpha.item()),
            0,
            0,
        )
        torch.cuda.synchronize()
        qk_official = torch.empty(
            (128, k_cutlass.shape[0]), device=device, dtype=torch.bfloat16
        )
        tactic = min(2, int(metadata["runner_tactic_count"]) - 1)
        ext.cutlass_runner_fp4_gemm(
            q_cutlass[:128].contiguous(),
            k_cutlass.contiguous(),
            q_cutlass_scales[:128].contiguous(),
            k_cutlass_scales.contiguous(),
            qk_alpha,
            qk_official,
            workspace,
            tactic,
        )
        torch.cuda.synchronize()
        logits = qk_official[:, :128].float() * (HEAD_DIM**-0.5)
        row_m_ref = logits.amax(dim=-1)
        p_ref = torch.exp(logits - row_m_ref[:, None])
        row_l_ref = p_ref.sum(dim=-1)
        p_dequant = nvfp4_rowmajor_to_fp32(p_packed, p_scales) / PROB_GLOBAL_SCALE
        p_delta = (p_dequant - p_ref).abs()
        row_m_delta = (row_m - row_m_ref).abs()
        row_l_delta = (row_l - row_l_ref).abs()
        p_cos = torch.sum(p_dequant * p_ref) / torch.clamp(
            torch.linalg.vector_norm(p_dequant) * torch.linalg.vector_norm(p_ref),
            min=1.0e-20,
        )
        result = {
            "persistent_owner_softmax_stage": True,
            "finite": bool(
                torch.isfinite(p_dequant).all().item()
                and torch.isfinite(row_m).all().item()
                and torch.isfinite(row_l).all().item()
            ),
            "p_mean_abs": float(p_delta.mean().item()),
            "p_max_abs": float(p_delta.max().item()),
            "p_cosine": float(p_cos.item()),
            "row_m_max_abs": float(row_m_delta.max().item()),
            "row_l_max_abs": float(row_l_delta.max().item()),
        }
        print(result)
        return
    if args.persistent_owner_pv_group_check_only:
        q_cutlass, q_cutlass_scales, q_cutlass_global = quantize_cutlass(
            q.reshape(Q_LEN * GROUP, HEAD_DIM)
        )
        k_cutlass, k_cutlass_scales, k_cutlass_global = quantize_cutlass(
            k_ref_f32.to(torch.bfloat16)
        )
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv_cutlass, v_pv_cutlass_scales, v_pv_cutlass_global = quantize_cutlass(
            v_ref_f32.T.contiguous().to(torch.bfloat16)
        )
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        out_group = torch.empty((128, 128), device=device, dtype=torch.float32)
        p_packed = torch.empty((128, 64), device=device, dtype=torch.uint8)
        p_scales = torch.empty((128, 8), device=device, dtype=torch.uint8)
        row_m = torch.empty((128,), device=device, dtype=torch.float32)
        row_l = torch.empty((128,), device=device, dtype=torch.float32)
        qk_alpha = 1.0 / (q_cutlass_global * k_cutlass_global)
        out_group_idx = 0
        ext.persistent_mainloop_owner_pv_group_stage(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            v_pv_cutlass,
            v_pv_cutlass_scales,
            out_group,
            p_packed,
            p_scales,
            row_m,
            row_l,
            workspace,
            float(qk_alpha.item()),
            0,
            0,
            out_group_idx,
        )
        torch.cuda.synchronize()

        pv_alpha = 1.0 / v_pv_cutlass_global
        group_start = out_group_idx * 128
        group_stop = group_start + 128
        full_out = torch.empty((128, HEAD_DIM), device=device, dtype=torch.float32)
        ext.fused_cutlass_tma_qk_tma_pv_128tile(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            v_pv_cutlass,
            v_pv_cutlass_scales,
            full_out,
            workspace,
            float(qk_alpha.item()),
            float(pv_alpha.item()),
            0,
            0,
        )
        torch.cuda.synchronize()

        owner_norm = out_group * (
            pv_alpha / (PROB_GLOBAL_SCALE * torch.clamp(row_l[:, None], min=1.0e-20))
        )
        full_ref = full_out[:, group_start:group_stop]
        full_delta = (owner_norm - full_ref).abs()
        full_cos = torch.sum(owner_norm * full_ref) / torch.clamp(
            torch.linalg.vector_norm(owner_norm) * torch.linalg.vector_norm(full_ref),
            min=1.0e-20,
        )

        p_dequant = nvfp4_rowmajor_to_fp32(p_packed, p_scales) / PROB_GLOBAL_SCALE
        p_norm = p_dequant / torch.clamp(row_l[:, None], min=1.0e-20)
        exact_ref = torch.matmul(
            p_norm.float(), v_ref_f32[:128, group_start:group_stop].float()
        )
        exact_delta = (owner_norm - exact_ref).abs()
        exact_cos = torch.sum(owner_norm * exact_ref) / torch.clamp(
            torch.linalg.vector_norm(owner_norm) * torch.linalg.vector_norm(exact_ref),
            min=1.0e-20,
        )

        result = {
            "persistent_owner_pv_group_stage": True,
            "finite": bool(
                torch.isfinite(out_group).all().item()
                and torch.isfinite(p_dequant).all().item()
                and torch.isfinite(row_m).all().item()
                and torch.isfinite(row_l).all().item()
            ),
            "owner_vs_full_tma_mean_abs": float(full_delta.mean().item()),
            "owner_vs_full_tma_max_abs": float(full_delta.max().item()),
            "owner_vs_full_tma_cosine": float(full_cos.item()),
            "owner_vs_exact_mean_abs": float(exact_delta.mean().item()),
            "owner_vs_exact_max_abs": float(exact_delta.max().item()),
            "owner_vs_exact_cosine": float(exact_cos.item()),
        }
        print(result)
        return
    if args.persistent_owner_full_tile_check_only:
        q_cutlass, q_cutlass_scales, q_cutlass_global = quantize_cutlass(
            q.reshape(Q_LEN * GROUP, HEAD_DIM)
        )
        k_cutlass, k_cutlass_scales, k_cutlass_global = quantize_cutlass(
            k_ref_f32.to(torch.bfloat16)
        )
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv_cutlass, v_pv_cutlass_scales, v_pv_cutlass_global = quantize_cutlass(
            v_ref_f32.T.contiguous().to(torch.bfloat16)
        )
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        owner_out = torch.empty((128, HEAD_DIM), device=device, dtype=torch.float32)
        full_ref = torch.empty_like(owner_out)
        qk_alpha = 1.0 / (q_cutlass_global * k_cutlass_global)
        pv_alpha = 1.0 / v_pv_cutlass_global
        ext.persistent_mainloop_owner_full_tile_stage(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            v_pv_cutlass,
            v_pv_cutlass_scales,
            owner_out,
            workspace,
            float(qk_alpha.item()),
            float(pv_alpha.item()),
            0,
            0,
        )
        ext.fused_cutlass_tma_qk_tma_pv_128tile(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            v_pv_cutlass,
            v_pv_cutlass_scales,
            full_ref,
            workspace,
            float(qk_alpha.item()),
            float(pv_alpha.item()),
            0,
            0,
        )
        torch.cuda.synchronize()
        delta = (owner_out - full_ref).abs()
        cos = torch.sum(owner_out * full_ref) / torch.clamp(
            torch.linalg.vector_norm(owner_out) * torch.linalg.vector_norm(full_ref),
            min=1.0e-20,
        )
        result = {
            "persistent_owner_full_tile_stage": True,
            "finite": bool(torch.isfinite(owner_out).all().item()),
            "owner_vs_full_tma_mean_abs": float(delta.mean().item()),
            "owner_vs_full_tma_max_abs": float(delta.max().item()),
            "owner_vs_full_tma_cosine": float(cos.item()),
        }
        if args.bench:
            result["bench_persistent_owner_full_tile"] = event_ms(
                lambda: ext.persistent_mainloop_owner_full_tile_stage(
                    q_cutlass,
                    q_cutlass_scales,
                    k_cutlass,
                    k_cutlass_scales,
                    v_pv_cutlass,
                    v_pv_cutlass_scales,
                    owner_out,
                    workspace,
                    float(qk_alpha.item()),
                    float(pv_alpha.item()),
                    0,
                    0,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
            result["bench_existing_full_tma_tile"] = event_ms(
                lambda: ext.fused_cutlass_tma_qk_tma_pv_128tile(
                    q_cutlass,
                    q_cutlass_scales,
                    k_cutlass,
                    k_cutlass_scales,
                    v_pv_cutlass,
                    v_pv_cutlass_scales,
                    full_ref,
                    workspace,
                    float(qk_alpha.item()),
                    float(pv_alpha.item()),
                    0,
                    0,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        print(result)
        return
    if (
        args.persistent_owner_group_online_check_only
        or args.persistent_owner_group_online_register_check_only
        or args.persistent_owner_group_online_register_q_check_only
    ):
        if args.persistent_owner_group_online_register_q_check_only:
            stage_name = "persistent_owner_group_online_register_q_stage"
            stage_fn = ext.persistent_mainloop_owner_group_online_register_q_stage
        elif args.persistent_owner_group_online_register_check_only:
            stage_name = "persistent_owner_group_online_register_stage"
            stage_fn = ext.persistent_mainloop_owner_group_online_register_stage
        else:
            stage_name = "persistent_owner_group_online_stage"
            stage_fn = ext.persistent_mainloop_owner_group_online_stage
        num_tiles = args.persistent_owner_online_kv_tiles
        if num_tiles <= 0 or num_tiles > (k_ref_f32.shape[0] // 128):
            raise ValueError("--persistent-owner-online-kv-tiles out of range")
        q_cutlass, q_cutlass_scales, q_cutlass_global = quantize_cutlass(
            q.reshape(Q_LEN * GROUP, HEAD_DIM)
        )
        k_cutlass, k_cutlass_scales, k_cutlass_global = quantize_cutlass(
            k_ref_f32.to(torch.bfloat16)
        )
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv_cutlass, v_pv_cutlass_scales, v_pv_cutlass_global = quantize_cutlass(
            v_ref_f32.T.contiguous().to(torch.bfloat16)
        )
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        out_group = torch.empty((128, 128), device=device, dtype=torch.float32)
        qk_alpha = 1.0 / (q_cutlass_global * k_cutlass_global)
        pv_alpha = 1.0 / v_pv_cutlass_global
        out_group_idx = 0
        stage_fn(
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
            num_tiles,
            out_group_idx,
        )
        torch.cuda.synchronize()

        kv_rows = num_tiles * 128
        tactic = min(2, int(metadata["runner_tactic_count"]) - 1)
        qk_ref = torch.empty((128, kv_rows), device=device, dtype=torch.bfloat16)
        ext.cutlass_runner_fp4_gemm(
            q_cutlass[:128].contiguous(),
            k_cutlass[:kv_rows].contiguous(),
            q_cutlass_scales[:128].contiguous(),
            k_cutlass_scales[:kv_rows].contiguous(),
            qk_alpha,
            qk_ref,
            workspace,
            tactic,
        )
        torch.cuda.synchronize()
        probs = torch.softmax(qk_ref.float() / (HEAD_DIM**0.5), dim=-1)
        group_start = out_group_idx * 128
        group_stop = group_start + 128
        exact_ref = torch.matmul(
            probs.float(), v_ref_f32[:kv_rows, group_start:group_stop].float()
        )
        delta = (out_group - exact_ref).abs()
        cos = torch.sum(out_group * exact_ref) / torch.clamp(
            torch.linalg.vector_norm(out_group) * torch.linalg.vector_norm(exact_ref),
            min=1.0e-20,
        )
        result = {
            stage_name: True,
            "num_kv_tiles": num_tiles,
            "finite": bool(torch.isfinite(out_group).all().item()),
            "owner_vs_exact_mean_abs": float(delta.mean().item()),
            "owner_vs_exact_max_abs": float(delta.max().item()),
            "owner_vs_exact_cosine": float(cos.item()),
        }
        if args.bench:
            result[f"bench_{stage_name}"] = event_ms(
                lambda: stage_fn(
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
                    num_tiles,
                    out_group_idx,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        print(result)
        return
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
        q_value = E2M1_VALUES[args.smem_atom_q_code].item()
        k_value = E2M1_VALUES[args.smem_atom_k_code].item()
        q_actual_f32 = torch.full_like(q_actual_f32, q_value)
        k_ref_f32 = torch.full_like(k_ref_f32, k_value)
    if args.smem_atom_ones:
        q_actual.fill_(0x22)
        q_scales_actual.fill_(0x38)
        k.fill_(0x22)
        k_scales.fill_(0x38)
        q_actual_f32 = torch.ones_like(q_actual_f32)
        k_ref_f32 = torch.ones_like(k_ref_f32)
    if args.fused_tma_qk_pv_128_check_only:
        q_cutlass, q_cutlass_scales, q_cutlass_global = quantize_cutlass(
            q.reshape(Q_LEN * GROUP, HEAD_DIM)
        )
        k_cutlass, k_cutlass_scales, k_cutlass_global = quantize_cutlass(
            k_ref_f32.to(torch.bfloat16)
        )
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        out = torch.empty((128, HEAD_DIM), device=device, dtype=torch.float32)
        qk_alpha = 1.0 / (q_cutlass_global * k_cutlass_global)
        ext.fused_cutlass_tma_qk_smem_pv_128tile(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            v_pv,
            v_pv_scales,
            out,
            workspace,
            float(qk_alpha.item()),
            0,
            0,
        )
        torch.cuda.synchronize()

        qk_official = torch.empty(
            (128, k_cutlass.shape[0]), device=device, dtype=torch.bfloat16
        )
        tactic = min(2, int(metadata["runner_tactic_count"]) - 1)
        ext.cutlass_runner_fp4_gemm(
            q_cutlass[:128].contiguous(),
            k_cutlass.contiguous(),
            q_cutlass_scales[:128].contiguous(),
            k_cutlass_scales.contiguous(),
            qk_alpha,
            qk_official,
            workspace,
            tactic,
        )
        torch.cuda.synchronize()
        p_ref = torch.softmax(qk_official[:, :128].float() / (HEAD_DIM**0.5), dim=-1)
        out_ref = torch.matmul(p_ref, v_pv_ref_f32[:, :128].float().T)
        delta = (out - out_ref).abs()
        ref_norm = torch.linalg.vector_norm(out_ref)
        out_norm = torch.linalg.vector_norm(out)
        cos = torch.sum(out * out_ref) / torch.clamp(ref_norm * out_norm, min=1.0e-20)
        result = {
            "fused_tma_qk_pv_128_finite": bool(torch.isfinite(out).all().item()),
            "fused_tma_qk_pv_128_mean_abs": float(delta.mean().item()),
            "fused_tma_qk_pv_128_max_abs": float(delta.max().item()),
            "fused_tma_qk_pv_128_cosine": float(cos.item()),
        }
        if args.bench:
            result["bench_fused_tma_qk_pv_128"] = event_ms(
                lambda: ext.fused_cutlass_tma_qk_smem_pv_128tile(
                    q_cutlass,
                    q_cutlass_scales,
                    k_cutlass,
                    k_cutlass_scales,
                    v_pv,
                    v_pv_scales,
                    out,
                    workspace,
                    float(qk_alpha.item()),
                    0,
                    0,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        print(result)
        return
    if args.fused_tma_qk_tma_pv_128_check_only:
        q_cutlass, q_cutlass_scales, q_cutlass_global = quantize_cutlass(
            q.reshape(Q_LEN * GROUP, HEAD_DIM)
        )
        k_cutlass, k_cutlass_scales, k_cutlass_global = quantize_cutlass(
            k_ref_f32.to(torch.bfloat16)
        )
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
        v_pv_cutlass, v_pv_cutlass_scales, v_pv_cutlass_global = quantize_cutlass(
            v_ref_f32.T.contiguous().to(torch.bfloat16)
        )
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        out = torch.empty((128, HEAD_DIM), device=device, dtype=torch.float32)
        qk_alpha = 1.0 / (q_cutlass_global * k_cutlass_global)
        ext.fused_cutlass_tma_qk_tma_pv_128tile(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            v_pv_cutlass,
            v_pv_cutlass_scales,
            out,
            workspace,
            float(qk_alpha.item()),
            float((1.0 / v_pv_cutlass_global).item()),
            0,
            0,
        )
        torch.cuda.synchronize()
        smem_out = torch.empty_like(out)
        ext.fused_cutlass_tma_qk_smem_pv_128tile(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            v_pv,
            v_pv_scales,
            smem_out,
            workspace,
            float(qk_alpha.item()),
            0,
            0,
        )
        torch.cuda.synchronize()

        qk_official = torch.empty(
            (128, k_cutlass.shape[0]), device=device, dtype=torch.bfloat16
        )
        tactic = min(2, int(metadata["runner_tactic_count"]) - 1)
        ext.cutlass_runner_fp4_gemm(
            q_cutlass[:128].contiguous(),
            k_cutlass.contiguous(),
            q_cutlass_scales[:128].contiguous(),
            k_cutlass_scales.contiguous(),
            qk_alpha,
            qk_official,
            workspace,
            tactic,
        )
        torch.cuda.synchronize()
        p_ref = torch.softmax(qk_official[:, :128].float() / (HEAD_DIM**0.5), dim=-1)
        out_ref = torch.matmul(p_ref, v_pv_ref_f32[:, :128].float().T)
        delta = (out - out_ref).abs()
        smem_delta = (smem_out - out_ref).abs()
        tma_vs_smem = (out - smem_out).abs()
        ref_norm = torch.linalg.vector_norm(out_ref)
        out_norm = torch.linalg.vector_norm(out)
        cos = torch.sum(out * out_ref) / torch.clamp(ref_norm * out_norm, min=1.0e-20)
        group_metrics = []
        for group_idx in range(HEAD_DIM // 128):
            start = group_idx * 128
            stop = start + 128
            group_out = out[:, start:stop]
            group_ref = out_ref[:, start:stop]
            group_ref_norm = torch.linalg.vector_norm(group_ref)
            group_out_norm = torch.linalg.vector_norm(group_out)
            group_cos = torch.sum(group_out * group_ref) / torch.clamp(
                group_ref_norm * group_out_norm,
                min=1.0e-20,
            )
            group_metrics.append(
                {
                    "group": group_idx,
                    "mean_abs": float((group_out - group_ref).abs().mean().item()),
                    "max_abs": float((group_out - group_ref).abs().max().item()),
                    "cosine": float(group_cos.item()),
                    "tma_vs_smem_mean_abs": float(
                        tma_vs_smem[:, start:stop].mean().item()
                    ),
                }
            )
        result = {
            "fused_tma_qk_tma_pv_128_finite": bool(torch.isfinite(out).all().item()),
            "fused_tma_qk_tma_pv_128_mean_abs": float(delta.mean().item()),
            "fused_tma_qk_tma_pv_128_max_abs": float(delta.max().item()),
            "fused_tma_qk_tma_pv_128_cosine": float(cos.item()),
            "smem_pv_reference_mean_abs": float(smem_delta.mean().item()),
            "tma_vs_smem_mean_abs": float(tma_vs_smem.mean().item()),
            "tma_vs_smem_max_abs": float(tma_vs_smem.max().item()),
            "fused_tma_qk_tma_pv_128_groups": group_metrics,
        }
        if args.bench:
            result["bench_fused_tma_qk_tma_pv_128"] = event_ms(
                lambda: ext.fused_cutlass_tma_qk_tma_pv_128tile(
                    q_cutlass,
                    q_cutlass_scales,
                    k_cutlass,
                    k_cutlass_scales,
                    v_pv_cutlass,
                    v_pv_cutlass_scales,
                    out,
                    workspace,
                    float(qk_alpha.item()),
                    float((1.0 / v_pv_cutlass_global).item()),
                    0,
                    0,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        print(result)
        return
    if args.smem_atom_block_check_only:
        qk_smem_atom_block = torch.empty(
            (128, k.shape[0]), device=device, dtype=torch.float32
        )
        ext.qk_cutlass_smem_atom_block(
            q_actual,
            q_scales_actual,
            k,
            k_scales,
            qk_smem_atom_block,
        )
        torch.cuda.synchronize()
        qk_smem_atom_block_ref = torch.matmul(
            q_actual_f32[:128].float(),
            k_ref_f32.float().T,
        )
        qk_smem_atom_block_delta = (
            qk_smem_atom_block - qk_smem_atom_block_ref
        ).abs()
        qk_smem_atom_block_ref_norm = torch.linalg.vector_norm(
            qk_smem_atom_block_ref
        )
        qk_smem_atom_block_norm = torch.linalg.vector_norm(qk_smem_atom_block)
        qk_smem_atom_block_cos = torch.sum(
            qk_smem_atom_block * qk_smem_atom_block_ref
        ) / torch.clamp(
            qk_smem_atom_block_ref_norm * qk_smem_atom_block_norm,
            min=1.0e-20,
        )
        result = {
            "qk_smem_atom_block_finite": bool(
                torch.isfinite(qk_smem_atom_block).all().item()
            ),
            "qk_smem_atom_block_mean_abs": float(
                qk_smem_atom_block_delta.mean().item()
            ),
            "qk_smem_atom_block_max_abs": float(
                qk_smem_atom_block_delta.max().item()
            ),
            "qk_smem_atom_block_cosine": float(qk_smem_atom_block_cos.item()),
            "qk_smem_atom_block_norm_ratio": float(
                (
                    qk_smem_atom_block_norm
                    / torch.clamp(qk_smem_atom_block_ref_norm, min=1.0e-20)
                ).item()
            ),
        }
        if args.bench:
            result["bench_qk_smem_atom_block_128x32768"] = event_ms(
                lambda: ext.qk_cutlass_smem_atom_block(
                    q_actual,
                    q_scales_actual,
                    k,
                    k_scales,
                    qk_smem_atom_block,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        print(result)
        return
    if args.pv_smem_atom_tile_check_only:
        kv_base = 0
        out_col_base = 0
        p_ref = torch.rand((128, k.shape[0]), device=device, dtype=torch.float32)
        p_scaled = p_ref * PROB_GLOBAL_SCALE
        p_packed, p_scales = fp32_to_nvfp4_rowmajor(p_scaled)
        p_dequant_scaled = nvfp4_rowmajor_to_fp32(p_packed, p_scales)
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
        pv_smem_atom_tile = torch.empty(
            (128, 128), device=device, dtype=torch.float32
        )
        ext.pv_cutlass_smem_atom_tile(
            p_packed,
            p_scales,
            v_pv,
            v_pv_scales,
            pv_smem_atom_tile,
            kv_base,
            out_col_base,
        )
        torch.cuda.synchronize()
        pv_smem_atom_ref = torch.matmul(
            p_dequant_scaled[:, kv_base : kv_base + 256].float(),
            v_pv_ref_f32[
                out_col_base : out_col_base + 128,
                kv_base : kv_base + 256,
            ].float().T,
        )
        pv_smem_atom_delta = (pv_smem_atom_tile - pv_smem_atom_ref).abs()
        pv_smem_atom_ref_norm = torch.linalg.vector_norm(pv_smem_atom_ref)
        pv_smem_atom_norm = torch.linalg.vector_norm(pv_smem_atom_tile)
        pv_smem_atom_cos = torch.sum(
            pv_smem_atom_tile * pv_smem_atom_ref
        ) / torch.clamp(
            pv_smem_atom_ref_norm * pv_smem_atom_norm,
            min=1.0e-20,
        )
        result = {
            "pv_smem_atom_tile_finite": bool(
                torch.isfinite(pv_smem_atom_tile).all().item()
            ),
            "pv_smem_atom_tile_mean_abs": float(
                pv_smem_atom_delta.mean().item()
            ),
            "pv_smem_atom_tile_max_abs": float(pv_smem_atom_delta.max().item()),
            "pv_smem_atom_tile_cosine": float(pv_smem_atom_cos.item()),
            "pv_smem_atom_tile_norm_ratio": float(
                (
                    pv_smem_atom_norm
                    / torch.clamp(pv_smem_atom_ref_norm, min=1.0e-20)
                ).item()
            ),
        }
        if args.bench:
            result["bench_pv_smem_atom_tile_128x128x256"] = event_ms(
                lambda: ext.pv_cutlass_smem_atom_tile(
                    p_packed,
                    p_scales,
                    v_pv,
                    v_pv_scales,
                    pv_smem_atom_tile,
                    kv_base,
                    out_col_base,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        print(result)
        return
    if args.fused_smem_one_kv_tile_check_only:
        kv_base = 0
        out_col_base = 0
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
        fused_smem_one_tile = torch.empty(
            (128, 128), device=device, dtype=torch.float32
        )
        ext.fused_cutlass_smem_one_kv_tile(
            q_actual,
            q_scales_actual,
            k,
            k_scales,
            v_pv,
            v_pv_scales,
            fused_smem_one_tile,
            kv_base,
            out_col_base,
        )
        torch.cuda.synchronize()
        qk_ref = torch.matmul(
            q_actual_f32[:128].float(),
            k_ref_f32[kv_base : kv_base + 256].float().T,
        )
        unnorm = torch.exp(qk_ref / (HEAD_DIM**0.5) - (qk_ref / (HEAD_DIM**0.5)).amax(dim=-1, keepdim=True))
        row_l = unnorm.sum(dim=-1, keepdim=True)
        p_unorm_scaled = unnorm * PROB_GLOBAL_SCALE
        p_packed, p_scales = fp32_to_nvfp4_rowmajor(p_unorm_scaled)
        p_dequant_unorm = nvfp4_rowmajor_to_fp32(p_packed, p_scales) / PROB_GLOBAL_SCALE
        fused_smem_quant_ref = torch.matmul(
            (p_dequant_unorm / row_l).float(),
            v_pv_ref_f32[
                out_col_base : out_col_base + 128,
                kv_base : kv_base + 256,
            ].float().T,
        )
        fused_smem_exact_ref = torch.matmul(
            (unnorm / row_l).float(),
            v_pv_ref_f32[
                out_col_base : out_col_base + 128,
                kv_base : kv_base + 256,
            ].float().T,
        )
        fused_smem_quant_delta = (
            fused_smem_one_tile - fused_smem_quant_ref
        ).abs()
        fused_smem_exact_delta = (
            fused_smem_one_tile - fused_smem_exact_ref
        ).abs()
        fused_smem_quant_ref_norm = torch.linalg.vector_norm(fused_smem_quant_ref)
        fused_smem_one_norm = torch.linalg.vector_norm(fused_smem_one_tile)
        fused_smem_quant_cos = torch.sum(
            fused_smem_one_tile * fused_smem_quant_ref
        ) / torch.clamp(
            fused_smem_quant_ref_norm * fused_smem_one_norm,
            min=1.0e-20,
        )
        result = {
            "fused_smem_one_tile_finite": bool(
                torch.isfinite(fused_smem_one_tile).all().item()
            ),
            "fused_smem_one_tile_vs_quant_mean_abs": float(
                fused_smem_quant_delta.mean().item()
            ),
            "fused_smem_one_tile_vs_quant_max_abs": float(
                fused_smem_quant_delta.max().item()
            ),
            "fused_smem_one_tile_vs_quant_cosine": float(
                fused_smem_quant_cos.item()
            ),
            "fused_smem_one_tile_vs_exact_mean_abs": float(
                fused_smem_exact_delta.mean().item()
            ),
            "fused_smem_one_tile_vs_exact_max_abs": float(
                fused_smem_exact_delta.max().item()
            ),
        }
        if args.bench:
            result["bench_fused_smem_one_kv_tile_128x128x256"] = event_ms(
                lambda: ext.fused_cutlass_smem_one_kv_tile(
                    q_actual,
                    q_scales_actual,
                    k,
                    k_scales,
                    v_pv,
                    v_pv_scales,
                    fused_smem_one_tile,
                    kv_base,
                    out_col_base,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        print(result)
        return
    if args.fused_smem_online_col_check_only:
        num_kv_tiles = args.fused_smem_online_kv_tiles
        if num_kv_tiles <= 0 or num_kv_tiles > k.shape[0] // 256:
            raise ValueError("--fused-smem-online-kv-tiles out of range")
        out_col_base = 0
        kv_tokens = num_kv_tiles * 256
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
        fused_online = torch.empty((128, 128), device=device, dtype=torch.float32)
        fused_online_scratch = torch.empty_like(fused_online)
        ext.fused_cutlass_smem_online_col_tile(
            q_actual,
            q_scales_actual,
            k,
            k_scales,
            v_pv,
            v_pv_scales,
            fused_online,
            fused_online_scratch,
            out_col_base,
            num_kv_tiles,
        )
        torch.cuda.synchronize()

        online_ref = torch.zeros_like(fused_online)
        online_m = torch.full((128, 1), -float("inf"), device=device)
        online_l = torch.zeros((128, 1), device=device)
        q_ref = q_actual_f32[:128].float()
        v_ref_cols = v_pv_ref_f32[out_col_base : out_col_base + 128].float()
        for tile in range(num_kv_tiles):
            start = tile * 256
            end = start + 256
            scores = (
                torch.matmul(q_ref, k_ref_f32[start:end].float().T)
                / (HEAD_DIM**0.5)
            )
            tile_m = scores.amax(dim=-1, keepdim=True)
            new_m = torch.maximum(online_m, tile_m)
            alpha = torch.where(
                online_l > 0,
                torch.exp(online_m - new_m),
                torch.zeros_like(online_l),
            )
            probs_unorm = torch.exp(scores - new_m)
            tile_l = probs_unorm.sum(dim=-1, keepdim=True)
            online_ref *= alpha
            online_l = online_l * alpha + tile_l
            online_m = new_m
            p_packed, p_scales = fp32_to_nvfp4_rowmajor(
                probs_unorm * PROB_GLOBAL_SCALE
            )
            p_dequant_unorm = (
                nvfp4_rowmajor_to_fp32(p_packed, p_scales) / PROB_GLOBAL_SCALE
            )
            online_ref += torch.matmul(
                p_dequant_unorm.float(),
                v_ref_cols[:, start:end].T,
            )
        online_ref = online_ref / online_l

        exact_scores = (
            torch.matmul(q_ref, k_ref_f32[:kv_tokens].float().T)
            / (HEAD_DIM**0.5)
        )
        exact_p = torch.softmax(exact_scores, dim=-1)
        exact_ref = torch.matmul(
            exact_p,
            v_ref_cols[:, :kv_tokens].T,
        )
        online_delta = (fused_online - online_ref).abs()
        exact_delta = (fused_online - exact_ref).abs()
        online_ref_norm = torch.linalg.vector_norm(online_ref)
        online_norm = torch.linalg.vector_norm(fused_online)
        online_cos = torch.sum(fused_online * online_ref) / torch.clamp(
            online_ref_norm * online_norm,
            min=1.0e-20,
        )
        result = {
            "fused_smem_online_kv_tiles": num_kv_tiles,
            "fused_smem_online_finite": bool(torch.isfinite(fused_online).all().item()),
            "fused_smem_online_vs_quant_mean_abs": float(
                online_delta.mean().item()
            ),
            "fused_smem_online_vs_quant_max_abs": float(online_delta.max().item()),
            "fused_smem_online_vs_quant_cosine": float(online_cos.item()),
            "fused_smem_online_vs_exact_mean_abs": float(exact_delta.mean().item()),
            "fused_smem_online_vs_exact_max_abs": float(exact_delta.max().item()),
        }
        if args.bench:
            result["bench_fused_smem_online_col_tile"] = event_ms(
                lambda: ext.fused_cutlass_smem_online_col_tile(
                    q_actual,
                    q_scales_actual,
                    k,
                    k_scales,
                    v_pv,
                    v_pv_scales,
                    fused_online,
                    fused_online_scratch,
                    out_col_base,
                    num_kv_tiles,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        print(result)
        return
    if args.fused_smem_online_full_width_check_only:
        num_kv_tiles = args.fused_smem_online_kv_tiles
        if num_kv_tiles <= 0 or num_kv_tiles > k.shape[0] // 256:
            raise ValueError("--fused-smem-online-kv-tiles out of range")
        kv_tokens = num_kv_tiles * 256
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
        fused_full = torch.empty((128, HEAD_DIM), device=device, dtype=torch.float32)
        fused_full_scratch = torch.empty((128, 128), device=device, dtype=torch.float32)
        ext.fused_cutlass_smem_online_full_width(
            q_actual,
            q_scales_actual,
            k,
            k_scales,
            v_pv,
            v_pv_scales,
            fused_full,
            fused_full_scratch,
            num_kv_tiles,
        )
        torch.cuda.synchronize()

        online_ref = torch.zeros_like(fused_full)
        online_m = torch.full((128, 1), -float("inf"), device=device)
        online_l = torch.zeros((128, 1), device=device)
        q_ref = q_actual_f32[:128].float()
        v_ref_all = v_pv_ref_f32.float()
        for tile in range(num_kv_tiles):
            start = tile * 256
            end = start + 256
            scores = (
                torch.matmul(q_ref, k_ref_f32[start:end].float().T)
                / (HEAD_DIM**0.5)
            )
            tile_m = scores.amax(dim=-1, keepdim=True)
            new_m = torch.maximum(online_m, tile_m)
            alpha = torch.where(
                online_l > 0,
                torch.exp(online_m - new_m),
                torch.zeros_like(online_l),
            )
            probs_unorm = torch.exp(scores - new_m)
            tile_l = probs_unorm.sum(dim=-1, keepdim=True)
            online_ref *= alpha
            online_l = online_l * alpha + tile_l
            online_m = new_m
            p_packed, p_scales = fp32_to_nvfp4_rowmajor(
                probs_unorm * PROB_GLOBAL_SCALE
            )
            p_dequant_unorm = (
                nvfp4_rowmajor_to_fp32(p_packed, p_scales) / PROB_GLOBAL_SCALE
            )
            online_ref += torch.matmul(
                p_dequant_unorm.float(),
                v_ref_all[:, start:end].T,
            )
        online_ref = online_ref / online_l

        exact_scores = (
            torch.matmul(q_ref, k_ref_f32[:kv_tokens].float().T)
            / (HEAD_DIM**0.5)
        )
        exact_p = torch.softmax(exact_scores, dim=-1)
        exact_ref = torch.matmul(
            exact_p,
            v_ref_all[:, :kv_tokens].T,
        )
        online_delta = (fused_full - online_ref).abs()
        exact_delta = (fused_full - exact_ref).abs()
        online_ref_norm = torch.linalg.vector_norm(online_ref)
        online_norm = torch.linalg.vector_norm(fused_full)
        online_cos = torch.sum(fused_full * online_ref) / torch.clamp(
            online_ref_norm * online_norm,
            min=1.0e-20,
        )
        result = {
            "fused_smem_online_full_width_kv_tiles": num_kv_tiles,
            "fused_smem_online_full_width_finite": bool(
                torch.isfinite(fused_full).all().item()
            ),
            "fused_smem_online_full_width_vs_quant_mean_abs": float(
                online_delta.mean().item()
            ),
            "fused_smem_online_full_width_vs_quant_max_abs": float(
                online_delta.max().item()
            ),
            "fused_smem_online_full_width_vs_quant_cosine": float(
                online_cos.item()
            ),
            "fused_smem_online_full_width_vs_exact_mean_abs": float(
                exact_delta.mean().item()
            ),
            "fused_smem_online_full_width_vs_exact_max_abs": float(
                exact_delta.max().item()
            ),
        }
        if args.bench:
            result["bench_fused_smem_online_full_width"] = event_ms(
                lambda: ext.fused_cutlass_smem_online_full_width(
                    q_actual,
                    q_scales_actual,
                    k,
                    k_scales,
                    v_pv,
                    v_pv_scales,
                    fused_full,
                    fused_full_scratch,
                    num_kv_tiles,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        print(result)
        return
    if (
        args.fused_smem_online_split_full_width_check_only
        or args.fused_smem_online_split_k128_full_width_check_only
        or args.fused_tma_pv_online_split_k128_full_width_check_only
        or args.fused_tma_qk_tma_pv_online_split_k128_full_width_check_only
    ):
        use_tma_pv = (
            args.fused_tma_pv_online_split_k128_full_width_check_only
            or args.fused_tma_qk_tma_pv_online_split_k128_full_width_check_only
        )
        use_k128 = args.fused_smem_online_split_k128_full_width_check_only or use_tma_pv
        num_splits = args.fused_smem_online_splits
        q_tiles = args.fused_smem_online_q_tiles
        kv_tile_tokens = 128 if use_k128 else 256
        total_kv_tiles = k.shape[0] // kv_tile_tokens
        if num_splits <= 0 or total_kv_tiles % num_splits != 0:
            raise ValueError(
                "--fused-smem-online-splits must divide "
                f"{total_kv_tiles}"
            )
        if q_tiles <= 0 or q_tiles > q_actual.shape[0] // 128:
            raise ValueError("--fused-smem-online-q-tiles out of range")
        q_rows = q_tiles * 128
        kv_tokens = k.shape[0]
        v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
        v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
        v_pv_cutlass, v_pv_cutlass_scales, v_pv_cutlass_global = quantize_cutlass(
            v_ref_f32.T.contiguous().to(torch.bfloat16)
        )
        q_cutlass = q_cutlass_scales = qk_alpha = None
        k_cutlass = k_cutlass_scales = None
        if use_tma_pv:
            q_cutlass, q_cutlass_scales, q_cutlass_global = quantize_cutlass(
                q.reshape(Q_LEN * GROUP, HEAD_DIM)
            )
            k_cutlass, k_cutlass_scales, k_cutlass_global = quantize_cutlass(
                k_ref_f32.to(torch.bfloat16)
            )
            qk_alpha = 1.0 / (q_cutlass_global * k_cutlass_global)
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        partial_out = torch.empty(
            (q_tiles, num_splits, 128, HEAD_DIM),
            device=device,
            dtype=torch.float32,
        )
        partial_m = torch.empty(
            (q_tiles, num_splits, 128), device=device, dtype=torch.float32
        )
        partial_l = torch.empty_like(partial_m)
        pv_scratch = torch.empty(
            (q_tiles, num_splits, 128, HEAD_DIM)
            if use_tma_pv
            else (q_tiles, num_splits, 128, 128),
            device=device,
            dtype=torch.float32,
        )
        fused_split = torch.empty((q_rows, HEAD_DIM), device=device, dtype=torch.float32)
        if use_tma_pv:
            split_kernel = ext.fused_cutlass_tma_qk_tma_pv_online_full_width_split_k128

            def run_split_kernel() -> None:
                split_kernel(
                    q_cutlass,
                    q_cutlass_scales,
                    k_cutlass,
                    k_cutlass_scales,
                    v_pv_cutlass,
                    v_pv_cutlass_scales,
                    partial_out,
                    partial_m,
                    partial_l,
                    pv_scratch,
                    fused_split,
                    workspace,
                    float(qk_alpha.item()),
                    float((1.0 / v_pv_cutlass_global).item()),
                    num_splits,
                    q_tiles,
                )

        else:
            split_kernel = (
                ext.fused_cutlass_smem_online_full_width_split_k128
                if use_k128
                else ext.fused_cutlass_smem_online_full_width_split
            )

            def run_split_kernel() -> None:
                split_kernel(
                    q_actual,
                    q_scales_actual,
                    k,
                    k_scales,
                    v_pv,
                    v_pv_scales,
                    partial_out,
                    partial_m,
                    partial_l,
                    pv_scratch,
                    fused_split,
                    num_splits,
                    q_tiles,
                )

        run_split_kernel()
        torch.cuda.synchronize()
        split_backend = (
            "tma_qk_tma_pv_128tile_partials"
            if use_tma_pv
            else ("smem_k128_online" if use_k128 else "smem_k256_online")
        )

        if args.fused_split_kernel_only:
            result = {
                "fused_split_kernel_only": True,
                "fused_split_backend": split_backend,
                "fused_smem_online_split_full_width_tile_k": kv_tile_tokens,
                "fused_smem_online_split_full_width_tma_pv": use_tma_pv,
                "fused_smem_online_split_full_width_splits": num_splits,
                "fused_smem_online_split_full_width_q_tiles": q_tiles,
                "fused_smem_online_split_full_width_finite": bool(
                    torch.isfinite(fused_split).all().item()
                ),
                "fused_smem_online_split_full_width_min": float(
                    fused_split.min().item()
                ),
                "fused_smem_online_split_full_width_max": float(
                    fused_split.max().item()
                ),
                "partial_m_finite": bool(torch.isfinite(partial_m).all().item()),
                "partial_l_finite": bool(torch.isfinite(partial_l).all().item()),
            }
            print(result)
            return

        online_ref = torch.zeros_like(fused_split)
        online_m = torch.full((q_rows, 1), -float("inf"), device=device)
        online_l = torch.zeros((q_rows, 1), device=device)
        q_ref = q_actual_f32[:q_rows].float()
        v_ref_all = v_pv_ref_f32.float()
        qk_ref_scores = None
        if use_tma_pv:
            qk_ref_scores = torch.empty(
                (q_rows, kv_tokens), device=device, dtype=torch.bfloat16
            )
            tactic = min(2, int(metadata["runner_tactic_count"]) - 1)
            ext.cutlass_runner_fp4_gemm(
                q_cutlass[:q_rows].contiguous(),
                k_cutlass.contiguous(),
                q_cutlass_scales[:q_rows].contiguous(),
                k_cutlass_scales.contiguous(),
                qk_alpha,
                qk_ref_scores,
                workspace,
                tactic,
            )
            torch.cuda.synchronize()
        for split in range(num_splits):
            split_start_tile = split * (total_kv_tiles // num_splits)
            split_end_tile = split_start_tile + (total_kv_tiles // num_splits)
            split_out = torch.zeros_like(fused_split)
            split_m = torch.full((q_rows, 1), -float("inf"), device=device)
            split_l = torch.zeros((q_rows, 1), device=device)
            for tile in range(split_start_tile, split_end_tile):
                start = tile * kv_tile_tokens
                end = start + kv_tile_tokens
                if qk_ref_scores is None:
                    scores = (
                        torch.matmul(q_ref, k_ref_f32[start:end].float().T)
                        / (HEAD_DIM**0.5)
                    )
                else:
                    scores = qk_ref_scores[:, start:end].float() / (HEAD_DIM**0.5)
                tile_m = scores.amax(dim=-1, keepdim=True)
                new_m = torch.maximum(split_m, tile_m)
                alpha = torch.where(
                    split_l > 0,
                    torch.exp(split_m - new_m),
                    torch.zeros_like(split_l),
                )
                probs_unorm = torch.exp(scores - new_m)
                tile_l = probs_unorm.sum(dim=-1, keepdim=True)
                split_out *= alpha
                split_l = split_l * alpha + tile_l
                split_m = new_m
                p_packed, p_scales = fp32_to_nvfp4_rowmajor(
                    probs_unorm * PROB_GLOBAL_SCALE
                )
                p_dequant_unorm = (
                    nvfp4_rowmajor_to_fp32(p_packed, p_scales)
                    / PROB_GLOBAL_SCALE
                )
                split_out += torch.matmul(
                    p_dequant_unorm.float(),
                    v_ref_all[:, start:end].T,
                )
            new_m = torch.maximum(online_m, split_m)
            alpha_old = torch.where(
                online_l > 0,
                torch.exp(online_m - new_m),
                torch.zeros_like(online_l),
            )
            alpha_split = torch.exp(split_m - new_m)
            online_ref = online_ref * alpha_old + split_out * alpha_split
            online_l = online_l * alpha_old + split_l * alpha_split
            online_m = new_m
        online_ref = online_ref / online_l

        exact_scores = (
            torch.matmul(q_ref, k_ref_f32[:kv_tokens].float().T)
            / (HEAD_DIM**0.5)
        )
        exact_ref = torch.matmul(
            torch.softmax(exact_scores, dim=-1),
            v_ref_all[:, :kv_tokens].T,
        )
        online_delta = (fused_split - online_ref).abs()
        exact_delta = (fused_split - exact_ref).abs()
        online_ref_norm = torch.linalg.vector_norm(online_ref)
        online_norm = torch.linalg.vector_norm(fused_split)
        online_cos = torch.sum(fused_split * online_ref) / torch.clamp(
            online_ref_norm * online_norm,
            min=1.0e-20,
        )
        result = {
            "fused_split_backend": split_backend,
            "fused_smem_online_split_full_width_tile_k": kv_tile_tokens,
            "fused_smem_online_split_full_width_tma_pv": use_tma_pv,
            "fused_smem_online_split_full_width_splits": num_splits,
            "fused_smem_online_split_full_width_q_tiles": q_tiles,
            "fused_smem_online_split_full_width_finite": bool(
                torch.isfinite(fused_split).all().item()
            ),
            "fused_smem_online_split_full_width_vs_quant_mean_abs": float(
                online_delta.mean().item()
            ),
            "fused_smem_online_split_full_width_vs_quant_max_abs": float(
                online_delta.max().item()
            ),
            "fused_smem_online_split_full_width_vs_quant_cosine": float(
                online_cos.item()
            ),
            "fused_smem_online_split_full_width_vs_exact_mean_abs": float(
                exact_delta.mean().item()
            ),
            "fused_smem_online_split_full_width_vs_exact_max_abs": float(
                exact_delta.max().item()
            ),
        }
        if args.bench:
            bench_result = event_ms(
                run_split_kernel,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            result["bench_fused_smem_online_split_full_width"] = bench_result
            if use_tma_pv:
                result["bench_fused_tma_qk_tma_pv_split_full_width"] = bench_result
        print(result)
        return
    if args.smem_atom_check_only:
        qk_smem_atom_tile = torch.empty((128, 128), device=device, dtype=torch.float32)
        def set_smem_atom_codes(
            q_codes_128: torch.Tensor,
            k_codes_128: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            q_actual.zero_()
            k.zero_()
            q_scales_actual.fill_(0x38)
            k_scales.fill_(0x38)
            q_actual[:128].copy_(
                (
                    q_codes_128[:, 0::2]
                    | (q_codes_128[:, 1::2] << 4)
                ).contiguous()
            )
            k[:128].copy_(
                (
                    k_codes_128[:, 0::2]
                    | (k_codes_128[:, 1::2] << 4)
                ).contiguous()
            )
            values = E2M1_VALUES.to(device=device)
            return values[q_codes_128.long()], values[k_codes_128.long()]

        def smem_atom_metrics(ref: torch.Tensor) -> dict[str, float | bool]:
            ext.qk_cutlass_smem_atom_tile(
                q_actual,
                q_scales_actual,
                k,
                k_scales,
                qk_smem_atom_tile,
                args.smem_atom_data_mode,
                args.smem_atom_scale_mode,
            )
            torch.cuda.synchronize()
            delta = (qk_smem_atom_tile - ref).abs()
            flat_argmax = int(delta.argmax().item())
            max_row = flat_argmax // delta.shape[1]
            max_col = flat_argmax % delta.shape[1]
            row_mean = delta.mean(dim=1)
            col_mean = delta.mean(dim=0)
            top_rows = torch.topk(row_mean, k=8).indices.tolist()
            top_cols = torch.topk(col_mean, k=8).indices.tolist()
            ref_norm = torch.linalg.vector_norm(ref)
            out_norm = torch.linalg.vector_norm(qk_smem_atom_tile)
            cosine = torch.sum(qk_smem_atom_tile * ref) / torch.clamp(
                ref_norm * out_norm,
                min=1.0e-20,
            )
            return {
                "finite": bool(torch.isfinite(qk_smem_atom_tile).all().item()),
                "mean_abs": float(delta.mean().item()),
                "max_abs": float(delta.max().item()),
                "cosine": float(cosine.item()),
                "norm_ratio": float(
                    (
                        out_norm
                        / torch.clamp(ref_norm, min=1.0e-20)
                    ).item()
                ),
                "out_0_0": float(qk_smem_atom_tile[0, 0].item()),
                "ref_0_0": float(ref[0, 0].item()),
                "max_loc": (max_row, max_col),
                "max_out": float(qk_smem_atom_tile[max_row, max_col].item()),
                "max_ref": float(ref[max_row, max_col].item()),
                "top_error_rows": top_rows,
                "top_error_cols": top_cols,
            }

        if args.smem_atom_code_sweep:
            values = E2M1_VALUES.to(device=device)
            mismatches = []
            max_abs = 0.0
            max_pair = None
            for q_code in range(16):
                q_byte = q_code | (q_code << 4)
                q_actual.fill_(q_byte)
                q_scales_actual.fill_(0x38)
                for k_code in range(16):
                    k_byte = k_code | (k_code << 4)
                    k.fill_(k_byte)
                    k_scales.fill_(0x38)
                    ext.qk_cutlass_smem_atom_tile(
                        q_actual,
                        q_scales_actual,
                        k,
                        k_scales,
                        qk_smem_atom_tile,
                        args.smem_atom_data_mode,
                        args.smem_atom_scale_mode,
                    )
                    torch.cuda.synchronize()
                    expected = float((512.0 * values[q_code] * values[k_code]).item())
                    delta = (qk_smem_atom_tile - expected).abs()
                    pair_max = float(delta.max().item())
                    if pair_max > max_abs:
                        max_abs = pair_max
                        max_pair = (q_code, k_code, expected, float(qk_smem_atom_tile[0, 0].item()))
                    if pair_max > 1.0e-3:
                        mismatches.append(
                            {
                                "q_code": q_code,
                                "k_code": k_code,
                                "expected": expected,
                                "actual_0_0": float(qk_smem_atom_tile[0, 0].item()),
                                "max_abs": pair_max,
                            }
                        )
            print(
                {
                    "cutlass_collective_metadata": metadata,
                    "qk_smem_atom_code_sweep_pairs": 256,
                    "qk_smem_atom_code_sweep_mismatch_count": len(mismatches),
                    "qk_smem_atom_code_sweep_max_abs": max_abs,
                    "qk_smem_atom_code_sweep_max_pair": max_pair,
                    "qk_smem_atom_code_sweep_first_mismatches": mismatches[:16],
                }
            )
            return
        if args.smem_atom_pattern_sweep:
            rows = torch.arange(128, device=device, dtype=torch.int64)[:, None]
            cols = torch.arange(512, device=device, dtype=torch.int64)[None, :]
            n_rows = torch.arange(128, device=device, dtype=torch.int64)[:, None]
            gen = torch.Generator(device=device)
            gen.manual_seed(20260427)
            random_q = torch.randint(
                0, 16, (128, 512), device=device, generator=gen, dtype=torch.int64
            )
            random_k = torch.randint(
                0, 16, (128, 512), device=device, generator=gen, dtype=torch.int64
            )
            cases: list[tuple[str, torch.Tensor, torch.Tensor]] = [
                (
                    "a_varies_by_row_b_ones",
                    (rows % 16).expand(128, 512),
                    torch.full((128, 512), 2, device=device, dtype=torch.int64),
                ),
                (
                    "a_ones_b_varies_by_n",
                    torch.full((128, 512), 2, device=device, dtype=torch.int64),
                    (n_rows % 16).expand(128, 512),
                ),
                (
                    "a_varies_by_k_b_ones",
                    (cols % 16).expand(128, 512),
                    torch.full((128, 512), 2, device=device, dtype=torch.int64),
                ),
                (
                    "a_ones_b_varies_by_k",
                    torch.full((128, 512), 2, device=device, dtype=torch.int64),
                    (cols % 16).expand(128, 512),
                ),
                (
                    "a_k_b_k_distinct",
                    (cols % 16).expand(128, 512),
                    ((3 * cols + 5) % 16).expand(128, 512),
                ),
                (
                    "checker_full",
                    ((rows + cols) % 16),
                    ((3 * n_rows + 5 * cols) % 16),
                ),
                (
                    "uniform_random_codes",
                    random_q,
                    random_k,
                ),
            ]
            results = {}
            for name, q_codes_i64, k_codes_i64 in cases:
                q_f32, k_f32 = set_smem_atom_codes(
                    q_codes_i64.to(torch.uint8),
                    k_codes_i64.to(torch.uint8),
                )
                ref = torch.matmul(q_f32.float(), k_f32.float().T)
                results[name] = smem_atom_metrics(ref)
            print(
                {
                    "cutlass_collective_metadata": metadata,
                    "qk_smem_atom_data_mode": args.smem_atom_data_mode,
                    "qk_smem_atom_scale_mode": args.smem_atom_scale_mode,
                    "qk_smem_atom_pattern_sweep": results,
                }
            )
            return
        if args.smem_atom_k_map_probe:
            values = E2M1_VALUES.to(device=device)
            gen = torch.Generator(device=device)
            gen.manual_seed(424242)
            seed_count = 16
            sample_coords = [
                (0, 0),
                (0, 1),
                (0, 8),
                (0, 16),
                (0, 32),
                (1, 0),
                (7, 7),
                (8, 0),
                (8, 8),
                (16, 16),
                (32, 32),
                (63, 63),
                (64, 0),
                (72, 8),
                (96, 96),
                (127, 127),
            ]
            k_patterns = torch.randint(
                0,
                16,
                (seed_count, 512),
                device=device,
                generator=gen,
                dtype=torch.int64,
            )
            observed = torch.empty(
                (len(sample_coords), 512, seed_count), device=device
            )
            q_codes = torch.zeros((128, 512), device=device, dtype=torch.uint8)
            for seed in range(seed_count):
                k_codes = k_patterns[seed].to(torch.uint8)[None, :].expand(128, 512)
                for k_pos in range(512):
                    q_codes.zero_()
                    q_codes[:, k_pos] = 2
                    set_smem_atom_codes(q_codes, k_codes)
                    ext.qk_cutlass_smem_atom_tile(
                        q_actual,
                        q_scales_actual,
                        k,
                        k_scales,
                        qk_smem_atom_tile,
                        args.smem_atom_data_mode,
                        args.smem_atom_scale_mode,
                    )
                    torch.cuda.synchronize()
                    for coord_idx, (row, col) in enumerate(sample_coords):
                        observed[coord_idx, k_pos, seed] = qk_smem_atom_tile[row, col]
            expected_signatures = values[k_patterns.long()].T.contiguous()
            identity = torch.arange(512, device=device)
            coord_results = {}
            for coord_idx, coord in enumerate(sample_coords):
                distances = (
                    observed[coord_idx, :, None, :]
                    - expected_signatures[None, :, :]
                ).abs().amax(dim=-1)
                best_distance, best_k = distances.min(dim=1)
                mismatches = (best_k != identity).nonzero().flatten()
                coord_results[str(coord)] = {
                    "mismatch_count": int(mismatches.numel()),
                    "max_distance": float(best_distance.max().item()),
                    "first_64_map": best_k[:64].tolist(),
                    "first_mismatches": [
                        (int(i), int(best_k[i].item()))
                        for i in mismatches[:32].tolist()
                    ],
                }
            print(
                {
                    "cutlass_collective_metadata": metadata,
                    "qk_smem_atom_data_mode": args.smem_atom_data_mode,
                    "qk_smem_atom_scale_mode": args.smem_atom_scale_mode,
                    "qk_smem_atom_k_map_seed_count": seed_count,
                    "qk_smem_atom_k_map_coords": coord_results,
                }
            )
            return
        if args.smem_atom_b_nk_map_probe:
            values = E2M1_VALUES.to(device=device)
            gen = torch.Generator(device=device)
            gen.manual_seed(515151)
            seed_count = 16
            sample_coords = [
                (0, 0),
                (0, 1),
                (0, 8),
                (0, 16),
                (0, 32),
                (1, 0),
                (7, 7),
                (8, 0),
                (8, 8),
                (16, 16),
                (32, 32),
                (63, 63),
                (64, 0),
                (72, 8),
                (96, 96),
                (127, 127),
            ]
            b_patterns = torch.randint(
                0,
                16,
                (seed_count, 128, 512),
                device=device,
                generator=gen,
                dtype=torch.int64,
            )
            observed = torch.empty(
                (len(sample_coords), 512, seed_count), device=device
            )
            q_codes = torch.zeros((128, 512), device=device, dtype=torch.uint8)
            for seed in range(seed_count):
                k_codes = b_patterns[seed].to(torch.uint8)
                for k_pos in range(512):
                    q_codes.zero_()
                    q_codes[:, k_pos] = 2
                    set_smem_atom_codes(q_codes, k_codes)
                    ext.qk_cutlass_smem_atom_tile(
                        q_actual,
                        q_scales_actual,
                        k,
                        k_scales,
                        qk_smem_atom_tile,
                        args.smem_atom_data_mode,
                        args.smem_atom_scale_mode,
                    )
                    torch.cuda.synchronize()
                    for coord_idx, (row, col) in enumerate(sample_coords):
                        observed[coord_idx, k_pos, seed] = qk_smem_atom_tile[row, col]
            expected_signatures = values[b_patterns.long()].permute(1, 2, 0)
            expected_flat = expected_signatures.reshape(128 * 512, seed_count)
            coord_results = {}
            identity_k = torch.arange(512, device=device)
            for coord_idx, coord in enumerate(sample_coords):
                distances = (
                    observed[coord_idx, :, None, :]
                    - expected_flat[None, :, :]
                ).abs().amax(dim=-1)
                best_distance, best_flat = distances.min(dim=1)
                best_n = torch.div(best_flat, 512, rounding_mode="floor")
                best_k = best_flat - best_n * 512
                expected_n = torch.full_like(best_n, coord[1])
                mismatches = (
                    (best_n != expected_n) | (best_k != identity_k)
                ).nonzero().flatten()
                coord_results[str(coord)] = {
                    "mismatch_count": int(mismatches.numel()),
                    "max_distance": float(best_distance.max().item()),
                    "first_32_map": [
                        (int(i), int(best_n[i].item()), int(best_k[i].item()))
                        for i in range(32)
                    ],
                    "first_mismatches": [
                        (int(i), int(best_n[i].item()), int(best_k[i].item()))
                        for i in mismatches[:32].tolist()
                    ],
                }
            print(
                {
                    "cutlass_collective_metadata": metadata,
                    "qk_smem_atom_data_mode": args.smem_atom_data_mode,
                    "qk_smem_atom_scale_mode": args.smem_atom_scale_mode,
                    "qk_smem_atom_b_nk_map_seed_count": seed_count,
                    "qk_smem_atom_b_nk_map_coords": coord_results,
                }
            )
            return
        if args.smem_atom_a_mk_map_probe:
            values = E2M1_VALUES.to(device=device)
            gen = torch.Generator(device=device)
            gen.manual_seed(616161)
            seed_count = 16
            sample_coords = [
                (0, 0),
                (0, 1),
                (0, 8),
                (0, 16),
                (0, 32),
                (1, 0),
                (7, 7),
                (8, 0),
                (8, 8),
                (16, 16),
                (32, 32),
                (63, 63),
                (64, 0),
                (72, 8),
                (96, 96),
                (127, 127),
            ]
            a_patterns = torch.randint(
                0,
                16,
                (seed_count, 128, 512),
                device=device,
                generator=gen,
                dtype=torch.int64,
            )
            observed = torch.empty(
                (len(sample_coords), 512, seed_count), device=device
            )
            k_codes = torch.zeros((128, 512), device=device, dtype=torch.uint8)
            for seed in range(seed_count):
                q_codes = a_patterns[seed].to(torch.uint8)
                for k_pos in range(512):
                    k_codes.zero_()
                    k_codes[:, k_pos] = 2
                    set_smem_atom_codes(q_codes, k_codes)
                    ext.qk_cutlass_smem_atom_tile(
                        q_actual,
                        q_scales_actual,
                        k,
                        k_scales,
                        qk_smem_atom_tile,
                        args.smem_atom_data_mode,
                        args.smem_atom_scale_mode,
                    )
                    torch.cuda.synchronize()
                    for coord_idx, (row, col) in enumerate(sample_coords):
                        observed[coord_idx, k_pos, seed] = qk_smem_atom_tile[row, col]
            expected_signatures = values[a_patterns.long()].permute(1, 2, 0)
            expected_flat = expected_signatures.reshape(128 * 512, seed_count)
            coord_results = {}
            identity_k = torch.arange(512, device=device)
            for coord_idx, coord in enumerate(sample_coords):
                distances = (
                    observed[coord_idx, :, None, :]
                    - expected_flat[None, :, :]
                ).abs().amax(dim=-1)
                best_distance, best_flat = distances.min(dim=1)
                best_m = torch.div(best_flat, 512, rounding_mode="floor")
                best_k = best_flat - best_m * 512
                expected_m = torch.full_like(best_m, coord[0])
                mismatches = (
                    (best_m != expected_m) | (best_k != identity_k)
                ).nonzero().flatten()
                coord_results[str(coord)] = {
                    "mismatch_count": int(mismatches.numel()),
                    "max_distance": float(best_distance.max().item()),
                    "first_32_map": [
                        (int(i), int(best_m[i].item()), int(best_k[i].item()))
                        for i in range(32)
                    ],
                    "first_mismatches": [
                        (int(i), int(best_m[i].item()), int(best_k[i].item()))
                        for i in mismatches[:32].tolist()
                    ],
                }
            print(
                {
                    "cutlass_collective_metadata": metadata,
                    "qk_smem_atom_data_mode": args.smem_atom_data_mode,
                    "qk_smem_atom_scale_mode": args.smem_atom_scale_mode,
                    "qk_smem_atom_a_mk_map_seed_count": seed_count,
                    "qk_smem_atom_a_mk_map_coords": coord_results,
                }
            )
            return
        ext.qk_cutlass_smem_atom_tile(
            q_actual,
            q_scales_actual,
            k,
            k_scales,
            qk_smem_atom_tile,
            args.smem_atom_data_mode,
            args.smem_atom_scale_mode,
        )
        torch.cuda.synchronize()
        qk_smem_atom_ref = torch.matmul(
            q_actual_f32[:128].float(),
            k_ref_f32[:128].float().T,
        )
        qk_smem_atom_delta = (qk_smem_atom_tile - qk_smem_atom_ref).abs()
        qk_smem_atom_ref_norm = torch.linalg.vector_norm(qk_smem_atom_ref)
        qk_smem_atom_norm = torch.linalg.vector_norm(qk_smem_atom_tile)
        qk_smem_atom_cos = torch.sum(
            qk_smem_atom_tile * qk_smem_atom_ref
        ) / torch.clamp(
            qk_smem_atom_ref_norm * qk_smem_atom_norm,
            min=1.0e-20,
        )
        qk_smem_atom_cos_t = torch.sum(
            qk_smem_atom_tile.T * qk_smem_atom_ref
        ) / torch.clamp(
            qk_smem_atom_ref_norm * qk_smem_atom_norm,
            min=1.0e-20,
        )
        qk_smem_atom_nonzero = (
            qk_smem_atom_tile.abs() > 1.0e-30
        ).sum()
        print(
            {
                "cutlass_collective_metadata": metadata,
                "qk_smem_atom_data_mode": args.smem_atom_data_mode,
                "qk_smem_atom_scale_mode": args.smem_atom_scale_mode,
                "qk_smem_atom_finite": bool(
                    torch.isfinite(qk_smem_atom_tile).all().item()
                ),
                "qk_smem_atom_mean_abs": float(qk_smem_atom_delta.mean().item()),
                "qk_smem_atom_max_abs": float(qk_smem_atom_delta.max().item()),
                "qk_smem_atom_cosine": float(qk_smem_atom_cos.item()),
                "qk_smem_atom_transpose_cosine": float(
                    qk_smem_atom_cos_t.item()
                ),
                "qk_smem_atom_norm_ratio": float(
                    (
                        qk_smem_atom_norm
                        / torch.clamp(qk_smem_atom_ref_norm, min=1.0e-20)
                    ).item()
                ),
                "qk_smem_atom_min": float(qk_smem_atom_tile.min().item()),
                "qk_smem_atom_max": float(qk_smem_atom_tile.max().item()),
                "qk_smem_atom_mean": float(qk_smem_atom_tile.mean().item()),
                "qk_smem_atom_ref_mean": float(qk_smem_atom_ref.mean().item()),
                "qk_smem_atom_nonzero": int(qk_smem_atom_nonzero.item()),
            }
        )
        return

    qk_runner_result = {}
    qk_runner_bench = {}
    if args.runner_check or args.bench or args.full_runner_bench:
        q_cutlass, q_cutlass_scales, q_cutlass_global = quantize_cutlass(
            q.reshape(Q_LEN * GROUP, HEAD_DIM)
        )
        k_cutlass, k_cutlass_scales, k_cutlass_global = quantize_cutlass(
            k_ref_f32.to(torch.bfloat16)
        )
        tactic = min(2, int(metadata["runner_tactic_count"]) - 1)
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        qk_runner_official = torch.empty(
            (128, k_cutlass.shape[0]), device=device, dtype=torch.bfloat16
        )
        qk_runner_ext = torch.empty_like(qk_runner_official)
        k_cutlass_native = k_cutlass.contiguous()
        k_cutlass_scales_native = k_cutlass_scales.contiguous()
        k_cutlass_for_python_runner = k_cutlass.T
        k_cutlass_scales_for_python_runner = k_cutlass_scales.T
        qk_alpha = 1.0 / (q_cutlass_global * k_cutlass_global)
        runner = gemm_base.get_cutlass_fp4_gemm_module(
            *torch.cuda.get_device_capability(device)
        ).cutlass_fp4_gemm_runner()
        runner(
            inputs=[
                q_cutlass[:128].contiguous(),
                k_cutlass_for_python_runner,
                q_cutlass_scales[:128].contiguous(),
                k_cutlass_scales_for_python_runner,
                qk_alpha,
                qk_runner_official.dtype,
                qk_runner_official,
                16,
                True,
                workspace,
            ],
            tactic=tactic,
        )
        ext.cutlass_runner_fp4_gemm(
            q_cutlass[:128].contiguous(),
            k_cutlass_native,
            q_cutlass_scales[:128].contiguous(),
            k_cutlass_scales_native,
            qk_alpha,
            qk_runner_ext,
            workspace,
            tactic,
        )
        torch.cuda.synchronize()
        qk_runner_delta = (qk_runner_ext.float() - qk_runner_official.float()).abs()
        qk_runner_ref_norm = torch.linalg.vector_norm(qk_runner_official.float())
        qk_runner_ext_norm = torch.linalg.vector_norm(qk_runner_ext.float())
        qk_runner_cos = torch.sum(
            qk_runner_ext.float() * qk_runner_official.float()
        ) / torch.clamp(
            qk_runner_ref_norm * qk_runner_ext_norm,
            min=1.0e-20,
        )
        qk_runner_result = {
            "qk_runner_tactic": tactic,
            "qk_runner_finite": bool(torch.isfinite(qk_runner_ext).all().item()),
            "qk_runner_vs_official_mean_abs": float(qk_runner_delta.mean().item()),
            "qk_runner_vs_official_max_abs": float(qk_runner_delta.max().item()),
            "qk_runner_vs_official_cosine": float(qk_runner_cos.item()),
        }
        p_runner_ref = torch.softmax(qk_runner_ext.float() / (HEAD_DIM**0.5), dim=-1)
        p_cutlass, p_cutlass_scales, p_cutlass_global = quantize_cutlass(
            p_runner_ref.to(torch.bfloat16)
        )
        v_runner_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
        v_cutlass, v_cutlass_scales, v_cutlass_global = quantize_cutlass(
            v_runner_ref_f32.T.contiguous().to(torch.bfloat16)
        )
        pv_alpha = 1.0 / (p_cutlass_global * v_cutlass_global)
        pv_runner_official = torch.empty(
            (p_cutlass.shape[0], v_cutlass.shape[0]),
            device=device,
            dtype=torch.bfloat16,
        )
        pv_runner_ext = torch.empty_like(pv_runner_official)
        runner(
            inputs=[
                p_cutlass.contiguous(),
                v_cutlass.T,
                p_cutlass_scales.contiguous(),
                v_cutlass_scales.T,
                pv_alpha,
                pv_runner_official.dtype,
                pv_runner_official,
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
            pv_runner_ext,
            workspace,
            tactic,
        )
        torch.cuda.synchronize()
        pv_runner_delta = (pv_runner_ext.float() - pv_runner_official.float()).abs()
        pv_runner_ref_norm = torch.linalg.vector_norm(pv_runner_official.float())
        pv_runner_ext_norm = torch.linalg.vector_norm(pv_runner_ext.float())
        pv_runner_cos = torch.sum(
            pv_runner_ext.float() * pv_runner_official.float()
        ) / torch.clamp(
            pv_runner_ref_norm * pv_runner_ext_norm,
            min=1.0e-20,
        )
        pv_exact_ref = torch.matmul(p_runner_ref.float(), v_runner_ref_f32.float())
        pv_exact_delta = (pv_runner_ext.float() - pv_exact_ref).abs()
        pv_exact_ref_norm = torch.linalg.vector_norm(pv_exact_ref)
        pv_runner_exact_cos = torch.sum(
            pv_runner_ext.float() * pv_exact_ref
        ) / torch.clamp(
            pv_runner_ext_norm * pv_exact_ref_norm,
            min=1.0e-20,
        )
        qk_runner_result.update(
            {
                "pv_runner_finite": bool(torch.isfinite(pv_runner_ext).all().item()),
                "pv_runner_vs_official_mean_abs": float(pv_runner_delta.mean().item()),
                "pv_runner_vs_official_max_abs": float(pv_runner_delta.max().item()),
                "pv_runner_vs_official_cosine": float(pv_runner_cos.item()),
                "pv_runner_vs_exact_mean_abs": float(pv_exact_delta.mean().item()),
                "pv_runner_vs_exact_max_abs": float(pv_exact_delta.max().item()),
                "pv_runner_vs_exact_cosine": float(pv_runner_exact_cos.item()),
            }
        )
        if args.bench:
            qk_runner_bench = {
                "bench_cutlass_runner_qk_128x32768": event_ms(
                    lambda: ext.cutlass_runner_fp4_gemm(
                        q_cutlass[:128].contiguous(),
                        k_cutlass_native,
                        q_cutlass_scales[:128].contiguous(),
                        k_cutlass_scales_native,
                        qk_alpha,
                        qk_runner_ext,
                        workspace,
                        tactic,
                    ),
                    warmup=args.warmup,
                    repeat=args.repeat,
                ),
                "bench_cutlass_runner_pv_128x512_k32768": event_ms(
                    lambda: ext.cutlass_runner_fp4_gemm(
                        p_cutlass.contiguous(),
                        v_cutlass.contiguous(),
                        p_cutlass_scales.contiguous(),
                        v_cutlass_scales.contiguous(),
                        pv_alpha,
                        pv_runner_ext,
                        workspace,
                        tactic,
                    ),
                    warmup=args.warmup,
                    repeat=args.repeat,
                ),
            }
        if args.full_runner_bench:
            qk_runner_full = torch.empty(
                (q_cutlass.shape[0], k_cutlass.shape[0]),
                device=device,
                dtype=torch.bfloat16,
            )
            full_qk_bench = event_ms(
                lambda: ext.cutlass_runner_fp4_gemm(
                    q_cutlass.contiguous(),
                    k_cutlass_native,
                    q_cutlass_scales.contiguous(),
                    k_cutlass_scales_native,
                    qk_alpha,
                    qk_runner_full,
                    workspace,
                    tactic,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
            ext.cutlass_runner_fp4_gemm(
                q_cutlass.contiguous(),
                k_cutlass_native,
                q_cutlass_scales.contiguous(),
                k_cutlass_scales_native,
                qk_alpha,
                qk_runner_full,
                workspace,
                tactic,
            )
            torch.cuda.synchronize()
            p_runner_full = torch.softmax(
                qk_runner_full.float() / (HEAD_DIM**0.5), dim=-1
            )
            p_cutlass_full, p_cutlass_full_scales, p_cutlass_full_global = quantize_cutlass(
                p_runner_full.to(torch.bfloat16)
            )
            pv_full_alpha = 1.0 / (p_cutlass_full_global * v_cutlass_global)
            pv_runner_full = torch.empty(
                (p_cutlass_full.shape[0], v_cutlass.shape[0]),
                device=device,
                dtype=torch.bfloat16,
            )
            full_pv_bench = event_ms(
                lambda: ext.cutlass_runner_fp4_gemm(
                    p_cutlass_full.contiguous(),
                    v_cutlass.contiguous(),
                    p_cutlass_full_scales.contiguous(),
                    v_cutlass_scales.contiguous(),
                    pv_full_alpha,
                    pv_runner_full,
                    workspace,
                    tactic,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
            qk_runner_result.update(
                {
                    "full_runner_qk_shape": tuple(qk_runner_full.shape),
                    "full_runner_pv_shape": tuple(pv_runner_full.shape),
                    "bench_cutlass_runner_qk_4096x32768": full_qk_bench,
                    "bench_cutlass_runner_pv_4096x512_k32768": full_pv_bench,
                }
            )

    qk_collective_result = {}
    if args.collective_check:
        q_cutlass, q_cutlass_scales, q_cutlass_global = quantize_cutlass(
            q.reshape(Q_LEN * GROUP, HEAD_DIM)
        )
        k_cutlass, k_cutlass_scales, k_cutlass_global = quantize_cutlass(
            k_ref_f32.to(torch.bfloat16)
        )
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        qk_collective_tile = torch.empty((128, 128), device=device, dtype=torch.float32)
        ext.qk_cutlass_collective_tile(
            q_cutlass,
            q_cutlass_scales,
            k_cutlass,
            k_cutlass_scales,
            qk_collective_tile,
            workspace,
            args.collective_debug_mode,
        )
        torch.cuda.synchronize()
        if args.collective_debug_mode != 0:
            qk_collective_result = {
                "qk_collective_debug_mode": args.collective_debug_mode,
                "qk_collective_debug_marker": float(
                    qk_collective_tile[0, 0].item()
                ),
            }
        else:
            qk_official = torch.empty(
                (128, k_cutlass.shape[0]), device=device, dtype=torch.bfloat16
            )
            qk_alpha = 1.0 / (q_cutlass_global * k_cutlass_global)
            tactic = min(2, int(metadata["runner_tactic_count"]) - 1)
            ext.cutlass_runner_fp4_gemm(
                q_cutlass[:128].contiguous(),
                k_cutlass.contiguous(),
                q_cutlass_scales[:128].contiguous(),
                k_cutlass_scales.contiguous(),
                qk_alpha,
                qk_official,
                workspace,
                tactic,
            )
            torch.cuda.synchronize()
            qk_collective_scaled = qk_collective_tile * qk_alpha
            qk_collective_ref = qk_official[:, :128].float()
            qk_collective_delta = (qk_collective_scaled - qk_collective_ref).abs()
            qk_collective_ref_norm = torch.linalg.vector_norm(qk_collective_ref)
            qk_collective_norm = torch.linalg.vector_norm(qk_collective_scaled)
            qk_collective_cos = torch.sum(
                qk_collective_scaled * qk_collective_ref
            ) / torch.clamp(
                qk_collective_ref_norm * qk_collective_norm,
                min=1.0e-20,
            )
            qk_collective_result = {
                "qk_collective_tile_k": 256,
                "qk_collective_finite": bool(
                    torch.isfinite(qk_collective_scaled).all().item()
                ),
                "qk_collective_vs_official_mean_abs": float(
                    qk_collective_delta.mean().item()
                ),
                "qk_collective_vs_official_max_abs": float(
                    qk_collective_delta.max().item()
                ),
                "qk_collective_vs_official_cosine": float(qk_collective_cos.item()),
            }
            if args.bench:
                bench_fn = lambda: ext.qk_cutlass_collective_tile(
                    q_cutlass,
                    q_cutlass_scales,
                    k_cutlass,
                    k_cutlass_scales,
                    qk_collective_tile,
                    workspace,
                    0,
                )
                qk_collective_result["bench_qk_collective_tile"] = event_ms(
                    bench_fn,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )

    if args.collective_check and (
        args.collective_debug_mode != 0 or args.collective_check_only
    ):
        print(qk_collective_result)
        return

    qk_smem_atom_result = {}
    if args.smem_atom_check:
        qk_smem_atom_tile = torch.empty((128, 128), device=device, dtype=torch.float32)
        ext.qk_cutlass_smem_atom_tile(
            q_actual,
            q_scales_actual,
            k,
            k_scales,
            qk_smem_atom_tile,
            args.smem_atom_data_mode,
            args.smem_atom_scale_mode,
        )
        torch.cuda.synchronize()
        qk_smem_atom_ref = torch.matmul(
            q_actual_f32[:128].float(),
            k_ref_f32[:128].float().T,
        )
        qk_smem_atom_delta = (qk_smem_atom_tile - qk_smem_atom_ref).abs()
        qk_smem_atom_ref_norm = torch.linalg.vector_norm(qk_smem_atom_ref)
        qk_smem_atom_norm = torch.linalg.vector_norm(qk_smem_atom_tile)
        qk_smem_atom_cos = torch.sum(
            qk_smem_atom_tile * qk_smem_atom_ref
        ) / torch.clamp(
            qk_smem_atom_ref_norm * qk_smem_atom_norm,
            min=1.0e-20,
        )
        qk_smem_atom_result = {
            "qk_smem_atom_data_mode": args.smem_atom_data_mode,
            "qk_smem_atom_scale_mode": args.smem_atom_scale_mode,
            "qk_smem_atom_finite": bool(
                torch.isfinite(qk_smem_atom_tile).all().item()
            ),
            "qk_smem_atom_mean_abs": float(qk_smem_atom_delta.mean().item()),
            "qk_smem_atom_max_abs": float(qk_smem_atom_delta.max().item()),
            "qk_smem_atom_cosine": float(qk_smem_atom_cos.item()),
        }

    q_rows = torch.arange(16, device=device) * GROUP
    qk_ref = torch.matmul(q_actual_f32[q_rows].float(), k_ref_f32[:16].float().T)
    qk_tile = torch.empty((16, 16), device=device, dtype=torch.float32)
    ext.qk_tile_cutlass_atom(q_actual, q_scales_actual, k, k_scales, qk_tile)
    torch.cuda.synchronize()

    qk_delta = (qk_tile - qk_ref).abs()
    qk_ref_norm = torch.linalg.vector_norm(qk_ref)
    qk_tile_norm = torch.linalg.vector_norm(qk_tile)
    qk_cos = torch.sum(qk_tile * qk_ref) / torch.clamp(
        qk_ref_norm * qk_tile_norm,
        min=1.0e-20,
    )

    qk_block_ref = torch.matmul(q_actual_f32[q_rows].float(), k_ref_f32.float().T)
    qk_block = torch.empty((16, k.shape[0]), device=device, dtype=torch.float32)
    ext.qk_block_cutlass_atom(q_actual, q_scales_actual, k, k_scales, qk_block)
    torch.cuda.synchronize()
    qk_block_delta = (qk_block - qk_block_ref).abs()
    qk_block_ref_norm = torch.linalg.vector_norm(qk_block_ref)
    qk_block_norm = torch.linalg.vector_norm(qk_block)
    qk_block_cos = torch.sum(qk_block * qk_block_ref) / torch.clamp(
        qk_block_ref_norm * qk_block_norm,
        min=1.0e-20,
    )

    p_ref = torch.softmax(qk_block_ref / (HEAD_DIM**0.5), dim=-1)
    p_packed = torch.empty((16, k.shape[0] // 2), device=device, dtype=torch.uint8)
    p_scales = torch.empty((16, k.shape[0] // 16), device=device, dtype=torch.uint8)
    ext.softmax_quant_p_block(qk_block, p_packed, p_scales)
    torch.cuda.synchronize()
    p_dequant_scaled = nvfp4_rowmajor_to_fp32(p_packed, p_scales)
    p_dequant = p_dequant_scaled / PROB_GLOBAL_SCALE
    p_delta = (p_dequant - p_ref).abs()
    p_ref_norm = torch.linalg.vector_norm(p_ref)
    p_dequant_norm = torch.linalg.vector_norm(p_dequant)
    p_cos = torch.sum(p_dequant * p_ref) / torch.clamp(
        p_ref_norm * p_dequant_norm,
        min=1.0e-20,
    )

    v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
    v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
    v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
    pv_ref_quant = torch.matmul(p_dequant.float(), v_pv_ref_f32.float().T)
    pv_ref_exact_p = torch.matmul(p_ref.float(), v_pv_ref_f32.float().T)
    pv_block = torch.empty((16, HEAD_DIM), device=device, dtype=torch.float32)
    ext.pv_block_cutlass_atom(p_packed, p_scales, v_pv, v_pv_scales, pv_block)
    torch.cuda.synchronize()
    pv_block_dequant = pv_block / PROB_GLOBAL_SCALE
    pv_quant_delta = (pv_block_dequant - pv_ref_quant).abs()
    pv_exact_delta = (pv_block_dequant - pv_ref_exact_p).abs()
    pv_ref_norm = torch.linalg.vector_norm(pv_ref_quant)
    pv_block_norm = torch.linalg.vector_norm(pv_block_dequant)
    pv_cos = torch.sum(pv_block_dequant * pv_ref_quant) / torch.clamp(
        pv_ref_norm * pv_block_norm,
        min=1.0e-20,
    )

    fused_atom = torch.empty((16, HEAD_DIM), device=device, dtype=torch.float32)
    ext.fused_atom_attention_block(
        q_actual,
        q_scales_actual,
        k,
        k_scales,
        v_pv,
        v_pv_scales,
        fused_atom,
    )
    torch.cuda.synchronize()
    fused_quant_delta = (fused_atom - pv_ref_quant).abs()
    fused_exact_delta = (fused_atom - pv_ref_exact_p).abs()
    fused_ref_norm = torch.linalg.vector_norm(pv_ref_exact_p)
    fused_norm = torch.linalg.vector_norm(fused_atom)
    fused_cos = torch.sum(fused_atom * pv_ref_exact_p) / torch.clamp(
        fused_ref_norm * fused_norm,
        min=1.0e-20,
    )
    split_partial_out = torch.empty((32, 16, HEAD_DIM), device=device, dtype=torch.float32)
    split_partial_m = torch.empty((32, 16), device=device, dtype=torch.float32)
    split_partial_l = torch.empty((32, 16), device=device, dtype=torch.float32)
    split_fused_atom = torch.empty((16, HEAD_DIM), device=device, dtype=torch.float32)
    ext.fused_atom_attention_split_block(
        q_actual,
        q_scales_actual,
        k,
        k_scales,
        v_pv,
        v_pv_scales,
        split_partial_out,
        split_partial_m,
        split_partial_l,
        split_fused_atom,
    )
    torch.cuda.synchronize()
    split_fused_quant_delta = (split_fused_atom - pv_ref_quant).abs()
    split_fused_exact_delta = (split_fused_atom - pv_ref_exact_p).abs()
    split_fused_norm = torch.linalg.vector_norm(split_fused_atom)
    split_fused_cos = torch.sum(split_fused_atom * pv_ref_exact_p) / torch.clamp(
        fused_ref_norm * split_fused_norm,
        min=1.0e-20,
    )
    q_flat_128 = q_actual_f32[:128].float()
    qk_flat_128_ref = torch.matmul(q_flat_128, k_ref_f32.float().T)
    p_flat_128_ref = torch.softmax(qk_flat_128_ref / (HEAD_DIM**0.5), dim=-1)
    pv_flat_128_ref = torch.matmul(p_flat_128_ref.float(), v_pv_ref_f32.float().T)
    split_128_default_splits = 32
    split_128_partial_out = torch.empty(
        (8, split_128_default_splits, 16, HEAD_DIM),
        device=device,
        dtype=torch.float32,
    )
    split_128_partial_m = torch.empty(
        (8, split_128_default_splits, 16), device=device, dtype=torch.float32
    )
    split_128_partial_l = torch.empty(
        (8, split_128_default_splits, 16), device=device, dtype=torch.float32
    )
    split_128_fused_atom = torch.empty((128, HEAD_DIM), device=device, dtype=torch.float32)
    ext.fused_atom_attention_split_128rows(
        q_actual,
        q_scales_actual,
        k,
        k_scales,
        v_pv,
        v_pv_scales,
        split_128_partial_out,
        split_128_partial_m,
        split_128_partial_l,
        split_128_fused_atom,
        split_128_default_splits,
    )
    torch.cuda.synchronize()
    split_128_delta = (split_128_fused_atom - pv_flat_128_ref).abs()
    split_128_ref_norm = torch.linalg.vector_norm(pv_flat_128_ref)
    split_128_norm = torch.linalg.vector_norm(split_128_fused_atom)
    split_128_cos = torch.sum(split_128_fused_atom * pv_flat_128_ref) / torch.clamp(
        split_128_ref_norm * split_128_norm,
        min=1.0e-20,
    )
    colsplit_128_partial_out = torch.empty_like(split_128_partial_out)
    colsplit_128_partial_m = torch.empty_like(split_128_partial_m)
    colsplit_128_partial_l = torch.empty_like(split_128_partial_l)
    colsplit_128_fused_atom = torch.empty_like(split_128_fused_atom)
    ext.fused_atom_attention_split_128rows_colsplit(
        q_actual,
        q_scales_actual,
        k,
        k_scales,
        v_pv,
        v_pv_scales,
        colsplit_128_partial_out,
        colsplit_128_partial_m,
        colsplit_128_partial_l,
        colsplit_128_fused_atom,
        split_128_default_splits,
    )
    torch.cuda.synchronize()
    colsplit_128_delta = (colsplit_128_fused_atom - pv_flat_128_ref).abs()
    colsplit_128_norm = torch.linalg.vector_norm(colsplit_128_fused_atom)
    colsplit_128_cos = torch.sum(
        colsplit_128_fused_atom * pv_flat_128_ref
    ) / torch.clamp(
        split_128_ref_norm * colsplit_128_norm,
        min=1.0e-20,
    )

    result = {
        "cutlass_collective_metadata": metadata,
        "q_quant_scale_match": bool(torch.equal(q_scales_actual, q_scales_expected)),
        "q_quant_packed_match": bool(torch.equal(q_actual, q_expected)),
        "q_quant_dequant_delta_mean": float(q_dequant_delta.mean().item()),
        "q_quant_dequant_delta_max": float(q_dequant_delta.max().item()),
        "qk_tile_finite": bool(torch.isfinite(qk_tile).all().item()),
        "qk_tile_mean_abs": float(qk_delta.mean().item()),
        "qk_tile_max_abs": float(qk_delta.max().item()),
        "qk_tile_cosine": float(qk_cos.item()),
        "qk_block_finite": bool(torch.isfinite(qk_block).all().item()),
        "qk_block_mean_abs": float(qk_block_delta.mean().item()),
        "qk_block_max_abs": float(qk_block_delta.max().item()),
        "qk_block_cosine": float(qk_block_cos.item()),
        "p_quant_finite": bool(torch.isfinite(p_dequant).all().item()),
        "p_quant_row_sum_min": float(p_dequant.sum(dim=-1).min().item()),
        "p_quant_row_sum_max": float(p_dequant.sum(dim=-1).max().item()),
        "p_quant_mean_abs": float(p_delta.mean().item()),
        "p_quant_max_abs": float(p_delta.max().item()),
        "p_quant_cosine": float(p_cos.item()),
        "pv_block_finite": bool(torch.isfinite(pv_block_dequant).all().item()),
        "pv_block_vs_quant_p_mean_abs": float(pv_quant_delta.mean().item()),
        "pv_block_vs_quant_p_max_abs": float(pv_quant_delta.max().item()),
        "pv_block_vs_quant_p_cosine": float(pv_cos.item()),
        "pv_block_vs_exact_p_mean_abs": float(pv_exact_delta.mean().item()),
        "pv_block_vs_exact_p_max_abs": float(pv_exact_delta.max().item()),
        "fused_atom_finite": bool(torch.isfinite(fused_atom).all().item()),
        "fused_atom_vs_quant_p_mean_abs": float(fused_quant_delta.mean().item()),
        "fused_atom_vs_quant_p_max_abs": float(fused_quant_delta.max().item()),
        "fused_atom_vs_exact_p_mean_abs": float(fused_exact_delta.mean().item()),
        "fused_atom_vs_exact_p_max_abs": float(fused_exact_delta.max().item()),
        "fused_atom_vs_exact_p_cosine": float(fused_cos.item()),
        "split_fused_atom_finite": bool(torch.isfinite(split_fused_atom).all().item()),
        "split_fused_atom_vs_quant_p_mean_abs": float(
            split_fused_quant_delta.mean().item()
        ),
        "split_fused_atom_vs_quant_p_max_abs": float(
            split_fused_quant_delta.max().item()
        ),
        "split_fused_atom_vs_exact_p_mean_abs": float(
            split_fused_exact_delta.mean().item()
        ),
        "split_fused_atom_vs_exact_p_max_abs": float(
            split_fused_exact_delta.max().item()
        ),
        "split_fused_atom_vs_exact_p_cosine": float(split_fused_cos.item()),
        "split_128_fused_atom_finite": bool(
            torch.isfinite(split_128_fused_atom).all().item()
        ),
        "split_128_fused_atom_vs_exact_p_mean_abs": float(
            split_128_delta.mean().item()
        ),
        "split_128_fused_atom_vs_exact_p_max_abs": float(split_128_delta.max().item()),
        "split_128_fused_atom_vs_exact_p_cosine": float(split_128_cos.item()),
        "colsplit_128_fused_atom_finite": bool(
            torch.isfinite(colsplit_128_fused_atom).all().item()
        ),
        "colsplit_128_fused_atom_vs_exact_p_mean_abs": float(
            colsplit_128_delta.mean().item()
        ),
        "colsplit_128_fused_atom_vs_exact_p_max_abs": float(
            colsplit_128_delta.max().item()
        ),
        "colsplit_128_fused_atom_vs_exact_p_cosine": float(
            colsplit_128_cos.item()
        ),
    }
    result.update(qk_runner_result)
    result.update(qk_collective_result)
    result.update(qk_smem_atom_result)

    if args.bench:
        result["bench_qk_block"] = event_ms(
            lambda: ext.qk_block_cutlass_atom(q_actual, q_scales_actual, k, k_scales, qk_block),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        result["bench_softmax_quant_p_block"] = event_ms(
            lambda: ext.softmax_quant_p_block(qk_block, p_packed, p_scales),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        result["bench_pv_block"] = event_ms(
            lambda: ext.pv_block_cutlass_atom(p_packed, p_scales, v_pv, v_pv_scales, pv_block),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        result["bench_fused_atom_attention_block"] = event_ms(
            lambda: ext.fused_atom_attention_block(
                q_actual,
                q_scales_actual,
                k,
                k_scales,
                v_pv,
                v_pv_scales,
                fused_atom,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        result["bench_fused_atom_attention_split_block"] = event_ms(
            lambda: ext.fused_atom_attention_split_block(
                q_actual,
                q_scales_actual,
                k,
                k_scales,
                v_pv,
                v_pv_scales,
                split_partial_out,
                split_partial_m,
                split_partial_l,
                split_fused_atom,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        result["bench_fused_atom_attention_split_128rows"] = event_ms(
            lambda: ext.fused_atom_attention_split_128rows(
                q_actual,
                q_scales_actual,
                k,
                k_scales,
                v_pv,
                v_pv_scales,
                split_128_partial_out,
                split_128_partial_m,
                split_128_partial_l,
                split_128_fused_atom,
                split_128_default_splits,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        result["bench_fused_atom_attention_split_128rows_colsplit"] = event_ms(
            lambda: ext.fused_atom_attention_split_128rows_colsplit(
                q_actual,
                q_scales_actual,
                k,
                k_scales,
                v_pv,
                v_pv_scales,
                colsplit_128_partial_out,
                colsplit_128_partial_m,
                colsplit_128_partial_l,
                colsplit_128_fused_atom,
                split_128_default_splits,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        split_sweep = {}
        for active_splits in (4, 8, 16, 32):
            sweep_partial_out = torch.empty(
                (8, active_splits, 16, HEAD_DIM),
                device=device,
                dtype=torch.float32,
            )
            sweep_partial_m = torch.empty(
                (8, active_splits, 16), device=device, dtype=torch.float32
            )
            sweep_partial_l = torch.empty(
                (8, active_splits, 16), device=device, dtype=torch.float32
            )
            sweep_out = torch.empty((128, HEAD_DIM), device=device, dtype=torch.float32)
            split_sweep[f"bench_fused_atom_attention_split_128rows_s{active_splits}"] = (
                event_ms(
                    lambda active_splits=active_splits,
                    sweep_partial_out=sweep_partial_out,
                    sweep_partial_m=sweep_partial_m,
                    sweep_partial_l=sweep_partial_l,
                    sweep_out=sweep_out: ext.fused_atom_attention_split_128rows(
                        q_actual,
                        q_scales_actual,
                        k,
                        k_scales,
                        v_pv,
                        v_pv_scales,
                        sweep_partial_out,
                        sweep_partial_m,
                        sweep_partial_l,
                        sweep_out,
                        active_splits,
                    ),
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            )
        result.update(split_sweep)
        colsplit_sweep = {}
        for active_splits in (16, 32):
            sweep_partial_out = torch.empty(
                (8, active_splits, 16, HEAD_DIM),
                device=device,
                dtype=torch.float32,
            )
            sweep_partial_m = torch.empty(
                (8, active_splits, 16), device=device, dtype=torch.float32
            )
            sweep_partial_l = torch.empty(
                (8, active_splits, 16), device=device, dtype=torch.float32
            )
            sweep_out = torch.empty((128, HEAD_DIM), device=device, dtype=torch.float32)
            colsplit_sweep[
                f"bench_fused_atom_attention_split_128rows_colsplit_s{active_splits}"
            ] = event_ms(
                lambda active_splits=active_splits,
                sweep_partial_out=sweep_partial_out,
                sweep_partial_m=sweep_partial_m,
                sweep_partial_l=sweep_partial_l,
                sweep_out=sweep_out: ext.fused_atom_attention_split_128rows_colsplit(
                    q_actual,
                    q_scales_actual,
                    k,
                    k_scales,
                    v_pv,
                    v_pv_scales,
                    sweep_partial_out,
                    sweep_partial_m,
                    sweep_partial_l,
                    sweep_out,
                    active_splits,
                ),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        result.update(colsplit_sweep)
        result.update(qk_runner_bench)

    print(result)


if __name__ == "__main__":
    main()
