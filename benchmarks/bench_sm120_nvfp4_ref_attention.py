from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


Q_LEN = 512
GROUP = 8
KV_LEN = 32768
HEAD_DIM = 512
PACKED_HEAD_DIM = HEAD_DIM // 2
SCALE_COLS = HEAD_DIM // 16
PROB_GLOBAL_SCALE = 6.0 * 448.0

E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


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
    split_kv_len = os.environ.get("SM120_NVFP4_SPLIT_KV_LEN", "2048")
    fused_kv_tile = os.environ.get("SM120_NVFP4_FUSED_KV_TILE", "64")
    return load(
        name="sm120_nvfp4_ref_attention_ext",
        sources=[str(root / "benchmarks" / "sm120_nvfp4_ref_attention.cu")],
        extra_include_paths=[
            str(root / "include"),
            str(root / "3rdparty" / "cutlass" / "include"),
            str(root / "3rdparty" / "cutlass" / "tools" / "util" / "include"),
        ],
        extra_cuda_cflags=[
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_120f,code=sm_120f",
            f"-DSM120_NVFP4_SPLIT_KV_LEN={split_kv_len}",
            f"-DSM120_NVFP4_FUSED_KV_TILE={fused_kv_tile}",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def fp32_to_nvfp4_rowmajor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    last_dim = x.shape[-1]
    if last_dim % 16 != 0:
        raise ValueError("NVFP4 row-major quantization requires last dim divisible by 16")

    values = E2M1_VALUES.to(device=x.device)
    scale_cols = last_dim // 16
    x_groups = x.float().reshape(-1, scale_cols, 16)
    max_abs = x_groups.abs().amax(dim=-1).clamp_min(1.0e-8)
    scale = (max_abs / 6.0).to(torch.float8_e4m3fn)
    scale_f32 = scale.float().clamp_min(1.0e-8)
    normalized = (x_groups / scale_f32.unsqueeze(-1)).clamp(-6.0, 6.0)
    distances = (normalized.unsqueeze(-1) - values).abs()
    codes = distances.argmin(dim=-1).to(torch.uint8).reshape(-1, last_dim)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scale_bytes = scale.view(torch.uint8).contiguous()
    return packed, scale_bytes


def nvfp4_rowmajor_to_fp32(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = E2M1_VALUES.to(device=packed.device)
    codes = torch.empty(
        (packed.shape[0], packed.shape[1] * 2),
        device=packed.device,
        dtype=torch.uint8,
    )
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = (packed >> 4) & 0x0F
    scale_f32 = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return values[codes.long()] * scale_f32


def make_inputs(device: torch.device):
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    q = (torch.randn((Q_LEN, GROUP, HEAD_DIM), device=device, generator=gen) / 4).to(
        torch.bfloat16
    )
    k_bf16 = (torch.randn((KV_LEN, HEAD_DIM), device=device, generator=gen) / 4).to(
        torch.bfloat16
    )
    v_bf16 = (torch.randn((KV_LEN, HEAD_DIM), device=device, generator=gen) / 4).to(
        torch.bfloat16
    )
    k, k_scales = fp32_to_nvfp4_rowmajor(k_bf16)
    v, v_scales = fp32_to_nvfp4_rowmajor(v_bf16)
    return q.contiguous(), k, v, k_scales, v_scales


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
) -> torch.Tensor:
    k_ref = nvfp4_rowmajor_to_fp32(k, k_scales)
    v_ref = nvfp4_rowmajor_to_fp32(v, v_scales)
    q_ref = q.float().transpose(0, 1).contiguous()
    scores = torch.matmul(q_ref, k_ref.T) / math.sqrt(HEAD_DIM)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_ref).transpose(0, 1).contiguous()
    return out.to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--skip-extension", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--full-staged", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--split-fused", action="store_true")
    parser.add_argument("--split-fused-pipelined", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    q, k, v, k_scales, v_scales = make_inputs(device)
    ref = reference_attention(q, k, v, k_scales, v_scales)
    print(
        {
            "reference_shape": tuple(ref.shape),
            "reference_dtype": str(ref.dtype),
            "reference_finite": bool(torch.isfinite(ref.float()).all().item()),
            "k_shape": tuple(k.shape),
            "scale_shape": tuple(k_scales.shape),
        }
    )

    if args.skip_extension:
        return

    ext = build_extension()
    split_kv_len = int(ext.split_kv_len)
    num_splits = int(ext.num_splits)
    fused_kv_tile = int(ext.fused_kv_tile)
    print(
        {
            "compiled_split_kv_len": split_kv_len,
            "compiled_num_splits": num_splits,
            "compiled_fused_kv_tile": fused_kv_tile,
        }
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
    q_input = q.reshape(Q_LEN * GROUP, HEAD_DIM).float()
    print(
        {
            "q_quant_packed_match": bool(torch.equal(q_actual, q_expected)),
            "q_quant_scale_match": bool(torch.equal(q_scales_actual, q_scales_expected)),
            "q_quant_packed_diff": int((q_actual != q_expected).sum().item()),
            "q_quant_scale_diff": int((q_scales_actual != q_scales_expected).sum().item()),
            "q_quant_dequant_delta_mean": float(q_dequant_delta.mean().item()),
            "q_quant_dequant_delta_max": float(q_dequant_delta.max().item()),
            "q_quant_expected_error_mean": float((q_expected_f32 - q_input).abs().mean().item()),
            "q_quant_actual_error_mean": float((q_actual_f32 - q_input).abs().mean().item()),
        }
    )

    q_rows = torch.arange(16, device=device) * GROUP
    k_ref_f32 = nvfp4_rowmajor_to_fp32(k, k_scales)
    v_ref_f32 = nvfp4_rowmajor_to_fp32(v, v_scales)
    v_pv, v_pv_scales = fp32_to_nvfp4_rowmajor(v_ref_f32.T.contiguous())
    qk_ref = torch.matmul(q_actual_f32[q_rows].float(), k_ref_f32[:16].float().T)
    qk_tile = torch.empty((16, 16), device=device, dtype=torch.float32)
    ext.qk_tile_mma_debug(q_actual, q_scales_actual, k, k_scales, qk_tile)
    torch.cuda.synchronize()
    qk_delta = (qk_tile - qk_ref).abs()
    qk_ref_norm = torch.linalg.vector_norm(qk_ref)
    qk_tile_norm = torch.linalg.vector_norm(qk_tile)
    qk_cos = torch.sum(qk_tile * qk_ref) / torch.clamp(qk_ref_norm * qk_tile_norm, min=1.0e-20)
    print(
        {
            "qk_tile_finite": bool(torch.isfinite(qk_tile).all().item()),
            "qk_tile_mean_abs": float(qk_delta.mean().item()),
            "qk_tile_max_abs": float(qk_delta.max().item()),
            "qk_tile_cosine": float(qk_cos.item()),
            "qk_tile_sample": [float(x) for x in qk_tile.flatten()[:4].tolist()],
            "qk_ref_sample": [float(x) for x in qk_ref.flatten()[:4].tolist()],
        }
    )

    qk_full_ref = torch.matmul(q_actual_f32[q_rows].float(), k_ref_f32.float().T)
    qk_full = torch.empty((16, KV_LEN), device=device, dtype=torch.float32)
    ext.qk_full_mma_debug(q_actual, q_scales_actual, k, k_scales, qk_full)
    torch.cuda.synchronize()
    qk_full_delta = (qk_full - qk_full_ref).abs()
    qk_full_ref_norm = torch.linalg.vector_norm(qk_full_ref)
    qk_full_norm = torch.linalg.vector_norm(qk_full)
    qk_full_cos = torch.sum(qk_full * qk_full_ref) / torch.clamp(
        qk_full_ref_norm * qk_full_norm, min=1.0e-20
    )
    print(
        {
            "qk_full_finite": bool(torch.isfinite(qk_full).all().item()),
            "qk_full_mean_abs": float(qk_full_delta.mean().item()),
            "qk_full_max_abs": float(qk_full_delta.max().item()),
            "qk_full_cosine": float(qk_full_cos.item()),
            "qk_full_sample": [float(x) for x in qk_full.flatten()[:4].tolist()],
            "qk_full_ref_sample": [float(x) for x in qk_full_ref.flatten()[:4].tolist()],
        }
    )

    p_ref = torch.softmax(qk_full_ref / math.sqrt(HEAD_DIM), dim=-1)
    p_packed = torch.empty((16, KV_LEN // 2), device=device, dtype=torch.uint8)
    p_scales = torch.empty((16, KV_LEN // 16), device=device, dtype=torch.uint8)
    ext.softmax_quant_p_debug(qk_full, p_packed, p_scales)
    torch.cuda.synchronize()
    p_dequant_scaled = nvfp4_rowmajor_to_fp32(p_packed, p_scales)
    p_dequant = p_dequant_scaled / PROB_GLOBAL_SCALE
    p_delta = (p_dequant - p_ref).abs()
    p_ref_norm = torch.linalg.vector_norm(p_ref)
    p_dequant_norm = torch.linalg.vector_norm(p_dequant)
    p_cos = torch.sum(p_dequant * p_ref) / torch.clamp(
        p_ref_norm * p_dequant_norm, min=1.0e-20
    )
    print(
        {
            "p_quant_finite": bool(torch.isfinite(p_dequant).all().item()),
            "p_quant_row_sum_min": float(p_dequant.sum(dim=-1).min().item()),
            "p_quant_row_sum_max": float(p_dequant.sum(dim=-1).max().item()),
            "p_quant_mean_abs": float(p_delta.mean().item()),
            "p_quant_max_abs": float(p_delta.max().item()),
            "p_quant_cosine": float(p_cos.item()),
            "p_quant_scale_nonzero": int((p_scales != 0).sum().item()),
            "p_quant_sample": [float(x) for x in p_dequant.flatten()[:4].tolist()],
            "p_ref_sample": [float(x) for x in p_ref.flatten()[:4].tolist()],
        }
    )

    v_pv_ref_f32 = nvfp4_rowmajor_to_fp32(v_pv, v_pv_scales)
    pv_tile = torch.empty((16, 16), device=device, dtype=torch.float32)
    ext.pv_tile_mma_debug(p_packed, p_scales, v_pv, v_pv_scales, pv_tile)
    torch.cuda.synchronize()
    pv_tile_dequant = pv_tile / PROB_GLOBAL_SCALE
    pv_ref_quant = torch.matmul(p_dequant.float(), v_pv_ref_f32[:16].float().T)
    pv_ref_exact_p = torch.matmul(p_ref.float(), v_pv_ref_f32[:16].float().T)
    pv_quant_delta = (pv_tile_dequant - pv_ref_quant).abs()
    pv_exact_delta = (pv_tile_dequant - pv_ref_exact_p).abs()
    pv_ref_norm = torch.linalg.vector_norm(pv_ref_quant)
    pv_tile_norm = torch.linalg.vector_norm(pv_tile_dequant)
    pv_cos = torch.sum(pv_tile_dequant * pv_ref_quant) / torch.clamp(
        pv_ref_norm * pv_tile_norm, min=1.0e-20
    )
    print(
        {
            "pv_tile_finite": bool(torch.isfinite(pv_tile_dequant).all().item()),
            "pv_tile_vs_quant_p_mean_abs": float(pv_quant_delta.mean().item()),
            "pv_tile_vs_quant_p_max_abs": float(pv_quant_delta.max().item()),
            "pv_tile_vs_quant_p_cosine": float(pv_cos.item()),
            "pv_tile_vs_exact_p_mean_abs": float(pv_exact_delta.mean().item()),
            "pv_tile_vs_exact_p_max_abs": float(pv_exact_delta.max().item()),
            "pv_tile_sample": [float(x) for x in pv_tile_dequant.flatten()[:4].tolist()],
            "pv_ref_sample": [float(x) for x in pv_ref_quant.flatten()[:4].tolist()],
        }
    )

    pv_full = torch.empty((16, HEAD_DIM), device=device, dtype=torch.float32)
    ext.pv_full_mma_debug(p_packed, p_scales, v_pv, v_pv_scales, pv_full)
    torch.cuda.synchronize()
    pv_full_dequant = pv_full / PROB_GLOBAL_SCALE
    pv_full_ref_quant = torch.matmul(p_dequant.float(), v_pv_ref_f32.float().T)
    pv_full_ref_exact_p = torch.matmul(p_ref.float(), v_pv_ref_f32.float().T)
    pv_full_quant_delta = (pv_full_dequant - pv_full_ref_quant).abs()
    pv_full_exact_delta = (pv_full_dequant - pv_full_ref_exact_p).abs()
    pv_full_ref_norm = torch.linalg.vector_norm(pv_full_ref_quant)
    pv_full_norm = torch.linalg.vector_norm(pv_full_dequant)
    pv_full_cos = torch.sum(pv_full_dequant * pv_full_ref_quant) / torch.clamp(
        pv_full_ref_norm * pv_full_norm, min=1.0e-20
    )
    print(
        {
            "pv_full_finite": bool(torch.isfinite(pv_full_dequant).all().item()),
            "pv_full_vs_quant_p_mean_abs": float(pv_full_quant_delta.mean().item()),
            "pv_full_vs_quant_p_max_abs": float(pv_full_quant_delta.max().item()),
            "pv_full_vs_quant_p_cosine": float(pv_full_cos.item()),
            "pv_full_vs_exact_p_mean_abs": float(pv_full_exact_delta.mean().item()),
            "pv_full_vs_exact_p_max_abs": float(pv_full_exact_delta.max().item()),
            "pv_full_sample": [float(x) for x in pv_full_dequant.flatten()[:4].tolist()],
            "pv_full_ref_sample": [float(x) for x in pv_full_ref_quant.flatten()[:4].tolist()],
        }
    )

    if args.bench:
        print(
            {
                "bench_qk_full": event_ms(
                    lambda: ext.qk_full_mma_debug(
                        q_actual, q_scales_actual, k, k_scales, qk_full
                    ),
                    warmup=args.warmup,
                    repeat=args.repeat,
                ),
                "bench_softmax_quant_p": event_ms(
                    lambda: ext.softmax_quant_p_debug(qk_full, p_packed, p_scales),
                    warmup=args.warmup,
                    repeat=args.repeat,
                ),
                "bench_pv_full": event_ms(
                    lambda: ext.pv_full_mma_debug(
                        p_packed, p_scales, v_pv, v_pv_scales, pv_full
                    ),
                    warmup=args.warmup,
                    repeat=args.repeat,
                ),
            }
        )

    staged_out_dequant = None
    if args.full_staged:
        qk_all = torch.empty((Q_LEN * GROUP, KV_LEN), device=device, dtype=torch.float32)
        p_all = torch.empty((Q_LEN * GROUP, KV_LEN // 2), device=device, dtype=torch.uint8)
        p_scales_all = torch.empty(
            (Q_LEN * GROUP, KV_LEN // 16), device=device, dtype=torch.uint8
        )
        out_all = torch.empty((Q_LEN * GROUP, HEAD_DIM), device=device, dtype=torch.float32)

        ext.qk_all_mma_debug(q_actual, q_scales_actual, k, k_scales, qk_all)
        ext.softmax_quant_p_all_debug(qk_all, p_all, p_scales_all)
        ext.pv_all_mma_debug(p_all, p_scales_all, v_pv, v_pv_scales, out_all)
        torch.cuda.synchronize()

        staged_out_dequant = (out_all / PROB_GLOBAL_SCALE).reshape(Q_LEN, GROUP, HEAD_DIM)
        full_delta = (staged_out_dequant - ref.float()).abs()
        first_block_delta = (
            out_all[q_rows] / PROB_GLOBAL_SCALE - pv_full_ref_quant
        ).abs()
        print(
            {
                "full_staged_finite": bool(torch.isfinite(staged_out_dequant).all().item()),
                "full_staged_vs_exact_ref_mean_abs": float(full_delta.mean().item()),
                "full_staged_vs_exact_ref_max_abs": float(full_delta.max().item()),
                "full_staged_first_block_vs_quant_ref_mean_abs": float(
                    first_block_delta.mean().item()
                ),
                "full_staged_first_block_vs_quant_ref_max_abs": float(
                    first_block_delta.max().item()
                ),
            }
        )

        if args.bench:
            print(
                {
                    "bench_full_qk_all": event_ms(
                        lambda: ext.qk_all_mma_debug(
                            q_actual, q_scales_actual, k, k_scales, qk_all
                        ),
                        warmup=args.warmup,
                        repeat=args.repeat,
                    ),
                    "bench_full_softmax_quant_p_all": event_ms(
                        lambda: ext.softmax_quant_p_all_debug(qk_all, p_all, p_scales_all),
                        warmup=args.warmup,
                        repeat=args.repeat,
                    ),
                    "bench_full_pv_all": event_ms(
                        lambda: ext.pv_all_mma_debug(
                            p_all, p_scales_all, v_pv, v_pv_scales, out_all
                        ),
                        warmup=args.warmup,
                        repeat=args.repeat,
                    ),
                }
            )

    if args.fused:
        fused_rows = torch.empty((Q_LEN * GROUP, HEAD_DIM), device=device, dtype=torch.float32)
        ext.fused_attention_all_debug(
            q_actual, q_scales_actual, k, k_scales, v_pv, v_pv_scales, fused_rows
        )
        torch.cuda.synchronize()
        fused = fused_rows.reshape(Q_LEN, GROUP, HEAD_DIM)
        fused_delta = (fused - ref.float()).abs()
        result = {
            "fused_finite": bool(torch.isfinite(fused).all().item()),
            "fused_vs_exact_ref_mean_abs": float(fused_delta.mean().item()),
            "fused_vs_exact_ref_max_abs": float(fused_delta.max().item()),
            "fused_sample": [float(x) for x in fused.flatten()[:4].tolist()],
            "exact_ref_sample": [float(x) for x in ref.float().flatten()[:4].tolist()],
        }
        if staged_out_dequant is not None:
            staged_delta = (fused - staged_out_dequant).abs()
            result.update(
                {
                    "fused_vs_staged_mean_abs": float(staged_delta.mean().item()),
                    "fused_vs_staged_max_abs": float(staged_delta.max().item()),
                }
            )
        print(result)

        if args.bench:
            print(
                {
                    "bench_fused_attention_all": event_ms(
                        lambda: ext.fused_attention_all_debug(
                            q_actual,
                            q_scales_actual,
                            k,
                            k_scales,
                            v_pv,
                            v_pv_scales,
                            fused_rows,
                        ),
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                }
            )

    if args.split_fused or args.split_fused_pipelined:
        k_frag_pre = torch.empty(
            (KV_LEN // 16, HEAD_DIM // 64, 32, 4), device=device, dtype=torch.int32
        )
        k_scale_pre = torch.empty(
            (KV_LEN // 16, HEAD_DIM // 64, 32, 2), device=device, dtype=torch.int32
        )
        v_frag_pre = torch.empty(
            (HEAD_DIM // 16, KV_LEN // 64, 32, 4), device=device, dtype=torch.int32
        )
        v_scale_pre = torch.empty(
            (HEAD_DIM // 16, KV_LEN // 64, 32, 2), device=device, dtype=torch.int32
        )
        ext.prepack_k_fragments_debug(k, k_scales, k_frag_pre, k_scale_pre)
        ext.prepack_v_fragments_debug(v_pv, v_pv_scales, v_frag_pre, v_scale_pre)
        torch.cuda.synchronize()

        partial_o = torch.empty(
            (num_splits, Q_LEN * GROUP, HEAD_DIM), device=device, dtype=torch.float32
        )
        partial_m = torch.empty((num_splits, Q_LEN * GROUP), device=device, dtype=torch.float32)
        partial_l = torch.empty((num_splits, Q_LEN * GROUP), device=device, dtype=torch.float32)
        split_weights = torch.empty_like(partial_m)
        split_rows = torch.empty((Q_LEN * GROUP, HEAD_DIM), device=device, dtype=torch.float32)
        partial_fn_name = (
            "fused_attention_split_partial_pipelined_debug"
            if args.split_fused_pipelined
            else "fused_attention_split_partial_debug"
        )
        partial_fn = getattr(ext, partial_fn_name)
        partial_fn(
            q_actual,
            q_scales_actual,
            k_frag_pre,
            k_scale_pre,
            v_frag_pre,
            v_scale_pre,
            partial_o,
            partial_m,
            partial_l,
        )
        ext.fused_attention_split_reduce_debug(
            partial_o, partial_m, partial_l, split_weights, split_rows
        )
        torch.cuda.synchronize()
        split_fused = split_rows.reshape(Q_LEN, GROUP, HEAD_DIM)
        split_delta = (split_fused - ref.float()).abs()
        result = {
            "split_fused_finite": bool(torch.isfinite(split_fused).all().item()),
            "split_fused_vs_exact_ref_mean_abs": float(split_delta.mean().item()),
            "split_fused_vs_exact_ref_max_abs": float(split_delta.max().item()),
            "split_fused_sample": [float(x) for x in split_fused.flatten()[:4].tolist()],
            "exact_ref_sample": [float(x) for x in ref.float().flatten()[:4].tolist()],
        }
        if staged_out_dequant is not None:
            staged_delta = (split_fused - staged_out_dequant).abs()
            result.update(
                {
                    "split_fused_vs_staged_mean_abs": float(staged_delta.mean().item()),
                    "split_fused_vs_staged_max_abs": float(staged_delta.max().item()),
                }
            )
        print(result)

        if args.bench:
            print(
                {
                    f"bench_{partial_fn_name}": event_ms(
                        lambda: partial_fn(
                            q_actual,
                            q_scales_actual,
                            k_frag_pre,
                            k_scale_pre,
                            v_frag_pre,
                            v_scale_pre,
                            partial_o,
                            partial_m,
                            partial_l,
                        ),
                        warmup=args.warmup,
                        repeat=args.repeat,
                    ),
                    "bench_split_fused_reduce": event_ms(
                        lambda: ext.fused_attention_split_reduce_debug(
                            partial_o, partial_m, partial_l, split_weights, split_rows
                        ),
                        warmup=args.warmup,
                        repeat=args.repeat,
                    ),
                    "bench_prepack_k": event_ms(
                        lambda: ext.prepack_k_fragments_debug(
                            k, k_scales, k_frag_pre, k_scale_pre
                        ),
                        warmup=args.warmup,
                        repeat=args.repeat,
                    ),
                    "bench_prepack_v": event_ms(
                        lambda: ext.prepack_v_fragments_debug(
                            v_pv, v_pv_scales, v_frag_pre, v_scale_pre
                        ),
                        warmup=args.warmup,
                        repeat=args.repeat,
                    ),
                }
            )

    out = torch.empty_like(q)
    ext.attention_b1_d512_g8_q512_kv32768(q, k, v, k_scales, v_scales, out)
    torch.cuda.synchronize()
    print(
        {
            "stub_out_shape": tuple(out.shape),
            "stub_out_dtype": str(out.dtype),
            "stub_out_finite": bool(torch.isfinite(out.float()).all().item()),
            "stub_mean_abs_vs_ref": float((out.float() - ref.float()).abs().mean().item()),
        }
    )


if __name__ == "__main__":
    main()
