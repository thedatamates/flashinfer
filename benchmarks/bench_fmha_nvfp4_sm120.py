from __future__ import annotations

import argparse
import json
import math

import torch

import flashinfer
from flashinfer.fmha_nvfp4_sm120 import BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper
from flashinfer.jit import gen_fmha_nvfp4_sm120_module

from bench_sm120_nvfp4_cutlass_fused_attention import (
    event_ms,
    make_shape_inputs,
    quantize_cutlass,
)
from bench_sm120_nvfp4_ref_attention import nvfp4_rowmajor_to_fp32


def default_output_group_span(head_dim: int) -> int:
    if head_dim == 128:
        return 1
    if head_dim == 256:
        return 2
    if head_dim == 512:
        return 4
    raise ValueError("head_dim must be one of {128, 256, 512}")


def bench_key(output_group_span: int) -> str:
    if output_group_span == 1:
        return "bench_sm120_qkv_online_register_q_splitkv_full_grid"
    return (
        f"bench_sm120_qkv_online_register_q_splitkv_reuse"
        f"{output_group_span}_full_grid"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the source-tree SM120 NVFP4 FMHA JIT module."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--q-len", type=int, required=True)
    parser.add_argument("--kv-len", type=int, required=True)
    parser.add_argument("--head-dim", type=int, choices=(128, 256, 512), required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--split-kv-len", type=int, required=True)
    parser.add_argument("--output-group-span", type=int, choices=(1, 2, 4), default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--sliding-window", type=int, default=-1)
    parser.add_argument("--logits-soft-cap", type=float, default=0.0)
    parser.add_argument(
        "--mode",
        choices=("dense", "paged-wrapper"),
        default="dense",
        help=(
            "dense benchmarks the low-level prepacked run_dense binding; "
            "paged-wrapper benchmarks the production Python wrapper, including "
            "Q quantization, native paged KV loads, fused attention, and output scatter."
        ),
    )
    parser.add_argument("--num-kv-heads", type=int, default=1)
    args = parser.parse_args()

    if args.kv_len % 128 != 0:
        raise ValueError("--kv-len must be a multiple of 128")
    if args.split_kv_len <= 0 or args.split_kv_len % 128 != 0:
        raise ValueError("--split-kv-len must be a positive multiple of 128")
    if args.q_len * args.group <= 0:
        raise ValueError("--q-len * --group must be positive")

    output_group_span = (
        default_output_group_span(args.head_dim)
        if args.output_group_span == 0
        else args.output_group_span
    )
    if args.head_dim == 128 and output_group_span != 1:
        raise ValueError("D128 only supports output_group_span=1")
    if args.head_dim % (output_group_span * 128) != 0:
        raise ValueError("head_dim must be divisible by output_group_span * 128")
    if args.num_kv_heads <= 0:
        raise ValueError("--num-kv-heads must be positive")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    q_rows = args.q_len * args.group
    key = bench_key(output_group_span)

    if args.mode == "paged-wrapper":
        gen = torch.Generator(device=device)
        gen.manual_seed(1234)
        num_qo_heads = args.group * args.num_kv_heads
        q = (
            torch.randn(
                (args.q_len, num_qo_heads, args.head_dim),
                device=device,
                generator=gen,
            )
            / 4
        ).to(torch.bfloat16)
        page_size = 16
        num_pages = math.ceil(args.kv_len / page_size)
        k_bf16 = (
            torch.randn(
                (num_pages, page_size, args.num_kv_heads, args.head_dim),
                device=device,
                generator=gen,
            )
            / 4
        ).to(torch.bfloat16)
        v_bf16 = (
            torch.randn(
                (num_pages, page_size, args.num_kv_heads, args.head_dim),
                device=device,
                generator=gen,
            )
            / 4
        ).to(torch.bfloat16)
        (k_pages, v_pages), (k_sf, v_sf), k_scale, v_scale = (
            flashinfer.nvfp4_quantize_paged_kv_cache(
                k_bf16,
                v_bf16,
                kv_layout="NHD",
                v_data_layout="pv",
                v_scale_layout="pv",
            )
        )
        block_tables = torch.arange(num_pages, dtype=torch.int32, device=device).view(
            1, num_pages
        )
        qo_indptr = torch.tensor([0, args.q_len], dtype=torch.int32, device="cpu")
        kv_lens = torch.tensor([args.kv_len], dtype=torch.int32, device="cpu")
        out = torch.empty_like(q)
        workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper(workspace)
        wrapper.plan(
            qo_indptr,
            block_tables,
            kv_lens,
            num_qo_heads=num_qo_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            causal=bool(args.causal),
            window_left=args.sliding_window,
            logits_soft_cap=float(args.logits_soft_cap),
            split_kv_len=args.split_kv_len,
            output_group_span=output_group_span,
        )

        def run() -> None:
            wrapper.run(
                q,
                (k_pages, v_pages),
                (k_sf, v_sf),
                k_scale=k_scale,
                v_scale=v_scale,
                out=out,
            )

        run()
        torch.cuda.synchronize()
        result = {
            "fmha_nvfp4_sm120_jit": True,
            "production_paged_wrapper": True,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "group": args.group,
            "num_kv_heads": args.num_kv_heads,
            "output_group_span": output_group_span,
            "split_kv_len": args.split_kv_len,
            "causal": bool(args.causal),
            "sliding_window": args.sliding_window,
            "logits_soft_cap": args.logits_soft_cap,
            "output_finite": bool(torch.isfinite(out.float()).all()),
            key: event_ms(run, warmup=args.warmup, repeat=args.repeat),
        }
        print(json.dumps(result, sort_keys=True))
        return

    q, k, v, k_scales_ref, v_scales_ref = make_shape_inputs(
        device,
        q_len=args.q_len,
        group=args.group,
        kv_len=args.kv_len,
        head_dim=args.head_dim,
    )
    k_ref = nvfp4_rowmajor_to_fp32(k, k_scales_ref).to(torch.bfloat16)
    v_ref = nvfp4_rowmajor_to_fp32(v, v_scales_ref).to(torch.bfloat16)
    q_packed, q_scales, q_global = quantize_cutlass(
        q.reshape(q_rows, args.head_dim)
    )
    k_packed, k_scales, k_global = quantize_cutlass(k_ref)
    v_pv_packed, v_pv_scales, v_global = quantize_cutlass(
        v_ref.T.contiguous()
    )
    qk_alpha = float((1.0 / (q_global * k_global)).item())
    pv_alpha = float((1.0 / v_global).item())
    split_kv_tiles = args.split_kv_len // 128
    num_splits = math.ceil(args.kv_len / args.split_kv_len)

    partial = torch.empty(
        (num_splits, q_rows, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    split_m = torch.empty((num_splits, q_rows), dtype=torch.float32, device=device)
    split_l = torch.empty((num_splits, q_rows), dtype=torch.float32, device=device)
    out = torch.empty((q_rows, args.head_dim), dtype=torch.bfloat16, device=device)
    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    module = gen_fmha_nvfp4_sm120_module().build_and_load()

    def run() -> None:
        module.run_dense(
            q_packed,
            q_scales,
            k_packed,
            k_scales,
            v_pv_packed,
            v_pv_scales,
            partial,
            split_m,
            split_l,
            out,
            workspace,
            qk_alpha,
            pv_alpha,
            split_kv_tiles,
            args.q_len,
            args.group,
            args.kv_len,
            bool(args.causal),
            args.sliding_window,
            float(args.logits_soft_cap),
            output_group_span,
        )

    run()
    torch.cuda.synchronize()
    result = {
        "fmha_nvfp4_sm120_jit": True,
        "production_paged_wrapper": False,
        "q_len": args.q_len,
        "kv_len": args.kv_len,
        "head_dim": args.head_dim,
        "group": args.group,
        "output_group_span": output_group_span,
        "causal": bool(args.causal),
        "sliding_window": args.sliding_window,
        "logits_soft_cap": args.logits_soft_cap,
        "output_finite": bool(torch.isfinite(out.float()).all()),
        key: event_ms(run, warmup=args.warmup, repeat=args.repeat),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
