from __future__ import annotations

import argparse
import json
import math

import torch

import flashinfer
from flashinfer import SfLayout
from flashinfer.fmha_nvfp4_sm120 import BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper
from flashinfer.jit import gen_fmha_nvfp4_sm120_module


BENCHMARK_KEY = "sm120_nvfp4_attention"


def default_output_group_span(head_dim: int) -> int:
    if head_dim == 128:
        return 1
    if head_dim == 256:
        return 2
    if head_dim == 512:
        return 4
    raise ValueError("head_dim must be one of {128, 256, 512}")


def tile_m_for_head_dim(head_dim: int) -> int:
    if head_dim in (128, 256):
        return 64
    if head_dim == 512:
        return 128
    raise ValueError("head_dim must be one of {128, 256, 512}")


def paged_tile_m_for_config(head_dim: int, sliding_window: int) -> int:
    if head_dim == 256 and sliding_window <= 0:
        return 128
    return tile_m_for_head_dim(head_dim)


def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def auto_split_kv_len(
    *,
    q_len: int,
    kv_len: int,
    group: int,
    head_dim: int,
    num_kv_heads: int,
    max_partial_bytes: int,
    api: str,
    sliding_window: int,
) -> int:
    tile_m = (
        paged_tile_m_for_config(head_dim, sliding_window)
        if api == "paged-wrapper"
        else tile_m_for_head_dim(head_dim)
    )
    if api == "paged-wrapper" and sliding_window > 0:
        split_kv_len = round_up(sliding_window, 128)
    else:
        q_tiles = max(1, round_up(q_len * group, tile_m) // tile_m)
        if api == "paged-wrapper" and head_dim == 512:
            q_tiles *= 3
        split_kv_len = min(max(q_tiles, 8), 96) * 128
    padded_rows = num_kv_heads * round_up(q_len * group, tile_m)
    bytes_per_split = padded_rows * (head_dim * 2 + 2 * 4)
    total_kv_tiles = math.ceil(kv_len / 128)
    split_kv_tiles = max(1, split_kv_len // 128)
    while True:
        num_splits = math.ceil(total_kv_tiles / split_kv_tiles)
        if num_splits * bytes_per_split <= max_partial_bytes:
            return split_kv_tiles * 128
        split_kv_tiles += 1


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


def make_bf16_shape_inputs(
    device: torch.device,
    *,
    q_len: int,
    group: int,
    kv_len: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    q = (torch.randn((q_len, group, head_dim), device=device, generator=gen) / 4).to(
        torch.bfloat16
    )
    k = (torch.randn((kv_len, head_dim), device=device, generator=gen) / 4).to(
        torch.bfloat16
    )
    v = (torch.randn((kv_len, head_dim), device=device, generator=gen) / 4).to(
        torch.bfloat16
    )
    return q.contiguous(), k.contiguous(), v.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the SM120 NVFP4 attention JIT module."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--q-len", type=int, required=True)
    parser.add_argument("--kv-len", type=int, required=True)
    parser.add_argument("--head-dim", type=int, choices=(128, 256, 512), required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument(
        "--split-kv-len",
        type=int,
        default=0,
        help=(
            "Split length in tokens. Use 0 to follow the production wrapper "
            "auto-selection while respecting --max-partial-bytes."
        ),
    )
    parser.add_argument(
        "--max-partial-bytes",
        type=int,
        default=1 << 30,
        help="Partial/split scratch budget used when --split-kv-len=0.",
    )
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
            "dense benchmarks the low-level prepacked dense_run binding; "
            "paged-wrapper benchmarks the production Python wrapper, including "
            "Q quantization, native paged KV loads, fused attention, and output scatter."
        ),
    )
    parser.add_argument(
        "--v-layout",
        choices=("linear", "pv"),
        default="linear",
        help=(
            "V cache layout for --mode=paged-wrapper. 'linear' matches "
            "nvfp4_quantize_paged_kv_cache(v_data_layout='linear') and measures "
            "the standard vLLM paged-KV input layout plus wrapper conversion "
            "to PV. 'pv' matches v_data_layout='pv' and skips conversion. "
            "Dense mode always uses v_layout='pv'."
        ),
    )
    parser.add_argument("--num-kv-heads", type=int, default=1)
    args = parser.parse_args()

    if args.kv_len % 128 != 0:
        raise ValueError("--kv-len must be a multiple of 128")
    if args.split_kv_len < 0 or args.split_kv_len % 128 != 0:
        raise ValueError("--split-kv-len must be 0 or a positive multiple of 128")
    if args.max_partial_bytes <= 0:
        raise ValueError("--max-partial-bytes must be positive")
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
    split_kv_len = (
        auto_split_kv_len(
            q_len=args.q_len,
            kv_len=args.kv_len,
            group=args.group,
            head_dim=args.head_dim,
            num_kv_heads=args.num_kv_heads,
            max_partial_bytes=args.max_partial_bytes,
            api=args.mode,
            sliding_window=args.sliding_window,
        )
        if args.split_kv_len == 0
        else args.split_kv_len
    )

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    q_rows = args.q_len * args.group

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
        if args.v_layout == "pv":
            (k_pages, v_pages), (k_sf, v_sf), k_scale, v_scale = (
                flashinfer.nvfp4_quantize_paged_kv_cache(
                    k_bf16,
                    v_bf16,
                    kv_layout="NHD",
                    v_data_layout="pv",
                    v_scale_layout="pv",
                )
            )
            v_cache_uses_pv_layout = True
            v_cache_sf_layout = "pv"
        else:
            (k_pages, v_pages), (k_sf, v_sf), k_scale, v_scale = (
                flashinfer.nvfp4_quantize_paged_kv_cache(
                    k_bf16,
                    v_bf16,
                    kv_layout="NHD",
                    v_data_layout="linear",
                    v_scale_layout="trtllm_interleaved",
                )
            )
            v_cache_uses_pv_layout = False
            v_cache_sf_layout = "trtllm_interleaved"
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
            split_kv_len=split_kv_len,
            output_group_span=output_group_span,
            v_cache_uses_pv_layout=v_cache_uses_pv_layout,
        )

        def run() -> None:
            wrapper.run(
                q,
                (k_pages, v_pages),
                (k_sf, v_sf),
                k_scale=k_scale,
                v_scale=v_scale,
                v_cache_uses_pv_layout=v_cache_uses_pv_layout,
                v_cache_sf_layout=v_cache_sf_layout,
                out=out,
            )

        run()
        torch.cuda.synchronize()
        result = {
            "fmha_nvfp4_sm120_jit": True,
            "api": "paged",
            "v_layout": args.v_layout,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "group": args.group,
            "num_kv_heads": args.num_kv_heads,
            "output_group_span": output_group_span,
            "split_kv_len": split_kv_len,
            "causal": bool(args.causal),
            "sliding_window": args.sliding_window,
            "logits_soft_cap": args.logits_soft_cap,
            "output_finite": bool(torch.isfinite(out.float()).all()),
            BENCHMARK_KEY: event_ms(run, warmup=args.warmup, repeat=args.repeat),
        }
        print(json.dumps(result, sort_keys=True))
        return

    q, k_bf16, v_bf16 = make_bf16_shape_inputs(
        device,
        q_len=args.q_len,
        group=args.group,
        kv_len=args.kv_len,
        head_dim=args.head_dim,
    )
    q_dense = q.reshape(q_rows, args.head_dim)
    q_rows_padded = round_up(q_rows, tile_m_for_head_dim(args.head_dim))
    if q_rows_padded != q_rows:
        q_dense_padded = torch.zeros(
            (q_rows_padded, args.head_dim), dtype=q.dtype, device=device
        )
        q_dense_padded[:q_rows, :] = q_dense
        q_dense = q_dense_padded
    q_packed, q_scales, q_global = quantize_cutlass(q_dense)
    if q_scales.size(0) > q_packed.size(0):
        q_packed_padded = torch.zeros(
            (q_scales.size(0), q_packed.size(1)),
            dtype=q_packed.dtype,
            device=device,
        )
        q_packed_padded[: q_packed.size(0), :] = q_packed
        q_packed = q_packed_padded
    if q_scales.size(0) != q_packed.size(0):
        raise RuntimeError("Q packed and scale row counts do not match after padding.")
    q_rows_packed = int(q_packed.size(0))
    k_packed, k_scales, k_global = quantize_cutlass(k_bf16)
    v_pv_packed, v_pv_scales, v_global = quantize_cutlass(
        v_bf16.T.contiguous()
    )
    qk_alpha = float((1.0 / (q_global * k_global)).item())
    pv_alpha = float((1.0 / v_global).item())
    split_kv_tiles = split_kv_len // 128
    num_splits = math.ceil(args.kv_len / split_kv_len)

    partial = torch.empty(
        (num_splits, q_rows_packed, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    split_m = torch.empty((num_splits, q_rows_packed), dtype=torch.float32, device=device)
    split_l = torch.empty((num_splits, q_rows_packed), dtype=torch.float32, device=device)
    out = torch.empty((q_rows_packed, args.head_dim), dtype=torch.bfloat16, device=device)
    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    module = gen_fmha_nvfp4_sm120_module(
        args.head_dim,
        causal=bool(args.causal),
        use_sliding_window=args.sliding_window > 0,
        use_logits_soft_cap=float(args.logits_soft_cap) > 0.0,
        v_cache_uses_pv_layout=True,
    ).build_and_load()

    def run() -> None:
        module.dense_run(
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
            torch.cuda.current_stream(device).cuda_stream,
        )

    run()
    torch.cuda.synchronize()
    result = {
        "fmha_nvfp4_sm120_jit": True,
        "api": "dense",
        "v_layout": "pv",
        "q_len": args.q_len,
        "kv_len": args.kv_len,
        "head_dim": args.head_dim,
        "group": args.group,
        "output_group_span": output_group_span,
        "split_kv_len": split_kv_len,
        "causal": bool(args.causal),
        "sliding_window": args.sliding_window,
        "logits_soft_cap": args.logits_soft_cap,
        "output_finite": bool(torch.isfinite(out[:q_rows].float()).all()),
        BENCHMARK_KEY: event_ms(run, warmup=args.warmup, repeat=args.repeat),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
