from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from flashinfer.decode import xqa_batch_decode_with_kv_cache
from flashinfer.fp4_quantization import nvfp4_quantize_paged_kv_cache


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


def _parse_groups(value: str, max_group_size: int) -> list[int]:
    if value:
        groups = [int(group) for group in value.split(",")]
    else:
        if max_group_size < 1 or max_group_size & (max_group_size - 1):
            raise ValueError("max_group_size must be a power of two.")
        groups = []
        group = 1
        while group <= max_group_size:
            groups.append(group)
            group *= 2
    if any(group < 1 for group in groups):
        raise ValueError("group sizes must be positive.")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XQA GQA table for NVFP4 paged-KV decode attention."
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-group-size", type=int, default=8)
    parser.add_argument(
        "--group-sizes",
        type=str,
        default="",
        help="Comma-separated group sizes. Defaults to powers of two up to max-group-size.",
    )
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "nvfp4", "bf16"],
        help="Run one benchmark target. Useful for profiler kernel filtering.",
    )
    args = parser.parse_args()

    if args.head_dim not in (128, 256, 512):
        raise ValueError("head_dim must be one of 128, 256, or 512.")
    if args.only == "bf16" and args.head_dim > 256:
        raise ValueError("BF16 XQA baseline only supports head_dim <= 256.")
    if args.kv_len <= 0:
        raise ValueError("kv_len must be positive.")
    if args.num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive.")

    group_sizes = _parse_groups(args.group_sizes, args.max_group_size)

    torch.manual_seed(0)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16
    workspace = torch.empty(
        args.workspace_mib * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )

    pages_per_seq = math.ceil(args.kv_len / args.page_size)
    total_pages = args.batch_size * pages_per_seq
    block_tables = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        args.batch_size,
        pages_per_seq,
    )
    seq_lens = torch.full(
        (args.batch_size,),
        args.kv_len,
        dtype=torch.uint32,
        device=device,
    )

    k_bf16 = torch.randn(
        total_pages,
        args.page_size,
        args.num_kv_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    ) / 4
    v_bf16 = torch.randn_like(k_bf16) / 4
    kv_cache, kv_cache_sf, k_scale, v_scale = nvfp4_quantize_paged_kv_cache(
        k_bf16,
        v_bf16,
        "NHD",
    )

    results: dict[str, object] = {
        "batch_size": args.batch_size,
        "kv_len": args.kv_len,
        "head_dim": args.head_dim,
        "num_kv_heads": args.num_kv_heads,
        "page_size": args.page_size,
        "groups": {},
    }

    for group in group_sizes:
        num_qo_heads = args.num_kv_heads * group
        q = torch.randn(
            args.batch_size,
            num_qo_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
        )
        out_nvfp4 = torch.empty_like(q)
        out_bf16 = torch.empty_like(q)

        def run_nvfp4() -> None:
            xqa_batch_decode_with_kv_cache(
                q,
                kv_cache,
                workspace,
                block_tables,
                seq_lens,
                max_seq_len=args.kv_len,
                bmm1_scale=k_scale / math.sqrt(args.head_dim),
                bmm2_scale=v_scale,
                window_left=-1,
                out=out_nvfp4,
                kv_layout="NHD",
                kv_cache_sf=kv_cache_sf,
            )

        def run_bf16() -> None:
            xqa_batch_decode_with_kv_cache(
                q,
                (k_bf16, v_bf16),
                workspace,
                block_tables,
                seq_lens,
                max_seq_len=args.kv_len,
                bmm1_scale=1.0 / math.sqrt(args.head_dim),
                bmm2_scale=1.0,
                window_left=-1,
                out=out_bf16,
                kv_layout="NHD",
            )

        nvfp4_samples = (
            _event_ms(run_nvfp4, warmup=args.warmup, repeat=args.repeat)
            if args.only in ("all", "nvfp4")
            else []
        )
        bf16_supported = args.head_dim <= 256
        bf16_samples = (
            _event_ms(run_bf16, warmup=args.warmup, repeat=args.repeat)
            if bf16_supported and args.only in ("all", "bf16")
            else []
        )

        decode_tokens = args.batch_size
        flops = (
            4
            * args.batch_size
            * num_qo_heads
            * args.kv_len
            * args.head_dim
        )
        group_result: dict[str, object] = {
            "num_qo_heads": num_qo_heads,
        }
        if nvfp4_samples:
            group_result["nvfp4_xqa"] = _summary(nvfp4_samples)
            group_result["nvfp4_tok_s_min_ms"] = decode_tokens / (
                min(nvfp4_samples) / 1000
            )
            group_result["nvfp4_tflops_min_ms"] = flops / (
                min(nvfp4_samples) / 1000
            ) / 1e12
        if bf16_samples:
            group_result["bf16_xqa"] = _summary(bf16_samples)
            group_result["bf16_tok_s_min_ms"] = decode_tokens / (
                min(bf16_samples) / 1000
            )
            group_result["bf16_tflops_min_ms"] = flops / (
                min(bf16_samples) / 1000
            ) / 1e12
        if nvfp4_samples and bf16_samples:
            group_result["nvfp4_vs_bf16"] = min(bf16_samples) / min(nvfp4_samples)
        elif not bf16_supported:
            group_result["bf16_xqa"] = "unsupported_for_head_dim"
        results["groups"][str(group)] = group_result

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.only != "all":
        return

    print("\nmarkdown")
    print(
        "| group | NVFP4 XQA min ms | BF16 XQA min ms | "
        "NVFP4 tok/s | BF16 tok/s | NVFP4 vs BF16 |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    for group_id, result in results["groups"].items():
        nvfp4 = result["nvfp4_xqa"]["min_ms"]
        bf16_result = result["bf16_xqa"]
        if isinstance(bf16_result, str):
            print(
                f"| {group_id} | {nvfp4:.4f} | n/a | "
                f"{result['nvfp4_tok_s_min_ms']:.0f} | n/a | n/a |"
            )
            continue
        bf16 = bf16_result["min_ms"]
        print(
            f"| {group_id} | {nvfp4:.4f} | {bf16:.4f} | "
            f"{result['nvfp4_tok_s_min_ms']:.0f} | "
            f"{result['bf16_tok_s_min_ms']:.0f} | {bf16 / nvfp4:.2f}x |"
        )


if __name__ == "__main__":
    main()
