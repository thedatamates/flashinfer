from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from flashinfer.decode import xqa_batch_decode_with_kv_cache


def _event_ms(fn, *, warmup: int, repeat: int) -> list[float]:
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
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if args.head_dim != 512:
        raise ValueError("This benchmark is intentionally scoped to D512.")
    if args.num_qo_heads % args.num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads.")

    torch.manual_seed(0)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    num_pages_per_seq = math.ceil(args.kv_len / args.page_size)
    num_pages = args.batch_size * num_pages_per_seq
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=device).view(
        args.batch_size, num_pages_per_seq
    )
    seq_lens = torch.full(
        (args.batch_size,), args.kv_len, dtype=torch.uint32, device=device
    )

    packed_dim = args.head_dim // 2
    sf_dim = args.head_dim // 16
    q = torch.randn(
        args.batch_size,
        args.num_qo_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    k_cache = torch.randint(
        0,
        256,
        (num_pages, args.page_size, args.num_kv_heads, packed_dim),
        dtype=torch.uint8,
        device=device,
    )
    v_cache = torch.randint(
        0,
        256,
        (num_pages, args.page_size, args.num_kv_heads, packed_dim),
        dtype=torch.uint8,
        device=device,
    )
    k_sf = torch.ones(
        (num_pages, args.page_size, args.num_kv_heads, sf_dim),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    v_sf = torch.ones_like(k_sf)
    out = torch.empty_like(q)

    def run() -> None:
        xqa_batch_decode_with_kv_cache(
            q,
            (k_cache, v_cache),
            workspace,
            block_tables,
            seq_lens,
            max_seq_len=args.kv_len,
            bmm1_scale=1.0 / math.sqrt(args.head_dim),
            bmm2_scale=1.0,
            window_left=-1,
            out=out,
            kv_layout="NHD",
            kv_cache_sf=(k_sf, v_sf),
        )

    samples = _event_ms(run, warmup=args.warmup, repeat=args.repeat)
    decode_tokens = args.batch_size
    flops = (
        4
        * args.batch_size
        * args.num_qo_heads
        * args.kv_len
        * args.head_dim
    )
    result = {
        "batch_size": args.batch_size,
        "decode_tokens": decode_tokens,
        "head_dim": args.head_dim,
        "kv_len": args.kv_len,
        "num_kv_heads": args.num_kv_heads,
        "num_pages": num_pages,
        "num_qo_heads": args.num_qo_heads,
        "page_size": args.page_size,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "tok_s_mean": decode_tokens / (statistics.mean(samples) / 1000),
        "tok_s_min_ms": decode_tokens / (min(samples) / 1000),
        "tflops_mean": flops / (statistics.mean(samples) / 1000) / 1e12,
        "tflops_min_ms": flops / (min(samples) / 1000) / 1e12,
        "samples_ms": samples,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
