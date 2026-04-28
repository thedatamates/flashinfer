from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

import flashinfer


def _make_indptr(lengths: torch.Tensor, page_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_or_kv_indptr = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=lengths.device)
    blocks = (lengths + page_size - 1) // page_size
    q_or_kv_indptr[1:] = torch.cumsum(blocks, dim=0)
    last_page_len = ((lengths - 1) % page_size + 1).to(torch.int32)
    return q_or_kv_indptr, blocks, last_page_len


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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--q-len", type=int, default=512)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Set both QO and KV heads. Kept for quick equal-head benchmarks.",
    )
    parser.add_argument("--num-qo-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--backend", type=str, default="fa2", choices=["auto", "fa2", "fmha_v2"])
    parser.add_argument("--kv-dtype", type=str, default="nvfp4", choices=["nvfp4", "fp8", "bf16"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument(
        "--nvfp4-v-pv-layout",
        action="store_true",
        help="Treat the V cache and V scales as already stored in FMHAv2 PV layout.",
    )
    parser.add_argument(
        "--no-stripe",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.head_dim != 512:
        raise ValueError("This benchmark is intentionally scoped to D512.")
    if args.num_heads is not None:
        args.num_qo_heads = args.num_heads
        args.num_kv_heads = args.num_heads
    if args.num_qo_heads % args.num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads.")
    if args.nvfp4_v_pv_layout and (args.backend != "fmha_v2" or args.kv_dtype != "nvfp4"):
        raise ValueError("--nvfp4-v-pv-layout requires --backend fmha_v2 --kv-dtype nvfp4.")

    torch.manual_seed(0)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16
    workspace = torch.empty(args.workspace_mib * 1024 * 1024, dtype=torch.uint8, device=device)

    q_lens = torch.full((args.batch_size,), args.q_len, dtype=torch.int32, device=device)
    kv_lens = torch.full((args.batch_size,), args.kv_len, dtype=torch.int32, device=device)
    qo_indptr = torch.zeros(args.batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens, dim=0)
    paged_kv_indptr, blocks_per_seq, last_page_len = _make_indptr(kv_lens, args.page_size)
    num_pages = int(paged_kv_indptr[-1].item())
    page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)

    q = torch.randn(
        int(qo_indptr[-1].item()),
        args.num_qo_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    if args.kv_dtype == "nvfp4":
        packed_dim = args.head_dim // 2
        sf_dim = args.head_dim // 16
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
        kv_data_type = torch.uint8
        kv_cache_sf = (k_sf, v_sf)
    elif args.kv_dtype == "fp8":
        k_cache = torch.randn(
            num_pages,
            args.page_size,
            args.num_kv_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
        ).to(torch.float8_e4m3fn)
        v_cache = torch.randn(
            num_pages,
            args.page_size,
            args.num_kv_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
        ).to(torch.float8_e4m3fn)
        kv_data_type = torch.float8_e4m3fn
        kv_cache_sf = None
    else:
        k_cache = torch.randn(
            num_pages,
            args.page_size,
            args.num_kv_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
        )
        v_cache = torch.randn_like(k_cache)
        kv_data_type = dtype
        kv_cache_sf = None
    out = torch.empty_like(q)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        backend=args.backend,
    )
    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        page_indices,
        last_page_len,
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
        args.page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=kv_data_type,
        o_data_type=dtype,
    )
    def run() -> None:
        if kv_cache_sf is None:
            if args.kv_dtype == "fp8":
                wrapper.run(q, (k_cache, v_cache), out=out, k_scale=1.0, v_scale=1.0)
            else:
                wrapper.run(q, (k_cache, v_cache), out=out)
        else:
            wrapper.run(
                q,
                (k_cache, v_cache),
                out=out,
                kv_cache_sf=kv_cache_sf,
                k_scale=1.0,
                v_scale=1.0,
                nvfp4_v_cache_uses_pv_layout=args.nvfp4_v_pv_layout,
            )

    samples = _event_ms(run, warmup=args.warmup, repeat=args.repeat)
    total_q = args.batch_size * args.q_len
    flops = (
        4
        * args.batch_size
        * args.num_qo_heads
        * args.q_len
        * args.kv_len
        * args.head_dim
    )
    result = {
        "backend": wrapper._backend,
        "batch_size": args.batch_size,
        "d512_stripe": False,
        "q_len": args.q_len,
        "kv_len": args.kv_len,
        "kv_dtype": args.kv_dtype,
        "nvfp4_v_pv_layout": args.nvfp4_v_pv_layout,
        "num_kv_heads": args.num_kv_heads,
        "num_qo_heads": args.num_qo_heads,
        "head_dim": args.head_dim,
        "page_size": args.page_size,
        "num_pages": num_pages,
        "tokens": total_q,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "tflops_mean": flops / (statistics.mean(samples) / 1000) / 1e12,
        "tflops_min_ms": flops / (min(samples) / 1000) / 1e12,
        "samples_ms": samples,
        "sm_scale": 1.0 / math.sqrt(args.head_dim),
        "blocks_per_seq": int(blocks_per_seq[0].item()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
