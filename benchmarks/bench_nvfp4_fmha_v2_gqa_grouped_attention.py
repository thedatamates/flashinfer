from __future__ import annotations

import argparse
import json
import statistics

import torch

import flashinfer
from flashinfer.fp4_quantization import fp4_quantize, nvfp4_quantize_paged_kv_cache


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


def _error_summary(exc: Exception) -> str:
    first_line = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {first_line}"


def _to_float8(
    x: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, float]:
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scaled = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scaled.to(dtype), scale.float().reciprocal().item()


def _quantize_v_pv_layout_nhd(
    v_cache: torch.Tensor,
    v_global_sf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_pages, page_size, num_kv_heads, head_dim = v_cache.shape
    if page_size != 16:
        raise ValueError("PV-layout NVFP4 V cache benchmark currently requires page_size=16.")

    v_by_col = (
        v_cache.permute(0, 2, 3, 1)
        .contiguous()
        .reshape(num_pages * num_kv_heads * head_dim, page_size)
    )
    packed_col_token, sf_col = fp4_quantize(
        v_by_col,
        v_global_sf,
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
    )

    packed_col_token = packed_col_token.view(torch.uint8).reshape(
        num_pages, num_kv_heads, head_dim, page_size // 2
    )
    nibbles = torch.empty(
        (num_pages, num_kv_heads, head_dim, page_size),
        device=v_cache.device,
        dtype=torch.uint8,
    )
    nibbles[..., 0::2] = packed_col_token & 0x0F
    nibbles[..., 1::2] = (packed_col_token >> 4) & 0x0F

    nibbles_by_row = nibbles.permute(0, 3, 1, 2).contiguous()
    v_packed = (
        nibbles_by_row[..., 0::2] | (nibbles_by_row[..., 1::2] << 4)
    ).contiguous()

    scale_dim = head_dim // 16
    sf_col = sf_col.view(torch.uint8).reshape(num_pages, num_kv_heads, head_dim)
    v_sf = torch.empty(
        (num_pages, page_size, num_kv_heads, scale_dim),
        device=v_cache.device,
        dtype=torch.uint8,
    )
    for col in range(head_dim):
        v_sf[:, col // scale_dim, :, col % scale_dim] = sf_col[:, :, col]

    return v_packed, v_sf.view(torch.float8_e4m3fn)


def _make_plan_tensors(
    *,
    batch_size: int,
    q_len: int,
    kv_len: int,
    page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_seq
    qo_indptr = torch.arange(
        0,
        (batch_size + 1) * q_len,
        q_len,
        dtype=torch.int32,
        device="cpu",
    )
    paged_kv_indptr = torch.arange(
        0,
        (batch_size + 1) * pages_per_seq,
        pages_per_seq,
        dtype=torch.int32,
        device="cpu",
    )
    paged_kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    last_page_len = (kv_len - 1) % page_size + 1
    paged_kv_last_page_len = torch.full(
        (batch_size,),
        last_page_len,
        dtype=torch.int32,
        device="cpu",
    )
    block_tables = paged_kv_indices.view(batch_size, pages_per_seq)
    return (
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        block_tables,
    )


def _make_wrapper(
    *,
    backend: str = "fmha_v2",
    workspace_mib: int,
    q_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
    o_data_type: torch.dtype,
    device: torch.device,
    plan_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    fixed_split_size: int | None,
    disable_split_kv: bool,
) -> flashinfer.BatchPrefillWithPagedKVCacheWrapper:
    qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, block_tables = (
        plan_tensors
    )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(workspace_mib * 1024 * 1024, dtype=torch.uint8, device=device),
        "NHD",
        backend=backend,
    )
    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        o_data_type=o_data_type,
        block_tables=block_tables,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
    )
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fused FMHAv2 GQA table for NVFP4 paged-KV prefill attention. "
            "Defaults to Gemma4 31B global-attention shape."
        )
    )
    parser.add_argument(
        "--gemma4-shape",
        type=str,
        default="global",
        choices=["global", "sliding", "custom"],
        help=(
            "Preset production Gemma4 31B shape. global: D512/group8/long KV; "
            "sliding: D256/group2/kv1024. Explicit shape flags override preset values."
        ),
    )
    parser.add_argument("--q-len", type=int, default=None)
    parser.add_argument("--kv-len", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-group-size", type=int, default=8)
    parser.add_argument(
        "--group-sizes",
        type=str,
        default=None,
        help=(
            "Comma-separated group sizes. Defaults to Gemma4 preset group, or powers "
            "of two up to max-group-size for custom."
        ),
    )
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Query/output and BF16 baseline dtype. Production Gemma4 uses bf16.",
    )
    parser.add_argument(
        "--fp4-v-layout",
        type=str,
        default="pv",
        choices=["pv", "nhd"],
        help="Use the production PV-layout V cache or the original NHD-reblocked V cache.",
    )
    parser.add_argument(
        "--fp4-v-sf-layout",
        type=str,
        default="trtllm_interleaved",
        choices=["trtllm_interleaved", "linear"],
        help="Physical layout for NHD V scale factors. Linear is only valid with --fp4-backend=fa2.",
    )
    parser.add_argument(
        "--fp4-backend",
        type=str,
        default="fmha_v2",
        choices=["fa2", "fmha_v2", "trtllm-gen"],
        help="Backend used for FP4 grouped/separate runs. FP8 baseline always uses FA2.",
    )
    parser.add_argument(
        "--bf16-backend",
        type=str,
        default="fmha_v2",
        choices=["fa2", "fmha_v2"],
        help="Backend used for the BF16 grouped baseline.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "separate-fp4", "grouped-fp4", "fp8", "bf16"],
        help="Run one benchmark target. Useful for Nsight kernel filtering.",
    )
    parser.add_argument(
        "--fixed-split-size",
        type=int,
        default=None,
        help="Forward fixed_split_size to FlashInfer prefill planning.",
    )
    parser.add_argument(
        "--disable-split-kv",
        action="store_true",
        help="Forward disable_split_kv=True to FlashInfer prefill planning.",
    )
    args = parser.parse_args()

    presets = {
        "global": {
            "q_len": 512,
            "kv_len": 32768,
            "head_dim": 512,
            "group_sizes": "8",
        },
        "sliding": {
            "q_len": 512,
            "kv_len": 1024,
            "head_dim": 256,
            "group_sizes": "2",
        },
        "custom": {
            "q_len": 512,
            "kv_len": 8192,
            "head_dim": 512,
            "group_sizes": "",
        },
    }
    preset = presets[args.gemma4_shape]
    args.q_len = args.q_len if args.q_len is not None else preset["q_len"]
    args.kv_len = args.kv_len if args.kv_len is not None else preset["kv_len"]
    args.head_dim = (
        args.head_dim if args.head_dim is not None else preset["head_dim"]
    )
    if args.group_sizes is None:
        args.group_sizes = preset["group_sizes"]

    if args.head_dim not in (128, 256, 512):
        raise ValueError("head_dim must be one of 128, 256, or 512.")
    if args.group_sizes:
        group_sizes = [int(group) for group in args.group_sizes.split(",")]
    else:
        if args.max_group_size < 1 or args.max_group_size & (args.max_group_size - 1):
            raise ValueError("max_group_size must be a power of two.")
        group_sizes = []
        group = 1
        while group <= args.max_group_size:
            group_sizes.append(group)
            group *= 2
    if any(group < 1 for group in group_sizes):
        raise ValueError("group sizes must be positive.")

    torch.manual_seed(0)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    pages_per_seq = (args.kv_len + args.page_size - 1) // args.page_size
    total_pages = args.batch_size * pages_per_seq
    plan_tensors = _make_plan_tensors(
        batch_size=args.batch_size,
        q_len=args.q_len,
        kv_len=args.kv_len,
        page_size=args.page_size,
        device=device,
    )

    k = torch.randn(
        total_pages,
        args.page_size,
        1,
        args.head_dim,
        dtype=dtype,
        device=device,
    ) / 4
    v = torch.randn_like(k) / 4
    if args.fp4_v_sf_layout == "linear" and args.fp4_backend != "fa2":
        raise ValueError("--fp4-v-sf-layout=linear is only supported with --fp4-backend=fa2")
    (k_fp4, v_reblocked), (k_sf, v_reblocked_sf), k_scale, v_scale = (
        nvfp4_quantize_paged_kv_cache(
            k, v, "NHD", v_scale_layout=args.fp4_v_sf_layout
        )
    )
    v_fp4_pv, v_sf_pv = _quantize_v_pv_layout_nhd(
        v,
        torch.tensor([1.0 / v_scale], device=device, dtype=torch.float32),
    )
    if args.fp4_v_layout == "pv":
        if args.fp4_backend != "fmha_v2":
            raise ValueError("--fp4-v-layout=pv is only supported with --fp4-backend=fmha_v2")
        v_fp4 = v_fp4_pv
        v_sf = v_sf_pv
        nvfp4_v_cache_uses_pv_layout = True
    else:
        v_fp4 = v_reblocked
        v_sf = v_reblocked_sf
        nvfp4_v_cache_uses_pv_layout = False

    k_bf16 = k.contiguous()
    v_bf16 = v.contiguous()
    k_fp8, k_fp8_scale = _to_float8(k)
    v_fp8, v_fp8_scale = _to_float8(v)

    results: dict[str, object] = {
        "shape": args.gemma4_shape,
        "batch_size": args.batch_size,
        "q_len": args.q_len,
        "kv_len": args.kv_len,
        "head_dim": args.head_dim,
        "page_size": args.page_size,
        "dtype": args.dtype,
        "fp8_backend": "fa2",
        "bf16_backend": args.bf16_backend,
        "fp4_backend": args.fp4_backend,
        "fp4_v_layout": args.fp4_v_layout,
        "fp4_v_sf_layout": args.fp4_v_sf_layout,
        "groups": {},
    }

    for group in group_sizes:
        need_grouped_fp4 = args.only in ("all", "grouped-fp4")
        need_separate_fp4 = args.only in ("all", "separate-fp4")
        need_fp8 = args.only in ("all", "fp8")
        need_bf16 = args.only in ("all", "bf16")
        q_grouped = torch.randn(
            args.batch_size * args.q_len,
            group,
            args.head_dim,
            dtype=dtype,
            device=device,
        )
        out_grouped_fp4 = torch.empty_like(q_grouped)
        out_grouped_fp8 = torch.empty_like(q_grouped)
        out_grouped_bf16 = torch.empty_like(q_grouped)
        wrapper_grouped_fp4 = (
            _make_wrapper(
                backend=args.fp4_backend,
                workspace_mib=args.workspace_mib,
                q_len=args.q_len,
                kv_len=args.kv_len,
                num_qo_heads=group,
                num_kv_heads=1,
                head_dim=args.head_dim,
                page_size=args.page_size,
                q_data_type=dtype,
                kv_data_type=torch.uint8,
                o_data_type=dtype,
                device=device,
                plan_tensors=plan_tensors,
                fixed_split_size=args.fixed_split_size,
                disable_split_kv=args.disable_split_kv,
            )
            if need_grouped_fp4
            else None
        )
        wrapper_grouped_fp8 = (
            _make_wrapper(
                backend="fa2",
                workspace_mib=args.workspace_mib,
                q_len=args.q_len,
                kv_len=args.kv_len,
                num_qo_heads=group,
                num_kv_heads=1,
                head_dim=args.head_dim,
                page_size=args.page_size,
                q_data_type=dtype,
                kv_data_type=torch.float8_e4m3fn,
                o_data_type=dtype,
                device=device,
                plan_tensors=plan_tensors,
                fixed_split_size=args.fixed_split_size,
                disable_split_kv=args.disable_split_kv,
            )
            if need_fp8
            else None
        )
        wrapper_grouped_bf16 = (
            _make_wrapper(
                backend=args.bf16_backend,
                workspace_mib=args.workspace_mib,
                q_len=args.q_len,
                kv_len=args.kv_len,
                num_qo_heads=group,
                num_kv_heads=1,
                head_dim=args.head_dim,
                page_size=args.page_size,
                q_data_type=dtype,
                kv_data_type=dtype,
                o_data_type=dtype,
                device=device,
                plan_tensors=plan_tensors,
                fixed_split_size=args.fixed_split_size,
                disable_split_kv=args.disable_split_kv,
            )
            if need_bf16
            else None
        )

        separate_wrappers = []
        separate_inputs = []
        if need_separate_fp4:
            for _ in range(group):
                q = torch.randn(
                    args.batch_size * args.q_len,
                    1,
                    args.head_dim,
                    dtype=dtype,
                    device=device,
                )
                out = torch.empty_like(q)
                separate_wrappers.append(
                    _make_wrapper(
                        backend=args.fp4_backend,
                        workspace_mib=args.workspace_mib,
                        q_len=args.q_len,
                        kv_len=args.kv_len,
                        num_qo_heads=1,
                        num_kv_heads=1,
                        head_dim=args.head_dim,
                        page_size=args.page_size,
                        q_data_type=dtype,
                        kv_data_type=torch.uint8,
                        o_data_type=dtype,
                        device=device,
                        plan_tensors=plan_tensors,
                        fixed_split_size=args.fixed_split_size,
                        disable_split_kv=args.disable_split_kv,
                    )
                )
                separate_inputs.append((q, out))

        def run_grouped_fp4() -> None:
            assert wrapper_grouped_fp4 is not None
            wrapper_grouped_fp4.run(
                q_grouped,
                (k_fp4, v_fp4),
                out=out_grouped_fp4,
                kv_cache_sf=(k_sf, v_sf),
                k_scale=k_scale,
                v_scale=v_scale,
                nvfp4_v_cache_uses_pv_layout=nvfp4_v_cache_uses_pv_layout,
                nvfp4_v_cache_sf_layout=args.fp4_v_sf_layout,
            )

        def run_separate_fp4() -> None:
            for wrapper, (q, out) in zip(separate_wrappers, separate_inputs):
                wrapper.run(
                    q,
                    (k_fp4, v_fp4),
                    out=out,
                    kv_cache_sf=(k_sf, v_sf),
                    k_scale=k_scale,
                        v_scale=v_scale,
                        nvfp4_v_cache_uses_pv_layout=nvfp4_v_cache_uses_pv_layout,
                        nvfp4_v_cache_sf_layout=args.fp4_v_sf_layout,
                    )

        def run_grouped_bf16() -> None:
            assert wrapper_grouped_bf16 is not None
            wrapper_grouped_bf16.run(
                q_grouped,
                (k_bf16, v_bf16),
                out=out_grouped_bf16,
            )

        def run_grouped_fp8() -> None:
            assert wrapper_grouped_fp8 is not None
            wrapper_grouped_fp8.run(
                q_grouped,
                (k_fp8, v_fp8),
                out=out_grouped_fp8,
                k_scale=k_fp8_scale,
                v_scale=v_fp8_scale,
            )

        grouped_fp4_samples = (
            _event_ms(run_grouped_fp4, warmup=args.warmup, repeat=args.repeat)
            if need_grouped_fp4
            else []
        )
        separate_fp4_samples = (
            _event_ms(run_separate_fp4, warmup=args.warmup, repeat=args.repeat)
            if need_separate_fp4
            else []
        )
        grouped_fp8_samples = []
        grouped_fp8_error = None
        if need_fp8:
            try:
                grouped_fp8_samples = _event_ms(
                    run_grouped_fp8,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            except Exception as exc:
                grouped_fp8_error = _error_summary(exc)
        grouped_bf16_samples = (
            _event_ms(run_grouped_bf16, warmup=args.warmup, repeat=args.repeat)
            if need_bf16
            else []
        )

        flops = (
            4
            * args.batch_size
            * group
            * args.q_len
            * args.kv_len
            * args.head_dim
        )
        group_result = {}
        if separate_fp4_samples:
            group_result["fused_fp4_separate"] = _summary(separate_fp4_samples)
            group_result["fp4_separate_tflops_min_ms"] = (
                flops / (min(separate_fp4_samples) / 1000) / 1e12
            )
        if grouped_fp4_samples:
            group_result["fused_fp4_grouped"] = _summary(grouped_fp4_samples)
            group_result["fp4_grouped_tflops_min_ms"] = (
                flops / (min(grouped_fp4_samples) / 1000) / 1e12
            )
        if grouped_fp8_samples:
            group_result["fp8_kv"] = _summary(grouped_fp8_samples)
            group_result["fp8_tflops_min_ms"] = (
                flops / (min(grouped_fp8_samples) / 1000) / 1e12
            )
        if grouped_fp8_error is not None:
            group_result["fp8_kv_error"] = grouped_fp8_error
        if grouped_bf16_samples:
            group_result["bf16_production"] = _summary(grouped_bf16_samples)
            group_result["bf16_tflops_min_ms"] = (
                flops / (min(grouped_bf16_samples) / 1000) / 1e12
            )
        if separate_fp4_samples and grouped_fp4_samples:
            group_result["fp4_grouped_vs_separate"] = (
                min(separate_fp4_samples) / min(grouped_fp4_samples)
            )
        if grouped_bf16_samples and grouped_fp4_samples:
            group_result["fp4_grouped_vs_bf16"] = (
                min(grouped_bf16_samples) / min(grouped_fp4_samples)
            )
        if grouped_fp8_samples and grouped_fp4_samples:
            group_result["fp4_grouped_vs_fp8"] = (
                min(grouped_fp8_samples) / min(grouped_fp4_samples)
            )
        if grouped_bf16_samples and grouped_fp8_samples:
            group_result["fp8_vs_bf16"] = (
                min(grouped_bf16_samples) / min(grouped_fp8_samples)
            )
        results["groups"][str(group)] = group_result

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.only != "all":
        return
    print("\nmarkdown")
    print(
        "| shape | group | q_len | kv_len | FP4 KV min ms | FP8 KV min ms | "
        "BF16 min ms | FP4 TFLOP/s | FP8 TFLOP/s | BF16 TFLOP/s | "
        "FP4 vs FP8 | FP4 vs BF16 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for group_id, result in results["groups"].items():
        grouped = result["fused_fp4_grouped"]["min_ms"]
        bf16 = result["bf16_production"]["min_ms"]
        grouped_tflops = result["fp4_grouped_tflops_min_ms"]
        bf16_tflops = result["bf16_tflops_min_ms"]
        if "fp8_kv" in result:
            fp8 = result["fp8_kv"]["min_ms"]
            fp8_ms = f"{fp8:.4f}"
            fp8_tflops = f"{result['fp8_tflops_min_ms']:.2f}"
            fp8_ratio = f"{fp8 / grouped:.2f}x"
        else:
            fp8_ms = "ERR"
            fp8_tflops = "ERR"
            fp8_ratio = "ERR"
        print(
            f"| {args.gemma4_shape} | {group_id} | {args.q_len} | {args.kv_len} | "
            f"{grouped:.4f} | {fp8_ms} | {bf16:.4f} | "
            f"{grouped_tflops:.2f} | {fp8_tflops} | {bf16_tflops:.2f} | "
            f"{fp8_ratio} | {bf16 / grouped:.2f}x |"
        )


if __name__ == "__main__":
    main()
