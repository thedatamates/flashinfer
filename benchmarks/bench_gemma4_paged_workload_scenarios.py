from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from typing import Callable

import torch

import flashinfer
from flashinfer.decode import xqa_batch_decode_with_kv_cache
from flashinfer.fp4_quantization import nvfp4_quantize_paged_kv_cache


@dataclass(frozen=True)
class Scenario:
    name: str
    mode: str
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]


def _event_ms(fn: Callable[[], None], *, warmup: int, repeat: int) -> list[float]:
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


def _scenario(name: str) -> Scenario:
    if name in ("few_long_decode", "global_few_long_decode"):
        return Scenario(
            name=name,
            mode="decode",
            q_lens=(1, 1, 1, 1),
            kv_lens=(262144, 262144, 262144, 262144),
        )
    if name in ("high_concurrency_mixed", "global_high_concurrency_mixed"):
        kv_lens = (
            (1024,) * 24
            + (8192,) * 24
            + (32768,) * 24
            + (131072,) * 16
            + (262144,) * 8
        )
        q_lens = (1,) * 92 + (512,) * 4
        return Scenario(
            name=name,
            mode="prefill",
            q_lens=q_lens,
            kv_lens=kv_lens,
        )
    if name == "global_high_concurrency_decode":
        kv_lens = (
            (1024,) * 24
            + (8192,) * 24
            + (32768,) * 24
            + (131072,) * 16
            + (262144,) * 8
        )
        return Scenario(
            name=name,
            mode="decode",
            q_lens=(1,) * len(kv_lens),
            kv_lens=kv_lens,
        )
    if name == "sliding_high_concurrency_decode":
        return Scenario(
            name=name,
            mode="decode",
            q_lens=(1,) * 96,
            kv_lens=(1024,) * 96,
        )
    if name == "sliding_few_long_decode":
        return Scenario(
            name=name,
            mode="decode",
            q_lens=(1, 1, 1, 1),
            kv_lens=(1024, 1024, 1024, 1024),
        )
    if name == "sliding_high_concurrency_mixed":
        return Scenario(
            name=name,
            mode="prefill",
            q_lens=(1,) * 92 + (512,) * 4,
            kv_lens=(1024,) * 96,
        )
    raise ValueError(f"unknown scenario: {name}")


def _page_plan(
    kv_lens: tuple[int, ...],
    *,
    page_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    pages_per_seq = tuple(math.ceil(kv_len / page_size) for kv_len in kv_lens)
    indptr = [0]
    for pages in pages_per_seq:
        indptr.append(indptr[-1] + pages)
    total_pages = indptr[-1]
    indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    last_page_len = torch.tensor(
        [(kv_len - 1) % page_size + 1 for kv_len in kv_lens],
        dtype=torch.int32,
        device="cpu",
    )
    max_pages = max(pages_per_seq)
    block_tables = torch.zeros(
        (len(kv_lens), max_pages),
        dtype=torch.int32,
        device=device,
    )
    for row, (start, stop) in enumerate(zip(indptr[:-1], indptr[1:])):
        block_tables[row, : stop - start] = indices[start:stop]
    return {
        "indptr_cpu": torch.tensor(indptr, dtype=torch.int32, device="cpu"),
        "indices": indices,
        "last_page_len_cpu": last_page_len,
        "block_tables": block_tables,
        "seq_lens": torch.tensor(kv_lens, dtype=torch.uint32, device=device),
        "seq_lens_cpu_i32": torch.tensor(kv_lens, dtype=torch.int32, device="cpu"),
        "total_pages": total_pages,
        "max_kv_len": max(kv_lens),
    }


def _qo_indptr(q_lens: tuple[int, ...]) -> torch.Tensor:
    indptr = [0]
    for q_len in q_lens:
        indptr.append(indptr[-1] + q_len)
    return torch.tensor(indptr, dtype=torch.int32, device="cpu")


def _make_kv(
    *,
    total_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    ) / 4
    v = torch.randn_like(k) / 4
    return k, v


def _bench_decode(
    scenario: Scenario,
    *,
    targets: list[str],
    page_size: int,
    group: int,
    head_dim: int,
    workspace_mib: int,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> dict[str, object]:
    plan = _page_plan(scenario.kv_lens, page_size=page_size, device=device)
    num_kv_heads = 1
    num_qo_heads = group
    dtype = torch.bfloat16
    k_bf16, v_bf16 = _make_kv(
        total_pages=int(plan["total_pages"]),
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    q = torch.randn(
        len(scenario.kv_lens),
        num_qo_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    workspace = torch.empty(workspace_mib * 1024 * 1024, dtype=torch.uint8, device=device)
    out = torch.empty_like(q)
    results: dict[str, object] = {}

    if "nvfp4_xqa" in targets:
        kv_cache, kv_cache_sf, k_scale, v_scale = nvfp4_quantize_paged_kv_cache(
            k_bf16,
            v_bf16,
            "NHD",
        )

        def run_nvfp4() -> None:
            xqa_batch_decode_with_kv_cache(
                q,
                kv_cache,
                workspace,
                plan["block_tables"],
                plan["seq_lens"],
                max_seq_len=int(plan["max_kv_len"]),
                bmm1_scale=k_scale / math.sqrt(head_dim),
                bmm2_scale=v_scale,
                out=out,
                kv_layout="NHD",
                kv_cache_sf=kv_cache_sf,
            )

        try:
            results["nvfp4_xqa"] = _summary(
                _event_ms(run_nvfp4, warmup=warmup, repeat=repeat)
            )
        except Exception as exc:  # noqa: BLE001
            results["nvfp4_xqa_error"] = _error_summary(exc)

    if "nvfp4_fa2_decode" in targets:
        kv_cache, kv_cache_sf, k_scale, v_scale = nvfp4_quantize_paged_kv_cache(
            k_bf16,
            v_bf16,
            "NHD",
        )
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(
                workspace_mib * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            ),
            "NHD",
            use_tensor_cores=True,
            backend="fa2",
        )
        wrapper.plan(
            plan["indptr_cpu"],
            plan["indices"],
            plan["last_page_len_cpu"],
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=dtype,
            kv_data_type=torch.uint8,
            o_data_type=dtype,
            block_tables=plan["block_tables"],
            seq_lens=plan["seq_lens_cpu_i32"],
        )

        def run_nvfp4_fa2() -> None:
            wrapper.run(
                q,
                kv_cache,
                out=out,
                k_scale=k_scale,
                v_scale=v_scale,
                kv_cache_sf=kv_cache_sf,
            )

        try:
            results["nvfp4_fa2_decode"] = _summary(
                _event_ms(run_nvfp4_fa2, warmup=warmup, repeat=repeat)
            )
        except Exception as exc:  # noqa: BLE001
            results["nvfp4_fa2_decode_error"] = _error_summary(exc)

    if "fp8_fa2_decode" in targets:
        k_fp8, k_scale = _to_float8(k_bf16)
        v_fp8, v_scale = _to_float8(v_bf16)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(
                workspace_mib * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            ),
            "NHD",
            use_tensor_cores=True,
            backend="fa2",
        )
        wrapper.plan(
            plan["indptr_cpu"],
            plan["indices"],
            plan["last_page_len_cpu"],
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=dtype,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=dtype,
            block_tables=plan["block_tables"],
            seq_lens=plan["seq_lens_cpu_i32"],
        )

        def run_fp8() -> None:
            wrapper.run(
                q,
                (k_fp8, v_fp8),
                out=out,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        try:
            results["fp8_fa2_decode"] = _summary(
                _event_ms(run_fp8, warmup=warmup, repeat=repeat)
            )
        except Exception as exc:  # noqa: BLE001
            results["fp8_fa2_decode_error"] = _error_summary(exc)

    if "bf16_fa2_decode" in targets:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(
                workspace_mib * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            ),
            "NHD",
            use_tensor_cores=True,
            backend="fa2",
        )
        wrapper.plan(
            plan["indptr_cpu"],
            plan["indices"],
            plan["last_page_len_cpu"],
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            block_tables=plan["block_tables"],
            seq_lens=plan["seq_lens_cpu_i32"],
        )

        def run_bf16() -> None:
            wrapper.run(
                q,
                (k_bf16, v_bf16),
                out=out,
            )

        try:
            results["bf16_fa2_decode"] = _summary(
                _event_ms(run_bf16, warmup=warmup, repeat=repeat)
            )
        except Exception as exc:  # noqa: BLE001
            results["bf16_fa2_decode_error"] = _error_summary(exc)

    return results


def _bench_prefill(
    scenario: Scenario,
    *,
    targets: list[str],
    page_size: int,
    group: int,
    head_dim: int,
    workspace_mib: int,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> dict[str, object]:
    plan = _page_plan(scenario.kv_lens, page_size=page_size, device=device)
    num_kv_heads = 1
    num_qo_heads = group
    dtype = torch.bfloat16
    k_bf16, v_bf16 = _make_kv(
        total_pages=int(plan["total_pages"]),
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    q = torch.randn(
        sum(scenario.q_lens),
        num_qo_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    qo_indptr = _qo_indptr(scenario.q_lens)
    results: dict[str, object] = {}

    def make_wrapper(kv_data_type: torch.dtype) -> flashinfer.BatchPrefillWithPagedKVCacheWrapper:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(workspace_mib * 1024 * 1024, dtype=torch.uint8, device=device),
            "NHD",
            backend="fa2",
        )
        wrapper.plan(
            qo_indptr,
            plan["indptr_cpu"],
            plan["indices"],
            plan["last_page_len_cpu"],
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=dtype,
            kv_data_type=kv_data_type,
            o_data_type=dtype,
            block_tables=plan["block_tables"],
            seq_lens=plan["seq_lens_cpu_i32"],
        )
        return wrapper

    if "nvfp4_fa2" in targets:
        (k_fp4, v_fp4), (k_sf, v_sf), k_scale, v_scale = nvfp4_quantize_paged_kv_cache(
            k_bf16,
            v_bf16,
            "NHD",
            v_scale_layout="linear",
        )
        wrapper = make_wrapper(torch.uint8)
        out = torch.empty_like(q)

        def run_nvfp4() -> None:
            wrapper.run(
                q,
                (k_fp4, v_fp4),
                out=out,
                kv_cache_sf=(k_sf, v_sf),
                k_scale=k_scale,
                v_scale=v_scale,
                nvfp4_v_cache_uses_pv_layout=False,
                nvfp4_v_cache_sf_layout="linear",
            )

        try:
            results["nvfp4_fa2_prefill"] = _summary(
                _event_ms(run_nvfp4, warmup=warmup, repeat=repeat)
            )
        except Exception as exc:  # noqa: BLE001
            results["nvfp4_fa2_prefill_error"] = _error_summary(exc)

    if "fp8_fa2" in targets:
        k_fp8, k_scale = _to_float8(k_bf16)
        v_fp8, v_scale = _to_float8(v_bf16)
        wrapper = make_wrapper(torch.float8_e4m3fn)
        out = torch.empty_like(q)

        def run_fp8() -> None:
            wrapper.run(
                q,
                (k_fp8, v_fp8),
                out=out,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        try:
            results["fp8_fa2_prefill"] = _summary(
                _event_ms(run_fp8, warmup=warmup, repeat=repeat)
            )
        except Exception as exc:  # noqa: BLE001
            results["fp8_fa2_prefill_error"] = _error_summary(exc)

    if "bf16_fa2" in targets:
        wrapper = make_wrapper(dtype)
        out = torch.empty_like(q)

        def run_bf16() -> None:
            wrapper.run(
                q,
                (k_bf16, v_bf16),
                out=out,
            )

        try:
            results["bf16_fa2_prefill"] = _summary(
                _event_ms(run_bf16, warmup=warmup, repeat=repeat)
            )
        except Exception as exc:  # noqa: BLE001
            results["bf16_fa2_prefill_error"] = _error_summary(exc)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=[
            "few_long_decode",
            "high_concurrency_mixed",
            "global_few_long_decode",
            "global_high_concurrency_decode",
            "global_high_concurrency_mixed",
            "sliding_few_long_decode",
            "sliding_high_concurrency_decode",
            "sliding_high_concurrency_mixed",
        ],
        required=True,
    )
    parser.add_argument("--targets", default="all")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    scenario = _scenario(args.scenario)
    if args.targets == "all":
        targets = (
            [
                "nvfp4_xqa",
                "nvfp4_fa2_decode",
                "fp8_fa2_decode",
                "bf16_fa2_decode",
            ]
            if scenario.mode == "decode"
            else ["nvfp4_fa2", "fp8_fa2", "bf16_fa2"]
        )
    else:
        targets = [target for target in args.targets.split(",") if target]

    torch.manual_seed(0)
    device = torch.device(f"cuda:{args.device}")
    if scenario.mode == "decode":
        results = _bench_decode(
            scenario,
            targets=targets,
            page_size=args.page_size,
            group=args.group,
            head_dim=args.head_dim,
            workspace_mib=args.workspace_mib,
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
        )
    else:
        results = _bench_prefill(
            scenario,
            targets=targets,
            page_size=args.page_size,
            group=args.group,
            head_dim=args.head_dim,
            workspace_mib=args.workspace_mib,
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
        )

    payload = {
        "scenario": scenario.name,
        "mode": scenario.mode,
        "batch_size": len(scenario.kv_lens),
        "sum_q": sum(scenario.q_lens),
        "min_q": min(scenario.q_lens),
        "max_q": max(scenario.q_lens),
        "min_kv": min(scenario.kv_lens),
        "max_kv": max(scenario.kv_lens),
        "sum_kv": sum(scenario.kv_lens),
        "head_dim": args.head_dim,
        "group": args.group,
        "page_size": args.page_size,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
