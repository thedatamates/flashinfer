from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Cell:
    shape: str
    use: str
    layer_count: int
    q_len: int
    kv_len: int
    head_dim: int
    group: int


KERNEL_ORDER = (
    "sm120_fused",
    "cutlass_two_stage",
    "flashinfer_nvfp4_fmha_v2",
    "flashinfer_nvfp4_fa2",
    "flashinfer_fp8_fa2",
    "flashinfer_bf16",
)


def default_cells() -> list[Cell]:
    cells: list[Cell] = []
    for q_len in (512, 2048):
        cells.append(
            Cell(
                shape="A",
                use="sliding",
                layer_count=50,
                q_len=q_len,
                kv_len=1024,
                head_dim=256,
                group=2,
            )
        )
    for q_len in (512, 2048):
        for kv_len in (8192, 32768, 131072, 262144):
            cells.append(
                Cell(
                    shape="B",
                    use="global",
                    layer_count=10,
                    q_len=q_len,
                    kv_len=kv_len,
                    head_dim=512,
                    group=8,
                )
            )
    return cells


def selected_cells(args: argparse.Namespace) -> list[Cell]:
    cells = default_cells()
    if args.shapes != "all":
        allowed = set(args.shapes.split(","))
        cells = [cell for cell in cells if cell.shape in allowed]
    if args.q_lens != "all":
        allowed = {int(value) for value in args.q_lens.split(",")}
        cells = [cell for cell in cells if cell.q_len in allowed]
    if args.kv_lens != "all":
        allowed = {int(value) for value in args.kv_lens.split(",")}
        cells = [cell for cell in cells if cell.kv_len in allowed]
    return cells


def report_stem(args: argparse.Namespace) -> Path:
    if args.output_prefix:
        return Path(args.output_prefix)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("reports") / f"gemma4_attention_grid_{stamp}"


def run_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(root / "benchmarks"), str(root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("CUDA_HOME", "/usr/local/cuda-13.2")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "12.0f")
    if Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6").exists():
        env.setdefault(
            "LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        )
    return env


def decode_first_json(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stdout[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in benchmark stdout")


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_sec: int,
    dry_run: bool,
) -> tuple[str, dict[str, Any] | None, str | None, str]:
    command_text = " ".join(cmd)
    if dry_run:
        return "dry_run", None, None, command_text
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return "timeout", None, f"timeout after {timeout_sec}s", command_text

    if proc.returncode != 0:
        stderr = proc.stderr.strip().splitlines()
        stdout = proc.stdout.strip().splitlines()
        detail = stderr[-1] if stderr else (stdout[-1] if stdout else "")
        return "error", None, detail, command_text
    try:
        return "ok", decode_first_json(proc.stdout), None, command_text
    except Exception as exc:  # noqa: BLE001
        return "parse_error", None, f"{type(exc).__name__}: {exc}", command_text


def unsupported(reason: str) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "reason": reason,
        "min_ms": None,
        "mean_ms": None,
        "tflops_min": None,
        "command": "",
    }


def summarize_result(
    *,
    status: str,
    data: dict[str, Any] | None,
    error: str | None,
    command: str,
    kernel: str,
    cell: Cell,
) -> dict[str, Any]:
    if status != "ok" or data is None:
        return {
            "status": status,
            "reason": error or "",
            "min_ms": None,
            "mean_ms": None,
            "tflops_min": None,
            "command": command,
        }

    if kernel == "sm120_fused":
        key = (
            "bench_sm120_qkv_online_register_q_splitkv_full_grid"
            if cell.head_dim == 256
            else "bench_sm120_qkv_online_register_q_splitkv_reuse4_full_grid"
        )
        bench = data[key]
        return {
            "status": "ok",
            "reason": "",
            "min_ms": bench["min_ms"],
            "mean_ms": bench["mean_ms"],
            "tflops_min": None,
            "command": command,
        }

    if kernel == "cutlass_two_stage":
        bench = data["two_stage_fp4_fused_softmax_quant"]
        return {
            "status": "ok",
            "reason": "",
            "min_ms": bench["min_ms"],
            "mean_ms": bench["mean_ms"],
            "tflops_min": None,
            "command": command,
        }

    group_result = data["groups"][str(cell.group)]
    if kernel in ("flashinfer_nvfp4_fmha_v2", "flashinfer_nvfp4_fa2"):
        bench = group_result["fused_fp4_grouped"]
        return {
            "status": "ok",
            "reason": "",
            "min_ms": bench["min_ms"],
            "mean_ms": bench["mean_ms"],
            "tflops_min": group_result.get("fp4_grouped_tflops_min_ms"),
            "command": command,
        }
    if kernel == "flashinfer_fp8_fa2":
        if "fp8_kv_error" in group_result:
            return {
                "status": "error",
                "reason": group_result["fp8_kv_error"],
                "min_ms": None,
                "mean_ms": None,
                "tflops_min": None,
                "command": command,
            }
        bench = group_result["fp8_kv"]
        return {
            "status": "ok",
            "reason": "",
            "min_ms": bench["min_ms"],
            "mean_ms": bench["mean_ms"],
            "tflops_min": group_result.get("fp8_tflops_min_ms"),
            "command": command,
        }
    if kernel == "flashinfer_bf16":
        bench = group_result["bf16_production"]
        return {
            "status": "ok",
            "reason": "",
            "min_ms": bench["min_ms"],
            "mean_ms": bench["mean_ms"],
            "tflops_min": group_result.get("bf16_tflops_min_ms"),
            "command": command,
        }
    raise ValueError(f"unknown kernel: {kernel}")


def command_for_kernel(
    *,
    kernel: str,
    cell: Cell,
    args: argparse.Namespace,
    root: Path,
) -> list[str] | dict[str, Any]:
    py = sys.executable
    common = [
        "--device",
        str(args.device),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
    ]
    if kernel == "sm120_fused":
        if cell.head_dim not in (128, 256, 512):
            return unsupported("current fused kernel supports D128/D256/D512 only")
        if (cell.q_len * cell.group) % 128 != 0:
            return unsupported("current fused kernel requires q_len * group multiple of 128")
        if cell.kv_len % 128 != 0:
            return unsupported("current fused kernel requires kv_len multiple of 128")
        span_flag = (
            "--sm120-qkv-online-splitkv-full-grid-bench"
            if cell.head_dim == 256
            else "--sm120-qkv-online-splitkv-reuse4-full-grid-bench"
        )
        split_kv_len = args.sm120_fused_split_kv_len
        if cell.shape == "A" and cell.head_dim == 256 and cell.kv_len == 1024:
            split_kv_len = 128 if cell.q_len <= 512 else 512
        return [
            py,
            str(root / "benchmarks" / "bench_sm120_nvfp4_cutlass_fused_attention.py"),
            *common,
            "--q-len",
            str(cell.q_len),
            "--kv-len",
            str(cell.kv_len),
            "--head-dim",
            str(cell.head_dim),
            "--group",
            str(cell.group),
            span_flag,
            "--split-kv-len",
            str(split_kv_len),
        ]

    if kernel == "cutlass_two_stage":
        return [
            py,
            str(root / "benchmarks" / "bench_nvfp4_native_attention_gemm.py"),
            "--device",
            str(args.device),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--m",
            str(cell.q_len * cell.group),
            "--n",
            str(cell.kv_len),
            "--d",
            str(cell.head_dim),
            "--softmax-quant-threads",
            "0",
        ]

    if kernel in (
        "flashinfer_nvfp4_fmha_v2",
        "flashinfer_nvfp4_fa2",
        "flashinfer_fp8_fa2",
        "flashinfer_bf16",
    ):
        only = {
            "flashinfer_nvfp4_fmha_v2": "grouped-fp4",
            "flashinfer_nvfp4_fa2": "grouped-fp4",
            "flashinfer_fp8_fa2": "fp8",
            "flashinfer_bf16": "bf16",
        }[kernel]
        fp4_backend = "fa2" if kernel == "flashinfer_nvfp4_fa2" else "fmha_v2"
        fp4_v_layout = "nhd" if kernel == "flashinfer_nvfp4_fa2" else "pv"
        fp4_v_sf_layout = (
            "linear" if kernel == "flashinfer_nvfp4_fa2" else "trtllm_interleaved"
        )
        gemma_shape = "sliding" if cell.shape == "A" else "global"
        return [
            py,
            str(root / "benchmarks" / "bench_nvfp4_fmha_v2_gqa_grouped_attention.py"),
            *common,
            "--gemma4-shape",
            gemma_shape,
            "--q-len",
            str(cell.q_len),
            "--kv-len",
            str(cell.kv_len),
            "--head-dim",
            str(cell.head_dim),
            "--group-sizes",
            str(cell.group),
            "--batch-size",
            "1",
            "--only",
            only,
            "--fp4-backend",
            fp4_backend,
            "--fp4-v-layout",
            fp4_v_layout,
            "--fp4-v-sf-layout",
            fp4_v_sf_layout,
        ]

    raise ValueError(f"unknown kernel: {kernel}")


def format_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def build_rows(
    cells: list[Cell],
    kernels: list[str],
    results: dict[tuple[Cell, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for cell in cells:
        best_kernel = None
        best_min = None
        for kernel in kernels:
            result = results[(cell, kernel)]
            detail_rows.append(
                {
                    "shape": cell.shape,
                    "use": cell.use,
                    "layers": cell.layer_count,
                    "q_len": cell.q_len,
                    "kv_len": cell.kv_len,
                    "head_dim": cell.head_dim,
                    "group": cell.group,
                    "kernel": kernel,
                    "status": result["status"],
                    "min_ms": result["min_ms"],
                    "mean_ms": result["mean_ms"],
                    "tflops_min": result["tflops_min"],
                    "reason": result["reason"],
                    "command": result["command"],
                }
            )
            if result["status"] != "ok" or result["min_ms"] is None:
                continue
            if best_min is None or result["min_ms"] < best_min:
                best_min = result["min_ms"]
                best_kernel = kernel
        best_rows.append(
            {
                "shape": cell.shape,
                "use": cell.use,
                "layers": cell.layer_count,
                "q_len": cell.q_len,
                "kv_len": cell.kv_len,
                "head_dim": cell.head_dim,
                "group": cell.group,
                "best_kernel": best_kernel or "",
                "best_min_ms": best_min,
                "weighted_layer_pass_min_ms": (
                    None if best_min is None else best_min * cell.layer_count
                ),
            }
        )
    return detail_rows, best_rows


def write_reports(
    *,
    stem: Path,
    cells: list[Cell],
    kernels: list[str],
    results: dict[tuple[Cell, str], dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    with stem.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)
    with stem.with_suffix(".best.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
        writer.writeheader()
        writer.writerows(best_rows)

    json_payload = {
        "kernels": kernels,
        "cells": [cell.__dict__ for cell in cells],
        "details": detail_rows,
        "best": best_rows,
        "weighted_total_min_ms_over_grid": sum(
            row["weighted_layer_pass_min_ms"] or 0.0 for row in best_rows
        ),
    }
    stem.with_suffix(".json").write_text(json.dumps(json_payload, indent=2))

    lines = [
        "# Gemma4 31B Attention Grid",
        "",
        "| shape | use | q | kv | d | group | layers | "
        + " | ".join(kernels)
        + " | best | weighted best ms |",
        "|---|---|---:|---:|---:|---:|---:|"
        + "|".join(["---:"] * len(kernels))
        + "|---|---:|",
    ]
    for cell, best in zip(cells, best_rows):
        values = []
        for kernel in kernels:
            result = results[(cell, kernel)]
            if result["status"] == "ok":
                values.append(format_ms(result["min_ms"]))
            elif result["status"] == "unsupported":
                values.append("-")
            else:
                values.append("ERR")
        lines.append(
            f"| {cell.shape} | {cell.use} | {cell.q_len} | {cell.kv_len} | "
            f"{cell.head_dim} | {cell.group} | {cell.layer_count} | "
            + " | ".join(values)
            + f" | {best['best_kernel'] or '-'} | "
            f"{format_ms(best['weighted_layer_pass_min_ms'])} |"
        )
    total = sum(row["weighted_layer_pass_min_ms"] or 0.0 for row in best_rows)
    lines.extend(
        [
            "",
            f"Weighted total over listed grid cells: `{total:.4f} ms`.",
            "",
            "Unsupported cells are marked `-`; failed runs are marked `ERR`.",
        ]
    )
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemma4 31B production attention prefill grid benchmarks."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument(
        "--kernels",
        type=str,
        default=",".join(KERNEL_ORDER),
        help=f"Comma-separated kernels. Valid: {','.join(KERNEL_ORDER)}",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="all",
        help="all, A, B, or comma-separated subset.",
    )
    parser.add_argument(
        "--q-lens",
        type=str,
        default="all",
        help="all or comma-separated q lengths.",
    )
    parser.add_argument(
        "--kv-lens",
        type=str,
        default="all",
        help="all or comma-separated KV lengths.",
    )
    parser.add_argument("--sm120-fused-split-kv-len", type=int, default=6656)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    kernels = [kernel for kernel in args.kernels.split(",") if kernel]
    unknown = sorted(set(kernels) - set(KERNEL_ORDER))
    if unknown:
        raise ValueError(f"unknown kernels: {unknown}")
    cells = selected_cells(args)
    if not cells:
        raise ValueError("no cells selected")

    env = run_env(root)
    results: dict[tuple[Cell, str], dict[str, Any]] = {}
    for cell in cells:
        for kernel in kernels:
            command_or_result = command_for_kernel(
                kernel=kernel, cell=cell, args=args, root=root
            )
            if isinstance(command_or_result, dict):
                results[(cell, kernel)] = command_or_result
                continue
            status, data, error, command = run_command(
                command_or_result,
                cwd=root,
                env=env,
                timeout_sec=args.timeout_sec,
                dry_run=args.dry_run,
            )
            results[(cell, kernel)] = summarize_result(
                status=status,
                data=data,
                error=error,
                command=command,
                kernel=kernel,
                cell=cell,
            )
            print(
                json.dumps(
                    {
                        "shape": cell.shape,
                        "q_len": cell.q_len,
                        "kv_len": cell.kv_len,
                        "head_dim": cell.head_dim,
                        "group": cell.group,
                        "kernel": kernel,
                        "status": results[(cell, kernel)]["status"],
                        "min_ms": results[(cell, kernel)]["min_ms"],
                        "reason": results[(cell, kernel)]["reason"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    detail_rows, best_rows = build_rows(cells, kernels, results)
    stem = report_stem(args)
    write_reports(
        stem=stem,
        cells=cells,
        kernels=kernels,
        results=results,
        detail_rows=detail_rows,
        best_rows=best_rows,
    )
    print(f"wrote {stem.with_suffix('.md')}")
    print(f"wrote {stem.with_suffix('.csv')}")
    print(f"wrote {stem.with_suffix('.best.csv')}")
    print(f"wrote {stem.with_suffix('.json')}")


if __name__ == "__main__":
    main()
