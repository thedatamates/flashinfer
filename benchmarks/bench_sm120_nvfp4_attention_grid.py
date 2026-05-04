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
    q_len: int
    kv_len: int
    head_dim: int = 256
    group: int = 2


DEFAULT_Q_LENS = (128, 256, 512, 1024, 2048, 4096)
DEFAULT_KV_LENS = (128, 512, 1024, 2048, 4096, 8192, 32768, 65536, 131072, 262144)
DEFAULT_GROUPS = (2, 4, 6, 8, 12, 16)

KERNELS = ("sm120_fused", "nvfp4_fa2", "fp8_fa2", "bf16_fa2")


def parse_int_list(value: str, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if value == "all":
        return default
    parsed = tuple(int(part) for part in value.split(",") if part)
    if not parsed:
        raise ValueError("empty integer list")
    return parsed


def parse_cells(value: str) -> tuple[tuple[int, int], ...]:
    cells: list[tuple[int, int]] = []
    for raw_part in value.replace(";", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            q_text, kv_text = part.split(":", 1)
        elif "x" in part:
            q_text, kv_text = part.split("x", 1)
        else:
            raise ValueError(
                "--cells entries must use q:kv or qxkv format, "
                f"got {part!r}"
            )
        q_len = int(q_text)
        kv_len = int(kv_text)
        if q_len <= 0 or kv_len <= 0:
            raise ValueError("--cells q and kv values must be positive")
        cells.append((q_len, kv_len))
    if not cells:
        raise ValueError("--cells did not contain any q:kv pairs")
    return tuple(cells)


def default_output_group_span(head_dim: int) -> int:
    if head_dim == 128:
        return 1
    if head_dim == 256:
        return 2
    if head_dim == 512:
        return 4
    raise ValueError("--head-dim must be one of 128, 256, or 512")


def fused_output_group_span(args: argparse.Namespace) -> int:
    if args.fused_output_group_span == 0:
        return default_output_group_span(args.head_dim)
    if args.fused_output_group_span not in (1, 2, 4):
        raise ValueError("--fused-output-group-span must be 0, 1, 2, or 4")
    return int(args.fused_output_group_span)


def fused_split_kv_len(args: argparse.Namespace) -> int:
    return int(args.fused_split_kv_len)


def build_cells(args: argparse.Namespace) -> list[Cell]:
    groups = parse_int_list(args.groups, default=DEFAULT_GROUPS)
    if args.cells:
        return [
            Cell(q_len=q_len, kv_len=kv_len, head_dim=args.head_dim, group=group)
            for group in groups
            for q_len, kv_len in parse_cells(args.cells)
        ]
    q_lens = parse_int_list(args.q_lens, default=DEFAULT_Q_LENS)
    kv_lens = parse_int_list(args.kv_lens, default=DEFAULT_KV_LENS)
    return [
        Cell(q_len=q_len, kv_len=kv_len, head_dim=args.head_dim, group=group)
        for group in groups
        for q_len in q_lens
        for kv_len in kv_lens
    ]


def run_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda-13.2")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "12.0f")
    env.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
    env["PYTHONPATH"] = (
        f"{root}:{root / 'benchmarks'}"
        + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    )
    return env


def run_json(command: list[str], *, env: dict[str, str], timeout_sec: int) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(command)}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError(f"command produced no stdout: {' '.join(command)}")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        for line in reversed(stdout.splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
        raise


def fused_command(root: Path, cell: Cell, args: argparse.Namespace) -> list[str]:
    output_group_span = fused_output_group_span(args)
    command = [
        sys.executable,
        str(root / "benchmarks" / "bench_sm120_nvfp4_attention.py"),
        "--device",
        str(args.device),
        "--mode",
        "paged-wrapper" if args.fused_api == "paged" else "dense",
        "--q-len",
        str(cell.q_len),
        "--kv-len",
        str(cell.kv_len),
        "--head-dim",
        str(cell.head_dim),
        "--group",
        str(cell.group),
        "--split-kv-len",
        str(fused_split_kv_len(args)),
        "--max-partial-bytes",
        str(args.fused_max_partial_bytes),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--output-group-span",
        str(output_group_span),
        "--sliding-window",
        str(args.sliding_window),
        "--logits-soft-cap",
        str(args.logits_soft_cap),
    ]
    if args.fused_api == "paged":
        command.extend(["--v-layout", args.fused_v_layout])
    if args.causal:
        command.append("--causal")
    return command


def flashinfer_command(
    root: Path,
    cell: Cell,
    args: argparse.Namespace,
    kernel: str,
) -> list[str]:
    only = {
        "nvfp4_fa2": "grouped-fp4",
        "fp8_fa2": "fp8",
        "bf16_fa2": "bf16",
    }[kernel]
    return [
        sys.executable,
        str(root / "benchmarks" / "bench_nvfp4_fmha_v2_gqa_grouped_attention.py"),
        "--device",
        str(args.device),
        "--gemma4-shape",
        "custom",
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
        "fa2",
        "--fp4-v-layout",
        "nhd",
        "--fp4-v-sf-layout",
        "linear",
        "--bf16-backend",
        "fa2",
        "--causal" if args.causal else "--no-causal",
        "--window-left",
        str(args.sliding_window),
        "--logits-soft-cap",
        str(args.logits_soft_cap),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
    ]


def summarize(
    *,
    cell: Cell,
    kernel: str,
    data: dict[str, Any],
    command: list[str],
    fused_output_group_span: int,
) -> dict[str, Any]:
    if kernel == "sm120_fused":
        bench = data["sm120_nvfp4_attention"]
        return {
            "q": cell.q_len,
            "kv": cell.kv_len,
            "d": cell.head_dim,
            "group": cell.group,
            "kernel": kernel,
            "api": data.get("api"),
            "v_layout": data.get("v_layout"),
            "causal": data.get("causal"),
            "sliding_window": data.get("sliding_window"),
            "logits_soft_cap": data.get("logits_soft_cap"),
            "fused_output_group_span": fused_output_group_span,
            "min_ms": bench["min_ms"],
            "mean_ms": bench["mean_ms"],
            "output_finite": data.get("output_finite"),
            "cosine": data.get("splitkv_full_grid_first_tile_vs_exact_cosine"),
            "splits": data.get("splits"),
            "split_kv_len": data.get("split_kv_len"),
            "storage_bytes": data.get("storage_bytes"),
            "command": " ".join(command),
        }
    group_result = data["groups"][str(cell.group)]
    key = {
        "nvfp4_fa2": "fused_fp4_grouped",
        "fp8_fa2": "fp8_kv",
        "bf16_fa2": "bf16_production",
    }[kernel]
    bench = group_result[key]
    return {
        "q": cell.q_len,
        "kv": cell.kv_len,
        "d": cell.head_dim,
        "group": cell.group,
        "kernel": kernel,
        "api": None,
        "v_layout": None,
        "causal": data.get("causal"),
        "sliding_window": data.get("window_left"),
        "logits_soft_cap": data.get("logits_soft_cap"),
        "min_ms": bench["min_ms"],
        "mean_ms": bench["mean_ms"],
        "output_finite": None,
        "cosine": None,
        "splits": None,
        "split_kv_len": None,
        "storage_bytes": None,
        "command": " ".join(command),
    }


def format_optional_float(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def format_optional_ratio(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}x"


def format_optional_percent(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1f}%"


def old_dense_sources(root: Path, head_dim: int) -> list[Path]:
    reports = root / "reports"
    candidates = [
        reports / f"d{head_dim}_focus_baseline_20260430.summary.csv",
        reports / f"d{head_dim}_hillclimb_postopt_20260429.summary.csv",
        reports / f"d{head_dim}_hillclimb_180cell_20260430.summary.csv",
        reports / f"d{head_dim}_group6_hillclimb_20260429.csv",
        reports / f"d{head_dim}_full_context_sm120_fused_dense.csv",
    ]
    return [path for path in candidates if path.exists()]


def load_old_dense_ms(root: Path, head_dim: int) -> dict[tuple[int, int, int], float]:
    old: dict[tuple[int, int, int], float] = {}
    for path in old_dense_sources(root, head_dim):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("kernel") not in (None, "", "sm120_fused"):
                    continue
                row_head_dim = row.get("d") or row.get("head_dim")
                if row_head_dim not in (None, "", str(head_dim)):
                    continue
                value = row.get("fused_ms") or row.get("min_ms")
                if not value:
                    continue
                q_len = row.get("q") or row.get("q_len")
                kv_len = row.get("kv") or row.get("kv_len")
                key = (int(row["group"]), int(q_len), int(kv_len))
                old.setdefault(key, float(value))
    return old


def old_dense_is_comparable(rows: list[dict[str, Any]]) -> bool:
    """Old dense reports were causal, no-SWA, no-softcap exploration runs."""
    fused_rows = [row for row in rows if row.get("kernel") == "sm120_fused"]
    if not fused_rows:
        return False
    for row in fused_rows:
        causal = row.get("causal")
        if causal is False:
            return False
        sliding_window = row.get("sliding_window")
        if sliding_window is not None and int(sliding_window) > 0:
            return False
        logits_soft_cap = row.get("logits_soft_cap")
        if logits_soft_cap is not None and abs(float(logits_soft_cap)) > 0.0:
            return False
    return True


def production_summary_rows(
    rows: list[dict[str, Any]],
    *,
    root: Path,
) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row.get("status", "ok") == "ok"]
    if not ok_rows:
        return []
    head_dim = int(ok_rows[0]["d"])
    old_dense = (
        load_old_dense_ms(root, head_dim)
        if old_dense_is_comparable(ok_rows)
        else {}
    )
    by_cell: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = {}
    for row in ok_rows:
        cell_key = (row["group"], row["q"], row["kv"])
        if row["kernel"] == "sm120_fused":
            if row.get("api") == "dense":
                slot = "dense"
            elif row.get("api") == "paged" and row.get("v_layout") == "pv":
                slot = "paged_pv"
            elif row.get("api") == "paged" and row.get("v_layout") == "linear":
                slot = "paged_linear"
            else:
                slot = f"fused_{row.get('api')}_{row.get('v_layout')}"
        else:
            slot = row["kernel"]
        by_cell.setdefault(cell_key, {})[slot] = row

    summary: list[dict[str, Any]] = []
    for cell_key in sorted(by_cell):
        data = by_cell[cell_key]
        dense_now = data.get("dense", {}).get("min_ms")
        dense_pre = old_dense.get(cell_key)
        paged_pv = data.get("paged_pv", {}).get("min_ms")
        paged_linear = data.get("paged_linear", {}).get("min_ms")
        nvfp4 = data.get("nvfp4_fa2", {}).get("min_ms")
        bf16 = data.get("bf16_fa2", {}).get("min_ms")
        fp8 = data.get("fp8_fa2", {}).get("min_ms")
        regression = (
            float(dense_now) / float(dense_pre)
            if dense_now is not None and dense_pre is not None
            else None
        )
        wrapper_overhead = (
            float(paged_pv) / float(dense_now)
            if paged_pv is not None and dense_now is not None
            else None
        )
        reblock_cost = (
            float(paged_linear) / float(paged_pv)
            if paged_linear is not None and paged_pv is not None
            else None
        )
        reblock_cost_pct = (
            (reblock_cost - 1.0) * 100.0 if reblock_cost is not None else None
        )
        speedup_vs_nvfp4 = (
            float(nvfp4) / float(paged_linear)
            if nvfp4 is not None and paged_linear is not None
            else None
        )
        speedup_vs_bf16 = (
            float(bf16) / float(paged_linear)
            if bf16 is not None and paged_linear is not None
            else None
        )
        summary.append(
            {
                "group": cell_key[0],
                "q": cell_key[1],
                "kv": cell_key[2],
                "dense_now_ms": dense_now,
                "dense_pre_ms": dense_pre,
                "dense_regression": regression,
                "paged_pv_ms": paged_pv,
                "paged_linear_ms": paged_linear,
                "wrapper_overhead_vs_dense": wrapper_overhead,
                "reblock_cost": reblock_cost,
                "reblock_cost_pct": reblock_cost_pct,
                "nvfp4_fa2_ms": nvfp4,
                "fp8_fa2_ms": fp8,
                "bf16_fa2_ms": bf16,
                "paged_linear_speedup_vs_nvfp4": speedup_vs_nvfp4,
                "paged_linear_speedup_vs_bf16": speedup_vs_bf16,
                "paged_linear_finite": data.get("paged_linear", {}).get(
                    "output_finite"
                ),
                "paged_pv_finite": data.get("paged_pv", {}).get("output_finite"),
                "dense_finite": data.get("dense", {}).get("output_finite"),
            }
        )
    return summary


def write_reports(rows: list[dict[str, Any]], *, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = prefix.with_suffix(".jsonl")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    summary_path = prefix.with_name(prefix.name + ".summary.csv")

    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    fieldnames = [
        "q",
        "kv",
        "d",
        "group",
        "kernel",
        "api",
        "v_layout",
        "causal",
        "sliding_window",
        "logits_soft_cap",
        "fused_output_group_span",
        "min_ms",
        "mean_ms",
        "output_finite",
        "cosine",
        "splits",
        "split_kv_len",
        "storage_bytes",
        "command",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_cell: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault((row["group"], row["q"], row["kv"]), {})[
            row["kernel"]
        ] = row

    summary_rows: list[dict[str, Any]] = []
    for cell_key in sorted(by_cell):
        data = by_cell[cell_key]
        fused = data.get("sm120_fused")
        nvfp4 = data.get("nvfp4_fa2")
        if not fused or not nvfp4:
            continue
        speedup = nvfp4["min_ms"] / fused["min_ms"]
        target_ms = nvfp4["min_ms"] / 2.0
        summary_rows.append(
            {
                "group": cell_key[0],
                "q": cell_key[1],
                "kv": cell_key[2],
                "fused_ms": fused["min_ms"],
                "nvfp4_fa2_ms": nvfp4["min_ms"],
                "fp8_fa2_ms": data.get("fp8_fa2", {}).get("min_ms"),
                "bf16_fa2_ms": data.get("bf16_fa2", {}).get("min_ms"),
                "fused_finite": fused.get("output_finite"),
                "fused_cosine": fused["cosine"],
                "fused_vs_nvfp4_fa2_speedup": speedup,
                "target_ms_for_2x": target_ms,
                "gap_to_target_ms": fused["min_ms"] - target_ms,
                "passes_2x": speedup >= 2.0,
                "splits": fused["splits"],
                "split_kv_len": fused["split_kv_len"],
            }
        )

    summary_fieldnames = [
        "group",
        "q",
        "kv",
        "fused_ms",
        "nvfp4_fa2_ms",
        "fp8_fa2_ms",
        "bf16_fa2_ms",
        "fused_finite",
        "fused_cosine",
        "fused_vs_nvfp4_fa2_speedup",
        "target_ms_for_2x",
        "gap_to_target_ms",
        "passes_2x",
        "splits",
        "split_kv_len",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with md_path.open("w") as f:
        f.write("| group | q | kv | fused | nvfp4_fa2 | fp8_fa2 | bf16_fa2 | speedup vs nvfp4 | cosine | pass |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
        for row in summary_rows:
            fp8_ms = row["fp8_fa2_ms"]
            bf16_ms = row["bf16_fa2_ms"]
            f.write(
                f"| {row['group']} | {row['q']} | {row['kv']} | {row['fused_ms']:.6f} | "
                f"{row['nvfp4_fa2_ms']:.6f} | "
                f"{'-' if fp8_ms is None else f'{fp8_ms:.6f}'} | "
                f"{'-' if bf16_ms is None else f'{bf16_ms:.6f}'} | "
                f"{row['fused_vs_nvfp4_fa2_speedup']:.3f}x | "
                f"{format_optional_float(row['fused_cosine'])} | "
                f"{'yes' if row['passes_2x'] else 'no'} |\n"
            )

    print(f"wrote {jsonl_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {md_path}")


def report_paths(prefix: Path) -> tuple[Path, Path, Path, Path]:
    return (
        prefix.with_suffix(".jsonl"),
        prefix.with_suffix(".csv"),
        prefix.with_name(prefix.name + ".summary.csv"),
        prefix.with_suffix(".md"),
    )


def row_fieldnames() -> list[str]:
    return [
        "q",
        "kv",
        "d",
        "group",
        "kernel",
        "api",
        "v_layout",
        "causal",
        "sliding_window",
        "logits_soft_cap",
        "fused_output_group_span",
        "min_ms",
        "mean_ms",
        "output_finite",
        "cosine",
        "splits",
        "split_kv_len",
        "storage_bytes",
        "status",
        "error",
        "command",
    ]


def append_row(row: dict[str, Any], *, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path, csv_path, _, _ = report_paths(prefix)

    with jsonl_path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_fieldnames())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_jsonl_rows(prefix: Path) -> list[dict[str, Any]]:
    jsonl_path, _, _, _ = report_paths(prefix)
    if not jsonl_path.exists():
        return []
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_summary_reports(rows: list[dict[str, Any]], *, prefix: Path) -> None:
    _, _, summary_path, md_path = report_paths(prefix)
    ok_rows = [row for row in rows if row.get("status", "ok") == "ok"]
    root = Path(__file__).resolve().parents[1]
    prod_rows = production_summary_rows(rows, root=root)
    fused_variants = {
        (row.get("api"), row.get("v_layout"))
        for row in ok_rows
        if row.get("kernel") == "sm120_fused"
    }

    baselines_by_cell: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = {}
    fused_rows: list[dict[str, Any]] = []
    for row in ok_rows:
        cell_key = (row["group"], row["q"], row["kv"])
        if row["kernel"] == "sm120_fused":
            fused_rows.append(row)
        else:
            baselines_by_cell.setdefault(cell_key, {})[row["kernel"]] = row

    summary_rows: list[dict[str, Any]] = []
    for fused in sorted(
        fused_rows,
        key=lambda row: (
            row["group"],
            row["q"],
            row["kv"],
            str(row.get("api")),
            str(row.get("v_layout")),
        ),
    ):
        cell_key = (fused["group"], fused["q"], fused["kv"])
        data = baselines_by_cell.get(cell_key, {})
        nvfp4 = data.get("nvfp4_fa2")
        if not nvfp4:
            continue
        speedup = nvfp4["min_ms"] / fused["min_ms"]
        target_ms = nvfp4["min_ms"] / 2.0
        summary_rows.append(
            {
                "group": cell_key[0],
                "q": cell_key[1],
                "kv": cell_key[2],
                "api": fused.get("api"),
                "v_layout": fused.get("v_layout"),
                "fused_ms": fused["min_ms"],
                "nvfp4_fa2_ms": nvfp4["min_ms"],
                "fp8_fa2_ms": data.get("fp8_fa2", {}).get("min_ms"),
                "bf16_fa2_ms": data.get("bf16_fa2", {}).get("min_ms"),
                "fused_finite": fused.get("output_finite"),
                "fused_cosine": fused["cosine"],
                "fused_vs_nvfp4_fa2_speedup": speedup,
                "target_ms_for_2x": target_ms,
                "gap_to_target_ms": fused["min_ms"] - target_ms,
                "passes_2x": speedup >= 2.0,
                "splits": fused["splits"],
                "split_kv_len": fused["split_kv_len"],
            }
        )

    summary_fieldnames = [
        "group",
        "q",
        "kv",
        "api",
        "v_layout",
        "fused_ms",
        "nvfp4_fa2_ms",
        "fp8_fa2_ms",
        "bf16_fa2_ms",
        "fused_finite",
        "fused_cosine",
        "fused_vs_nvfp4_fa2_speedup",
        "target_ms_for_2x",
        "gap_to_target_ms",
        "passes_2x",
        "splits",
        "split_kv_len",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    if len(fused_variants) > 1:
        prod_path = prefix.with_name(prefix.name + ".production.csv")
        prod_fieldnames = [
            "group",
            "q",
            "kv",
            "dense_now_ms",
            "dense_pre_ms",
            "dense_regression",
            "paged_pv_ms",
            "paged_linear_ms",
            "wrapper_overhead_vs_dense",
            "reblock_cost",
            "reblock_cost_pct",
            "nvfp4_fa2_ms",
            "fp8_fa2_ms",
            "bf16_fa2_ms",
            "paged_linear_speedup_vs_nvfp4",
            "paged_linear_speedup_vs_bf16",
            "paged_linear_finite",
            "paged_pv_finite",
            "dense_finite",
        ]
        with prod_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=prod_fieldnames)
            writer.writeheader()
            writer.writerows(prod_rows)

    with md_path.open("w") as f:
        if len(fused_variants) > 1:
            f.write(
                "| q | kv | dense (now) | dense (pre) | regression | "
                "paged-pv | paged-linear | wrapper overhead | reblock cost | "
                "nvfp4_fa2 | fp8_fa2 | bf16_fa2 | speedup vs nvfp4 | "
                "speedup vs bf16 |\n"
            )
            f.write(
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
                "---:|---:|---:|\n"
            )
            for row in prod_rows:
                f.write(
                    f"| {row['q']} | {row['kv']} | "
                    f"{format_optional_float(row['dense_now_ms'])} | "
                    f"{format_optional_float(row['dense_pre_ms'])} | "
                    f"{format_optional_ratio(row['dense_regression'])} | "
                    f"{format_optional_float(row['paged_pv_ms'])} | "
                    f"{format_optional_float(row['paged_linear_ms'])} | "
                    f"{format_optional_ratio(row['wrapper_overhead_vs_dense'])} | "
                    f"{format_optional_ratio(row['reblock_cost'])} "
                    f"({format_optional_percent(row['reblock_cost_pct'])}) | "
                    f"{format_optional_float(row['nvfp4_fa2_ms'])} | "
                    f"{format_optional_float(row['fp8_fa2_ms'])} | "
                    f"{format_optional_float(row['bf16_fa2_ms'])} | "
                    f"{format_optional_ratio(row['paged_linear_speedup_vs_nvfp4'])} | "
                    f"{format_optional_ratio(row['paged_linear_speedup_vs_bf16'])} |\n"
                )

            failures = [row for row in rows if row.get("status") == "error"]
            if failures:
                f.write("\n## Failures\n\n")
                f.write("| group | q | kv | kernel | api | v_layout | error |\n")
                f.write("|---:|---:|---:|---|---|---|---|\n")
                for row in failures:
                    error = str(row.get("error", "")).splitlines()[0][:180]
                    f.write(
                        f"| {row['group']} | {row['q']} | {row['kv']} | "
                        f"{row['kernel']} | {row.get('api')} | "
                        f"{row.get('v_layout')} | {error} |\n"
                    )
            return

        f.write("| group | q | kv | api | v_layout | fused | nvfp4_fa2 | fp8_fa2 | bf16_fa2 | speedup vs nvfp4 | cosine | pass |\n")
        f.write("|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|\n")
        for row in summary_rows:
            fp8_ms = row["fp8_fa2_ms"]
            bf16_ms = row["bf16_fa2_ms"]
            f.write(
                f"| {row['group']} | {row['q']} | {row['kv']} | "
                f"{row['api']} | {row['v_layout']} | {row['fused_ms']:.6f} | "
                f"{row['nvfp4_fa2_ms']:.6f} | "
                f"{'-' if fp8_ms is None else f'{fp8_ms:.6f}'} | "
                f"{'-' if bf16_ms is None else f'{bf16_ms:.6f}'} | "
                f"{row['fused_vs_nvfp4_fa2_speedup']:.3f}x | "
                f"{format_optional_float(row['fused_cosine'])} | "
                f"{'yes' if row['passes_2x'] else 'no'} |\n"
            )

        failures = [row for row in rows if row.get("status") == "error"]
        if failures:
            f.write("\n## Failures\n\n")
            f.write("| group | q | kv | kernel | error |\n")
            f.write("|---:|---:|---:|---|---|\n")
            for row in failures:
                error = str(row.get("error", "")).splitlines()[0][:180]
                f.write(
                    f"| {row['group']} | {row['q']} | {row['kv']} | "
                    f"{row['kernel']} | {error} |\n"
                )


def error_row(
    *,
    cell: Cell,
    kernel: str,
    command: list[str],
    fused_output_group_span: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "q": cell.q_len,
        "kv": cell.kv_len,
        "d": cell.head_dim,
        "group": cell.group,
        "kernel": kernel,
        "api": None,
        "v_layout": None,
        "causal": None,
        "sliding_window": None,
        "logits_soft_cap": None,
        "fused_output_group_span": (
            fused_output_group_span if kernel == "sm120_fused" else None
        ),
        "min_ms": None,
        "mean_ms": None,
        "output_finite": None,
        "cosine": None,
        "splits": None,
        "split_kv_len": None,
        "storage_bytes": None,
        "status": "error",
        "error": str(error),
        "command": " ".join(command),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SM120 NVFP4 attention sweeps across q/kv/group cells."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument(
        "--fused-split-kv-len",
        type=int,
        default=0,
        help=(
            "Split length passed to the SM120 fused benchmark. Use 0 to "
            "follow the production wrapper auto-selection while respecting "
            "--fused-max-partial-bytes."
        ),
    )
    parser.add_argument(
        "--fused-max-partial-bytes",
        type=int,
        default=1 << 30,
        help="Partial/split scratch budget used when --fused-split-kv-len=0.",
    )
    parser.add_argument(
        "--fused-api",
        choices=("paged", "dense"),
        default="paged",
        help="Entry point used for the sm120_fused column.",
    )
    parser.add_argument(
        "--fused-v-layout",
        choices=("linear", "pv"),
        default="linear",
        help=(
            "V cache layout for --fused-api=paged. 'linear' matches "
            "nvfp4_quantize_paged_kv_cache(v_data_layout='linear') and "
            "measures the standard vLLM paged-KV input layout; 'pv' measures "
            "preconverted PV-layout cache."
        ),
    )
    parser.add_argument(
        "--fused-output-group-span",
        type=int,
        default=0,
        help="0 selects the default span for --head-dim: D128=1, D256=2, D512=4.",
    )
    parser.add_argument("--sliding-window", type=int, default=-1)
    parser.add_argument("--logits-soft-cap", type=float, default=50.0)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument(
        "--groups",
        type=str,
        default="all",
        help="all or comma-separated GQA group sizes.",
    )
    parser.add_argument(
        "--cells",
        type=str,
        default="",
        help=(
            "Explicit q:kv cell list, e.g. '1:4096,512:8192'. "
            "When set, --q-lens and --kv-lens are ignored."
        ),
    )
    parser.add_argument(
        "--kernels",
        type=str,
        default=",".join(KERNELS),
        help="Comma-separated subset of sm120_fused,nvfp4_fa2,fp8_fa2,bf16_fa2.",
    )
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    env = run_env(root)
    kernels = tuple(k for k in args.kernels.split(",") if k)
    unknown = sorted(set(kernels) - set(KERNELS))
    if unknown:
        raise ValueError(f"unknown kernels: {unknown}")
    output_group_span = fused_output_group_span(args)
    cells = build_cells(args)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else root / "reports" / f"d{args.head_dim}_sm120_nvfp4_attention_grid_{stamp}"
    )
    rows: list[dict[str, Any]] = load_jsonl_rows(prefix)
    existing = {
        (
            row.get("group"),
            row.get("q"),
            row.get("kv"),
            row.get("d"),
            row.get("kernel"),
            row.get("api"),
            row.get("v_layout"),
        )
        for row in rows
    }
    for cell in cells:
        for kernel in kernels:
            row_key = (
                cell.group,
                cell.q_len,
                cell.kv_len,
                cell.head_dim,
                kernel,
                args.fused_api if kernel == "sm120_fused" else None,
                (
                    args.fused_v_layout
                    if kernel == "sm120_fused" and args.fused_api == "paged"
                    else ("pv" if kernel == "sm120_fused" else None)
                ),
            )
            if row_key in existing:
                print(
                    json.dumps(
                        {
                            "status": "skipped_existing",
                            "group": cell.group,
                            "q": cell.q_len,
                            "kv": cell.kv_len,
                            "d": cell.head_dim,
                            "kernel": kernel,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            command = (
                fused_command(root, cell, args)
                if kernel == "sm120_fused"
                else flashinfer_command(root, cell, args, kernel)
            )
            try:
                data = run_json(command, env=env, timeout_sec=args.timeout_sec)
                row = summarize(
                    cell=cell,
                    kernel=kernel,
                    data=data,
                    command=command,
                    fused_output_group_span=output_group_span,
                )
                row["status"] = "ok"
                row["error"] = None
            except Exception as exc:
                row = error_row(
                    cell=cell,
                    kernel=kernel,
                    command=command,
                    fused_output_group_span=output_group_span,
                    error=exc,
                )
            rows.append(row)
            existing.add(row_key)
            append_row(row, prefix=prefix)
            write_summary_reports(rows, prefix=prefix)
            print(json.dumps(row, sort_keys=True), flush=True)
    write_summary_reports(rows, prefix=prefix)
    print(f"wrote {report_paths(prefix)[0]}")
    print(f"wrote {report_paths(prefix)[1]}")
    print(f"wrote {report_paths(prefix)[2]}")
    print(f"wrote {report_paths(prefix)[3]}")


if __name__ == "__main__":
    main()
