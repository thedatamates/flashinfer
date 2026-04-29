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
DEFAULT_KV_LENS = (8192, 32768, 65536, 131072, 262144)
DEFAULT_GROUPS = (2, 4, 6, 8, 12, 16)

KERNELS = ("sm120_fused", "nvfp4_fa2", "fp8_fa2", "bf16_fa2")


def parse_int_list(value: str, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if value == "all":
        return default
    parsed = tuple(int(part) for part in value.split(",") if part)
    if not parsed:
        raise ValueError("empty integer list")
    return parsed


def build_cells(args: argparse.Namespace) -> list[Cell]:
    q_lens = parse_int_list(args.q_lens, default=DEFAULT_Q_LENS)
    kv_lens = parse_int_list(args.kv_lens, default=DEFAULT_KV_LENS)
    groups = parse_int_list(args.groups, default=DEFAULT_GROUPS)
    return [
        Cell(q_len=q_len, kv_len=kv_len, group=group)
        for group in groups
        for q_len in q_lens
        for kv_len in kv_lens
    ]


def run_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda-13.2")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "12.0f")
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
    return [
        sys.executable,
        str(root / "benchmarks" / "bench_sm120_nvfp4_cutlass_fused_attention.py"),
        "--device",
        str(args.device),
        "--q-len",
        str(cell.q_len),
        "--kv-len",
        str(cell.kv_len),
        "--head-dim",
        str(cell.head_dim),
        "--group",
        str(cell.group),
        "--split-kv-len",
        str(args.fused_split_kv_len),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--sm120-qkv-online-splitkv-full-grid-bench",
    ]


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
) -> dict[str, Any]:
    if kernel == "sm120_fused":
        bench = data["bench_sm120_qkv_online_register_q_splitkv_full_grid"]
        return {
            "q": cell.q_len,
            "kv": cell.kv_len,
            "d": cell.head_dim,
            "group": cell.group,
            "kernel": kernel,
            "min_ms": bench["min_ms"],
            "mean_ms": bench["mean_ms"],
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
        "min_ms": bench["min_ms"],
        "mean_ms": bench["mean_ms"],
        "cosine": None,
        "splits": None,
        "split_kv_len": None,
        "storage_bytes": None,
        "command": " ".join(command),
    }


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
        "min_ms",
        "mean_ms",
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
                "fused_cosine": fused["cosine"],
                "fused_vs_nvfp4_fa2_speedup": speedup,
                "target_ms_for_2x": target_ms,
                "gap_to_target_ms": fused["min_ms"] - target_ms,
                "passes_2x": speedup >= 2.0,
                "splits": fused["splits"],
                "split_kv_len": fused["split_kv_len"],
            }
        )

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
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
                f"{row['fused_cosine']:.6f} | "
                f"{'yes' if row['passes_2x'] else 'no'} |\n"
            )

    print(f"wrote {jsonl_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--fused-split-kv-len", type=int, default=6656)
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
    cells = build_cells(args)

    rows: list[dict[str, Any]] = []
    for cell in cells:
        for kernel in kernels:
            command = (
                fused_command(root, cell, args)
                if kernel == "sm120_fused"
                else flashinfer_command(root, cell, args, kernel)
            )
            data = run_json(command, env=env, timeout_sec=args.timeout_sec)
            row = summarize(cell=cell, kernel=kernel, data=data, command=command)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else root / "reports" / f"d256_hillclimb_{stamp}"
    )
    write_reports(rows, prefix=prefix)


if __name__ == "__main__":
    main()
