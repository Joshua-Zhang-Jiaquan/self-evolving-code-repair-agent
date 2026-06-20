#!/usr/bin/env python3
"""Generate Markdown report tables from a summary JSON file.

Usage:
    python scripts/generate_report_tables.py --summary outputs/summary.json --out-dir report/tables
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate report tables from summary JSON")
    _ = parser.add_argument("--summary", required=True, help="Path to summary JSON")
    _ = parser.add_argument("--out-dir", required=True, help="Output directory for table files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary_path = Path(cast(str, args.summary))
    out_dir = Path(cast(str, args.out_dir))
    if not summary_path.is_file():
        print(f"summary file not found: {summary_path}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    data = cast(dict[str, object], json.loads(summary_path.read_text(encoding="utf-8")))
    runs: list[dict[str, object]] = cast(list[dict[str, object]], data["runs"])

    _write_results_table(runs, out_dir / "results_table.md")
    _write_ablation_table(runs, out_dir / "ablation_comparison.md")
    _write_device_table(runs, out_dir / "device_utilization.md")
    return 0


def _write_results_table(runs: list[dict[str, object]], path: Path) -> None:
    lines = ["| Run ID | Type | Predictions | Resolved | Pass@1 | Empty Patch Rate | Official Harness Status |"]
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for run in _sorted_runs(runs):
        run_id = str(run.get("run_id", ""))
        run_type = str(run.get("run_type", ""))
        preds = _int(run.get("prediction_rows"))
        resolved = _int(run.get("resolved"))
        pass1 = _fmt_float(run.get("pass_at_1"))
        empty_rate = _fmt_float(run.get("empty_patch_rate"))
        harness = cast(dict[str, object], run.get("official_harness", {}))
        harness_status = str(harness.get("status", "N/A"))
        lines.append(f"| {run_id} | {run_type} | {preds} | {resolved} | {pass1} | {empty_rate} | {harness_status} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ablation_table(runs: list[dict[str, object]], path: Path) -> None:
    lines = ["| Run ID | Type | Predictions | Resolved | Pass@1 | Empty Patch Rate | Official Harness Status |"]
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    run_map = {str(r.get("run_id", "")): r for r in runs}
    baseline_ids = ["baseline_main", "feedback_main", "learning_main"]
    ablation_ids = ["ablation_no_process_reward", "ablation_no_feedback_features", "ablation_reduced_test_budget"]
    for run_id in baseline_ids:
        run = run_map.get(run_id)
        if run is None:
            continue
        run_type = str(run.get("run_type", ""))
        preds = _int(run.get("prediction_rows"))
        resolved = _int(run.get("resolved"))
        pass1 = _fmt_float(run.get("pass_at_1"))
        empty_rate = _fmt_float(run.get("empty_patch_rate"))
        harness = cast(dict[str, object], run.get("official_harness", {}))
        harness_status = str(harness.get("status", "N/A"))
        lines.append(f"| {run_id} | {run_type} | {preds} | {resolved} | {pass1} | {empty_rate} | {harness_status} |")
    for run_id in ablation_ids:
        run = run_map.get(run_id)
        if run is None:
            continue
        run_type = str(run.get("run_type", ""))
        preds = _int(run.get("prediction_rows"))
        resolved = _int(run.get("resolved"))
        pass1 = _fmt_float(run.get("pass_at_1"))
        empty_rate = _fmt_float(run.get("empty_patch_rate"))
        harness = cast(dict[str, object], run.get("official_harness", {}))
        harness_status = str(harness.get("status", "N/A"))
        lines.append(f"| {run_id} | {run_type} | {preds} | {resolved} | {pass1} | {empty_rate} | {harness_status} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_device_table(runs: list[dict[str, object]], path: Path) -> None:
    lines = ["| Run ID | Device IDs | Utilization (avg %) | GPU Memory Peak (MB) | Fallback Reasons |"]
    lines.append("|---|---:|---:|---|")
    for run in _sorted_runs(runs):
        run_id = str(run.get("run_id", ""))
        du = cast(dict[str, dict[str, object]], run.get("device_utilization", {}))
        gm = cast(dict[str, dict[str, object]], run.get("gpu_memory_peak", {}))
        fr = cast(dict[str, object], run.get("fallback_reasons", {}))
        device_ids = ", ".join(sorted(du.keys())) if du else "none"
        util_parts: list[str] = []
        for dev_id in sorted(du.keys(), key=lambda x: int(x)):
            dev = du[dev_id]
            avg = dev.get("average_utilization_percent")
            if isinstance(avg, (int, float)):
                util_parts.append(f"GPU{dev_id}: {avg:.1f}%")
            else:
                util_parts.append(f"GPU{dev_id}: not_reported")
        util_str = ", ".join(util_parts) if util_parts else "not_reported"
        mem_parts: list[str] = []
        for dev_id in sorted(gm.keys(), key=lambda x: int(x)):
            dev = gm[dev_id]
            peak = dev.get("memory_peak_mb")
            if isinstance(peak, (int, float)):
                mem_parts.append(f"GPU{dev_id}: {peak}")
            else:
                mem_parts.append(f"GPU{dev_id}: not_reported")
        mem_str = ", ".join(mem_parts) if mem_parts else "not_reported"
        if fr:
            fr_str = ", ".join(f"{k}: {v}" for k, v in sorted((str(k), v) for k, v in fr.items()))
        else:
            fr_str = "none"
        lines.append(f"| {run_id} | {device_ids} | {util_str} | {mem_str} | {fr_str} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sorted_runs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    def _key(run: dict[str, object]) -> tuple[int, str]:
        run_id = str(run.get("run_id", ""))
        main_abl = {"baseline_main", "feedback_main", "learning_main",
                    "ablation_no_process_reward", "ablation_no_feedback_features",
                    "ablation_reduced_test_budget"}
        if run_id in main_abl:
            return (0, run_id)
        if "smoke" in run_id:
            return (1, run_id)
        return (2, run_id)
    return sorted(runs, key=_key)


def _int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _fmt_float(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
