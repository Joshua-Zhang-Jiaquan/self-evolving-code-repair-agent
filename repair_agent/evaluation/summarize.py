from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import yaml

from repair_agent.evaluation.metrics import (
    aggregate_run_metrics,
    summarize_model_gates,
    summarize_official_harness,
    summarize_resources,
)
from repair_agent.logging import JsonMap, read_json_object, read_jsonl, write_json_atomic


EXPECTED_RUN_TYPES = ("baseline", "feedback", "learning")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize repair_agent run directories")
    _ = parser.add_argument("--runs", required=True, help="Directory containing run subdirectories")
    _ = parser.add_argument("--out", required=True, help="Summary JSON output path")
    _ = parser.add_argument("--include-resources", action="store_true", help="Include resource_usage.jsonl summaries")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_arg = cast(str, args.runs)
    out_arg = cast(str, args.out)
    include_resources = cast(bool, args.include_resources)
    try:
        summary = summarize_runs(Path(runs_arg), include_resources=include_resources)
        write_json_atomic(Path(out_arg), summary)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"summary error: {exc}", file=sys.stderr)
        return 2
    return 0


def summarize_runs(runs_root: Path, *, include_resources: bool = False) -> JsonMap:
    if not runs_root.is_dir():
        raise ValueError(f"runs directory not found: {runs_root}")
    run_rows: list[JsonMap] = []
    resource_rows: list[JsonMap] = []
    harness_reports: list[JsonMap] = []
    for run_dir in _run_directories(runs_root):
        row = summarize_run_dir(run_dir, runs_root=runs_root, include_resources=include_resources)
        run_rows.append(row)
        harness = row.get("official_harness")
        if isinstance(harness, dict):
            harness_map = cast(dict[object, object], harness)
            harness_reports.append({str(key): item for key, item in harness_map.items()})
        if include_resources:
            resource = row.get("resources")
            if isinstance(resource, dict):
                resource_rows.extend(read_jsonl(run_dir / "resource_usage.jsonl"))
    aggregate = aggregate_summary(run_rows)
    output: JsonMap = {
        "runs": run_rows,
        "aggregate": aggregate,
        "missing_runs": missing_run_types(run_rows),
    }
    gate_dir = runs_root.parent / "model_gates"
    gates = summarize_model_gates(sorted(gate_dir.glob("*.json")) if gate_dir.is_dir() else [])
    if gates.get("models"):
        output["model_gates"] = gates
    global_harness = _load_global_harness(runs_root)
    if global_harness or harness_reports:
        output["harness"] = {
            "global": summarize_official_harness(global_harness) if global_harness else {},
            "runs": harness_reports,
        }
    if include_resources:
        output["resources"] = summarize_resources(resource_rows)
    return output


def summarize_run_dir(run_dir: Path, *, runs_root: Path, include_resources: bool) -> JsonMap:
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    trajectories = read_jsonl(run_dir / "trajectories.jsonl")
    metrics = read_json_object(run_dir / "metrics.json", {})
    run_name = _run_name(run_dir, trajectories)
    harness = _load_run_harness(run_dir) or _load_global_harness(runs_root)
    aggregate = aggregate_run_metrics(predictions, trajectories, metrics, harness)
    row: JsonMap = {
        "run_id": run_dir.name,
        "run_name": run_name,
        "run_type": _canonical_run_type(run_name, run_dir.name),
        "run_dir": str(run_dir),
        "prediction_rows": len(predictions),
        "trajectory_rows": len(trajectories),
        "metrics_rows": _metrics_row_count(metrics),
    }
    row.update(aggregate)
    if include_resources:
        resource_summary = summarize_resources(read_jsonl(run_dir / "resource_usage.jsonl"))
        row["resources"] = resource_summary
        row["device_utilization"] = resource_summary["device_utilization"]
        row["gpu_memory_peak"] = resource_summary["gpu_memory_peak"]
        row["fallback_reasons"] = resource_summary["fallback_reasons"]
    return row


def aggregate_summary(rows: list[JsonMap]) -> JsonMap:
    denominators = [_int_value(row.get("denominator")) for row in rows if isinstance(row.get("denominator"), int)]
    resolved = [_int_value(row.get("resolved")) for row in rows if isinstance(row.get("resolved"), int)]
    pass_at_1_values = [_float_value(row.get("pass_at_1")) for row in rows if isinstance(row.get("pass_at_1"), int | float)]
    total_denominator = sum(denominators)
    total_resolved = sum(resolved)
    return {
        "run_count": len(rows),
        "total_denominator": total_denominator,
        "total_resolved": total_resolved,
        "resolved_rate": (total_resolved / total_denominator) if total_denominator else None,
        "mean_pass_at_1": (sum(pass_at_1_values) / len(pass_at_1_values)) if pass_at_1_values else None,
        "available_run_types": sorted({str(row.get("run_type")) for row in rows if row.get("run_type")}),
    }


def missing_run_types(rows: list[JsonMap]) -> list[str]:
    available = {str(row.get("run_type")) for row in rows if row.get("run_type")}
    return [item for item in EXPECTED_RUN_TYPES if item not in available]


def _run_directories(runs_root: Path) -> list[Path]:
    dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    return sorted(path for path in dirs if _has_run_artifact(path) and ".archived." not in path.name)


def _has_run_artifact(path: Path) -> bool:
    return any((path / name).exists() for name in ("metrics.json", "predictions.jsonl", "trajectories.jsonl", "resource_usage.jsonl"))


def _run_name(run_dir: Path, trajectories: list[JsonMap]) -> str:
    for row in trajectories:
        value = row.get("run_name")
        if isinstance(value, str) and value.strip():
            return value
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        loaded = cast(object, yaml.safe_load(config_path.read_text(encoding="utf-8")))
        if isinstance(loaded, dict):
            loaded_map = cast(dict[object, object], loaded)
            run = loaded_map.get("run")
            if isinstance(run, dict):
                run_map = cast(dict[object, object], run)
                name = run_map.get("name")
                if isinstance(name, str) and name.strip():
                    return name
    return run_dir.name


def _canonical_run_type(run_name: str, run_id: str) -> str:
    label = f"{run_name} {run_id}".lower()
    for expected in EXPECTED_RUN_TYPES:
        if expected in label:
            return expected
    return run_name


def _load_run_harness(run_dir: Path) -> JsonMap:
    for name in ("harness_report.json", "harness_status.json", "swebench_report.json", "report.json"):
        path = run_dir / name
        if path.is_file():
            payload = read_json_object(path, {})
            payload["source"] = str(path)
            return payload
    return {}


def _load_global_harness(runs_root: Path) -> JsonMap:
    candidates = [runs_root.parent / "harness_status.json", runs_root / "harness_status.json"]
    for path in candidates:
        if path.is_file():
            payload = read_json_object(path, {})
            payload["source"] = str(path)
            return payload
    return {}


def _metrics_row_count(metrics: JsonMap) -> int:
    instances = metrics.get("instances", [])
    return len(cast(list[object], instances)) if isinstance(instances, list) else 0


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    return value if isinstance(value, int) else 0


def _float_value(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    return float(value) if isinstance(value, int | float) else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
