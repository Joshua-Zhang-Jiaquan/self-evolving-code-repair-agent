from __future__ import annotations
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from repair_agent.logging import JsonMap


REQUIRED_PREDICTION_KEYS = ("instance_id", "model_name_or_path", "model_patch")
RESOLVED_STATUSES = {"pass", "passed", "resolved", "success", "succeeded"}
ERROR_STATUSES = {"error", "errored", "timeout", "timed_out", "failed", "failure", "crashed"}


@dataclass(frozen=True)
class PredictionValidationSummary:
    path: Path
    row_count: int
    instance_ids: tuple[str, ...]


def validate_prediction_row(row: Mapping[object, object], line_number: int) -> None:
    for key in REQUIRED_PREDICTION_KEYS:
        if key not in row:
            raise ValueError(f"line {line_number} missing required key '{key}'")
        if not isinstance(row[key], str):
            raise ValueError(f"line {line_number} key '{key}' must be a string")


def validate_predictions_file(path: Path) -> PredictionValidationSummary:
    if not path.is_file():
        raise ValueError(f"predictions file not found: {path}")
    instance_ids: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            loaded = cast(object, json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number} is not valid JSON: {exc.msg}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"line {line_number} must be a JSON object")
        row = cast(dict[object, object], loaded)
        validate_prediction_row(row, line_number)
        instance_ids.append(cast(str, row["instance_id"]))
    if not instance_ids:
        raise ValueError(f"predictions file has no nonblank rows: {path}")
    return PredictionValidationSummary(path=path, row_count=len(instance_ids), instance_ids=tuple(instance_ids))


def prediction_stats(predictions: Sequence[JsonMap]) -> JsonMap:
    sizes = [_patch_size(row.get("model_patch")) for row in predictions]
    empty = sum(1 for row in predictions if not str(row.get("model_patch", "")).strip())
    return {
        "prediction_rows": len(predictions),
        "empty_patch_count": empty,
        "empty_patch_rate": _safe_rate(empty, len(predictions)),
        "patch_apply_count": len(predictions) - empty,
        "patch_apply_rate": _safe_rate(len(predictions) - empty, len(predictions)),
        "patch_size": {
            "avg_chars": _mean([item["chars"] for item in sizes]),
            "avg_lines": _mean([item["lines"] for item in sizes]),
            "max_chars": max([item["chars"] for item in sizes], default=0),
        },
    }


def denominator_counts(rows: Sequence[JsonMap]) -> JsonMap:
    resolved = 0
    unresolved = 0
    empty = 0
    errors = 0
    timeouts = 0
    failures = 0
    for row in rows:
        status = _status(row)
        patch_text = row.get("model_patch")
        patch_status = str(row.get("patch_status", "")).lower()
        is_empty = patch_status in {"empty", "no_patch"} or (isinstance(patch_text, str) and not patch_text.strip())
        if status in RESOLVED_STATUSES:
            resolved += 1
        elif is_empty:
            empty += 1
        elif status in {"timeout", "timed_out"}:
            timeouts += 1
            errors += 1
        elif status in ERROR_STATUSES:
            failures += 1
            errors += 1
        else:
            unresolved += 1
    denominator = resolved + unresolved + empty + errors
    return {
        "resolved": resolved,
        "unresolved": unresolved,
        "empty": empty,
        "error": errors,
        "timeout": timeouts,
        "failure": failures,
        "denominator": denominator,
        "resolved_rate": _safe_rate(resolved, denominator),
    }


def pass_at_k(attempts_by_instance: Mapping[str, Sequence[bool]], k: int) -> float | None:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not attempts_by_instance:
        return None
    passed = 0
    for attempts in attempts_by_instance.values():
        if any(attempts[:k]):
            passed += 1
    return passed / len(attempts_by_instance)


def aggregate_run_metrics(
    predictions: Sequence[JsonMap],
    trajectories: Sequence[JsonMap],
    metrics: JsonMap,
    official_harness: JsonMap | None = None,
) -> JsonMap:
    instances = _instance_rows(predictions, trajectories, metrics)
    denominator = denominator_counts(instances)
    trajectory = trajectory_stats(trajectories, metrics)
    patch = prediction_stats(predictions)
    attempts = _attempts_by_instance(instances)
    official = summarize_official_harness(official_harness or {})
    official_rate = official.get("official_resolved_rate") if official.get("official") else None
    resolved_rate = official_rate if official_rate is not None else denominator["resolved_rate"]
    result: JsonMap = {
        "denominator": denominator["denominator"],
        "resolved": denominator["resolved"],
        "unresolved": denominator["unresolved"],
        "empty": denominator["empty"],
        "error": denominator["error"],
        "timeout": denominator["timeout"],
        "failure": denominator["failure"],
        "resolved_rate": resolved_rate,
        "official_resolved_rate": official_rate,
        "official_harness": official,
        "pass_at_1": pass_at_k(attempts, 1),
        "pass_at_k": pass_at_k(attempts, max((len(value) for value in attempts.values()), default=1)),
        "visible_test_pass_rate": denominator["resolved_rate"],
        "patch_apply_rate": patch["patch_apply_rate"],
        "empty_patch_rate": patch["empty_patch_rate"],
        "error_rate": _safe_rate(cast(int, denominator["error"]), cast(int, denominator["denominator"])),
        "average_tool_calls": trajectory["average_tool_calls"],
        "average_test_runs": trajectory["average_test_runs"],
        "wall_time_seconds": trajectory["wall_time_seconds"],
        "patch_size": patch["patch_size"],
        "unsafe_edit_rate": trajectory["unsafe_edit_rate"],
        "cost_proxy": cost_proxy(trajectory, patch),
    }
    return result


def trajectory_stats(trajectories: Sequence[JsonMap], metrics: JsonMap) -> JsonMap:
    per_instance_tool: dict[str, int] = defaultdict(int)
    per_instance_tests: dict[str, int] = defaultdict(int)
    unsafe_edits = 0
    edit_count = 0
    wall_times: list[float] = []
    for row in trajectories:
        instance_id = str(row.get("instance_id", ""))
        if instance_id:
            per_instance_tool[instance_id] = max(per_instance_tool[instance_id], _int(row.get("tool_call_count")))
            per_instance_tests[instance_id] = max(per_instance_tests[instance_id], _int(row.get("test_run_count")))
        if row.get("tool") == "edit_file" or row.get("action") == "edit_file":
            edit_count += 1
            status = str(row.get("status", "")).lower()
            metadata = row.get("metadata")
            metadata_map = cast(dict[object, object], metadata) if isinstance(metadata, dict) else {}
            unsafe = bool(metadata_map.get("unsafe_edit"))
            if status in {"denied", "unsafe", "blocked"} or unsafe:
                unsafe_edits += 1
        for key in ("wall_time_seconds", "elapsed_seconds", "duration_seconds"):
            value = row.get(key)
            if isinstance(value, int | float):
                wall_times.append(float(value))
                break
    for item in _metric_instance_list(metrics):
        instance_id = str(item.get("instance_id", ""))
        if not instance_id:
            continue
        per_instance_tool[instance_id] = max(per_instance_tool[instance_id], _int(item.get("tool_call_count")))
        per_instance_tests[instance_id] = max(per_instance_tests[instance_id], _int(item.get("test_run_count")))
        wall_value = item.get("wall_time_seconds")
        if isinstance(wall_value, int | float):
            wall_times.append(float(wall_value))
    return {
        "average_tool_calls": _mean(list(per_instance_tool.values())),
        "average_test_runs": _mean(list(per_instance_tests.values())),
        "unsafe_edit_rate": _safe_rate(unsafe_edits, edit_count),
        "wall_time_seconds": {"average": _mean(wall_times), "max": max(wall_times, default=0.0), "source": "reported" if wall_times else "not_reported"},
    }


def cost_proxy(trajectory: JsonMap, patch: JsonMap) -> float:
    patch_size = patch.get("patch_size")
    avg_chars = 0.0
    if isinstance(patch_size, dict) and isinstance(patch_size.get("avg_chars"), int | float):
        avg_chars = float(patch_size["avg_chars"])
    tool_calls = _float(trajectory.get("average_tool_calls"))
    test_runs = _float(trajectory.get("average_test_runs"))
    return round(tool_calls + (5.0 * test_runs) + (avg_chars / 1000.0), 6)


def summarize_resources(rows: Sequence[JsonMap]) -> JsonMap:
    visible = sorted(_visible_device_ids(rows))
    assigned = Counter(_assigned_device_ids(rows))
    utilization_values: dict[int, list[float]] = defaultdict(list)
    memory_values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        for device_id, value in _device_metric_values(row, ("gpu_utilization", "gpu_utilization_percent", "utilization")):
            utilization_values[device_id].append(value)
        for device_id, value in _device_metric_values(row, ("gpu_memory_peak_mb", "memory_peak_mb", "memory_used_mb")):
            memory_values[device_id].append(value)
    device_utilization: JsonMap = {}
    gpu_memory_peak: JsonMap = {}
    for device_id in visible:
        values = utilization_values.get(device_id, [])
        device_utilization[str(device_id)] = {
            "assigned_task_count": assigned[device_id],
            "run_count": len(rows),
            "average_utilization_percent": _mean(values) if values else None,
            "source": "reported" if values else "not_reported",
        }
        memory = memory_values.get(device_id, [])
        gpu_memory_peak[str(device_id)] = {
            "memory_peak_mb": max(memory) if memory else None,
            "source": "reported" if memory else "not_reported",
        }
    fallback_reasons = _fallback_reasons(rows)
    return {
        "visible_device_ids": visible,
        "device_utilization": device_utilization,
        "gpu_memory_peak": gpu_memory_peak,
        "worker_settings": _worker_settings(rows),
        "cpu_worker_utilization": _worker_utilization(rows, "cpu"),
        "docker_worker_utilization": _worker_utilization(rows, "docker"),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
    }


def summarize_model_gates(paths: Sequence[Path]) -> JsonMap:
    models: JsonMap = {}
    fallback_counts: Counter[str] = Counter()
    for path in sorted(paths):
        if not path.is_file():
            continue
        loaded = _load_json_object(path)
        model = str(loaded.get("model") or path.stem)
        reasons = _reasons_from_mapping(loaded.get("fallback"))
        fallback_counts.update(reasons)
        models[model] = {
            "path": str(path),
            "status": loaded.get("status"),
            "reason": loaded.get("reason"),
            "device_ids": loaded.get("device_ids", []),
            "device_strategy": loaded.get("device_strategy"),
            "fallback_reasons": reasons,
        }
    return {"models": models, "fallback_reasons": dict(sorted(fallback_counts.items()))}


def summarize_official_harness(report: JsonMap) -> JsonMap:
    if not report:
        return {"official": False, "official_resolved_rate": None, "source": "missing"}
    executed = report.get("official_harness_executed")
    status = str(report.get("status", "")).lower()
    if executed is False or status in {"blocked", "not_run", "unavailable"}:
        return {
            "official": False,
            "official_resolved_rate": None,
            "status": report.get("status"),
            "blocked_reason": report.get("blocked_reason") or report.get("fallback_reason"),
            "source": report.get("source", "harness_status"),
        }
    resolved = _count_field(report, ("resolved", "resolved_ids"))
    unresolved = _count_field(report, ("unresolved", "unresolved_ids"))
    empty = _count_field(report, ("empty_patch", "empty", "empty_patch_ids"))
    errors = _count_field(report, ("error", "errors", "error_ids", "timeout", "timeouts", "failure", "failures"))
    denominator = resolved + unresolved + empty + errors
    return {
        "official": denominator > 0,
        "official_resolved_rate": _safe_rate(resolved, denominator),
        "resolved": resolved,
        "unresolved": unresolved,
        "empty": empty,
        "error": errors,
        "denominator": denominator,
        "source": report.get("source", "harness_report"),
    }


def _instance_rows(predictions: Sequence[JsonMap], trajectories: Sequence[JsonMap], metrics: JsonMap) -> list[JsonMap]:
    rows: dict[str, JsonMap] = {}
    for row in predictions:
        instance_id = str(row.get("instance_id", ""))
        if instance_id:
            rows.setdefault(instance_id, {"instance_id": instance_id}).update(row)
    for item in _metric_instance_list(metrics):
        instance_id = str(item.get("instance_id", ""))
        if instance_id:
            rows.setdefault(instance_id, {"instance_id": instance_id}).update(item)
    for row in trajectories:
        instance_id = str(row.get("instance_id", ""))
        if not instance_id:
            continue
        entry = rows.setdefault(instance_id, {"instance_id": instance_id})
        if "final_status" in row:
            entry.setdefault("final_status", row["final_status"])
        if "status" in row and "final_status" not in entry:
            entry.setdefault("status", row["status"])
    return list(rows.values())


def _attempts_by_instance(instances: Sequence[JsonMap]) -> dict[str, list[bool]]:
    attempts: dict[str, list[bool]] = defaultdict(list)
    for row in instances:
        instance_id = str(row.get("instance_id", ""))
        if instance_id:
            attempts[instance_id].append(_status(row) in RESOLVED_STATUSES)
    return dict(attempts)


def _metric_instance_list(metrics: JsonMap) -> list[JsonMap]:
    instances = metrics.get("instances", [])
    if not isinstance(instances, list):
        return []
    return [cast(JsonMap, item) for item in instances if isinstance(item, dict)]


def _patch_size(value: object) -> dict[str, int]:
    if not isinstance(value, str):
        return {"chars": 0, "lines": 0}
    return {"chars": len(value), "lines": len(value.splitlines())}


def _status(row: JsonMap) -> str:
    for key in ("official_status", "final_status", "outcome", "status"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return "unresolved"


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _mean(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return 0


def _float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def _visible_device_ids(rows: Sequence[JsonMap]) -> set[int]:
    ids: set[int] = set()
    for row in rows:
        for key in ("visible_gpus", "device_ids"):
            value = row.get(key)
            if isinstance(value, list):
                ids.update(_ints(value))
        assigned = row.get("assigned_device")
        parsed = _parse_device_id(assigned)
        if parsed is not None:
            ids.add(parsed)
    return ids


def _assigned_device_ids(rows: Sequence[JsonMap]) -> list[int]:
    result: list[int] = []
    for row in rows:
        parsed = _parse_device_id(row.get("assigned_device"))
        if parsed is not None:
            result.append(parsed)
    return result


def _parse_device_id(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    lower = value.strip().lower()
    if lower.isdecimal():
        return int(lower)
    if lower.startswith("cuda:") and lower.split(":", 1)[1].isdigit():
        return int(lower.split(":", 1)[1])
    if lower.startswith("gpu"):
        digits = "".join(char for char in lower if char.isdigit())
        if digits:
            return int(digits)
    return None


def _device_metric_values(row: JsonMap, keys: Iterable[str]) -> list[tuple[int, float]]:
    result: list[tuple[int, float]] = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, dict):
            for raw_device, raw_value in value.items():
                parsed = _parse_device_id(raw_device)
                if parsed is not None and isinstance(raw_value, int | float):
                    result.append((parsed, float(raw_value)))
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                item_map = cast(dict[object, object], item)
                parsed = _parse_device_id(item_map.get("device_id", item_map.get("index")))
                metric = item_map.get(key, item_map.get("value"))
                if parsed is not None and isinstance(metric, int | float):
                    result.append((parsed, float(metric)))
        elif isinstance(value, int | float):
            parsed = _parse_device_id(row.get("assigned_device"))
            if parsed is not None:
                result.append((parsed, float(value)))
    return result


def _fallback_reasons(rows: Sequence[JsonMap]) -> Counter[str]:
    reasons: Counter[str] = Counter()
    for row in rows:
        reasons.update(_reasons_from_mapping(row.get("fallback")))
        for key in ("fallback_reason", "blocked_reason"):
            value = row.get(key)
            if isinstance(value, str) and value:
                reasons[value] += 1
    return reasons


def _reasons_from_mapping(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    reasons = value.get("reasons", [])
    if not isinstance(reasons, list):
        return []
    return [item for item in reasons if isinstance(item, str) and item]


def _worker_settings(rows: Sequence[JsonMap]) -> JsonMap:
    latest: JsonMap = {}
    for row in rows:
        worker = row.get("worker_settings")
        if isinstance(worker, dict):
            latest = {str(key): item for key, item in worker.items()}
    return latest


def _worker_utilization(rows: Sequence[JsonMap], kind: str) -> JsonMap:
    key = f"{kind}_worker_utilization"
    values = [_float(row[key]) for row in rows if isinstance(row.get(key), int | float)]
    return {"average": _mean(values), "source": "reported" if values else "not_reported"}


def _count_field(report: JsonMap, keys: Sequence[str]) -> int:
    total = 0
    for key in keys:
        value = report.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            total += value
        elif isinstance(value, list | tuple | set):
            total += len(value)
        elif isinstance(value, dict):
            total += len(value)
    return total


def _ints(values: Sequence[object]) -> set[int]:
    result: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            result.add(value)
        elif isinstance(value, str) and value.isdecimal():
            result.add(int(value))
    return result


def _load_json_object(path: Path) -> JsonMap:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return {str(key): item for key, item in loaded.items()}
