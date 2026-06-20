from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeAlias


JsonMap: TypeAlias = dict[str, object]


def ensure_run_dir(output_root: str | Path, run_id: str) -> Path:
    if not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_jsonl(path: str | Path, row: JsonMap) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        json.dump(row, handle, sort_keys=True, separators=(",", ":"))
        _ = handle.write("\n")


def read_jsonl(path: str | Path) -> list[JsonMap]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[JsonMap] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        loaded: object = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(f"JSONL row {line_number} in {source} is not an object")
        rows.append(_string_key_mapping(loaded, f"JSONL row {line_number} in {source}"))
    return rows


def write_json_atomic(path: str | Path, payload: JsonMap) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    _ = tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def read_json_object(path: str | Path, default: JsonMap | None = None) -> JsonMap:
    source = Path(path)
    if not source.exists():
        return {} if default is None else dict(default)
    loaded: object = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON file must contain an object: {source}")
    return _string_key_mapping(loaded, f"JSON file {source}")


def initialize_run_files(run_dir: str | Path, force: bool = False) -> dict[str, Path]:
    base = Path(run_dir)
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "trajectories": base / "trajectories.jsonl",
        "predictions": base / "predictions.jsonl",
        "metrics": base / "metrics.json",
        "state": base / "run_state.json",
    }
    for key in ("trajectories", "predictions"):
        if force or not paths[key].exists():
            _ = paths[key].write_text("", encoding="utf-8")
    if force or not paths["metrics"].exists():
        write_json_atomic(paths["metrics"], {"completed": 0, "skipped": 0, "total": 0})
    if force or not paths["state"].exists():
        write_json_atomic(
            paths["state"],
            {"completed_instances": [], "dry_run": True, "status": "initialized"},
        )
    return paths


def _string_key_mapping(value: dict[object, object], error: str) -> JsonMap:
    result: JsonMap = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{error} must use string keys")
        result[key] = item
    return result
