#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

_ = sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from repair_agent.config import ConfigError
from repair_agent.env.swebench_loader import (
    load_dataset_gold_patches,
    load_gold_patch_source,
    load_task_manifest,
    smoke_gold_patch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write gold-patch SWE-bench smoke predictions")
    _ = parser.add_argument("--manifest", required=True, help="Task manifest YAML")
    _ = parser.add_argument("--out", required=True, help="Output prediction JSONL path")
    _ = parser.add_argument("--gold-source", help="Local JSONL/YAML source containing actual SWE-bench patch fields")
    _ = parser.add_argument(
        "--model-name",
        default="gold-smoke",
        help="model_name_or_path value for official SWE-bench prediction rows",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = cast(dict[str, object], vars(build_parser().parse_args(argv)))
    manifest_path = _required_path(args.get("manifest"), "--manifest")
    out_path = _required_path(args.get("out"), "--out")
    gold_source = _optional_path(args.get("gold_source"), "--gold-source")
    model_name = _required_string(args.get("model_name"), "--model-name")
    try:
        count = write_gold_smoke_predictions(
            manifest_path=manifest_path, out_path=out_path, model_name=model_name, gold_source=gold_source
        )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {count} gold smoke prediction rows to {out_path}")
    return 0


def write_gold_smoke_predictions(
    manifest_path: Path, out_path: Path, model_name: str, gold_source: Path | None = None
) -> int:
    if not model_name.strip():
        raise ConfigError("--model-name must be a non-empty string")
    manifest = load_task_manifest(manifest_path)
    gold_patches = (
        load_dataset_gold_patches(manifest.dataset_name, manifest.split, manifest.smoke_ids)
        if gold_source is None
        else load_gold_patch_source(gold_source)
    )
    rows = [
        {
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "model_patch": smoke_gold_patch(instance_id, manifest, gold_patches),
        }
        for instance_id in manifest.smoke_ids
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True, separators=(",", ":"))
            _ = handle.write("\n")
    return len(rows)


def _required_path(value: object, label: str) -> Path:
    return Path(_required_string(value, label))


def _optional_path(value: object, label: str) -> Path | None:
    if value is None:
        return None
    return _required_path(value, label)


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
