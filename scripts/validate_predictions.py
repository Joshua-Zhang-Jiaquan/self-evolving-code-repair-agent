#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repair_agent.evaluation.metrics import PredictionValidationSummary, validate_predictions_file

__all__ = ["PredictionValidationSummary", "build_parser", "main", "validate_predictions_file"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate SWE-bench official prediction JSONL")
    _ = parser.add_argument("predictions", help="Prediction JSONL path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = cast(dict[str, object], vars(build_parser().parse_args(argv)))
    predictions = args.get("predictions")
    if not isinstance(predictions, str) or not predictions.strip():
        print("prediction validation error: predictions path must be a non-empty string", file=sys.stderr)
        return 2
    try:
        summary = validate_predictions_file(Path(predictions))
    except ValueError as exc:
        print(f"prediction validation error: {exc}", file=sys.stderr)
        return 2
    print(f"valid {summary.row_count} prediction rows in {summary.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
