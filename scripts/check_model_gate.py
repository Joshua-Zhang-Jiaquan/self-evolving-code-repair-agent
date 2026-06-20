#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repair_agent.agent.models import (  # noqa: E402
    check_diffrwkv_gate,
    check_qwable_gate,
    load_models_config,
    write_gate_record,
)
from repair_agent.config import ConfigError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run safe local model compatibility gates")
    _ = parser.add_argument("--model", choices=["qwable", "diffrwkv"], required=True)
    _ = parser.add_argument("--checkpoint", type=str, default=None, help="DiffRWKV checkpoint directory")
    _ = parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use --dry-run for the safe metadata gate, or --no-dry-run for real local inference",
    )
    _ = parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Tokens to generate during real (non-dry-run) Qwable inference",
    )
    _ = parser.add_argument("--models-config", type=str, default="configs/models.yaml")
    _ = parser.add_argument("--resources", type=str, default="configs/resources.yaml")
    _ = parser.add_argument("--out-dir", type=str, default="outputs/model_gates")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    model = cast(str, args.model)
    models_config_path = cast(str, args.models_config)
    resources_path = cast(str, args.resources)
    checkpoint = cast(str | None, args.checkpoint)
    dry_run = cast(bool, args.dry_run)
    max_new_tokens = cast(int, args.max_new_tokens)
    out_dir = cast(str, args.out_dir)
    try:
        models_config = load_models_config(models_config_path)
        if model == "qwable":
            record = check_qwable_gate(
                models_config=models_config,
                resources_path=resources_path,
                dry_run=dry_run,
                max_new_tokens=max_new_tokens,
            )
        else:
            record = check_diffrwkv_gate(
                models_config=models_config,
                checkpoint=checkpoint,
                resources_path=resources_path,
                dry_run=dry_run,
            )
        output_path = write_gate_record(record, out_dir)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"model gate error: {exc}", file=sys.stderr)
        return 2

    print(f"{record['model']} gate {record['status']}: {record['reason']} -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
