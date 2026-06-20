from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from repair_agent.tools.core import ToolRegistry, build_default_registry


REGISTRY: ToolRegistry = build_default_registry()


def get_registry() -> ToolRegistry:
    return REGISTRY


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safe repair-agent tool registry")
    _ = parser.add_argument("--list", action="store_true", help="print registered tool names")
    _ = parser.add_argument("--schemas", action="store_true", help="print tool schemas as JSON")
    args: dict[str, object] = vars(parser.parse_args(argv))
    if args.get("list") is True:
        for name in REGISTRY.list_tools():
            print(name)
        return 0
    if args.get("schemas") is True:
        print(json.dumps(REGISTRY.schemas(), indent=2, sort_keys=True))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
