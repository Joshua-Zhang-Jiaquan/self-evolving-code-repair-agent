"""Run a small Defects4J smoke protocol from a JSON config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .defects4j import Defects4JCase, Defects4JClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/defects4j_smoke.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/defects4j_smoke.json"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    client = Defects4JClient()
    payload = {"available": client.available(), "cases": []}
    if not client.available():
        payload["error"] = "defects4j CLI not found on PATH"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"out": str(args.out), "available": False, "error": payload["error"]}, ensure_ascii=False))
        raise SystemExit(2)

    for item in config["cases"]:
        case = Defects4JCase(
            project=item["project"],
            bug_id=int(item["bug_id"]),
            version=item.get("version", "b"),
            workdir=Path(item["workdir"]),
        )
        checkout_output = client.checkout(case)
        compile_output = client.compile(case.workdir)
        test_output = client.test(case.workdir)
        metadata = client.metadata(case.workdir)
        payload["cases"].append(
            {
                "project": case.project,
                "bug_id": case.bug_id,
                "version": case.checkout_version,
                "workdir": str(case.workdir),
                "checkout_output": checkout_output[-2000:],
                "compile_output": compile_output[-2000:],
                "test_output": test_output[-4000:],
                "metadata": metadata,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "cases": len(payload["cases"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
