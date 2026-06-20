from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import cast


PATH_RE = re.compile(r"`([^`]+)`")
ARTIFACT_PREFIXES = ("configs/", "outputs/", "report/", "scripts/", "repair_agent/")


def _project_root(report_path: Path) -> Path:
    return report_path.resolve().parents[1]


def _looks_like_artifact(value: str) -> bool:
    if value.startswith(ARTIFACT_PREFIXES):
        return True
    return value.endswith((".json", ".jsonl", ".yaml", ".yml", ".md", ".py")) and "/" in value


def _expand_reference(root: Path, reference: str) -> list[Path]:
    if "*" in reference:
        return sorted(root.glob(reference))
    return [root / reference]


def collect_references(markdown: str) -> list[str]:
    refs: list[str] = []
    for match in PATH_RE.finditer(markdown):
        candidate = match.group(1).strip()
        if _looks_like_artifact(candidate):
            refs.append(candidate)
    return sorted(set(refs))


def check_report(report_path: Path, summary_path: Path) -> list[str]:
    root = _project_root(report_path)
    errors: list[str] = []

    if not report_path.is_file():
        return [f"missing report: {report_path}"]
    if not summary_path.is_file():
        errors.append(f"missing summary: {summary_path}")
    else:
        try:
            json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid summary JSON: {summary_path}: {exc}")

    markdown = report_path.read_text(encoding="utf-8")
    for reference in collect_references(markdown):
        matches = _expand_reference(root, reference)
        if not matches:
            errors.append(f"unmatched glob reference: {reference}")
            continue
        missing = [path for path in matches if not path.exists()]
        if missing:
            errors.extend(f"missing reference: {reference} -> {path}" for path in missing)

    required = [
        "outputs/run_manifest.json",
        "outputs/run_schedule.json",
        "outputs/summary.json",
        "outputs/harness_status.json",
        "report/figures/credit_assignment.json",
        "report/figures/credit_assignment_tables.md",
    ]
    for reference in required:
        if not (root / reference).is_file():
            errors.append(f"missing required artifact: {reference}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check report artifact references")
    _ = parser.add_argument("report", type=Path, help="Markdown report path")
    _ = parser.add_argument("summary", type=Path, help="outputs/summary.json path")
    namespace = parser.parse_args(argv)
    report = cast(Path, namespace.report)
    summary = cast(Path, namespace.summary)

    errors = check_report(report, summary)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"report references ok: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
