from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast


STRICT_FLAG = "--strict-official"
SIMULATE_FLAG = "--simulate-docker-failure"
FIXTURE_MARKER = "local-"
OFFICIAL_ID_SEPARATOR = "__"
SKIPPED_EXISTING = "skipped_existing"
ARCHIVED_MARKER = ".archived."
SENSITIVE_TRAJECTORY_KEYS = frozenset({"patch", "test_patch"})


@dataclass(frozen=True)
class CheckConfig:
    """Resolved filesystem locations and policy inputs for the integrity gate."""

    manifest: Path
    summary: Path
    harness: Path
    qwable: Path
    schedule: Path
    readme: Path
    report: Path
    runs_dir: Path
    results_json: Path
    tables_dir: Path
    forbidden_paths: tuple[str, ...]


def _read_json_object(path: Path) -> tuple[dict[str, object] | None, str | None]:
    if not path.is_file():
        return None, f"missing file: {path}"
    try:
        loaded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {path}: {exc}"
    if not isinstance(loaded, dict):
        return None, f"not a JSON object: {path}"
    return cast(dict[str, object], loaded), None


def _iter_jsonl_objects(path: Path) -> tuple[list[tuple[int, dict[str, object]]], list[str]]:
    rows: list[tuple[int, dict[str, object]]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            node = cast(object, json.loads(stripped))
        except json.JSONDecodeError as exc:
            errors.append(f"{path} line {line_number}: invalid JSON: {exc}")
            continue
        if isinstance(node, dict):
            rows.append((line_number, cast(dict[str, object], node)))
        else:
            errors.append(f"{path} line {line_number}: JSONL row is not an object")
    return rows, errors


def _iter_jsonl_nodes(path: Path) -> tuple[list[tuple[int, object]], list[str]]:
    nodes: list[tuple[int, object]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            node = cast(object, json.loads(stripped))
        except json.JSONDecodeError as exc:
            errors.append(f"{path} line {line_number}: invalid JSON: {exc}")
            continue
        nodes.append((line_number, node))
    return nodes, errors


def _find_sensitive_keys(node: object, keys: frozenset[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in cast(dict[str, object], node).items():
            if key in keys:
                found.add(key)
            found |= _find_sensitive_keys(value, keys)
    elif isinstance(node, list):
        for item in cast(list[object], node):
            found |= _find_sensitive_keys(item, keys)
    return found


def _strict_stages(manifest: dict[str, object]) -> list[dict[str, object]]:
    stages = manifest.get("stages")
    strict: list[dict[str, object]] = []
    if not isinstance(stages, list):
        return strict
    for stage in cast(list[object], stages):
        if not isinstance(stage, dict):
            continue
        stage_dict = cast(dict[str, object], stage)
        command = stage_dict.get("command")
        command_items = cast(list[object], command) if isinstance(command, list) else []
        if STRICT_FLAG in command_items:
            strict.append(stage_dict)
    return strict


def _strict_run_ids(manifest: dict[str, object]) -> list[str]:
    run_ids: list[str] = []
    for stage in _strict_stages(manifest):
        run_id = stage.get("run_id")
        if isinstance(run_id, str) and run_id:
            run_ids.append(run_id)
    return run_ids


def check_no_fixture_ids_in_strict_runs(cfg: CheckConfig) -> list[str]:
    manifest, error = _read_json_object(cfg.manifest)
    if manifest is None:
        return [cast(str, error)]
    errors: list[str] = []
    run_ids = _strict_run_ids(manifest)
    if not run_ids:
        return ["no strict run found in manifest (expected a --strict-official stage with a run_id)"]
    for run_id in run_ids:
        predictions = cfg.runs_dir / run_id / "predictions.jsonl"
        if not predictions.is_file():
            errors.append(f"strict run '{run_id}' is missing predictions.jsonl at {predictions}")
            continue
        rows, parse_errors = _iter_jsonl_objects(predictions)
        errors.extend(parse_errors)
        for line_number, row in rows:
            instance_id = row.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                errors.append(f"strict run '{run_id}' line {line_number}: missing instance_id")
                continue
            if FIXTURE_MARKER in instance_id or OFFICIAL_ID_SEPARATOR not in instance_id:
                errors.append(f"strict run '{run_id}' line {line_number}: fixture-style instance_id '{instance_id}' (contains '{FIXTURE_MARKER}' or lacks '{OFFICIAL_ID_SEPARATOR}')")
    return errors


def check_no_skipped_strict_stage(cfg: CheckConfig) -> list[str]:
    manifest, error = _read_json_object(cfg.manifest)
    if manifest is None:
        return [cast(str, error)]
    strict = _strict_stages(manifest)
    if not strict:
        return ["no strict stage found in manifest (expected at least one --strict-official stage)"]
    errors: list[str] = []
    for stage in strict:
        if stage.get("status") == SKIPPED_EXISTING:
            stage_id = stage.get("stage_id")
            errors.append(f"strict stage '{stage_id}' has status '{SKIPPED_EXISTING}' (strict stages must execute fresh, not reuse prior outputs)")
    return errors


def check_no_simulate_flag(cfg: CheckConfig) -> list[str]:
    errors: list[str] = []
    targets = (("README", cfg.readme), ("report", cfg.report), ("run schedule", cfg.schedule))
    for label, path in targets:
        if not path.is_file():
            errors.append(f"cannot check {label}: missing file {path}")
            continue
        if SIMULATE_FLAG in path.read_text(encoding="utf-8", errors="ignore"):
            errors.append(f"{label} ({path}) contains forbidden flag '{SIMULATE_FLAG}'")
    return errors


def check_no_patch_keys_in_trajectories(cfg: CheckConfig) -> list[str]:
    if not cfg.runs_dir.is_dir():
        return [f"runs directory missing: {cfg.runs_dir}"]
    errors: list[str] = []
    trajectory_files = sorted(
        path for path in cfg.runs_dir.glob("*/trajectories.jsonl") if ARCHIVED_MARKER not in path.parent.name
    )
    for path in trajectory_files:
        nodes, parse_errors = _iter_jsonl_nodes(path)
        errors.extend(parse_errors)
        for line_number, node in nodes:
            found = _find_sensitive_keys(node, SENSITIVE_TRAJECTORY_KEYS)
            if found:
                keys = ", ".join(sorted(found))
                errors.append(f"trajectory {path} line {line_number} exposes hidden gold key(s): {keys}")
    return errors


def check_qwable_not_dry_run(cfg: CheckConfig) -> list[str]:
    obj, error = _read_json_object(cfg.qwable)
    if obj is None:
        return [cast(str, error)]
    details = obj.get("details")
    if not isinstance(details, dict):
        return [f"qwable gate {cfg.qwable} is missing a 'details' object"]
    dry_run = cast(dict[str, object], details).get("dry_run")
    if dry_run is not False:
        return [f"qwable gate {cfg.qwable} is not real inference: details.dry_run={dry_run!r} (expected False)"]
    return []


def check_harness_executed_or_blocked(cfg: CheckConfig) -> list[str]:
    obj, error = _read_json_object(cfg.harness)
    if obj is None:
        return [cast(str, error)]
    if "official_harness_executed" not in obj:
        return [f"harness status {cfg.harness} is missing the 'official_harness_executed' field"]
    executed = obj.get("official_harness_executed")
    if executed is True:
        return []
    blockers = obj.get("blockers")
    has_blockers = isinstance(blockers, list) and bool(cast(list[object], blockers))
    if has_blockers:
        return []
    return [f"harness {cfg.harness}: official_harness_executed={executed!r} but no non-empty 'blockers' list recorded (must be executed or honestly blocked)"]


def check_forbidden_paths_absent(cfg: CheckConfig) -> list[str]:
    errors: list[str] = []
    targets = (("run manifest", cfg.manifest), ("summary", cfg.summary))
    for label, path in targets:
        if not path.is_file():
            errors.append(f"cannot check {label}: missing file {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for forbidden in cfg.forbidden_paths:
            if forbidden and forbidden in text:
                errors.append(f"forbidden path '{forbidden}' appears in {label} ({path})")
    return errors


def check_required_files_exist(cfg: CheckConfig) -> list[str]:
    required = (
        cfg.summary,
        cfg.manifest,
        cfg.harness,
        cfg.qwable,
        cfg.results_json,
        cfg.tables_dir / "results_table.md",
        cfg.tables_dir / "ablation_comparison.md",
        cfg.tables_dir / "device_utilization.md",
    )
    return [f"missing required result file: {path}" for path in required if not path.is_file()]


def run_all_checks(cfg: CheckConfig) -> list[tuple[str, list[str]]]:
    return [
        ("check 1: strict runs use only official instance IDs", check_no_fixture_ids_in_strict_runs(cfg)),
        ("check 2: no strict stage is skipped_existing", check_no_skipped_strict_stage(cfg)),
        ("check 3: no --simulate-docker-failure in README/report/schedule", check_no_simulate_flag(cfg)),
        ("check 4: no hidden patch/test_patch in trajectories", check_no_patch_keys_in_trajectories(cfg)),
        ("check 5: qwable gate ran real inference (not dry-run)", check_qwable_not_dry_run(cfg)),
        ("check 6: official harness executed or honestly blocked", check_harness_executed_or_blocked(cfg)),
        ("check 7: forbidden paths absent from manifests", check_forbidden_paths_absent(cfg)),
        ("check 8: all planned result files exist", check_required_files_exist(cfg)),
    ]


def _required_path(namespace: argparse.Namespace, attribute: str) -> Path:
    value = cast(object, getattr(namespace, attribute))
    if not isinstance(value, Path):
        raise TypeError(f"--{attribute} must be a path")
    return value.resolve()


def _optional_path(namespace: argparse.Namespace, attribute: str) -> Path | None:
    value = cast(object, getattr(namespace, attribute))
    return value.resolve() if isinstance(value, Path) else None


def build_config(namespace: argparse.Namespace) -> CheckConfig:
    manifest = _required_path(namespace, "manifest")
    summary = _required_path(namespace, "summary")
    harness = _required_path(namespace, "harness")
    qwable = _required_path(namespace, "qwable")
    outputs_dir = manifest.parent

    root_override = _optional_path(namespace, "root")
    root = root_override if root_override is not None else outputs_dir.parent

    schedule_override = _optional_path(namespace, "schedule")
    schedule = schedule_override if schedule_override is not None else outputs_dir / "run_schedule.json"

    runs_override = _optional_path(namespace, "runs_dir")
    runs_dir = runs_override if runs_override is not None else outputs_dir / "runs"

    readme_override = _optional_path(namespace, "readme")
    readme = readme_override if readme_override is not None else root / "README.md"

    report_override = _optional_path(namespace, "report")
    report = report_override if report_override is not None else root / "report" / "report.md"

    results_override = _optional_path(namespace, "results_json")
    results_json = results_override if results_override is not None else root / "report" / "figures" / "results.json"

    tables_override = _optional_path(namespace, "tables_dir")
    tables_dir = tables_override if tables_override is not None else root / "report" / "tables"

    forbidden_raw = cast(object, getattr(namespace, "forbidden_path"))
    forbidden_paths = tuple(cast(list[str], forbidden_raw)) if isinstance(forbidden_raw, list) else ()

    return CheckConfig(
        manifest=manifest,
        summary=summary,
        harness=harness,
        qwable=qwable,
        schedule=schedule,
        readme=readme,
        report=report,
        runs_dir=runs_dir,
        results_json=results_json,
        tables_dir=tables_dir,
        forbidden_paths=forbidden_paths,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict run integrity, leakage, and result-completeness gate")
    _ = parser.add_argument("--manifest", type=Path, required=True, help="outputs/run_manifest.json path")
    _ = parser.add_argument("--summary", type=Path, required=True, help="outputs/summary.json path")
    _ = parser.add_argument("--harness", type=Path, required=True, help="outputs/harness_status.json path")
    _ = parser.add_argument("--qwable", type=Path, required=True, help="outputs/model_gates/qwable.json path")
    _ = parser.add_argument("--forbidden-path", dest="forbidden_path", action="append", default=None, help="path fragment that must not appear in the manifests (repeatable)")
    _ = parser.add_argument("--root", type=Path, default=None, help="project root (defaults to the manifest's grandparent)")
    _ = parser.add_argument("--schedule", type=Path, default=None, help="run schedule path (defaults to <outputs>/run_schedule.json)")
    _ = parser.add_argument("--runs-dir", dest="runs_dir", type=Path, default=None, help="runs directory (defaults to <outputs>/runs)")
    _ = parser.add_argument("--readme", type=Path, default=None, help="README path (defaults to <root>/README.md)")
    _ = parser.add_argument("--report", type=Path, default=None, help="report path (defaults to <root>/report/report.md)")
    _ = parser.add_argument("--results-json", dest="results_json", type=Path, default=None, help="results figure JSON (defaults to <root>/report/figures/results.json)")
    _ = parser.add_argument("--tables-dir", dest="tables_dir", type=Path, default=None, help="tables directory (defaults to <root>/report/tables)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    cfg = build_config(namespace)

    results = run_all_checks(cfg)
    failed = [(name, errors) for name, errors in results if errors]

    for name, errors in results:
        marker = "FAIL" if errors else "PASS"
        print(f"[{marker}] {name}")
        for error in errors:
            print(f"    - {error}", file=sys.stderr)

    if failed:
        print(f"strict integrity FAILED: {len(failed)}/{len(results)} checks reported errors", file=sys.stderr)
        return 1
    print(f"strict integrity OK: all {len(results)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
