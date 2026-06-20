from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from repair_agent.config import ConfigError, ConfigMap
from repair_agent.logging import write_json_atomic


DEFECTS4J_ID_PATTERN = re.compile(r"^(?P<project>[A-Z][a-zA-Z0-9]*)_(?P<bug_id>\d+)$")
SUPPORTED_PROJECTS = frozenset(
    {
        "Chart",
        "Cli",
        "Closure",
        "Codec",
        "Collections",
        "Compress",
        "Csv",
        "Gson",
        "JacksonCore",
        "JacksonDatabind",
        "JacksonXml",
        "Jsoup",
        "JxPath",
        "Lang",
        "Math",
        "Mockito",
        "Time",
    }
)


@dataclass(frozen=True)
class Defects4JInstance:
    instance_id: str
    project: str
    bug_id: int


@dataclass(frozen=True)
class Defects4JResult:
    instance_id: str
    resolved: bool
    patch_applied: bool
    compiled: bool
    test_count: int
    failing_tests: list[str]
    error: str | None


def find_defects4j_home() -> Path | None:
    """Return the Defects4J installation directory, or None if not found."""
    env = os.environ.get("DEFECTS4J_HOME")
    if env:
        path = Path(env)
        if _is_valid_home(path):
            return path
    for candidate in (
        Path("/tmp/opencode/defects4j"),
        Path.home() / "defects4j",
        Path.home() / ".local" / "defects4j",
        Path("/usr/local/defects4j"),
    ):
        if _is_valid_home(candidate):
            return candidate
    return None


def _is_valid_home(path: Path) -> bool:
    return path.is_dir() and (path / "framework" / "bin" / "defects4j").is_file()


def is_available() -> bool:
    """Return True if the `defects4j` command can run a trivial query."""
    home = find_defects4j_home()
    if home is None:
        return False
    cmd = str(home / "framework" / "bin" / "defects4j")
    try:
        completed = subprocess.run(
            [cmd, "info", "-p", "Lang", "-b", "1"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def parse_instance_id(instance_id: str) -> Defects4JInstance | None:
    """Parse a Defects4J instance id such as ``Lang_1``."""
    match = DEFECTS4J_ID_PATTERN.match(instance_id)
    if not match:
        return None
    project = match.group("project")
    bug_id = int(match.group("bug_id"))
    if project not in SUPPORTED_PROJECTS:
        return None
    return Defects4JInstance(instance_id=instance_id, project=project, bug_id=bug_id)


def defects4j_ids_in_predictions(predictions_path: Path) -> tuple[Defects4JInstance, ...]:
    """Return the unique Defects4J instances referenced in a prediction JSONL file."""
    instances: list[Defects4JInstance] = []
    try:
        lines = predictions_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    for line in lines:
        if not line.strip():
            continue
        try:
            row_object = cast(object, json.loads(line))
        except json.JSONDecodeError:
            continue
        if not isinstance(row_object, dict):
            continue
        row = cast(dict[str, object], row_object)
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str):
            continue
        parsed = parse_instance_id(instance_id)
        if parsed is not None:
            instances.append(parsed)
    return tuple(dict.fromkeys(instances))


def run_defects4j_command(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    """Run a Defects4J subcommand with DEFECTS4J_HOME and PATH configured."""
    home = find_defects4j_home()
    if home is None:
        raise ConfigError("Defects4J home not found")
    env = dict(os.environ)
    env["DEFECTS4J_HOME"] = str(home)
    bin_dir = str(home / "framework" / "bin")
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def checkout_bug(instance: Defects4JInstance, workdir: Path, version: str = "b") -> None:
    """Checkout a Defects4J bug version (``b`` = buggy, ``f`` = fixed)."""
    workdir.mkdir(parents=True, exist_ok=True)
    if any(workdir.iterdir()):
        shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    result = run_defects4j_command(
        [
            "defects4j",
            "checkout",
            "-p",
            instance.project,
            "-v",
            f"{instance.bug_id}{version}",
            "-w",
            str(workdir),
        ],
        timeout=300,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeError(f"defects4j checkout failed for {instance.instance_id}: {detail}")


def apply_patch(workdir: Path, patch_text: str) -> bool:
    """Apply ``patch_text`` to the checked-out working directory."""
    if not patch_text.strip():
        return False
    # Try git apply first because Defects4J checkouts are git repositories.
    check = subprocess.run(
        ["git", "apply", "--check"],
        input=patch_text,
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode == 0:
        applied = subprocess.run(
            ["git", "apply"],
            input=patch_text,
            cwd=workdir,
            capture_output=True,
            text=True,
            check=False,
        )
        return applied.returncode == 0
    # Fall back to the standard patch command.
    applied = subprocess.run(
        ["patch", "-p1"],
        input=patch_text,
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    return applied.returncode == 0


def compile_bug(workdir: Path) -> bool:
    """Compile the checked-out bug."""
    result = run_defects4j_command(
        ["defects4j", "compile", "-w", str(workdir)],
        timeout=300,
    )
    return result.returncode == 0


def run_tests(workdir: Path) -> tuple[int, list[str]]:
    """Run the relevant tests and return (failing_count, failing_tests)."""
    result = run_defects4j_command(
        ["defects4j", "test", "-w", str(workdir)],
        timeout=600,
    )
    if result.returncode != 0:
        return 0, [f"test_execution_error: {(result.stderr or result.stdout).strip()[-500:]}"]

    failing: list[str] = []
    total = 0
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("failing tests:"):
            count_text = line.split(":", 1)[1].strip()
            try:
                total = int(count_text)
            except ValueError:
                total = 0
        elif "::" in line:
            test_name = line.lstrip("-\t ").split()[0]
            failing.append(test_name)

    if total == 0:
        total = len(failing)
    return total, failing


def evaluate_instance(
    instance: Defects4JInstance,
    patch_text: str,
    workdir_root: Path,
) -> Defects4JResult:
    """Evaluate a single Defects4J instance against a patch."""
    workdir = workdir_root / instance.instance_id
    try:
        checkout_bug(instance, workdir, version="b")
        patch_applied = apply_patch(workdir, patch_text)
        if not patch_applied:
            return Defects4JResult(
                instance.instance_id,
                resolved=False,
                patch_applied=False,
                compiled=False,
                test_count=0,
                failing_tests=[],
                error="patch_not_applied",
            )
        compiled = compile_bug(workdir)
        if not compiled:
            return Defects4JResult(
                instance.instance_id,
                resolved=False,
                patch_applied=True,
                compiled=False,
                test_count=0,
                failing_tests=[],
                error="compile_failed",
            )
        failing_count, failing_tests = run_tests(workdir)
        resolved = failing_count == 0 and len(failing_tests) == 0
        return Defects4JResult(
            instance.instance_id,
            resolved=resolved,
            patch_applied=True,
            compiled=True,
            test_count=failing_count,
            failing_tests=failing_tests,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - execution boundary
        return Defects4JResult(
            instance.instance_id,
            resolved=False,
            patch_applied=False,
            compiled=False,
            test_count=0,
            failing_tests=[],
            error=f"{exc.__class__.__name__}: {exc}",
        )


def result_to_dict(result: Defects4JResult) -> ConfigMap:
    return {
        "instance_id": result.instance_id,
        "resolved": result.resolved,
        "patch_applied": result.patch_applied,
        "compiled": result.compiled,
        "test_count": result.test_count,
        "failing_tests": result.failing_tests,
        "error": result.error,
    }


def evaluate_predictions(
    predictions_path: Path,
    run_id: str,
    max_workers: int,
    workdir_root: Path | None = None,
) -> ConfigMap:
    """Evaluate all Defects4J predictions locally without Docker.

    Returns a status dictionary compatible with the SWE-bench harness wrapper.
    """
    if workdir_root is None:
        workdir_root = Path("outputs") / "defects4j_work" / run_id
    workdir_root.mkdir(parents=True, exist_ok=True)

    patch_by_instance: dict[str, str] = {}
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row_object = cast(object, json.loads(line))
        except json.JSONDecodeError:
            continue
        if not isinstance(row_object, dict):
            continue
        row = cast(dict[str, object], row_object)
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str):
            continue
        if parse_instance_id(instance_id) is None:
            continue
        patch = row.get("model_patch", "")
        patch_by_instance[instance_id] = patch if isinstance(patch, str) else ""

    if not patch_by_instance:
        return {
            "status": "blocked",
            "official_harness_executed": False,
            "defects4j_harness_executed": False,
            "fallback_reason": "no_defects4j_predictions",
            "resolved": 0,
            "total": 0,
            "resolved_rate": 0.0,
            "report_dir": str(Path("logs") / "run_evaluation" / run_id),
            "stderr_tail": "",
            "stdout_tail": "No Defects4J predictions found",
        }

    results: list[Defects4JResult] = []
    with ThreadPoolExecutor(max_workers=max(max_workers, 1)) as executor:
        future_to_instance: dict[Future[Defects4JResult], str] = {}
        for instance_id, patch_text in patch_by_instance.items():
            parsed = parse_instance_id(instance_id)
            assert parsed is not None
            future = executor.submit(evaluate_instance, parsed, patch_text, workdir_root)
            future_to_instance[future] = instance_id

        for future in as_completed(future_to_instance):
            results.append(future.result())

    resolved = sum(1 for r in results if r.resolved)
    total = len(results)
    errored = sum(1 for r in results if r.error is not None)
    resolved_rate = round(resolved / total, 4) if total > 0 else 0.0

    report_dir = Path("logs") / "run_evaluation" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report: ConfigMap = {
        "resolved_instances": resolved,
        "total_instances": total,
        "resolved_rate": resolved_rate,
        "results": [result_to_dict(r) for r in results],
    }
    write_json_atomic(report_dir / "report.json", report)

    # If every instance failed at the infrastructure level (checkout/compile/etc.),
    # treat the run as fallback rather than completed so strict mode exits nonzero.
    status = "completed" if total > 0 and errored < total else "fallback"
    return {
        "status": status,
        "official_harness_executed": False,
        "defects4j_harness_executed": True,
        "fallback_reason": None,
        "resolved": resolved,
        "total": total,
        "resolved_rate": resolved_rate,
        "report_dir": str(report_dir),
        "stderr_tail": "",
        "stdout_tail": f"Defects4J evaluation: {resolved}/{total} resolved",
    }
