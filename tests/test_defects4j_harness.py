from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from repair_agent.env import defects4j_harness, harness as harness_module
from repair_agent.env.harness import main as harness_main


# --------------------------------------------------------------------------- #
# Defects4J id parsing
# --------------------------------------------------------------------------- #
def test_parse_instance_id_accepts_supported_projects():
    parsed = defects4j_harness.parse_instance_id("Lang_1")
    assert parsed is not None
    assert parsed.project == "Lang"
    assert parsed.bug_id == 1
    assert parsed.instance_id == "Lang_1"


def test_parse_instance_id_rejects_unsupported_project():
    assert defects4j_harness.parse_instance_id("Unknown_1") is None


def test_parse_instance_id_rejects_malformed_ids():
    assert defects4j_harness.parse_instance_id("Lang_1_2") is None
    assert defects4j_harness.parse_instance_id("lang_1") is None
    assert defects4j_harness.parse_instance_id("Lang") is None
    assert defects4j_harness.parse_instance_id("1_Lang") is None


# --------------------------------------------------------------------------- #
# Prediction scanning
# --------------------------------------------------------------------------- #
def test_defects4j_ids_in_predictions_extracts_unique_instances(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        '{"instance_id":"Lang_1","model_patch":"diff1"}',
        '{"instance_id":"Math_2","model_patch":"diff2"}',
        '{"instance_id":"Lang_1","model_patch":"diff3"}',
        '{"instance_id":"django__django-11099","model_patch":"diff4"}',
    ]
    _ = predictions.write_text("\n".join(rows) + "\n", encoding="utf-8")

    instances = defects4j_harness.defects4j_ids_in_predictions(predictions)
    assert len(instances) == 2
    assert instances[0].instance_id == "Lang_1"
    assert instances[1].instance_id == "Math_2"


def test_run_tests_parses_passing_output():
    def fake_run(
        args: list[str],
        *,
        capture_output: bool = True,
        text: bool = True,
        timeout: int = 600,
        check: bool = False,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        _ = (args, capture_output, text, timeout, check, env)
        return subprocess.CompletedProcess(args, 0, "Failing tests: 0", "")

    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.setattr(defects4j_harness, "run_defects4j_command", fake_run)
        count, failing = defects4j_harness.run_tests(Path("/tmp/dummy"))
    assert count == 0
    assert failing == []


def test_run_tests_parses_failing_output():
    def fake_run(
        args: list[str],
        *,
        capture_output: bool = True,
        text: bool = True,
        timeout: int = 600,
        check: bool = False,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        _ = (args, capture_output, text, timeout, check, env)
        stdout = """Failing tests: 2
  - org.foo.Bar::testOne
  - org.foo.Baz::testTwo"""
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.setattr(defects4j_harness, "run_defects4j_command", fake_run)
        count, failing = defects4j_harness.run_tests(Path("/tmp/dummy"))
    assert count == 2
    assert failing == ["org.foo.Bar::testOne", "org.foo.Baz::testTwo"]


def test_run_tests_reports_error_on_nonzero_return():
    def fake_run(
        args: list[str],
        *,
        capture_output: bool = True,
        text: bool = True,
        timeout: int = 600,
        check: bool = False,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        _ = (args, capture_output, text, timeout, check, env)
        return subprocess.CompletedProcess(args, 1, "", "execution failed")

    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.setattr(defects4j_harness, "run_defects4j_command", fake_run)
        count, failing = defects4j_harness.run_tests(Path("/tmp/dummy"))
    assert count == 0
    assert len(failing) == 1
    assert "test_execution_error" in failing[0]


def test_apply_patch_tries_git_first_then_patch(tmp_path: Path):
    workdir = tmp_path / "work"
    _ = workdir.mkdir()
    _ = (workdir / "file.txt").write_text("old\n", encoding="utf-8")

    # git apply requires the workdir to be a git repository.
    _ = subprocess.run(["git", "init"], cwd=workdir, check=True, capture_output=True)
    _ = subprocess.run(["git", "config", "user.email", "x@x"], cwd=workdir, check=True, capture_output=True)
    _ = subprocess.run(["git", "config", "user.name", "x"], cwd=workdir, check=True, capture_output=True)
    _ = subprocess.run(["git", "add", "."], cwd=workdir, check=True, capture_output=True)
    _ = subprocess.run(["git", "commit", "-m", "init"], cwd=workdir, check=True, capture_output=True)

    patch_text = "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n"
    assert defects4j_harness.apply_patch(workdir, patch_text) is True
    assert (workdir / "file.txt").read_text(encoding="utf-8").strip() == "new"


def test_apply_patch_returns_false_for_empty_patch(tmp_path: Path):
    assert defects4j_harness.apply_patch(tmp_path / "work", "   ") is False


# --------------------------------------------------------------------------- #
# Local evaluation with mocked Defects4J commands
# --------------------------------------------------------------------------- #
def test_evaluate_predictions_reports_resolved_when_tests_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        '{"instance_id":"Lang_1","model_patch":"diff"}\n',
        encoding="utf-8",
    )
    workdir = tmp_path / "work"

    call_log: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        capture_output: bool = True,
        text: bool = True,
        timeout: int = 300,
        check: bool = False,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        _ = (capture_output, text, timeout, check, env)
        call_log.append(args)
        cmd = " ".join(args)
        if "checkout" in cmd:
            return subprocess.CompletedProcess(args, 0, "checked out", "")
        if "compile" in cmd:
            return subprocess.CompletedProcess(args, 0, "compiled", "")
        if "test" in cmd:
            return subprocess.CompletedProcess(args, 0, "Failing tests: 0", "")
        return subprocess.CompletedProcess(args, 1, "", "unexpected command")

    def fake_apply_patch(workdir: Path, patch_text: str) -> bool:
        _ = (workdir, patch_text)
        return True

    def empty_iterdir(_self: Path) -> Iterator[Path]:
        return iter([])

    monkeypatch.setattr(defects4j_harness, "run_defects4j_command", fake_run)
    monkeypatch.setattr(defects4j_harness, "apply_patch", fake_apply_patch)
    monkeypatch.setattr(Path, "iterdir", empty_iterdir)

    result = defects4j_harness.evaluate_predictions(
        predictions_path=predictions,
        run_id="unit_d4j",
        max_workers=1,
        workdir_root=workdir,
    )

    assert result["defects4j_harness_executed"] is True
    assert result["resolved"] == 1
    assert result["total"] == 1
    assert result["resolved_rate"] == 1.0
    assert result["status"] == "completed"
    report_path = Path(cast(str, result["report_dir"])) / "report.json"
    report = cast(dict[str, object], json.loads(report_path.read_text(encoding="utf-8")))
    assert report["resolved_instances"] == 1


def test_evaluate_predictions_reports_unresolved_when_tests_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        '{"instance_id":"Lang_1","model_patch":"diff"}\n',
        encoding="utf-8",
    )
    workdir = tmp_path / "work"

    def fake_run(
        args: list[str],
        *,
        capture_output: bool = True,
        text: bool = True,
        timeout: int = 300,
        check: bool = False,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        _ = (capture_output, text, timeout, check, env)
        cmd = " ".join(args)
        if "checkout" in cmd:
            return subprocess.CompletedProcess(args, 0, "checked out", "")
        if "compile" in cmd:
            return subprocess.CompletedProcess(args, 0, "compiled", "")
        if "test" in cmd:
            return subprocess.CompletedProcess(
                args,
                0,
                "Failing tests: 1\n  - org.foo.Bar::testBaz",
                "",
            )
        return subprocess.CompletedProcess(args, 1, "", "unexpected command")

    def always_apply(workdir: Path, patch_text: str) -> bool:
        _ = (workdir, patch_text)
        return True

    def empty_iterdir(_self: Path) -> Iterator[Path]:
        return iter([])

    monkeypatch.setattr(defects4j_harness, "run_defects4j_command", fake_run)
    monkeypatch.setattr(defects4j_harness, "apply_patch", always_apply)
    monkeypatch.setattr(Path, "iterdir", empty_iterdir)

    result = defects4j_harness.evaluate_predictions(
        predictions_path=predictions,
        run_id="unit_d4j_fail",
        max_workers=1,
        workdir_root=workdir,
    )

    assert result["resolved"] == 0
    assert result["total"] == 1


# --------------------------------------------------------------------------- #
# Harness fallback integration
# --------------------------------------------------------------------------- #
def test_harness_falls_back_to_defects4j_when_docker_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        '{"instance_id":"Lang_1","model_patch":"diff"}\n',
        encoding="utf-8",
    )
    status_path = tmp_path / "harness_status.json"

    def blocked(args: object) -> str:
        _ = args
        return "docker_daemon_unavailable:unshare"

    def available() -> bool:
        return True

    def evaluate(**_kwargs: object) -> dict[str, object]:
        return {
            "status": "completed",
            "official_harness_executed": False,
            "defects4j_harness_executed": True,
            "fallback_reason": None,
            "resolved": 1,
            "total": 1,
            "resolved_rate": 1.0,
            "report_dir": str(tmp_path / "report"),
            "stderr_tail": "",
            "stdout_tail": "ok",
        }

    monkeypatch.setattr(harness_module, "_blocked_reason", blocked)
    monkeypatch.setattr(defects4j_harness, "is_available", available)
    monkeypatch.setattr(defects4j_harness, "evaluate_predictions", evaluate)

    result = harness_main(
        [
            "--predictions",
            str(predictions),
            "--run-id",
            "d4j_fallback",
            "--max-workers",
            "1",
            "--strict-official",
            "--status-out",
            str(status_path),
        ]
    )
    status = cast(dict[str, object], json.loads(status_path.read_text(encoding="utf-8")))

    assert result == 0
    assert status["status"] == "completed"
    assert status["defects4j_harness_executed"] is True
    assert status["fallback_reason"] == "docker_daemon_unavailable:unshare"


def test_harness_falls_back_to_defects4j_when_official_harness_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        '{"instance_id":"Lang_1","model_patch":"diff"}\n',
        encoding="utf-8",
    )
    status_path = tmp_path / "harness_status.json"

    def not_blocked(args: object) -> str | None:
        _ = args
        return None

    def failing_official(command: list[str], timeout_seconds: int) -> dict[str, object]:
        _ = (command, timeout_seconds)
        return {
            "official_harness_executed": True,
            "returncode": 1,
            "status": "fallback",
            "stderr_tail": "unshare: operation not permitted",
            "stdout_tail": "",
            "fallback_reason": "official_harness_returned_1",
        }

    def evaluate(**_kwargs: object) -> dict[str, object]:
        return {
            "status": "completed",
            "official_harness_executed": False,
            "defects4j_harness_executed": True,
            "fallback_reason": None,
            "resolved": 1,
            "total": 1,
            "resolved_rate": 1.0,
            "report_dir": str(tmp_path / "report"),
            "stderr_tail": "",
            "stdout_tail": "ok",
        }

    monkeypatch.setattr(harness_module, "_blocked_reason", not_blocked)
    monkeypatch.setattr(harness_module, "_run_official", failing_official)
    monkeypatch.setattr(defects4j_harness, "is_available", lambda: True)
    monkeypatch.setattr(defects4j_harness, "evaluate_predictions", evaluate)

    result = harness_main(
        [
            "--predictions",
            str(predictions),
            "--run-id",
            "d4j_after_official_fail",
            "--max-workers",
            "1",
            "--strict-official",
            "--status-out",
            str(status_path),
        ]
    )
    status = cast(dict[str, object], json.loads(status_path.read_text(encoding="utf-8")))

    assert result == 0
    assert status["status"] == "completed"
    assert status["defects4j_harness_executed"] is True
    assert "official_harness_failed" in cast(str, status["fallback_reason"])


def test_harness_still_blocked_when_defects4j_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        '{"instance_id":"Lang_1","model_patch":"diff"}\n',
        encoding="utf-8",
    )
    status_path = tmp_path / "harness_status.json"

    def blocked(args: object) -> str:
        _ = args
        return "docker_daemon_unavailable:unshare"

    monkeypatch.setattr(harness_module, "_blocked_reason", blocked)
    monkeypatch.setattr(defects4j_harness, "is_available", lambda: False)

    result = harness_main(
        [
            "--predictions",
            str(predictions),
            "--run-id",
            "d4j_unavailable",
            "--max-workers",
            "1",
            "--strict-official",
            "--status-out",
            str(status_path),
        ]
    )
    status = cast(dict[str, object], json.loads(status_path.read_text(encoding="utf-8")))

    assert result == 1
    assert status["status"] == "blocked"
    assert status["blocked_reason"] == "docker_daemon_unavailable:unshare"


def test_skip_defects4j_fallback_flag_returns_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        '{"instance_id":"Lang_1","model_patch":"diff"}\n',
        encoding="utf-8",
    )
    status_path = tmp_path / "harness_status.json"

    def blocked(args: object) -> str:
        _ = args
        return "docker_daemon_unavailable:unshare"

    # If this were called, it would succeed; the flag should prevent the call.
    def available() -> bool:
        return True

    monkeypatch.setattr(harness_module, "_blocked_reason", blocked)
    monkeypatch.setattr(defects4j_harness, "is_available", available)

    result = harness_main(
        [
            "--predictions",
            str(predictions),
            "--run-id",
            "d4j_skipped",
            "--max-workers",
            "1",
            "--strict-official",
            "--skip-defects4j-fallback",
            "--status-out",
            str(status_path),
        ]
    )
    status = cast(dict[str, object], json.loads(status_path.read_text(encoding="utf-8")))

    assert result == 1
    assert status["status"] == "blocked"


def test_defects4j_id_passes_instance_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        '{"instance_id":"Lang_1","model_patch":"diff"}\n',
        encoding="utf-8",
    )
    status_path = tmp_path / "harness_status.json"

    def blocked(args: object) -> str:
        _ = args
        return "docker_daemon_unavailable:unshare"

    monkeypatch.setattr(harness_module, "_blocked_reason", blocked)
    monkeypatch.setattr(defects4j_harness, "is_available", lambda: False)

    result = harness_main(
        [
            "--predictions",
            str(predictions),
            "--run-id",
            "d4j_id_validation",
            "--max-workers",
            "1",
            "--status-out",
            str(status_path),
        ]
    )
    status = cast(dict[str, object], json.loads(status_path.read_text(encoding="utf-8")))

    assert result == 0  # non-strict mode returns 0 when blocked
    assert status["status"] == "blocked"
    assert status["total"] == 1
