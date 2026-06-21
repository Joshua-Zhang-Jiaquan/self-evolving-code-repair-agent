from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from repair_agent.env import defects4j_harness, harness as harness_module
from repair_agent.env import defects4j_cache
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


# --------------------------------------------------------------------------- #
# Cache module
# --------------------------------------------------------------------------- #
def test_cache_path_returns_stable_location():
    workdir_root = Path("/tmp/work")
    instance = defects4j_harness.parse_instance_id("Lang_1")
    assert instance is not None
    path = defects4j_cache.cache_path(workdir_root, instance, "b")
    assert path == workdir_root / ".d4j_cache" / "Lang_1b"


def test_ensure_cached_skips_when_sentinel_exists(tmp_path: Path):
    instance = defects4j_harness.parse_instance_id("Lang_1")
    assert instance is not None
    cached = defects4j_cache.cache_path(tmp_path, instance, "b")
    cached.mkdir(parents=True)
    (cached / ".defects4j.config").touch()

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
        return subprocess.CompletedProcess(args, 1, "", "should not be called")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(defects4j_harness, "run_defects4j_command", fake_run)

    result = defects4j_cache.ensure_cached(
        instance, "b", workdir_root=tmp_path
    )
    assert result == cached
    assert len(call_log) == 0


def test_ensure_cached_runs_checkout_when_cache_missing(tmp_path: Path):
    instance = defects4j_harness.parse_instance_id("Lang_1")
    assert instance is not None
    cached = defects4j_cache.cache_path(tmp_path, instance, "b")

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
        return subprocess.CompletedProcess(args, 0, "checked out", "")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(defects4j_harness, "run_defects4j_command", fake_run)

    result = defects4j_cache.ensure_cached(
        instance, "b", workdir_root=tmp_path
    )
    assert result == cached
    assert len(call_log) == 1
    assert "checkout" in call_log[0]
    assert (cached / ".defects4j.config").exists()


def test_ensure_cached_raises_on_checkout_failure(tmp_path: Path):
    instance = defects4j_harness.parse_instance_id("Lang_1")
    assert instance is not None

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
        return subprocess.CompletedProcess(args, 1, "", "checkout error")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(defects4j_harness, "run_defects4j_command", fake_run)

    with pytest.raises(RuntimeError, match="checkout.*failed"):
        defects4j_cache.ensure_cached(instance, "b", workdir_root=tmp_path)


def test_materialize_workdir_copies_directory(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "README").write_text("hello", encoding="utf-8")
    (cache_dir / "src").mkdir()
    (cache_dir / "src" / "Foo.java").write_text("class Foo {}", encoding="utf-8")

    workdir = tmp_path / "work"
    result = defects4j_cache.materialize_workdir(cache_dir, workdir)
    assert result == workdir
    assert workdir.is_dir()
    assert (workdir / "README").read_text(encoding="utf-8") == "hello"
    assert (workdir / "src" / "Foo.java").read_text(encoding="utf-8") == "class Foo {}"


def test_materialize_workdir_overwrites_existing(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "file.txt").write_text("cached", encoding="utf-8")

    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "file.txt").write_text("stale", encoding="utf-8")

    defects4j_cache.materialize_workdir(cache_dir, workdir)
    assert (workdir / "file.txt").read_text(encoding="utf-8") == "cached"


def test_materialize_workdir_falls_back_to_copytree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When cp and git clone both fail, copytree should handle it."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "data.txt").write_text("fallback", encoding="utf-8")

    def fake_subprocess_run(args, **kwargs):
        _ = kwargs
        raise FileNotFoundError("command not found")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    workdir = tmp_path / "work"
    result = defects4j_cache.materialize_workdir(cache_dir, workdir)
    assert result == workdir
    assert (workdir / "data.txt").read_text(encoding="utf-8") == "fallback"


def test_reset_workdir_runs_git_commands(tmp_path: Path):
    call_log: list[list[str]] = []

    def fake_run(
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        call_log.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(subprocess, "run", fake_run)

    workdir = tmp_path / "work"
    workdir.mkdir()
    assert defects4j_cache.reset_workdir(workdir) is True
    assert call_log[0] == ["git", "checkout", "--", "."]
    assert call_log[1] == ["git", "clean", "-fdq"]


def test_reset_workdir_returns_false_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_run(
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert defects4j_cache.reset_workdir(tmp_path / "work") is False


def test_prewarm_deduplicates_instances(tmp_path: Path):
    instance_a = defects4j_harness.parse_instance_id("Lang_1")
    instance_b = defects4j_harness.parse_instance_id("Math_5")
    assert instance_a is not None and instance_b is not None

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
        return subprocess.CompletedProcess(args, 0, "checked out", "")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(defects4j_harness, "run_defects4j_command", fake_run)

    # Lang_1 appears twice, Math_5 once → 2 unique checkouts.
    result = defects4j_cache.prewarm(
        [instance_a, instance_b, instance_a], workdir_root=tmp_path
    )
    assert len(result) == 2
    assert "Lang_1b" in result
    assert "Math_5b" in result
    assert len(call_log) == 2


def test_prewarm_creates_sentinels(tmp_path: Path):
    instance = defects4j_harness.parse_instance_id("Lang_1")
    assert instance is not None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        defects4j_harness,
        "run_defects4j_command",
        lambda args, **kw: subprocess.CompletedProcess(args, 0, "ok", ""),
    )

    _ = defects4j_cache.prewarm([instance], workdir_root=tmp_path)
    cached = defects4j_cache.cache_path(tmp_path, instance, "b")
    assert (cached / ".defects4j.config").exists()


# --------------------------------------------------------------------------- #
# run_trigger_tests
# --------------------------------------------------------------------------- #
def test_run_trigger_tests_parses_passing_output():
    def fake_run(
        args: list[str],
        *,
        capture_output: bool = True,
        text: bool = True,
        timeout: int = 600,
        check: bool = False,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        _ = (capture_output, text, timeout, check, env)
        assert "-t" in args
        assert "org.foo.Bar::testOne" in args
        return subprocess.CompletedProcess(args, 0, "Failing tests: 0", "")

    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.setattr(defects4j_harness, "run_defects4j_command", fake_run)
        count, failing = defects4j_harness.run_trigger_tests(
            Path("/tmp/dummy"), ["org.foo.Bar::testOne"]
        )
    assert count == 0
    assert failing == []


def test_run_trigger_tests_parses_failing_output():
    def fake_run(
        args: list[str],
        *,
        capture_output: bool = True,
        text: bool = True,
        timeout: int = 600,
        check: bool = False,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        _ = (capture_output, text, timeout, check, env)
        stdout = "Failing tests: 2\n  - org.foo.Bar::testOne\n  - org.foo.Baz::testTwo"
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.setattr(defects4j_harness, "run_defects4j_command", fake_run)
        count, failing = defects4j_harness.run_trigger_tests(
            Path("/tmp/dummy"), ["org.foo.Bar::testOne", "org.foo.Baz::testTwo"]
        )
    assert count == 2
    assert failing == ["org.foo.Bar::testOne", "org.foo.Baz::testTwo"]


def test_run_trigger_tests_reports_error_on_nonzero_return():
    def fake_run(
        args: list[str],
        *,
        capture_output: bool = True,
        text: bool = True,
        timeout: int = 600,
        check: bool = False,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        _ = (capture_output, text, timeout, check, env)
        return subprocess.CompletedProcess(args, 1, "", "trigger test error")

    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.setattr(defects4j_harness, "run_defects4j_command", fake_run)
        count, failing = defects4j_harness.run_trigger_tests(
            Path("/tmp/dummy"), ["org.foo.Bar::testFail"]
        )
    assert count == 0
    assert len(failing) == 1
    assert "test_execution_error" in failing[0]
