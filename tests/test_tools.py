from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from repair_agent.tools import TaskWorkspace, ToolResult
from repair_agent.tools.core import BUDGET_EXCEEDED, DENIED, ERROR, MALFORMED, OK, TIMEOUT, UNSUPPORTED
from repair_agent.tools.registry import get_registry


REQUIRED_TOOLS = {
    "search",
    "read_file",
    "inspect_test",
    "edit_file",
    "run_tests",
    "rollback",
    "git_diff",
    "final_answer",
}


def make_workspace(
    tmp_path: Path,
    *,
    visible_tests: Sequence[str] = (),
    visible_failures: Mapping[str, str] | None = None,
    max_output_chars: int = 4000,
    max_test_runs: int = 3,
    test_timeout_seconds: float = 10.0,
) -> TaskWorkspace:
    return TaskWorkspace(
        checkout_root=tmp_path,
        visible_tests=visible_tests,
        visible_failures=visible_failures or {},
        max_output_chars=max_output_chars,
        max_test_runs=max_test_runs,
        test_timeout_seconds=test_timeout_seconds,
    )


def test_registry_lists_required_tools_and_json_like_schemas():
    registry = get_registry()

    assert REQUIRED_TOOLS.issubset(set(registry.list_tools()))
    schemas = registry.schemas()
    for name in REQUIRED_TOOLS:
        schema = schemas[name]
        assert schema["type"] == "object"
        assert isinstance(schema["required"], list)
        assert isinstance(schema["properties"], dict)


def test_registry_cli_list_prints_required_tools(project_root: Path):
    completed = subprocess.run(
        [sys.executable, "-m", "repair_agent.tools.registry", "--list"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert REQUIRED_TOOLS.issubset(set(completed.stdout.splitlines()))


def test_tool_result_serializes_structured_fields():
    result = ToolResult(
        tool="read_file",
        status=OK,
        output="body",
        cost={"tool_calls": 1.0},
        elapsed_seconds=0.1,
        truncated=False,
        metadata={"path": "src/app.py"},
    )

    assert result.to_dict() == {
        "tool": "read_file",
        "status": OK,
        "output": "body",
        "error": "",
        "cost": {"tool_calls": 1.0},
        "elapsed_seconds": 0.1,
        "truncated": False,
        "metadata": {"path": "src/app.py"},
    }


def test_read_file_denies_absolute_and_traversal_outside_workspace(tmp_path: Path):
    registry = get_registry()
    workspace = make_workspace(tmp_path)
    outside = tmp_path.parent / "outside_secret.txt"
    _ = outside.write_text("secret", encoding="utf-8")

    absolute = registry.execute("read_file", workspace, {"path": "/etc/passwd"})
    traversal = registry.execute("read_file", workspace, {"path": "../outside_secret.txt"})

    assert absolute.status == DENIED
    assert traversal.status == DENIED
    assert "secret" not in traversal.output


def test_read_file_truncates_large_output_deterministically(tmp_path: Path):
    registry = get_registry()
    source = tmp_path / "src.py"
    _ = source.write_text("\n".join(f"line-{index:03d}" for index in range(100)), encoding="utf-8")
    workspace = make_workspace(tmp_path, max_output_chars=160)

    result = registry.execute("read_file", workspace, {"path": "src.py"})

    assert result.status == OK
    assert result.truncated
    assert result.output.endswith("...[truncated]")
    assert len(result.output) <= 160


def test_search_is_bounded_and_skips_protected_directories(tmp_path: Path):
    registry = get_registry()
    (tmp_path / "pkg").mkdir()
    _ = (tmp_path / "pkg" / "app.py").write_text("needle = 'visible'\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    _ = (tmp_path / ".venv" / "secret.py").write_text("needle = 'hidden'\n", encoding="utf-8")
    workspace = make_workspace(tmp_path, max_output_chars=500)

    result = registry.execute("search", workspace, {"query": "needle", "max_matches": 5})

    assert result.status == OK
    assert "pkg/app.py:1" in result.output
    assert ".venv" not in result.output
    assert result.metadata["matches"] == 1


def test_inspect_test_only_reads_visible_tests_or_named_failures(tmp_path: Path):
    registry = get_registry()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _ = (tests_dir / "test_visible.py").write_text("def test_ok():\n    assert 'visible'.upper() == 'VISIBLE'\n", encoding="utf-8")
    _ = (tests_dir / "test_hidden.py").write_text("def test_hidden():\n    assert False\n", encoding="utf-8")
    workspace = make_workspace(
        tmp_path,
        visible_tests=("tests/test_visible.py",),
        visible_failures={"fail-1": "AssertionError: visible failure"},
    )

    visible_file = registry.execute("inspect_test", workspace, {"target": "tests/test_visible.py"})
    visible_failure = registry.execute("inspect_test", workspace, {"target": "fail-1"})
    hidden_file = registry.execute("inspect_test", workspace, {"target": "tests/test_hidden.py"})

    assert visible_file.status == OK
    assert visible_failure.status == OK
    assert "visible failure" in visible_failure.output
    assert hidden_file.status == DENIED


def test_edit_file_applies_bounded_line_edit_and_rollback_restores_last_edit(tmp_path: Path):
    registry = get_registry()
    source = tmp_path / "pkg.py"
    _ = source.write_text("a = 1\nb = 2\n", encoding="utf-8")
    workspace = make_workspace(tmp_path)

    edit = registry.execute(
        "edit_file",
        workspace,
        {"path": "pkg.py", "replacement": "b = 3\n", "start_line": 2, "end_line": 2},
    )
    diff = registry.execute("git_diff", workspace, {})
    rollback = registry.execute("rollback", workspace, {"reason": "failed tests"})

    assert edit.status == OK
    assert source.read_text(encoding="utf-8") == "a = 1\nb = 2\n"
    assert diff.status == OK
    assert "-b = 2" in diff.output
    assert "+b = 3" in diff.output
    assert rollback.status == OK


def test_rollback_without_history_fails_gracefully(tmp_path: Path):
    registry = get_registry()
    workspace = make_workspace(tmp_path)

    result = registry.execute("rollback", workspace, {})

    assert result.status == ERROR
    assert "no edit history" in result.error


def test_edit_file_denies_outside_paths_and_test_modifications(tmp_path: Path):
    registry = get_registry()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _ = (tests_dir / "test_app.py").write_text("def test_app():\n    assert 2 * 3 == 6\n", encoding="utf-8")
    outside = tmp_path.parent / "outside_edit.py"
    workspace = make_workspace(tmp_path)

    outside_result = registry.execute("edit_file", workspace, {"path": str(outside), "replacement": "x = 1\n"})
    test_result = registry.execute("edit_file", workspace, {"path": "tests/test_app.py", "replacement": ""})

    assert outside_result.status == DENIED
    assert test_result.status == DENIED
    assert (tests_dir / "test_app.py").read_text(encoding="utf-8") == "def test_app():\n    assert 2 * 3 == 6\n"


def test_git_diff_without_git_repo_returns_controlled_unsupported_result(tmp_path: Path):
    registry = get_registry()
    workspace = make_workspace(tmp_path)

    result = registry.execute("git_diff", workspace, {})

    assert result.status == UNSUPPORTED
    assert result.metadata["git_repo"] is False


def test_run_tests_respects_budget_and_visible_test_allowlist(tmp_path: Path):
    registry = get_registry()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _ = (tests_dir / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    _ = (tests_dir / "test_hidden.py").write_text("def test_hidden():\n    assert 'hidden'.endswith('den')\n", encoding="utf-8")
    workspace = make_workspace(tmp_path, visible_tests=("tests/test_ok.py",), max_test_runs=1, test_timeout_seconds=5.0)

    denied = registry.execute("run_tests", workspace, {"target": "tests/test_hidden.py"})
    first = registry.execute("run_tests", workspace, {"target": "tests/test_ok.py"})
    second = registry.execute("run_tests", workspace, {"target": "tests/test_ok.py"})

    assert denied.status == DENIED
    assert first.status == OK
    assert second.status == BUDGET_EXCEEDED
    assert workspace.test_run_count == 1


def test_run_tests_returns_timeout_status(tmp_path: Path):
    registry = get_registry()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _ = (tests_dir / "test_sleep.py").write_text(
        "import time\n\ndef test_sleep():\n    time.sleep(5)\n",
        encoding="utf-8",
    )
    workspace = make_workspace(
        tmp_path,
        visible_tests=("tests/test_sleep.py",),
        max_test_runs=2,
        test_timeout_seconds=0.2,
    )

    result = registry.execute("run_tests", workspace, {"target": "tests/test_sleep.py"})

    assert result.status == TIMEOUT
    assert result.cost["test_runs"] == 1.0


def test_run_tests_denies_network_or_privileged_commands(tmp_path: Path):
    registry = get_registry()
    workspace = make_workspace(tmp_path)

    result = registry.execute("run_tests", workspace, {"target": "curl https://example.com"})

    assert result.status == DENIED


def test_malformed_inputs_return_structured_errors(tmp_path: Path):
    registry = get_registry()
    workspace = make_workspace(tmp_path)

    missing = registry.execute("read_file", workspace, {})
    unknown = registry.execute("read_file", workspace, {"path": "x.py", "extra": "nope"})
    bad_edit = registry.execute("edit_file", workspace, {"path": "x.py", "replacement": 123})

    assert missing.status == MALFORMED
    assert unknown.status == MALFORMED
    assert bad_edit.status == MALFORMED


def test_final_answer_validates_text_without_side_effects(tmp_path: Path):
    registry = get_registry()
    workspace = make_workspace(tmp_path)

    result = registry.execute("final_answer", workspace, {"answer": "Fixed parser guard."})
    empty = registry.execute("final_answer", workspace, {"answer": "   "})

    assert result.status == OK
    assert result.output == "Fixed parser guard."
    assert empty.status == MALFORMED
    assert list(tmp_path.iterdir()) == []
