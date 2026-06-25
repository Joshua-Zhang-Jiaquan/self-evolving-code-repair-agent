"""Reproducible code repair environment with tool actions and rewards."""

from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .tasks import RepairTask


@dataclass
class ToolEvent:
    action: str
    ok: bool
    detail: str
    cost: float = 1.0


@dataclass
class EnvironmentState:
    issue: str
    diff: str = ""
    observed_files: Dict[str, str] = field(default_factory=dict)
    tool_events: List[ToolEvent] = field(default_factory=list)
    last_test_output: str = ""
    max_steps: int = 30
    max_test_runs: int = 8
    test_runs: int = 0
    unsafe_edit: bool = False
    test_deletion: bool = False
    relevant_file_found: bool = False


@dataclass
class RunResult:
    returncode: int
    output: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


@dataclass
class RewardBreakdown:
    pass_visible_tests: int
    pass_hidden_tests: int
    relevant_file_found: int
    num_tool_calls: int
    num_test_runs: int
    unsafe_edit: int
    test_deletion: int
    reward: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "pass_visible_tests": self.pass_visible_tests,
            "pass_hidden_tests": self.pass_hidden_tests,
            "relevant_file_found": self.relevant_file_found,
            "num_tool_calls": self.num_tool_calls,
            "num_test_runs": self.num_test_runs,
            "unsafe_edit": self.unsafe_edit,
            "test_deletion": self.test_deletion,
            "reward": self.reward,
        }


class CodeRepairEnvironment:
    """A small POMDP-like environment for code repair tasks.

    The hidden state is the full repository plus hidden tests. The agent observes
    issue text, selected file snippets, visible test output, and diffs through
    explicit tool calls.
    """

    def __init__(
        self,
        task: RepairTask,
        max_steps: int = 30,
        max_test_runs: int = 8,
        test_timeout: Optional[int] = None,
    ):
        self.task = task
        self.root_obj: Optional[tempfile.TemporaryDirectory[str]] = None
        self.root: Optional[Path] = None
        self.original_files: Dict[str, str] = {}
        self.snapshots: List[Dict[str, str]] = []
        self.test_timeout = int(test_timeout or os.environ.get("CODE_REPAIR_TEST_TIMEOUT", "60"))
        self.state = EnvironmentState(issue=task.issue, max_steps=max_steps, max_test_runs=max_test_runs)

    def __enter__(self) -> "CodeRepairEnvironment":
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self) -> EnvironmentState:
        self.close()
        self.root_obj = tempfile.TemporaryDirectory(prefix=f"repair_{self.task.task_id}_")
        self.root = Path(self.root_obj.name)
        for rel_path, content in self.task.files.items():
            path = self.root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.original_files = dict(self.task.files)
        self.snapshots = []
        self.state = EnvironmentState(
            issue=self.task.issue,
            max_steps=self.state.max_steps,
            max_test_runs=self.state.max_test_runs,
        )
        return self.state

    def close(self) -> None:
        if self.root_obj is not None:
            self.root_obj.cleanup()
            self.root_obj = None
            self.root = None

    def search(self, pattern: str, max_matches: int = 20) -> str:
        matches: List[str] = []
        needle = pattern.lower()
        for path in self._repo_files():
            if path.parts[0] == "tests":
                continue
            text = self._read(path)
            for line_no, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    matches.append(f"{path}:{line_no}: {line.strip()}")
                    self.state.relevant_file_found = True
                    if len(matches) >= max_matches:
                        return self._record("search", True, "\n".join(matches))
        detail = "\n".join(matches) if matches else "no matches"
        return self._record("search", bool(matches), detail)

    def read_file(self, rel_path: str, start: int = 1, end: Optional[int] = None) -> str:
        path = self._safe_path(rel_path)
        if path is None:
            self.state.unsafe_edit = True
            return self._record("read_file", False, f"{rel_path}: unsafe path is blocked")
        try:
            text = self._read(path)
        except FileNotFoundError:
            return self._record("read_file", False, f"{rel_path} not found")
        lines = text.splitlines()
        if end is None:
            end = len(lines)
        snippet = "\n".join(f"{idx}: {line}" for idx, line in enumerate(lines[start - 1 : end], start=start))
        self.state.observed_files[rel_path] = snippet
        if not path.parts[0] == "tests":
            self.state.relevant_file_found = True
        return self._record("read_file", True, snippet)

    def inspect_test(self) -> RunResult:
        result = self._run(self.task.visible_test_command)
        self._record("inspect_test", result.passed, result.output)
        return result

    def run_tests(self, scope: str = "visible") -> RunResult:
        command = self.task.hidden_test_command if scope == "hidden" else self.task.visible_test_command
        result = self._run(command)
        self._record(f"run_tests:{scope}", result.passed, result.output)
        return result

    def edit_file(self, rel_path: str, old: str, new: str) -> str:
        path = self._safe_path(rel_path)
        if path is None:
            self.state.unsafe_edit = True
            return self._record("edit_file", False, f"{rel_path}: unsafe path is blocked")
        if path.parts and path.parts[0] == "tests":
            self.state.unsafe_edit = True
            self.state.test_deletion = True
            return self._record("edit_file", False, "editing tests is blocked")
        try:
            text = self._read(path)
        except FileNotFoundError:
            return self._record("edit_file", False, f"{rel_path} not found")
        if old not in text:
            return self._record("edit_file", False, "old text not found")
        self.snapshots.append(self._current_source_snapshot())
        updated = text.replace(old, new, 1)
        self._write(path, updated)
        self.state.diff = self.diff()
        return self._record("edit_file", True, f"patched {rel_path}")

    def rollback(self) -> str:
        if not self.snapshots:
            return self._record("rollback", False, "no snapshot available")
        snapshot = self.snapshots.pop()
        for rel_path, content in snapshot.items():
            self._write(Path(rel_path), content)
        self.state.diff = self.diff()
        return self._record("rollback", True, "restored previous source snapshot")

    def final_answer(self, summary: str) -> str:
        return self._record("final_answer", True, summary, cost=0.2)

    def diff(self) -> str:
        chunks: List[str] = []
        for rel_path, original in sorted(self.original_files.items()):
            if rel_path.startswith("tests/"):
                continue
            current = self._read(Path(rel_path))
            if current == original:
                continue
            chunks.extend(
                difflib.unified_diff(
                    original.splitlines(),
                    current.splitlines(),
                    fromfile=f"a/{rel_path}",
                    tofile=f"b/{rel_path}",
                    lineterm="",
                )
            )
        return "\n".join(chunks)

    def reward(self) -> RewardBreakdown:
        visible = self._run_no_record(self.task.visible_test_command).passed
        hidden = self._run_no_record(self.task.hidden_test_command).passed
        num_tool_calls = len(self.state.tool_events)
        num_test_runs = self.state.test_runs
        value = (
            1.0 * int(visible)
            + 2.0 * int(hidden)
            + 0.2 * int(self.state.relevant_file_found)
            - 0.05 * num_tool_calls
            - 0.10 * num_test_runs
            - 1.0 * int(self.state.unsafe_edit)
            - 1.0 * int(self.state.test_deletion)
        )
        return RewardBreakdown(
            pass_visible_tests=int(visible),
            pass_hidden_tests=int(hidden),
            relevant_file_found=int(self.state.relevant_file_found),
            num_tool_calls=num_tool_calls,
            num_test_runs=num_test_runs,
            unsafe_edit=int(self.state.unsafe_edit),
            test_deletion=int(self.state.test_deletion),
            reward=round(value, 4),
        )

    def trace_json(self) -> str:
        return json.dumps(
            {
                "task_id": self.task.task_id,
                "events": [event.__dict__ for event in self.state.tool_events],
                "diff": self.state.diff,
                "reward": self.reward().as_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )

    def _record(self, action: str, ok: bool, detail: str, cost: float = 1.0) -> str:
        self.state.tool_events.append(ToolEvent(action=action, ok=ok, detail=detail[:2000], cost=cost))
        return detail

    def _run(self, command: List[str]) -> RunResult:
        if self.state.test_runs >= self.state.max_test_runs:
            result = RunResult(returncode=124, output="test budget exceeded")
        else:
            self.state.test_runs += 1
            result = self._run_no_record(command)
        self.state.last_test_output = result.output
        return result

    def _run_no_record(self, command: List[str]) -> RunResult:
        assert self.root is not None
        for cache_dir in self.root.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)
        try:
            proc = subprocess.run(
                command,
                cwd=self.root,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.test_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            return RunResult(returncode=124, output=f"{output}\ncommand timed out after {self.test_timeout}s")
        return RunResult(returncode=proc.returncode, output=proc.stdout)

    def _repo_files(self) -> List[Path]:
        assert self.root is not None
        return sorted(
            path.relative_to(self.root)
            for path in self.root.rglob("*.py")
            if "__pycache__" not in path.parts
        )

    def _current_source_snapshot(self) -> Dict[str, str]:
        return {
            str(path): self._read(path)
            for path in self._repo_files()
            if not path.parts[0] == "tests"
        }

    def _read(self, rel_path: Path) -> str:
        assert self.root is not None
        return (self.root / rel_path).read_text(encoding="utf-8")

    def _write(self, rel_path: Path, content: str) -> None:
        assert self.root is not None
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _safe_path(self, rel_path: str) -> Optional[Path]:
        path = Path(rel_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            return None
        return path
