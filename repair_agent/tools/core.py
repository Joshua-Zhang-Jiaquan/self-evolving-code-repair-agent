from __future__ import annotations

import difflib
import hashlib
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast


JsonMap = dict[str, object]
Args = Mapping[str, object]

OK = "ok"
DENIED = "denied"
ERROR = "error"
TIMEOUT = "timeout"
BUDGET_EXCEEDED = "budget_exceeded"
MALFORMED = "malformed"
UNSUPPORTED = "unsupported"

DENIED_READ_COMPONENTS = frozenset(
    {".git", ".hg", ".svn", ".venv", "venv", "env", "homework1", "homework2", "homework3", "homework4", "submissions"}
)
DENIED_SEARCH_DIRS = DENIED_READ_COMPONENTS | frozenset({"__pycache__", ".pytest_cache"})
PROTECTED_EDIT_COMPONENTS = frozenset(
    {"tests", "test", "benchmarks", "benchmark", ".omo", "configs", ".github"}
)
TEXT_FILE_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".txt",
        ".md",
        ".rst",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".sh",
        ".sql",
        ".html",
        ".css",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
    }
)


@dataclass(frozen=True)
class ToolResult:
    tool: str
    status: str
    output: str = ""
    error: str = ""
    cost: Mapping[str, float] = field(default_factory=lambda: {})
    elapsed_seconds: float = 0.0
    truncated: bool = False
    metadata: Mapping[str, object] = field(default_factory=lambda: {})

    def to_dict(self) -> JsonMap:
        return {
            "tool": self.tool,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "cost": dict(self.cost),
            "elapsed_seconds": self.elapsed_seconds,
            "truncated": self.truncated,
            "metadata": dict(self.metadata),
        }


@dataclass
class EditRecord:
    path: Path
    relative_path: str
    previous_text: str
    previous_existed: bool
    new_hash: str


@dataclass
class TaskWorkspace:
    checkout_root: str | Path
    visible_tests: Sequence[str] = ()
    visible_failures: Mapping[str, str] = field(default_factory=lambda: {})
    max_output_chars: int = 4000
    timeout_seconds: float = 10.0
    test_timeout_seconds: float = 10.0
    max_test_runs: int = 3
    max_edit_chars: int = 100_000
    edit_history: list[EditRecord] = field(default_factory=list)
    test_run_count: int = 0

    def __post_init__(self) -> None:
        root = Path(self.checkout_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"checkout_root must be an existing directory: {root}")
        self.checkout_root = root
        self.visible_tests = tuple(_normalize_relative(test) for test in self.visible_tests)
        self.visible_failures = {str(name): str(text) for name, text in self.visible_failures.items()}
        if self.max_output_chars < 128:
            raise ValueError("max_output_chars must be at least 128")
        if self.timeout_seconds <= 0 or self.test_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_test_runs < 0:
            raise ValueError("max_test_runs must be non-negative")
        if self.max_edit_chars < 1:
            raise ValueError("max_edit_chars must be positive")

    @property
    def root(self) -> Path:
        return Path(self.checkout_root)

    def resolve_path(self, raw_path: object, *, allow_missing: bool = False) -> tuple[Path | None, str | None]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None, "path must be a non-empty string"
        text = raw_path.strip()
        if "\x00" in text:
            return None, "path contains NUL byte"
        raw = Path(text).expanduser()
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve(strict=False)
        if not _is_relative_to(resolved, self.root):
            return None, f"path is outside checkout root: {text}"
        relative_parts = resolved.relative_to(self.root).parts
        if any(part in DENIED_READ_COMPONENTS for part in relative_parts):
            return None, f"path is in a protected directory: {text}"
        if not allow_missing and not resolved.exists():
            return None, f"path does not exist: {text}"
        return resolved, None

    def relative(self, path: Path) -> str:
        return path.resolve(strict=False).relative_to(self.root).as_posix()

    def is_visible_test_path(self, raw_target: str) -> bool:
        test_path = _normalize_relative(raw_target.split("::", 1)[0])
        return test_path in set(self.visible_tests)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: Mapping[str, object]
    handler: Callable[[TaskWorkspace, Args], ToolResult]


class ToolRegistry:
    def __init__(self, tools: Sequence[ToolSpec]) -> None:
        self._tools: dict[str, ToolSpec] = {tool.name: tool for tool in tools}

    def list_tools(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> dict[str, Mapping[str, object]]:
        return {name: dict(spec.schema) for name, spec in self._tools.items()}

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def execute(self, name: str, workspace: TaskWorkspace, args: Args | None = None) -> ToolResult:
        if name not in self._tools:
            return ToolResult(tool=name, status=MALFORMED, error=f"unknown tool: {name}")
        payload: Args = args or {}
        spec = self._tools[name]
        validation_error = _validate_args(spec.schema, payload)
        if validation_error:
            return ToolResult(tool=name, status=MALFORMED, error=validation_error, cost={"tool_calls": 1.0})
        start = time.perf_counter()
        try:
            result = spec.handler(workspace, payload)
        except Exception as exc:
            result = ToolResult(tool=name, status=ERROR, error=f"{type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - start
        cost = {"tool_calls": 1.0, **dict(result.cost)}
        return replace(result, elapsed_seconds=elapsed, cost=cost)


def build_default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec("search", "Search bounded text inside checkout", _schema(["query"], {"query": "string", "path": "string", "max_matches": "integer"}), search),
            ToolSpec("read_file", "Read a checkout-relative file or directory", _schema(["path"], {"path": "string", "start_line": "integer", "max_lines": "integer"}), read_file),
            ToolSpec("inspect_test", "Inspect visible tests or named visible failures", _schema(["target"], {"target": "string"}), inspect_test),
            ToolSpec("edit_file", "Apply bounded source edits inside checkout", _schema(["path", "replacement"], {"path": "string", "replacement": "string", "start_line": "integer", "end_line": "integer"}), edit_file),
            ToolSpec("run_tests", "Run allowed visible pytest command within budget", _schema([], {"target": "string", "timeout_seconds": "number"}), run_tests),
            ToolSpec("rollback", "Restore the last edit applied by this workspace", _schema([], {"reason": "string"}), rollback),
            ToolSpec("git_diff", "Return git diff or controlled edit-history diff", _schema([], {}), git_diff),
            ToolSpec("final_answer", "Validate and return final repair summary", _schema(["answer"], {"answer": "string"}), final_answer),
        ]
    )


def search(workspace: TaskWorkspace, args: Args) -> ToolResult:
    query = str(args["query"])
    if not query.strip():
        return ToolResult("search", MALFORMED, error="query must be non-empty")
    max_matches = _bounded_int(args.get("max_matches", 50), default=50, minimum=1, maximum=200)
    root, error = workspace.resolve_path(args.get("path", "."))
    if error or root is None:
        return ToolResult("search", DENIED, error=error or "invalid search path")
    paths = [root] if root.is_file() else _iter_search_files(root)
    matches: list[str] = []
    lowered = query.lower()
    files_seen = 0
    for path in paths:
        if len(matches) >= max_matches:
            break
        if not path.is_file() or not _looks_text(path):
            continue
        files_seen += 1
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if lowered in line.lower():
                    rel = workspace.relative(path)
                    matches.append(f"{rel}:{line_number}: {line[:300]}")
                    if len(matches) >= max_matches:
                        break
        except OSError:
            continue
    output, truncated = _truncate("\n".join(matches), workspace.max_output_chars)
    return ToolResult(
        "search",
        OK,
        output=output,
        truncated=truncated or len(matches) >= max_matches,
        cost={"files_scanned": float(files_seen), "matches": float(len(matches))},
        metadata={"matches": len(matches), "max_matches": max_matches},
    )


def read_file(workspace: TaskWorkspace, args: Args) -> ToolResult:
    path, error = workspace.resolve_path(args.get("path"))
    if error or path is None:
        return ToolResult("read_file", DENIED, error=error or "invalid path")
    if path.is_dir():
        entries = sorted(child.name + ("/" if child.is_dir() else "") for child in path.iterdir() if child.name not in DENIED_READ_COMPONENTS)
        output, truncated = _truncate("\n".join(entries), workspace.max_output_chars)
        return ToolResult("read_file", OK, output=output, truncated=truncated, metadata={"type": "directory", "path": workspace.relative(path)})
    if not _looks_text(path):
        return ToolResult("read_file", DENIED, error="refusing to read non-text or oversized file", metadata={"path": workspace.relative(path)})
    start_line = _bounded_int(args.get("start_line", 1), default=1, minimum=1, maximum=1_000_000)
    max_lines = _bounded_int(args.get("max_lines", 400), default=400, minimum=1, maximum=5000)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = lines[start_line - 1 : start_line - 1 + max_lines]
    numbered = [f"{line_number}: {line}" for line_number, line in enumerate(selected, start=start_line)]
    output, truncated = _truncate("\n".join(numbered), workspace.max_output_chars)
    more_lines = start_line - 1 + max_lines < len(lines)
    return ToolResult(
        "read_file",
        OK,
        output=output,
        truncated=truncated or more_lines,
        cost={"bytes_read": float(path.stat().st_size)},
        metadata={"path": workspace.relative(path), "line_count": len(lines), "start_line": start_line},
    )


def inspect_test(workspace: TaskWorkspace, args: Args) -> ToolResult:
    target = str(args["target"])
    if target in workspace.visible_failures:
        output, truncated = _truncate(workspace.visible_failures[target], workspace.max_output_chars)
        return ToolResult("inspect_test", OK, output=output, truncated=truncated, metadata={"target": target, "kind": "failure"})
    if not workspace.is_visible_test_path(target):
        return ToolResult("inspect_test", DENIED, error=f"test target is not visible/allowed: {target}")
    return replace(read_file(workspace, {"path": target.split("::", 1)[0]}), tool="inspect_test")


def edit_file(workspace: TaskWorkspace, args: Args) -> ToolResult:
    replacement = args["replacement"]
    if not isinstance(replacement, str):
        return ToolResult("edit_file", MALFORMED, error="replacement must be a string")
    if len(replacement) > workspace.max_edit_chars:
        return ToolResult("edit_file", DENIED, error="replacement exceeds max_edit_chars")
    path, error = workspace.resolve_path(args.get("path"), allow_missing=True)
    if error or path is None:
        return ToolResult("edit_file", DENIED, error=error or "invalid path")
    rel = workspace.relative(path)
    if _is_protected_edit_path(path, workspace.root):
        return ToolResult("edit_file", DENIED, error=f"editing tests or benchmark metadata is blocked: {rel}")
    if not path.parent.exists():
        return ToolResult("edit_file", DENIED, error=f"parent directory does not exist: {path.parent}")
    previous_existed = path.exists()
    previous_text = path.read_text(encoding="utf-8", errors="replace") if previous_existed else ""
    if previous_existed and not _looks_text(path):
        return ToolResult("edit_file", DENIED, error="refusing to edit non-text or oversized file")
    new_text = _apply_replacement(previous_text, replacement, args)
    if isinstance(new_text, ToolResult):
        return new_text
    _ = path.write_text(new_text, encoding="utf-8")
    new_hash = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
    workspace.edit_history.append(EditRecord(path=path, relative_path=rel, previous_text=previous_text, previous_existed=previous_existed, new_hash=new_hash))
    return ToolResult(
        "edit_file",
        OK,
        output=f"edited {rel}",
        cost={"edited_chars": float(len(new_text))},
        metadata={"path": rel, "previous_existed": previous_existed, "sha256": new_hash},
    )


def run_tests(workspace: TaskWorkspace, args: Args) -> ToolResult:
    if workspace.test_run_count >= workspace.max_test_runs:
        return ToolResult("run_tests", BUDGET_EXCEEDED, error="test run budget exhausted", metadata={"test_run_count": workspace.test_run_count, "max_test_runs": workspace.max_test_runs})
    command_result = _build_pytest_command(workspace, args.get("target", ""))
    if isinstance(command_result, ToolResult):
        return command_result
    timeout_seconds = min(
        _bounded_float(args.get("timeout_seconds"), default=workspace.test_timeout_seconds),
        workspace.test_timeout_seconds,
    )
    workspace.test_run_count += 1
    try:
        completed = subprocess.run(command_result, cwd=workspace.root, text=True, capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        partial = _process_text(exc.stdout) + _process_text(exc.stderr)
        output, truncated = _truncate(partial, workspace.max_output_chars)
        return ToolResult("run_tests", TIMEOUT, output=output, error=f"test command timed out after {timeout_seconds:g}s", truncated=truncated, cost={"test_runs": 1.0}, metadata={"command": command_result, "timeout_seconds": timeout_seconds})
    combined = completed.stdout + completed.stderr
    output, truncated = _truncate(combined, workspace.max_output_chars)
    status = OK if completed.returncode == 0 else ERROR
    return ToolResult("run_tests", status, output=output, error="" if status == OK else f"pytest exited with {completed.returncode}", truncated=truncated, cost={"test_runs": 1.0}, metadata={"command": command_result, "returncode": completed.returncode, "test_run_count": workspace.test_run_count})


def rollback(workspace: TaskWorkspace, args: Args) -> ToolResult:
    if not workspace.edit_history:
        return ToolResult("rollback", ERROR, error="no edit history to rollback")
    record = workspace.edit_history.pop()
    if not _is_relative_to(record.path.resolve(strict=False), workspace.root):
        return ToolResult("rollback", DENIED, error="recorded edit path is outside checkout root")
    if record.previous_existed:
        _ = record.path.write_text(record.previous_text, encoding="utf-8")
    elif record.path.exists():
        record.path.unlink()
    return ToolResult("rollback", OK, output=f"rolled back {record.relative_path}", metadata={"path": record.relative_path, "reason": args.get("reason", "")})


def git_diff(workspace: TaskWorkspace, _args: Args) -> ToolResult:
    if (workspace.root / ".git").is_dir():
        try:
            completed = subprocess.run(["git", "diff", "--"], cwd=workspace.root, text=True, capture_output=True, timeout=workspace.timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            return ToolResult("git_diff", TIMEOUT, error="git diff timed out")
        output, truncated = _truncate(completed.stdout + completed.stderr, workspace.max_output_chars)
        return ToolResult("git_diff", OK if completed.returncode == 0 else ERROR, output=output, truncated=truncated, metadata={"git_repo": True, "returncode": completed.returncode})
    hunks: list[str] = []
    for record in workspace.edit_history:
        current = record.path.read_text(encoding="utf-8", errors="replace") if record.path.exists() else ""
        hunks.extend(
            difflib.unified_diff(
                record.previous_text.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile=f"a/{record.relative_path}",
                tofile=f"b/{record.relative_path}",
            )
        )
    output, truncated = _truncate("".join(hunks), workspace.max_output_chars)
    status = OK if hunks else UNSUPPORTED
    error = "checkout is not a git repository and no tool edits are recorded" if not hunks else ""
    return ToolResult("git_diff", status, output=output, error=error, truncated=truncated, metadata={"git_repo": False, "edit_records": len(workspace.edit_history)})


def final_answer(workspace: TaskWorkspace, args: Args) -> ToolResult:
    answer = args["answer"]
    if not isinstance(answer, str) or not answer.strip():
        return ToolResult("final_answer", MALFORMED, error="answer must be a non-empty string")
    output, truncated = _truncate(answer.strip(), workspace.max_output_chars)
    return ToolResult("final_answer", OK, output=output, truncated=truncated, metadata={"chars": len(answer.strip())})


def _schema(required: Sequence[str], properties: Mapping[str, str]) -> JsonMap:
    return {
        "type": "object",
        "required": list(required),
        "properties": dict(properties),
        "additionalProperties": False,
    }


def _validate_args(schema: Mapping[str, object], args: Args) -> str:
    required_value = schema.get("required", [])
    required = cast(list[str], required_value) if isinstance(required_value, list) else []
    for item in required:
        if item not in args:
            return f"missing required argument: {item}"
    properties_value = schema.get("properties", {})
    empty_properties: Mapping[str, object] = {}
    properties = (
        cast(Mapping[str, object], properties_value)
        if isinstance(properties_value, Mapping)
        else empty_properties
    )
    allowed = set(properties.keys())
    unknown = sorted(set(args) - allowed)
    if unknown:
        return f"unknown argument(s): {', '.join(unknown)}"
    return ""


def _normalize_relative(path: str) -> str:
    return Path(path).as_posix().lstrip("/")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        _ = path.relative_to(root)
    except ValueError:
        return False
    return True


def _iter_search_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in DENIED_SEARCH_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def _looks_text(path: Path) -> bool:
    if path.name.startswith(".") and path.suffix not in TEXT_FILE_SUFFIXES:
        return False
    try:
        if path.stat().st_size > 2_000_000:
            return False
        sample = path.read_bytes()[:2048]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    return path.suffix in TEXT_FILE_SUFFIXES or not sample or all(byte in b"\t\n\r" or 32 <= byte < 127 for byte in sample[:256])


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    suffix = "\n...[truncated]"
    return text[: max(0, max_chars - len(suffix))] + suffix, True


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int):
        return default
    return max(minimum, min(maximum, value))


def _bounded_float(value: object, *, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _is_protected_edit_path(path: Path, root: Path) -> bool:
    parts = path.resolve(strict=False).relative_to(root).parts
    name = path.name
    return any(part in PROTECTED_EDIT_COMPONENTS for part in parts) or name.startswith("test_") or "benchmark" in name.lower()


def _apply_replacement(previous_text: str, replacement: str, args: Args) -> str | ToolResult:
    has_start = "start_line" in args
    has_end = "end_line" in args
    if has_start != has_end:
        return ToolResult("edit_file", MALFORMED, error="start_line and end_line must be supplied together")
    if not has_start:
        return replacement
    start_line = args.get("start_line")
    end_line = args.get("end_line")
    if not isinstance(start_line, int) or not isinstance(end_line, int):
        return ToolResult("edit_file", MALFORMED, error="start_line and end_line must be integers")
    lines = previous_text.splitlines(keepends=True)
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        return ToolResult("edit_file", MALFORMED, error="line range is outside file bounds")
    replacement_lines = replacement.splitlines(keepends=True)
    if replacement and not replacement.endswith(("\n", "\r")):
        replacement_lines = [*replacement_lines[:-1], replacement_lines[-1] if replacement_lines else replacement]
    return "".join(lines[: start_line - 1] + replacement_lines + lines[end_line:])


def _build_pytest_command(workspace: TaskWorkspace, target: object) -> list[str] | ToolResult:
    if target is None or target == "":
        return ["python", "-m", "pytest", "-q"]
    if not isinstance(target, str):
        return ToolResult("run_tests", MALFORMED, error="target must be a string")
    try:
        tokens = shlex.split(target)
    except ValueError as exc:
        return ToolResult("run_tests", MALFORMED, error=f"invalid command syntax: {exc}")
    if not tokens:
        return ["python", "-m", "pytest", "-q"]
    forbidden = {"sudo", "su", "curl", "wget", "ssh", "scp", "rm", "mv", "chmod", "chown", "git", "pip", "python -c"}
    if any(token in forbidden or token.startswith(("http://", "https://")) for token in tokens):
        return ToolResult("run_tests", DENIED, error="target contains disallowed command tokens")
    if tokens[0] == "pytest":
        command = ["python", "-m", "pytest", *tokens[1:]]
    elif tokens[:3] == ["python", "-m", "pytest"]:
        command = tokens
    elif tokens[:3] == ["python3", "-m", "pytest"]:
        command = ["python", *tokens[1:]]
    else:
        path = target.split("::", 1)[0]
        resolved, error = workspace.resolve_path(path)
        if error or resolved is None:
            return ToolResult("run_tests", DENIED, error=error or "invalid test target")
        if workspace.visible_tests and not workspace.is_visible_test_path(target):
            return ToolResult("run_tests", DENIED, error=f"test target is not visible/allowed: {target}")
        return ["python", "-m", "pytest", "-q", target]
    for token in command[3:]:
        if token.startswith("-") or token == "-q":
            continue
        path = token.split("::", 1)[0]
        if path.startswith("http"):
            return ToolResult("run_tests", DENIED, error="network targets are not allowed")
        resolved, error = workspace.resolve_path(path)
        if error or resolved is None:
            return ToolResult("run_tests", DENIED, error=error or "invalid test path")
        if workspace.visible_tests and not workspace.is_visible_test_path(token):
            return ToolResult("run_tests", DENIED, error=f"test target is not visible/allowed: {token}")
    return command


def _process_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return ""
