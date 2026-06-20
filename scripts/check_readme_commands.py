from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast


SHELL_FENCES = {"bash", "sh", "shell", "console"}


@dataclass(frozen=True)
class ShellCommand:
    line_number: int
    text: str
    marker: str


def _fence_language(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("```"):
        return None
    return stripped[3:].strip().lower()


def collect_commands(readme: str) -> list[ShellCommand]:
    commands: list[ShellCommand] = []
    in_shell = False
    marker = ""
    for index, line in enumerate(readme.splitlines(), start=1):
        language = _fence_language(line)
        if language is not None:
            if in_shell:
                in_shell = False
                marker = ""
            else:
                in_shell = language in SHELL_FENCES
                marker = ""
            continue
        if not in_shell:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            marker = stripped.lower()
            continue
        commands.append(ShellCommand(index, stripped, marker))
    return commands


def _parseable(command: str) -> bool:
    try:
        _ = shlex.split(command)
    except ValueError:
        return False
    return True


def check_commands(path: Path, dry_run_safe: bool) -> list[str]:
    if not path.is_file():
        return [f"missing README: {path}"]
    commands = collect_commands(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not commands:
        errors.append("no shell commands found")
    for command in commands:
        if not _parseable(command.text):
            errors.append(f"line {command.line_number}: shell command is not parseable: {command.text}")
        if dry_run_safe and not ("safe:" in command.marker or "prereq:" in command.marker):
            errors.append(
                f"line {command.line_number}: missing preceding '# safe:' or '# prereq:' marker: {command.text}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check README shell commands")
    _ = parser.add_argument("readme", type=Path)
    _ = parser.add_argument("--dry-run-safe", action="store_true")
    namespace = parser.parse_args(argv)
    readme = cast(Path, namespace.readme)
    dry_run_safe = cast(bool, namespace.dry_run_safe)

    errors = check_commands(readme, dry_run_safe)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"readme commands ok: {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
