"""Optional Defects4J adapter.

This module does not vendor Defects4J. It uses the official `defects4j` CLI
when it is installed and available on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Defects4JCase:
    project: str
    bug_id: int
    workdir: Path
    version: str = "b"

    @property
    def checkout_version(self) -> str:
        return f"{self.bug_id}{self.version}"


class Defects4JUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandResult:
    command: List[str]
    cwd: str
    returncode: int
    output: str
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "output": self.output,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "ok": self.ok,
        }


class Defects4JClient:
    """Small wrapper around official Defects4J commands."""

    def __init__(self, binary: str = "defects4j", timeout: Optional[int] = None):
        self.binary = binary
        self.timeout = int(timeout or os.environ.get("DEFECTS4J_TIMEOUT", "3600"))

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def require(self) -> None:
        if not self.available():
            raise Defects4JUnavailable(
                "`defects4j` is not on PATH. Install rjust/defects4j, run init.sh, "
                "and export defects4j/framework/bin into PATH."
            )

    def checkout(self, case: Defects4JCase) -> str:
        self.require()
        workdir = case.workdir.resolve()
        workdir.parent.mkdir(parents=True, exist_ok=True)
        return self.run(
            [
                self.binary,
                "checkout",
                "-p",
                case.project,
                "-v",
                case.checkout_version,
                "-w",
                str(workdir),
            ],
            cwd=workdir.parent,
            check=True,
        ).output

    def compile(self, workdir: Path) -> str:
        self.require()
        return self.run([self.binary, "compile"], cwd=workdir, check=True).output

    def test(self, workdir: Path, test: Optional[str] = None) -> str:
        self.require()
        command = [self.binary, "test"]
        if test:
            command.extend(["-t", test])
        return self.run(command, cwd=workdir, check=True).output

    def export(self, workdir: Path, prop: str) -> str:
        self.require()
        return self.run([self.binary, "export", "-p", prop], cwd=workdir, check=True).output

    def metadata(self, workdir: Path) -> Dict[str, str]:
        props = [
            "dir.src.classes",
            "dir.src.tests",
            "tests.trigger",
            "tests.relevant",
            "classes.modified",
        ]
        return {prop: _clean_export_output(self.export(workdir, prop)) for prop in props}

    def run(self, command: List[str], cwd: Path, check: bool = False) -> CommandResult:
        env = {**os.environ, "TZ": "America/Los_Angeles"}
        started = time.perf_counter()
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout,
            check=False,
        )
        result = CommandResult(
            command=command,
            cwd=str(cwd),
            returncode=proc.returncode,
            output=proc.stdout,
            elapsed_seconds=time.perf_counter() - started,
        )
        if check and not result.ok:
            raise RuntimeError(f"{' '.join(command)} failed with {proc.returncode}\n{proc.stdout}")
        return result


def _clean_export_output(output: str) -> str:
    """Keep only exported Defects4J values, not CLI progress log lines."""
    values: List[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Running ant "):
            continue
        values.append(stripped)
    return "\n".join(values)
