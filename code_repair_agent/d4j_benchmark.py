"""Defects4J benchmark case selection and config parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class BenchmarkCase:
    project: str
    bug_id: int

    @property
    def case_id(self) -> str:
        return f"{self.project}-{self.bug_id}"

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "BenchmarkCase":
        return cls(project=str(raw["project"]), bug_id=int(raw["bug_id"]))

    def as_dict(self) -> Dict[str, object]:
        return {"project": self.project, "bug_id": self.bug_id, "case_id": self.case_id}


def default_30_cases() -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []
    cases.extend(BenchmarkCase("Chart", bug_id) for bug_id in range(1, 11))
    cases.extend(BenchmarkCase("Lang", bug_id) for bug_id in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    cases.extend(BenchmarkCase("Math", bug_id) for bug_id in range(1, 11))
    return cases


def load_cases(path: Optional[Path]) -> List[BenchmarkCase]:
    if path is None:
        return default_30_cases()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkCase.from_dict(item) for item in raw["cases"]]


def write_cases(path: Path, cases: Iterable[BenchmarkCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "Defects4J",
        "cases": [case.as_dict() for case in cases],
        "notes": "30-bug broad benchmark: Chart 1-10, Lang 1 and 3-11, Math 1-10.",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
