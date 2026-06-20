from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from repair_agent.config import ConfigMap


AGENT_INTERFACE_VERSION = "agent-interface-v1"


def _empty_str_map() -> dict[str, str]:
    return {}


def _empty_object_map() -> dict[str, object]:
    return {}


@dataclass(frozen=True)
class AgentTask:
    instance_id: str
    repo: str
    problem_statement: str
    checkout_root: Path
    visible_tests: tuple[str, ...] = ()
    visible_failures: Mapping[str, str] = field(default_factory=_empty_str_map)
    model_name_or_path: str = "rule_based_local"
    max_steps: int = 12
    max_test_runs: int = 1
    test_timeout_seconds: float = 10.0
    max_output_chars: int = 4000
    metadata: Mapping[str, object] = field(default_factory=_empty_object_map)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkout_root", Path(self.checkout_root))
        object.__setattr__(self, "visible_tests", tuple(str(item) for item in self.visible_tests))
        object.__setattr__(self, "visible_failures", {str(key): str(value) for key, value in self.visible_failures.items()})
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.max_test_runs < 0:
            raise ValueError("max_test_runs must be non-negative")
        if self.test_timeout_seconds <= 0:
            raise ValueError("test_timeout_seconds must be positive")
        if self.max_output_chars < 128:
            raise ValueError("max_output_chars must be at least 128")


@dataclass(frozen=True)
class TrajectoryStepRecord:
    instance_id: str
    run_id: str
    model_name_or_path: str
    agent_version: str
    step_index: int
    action: str
    tool: str
    status: str
    output_summary: str = ""
    error: str = ""
    tool_call_count: int = 0
    test_run_count: int = 0
    edit_count: int = 0
    final_status: str = "running"
    args_hash: str = ""
    timestamp: str = ""
    metadata: Mapping[str, object] = field(default_factory=_empty_object_map)

    def to_row(self) -> ConfigMap:
        return {
            "action": self.action,
            "agent_version": self.agent_version,
            "args_hash": self.args_hash,
            "edit_count": self.edit_count,
            "error": self.error,
            "event": "trajectory_step",
            "final_status": self.final_status,
            "instance_id": self.instance_id,
            "metadata": dict(self.metadata),
            "model_name_or_path": self.model_name_or_path,
            "output_summary": self.output_summary,
            "run_id": self.run_id,
            "status": self.status,
            "step_index": self.step_index,
            "test_run_count": self.test_run_count,
            "timestamp": self.timestamp or utc_now(),
            "tool": self.tool,
            "tool_call_count": self.tool_call_count,
        }


@dataclass(frozen=True)
class AgentFinalAnswer:
    instance_id: str
    model_name_or_path: str
    model_patch: str
    status: str
    explanation: str
    patch_sha256: str = ""
    patch_path: str = ""
    metadata: Mapping[str, object] = field(default_factory=_empty_object_map)

    def __post_init__(self) -> None:
        digest = self.patch_sha256 or hashlib.sha256(self.model_patch.encode("utf-8")).hexdigest()
        object.__setattr__(self, "patch_sha256", digest)

    def prediction_row(self) -> ConfigMap:
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
        }


@dataclass(frozen=True)
class AgentResult:
    final: AgentFinalAnswer
    trajectory: Sequence[TrajectoryStepRecord]
    metrics: Mapping[str, object] = field(default_factory=_empty_object_map)

    def trajectory_rows(self) -> list[ConfigMap]:
        return [step.to_row() for step in self.trajectory]


class RepairAgent(Protocol):
    agent_version: str

    def run(self, task: AgentTask, run_id: str) -> AgentResult:
        ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
