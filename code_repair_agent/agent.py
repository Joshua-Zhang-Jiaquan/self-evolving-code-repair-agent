"""Tool-using repair agent and self-evolving policy memory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .environment import CodeRepairEnvironment, RewardBreakdown


STRATEGY_ALIASES = {
    "minus_to_plus": ["add", "sum", "plus", "subtraction", "arithmetic"],
    "factorial_identity": ["factorial", "identity", "product", "zero", "base"],
    "zero_division_guard": ["normalize", "zero", "division", "total", "sum"],
    "slugify_split_join": ["slugify", "trim", "collapse", "spaces", "whitespace"],
}


@dataclass
class CandidatePatch:
    strategy: str
    path: str
    old: str
    new: str


@dataclass
class EpisodeResult:
    task_id: str
    solved_visible: bool
    solved_hidden: bool
    reward: RewardBreakdown
    strategy: Optional[str]
    diff: str
    tool_calls: int
    test_runs: int
    patch_attempts: int
    pass_at_1: bool
    patch_size: int
    events: List[Dict[str, object]]

    def as_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "solved_visible": self.solved_visible,
            "solved_hidden": self.solved_hidden,
            "reward": self.reward.as_dict(),
            "strategy": self.strategy,
            "diff": self.diff,
            "tool_calls": self.tool_calls,
            "test_runs": self.test_runs,
            "patch_attempts": self.patch_attempts,
            "pass_at_1": self.pass_at_1,
            "patch_size": self.patch_size,
            "unsafe_edit": bool(self.reward.unsafe_edit),
            "events": self.events,
        }


@dataclass
class PolicyMemory:
    strategy_scores: Dict[str, float] = field(default_factory=dict)
    keyword_strategy_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "PolicyMemory":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            strategy_scores={k: float(v) for k, v in raw.get("strategy_scores", {}).items()},
            keyword_strategy_scores={
                key: {name: float(score) for name, score in value.items()}
                for key, value in raw.get("keyword_strategy_scores", {}).items()
            },
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "strategy_scores": self.strategy_scores,
                    "keyword_strategy_scores": self.keyword_strategy_scores,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def update(self, keywords: Iterable[str], strategy: Optional[str], reward: RewardBreakdown) -> None:
        if strategy is None:
            return
        delta = reward.reward
        self.strategy_scores[strategy] = self.strategy_scores.get(strategy, 0.0) + delta
        reflected_keywords = list(dict.fromkeys(list(keywords) + STRATEGY_ALIASES.get(strategy, [])))
        for keyword in reflected_keywords:
            bucket = self.keyword_strategy_scores.setdefault(keyword, {})
            bucket[strategy] = bucket.get(strategy, 0.0) + delta

    def score(self, strategy: str, keywords: Iterable[str]) -> float:
        score = 0.0
        for keyword in keywords:
            score += self.keyword_strategy_scores.get(keyword, {}).get(strategy, 0.0)
        return score


class SelfEvolvingRepairAgent:
    """A compact repair agent that improves strategy ordering from feedback."""

    def __init__(
        self,
        memory: Optional[PolicyMemory] = None,
        *,
        use_test_feedback: bool = True,
        max_patch_attempts: Optional[int] = None,
        max_strategy_scans: Optional[int] = None,
    ):
        self.memory = memory or PolicyMemory()
        self.use_test_feedback = use_test_feedback
        self.max_patch_attempts = max_patch_attempts
        self.max_strategy_scans = max_strategy_scans
        self.strategy_order = [
            "slugify_split_join",
            "zero_division_guard",
            "factorial_identity",
            "minus_to_plus",
        ]

    def run_episode(self, env: CodeRepairEnvironment, learn: bool = False) -> EpisodeResult:
        env.inspect_test()
        keywords = extract_keywords(env.task.issue + " " + env.task.optional_hints)
        candidate_files = self._candidate_files(env, keywords)
        strategy_used: Optional[str] = None

        for path in candidate_files:
            env.read_file(path)

        patch_attempts = 0
        pass_at_1 = False
        for patch in self._candidate_patches(env, candidate_files, keywords):
            if self.max_patch_attempts is not None and patch_attempts >= self.max_patch_attempts:
                break
            patch_attempts += 1
            env.edit_file(patch.path, patch.old, patch.new)
            if not self.use_test_feedback:
                strategy_used = patch.strategy
                break
            visible = env.run_tests("visible")
            if patch_attempts == 1:
                pass_at_1 = visible.passed
            if visible.passed:
                strategy_used = patch.strategy
                break
            env.rollback()

        hidden = env.run_tests("hidden")
        reward = env.reward()
        env.final_answer("fixed" if hidden.passed else "not fixed")
        if patch_attempts == 1 and not self.use_test_feedback:
            pass_at_1 = reward.pass_hidden_tests == 1
        if learn:
            self.memory.update(keywords, strategy_used, reward)
        return EpisodeResult(
            task_id=env.task.task_id,
            solved_visible=reward.pass_visible_tests == 1,
            solved_hidden=reward.pass_hidden_tests == 1,
            reward=reward,
            strategy=strategy_used,
            diff=env.diff(),
            tool_calls=len(env.state.tool_events),
            test_runs=env.state.test_runs,
            patch_attempts=patch_attempts,
            pass_at_1=pass_at_1,
            patch_size=_patch_size(env.diff()),
            events=[event.__dict__ for event in env.state.tool_events],
        )

    def _candidate_files(self, env: CodeRepairEnvironment, keywords: List[str]) -> List[str]:
        files: List[str] = []
        for keyword in keywords:
            detail = env.search(keyword)
            for line in detail.splitlines():
                if ":" in line:
                    rel_path = line.split(":", 1)[0]
                    if rel_path.endswith(".py") and rel_path not in files and not rel_path.startswith("tests/"):
                        files.append(rel_path)
        if files:
            return files
        for path in sorted(env.root.rglob("*.py")):  # type: ignore[union-attr]
            rel = str(path.relative_to(env.root))  # type: ignore[arg-type]
            if not rel.startswith("tests/"):
                files.append(rel)
        return files

    def _ordered_strategies(self, keywords: List[str]) -> List[str]:
        indexed = list(enumerate(self.strategy_order))
        indexed.sort(key=lambda item: (-self.memory.score(item[1], keywords), item[0]))
        return [name for _, name in indexed]

    def _candidate_patches(
        self, env: CodeRepairEnvironment, files: List[str], keywords: List[str]
    ) -> Iterable[CandidatePatch]:
        strategy_map: Dict[str, Callable[[str, str], Optional[CandidatePatch]]] = {
            "minus_to_plus": self._minus_to_plus,
            "factorial_identity": self._factorial_identity,
            "zero_division_guard": self._zero_division_guard,
            "slugify_split_join": self._slugify_split_join,
        }
        strategies = self._ordered_strategies(keywords)
        if self.max_strategy_scans is not None:
            strategies = strategies[: self.max_strategy_scans]
        for strategy in strategies:
            maker = strategy_map[strategy]
            for rel_path in files:
                text = env.read_file(rel_path)
                observed = "\n".join(line.split(": ", 1)[1] for line in text.splitlines() if ": " in line)
                patch = maker(rel_path, observed)
                if patch is not None:
                    yield patch

    def _minus_to_plus(self, rel_path: str, text: str) -> Optional[CandidatePatch]:
        match = re.search(r"return\s+([A-Za-z_][A-Za-z0-9_]*)\s*-\s*([A-Za-z_][A-Za-z0-9_]*)", text)
        if not match:
            return None
        old = match.group(0)
        new = f"return {match.group(1)} + {match.group(2)}"
        return CandidatePatch("minus_to_plus", rel_path, old, new)

    def _factorial_identity(self, rel_path: str, text: str) -> Optional[CandidatePatch]:
        old = "if n == 0:\n        return 0"
        if old not in text:
            return None
        return CandidatePatch("factorial_identity", rel_path, old, "if n == 0:\n        return 1")

    def _zero_division_guard(self, rel_path: str, text: str) -> Optional[CandidatePatch]:
        old = "total = sum(values)\n    return [v / total for v in values]"
        if old not in text:
            return None
        new = "total = sum(values)\n    if total == 0:\n        return [0 for _ in values]\n    return [v / total for v in values]"
        return CandidatePatch("zero_division_guard", rel_path, old, new)

    def _slugify_split_join(self, rel_path: str, text: str) -> Optional[CandidatePatch]:
        old = 'return text.lower().replace(" ", "-")'
        if old not in text:
            return None
        new = 'return "-".join(text.strip().lower().split())'
        return CandidatePatch("slugify_split_join", rel_path, old, new)


def extract_keywords(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text.lower())
    stop = {
        "the",
        "and",
        "but",
        "should",
        "returns",
        "return",
        "wrong",
        "test",
        "tests",
        "self",
        "assert",
        "equal",
        "visible",
        "failure",
        "fail",
        "failed",
        "expected",
        "got",
        "traceback",
        "assertionerror",
        "unittest",
        "line",
        "ran",
    }
    result: List[str] = []
    for word in words:
        if len(word) < 3 or word in stop or word.startswith("test"):
            continue
        if word not in result:
            result.append(word)
    return result[:12]


def _patch_size(diff_text: str) -> int:
    return sum(
        1
        for line in diff_text.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )
