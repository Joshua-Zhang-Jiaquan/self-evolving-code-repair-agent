"""Persistent self-improvement memory for Defects4J runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class BenchmarkMemory:
    patch_ranking: Dict[str, Dict[str, float]] = field(default_factory=dict)
    test_selection: Dict[str, Dict[str, float]] = field(default_factory=dict)
    repair_skill_memory: Dict[str, Dict[str, float]] = field(default_factory=dict)
    test_skill_memory: Dict[str, Dict[str, float]] = field(default_factory=dict)
    regression_outcomes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    failure_reflections: List[Dict[str, str]] = field(default_factory=list)
    success_strategies: List[Dict[str, str]] = field(default_factory=list)
    max_failure_reflections: int = 100
    max_success_strategies: int = 100

    @classmethod
    def load(cls, path: Path) -> "BenchmarkMemory":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            patch_ranking={
                feature: {name: float(score) for name, score in scores.items()}
                for feature, scores in raw.get("patch_ranking", {}).items()
            },
            test_selection={
                feature: {name: float(score) for name, score in scores.items()}
                for feature, scores in raw.get("test_selection", {}).items()
            },
            repair_skill_memory={
                feature: {name: float(score) for name, score in scores.items()}
                for feature, scores in raw.get("repair_skill_memory", {}).items()
            },
            test_skill_memory={
                feature: {name: float(score) for name, score in scores.items()}
                for feature, scores in raw.get("test_skill_memory", {}).items()
            },
            regression_outcomes={
                feature: {name: float(score) for name, score in scores.items()}
                for feature, scores in raw.get("regression_outcomes", {}).items()
            },
            failure_reflections=list(raw.get("failure_reflections", [])),
            success_strategies=list(raw.get("success_strategies", [])),
            max_failure_reflections=int(raw.get("max_failure_reflections", 100)),
            max_success_strategies=int(raw.get("max_success_strategies", 100)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def as_dict(self) -> Dict[str, object]:
        return {
            "patch_ranking": self.patch_ranking,
            "test_selection": self.test_selection,
            "repair_skill_memory": self.repair_skill_memory,
            "test_skill_memory": self.test_skill_memory,
            "regression_outcomes": self.regression_outcomes,
            "failure_reflections": self.failure_reflections,
            "success_strategies": self.success_strategies,
            "max_failure_reflections": self.max_failure_reflections,
            "max_success_strategies": self.max_success_strategies,
        }

    def apply_decay(self, factor: float) -> None:
        """Multiply all score tables by a decay factor (0 < factor <= 1.0).

        This allows old evidence to be gradually discounted as new evidence
        accumulates, preventing stale early experiences from dominating.
        Called between cases by the benchmark runner when decay is enabled.
        """
        if factor <= 0 or factor > 1.0:
            return
        for table in (
            self.patch_ranking,
            self.test_selection,
            self.repair_skill_memory,
            self.test_skill_memory,
            self.regression_outcomes,
        ):
            for feature in list(table):
                for name in list(table[feature]):
                    table[feature][name] *= factor
                    if abs(table[feature][name]) < 0.01:
                        del table[feature][name]
                if not table[feature]:
                    del table[feature]

    def prompt_preferences(self, features: Iterable[str]) -> List[str]:
        scores = self._aggregate(self.patch_ranking, features)
        positive = [(name, score) for name, score in scores.items() if score > 0]
        return [name for name, _ in sorted(positive, key=lambda item: (-item[1], item[0]))]

    def preferred_test_scope(self, features: Iterable[str], default: str = "trigger") -> str:
        scores = self._aggregate(self.test_selection, features)
        if not scores:
            return default
        valid = [(scope, score) for scope, score in scores.items() if scope in {"trigger", "relevant", "all"} and score > 0]
        if not valid:
            return default
        return sorted(valid, key=lambda item: (-item[1], item[0]))[0][0]

    def repair_skill_preferences(self, features: Iterable[str], limit: int = 5) -> List[str]:
        return self._ranked_names(self.repair_skill_memory, features, limit=limit)

    def test_skill_preferences(self, features: Iterable[str], limit: int = 5) -> List[str]:
        return self._ranked_names(self.test_skill_memory, features, limit=limit)

    def regression_warnings(self, features: Iterable[str], limit: int = 5) -> List[str]:
        scores = self._aggregate(self.regression_outcomes, features)
        risky = [(name, score) for name, score in scores.items() if score < 0]
        return [
            f"{name} has failed regression before; avoid visible-only overfitting"
            for name, _ in sorted(risky, key=lambda item: (item[1], item[0]))[:limit]
        ]

    def attempt_bonus(self, features: Iterable[str], *, max_bonus: int = 2) -> int:
        """Return a bounded extra-attempt budget when memory says retries help.

        The value is feature-derived, not case-id derived, so it can transfer to
        new bugs without hard-coding benchmark answers.
        """
        scores = self._aggregate(self.repair_skill_memory, features)
        retry_score = scores.get("retry-after-feedback", 0.0) + scores.get("repair-after-regression", 0.0)
        if retry_score <= 0:
            return 0
        return min(max_bonus, 1 + int(retry_score // 2))

    def relevant_reflections(self, features: Iterable[str], limit: int = 5) -> List[str]:
        feature_set = set(features)
        matches: List[str] = []
        for item in reversed(self.failure_reflections):
            item_features = set(item.get("features", "").split(","))
            if feature_set & item_features and item.get("reflection"):
                matches.append(_sanitize_reflection(item))
            if len(matches) >= limit:
                break
        return list(reversed(matches))

    def relevant_success_strategies(self, features: Iterable[str], limit: int = 5) -> List[str]:
        feature_set = set(features)
        matches: List[str] = []
        for item in reversed(self.success_strategies):
            item_features = set(item.get("features", "").split(","))
            if feature_set & item_features and item.get("strategy"):
                matches.append(item["strategy"])
            if len(matches) >= limit:
                break
        return list(reversed(matches))

    def update(
        self,
        *,
        features: Iterable[str],
        patch_style: str,
        test_scope: str,
        solved: bool,
        failure_reason: Optional[str],
        reflection: Optional[str],
        repair_skill: Optional[str] = None,
        test_skill: Optional[str] = None,
        visible_passed: bool = False,
        regression_checked: bool = False,
        regression_passed: bool = False,
        success_strategy: Optional[str] = None,
        update_check_memory: bool = True,
        update_repair_memory: bool = True,
    ) -> None:
        delta = _outcome_delta(
            solved=solved,
            failure_reason=failure_reason,
            visible_passed=visible_passed,
            regression_checked=regression_checked,
            regression_passed=regression_passed,
        )
        skill_delta = _skill_delta(
            solved=solved,
            failure_reason=failure_reason,
            visible_passed=visible_passed,
            regression_checked=regression_checked,
            regression_passed=regression_passed,
        )
        features = list(dict.fromkeys(features))
        if not features:
            return
        for feature in features:
            if update_repair_memory:
                self.patch_ranking.setdefault(feature, {})[patch_style] = (
                    self.patch_ranking.setdefault(feature, {}).get(patch_style, 0.0) + delta
                )
                if repair_skill:
                    self.repair_skill_memory.setdefault(feature, {})[repair_skill] = (
                        self.repair_skill_memory.setdefault(feature, {}).get(repair_skill, 0.0) + skill_delta
                    )
            if update_check_memory:
                self.test_selection.setdefault(feature, {})[test_scope] = (
                    self.test_selection.setdefault(feature, {}).get(test_scope, 0.0) + delta
                )
                if test_skill:
                    self.test_skill_memory.setdefault(feature, {})[test_skill] = (
                        self.test_skill_memory.setdefault(feature, {}).get(test_skill, 0.0) + skill_delta
                    )
            if update_check_memory and regression_checked:
                key = f"style:{patch_style}|scope:{test_scope}"
                regression_delta = 1.0 if regression_passed else -1.0
                self.regression_outcomes.setdefault(feature, {})[key] = (
                    self.regression_outcomes.setdefault(feature, {}).get(key, 0.0) + regression_delta
                )
        if update_repair_memory and not solved and reflection:
            feature_string = ",".join(features)
            reason = failure_reason or "unknown"
            replacement = {
                "features": feature_string,
                "failure_reason": reason,
                "reflection": reflection[:1000],
            }
            for index, item in enumerate(self.failure_reflections):
                if item.get("features") == feature_string and item.get("failure_reason") == reason:
                    self.failure_reflections[index] = replacement
                    break
            else:
                self.failure_reflections.append(replacement)
                if len(self.failure_reflections) > self.max_failure_reflections:
                    del self.failure_reflections[0]
        if update_repair_memory and solved and success_strategy:
            feature_string = ",".join(features)
            replacement = {
                "features": feature_string,
                "strategy": success_strategy[:1000],
            }
            for index, item in enumerate(self.success_strategies):
                if item.get("features") == feature_string:
                    self.success_strategies[index] = replacement
                    break
            else:
                self.success_strategies.append(replacement)
                if len(self.success_strategies) > self.max_success_strategies:
                    del self.success_strategies[0]

    def _aggregate(self, table: Dict[str, Dict[str, float]], features: Iterable[str]) -> Dict[str, float]:
        aggregate: Dict[str, float] = {}
        for feature in features:
            weight = _feature_weight(feature)
            for name, score in table.get(feature, {}).items():
                aggregate[name] = aggregate.get(name, 0.0) + score * weight
        return aggregate

    def _ranked_names(self, table: Dict[str, Dict[str, float]], features: Iterable[str], *, limit: int) -> List[str]:
        scores = self._aggregate(table, features)
        positive = [(name, score) for name, score in scores.items() if score > 0]
        return [name for name, _ in sorted(positive, key=lambda item: (-item[1], item[0]))[:limit]]


def extract_features(project: str, test_output: str, metadata: Dict[str, str]) -> List[str]:
    features = [f"project:{project}"]
    trigger = metadata.get("tests.trigger", "")
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.$]+", trigger)[:5]:
        features.append(f"trigger:{token.split('.')[-1].lower()}")
    for exception in re.findall(r"([A-Za-z_][A-Za-z0-9_.]*Exception|AssertionError|Error)", test_output)[:5]:
        features.append(f"exception:{exception.split('.')[-1].lower()}")
    modified = metadata.get("classes.modified", "")
    for klass in re.findall(r"[A-Za-z_][A-Za-z0-9_.]+", modified)[:5]:
        features.append(f"class:{klass.split('.')[-1].lower()}")
    return list(dict.fromkeys(features))


def _outcome_delta(
    *,
    solved: bool,
    failure_reason: Optional[str],
    visible_passed: bool,
    regression_checked: bool,
    regression_passed: bool,
) -> float:
    if solved:
        return 1.5
    if regression_checked and visible_passed and not regression_passed:
        return -1.0
    if failure_reason == "compile_failure":
        return -0.75
    if failure_reason == "visible_failure":
        return -0.5
    return -0.25


def _skill_delta(
    *,
    solved: bool,
    failure_reason: Optional[str],
    visible_passed: bool,
    regression_checked: bool,
    regression_passed: bool,
) -> float:
    if solved:
        return 1.5
    if regression_checked and visible_passed and not regression_passed:
        return 0.75
    if failure_reason == "compile_failure":
        return -0.5
    return -0.25


def _sanitize_reflection(item: Dict[str, str]) -> str:
    reason = item.get("failure_reason", "unknown")
    reflection = item.get("reflection", "")
    if reason == "patch_apply_failure" and "old text not found" in reflection:
        return (
            "patch_apply_failure: previous old text did not match the current source; "
            "use only exact lines visible in source snippets or a minimal anchored window, "
            "and do not reuse absent identifiers from failed attempts"
        )
    if reason == "patch_apply_failure":
        return (
            "patch_apply_failure: previous patch could not be applied safely; "
            "ground the next patch in exact current source text"
        )
    if reason == "visible_failure":
        return (
            "visible_failure: previous patch compiled but did not pass trigger tests; "
            "use failing_tests assertion details and avoid repeating the same patch style unchanged"
        )
    return reflection


def _feature_weight(feature: str) -> float:
    """Prefer specific transfer evidence over broad project-level evidence."""
    if feature.startswith(("class:", "exception:", "trigger:")):
        return 2.0
    return 1.0
