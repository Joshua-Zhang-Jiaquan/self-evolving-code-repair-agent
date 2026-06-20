from __future__ import annotations
# pyright: reportAny=false, reportExplicitAny=false, reportGeneralTypeIssues=false, reportUnannotatedClassAttribute=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


ACTION_VOCABULARY: tuple[str, ...] = (
    "search",
    "read_file",
    "inspect_test",
    "edit_file",
    "run_tests",
    "rollback",
    "git_diff",
    "final_answer",
)
ACTION_SCHEMA_VERSION = "safe-tool-selection-v1"
POLICY_CHECKPOINT_VERSION = 1
MODEL_GATE_STATUSES: tuple[str, ...] = ("unknown", "pass", "blocked", "skipped")
TEST_STATUSES: tuple[str, ...] = (
    "not_run",
    "ok",
    "error",
    "timeout",
    "budget_exceeded",
    "denied",
    "malformed",
    "failed",
)


@dataclass(frozen=True)
class LearningContext:
    step_index: int = 0
    max_steps: int = 12
    test_run_count: int = 0
    max_test_runs: int = 1
    last_action_type: str = "none"
    last_test_status: str = "not_run"
    relevant_file_score: float = 0.0
    patch_exists: bool = False
    repeated_action_count: int = 0
    model_gate_status: str = "unknown"
    tool_call_count: int = 0
    visible_gpu_count: int = 0
    rollout_parallelism: int = 1


@dataclass(frozen=True)
class PolicyTransition:
    features: Mapping[str, float]
    action: str
    reward: float


class FeatureExtractor:
    def __init__(self, *, action_vocabulary: Sequence[str] = ACTION_VOCABULARY) -> None:
        self.action_vocabulary = tuple(action_vocabulary)
        self.feature_names = self._feature_names()

    def extract(self, context: LearningContext) -> dict[str, float]:
        max_steps = max(1, context.max_steps)
        max_tests = max(1, context.max_test_runs)
        features: dict[str, float] = {
            "bias": 1.0,
            "step_fraction": _clamp(context.step_index / max_steps),
            "remaining_budget_fraction": _clamp((max_steps - context.step_index) / max_steps),
            "test_budget_fraction": _clamp((max_tests - context.test_run_count) / max_tests),
            "relevant_file_score": _clamp(context.relevant_file_score),
            "patch_exists": 1.0 if context.patch_exists else 0.0,
            "repeated_action_count": _clamp(context.repeated_action_count / max_steps),
            "tool_call_fraction": _clamp(context.tool_call_count / max_steps),
            "max_steps_norm": _clamp(max_steps / 64.0),
            "max_test_runs_norm": _clamp(max_tests / 8.0),
            "visible_gpu_count_norm": _clamp(context.visible_gpu_count / 8.0),
            "rollout_parallelism_norm": _clamp(context.rollout_parallelism / 16.0),
        }
        for name in ("none", *self.action_vocabulary):
            features[f"last_action={name}"] = 1.0 if context.last_action_type == name else 0.0
        status = context.last_test_status if context.last_test_status in TEST_STATUSES else "failed"
        for name in TEST_STATUSES:
            features[f"last_test_status={name}"] = 1.0 if status == name else 0.0
        gate = context.model_gate_status if context.model_gate_status in MODEL_GATE_STATUSES else "unknown"
        for name in MODEL_GATE_STATUSES:
            features[f"model_gate_status={name}"] = 1.0 if gate == name else 0.0
        return {name: float(features.get(name, 0.0)) for name in self.feature_names}

    def _feature_names(self) -> tuple[str, ...]:
        base = (
            "bias",
            "step_fraction",
            "remaining_budget_fraction",
            "test_budget_fraction",
            "relevant_file_score",
            "patch_exists",
            "repeated_action_count",
            "tool_call_fraction",
            "max_steps_norm",
            "max_test_runs_norm",
            "visible_gpu_count_norm",
            "rollout_parallelism_norm",
        )
        last_actions = tuple(f"last_action={name}" for name in ("none", *self.action_vocabulary))
        tests = tuple(f"last_test_status={name}" for name in TEST_STATUSES)
        gates = tuple(f"model_gate_status={name}" for name in MODEL_GATE_STATUSES)
        return (*base, *last_actions, *tests, *gates)


class LinearSoftmaxPolicy:
    def __init__(
        self,
        *,
        feature_names: Sequence[str] | None = None,
        action_vocabulary: Sequence[str] = ACTION_VOCABULARY,
        learning_rate: float = 0.05,
        gamma: float = 1.0,
        baseline_decay: float = 0.9,
        baseline_value: float = 0.0,
        weights: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self.action_vocabulary = tuple(action_vocabulary)
        if self.action_vocabulary != ACTION_VOCABULARY:
            raise ValueError("learning policy action vocabulary is frozen by safe-tool-selection-v1")
        self.feature_names = tuple(feature_names or FeatureExtractor().feature_names)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.baseline_decay = float(baseline_decay)
        self.baseline_value = float(baseline_value)
        if not (0.0 <= self.baseline_decay < 1.0):
            raise ValueError("baseline_decay must be in [0, 1)")
        self.weights = self._initial_weights(weights)

    def logits(self, features: Mapping[str, float]) -> dict[str, float]:
        return {
            action: sum(self.weights[action][name] * float(features.get(name, 0.0)) for name in self.feature_names)
            for action in self.action_vocabulary
        }

    def probabilities(
        self,
        features: Mapping[str, float],
        available_actions: Sequence[str] | None = None,
    ) -> dict[str, float]:
        allowed = tuple(available_actions or self.action_vocabulary)
        if not allowed:
            raise ValueError("available_actions must not be empty")
        unknown = [action for action in allowed if action not in self.action_vocabulary]
        if unknown:
            raise ValueError(f"unknown policy action(s): {', '.join(unknown)}")
        logits = self.logits(features)
        max_logit = max(logits[action] for action in allowed)
        exp_values = {action: math.exp(logits[action] - max_logit) for action in allowed}
        denominator = sum(exp_values.values())
        probabilities = {action: 0.0 for action in self.action_vocabulary}
        for action in allowed:
            probabilities[action] = exp_values[action] / denominator
        return probabilities

    def select_action(
        self,
        features: Mapping[str, float],
        *,
        available_actions: Sequence[str] | None = None,
        rng: random.Random | None = None,
        deterministic: bool = False,
    ) -> str:
        probs = self.probabilities(features, available_actions)
        allowed = tuple(available_actions or self.action_vocabulary)
        if deterministic:
            ranked = list(enumerate(allowed))
            return max(ranked, key=lambda item: (probs[item[1]], -item[0]))[1]
        draw = (rng or random).random()
        cumulative = 0.0
        for action in allowed:
            cumulative += probs[action]
            if draw <= cumulative:
                return action
        return allowed[-1]

    def update_episode(self, transitions: Sequence[PolicyTransition]) -> dict[str, float | int | list[float]]:
        returns = compute_returns([transition.reward for transition in transitions], self.gamma)
        baseline_before = self.baseline_value
        total_abs_update = 0.0
        for transition, reward_to_go in zip(transitions, returns, strict=True):
            advantage = reward_to_go - baseline_before
            probs = self.probabilities(transition.features)
            for action in self.action_vocabulary:
                indicator = 1.0 if action == transition.action else 0.0
                coefficient = self.learning_rate * advantage * (indicator - probs[action])
                if coefficient == 0.0:
                    continue
                for name in self.feature_names:
                    delta = coefficient * float(transition.features.get(name, 0.0))
                    self.weights[action][name] += delta
                    total_abs_update += abs(delta)
        episode_return = returns[0] if returns else 0.0
        self.baseline_value = update_moving_average_baseline(
            self.baseline_value,
            episode_return,
            self.baseline_decay,
        )
        return {
            "baseline_before": baseline_before,
            "baseline_after": self.baseline_value,
            "episode_return": episode_return,
            "returns": returns,
            "transition_count": len(transitions),
            "total_abs_update": total_abs_update,
        }

    def to_checkpoint(self) -> dict[str, object]:
        return {
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "action_vocabulary": list(self.action_vocabulary),
            "baseline_decay": self.baseline_decay,
            "baseline_value": self.baseline_value,
            "checkpoint_version": POLICY_CHECKPOINT_VERSION,
            "feature_names": list(self.feature_names),
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "policy_type": "linear_softmax_reinforce",
            "weights": self.weights,
        }

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = target.write_text(json.dumps(self.to_checkpoint(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "LinearSoftmaxPolicy":
        loaded: object = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("policy checkpoint must contain a JSON object")
        checkpoint = {str(key): value for key, value in loaded.items()}
        if checkpoint.get("action_schema_version") != ACTION_SCHEMA_VERSION:
            raise ValueError("incompatible policy action schema version")
        if tuple(checkpoint.get("action_vocabulary", ())) != ACTION_VOCABULARY:
            raise ValueError("policy action vocabulary does not match frozen safe tools")
        weights_value = checkpoint.get("weights")
        if not isinstance(weights_value, dict):
            raise ValueError("policy checkpoint missing weights")
        weights: dict[str, dict[str, float]] = {}
        for action, row in weights_value.items():
            if not isinstance(row, dict):
                raise ValueError("policy checkpoint weight rows must be mappings")
            weights[str(action)] = {str(name): float(value) for name, value in row.items() if isinstance(value, int | float)}
        feature_names_value = checkpoint.get("feature_names")
        if not isinstance(feature_names_value, list):
            raise ValueError("policy checkpoint missing feature_names")
        return cls(
            feature_names=[str(item) for item in feature_names_value],
            learning_rate=_float(checkpoint.get("learning_rate"), 0.05),
            gamma=_float(checkpoint.get("gamma"), 1.0),
            baseline_decay=_float(checkpoint.get("baseline_decay"), 0.9),
            baseline_value=_float(checkpoint.get("baseline_value"), 0.0),
            weights=weights,
        )

    def _initial_weights(self, weights: Mapping[str, Mapping[str, float]] | None) -> dict[str, dict[str, float]]:
        initialized: dict[str, dict[str, float]] = {}
        for action in self.action_vocabulary:
            row = dict(weights.get(action, {})) if weights is not None else {}
            initialized[action] = {name: float(row.get(name, 0.0)) for name in self.feature_names}
        return initialized


def compute_returns(rewards: Sequence[float], gamma: float = 1.0) -> list[float]:
    running = 0.0
    returns: list[float] = []
    for reward in reversed(rewards):
        running = float(reward) + float(gamma) * running
        returns.append(running)
    returns.reverse()
    return returns


def update_moving_average_baseline(current: float, target: float, decay: float) -> float:
    if not (0.0 <= decay < 1.0):
        raise ValueError("decay must be in [0, 1)")
    return float(decay) * float(current) + (1.0 - float(decay)) * float(target)


def assert_frozen_tool_schema(action_vocabulary: Sequence[str]) -> None:
    if tuple(action_vocabulary) != ACTION_VOCABULARY:
        raise ValueError("learning action vocabulary must stay aligned with the frozen safe tool registry")


def policy_from_config(config: Mapping[str, object]) -> LinearSoftmaxPolicy:
    learning = config.get("learning", {})
    if not isinstance(learning, Mapping):
        learning = {}
    raw_action_vocab = learning.get("action_vocabulary", ACTION_VOCABULARY)
    action_vocab = tuple(str(item) for item in raw_action_vocab) if isinstance(raw_action_vocab, Sequence) and not isinstance(raw_action_vocab, str) else ACTION_VOCABULARY
    assert_frozen_tool_schema(action_vocab)
    return LinearSoftmaxPolicy(
        learning_rate=_float(learning.get("learning_rate"), 0.05),
        gamma=_float(learning.get("gamma"), 1.0),
        baseline_decay=_float(learning.get("baseline_decay"), 0.9),
    )


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _float(value: object, default: float) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default
