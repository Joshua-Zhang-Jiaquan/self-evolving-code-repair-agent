from __future__ import annotations
# pyright: reportAny=false, reportUnknownMemberType=false

import json
from pathlib import Path
from typing import cast

import pytest
import yaml

from repair_agent.training.policy import (
    ACTION_SCHEMA_VERSION,
    ACTION_VOCABULARY,
    FeatureExtractor,
    LearningContext,
    LinearSoftmaxPolicy,
    PolicyTransition,
    assert_frozen_tool_schema,
    compute_returns,
    update_moving_average_baseline,
)
from repair_agent.training.train import NO_SIGNAL, allocate_rollout_workers, main as train_main, training_status


def test_compute_returns_uses_reward_to_go_discounting():
    returns = compute_returns([1.0, 2.0, 3.0], gamma=0.5)

    assert returns == [2.75, 3.5, 3.0]


def test_moving_average_baseline_updates_toward_return():
    updated = update_moving_average_baseline(current=2.0, target=10.0, decay=0.75)

    assert updated == 4.0


def test_positive_reward_updates_selected_action_probability():
    policy = LinearSoftmaxPolicy(learning_rate=0.2, baseline_decay=0.5)
    features = {name: 0.0 for name in policy.feature_names}
    features["bias"] = 1.0
    before = policy.probabilities(features)["search"]

    update = policy.update_episode([PolicyTransition(features=features, action="search", reward=5.0)])
    after = policy.probabilities(features)["search"]

    total_abs_update = update["total_abs_update"]
    assert isinstance(total_abs_update, float)
    assert total_abs_update > 0.0
    assert after > before
    assert policy.logits(features)["search"] > 0.0


def test_checkpoint_save_load_preserves_policy_distribution(tmp_path: Path):
    policy = LinearSoftmaxPolicy(learning_rate=0.1, gamma=0.9, baseline_decay=0.8)
    features = {name: 0.0 for name in policy.feature_names}
    features["bias"] = 1.0
    _ = policy.update_episode([PolicyTransition(features=features, action="run_tests", reward=3.0)])
    path = tmp_path / "policy.json"

    policy.save_json(path)
    loaded = LinearSoftmaxPolicy.load_json(path)

    assert loaded.to_checkpoint()["action_schema_version"] == ACTION_SCHEMA_VERSION
    assert loaded.probabilities(features)["run_tests"] == pytest.approx(policy.probabilities(features)["run_tests"])
    assert loaded.baseline_value == pytest.approx(policy.baseline_value)


def test_feature_extraction_encodes_required_context():
    extractor = FeatureExtractor()
    features = extractor.extract(
        LearningContext(
            step_index=3,
            max_steps=12,
            test_run_count=1,
            max_test_runs=2,
            last_action_type="run_tests",
            last_test_status="error",
            relevant_file_score=0.75,
            patch_exists=True,
            repeated_action_count=2,
            model_gate_status="pass",
            tool_call_count=4,
            visible_gpu_count=4,
            rollout_parallelism=4,
        )
    )

    assert features["step_fraction"] == pytest.approx(0.25)
    assert features["remaining_budget_fraction"] == pytest.approx(0.75)
    assert features["test_budget_fraction"] == pytest.approx(0.5)
    assert features["last_action=run_tests"] == 1.0
    assert features["last_test_status=error"] == 1.0
    assert features["relevant_file_score"] == pytest.approx(0.75)
    assert features["patch_exists"] == 1.0
    assert features["repeated_action_count"] > 0.0
    assert features["model_gate_status=pass"] == 1.0
    assert features["visible_gpu_count_norm"] == pytest.approx(0.5)


def test_tool_schema_freeze_matches_safe_registry_names():
    assert ACTION_VOCABULARY == ("search", "read_file", "inspect_test", "edit_file", "run_tests", "rollback", "git_diff", "final_answer")
    assert_frozen_tool_schema(ACTION_VOCABULARY)
    with pytest.raises(ValueError, match="frozen"):
        assert_frozen_tool_schema((*ACTION_VOCABULARY, "unsafe_shell"))


def test_no_signal_status_when_all_rewards_zero(tmp_path: Path):
    reward_config = tmp_path / "zero_rewards.yaml"
    _ = reward_config.write_text(
        yaml.safe_dump(
            {
                "weights": {
                    "pass": 0.0,
                    "visible_test_pass": 0.0,
                    "visible_test_failure": 0.0,
                    "hidden_regression_ready": 0.0,
                    "partial_progress": 0.0,
                    "relevant_file": 0.0,
                    "tool_calls": 0.0,
                    "test_runs": 0.0,
                    "unsafe_edit": 0.0,
                    "test_deletion": 0.0,
                    "timeout": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    config_path = write_learning_config(tmp_path, reward_config=reward_config)

    assert train_main(["--config", str(config_path), "--limit", "1", "--episodes", "1", "--run-id", "no_signal_unit"]) == 0
    status = cast(dict[str, object], json.loads((tmp_path / "runs" / "no_signal_unit" / "status.json").read_text(encoding="utf-8")))
    metrics = cast(dict[str, object], json.loads((tmp_path / "runs" / "no_signal_unit" / "metrics.json").read_text(encoding="utf-8")))

    assert status["status"] == NO_SIGNAL
    assert "Reward signal is all zero" in str(status["recommendation"])
    assert metrics["status"] == NO_SIGNAL


def test_training_status_reports_completed_for_nonzero_returns():
    status = training_status({"all_step_rewards_zero": False, "episode_returns": [1.25]})

    assert status["status"] == "COMPLETED"


def test_device_allocation_covers_all_healthy_visible_gpus(tmp_path: Path):
    resources_path = write_resources(tmp_path)
    inventory_path = tmp_path / "device_inventory.json"
    _ = inventory_path.write_text(
        json.dumps(
            {
                "gpus": [
                    {"index": 0, "memory_free_mb": 48000},
                    {"index": 1, "memory_free_mb": 47000},
                    {"index": 2, "memory_free_mb": 46000},
                    {"index": 3, "memory_free_mb": 45000},
                ],
                "swebench_workers": {"recommended_swebench_max_workers": 8},
            }
        ),
        encoding="utf-8",
    )

    allocation = allocate_rollout_workers(resources_path, rollout_count=8, inventory_path=inventory_path)
    workers = cast(list[dict[str, object]], allocation["workers"])

    assert allocation["healthy_visible_gpus"] == [0, 1, 2, 3]
    assert {worker["gpu_id"] for worker in workers} == {0, 1, 2, 3}
    assert all(cast(int, worker["planned_rollouts"]) >= 2 for worker in workers)
    assert allocation["missing_gpus"] == []


def write_learning_config(tmp_path: Path, *, reward_config: Path) -> Path:
    config = {
        "agent": {
            "instances": [
                {
                    "fixture": {
                        "files": {
                            "README.md": "learning fixture\n",
                            "math_utils.py": "def add_numbers(left, right):\n    return left - right\n",
                            "tests/test_math_utils.py": "from math_utils import add_numbers\n\n\ndef test_add_numbers_visible():\n    assert add_numbers(2, 3) == 5\n",
                        }
                    },
                    "instance_id": "learning-unit-0001",
                    "problem_statement": "The visible test for add_numbers fails because the helper should add two numbers.",
                    "repo": "local/learning-fixture",
                    "visible_failures": {"visible-failure": "AssertionError: add_numbers(2, 3) should equal 5"},
                    "visible_tests": ["tests/test_math_utils.py"],
                }
            ],
            "max_output_chars": 4000,
            "max_steps": 12,
            "max_test_runs": 1,
            "model_name_or_path": "rule_based_local",
            "test_timeout_seconds": 5.0,
            "type": "learning",
        },
        "dry_run": {"instances": [{"instance_id": "dry-unit", "repo": "local/dry"}]},
        "learning": {
            "action_vocabulary": list(ACTION_VOCABULARY),
            "baseline_decay": 0.9,
            "gamma": 1.0,
            "learning_rate": 0.05,
            "reward_config": str(reward_config),
            "tool_schema_version": ACTION_SCHEMA_VERSION,
        },
        "run": {"name": "learning", "output_dir": str(tmp_path / "runs")},
    }
    path = tmp_path / "learning.yaml"
    _ = path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def write_resources(tmp_path: Path) -> Path:
    resources = {
        "cpu": {"max_workers": 8},
        "device_policy": "maximize_local",
        "docker_cache_level": "env",
        "fallback": {"max_retries": 1, "on_cpu_saturated": "wait_and_retry", "on_gpu_oom": "reduce_batch_and_retry", "on_gpu_unavailable": "record_and_continue"},
        "gpus": {"expected_ids": [0, 1, 2, 3], "per_device": {"fallback": None, "min_memory_mb": 4096}},
        "memory": {"per_swebench_worker_mb": 8192, "reserve_mb": 8192},
        "model_shards": {"max_gpus_per_model": 4, "per_gpu_batch_size": 1, "strategy": "device_map_auto"},
        "swebench_max_workers": "auto",
        "trainer_devices": {"policy_device": 0, "rollout_gpus": [0, 1, 2, 3], "rollout_parallelism": 4},
    }
    path = tmp_path / "resources.yaml"
    _ = path.write_text(yaml.safe_dump(resources, sort_keys=True), encoding="utf-8")
    return path
