from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_configs import (
    compare_budget_fields,
    compare_task_budget_shape,
    diff_configs,
    main as compare_main,
)
from repair_agent.config import load_yaml_config

ABLATIONS_DIR = ROOT / "configs" / "ablations"
LEARNING_PATH = ROOT / "configs" / "learning.yaml"

A1_PATH = ABLATIONS_DIR / "no_process_reward.yaml"
A2_PATH = ABLATIONS_DIR / "no_feedback_features.yaml"
A3_PATH = ABLATIONS_DIR / "reduced_test_budget.yaml"

A1_ALLOWED = {"reward.process_weight"}
A2_ALLOWED = {"learning.report_label", "learning.feedback_features_enabled"}
A3_ALLOWED = {"learning.report_label", "agent.max_steps"}

FROZEN_VOCABULARY = ["search", "read_file", "inspect_test", "edit_file", "run_tests", "rollback", "git_diff", "final_answer"]
FROZEN_SCHEMA = "safe-tool-selection-v1"


# ── helper ────────────────────────────────────────────────────────────────────


def load_config(path: Path) -> dict[str, object]:
    return cast(dict[str, object], yaml.safe_load(path.read_text(encoding="utf-8")))


def assert_no_unexpected_diffs(
    baseline: dict[str, object],
    ablation: dict[str, object],
    allowed: set[str],
    label: str,
) -> None:
    diffs = diff_configs(baseline, ablation, allowed=allowed)
    unexpected = [(d.path, d.left, d.right) for d in diffs if d.path not in allowed]
    assert not unexpected, f"{label}: unexpected diffs {unexpected}"


# ── baseline loading ──────────────────────────────────────────────────────────


def test_learning_yaml_loads():
    cfg = load_yaml_config(LEARNING_PATH)
    assert cast(dict[str, object], cfg["run"])["name"] == "learning"
    assert cast(dict[str, object], cfg["learning"])["tool_schema_version"] == FROZEN_SCHEMA


# ── A1: no process reward ─────────────────────────────────────────────────────


def test_a1_config_loads():
    config = load_yaml_config(A1_PATH)
    reward = cast(dict[str, object], config["reward"])
    assert reward["report_label"] == "A1-no-process-reward"
    assert reward["process_weight"] == 0.0


def test_a1_preserves_frozen_schema_and_vocab():
    config = load_yaml_config(A1_PATH)
    learning = cast(dict[str, object], config["learning"])
    assert learning["tool_schema_version"] == FROZEN_SCHEMA
    assert learning["action_vocabulary"] == FROZEN_VOCABULARY


def test_a1_diff_vs_learning_is_exactly_process_reward_toggle():
    baseline = load_config(LEARNING_PATH)
    ablation = load_config(A1_PATH)
    assert_no_unexpected_diffs(baseline, ablation, A1_ALLOWED, "A1")


def test_a1_budget_fields_match_learning():
    baseline = load_config(LEARNING_PATH)
    ablation = load_config(A1_PATH)
    budget_diffs = compare_budget_fields(baseline, ablation)
    assert budget_diffs == [], f"A1 budget drift: {budget_diffs}"
    shape_diffs = compare_task_budget_shape(baseline, ablation)
    assert shape_diffs == [], f"A1 task shape drift: {shape_diffs}"


def test_a1_compare_configs_cli_no_process_reward():
    exit_code = compare_main(
        [str(LEARNING_PATH), str(A1_PATH), "--allowed-diff", "reward.process_weight"]
    )
    assert exit_code == 0


# ── A2: no feedback features ──────────────────────────────────────────────────


def test_a2_config_loads():
    config = load_yaml_config(A2_PATH)
    learning = cast(dict[str, object], config["learning"])
    assert learning["report_label"] == "A2-no-feedback-features"
    assert learning["feedback_features_enabled"] is False


def test_a2_preserves_frozen_schema_and_vocab():
    config = load_yaml_config(A2_PATH)
    learning = cast(dict[str, object], config["learning"])
    assert learning["tool_schema_version"] == FROZEN_SCHEMA
    assert learning["action_vocabulary"] == FROZEN_VOCABULARY


def test_a2_diff_vs_learning_is_exactly_feedback_features_toggle():
    baseline = load_config(LEARNING_PATH)
    ablation = load_config(A2_PATH)
    assert_no_unexpected_diffs(baseline, ablation, A2_ALLOWED, "A2")


def test_a2_budget_fields_match_learning():
    baseline = load_config(LEARNING_PATH)
    ablation = load_config(A2_PATH)
    budget_diffs = compare_budget_fields(baseline, ablation)
    assert budget_diffs == [], f"A2 budget drift: {budget_diffs}"
    shape_diffs = compare_task_budget_shape(baseline, ablation)
    assert shape_diffs == [], f"A2 task shape drift: {shape_diffs}"


def test_a2_compare_configs_cli_no_feedback_features():
    exit_code = compare_main(
        [
            str(LEARNING_PATH),
            str(A2_PATH),
            "--allowed-diff",
            "learning.report_label",
            "--allowed-diff",
            "learning.feedback_features_enabled",
        ]
    )
    assert exit_code == 0


# ── A3: reduced test budget ───────────────────────────────────────────────────


def test_a3_config_loads():
    config = load_yaml_config(A3_PATH)
    agent = cast(dict[str, object], config.get("agent", {}))
    learning = cast(dict[str, object], config["learning"])
    assert agent["max_steps"] == 6
    assert learning["report_label"] == "A3-reduced-test-budget"


def test_a3_preserves_frozen_schema_and_vocab():
    config = load_yaml_config(A3_PATH)
    learning = cast(dict[str, object], config["learning"])
    assert learning["tool_schema_version"] == FROZEN_SCHEMA
    assert learning["action_vocabulary"] == FROZEN_VOCABULARY


def test_a3_diff_vs_learning_is_exactly_reduced_budget_toggle():
    baseline = load_config(LEARNING_PATH)
    ablation = load_config(A3_PATH)
    assert_no_unexpected_diffs(baseline, ablation, A3_ALLOWED, "A3")


def test_a3_budget_fields_differ_only_in_max_steps():
    baseline = load_config(LEARNING_PATH)
    ablation = load_config(A3_PATH)
    budget_diffs = compare_budget_fields(baseline, ablation)
    assert len(budget_diffs) == 1
    assert budget_diffs[0].path == "agent.max_steps"
    assert budget_diffs[0].left == 12
    assert budget_diffs[0].right == 6
    shape_diffs = compare_task_budget_shape(baseline, ablation)
    assert shape_diffs == [], f"A3 task shape drift: {shape_diffs}"


def test_a3_compare_configs_cli_reduced_test_budget():
    exit_code = compare_main(
        [str(LEARNING_PATH), str(A3_PATH), "--allowed-diff", "learning.report_label", "--allowed-diff", "agent.max_steps"]
    )
    assert exit_code == 0


# ── cross-ablation uniqueness ─────────────────────────────────────────────────


def test_ablation_report_labels_are_unique():
    labels: set[str] = set()
    for path in (A1_PATH, A2_PATH, A3_PATH):
        config = load_yaml_config(path)
        # A1 stores its label in the reward section; A2/A3 in learning
        reward = config.get("reward")
        if isinstance(reward, dict) and "report_label" in reward:
            labels.add(str(cast(dict[str, object], reward)["report_label"]))
        learning = cast(dict[str, object], config["learning"])
        if "report_label" in learning:
            labels.add(str(learning["report_label"]))
    assert len(labels) == 3, f"Duplicate or missing report labels: {labels}"


def test_ablation_configs_have_three_instances_like_learning():
    baseline = load_config(LEARNING_PATH)
    for path in (A1_PATH, A2_PATH, A3_PATH):
        ablation = load_config(path)
        baseline_len = len(cast(list[object], cast(dict[str, object], baseline["agent"])["instances"]))
        ablation_len = len(cast(list[object], cast(dict[str, object], ablation["agent"])["instances"]))
        assert ablation_len == baseline_len, f"{path.name}: instance count {ablation_len} != baseline {baseline_len}"


def test_ablation_configs_preserve_dry_run_like_learning():
    baseline = load_config(LEARNING_PATH)
    baseline_dry = cast(list[object], cast(dict[str, object], baseline["dry_run"])["instances"])
    for path in (A1_PATH, A2_PATH, A3_PATH):
        ablation = load_config(path)
        ablation_dry = cast(list[object], cast(dict[str, object], ablation["dry_run"])["instances"])
        assert len(ablation_dry) == len(baseline_dry), f"{path.name}: dry_run instance count mismatch"


def test_ablation_configs_preserve_model_output_run_dir_like_learning():
    baseline = load_config(LEARNING_PATH)
    ref_model = cast(dict[str, object], baseline["agent"])["model_name_or_path"]
    ref_output = cast(str, cast(dict[str, object], baseline["agent"])["max_output_chars"])
    ref_run = cast(dict[str, object], baseline["run"])
    for path in (A1_PATH, A2_PATH, A3_PATH):
        ablation = load_config(path)
        agent = cast(dict[str, object], ablation["agent"])
        assert agent["model_name_or_path"] == ref_model, f"{path.name}: model_name_or_path changed"
        assert agent["max_output_chars"] == ref_output, f"{path.name}: max_output_chars changed"
        assert ablation["run"] == ref_run, f"{path.name}: run section changed"
