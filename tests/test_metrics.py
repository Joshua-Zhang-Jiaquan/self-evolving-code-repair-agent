from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import yaml

from repair_agent.evaluation.metrics import (
    denominator_counts,
    pass_at_k,
    summarize_model_gates,
    summarize_official_harness,
    summarize_resources,
    validate_predictions_file,
)
from repair_agent.evaluation.summarize import main as summarize_main


def test_validate_predictions_rejects_missing_model_patch(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        json.dumps({"instance_id": "case-1", "model_name_or_path": "local-model"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_patch"):
        _ = validate_predictions_file(predictions)


def test_error_instances_count_in_submitted_denominator():
    counts = denominator_counts(
        [
            {"instance_id": "resolved", "final_status": "passed", "model_patch": "diff --git a b"},
            {"instance_id": "unresolved", "final_status": "patch_unverified", "model_patch": "diff --git a b"},
            {"instance_id": "empty", "final_status": "no_patch", "model_patch": ""},
            {"instance_id": "timeout", "final_status": "timeout", "model_patch": "diff --git a b"},
        ]
    )

    assert counts["resolved"] == 1
    assert counts["unresolved"] == 1
    assert counts["empty"] == 1
    assert counts["error"] == 1
    assert counts["denominator"] == 4
    assert counts["resolved_rate"] == 0.25


def test_pass_at_k_uses_first_k_attempts_per_instance():
    attempts = {"case-a": [False, True], "case-b": [False, False], "case-c": [True]}

    first = pass_at_k(attempts, 1)
    second = pass_at_k(attempts, 2)
    assert first is not None and abs(first - (1 / 3)) < 0.000001
    assert second is not None and abs(second - (2 / 3)) < 0.000001


def test_resource_usage_summary_includes_all_visible_devices():
    summary = summarize_resources(
        [
            {
                "assigned_device": "cuda:0",
                "fallback": {"reasons": ["gpu_3_reserved"]},
                "gpu_memory_peak_mb": {"0": 12000, "2": 9000},
                "gpu_utilization_percent": {"0": 75.0},
                "visible_gpus": [0, 1, 2, 3],
                "worker_settings": {"cpu_max_workers": 8, "swebench_max_workers": 2},
            }
        ]
    )

    assert summary["visible_device_ids"] == [0, 1, 2, 3]
    device_utilization = cast(dict[str, dict[str, object]], summary["device_utilization"])
    gpu_memory_peak = cast(dict[str, dict[str, object]], summary["gpu_memory_peak"])
    assert set(device_utilization) == {"0", "1", "2", "3"}
    assert device_utilization["0"]["assigned_task_count"] == 1
    assert device_utilization["3"]["assigned_task_count"] == 0
    assert gpu_memory_peak["2"]["memory_peak_mb"] == 9000
    assert gpu_memory_peak["1"]["source"] == "not_reported"
    assert summary["fallback_reasons"] == {"gpu_3_reserved": 1}


def test_summary_cli_writes_synthetic_run_with_resources_and_gates(tmp_path: Path):
    runs = tmp_path / "outputs" / "runs"
    run_dir = runs / "baseline_synthetic"
    run_dir.mkdir(parents=True)
    _write_jsonl(
        run_dir / "predictions.jsonl",
        [{"instance_id": "case-1", "model_name_or_path": "local", "model_patch": "--- a/x.py\n+++ b/x.py\n"}],
    )
    _write_jsonl(
        run_dir / "trajectories.jsonl",
        [
            {
                "final_status": "passed",
                "instance_id": "case-1",
                "run_name": "baseline",
                "status": "ok",
                "test_run_count": 1,
                "tool": "run_tests",
                "tool_call_count": 4,
            }
        ],
    )
    _ = (run_dir / "metrics.json").write_text(
        json.dumps({"instances": [{"final_status": "passed", "instance_id": "case-1", "test_run_count": 1, "tool_call_count": 4}]}),
        encoding="utf-8",
    )
    _ = (run_dir / "config.yaml").write_text(yaml.safe_dump({"run": {"name": "baseline"}}), encoding="utf-8")
    _write_jsonl(
        run_dir / "resource_usage.jsonl",
        [{"assigned_device": "cuda:0", "fallback": {"reasons": []}, "visible_gpus": [0, 1]}],
    )
    gate_dir = tmp_path / "outputs" / "model_gates"
    gate_dir.mkdir()
    _ = (gate_dir / "local.json").write_text(
        json.dumps({"device_ids": [0, 1], "model": "local", "reason": "ok", "status": "pass"}),
        encoding="utf-8",
    )
    out = tmp_path / "outputs" / "summary.json"

    assert summarize_main(["--runs", str(runs), "--out", str(out), "--include-resources"]) == 0
    payload = cast(dict[str, object], json.loads(out.read_text(encoding="utf-8")))
    run_rows = cast(list[dict[str, object]], payload["runs"])
    resources = cast(dict[str, object], payload["resources"])
    aggregate = cast(dict[str, object], payload["aggregate"])
    model_gates = cast(dict[str, object], payload["model_gates"])
    models = cast(dict[str, dict[str, object]], model_gates["models"])
    first_run = run_rows[0]
    run_devices = cast(dict[str, dict[str, object]], first_run["device_utilization"])

    assert first_run["run_type"] == "baseline"
    assert first_run["resolved"] == 1
    assert run_devices["1"]["assigned_task_count"] == 0
    assert aggregate["total_denominator"] == 1
    assert payload["missing_runs"] == ["feedback", "learning"]
    assert models["local"]["status"] == "pass"
    assert "gpu_memory_peak" in resources


def test_official_harness_rate_comes_from_report_counts():
    report = summarize_official_harness(
        {"official_harness_executed": True, "resolved": ["a", "b"], "unresolved": ["c"], "empty_patch": ["d"], "error": ["e"]}
    )

    assert report["official"] is True
    official_rate = cast(float, report["official_resolved_rate"])
    assert abs(official_rate - 0.4) < 0.000001
    assert report["denominator"] == 5


def test_model_gate_summary_includes_blocked_reason_and_fallback(tmp_path: Path):
    path = tmp_path / "diffrwkv.json"
    _ = path.write_text(
        json.dumps(
            {
                "device_ids": [0, 1, 2, 3],
                "device_strategy": "per_worker_cuda_visible_devices",
                "fallback": {"reasons": ["not_instruction_model"]},
                "model": "diffrwkv",
                "reason": "trajectory_model",
                "status": "blocked",
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_model_gates([path])

    models = cast(dict[str, dict[str, object]], summary["models"])
    assert models["diffrwkv"]["status"] == "blocked"
    assert models["diffrwkv"]["fallback_reasons"] == ["not_instruction_model"]
    assert summary["fallback_reasons"] == {"not_instruction_model": 1}


def test_summary_handles_missing_optional_files(tmp_path: Path):
    runs = tmp_path / "runs"
    run_dir = runs / "baseline_minimal"
    run_dir.mkdir(parents=True)
    _write_jsonl(run_dir / "predictions.jsonl", [{"instance_id": "case-1", "model_name_or_path": "local", "model_patch": ""}])
    out = tmp_path / "summary.json"

    assert summarize_main(["--runs", str(runs), "--out", str(out)]) == 0
    payload = cast(dict[str, object], json.loads(out.read_text(encoding="utf-8")))
    run_rows = cast(list[dict[str, object]], payload["runs"])

    assert run_rows[0]["empty"] == 1
    assert run_rows[0]["denominator"] == 1
    assert payload["missing_runs"] == ["feedback", "learning"]


def test_summarize_excludes_archived_directories(tmp_path: Path):
    runs = tmp_path / "runs"
    active = runs / "baseline_active"
    archived = runs / "baseline_active.archived.20260101T000000Z"
    active.mkdir(parents=True)
    archived.mkdir(parents=True)
    for d in (active, archived):
        _write_jsonl(d / "predictions.jsonl", [{"instance_id": "case-1", "model_name_or_path": "local", "model_patch": "diff"}])
    out = tmp_path / "summary.json"

    assert summarize_main(["--runs", str(runs), "--out", str(out)]) == 0
    payload = cast(dict[str, object], json.loads(out.read_text(encoding="utf-8")))
    run_rows = cast(list[dict[str, object]], payload["runs"])

    assert len(run_rows) == 1
    assert run_rows[0]["run_id"] == "baseline_active"


def test_real_aggregate_counts_match_across_summary_files(project_root: Path):
    summary_a = project_root / "outputs" / "summary.json"
    summary_b = project_root / "report" / "figures" / "results.json"
    if not summary_a.is_file() or not summary_b.is_file():
        pytest.skip("summary files not found; run summarize first")
    a = cast(dict[str, object], json.loads(summary_a.read_text(encoding="utf-8")))
    b = cast(dict[str, object], json.loads(summary_b.read_text(encoding="utf-8")))
    agg_a = cast(dict[str, object], a["aggregate"])
    agg_b = cast(dict[str, object], b["aggregate"])
    assert agg_a["total_denominator"] == agg_b["total_denominator"]
    assert agg_a["total_resolved"] == agg_b["total_resolved"]
    assert agg_a["run_count"] == agg_b["run_count"]
    assert agg_a["mean_pass_at_1"] == agg_b["mean_pass_at_1"]
    assert agg_a["resolved_rate"] == agg_b["resolved_rate"]


def test_every_run_has_resource_fields(project_root: Path):
    summary_path = project_root / "outputs" / "summary.json"
    if not summary_path.is_file():
        pytest.skip("summary.json not found; run summarize first")
    data = cast(dict[str, object], json.loads(summary_path.read_text(encoding="utf-8")))
    run_rows = cast(list[dict[str, object]], data["runs"])
    assert len(run_rows) > 0
    for run in run_rows:
        run_id = run.get("run_id", "unknown")
        assert "device_utilization" in run, f"{run_id} missing device_utilization"
        assert "gpu_memory_peak" in run, f"{run_id} missing gpu_memory_peak"
        assert "fallback_reasons" in run, f"{run_id} missing fallback_reasons"


def test_report_table_files_nonempty_with_real_run_ids(project_root: Path):
    table_dir = project_root / "report" / "tables"
    expected = ["results_table.md", "ablation_comparison.md", "device_utilization.md"]
    required_ids = {
        "baseline_main",
        "feedback_main",
        "learning_main",
        "ablation_no_process_reward",
        "ablation_no_feedback_features",
        "ablation_reduced_test_budget",
    }
    for name in expected:
        path = table_dir / name
        assert path.is_file(), f"{name} does not exist"
        content = path.read_text(encoding="utf-8")
        assert content.strip(), f"{name} is empty"
        missing = required_ids - set(content.split())
        assert not missing, f"{name} missing run IDs: {missing}"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
