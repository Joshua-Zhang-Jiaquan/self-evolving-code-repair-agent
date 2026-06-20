"""Scaffold tests: verify that the repair_agent package and project structure exist."""
from __future__ import annotations

import json
import subprocess
import sys

import yaml


# --- Package import tests ---

def test_package_imports():
    """Verify repair_agent can be imported and has expected __name__."""
    import repair_agent
    assert repair_agent.__name__ == "repair_agent"
    assert repair_agent.__version__ == "0.1.0"


def test_subpackage_imports():
    """Verify all subpackages are importable."""
    subpackages = ["agent", "env", "tools", "training", "evaluation"]
    for sp in subpackages:
        mod = __import__(f"repair_agent.{sp}", fromlist=[sp])
        assert mod is not None, f"Failed to import repair_agent.{sp}"


# --- Directory structure tests ---

def test_required_directories_exist(project_root):
    """Verify the core directory layout exists."""
    required_dirs = [
        "repair_agent/agent",
        "repair_agent/env",
        "repair_agent/tools",
        "repair_agent/training",
        "repair_agent/evaluation",
        "scripts",
        "configs",
        "configs/ablations",
        "tests",
        "logs",
        "outputs",
        "report/figures",
        "report/tables",
    ]
    for rel in required_dirs:
        full = project_root / rel
        assert full.is_dir(), f"Missing directory: {rel}"
        # Check for .gitkeep in leaf directories
        assert any(full.iterdir()) or full.joinpath(".gitkeep").exists(), \
            f"Empty directory without .gitkeep: {rel}"


def test_required_files_exist(project_root):
    """Verify key project files are present."""
    required_files = [
        "pyproject.toml",
        "README.md",
        "Dockerfile",
        "environment.yml",
        ".gitignore",
        "configs/resources.yaml",
        "scripts/probe_local_devices.py",
    ]
    for rel in required_files:
        full = project_root / rel
        assert full.is_file(), f"Missing file: {rel}"


# --- Resources YAML config tests ---

def test_resources_yaml_shape(project_root):
    """Verify configs/resources.yaml has the required shape."""
    cfg_path = project_root / "configs" / "resources.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    assert cfg["device_policy"] == "maximize_local", \
        "device_policy must be maximize_local"

    gpus = cfg["gpus"]
    assert "expected_ids" in gpus
    assert set(gpus["expected_ids"]) == {0, 1, 2, 3}, \
        f"Expected GPU IDs {{0,1,2,3}}, got {set(gpus['expected_ids'])}"

    # Validate sub-sections exist
    for key in ["model_shards", "trainer_devices", "cpu", "memory", "disk", "fallback"]:
        assert key in cfg, f"Missing top-level key: {key}"

    # Validate model_shards sub-keys
    ms = cfg["model_shards"]
    assert "strategy" in ms
    assert "max_gpus_per_model" in ms

    # Validate trainer_devices sub-keys
    td = cfg["trainer_devices"]
    assert "policy_device" in td
    assert "rollout_parallelism" in td

    # Validate fallback sub-keys
    fb = cfg["fallback"]
    assert "on_gpu_unavailable" in fb
    assert "on_gpu_oom" in fb


# --- Device probe tests ---

def test_probe_script_executable(project_root):
    """Verify the probe script can be executed and produces valid JSON."""
    probe_script = project_root / "scripts" / "probe_local_devices.py"
    assert probe_script.is_file()
    result = subprocess.run(
        [sys.executable, str(probe_script)],
        capture_output=True, text=True, timeout=30,
        cwd=str(project_root),
    )
    assert result.returncode == 0, f"Probe script failed:\n{result.stderr}"

    data = json.loads(result.stdout)
    # Verify top-level keys
    for key in ["gpus", "cpu", "memory", "disk", "swebench_workers", "resource_summary"]:
        assert key in data, f"Missing key in probe output: {key}"
    assert isinstance(data["gpus"], list)
    assert "logical_cores" in data["cpu"]


def test_probe_writes_to_file(project_root, temp_json_file):
    """Verify probe script writes JSON to the --out path."""
    probe_script = project_root / "scripts" / "probe_local_devices.py"
    result = subprocess.run(
        [sys.executable, str(probe_script), "--out", str(temp_json_file)],
        capture_output=True, text=True, timeout=30,
        cwd=str(project_root),
    )
    assert result.returncode == 0, f"Probe script failed:\n{result.stderr}"
    assert temp_json_file.is_file(), f"Output file not created: {temp_json_file}"
    data = json.loads(temp_json_file.read_text())
    assert "resource_summary" in data
    assert data["resource_summary"]["gpu_usable"] >= 0


def test_probe_includes_fallback_for_missing_gpus(project_root):
    """If expected GPUs are missing, fallback reasons must be recorded."""
    probe_script = project_root / "scripts" / "probe_local_devices.py"
    result = subprocess.run(
        [sys.executable, str(probe_script)],
        capture_output=True, text=True, timeout=30,
        cwd=str(project_root),
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    detected = data.get("gpu_count_detected", 0)
    expected = len(data.get("gpus_expected", [0, 1, 2, 3]))
    # If fewer GPUs were detected than expected, fallback.gpus_missing should not be empty
    fallback = data.get("fallback", {})
    if detected < expected:
        missing = data.get("gpus_missing", [])
        # Should have recorded missing GPU IDs
        assert len(missing) > 0 or fallback.get("missing_gpus"), \
            "Fewer GPUs detected than expected but no fallback reason recorded."


def test_probe_swebench_recommendation_positive(project_root):
    """Verify the probe recommends a positive SWE-bench worker count."""
    probe_script = project_root / "scripts" / "probe_local_devices.py"
    result = subprocess.run(
        [sys.executable, str(probe_script)],
        capture_output=True, text=True, timeout=30,
        cwd=str(project_root),
    )
    data = json.loads(result.stdout)
    workers = data.get("swebench_workers", {})
    rec = workers.get("recommended_swebench_max_workers", 0)
    assert rec >= 1, f"swebench_max_workers must be >= 1, got {rec}"
    assert "rationale" in workers, "swebench_workers missing rationale"


# --- Edge-case / safety tests ---

def test_configs_ablations_dir_is_not_empty(project_root):
    """Verify the configs/ablations directory has a .gitkeep."""
    abl_dir = project_root / "configs" / "ablations"
    assert abl_dir.is_dir()
    contents = list(abl_dir.iterdir())
    assert len(contents) >= 1, f"configs/ablations/ is empty; expected at least .gitkeep"


def test_outputs_has_gitkeep(project_root):
    """Verify outputs/ and subdirs have .gitkeep files."""
    for sub in ["runs", "model_gates"]:
        p = project_root / "outputs" / sub / ".gitkeep"
        assert p.exists(), f"Missing .gitkeep in outputs/{sub}/"


def test_logs_has_gitkeep(project_root):
    """Verify logs/ has .gitkeep or contains files."""
    logs_dir = project_root / "logs"
    assert logs_dir.is_dir()
    gitkeep = logs_dir / ".gitkeep"
    has_gitkeep = gitkeep.exists()
    has_content = any(logs_dir.iterdir())
    assert has_gitkeep or has_content,         f"logs/ is empty and has no .gitkeep: {logs_dir}" 
