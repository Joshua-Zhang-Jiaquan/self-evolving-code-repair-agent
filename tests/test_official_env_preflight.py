from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import cast

import pytest

import scripts.check_official_swebench_env as preflight
from scripts.check_official_swebench_env import main as preflight_main


def _all_found(smoke_ids: list[str], main_ids: list[str]) -> preflight.ConfigMap:
    return {
        "smoke_found": list(smoke_ids),
        "main_found": list(main_ids),
        "missing": [],
        "error": None,
    }


def _manifest_ids(project_root: Path) -> tuple[str, str, list[str], list[str]]:
    return preflight.load_manifest_ids(project_root / "configs" / "task_manifest.yaml")


def _run(project_root: Path, out_path: Path, extra: list[str] | None = None) -> int:
    argv = [
        "--manifest",
        str(project_root / "configs" / "task_manifest.yaml"),
        "--models-config",
        str(project_root / "configs" / "models.yaml"),
        "--resources",
        str(project_root / "configs" / "resources.yaml"),
        "--out",
        str(out_path),
    ]
    if extra:
        argv.extend(extra)
    return preflight_main(argv)


def _load(out_path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(out_path.read_text(encoding="utf-8")))


def _blocker_codes(status: dict[str, object]) -> list[str]:
    blockers = cast(list[dict[str, object]], status["blockers"])
    return [cast(str, b["code"]) for b in blockers]


def _patch_all_available(
    monkeypatch: pytest.MonkeyPatch, project_root: Path
) -> None:
    _, _, smoke_ids, main_ids = _manifest_ids(project_root)
    monkeypatch.setattr(preflight, "swebench_importable", lambda: True)
    monkeypatch.setattr(preflight, "docker_cli_path", lambda: "/usr/bin/docker")
    monkeypatch.setattr(preflight, "docker_daemon_reachable", lambda path: (True, ""))
    monkeypatch.setattr(
        preflight,
        "verify_dataset_ids",
        lambda dataset_name, split, s_ids, m_ids, **_: _all_found(s_ids, m_ids),
    )
    monkeypatch.setattr(preflight, "qwable_availability", lambda model_id, **_: (True, "cache"))


def test_preflight_passes_when_all_available(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    _patch_all_available(monkeypatch, project_root)
    out_path = tmp_path / "official_env_status.json"

    result = _run(project_root, out_path)
    status = _load(out_path)
    _, _, smoke_ids, main_ids = _manifest_ids(project_root)

    assert result == 0
    assert status["status"] == "pass"
    assert _blocker_codes(status) == []
    dataset = cast(dict[str, object], status["dataset"])
    assert dataset["main_ids_count"] == len(main_ids)
    assert dataset["smoke_ids_count"] == len(smoke_ids)
    docker = cast(dict[str, object], status["docker"])
    assert docker["daemon_reachable"] is True
    swebench = cast(dict[str, object], status["swebench"])
    assert swebench["importable"] is True


def test_preflight_blocks_on_missing_docker(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    _patch_all_available(monkeypatch, project_root)
    real_which = shutil.which
    monkeypatch.setattr(
        preflight.shutil,
        "which",
        lambda name: None if name == "docker" else real_which(name),
    )
    monkeypatch.setattr(preflight, "docker_cli_path", lambda: preflight.shutil.which("docker"))
    out_path = tmp_path / "official_env_status.json"

    result = _run(project_root, out_path)
    status = _load(out_path)

    assert result == 1
    assert status["status"] == "blocked"
    assert "docker_cli_unavailable" in _blocker_codes(status)
    docker = cast(dict[str, object], status["docker"])
    assert docker["cli_path"] is None
    assert docker["daemon_reachable"] is False


def test_preflight_blocks_on_missing_swebench(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    _patch_all_available(monkeypatch, project_root)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "swebench" else real_find_spec(name, *a, **k),
    )
    # Restore the real check so it consults the patched find_spec (it was
    # overridden to True by _patch_all_available).
    monkeypatch.setattr(
        preflight,
        "swebench_importable",
        lambda: preflight.importlib.util.find_spec("swebench") is not None,
    )
    out_path = tmp_path / "official_env_status.json"

    result = _run(project_root, out_path)
    status = _load(out_path)

    assert result == 1
    assert status["status"] == "blocked"
    assert "swebench_package_unavailable" in _blocker_codes(status)
    swebench = cast(dict[str, object], status["swebench"])
    assert swebench["importable"] is False


def test_preflight_blocks_on_missing_dataset_ids(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    _patch_all_available(monkeypatch, project_root)

    def missing_dataset(dataset_name: str, split: str, s_ids: list[str], m_ids: list[str], **_):
        return {
            "smoke_found": list(s_ids),
            "main_found": list(m_ids[:-1]),
            "missing": [m_ids[-1]],
            "error": None,
        }

    monkeypatch.setattr(preflight, "verify_dataset_ids", missing_dataset)
    out_path = tmp_path / "official_env_status.json"

    result = _run(project_root, out_path)
    status = _load(out_path)

    assert result == 1
    assert status["status"] == "blocked"
    assert "dataset_ids_missing" in _blocker_codes(status)
    dataset = cast(dict[str, object], status["dataset"])
    assert dataset["missing_ids"] != []


def test_preflight_non_strict_exits_zero_with_blockers(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    # Force a hard official blocker so blockers are non-empty.
    monkeypatch.setattr(preflight, "swebench_importable", lambda: False)
    monkeypatch.setattr(preflight, "docker_cli_path", lambda: None)
    monkeypatch.setattr(
        preflight,
        "verify_dataset_ids",
        lambda dataset_name, split, s_ids, m_ids, **_: _all_found(s_ids, m_ids),
    )
    monkeypatch.setattr(preflight, "qwable_availability", lambda model_id, **_: (True, "cache"))
    out_path = tmp_path / "official_env_status.json"

    result = _run(project_root, out_path, extra=["--no-strict"])
    status = _load(out_path)

    assert result == 0
    assert status["status"] == "blocked"
    assert _blocker_codes(status) != []
    assert status["strict"] is False


def test_preflight_qz_schema_no_token_leak(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    _patch_all_available(monkeypatch, project_root)
    out_path = tmp_path / "official_env_status.json"

    result = _run(project_root, out_path)
    raw_json = out_path.read_text(encoding="utf-8")
    status = _load(out_path)

    assert result == 0
    # The qz auth token is a JWT beginning with "eyJ"; it must never appear.
    assert "eyJ" not in raw_json
    qz = cast(dict[str, object], status["qz"])
    assert "token" not in {k.lower() for k in qz}
    schema_path = qz.get("schema_path")
    if isinstance(schema_path, str):
        schema_text = Path(schema_path).read_text(encoding="utf-8")
        assert "eyJ" not in schema_text


def test_preflight_records_qz_readiness(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    _patch_all_available(monkeypatch, project_root)
    out_path = tmp_path / "official_env_status.json"

    _ = _run(project_root, out_path)
    status = _load(out_path)
    qz = cast(dict[str, object], status["qz"])

    assert "available" in qz
    assert "schema_checked" in qz
    assert qz["dry_run_required"] is True
