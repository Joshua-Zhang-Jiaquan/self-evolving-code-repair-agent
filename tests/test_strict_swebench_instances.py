from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pytest

from repair_agent.config import ConfigError
from repair_agent.env.swebench_loader import (
    DatasetLoader,
    assert_agent_record_safe,
    load_task_instances,
    load_task_manifest,
    sanitize_instance_record,
)


_GOLD_MARKER = "GOLD-SECRET"


def _fake_row(instance_id: str) -> dict[str, object]:
    """Build a fake HuggingFace SWE-bench Lite row that carries oracle fields.

    The ``patch``/``test_patch``/``FAIL_TO_PASS``/``PASS_TO_PASS`` fields are seeded
    with the ``GOLD-SECRET`` marker so tests can prove they never reach a record.
    """
    org, rest = instance_id.split("__", 1)
    name = rest.rsplit("-", 1)[0]
    return {
        "instance_id": instance_id,
        "repo": f"{org}/{name}",
        "base_commit": f"basecommit-{instance_id}",
        "problem_statement": f"Problem statement for {instance_id}.",
        "hints_text": f"hint for {instance_id}",
        "version": "1.0",
        "environment_setup_commit": f"envcommit-{instance_id}",
        "created_at": "2020-01-01T00:00:00Z",
        "patch": f"diff --git a/g.py b/g.py {_GOLD_MARKER}-PATCH-{instance_id}",
        "test_patch": f"diff --git a/t.py b/t.py {_GOLD_MARKER}-TESTPATCH-{instance_id}",
        "FAIL_TO_PASS": json.dumps(
            [f"tests/test_{name}.py::test_fail_{index}" for index in range(2)]
        ),
        "PASS_TO_PASS": json.dumps(
            [f"tests/test_{name}.py::test_pass_{index}" for index in range(3)]
        ),
    }


def _make_fake_loader(rows: list[dict[str, object]]) -> DatasetLoader:
    def fake_loader(path: str, *, split: str, streaming: bool) -> Iterable[object]:
        _ = (path, split, streaming)
        yield from rows

    return fake_loader


def _patch_loader(
    monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]
) -> None:
    loader = _make_fake_loader(rows)
    monkeypatch.setattr(
        "repair_agent.env.swebench_loader._import_load_dataset", lambda: loader
    )


_EXPECTED_RECORD_KEYS = {
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "hints_text",
    "visible_test_metadata",
    "workspace_setup",
    "source",
    "model_patch",
}


def test_load_all_manifest_ids(monkeypatch: pytest.MonkeyPatch, project_root: Path):
    manifest_path = project_root / "configs" / "task_manifest.yaml"
    manifest = load_task_manifest(manifest_path)
    _patch_loader(monkeypatch, [_fake_row(instance_id) for instance_id in manifest.all_ids])

    result = load_task_instances(manifest_path, strict=True)

    assert set(result.keys()) == {"main", "smoke"}
    assert len(result["main"]) == 40
    assert len(result["smoke"]) == 2
    assert {str(record["instance_id"]) for record in result["main"]} == set(manifest.main_ids)
    assert {str(record["instance_id"]) for record in result["smoke"]} == set(manifest.smoke_ids)
    assert all("__" in str(record["instance_id"]) for record in result["main"])

    for record in [*result["main"], *result["smoke"]]:
        assert set(record.keys()) == _EXPECTED_RECORD_KEYS
        assert record["source"] == "swebench_lite_official"
        assert record["model_patch"] == ""

    metadata = cast(dict[str, object], result["smoke"][0]["visible_test_metadata"])
    assert metadata["fail_to_pass_count"] == 2
    assert metadata["pass_to_pass_count"] == 3
    assert metadata["oracle_tests_hidden"] is True


def test_smoke_ids_load(monkeypatch: pytest.MonkeyPatch, project_root: Path):
    manifest_path = project_root / "configs" / "task_manifest.yaml"
    manifest = load_task_manifest(manifest_path)
    _patch_loader(monkeypatch, [_fake_row(instance_id) for instance_id in manifest.smoke_ids])

    result = load_task_instances(manifest_path, ids=list(manifest.smoke_ids), strict=True)

    assert set(result.keys()) == {"requested"}
    assert len(result["requested"]) == 2
    assert {str(record["instance_id"]) for record in result["requested"]} == set(manifest.smoke_ids)
    for record in result["requested"]:
        workspace = cast(dict[str, object], record["workspace_setup"])
        assert workspace["version"] == "1.0"
        assert str(workspace["environment_setup_commit"]).startswith("envcommit-")


def test_strict_mode_rejects_local_fixture_ids(
    monkeypatch: pytest.MonkeyPatch, project_root: Path
):
    manifest_path = project_root / "configs" / "task_manifest.yaml"

    def must_not_load() -> DatasetLoader:
        raise AssertionError("dataset loader must not run for local fixture IDs")

    monkeypatch.setattr(
        "repair_agent.env.swebench_loader._import_load_dataset", must_not_load
    )

    with pytest.raises(ConfigError, match="local fixture"):
        _ = load_task_instances(manifest_path, ids=["baseline-local-0001"], strict=True)

    with pytest.raises(ConfigError, match="local fixture"):
        _ = load_task_instances(manifest_path, ids=["no-double-underscore"], strict=True)


def test_strict_mode_rejects_missing_ids(
    monkeypatch: pytest.MonkeyPatch, project_root: Path
):
    manifest_path = project_root / "configs" / "task_manifest.yaml"
    present = "django__django-11099"
    absent = "django__django-99999"
    _patch_loader(monkeypatch, [_fake_row(present)])

    with pytest.raises(ConfigError, match="missing requested instance IDs"):
        _ = load_task_instances(manifest_path, ids=[present, absent], strict=True)


def test_non_strict_mode_skips_missing_ids_with_warning(
    monkeypatch: pytest.MonkeyPatch, project_root: Path
):
    manifest_path = project_root / "configs" / "task_manifest.yaml"
    present = "django__django-11099"
    absent = "django__django-99999"
    _patch_loader(monkeypatch, [_fake_row(present)])

    with pytest.warns(UserWarning, match="missing requested instance ID"):
        result = load_task_instances(manifest_path, ids=[present, absent], strict=False)

    assert [str(record["instance_id"]) for record in result["requested"]] == [present]


def test_sanitizer_removes_hidden_fields():
    raw = {
        "instance_id": "django__django-11099",
        "repo": "django/django",
        "patch": f"{_GOLD_MARKER}-PATCH",
        "test_patch": f"{_GOLD_MARKER}-TESTPATCH",
        "FAIL_TO_PASS": ["tests/x.py::test_a"],
        "PASS_TO_PASS": ["tests/x.py::test_b"],
        "nested": {
            "patch": f"{_GOLD_MARKER}-NESTED",
            "keep": "ok",
            "deeper": [{"test_patch": f"{_GOLD_MARKER}-DEEP", "fine": 1}],
        },
    }

    result = sanitize_instance_record(raw)

    assert isinstance(result, dict)
    sanitized = cast(dict[str, object], result)
    assert "patch" not in sanitized
    assert "test_patch" not in sanitized
    assert "FAIL_TO_PASS" not in sanitized
    assert "PASS_TO_PASS" not in sanitized
    assert sanitized["instance_id"] == "django__django-11099"
    assert sanitized["repo"] == "django/django"

    nested = cast(dict[str, object], sanitized["nested"])
    assert "patch" not in nested
    assert nested["keep"] == "ok"
    deeper = cast(list[object], nested["deeper"])
    assert deeper[0] == {"fine": 1}

    blob = str(sanitized)
    assert _GOLD_MARKER not in blob
    assert "test_patch" not in blob
    assert "FAIL_TO_PASS" not in blob
    assert "PASS_TO_PASS" not in blob


def test_agent_records_have_no_hidden_fields(
    monkeypatch: pytest.MonkeyPatch, project_root: Path
):
    manifest_path = project_root / "configs" / "task_manifest.yaml"
    manifest = load_task_manifest(manifest_path)
    _patch_loader(monkeypatch, [_fake_row(instance_id) for instance_id in manifest.all_ids])

    result = load_task_instances(manifest_path, strict=True)

    for record in [*result["main"], *result["smoke"]]:
        assert_agent_record_safe(record)
        assert "patch" not in record
        assert "test_patch" not in record
        assert "FAIL_TO_PASS" not in record
        assert "PASS_TO_PASS" not in record
        assert record["model_patch"] == ""

    blob = str(result)
    assert _GOLD_MARKER not in blob
    assert "test_patch" not in blob
    assert "FAIL_TO_PASS" not in blob
    assert "PASS_TO_PASS" not in blob
