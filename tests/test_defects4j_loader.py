from __future__ import annotations

from pathlib import Path

import pytest

from repair_agent.config import ConfigError
from repair_agent.env import defects4j_loader as loader
from repair_agent.env.defects4j_harness import parse_instance_id


REAL_HEADER = "bug.id,revision.id.buggy,revision.id.fixed,report.id,report.url"


def _write_active_bugs(path: Path, body_lines: list[str], *, header: str = REAL_HEADER) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text("\n".join([header, *body_lines]) + "\n", encoding="utf-8")
    return path


def _write_fake_home(tmp_path: Path, project: str, body_lines: list[str]) -> Path:
    home = tmp_path / "defects4j_home"
    csv_path = home / "framework" / "projects" / project / "active-bugs.csv"
    _ = _write_active_bugs(csv_path, body_lines)
    return home


# --------------------------------------------------------------------------- #
# CSV parsing
# --------------------------------------------------------------------------- #
def test_read_active_bugs_parses_real_header(tmp_path: Path):
    csv_path = _write_active_bugs(
        tmp_path / "Lang" / "active-bugs.csv",
        [
            "1,396afc3e4693cfee182efe582455f2d97058c068,d1a45e9738de5b3e299bb51e987565dcce55fee6,LANG-747,https://issues.apache.org/jira/browse/LANG-747",
            "3,64cfee77e333d0c31a0fde0abe6dac3d97b0f078,8a1042959df80c06dbfa83896594caa8e20ff9d6,LANG-693,https://issues.apache.org/jira/browse/LANG-693",
        ],
    )

    bugs = loader.read_active_bugs(csv_path, "Lang")

    assert [bug.instance_id for bug in bugs] == ["Lang_1", "Lang_3"]
    first = bugs[0]
    assert first.project == "Lang"
    assert first.bug_id == 1
    assert first.revision_id_buggy == "396afc3e4693cfee182efe582455f2d97058c068"
    assert first.revision_id_fixed == "d1a45e9738de5b3e299bb51e987565dcce55fee6"
    assert first.report_id == "LANG-747"
    assert first.report_url == "https://issues.apache.org/jira/browse/LANG-747"


def test_read_active_bugs_sorts_and_dedupes_bug_ids(tmp_path: Path):
    csv_path = _write_active_bugs(
        tmp_path / "Math" / "active-bugs.csv",
        [
            "5,aaaa,bbbb,MATH-934,url5",
            "2,cccc,dddd,MATH-1021,url2",
            "5,eeee,ffff,MATH-DUP,url5dup",
        ],
    )

    bugs = loader.read_active_bugs(csv_path, "Math")

    assert [bug.bug_id for bug in bugs] == [2, 5]
    # First occurrence of the duplicate id wins.
    assert next(bug for bug in bugs if bug.bug_id == 5).revision_id_buggy == "aaaa"


def test_read_active_bugs_skips_blank_and_noninteger_rows(tmp_path: Path):
    csv_path = _write_active_bugs(
        tmp_path / "Cli" / "active-bugs.csv",
        [
            "1,aaaa,bbbb,CLI-1,url1",
            "",
            "not_an_int,xxxx,yyyy,CLI-bad,urlbad",
            "  ,zzzz,wwww,CLI-empty,urlempty",
            "4,cccc,dddd,CLI-4,url4",
        ],
    )

    bugs = loader.read_active_bugs(csv_path, "Cli")

    assert [bug.instance_id for bug in bugs] == ["Cli_1", "Cli_4"]


def test_read_active_bugs_tolerates_missing_optional_columns(tmp_path: Path):
    csv_path = _write_active_bugs(
        tmp_path / "Csv" / "active-bugs.csv",
        ["1,aaaa,bbbb", "2,cccc,dddd"],
        header="bug.id,revision.id.buggy,revision.id.fixed",
    )

    bugs = loader.read_active_bugs(csv_path, "Csv")

    assert [bug.instance_id for bug in bugs] == ["Csv_1", "Csv_2"]
    assert bugs[0].report_id == ""
    assert bugs[0].report_url == ""
    assert bugs[0].revision_id_buggy == "aaaa"


def test_read_active_bugs_rejects_unsupported_project(tmp_path: Path):
    csv_path = _write_active_bugs(tmp_path / "Nope" / "active-bugs.csv", ["1,a,b,N-1,url"])
    with pytest.raises(ConfigError, match="defects4j_unsupported_project"):
        _ = loader.read_active_bugs(csv_path, "Nope")


def test_read_active_bugs_rejects_missing_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="defects4j_active_bugs_not_found"):
        _ = loader.read_active_bugs(tmp_path / "Lang" / "active-bugs.csv", "Lang")


def test_read_active_bugs_rejects_header_without_bug_id(tmp_path: Path):
    csv_path = _write_active_bugs(
        tmp_path / "Lang" / "active-bugs.csv",
        ["1,a,b"],
        header="id,revision.id.buggy,revision.id.fixed",
    )
    with pytest.raises(ConfigError, match="defects4j_active_bugs_header_invalid"):
        _ = loader.read_active_bugs(csv_path, "Lang")


def test_active_bugs_csv_path_rejects_unsupported_project(tmp_path: Path):
    with pytest.raises(ConfigError, match="defects4j_unsupported_project"):
        _ = loader.active_bugs_csv_path(tmp_path, "DefinitelyNotAProject")


# --------------------------------------------------------------------------- #
# id filtering
# --------------------------------------------------------------------------- #
def _sample_bugs() -> list[loader.ActiveBug]:
    return [
        loader.ActiveBug("Lang_1", "Lang", 1, "a1", "b1", "LANG-1", "url1"),
        loader.ActiveBug("Lang_3", "Lang", 3, "a3", "b3", "LANG-3", "url3"),
        loader.ActiveBug("Lang_4", "Lang", 4, "a4", "b4", "LANG-4", "url4"),
    ]


def test_filter_active_bugs_selects_requested_in_order():
    bugs = _sample_bugs()
    selected = loader.filter_active_bugs(bugs, ["Lang_4", "Lang_1"])
    assert [bug.instance_id for bug in selected] == ["Lang_4", "Lang_1"]


def test_filter_active_bugs_dedupes_requested():
    bugs = _sample_bugs()
    selected = loader.filter_active_bugs(bugs, ["Lang_1", "Lang_1", "Lang_3"])
    assert [bug.instance_id for bug in selected] == ["Lang_1", "Lang_3"]


def test_filter_active_bugs_rejects_unknown_id():
    bugs = _sample_bugs()
    with pytest.raises(ConfigError, match="defects4j_instance_missing"):
        _ = loader.filter_active_bugs(bugs, ["Lang_999"])


def test_filter_active_bugs_rejects_invalid_id():
    bugs = _sample_bugs()
    with pytest.raises(ConfigError, match="defects4j_instance_id_invalid"):
        _ = loader.filter_active_bugs(bugs, ["lang_1"])
    with pytest.raises(ConfigError, match="defects4j_instance_id_invalid"):
        _ = loader.filter_active_bugs(bugs, ["Unknown_1"])


def test_parse_id_argument_splits_comma_and_space():
    assert loader.parse_id_argument("Lang_1, Math_5  Cli_2") == ["Lang_1", "Math_5", "Cli_2"]
    assert loader.parse_id_argument(None) == []
    assert loader.parse_id_argument("  ") == []


# --------------------------------------------------------------------------- #
# instance record shape
# --------------------------------------------------------------------------- #
def test_bug_to_instance_record_shape():
    bug = loader.ActiveBug("Lang_1", "Lang", 1, "buggyrev", "fixedrev", "LANG-747", "url")
    record = loader.bug_to_instance_record(bug)

    required = {"source", "language", "instance_id", "repo", "problem_statement", "visible_tests", "visible_failures", "workspace_setup"}
    assert required <= set(record)
    assert record["source"] == loader.DEFECTS4J_INSTANCE_SOURCE
    assert record["language"] == "java"
    assert record["instance_id"] == "Lang_1"
    assert record["repo"] == "defects4j/Lang"
    assert record["visible_tests"] == []
    assert record["visible_failures"] == {}
    assert "Lang_1" in str(record["problem_statement"])
    workspace = record["workspace_setup"]
    assert isinstance(workspace, dict)
    assert workspace["project"] == "Lang"
    assert workspace["bug_id"] == 1
    assert workspace["revision_id_buggy"] == "buggyrev"
    assert workspace["revision_id_fixed"] == "fixedrev"


def test_instance_record_for_id_minimal_without_bug():
    record = loader.instance_record_for_id("Math_5")
    assert record["instance_id"] == "Math_5"
    assert record["source"] == loader.DEFECTS4J_INSTANCE_SOURCE
    assert record["language"] == "java"
    workspace = record["workspace_setup"]
    assert isinstance(workspace, dict)
    assert workspace["project"] == "Math"
    assert workspace["bug_id"] == 5
    assert workspace["revision_id_buggy"] == ""


def test_instance_record_for_id_rejects_invalid():
    with pytest.raises(ConfigError, match="defects4j_instance_id_invalid"):
        _ = loader.instance_record_for_id("not-a-defects4j-id")


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def test_load_defects4j_manifest_parses_smoke_and_main(tmp_path: Path):
    manifest_path = tmp_path / "manifest.yaml"
    _ = manifest_path.write_text(
        "smoke_ids:\n  - Lang_1\n  - Math_5\nmain_ids:\n  - Chart_1\n  - Cli_1\n",
        encoding="utf-8",
    )
    manifest = loader.load_defects4j_manifest(manifest_path)
    assert manifest.smoke_ids == ("Lang_1", "Math_5")
    assert manifest.main_ids == ("Chart_1", "Cli_1")
    assert manifest.all_ids == ("Lang_1", "Math_5", "Chart_1", "Cli_1")
    assert manifest.ids_for_split("smoke") == ("Lang_1", "Math_5")
    assert manifest.ids_for_split("main") == ("Chart_1", "Cli_1")


def test_load_defects4j_manifest_rejects_invalid_id(tmp_path: Path):
    manifest_path = tmp_path / "manifest.yaml"
    _ = manifest_path.write_text("smoke_ids:\n  - django__django-11099\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="defects4j_manifest_invalid_id"):
        _ = loader.load_defects4j_manifest(manifest_path)


def test_manifest_ids_for_split_rejects_unknown_split(tmp_path: Path):
    manifest_path = tmp_path / "manifest.yaml"
    _ = manifest_path.write_text("smoke_ids:\n  - Lang_1\n", encoding="utf-8")
    manifest = loader.load_defects4j_manifest(manifest_path)
    with pytest.raises(ConfigError, match="defects4j_manifest_invalid_split"):
        _ = manifest.ids_for_split("bogus")


def test_bundled_manifest_parses_with_expected_ids(project_root: Path):
    manifest = loader.load_defects4j_manifest(project_root / "configs" / "defects4j_manifest.yaml")
    assert manifest.smoke_ids == ("Lang_1", "Math_5")
    assert len(manifest.main_ids) == 20
    for instance_id in manifest.all_ids:
        assert parse_instance_id(instance_id) is not None


# --------------------------------------------------------------------------- #
# collect_instances
# --------------------------------------------------------------------------- #
def test_collect_instances_from_ids_enriches_from_home(tmp_path: Path):
    home = _write_fake_home(tmp_path, "Lang", ["1,buggyrev,fixedrev,LANG-747,url"])
    records = loader.collect_instances(defects4j_home=home, ids=["Lang_1"])
    assert len(records) == 1
    workspace = records[0]["workspace_setup"]
    assert isinstance(workspace, dict)
    assert workspace["revision_id_buggy"] == "buggyrev"


def test_collect_instances_from_ids_without_home_is_minimal():
    records = loader.collect_instances(defects4j_home=None, ids=["Lang_1", "Math_5"])
    assert [record["instance_id"] for record in records] == ["Lang_1", "Math_5"]
    workspace = records[0]["workspace_setup"]
    assert isinstance(workspace, dict)
    assert workspace["revision_id_buggy"] == ""


def test_collect_instances_from_manifest_split(tmp_path: Path):
    manifest_path = tmp_path / "manifest.yaml"
    _ = manifest_path.write_text(
        "smoke_ids:\n  - Lang_1\n  - Math_5\nmain_ids:\n  - Chart_1\n  - Cli_1\n",
        encoding="utf-8",
    )
    smoke = loader.collect_instances(defects4j_home=None, manifest_path=manifest_path, split="smoke")
    assert [record["instance_id"] for record in smoke] == ["Lang_1", "Math_5"]
    main = loader.collect_instances(defects4j_home=None, manifest_path=manifest_path, split="main")
    assert [record["instance_id"] for record in main] == ["Chart_1", "Cli_1"]


def test_collect_instances_from_projects_requires_home():
    with pytest.raises(ConfigError, match="defects4j_projects_requires_home"):
        _ = loader.collect_instances(defects4j_home=None, projects=["Lang"])


def test_collect_instances_from_projects_loads_all_active_bugs(tmp_path: Path):
    home = _write_fake_home(tmp_path, "Lang", ["1,a1,b1,LANG-1,url1", "3,a3,b3,LANG-3,url3"])
    records = loader.collect_instances(defects4j_home=home, projects=["Lang"])
    assert [record["instance_id"] for record in records] == ["Lang_1", "Lang_3"]


def test_collect_instances_applies_limit():
    records = loader.collect_instances(defects4j_home=None, ids=["Lang_1", "Math_5", "Cli_2"], limit=2)
    assert [record["instance_id"] for record in records] == ["Lang_1", "Math_5"]


def test_collect_instances_rejects_negative_limit():
    with pytest.raises(ConfigError, match="defects4j_limit_invalid"):
        _ = loader.collect_instances(defects4j_home=None, ids=["Lang_1"], limit=0)


def test_collect_instances_requires_some_selector():
    with pytest.raises(ConfigError, match="defects4j_requires_ids_projects_or_manifest"):
        _ = loader.collect_instances(defects4j_home=None)


def test_collect_instances_rejects_invalid_id():
    with pytest.raises(ConfigError, match="defects4j_instance_id_invalid"):
        _ = loader.collect_instances(defects4j_home=None, ids=["Lang_1", "bogus-id"])
