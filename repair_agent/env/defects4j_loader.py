"""Defects4J task loader.

Reads Defects4J ``framework/projects/<PID>/active-bugs.csv`` files and emits
instance records that are compatible with :mod:`repair_agent.run`. Instance ids
are validated with :func:`repair_agent.env.defects4j_harness.parse_instance_id`
so only ids of the shape ``<Project>_<bugId>`` for a supported project are
accepted.

This module is intentionally independent of the SWE-bench Lite loader: it does
not enforce SWE-bench id conventions (``__``) and does not require the
``datasets`` package or network access. When a Defects4J installation is
available it enriches records with the buggy/fixed revision metadata read from
``active-bugs.csv``; without it, minimal-but-valid records are still produced
from the instance id alone.
"""
from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from repair_agent.config import ConfigError, ConfigMap, load_yaml_config
from repair_agent.env.defects4j_harness import (
    SUPPORTED_PROJECTS,
    Defects4JInstance,
    parse_instance_id,
)


# Source tag stamped on every emitted record so ``repair_agent.run`` can route
# Defects4J instances through the Java checkout/test path instead of the
# SWE-bench official path.
DEFECTS4J_INSTANCE_SOURCE = "defects4j"
DEFECTS4J_LANGUAGE = "java"
DEFECTS4J_CHECKOUT_VERSION = "b"

# The real Defects4J active-bugs.csv header.
ACTIVE_BUGS_BUG_ID = "bug.id"
ACTIVE_BUGS_REVISION_BUGGY = "revision.id.buggy"
ACTIVE_BUGS_REVISION_FIXED = "revision.id.fixed"
ACTIVE_BUGS_REPORT_ID = "report.id"
ACTIVE_BUGS_REPORT_URL = "report.url"

INSTANCE_SPLIT_CHOICES = ("main", "smoke")


@dataclass(frozen=True)
class ActiveBug:
    """A single row of a Defects4J ``active-bugs.csv`` file."""

    instance_id: str
    project: str
    bug_id: int
    revision_id_buggy: str
    revision_id_fixed: str
    report_id: str
    report_url: str


@dataclass(frozen=True)
class Defects4JManifest:
    """A pinned set of Defects4J instance ids, mirroring the SWE-bench manifest."""

    smoke_ids: tuple[str, ...]
    main_ids: tuple[str, ...]

    @property
    def all_ids(self) -> tuple[str, ...]:
        return (*self.smoke_ids, *self.main_ids)

    def ids_for_split(self, split: str) -> tuple[str, ...]:
        if split == "smoke":
            return self.smoke_ids
        if split == "main":
            return self.main_ids
        raise ConfigError(f"defects4j_manifest_invalid_split: {split!r} must be one of {INSTANCE_SPLIT_CHOICES}")


def active_bugs_csv_path(defects4j_home: str | Path, project: str) -> Path:
    """Return the path of a project's ``active-bugs.csv`` under a Defects4J home."""
    if project not in SUPPORTED_PROJECTS:
        raise ConfigError(f"defects4j_unsupported_project: {project!r}")
    return Path(defects4j_home) / "framework" / "projects" / project / "active-bugs.csv"


def read_active_bugs(csv_path: str | Path, project: str) -> list[ActiveBug]:
    """Parse a Defects4J ``active-bugs.csv`` into :class:`ActiveBug` records.

    Tolerates the real header
    ``bug.id,revision.id.buggy,revision.id.fixed,report.id,report.url`` and
    skips blank lines, rows with a non-integer ``bug.id``, and duplicate bug
    ids. Rows whose derived instance id is not a supported Defects4J id are
    skipped. The returned list is sorted by ascending ``bug_id``.
    """
    if project not in SUPPORTED_PROJECTS:
        raise ConfigError(f"defects4j_unsupported_project: {project!r}")
    source = Path(csv_path)
    if not source.is_file():
        raise ConfigError(f"defects4j_active_bugs_not_found: {source}")
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"defects4j_active_bugs_unreadable: {source}: {exc}") from exc

    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise ConfigError(f"defects4j_active_bugs_empty: {source}")
    if ACTIVE_BUGS_BUG_ID not in reader.fieldnames:
        header = ",".join(reader.fieldnames)
        raise ConfigError(
            f"defects4j_active_bugs_header_invalid: {source} header {header!r} lacks {ACTIVE_BUGS_BUG_ID!r}"
        )

    bugs: list[ActiveBug] = []
    seen: set[int] = set()
    for row in reader:
        raw_id = (row.get(ACTIVE_BUGS_BUG_ID) or "").strip()
        if not raw_id:
            continue
        try:
            bug_id = int(raw_id)
        except ValueError:
            continue
        if bug_id in seen:
            continue
        instance_id = f"{project}_{bug_id}"
        if parse_instance_id(instance_id) is None:
            continue
        seen.add(bug_id)
        bugs.append(
            ActiveBug(
                instance_id=instance_id,
                project=project,
                bug_id=bug_id,
                revision_id_buggy=(row.get(ACTIVE_BUGS_REVISION_BUGGY) or "").strip(),
                revision_id_fixed=(row.get(ACTIVE_BUGS_REVISION_FIXED) or "").strip(),
                report_id=(row.get(ACTIVE_BUGS_REPORT_ID) or "").strip(),
                report_url=(row.get(ACTIVE_BUGS_REPORT_URL) or "").strip(),
            )
        )
    bugs.sort(key=lambda bug: bug.bug_id)
    return bugs


def load_active_bugs(defects4j_home: str | Path, project: str) -> list[ActiveBug]:
    """Read all active bugs for ``project`` from a Defects4J installation."""
    return read_active_bugs(active_bugs_csv_path(defects4j_home, project), project)


def filter_active_bugs(bugs: Sequence[ActiveBug], ids: Iterable[str]) -> list[ActiveBug]:
    """Return the bugs whose ids match ``ids``, preserving the requested order.

    Raises :class:`ConfigError` for ids that are not valid Defects4J ids or that
    are absent from ``bugs``. Duplicate requested ids are de-duplicated while
    keeping their first occurrence.
    """
    by_id = {bug.instance_id: bug for bug in bugs}
    selected: list[ActiveBug] = []
    seen: set[str] = set()
    for instance_id in ids:
        if parse_instance_id(instance_id) is None:
            raise ConfigError(f"defects4j_instance_id_invalid: {instance_id!r}")
        if instance_id in seen:
            continue
        bug = by_id.get(instance_id)
        if bug is None:
            raise ConfigError(f"defects4j_instance_missing: {instance_id!r} is not in the active-bugs set")
        seen.add(instance_id)
        selected.append(bug)
    return selected


def bug_to_instance_record(bug: ActiveBug) -> ConfigMap:
    """Build a ``repair_agent.run`` instance record from an :class:`ActiveBug`."""
    workspace_setup: ConfigMap = {
        "project": bug.project,
        "bug_id": bug.bug_id,
        "version": DEFECTS4J_CHECKOUT_VERSION,
        "revision_id_buggy": bug.revision_id_buggy,
        "revision_id_fixed": bug.revision_id_fixed,
        "report_id": bug.report_id,
        "report_url": bug.report_url,
    }
    return {
        "instance_id": bug.instance_id,
        "source": DEFECTS4J_INSTANCE_SOURCE,
        "language": DEFECTS4J_LANGUAGE,
        "repo": f"defects4j/{bug.project}",
        "problem_statement": _problem_statement(bug),
        "visible_tests": [],
        "visible_failures": {},
        "workspace_setup": workspace_setup,
    }


def instance_record_for_id(instance_id: str, bug: ActiveBug | None = None) -> ConfigMap:
    """Build an instance record for an id, using ``bug`` metadata when present.

    When ``bug`` is ``None`` (no ``active-bugs.csv`` available) a minimal but
    valid record is produced from the parsed id alone.
    """
    parsed = parse_instance_id(instance_id)
    if parsed is None:
        message = (
            f"defects4j_instance_id_invalid: {instance_id!r} is not a supported Defects4J id "
            "(expected '<Project>_<bugId>' for a supported project)"
        )
        raise ConfigError(message)
    if bug is not None:
        return bug_to_instance_record(bug)
    placeholder = ActiveBug(
        instance_id=parsed.instance_id,
        project=parsed.project,
        bug_id=parsed.bug_id,
        revision_id_buggy="",
        revision_id_fixed="",
        report_id="",
        report_url="",
    )
    return bug_to_instance_record(placeholder)


def load_defects4j_manifest(path: str | Path) -> Defects4JManifest:
    """Load a Defects4J manifest YAML (``smoke_ids`` / ``main_ids``)."""
    raw = load_yaml_config(path)
    smoke_ids = tuple(_id_list(raw.get("smoke_ids"), "smoke_ids"))
    main_ids = tuple(_id_list(raw.get("main_ids"), "main_ids"))
    if not smoke_ids and not main_ids:
        raise ConfigError("defects4j_manifest_invalid: at least one of 'smoke_ids' or 'main_ids' must be non-empty")
    for instance_id in (*smoke_ids, *main_ids):
        if parse_instance_id(instance_id) is None:
            raise ConfigError(f"defects4j_manifest_invalid_id: {instance_id!r} is not a supported Defects4J id")
    return Defects4JManifest(smoke_ids=smoke_ids, main_ids=main_ids)


def parse_id_argument(value: str | None) -> list[str]:
    """Split a comma/space separated CLI argument into a list of ids/projects."""
    if value is None:
        return []
    parts = [chunk.strip() for chunk in value.replace(",", " ").split()]
    return [part for part in parts if part]


def collect_instances(
    *,
    defects4j_home: str | Path | None,
    ids: Sequence[str] | None = None,
    projects: Sequence[str] | None = None,
    manifest_path: str | Path | None = None,
    split: str = "main",
    limit: int | None = None,
) -> list[ConfigMap]:
    """Resolve Defects4J instance records from explicit ids, a manifest, or projects.

    Selection priority: explicit ``ids`` > ``manifest_path`` (``split``) >
    ``projects``. Records are enriched from ``active-bugs.csv`` when
    ``defects4j_home`` is available. Project enumeration requires a home.
    """
    home = Path(defects4j_home) if defects4j_home is not None else None
    explicit_ids = list(ids or [])
    project_names = list(projects or [])

    if explicit_ids:
        records = _records_for_ids(explicit_ids, home)
    elif manifest_path is not None:
        manifest = load_defects4j_manifest(manifest_path)
        records = _records_for_ids(list(manifest.ids_for_split(split)), home)
    elif project_names:
        if home is None:
            raise ConfigError(
                "defects4j_projects_requires_home: a Defects4J home is required to enumerate project bugs"
            )
        records = _records_for_projects(project_names, home)
    else:
        raise ConfigError("defects4j_requires_ids_projects_or_manifest")

    return _apply_limit(records, limit)


def _records_for_ids(ids: Sequence[str], home: Path | None) -> list[ConfigMap]:
    parsed_ids: list[Defects4JInstance] = []
    for instance_id in ids:
        parsed = parse_instance_id(instance_id)
        if parsed is None:
            message = (
                f"defects4j_instance_id_invalid: {instance_id!r} is not a supported Defects4J id "
                "(expected '<Project>_<bugId>' for a supported project)"
            )
            raise ConfigError(message)
        parsed_ids.append(parsed)

    bug_cache: dict[str, ActiveBug] = {}
    if home is not None:
        for project in sorted({parsed.project for parsed in parsed_ids}):
            csv_path = active_bugs_csv_path(home, project)
            if csv_path.is_file():
                for bug in read_active_bugs(csv_path, project):
                    bug_cache[bug.instance_id] = bug

    records: list[ConfigMap] = []
    seen: set[str] = set()
    for parsed in parsed_ids:
        if parsed.instance_id in seen:
            continue
        seen.add(parsed.instance_id)
        records.append(instance_record_for_id(parsed.instance_id, bug_cache.get(parsed.instance_id)))
    return records


def _records_for_projects(projects: Sequence[str], home: Path) -> list[ConfigMap]:
    records: list[ConfigMap] = []
    seen: set[str] = set()
    for project in projects:
        for bug in load_active_bugs(home, project):
            if bug.instance_id in seen:
                continue
            seen.add(bug.instance_id)
            records.append(bug_to_instance_record(bug))
    return records


def _apply_limit(records: list[ConfigMap], limit: int | None) -> list[ConfigMap]:
    if limit is None:
        return records
    if limit < 1:
        raise ConfigError("defects4j_limit_invalid: --limit must be a positive integer")
    return records[:limit]


def _problem_statement(bug: ActiveBug) -> str:
    report = f" (issue {bug.report_id})" if bug.report_id else ""
    return (
        f"Repair Defects4J bug {bug.instance_id}{report}. The buggy revision of the {bug.project} "
        "project fails its triggering tests; produce a source patch so that the relevant Defects4J "
        "tests pass."
    )


def _id_list(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"defects4j_manifest_invalid: field {field_name!r} must be a list")
    values = cast(list[object], value)
    result: list[str] = []
    for index, item in enumerate(values):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"defects4j_manifest_invalid: field '{field_name}[{index}]' must be a non-empty string")
        result.append(item.strip())
    return result
