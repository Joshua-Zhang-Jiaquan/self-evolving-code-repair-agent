from __future__ import annotations

from collections.abc import Iterable, Mapping
import importlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, cast

from repair_agent.config import ConfigError, load_yaml_config, require_string


JsonLike: TypeAlias = dict[str, object]
SENSITIVE_BENCHMARK_KEYS = frozenset({"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"})


class DatasetLoader(Protocol):
    def __call__(self, path: str, *, split: str, streaming: bool) -> Iterable[object]: ...


@dataclass(frozen=True)
class TaskManifest:
    dataset_name: str
    split: str
    smoke_ids: tuple[str, ...]
    main_ids: tuple[str, ...]

    @property
    def all_ids(self) -> tuple[str, ...]:
        return (*self.smoke_ids, *self.main_ids)


@dataclass(frozen=True)
class BenchmarkMetadata:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    source: str

    def agent_record(self) -> JsonLike:
        record: JsonLike = {
            "base_commit": self.base_commit,
            "fail_to_pass": list(self.fail_to_pass),
            "hints_text": self.hints_text,
            "instance_id": self.instance_id,
            "pass_to_pass": list(self.pass_to_pass),
            "problem_statement": self.problem_statement,
            "repo": self.repo,
            "source": self.source,
        }
        assert_agent_record_safe(record)
        return record


@dataclass(frozen=True)
class BenchmarkGold:
    instance_id: str
    patch: str
    test_patch: str
    source: str


@dataclass(frozen=True)
class BenchmarkRecord:
    metadata: BenchmarkMetadata
    gold: BenchmarkGold | None = None

    def agent_record(self) -> JsonLike:
        return self.metadata.agent_record()


def load_task_manifest(path: str | Path) -> TaskManifest:
    raw = load_yaml_config(path)
    dataset_name = require_string(
        raw.get("dataset_name", "princeton-nlp/SWE-bench_Lite"),
        "Task manifest field 'dataset_name' must be a non-empty string",
    )
    split = require_string(
        raw.get("split", "test"), "Task manifest field 'split' must be a non-empty string"
    )
    smoke_ids = tuple(_string_list(raw.get("smoke_ids"), "smoke_ids"))
    main_ids = tuple(_string_list(raw.get("main_ids"), "main_ids"))
    _validate_manifest_ids(smoke_ids, main_ids)

    if "smoke_gold_patches" in raw:
        raise ConfigError(
            "Task manifest field 'smoke_gold_patches' is disabled; provide actual local gold data with --gold-source"
        )

    return TaskManifest(
        dataset_name=dataset_name,
        split=split,
        smoke_ids=smoke_ids,
        main_ids=main_ids,
    )


def load_manifest_records(path: str | Path, include_main: bool = True) -> list[BenchmarkRecord]:
    manifest = load_task_manifest(path)
    selected_ids = manifest.all_ids if include_main else manifest.smoke_ids
    return [benchmark_record(instance_id, manifest) for instance_id in selected_ids]


def benchmark_record(instance_id: str, manifest: TaskManifest | None = None) -> BenchmarkRecord:
    metadata = _CATALOG_METADATA.get(instance_id) or _fallback_metadata(instance_id)
    _ = manifest
    return BenchmarkRecord(metadata=metadata, gold=None)


def load_gold_patch_source(path: str | Path) -> dict[str, str]:
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"Gold patch source not found: {source}")
    if source.suffix == ".jsonl":
        return _load_jsonl_gold_patch_source(source)
    raw = load_yaml_config(source)
    patches: dict[str, str] = {}
    for instance_id, value in raw.items():
        if isinstance(value, str):
            patch = value
        elif isinstance(value, dict):
            value_map = cast(dict[object, object], value)
            candidate = value_map.get("patch")
            if not isinstance(candidate, str):
                raise ConfigError(f"Gold patch source entry {instance_id} must contain a string 'patch'")
            patch = candidate
        else:
            raise ConfigError(f"Gold patch source entry {instance_id} must be a string or mapping")
        _store_gold_patch(patches, instance_id, patch, source)
    return patches


def load_dataset_gold_patches(
    dataset_name: str,
    split: str,
    instance_ids: Iterable[str],
    loader: DatasetLoader | None = None,
) -> dict[str, str]:
    wanted = tuple(instance_ids)
    if not wanted:
        raise ConfigError("gold_patch_unavailable: no smoke IDs were requested")
    load_dataset = loader or _import_load_dataset()
    try:
        rows = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as exc:
        raise ConfigError(f"gold_patch_unavailable: could not load dataset {dataset_name} split {split}: {exc}") from exc

    patches: dict[str, str] = {}
    wanted_set = set(wanted)
    for row_object in rows:
        if not isinstance(row_object, Mapping):
            continue
        row = cast(Mapping[object, object], row_object)
        instance_id = row.get("instance_id")
        if instance_id not in wanted_set or not isinstance(instance_id, str):
            continue
        patch = row.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            raise ConfigError(
                f"gold_patch_unavailable: dataset row {instance_id} has missing or empty string patch"
            )
        patches[instance_id] = patch
        if wanted_set <= set(patches):
            break
    missing = [instance_id for instance_id in wanted if instance_id not in patches]
    if missing:
        raise ConfigError(f"gold_patch_unavailable: dataset is missing smoke IDs: {', '.join(missing)}")
    return patches


def smoke_gold_patch(
    instance_id: str, manifest: TaskManifest, gold_patches: Mapping[str, str] | None = None
) -> str:
    _ = manifest
    patch = gold_patches.get(instance_id) if gold_patches is not None else None
    if patch is None:
        message = (
            f"gold_patch_unavailable: actual SWE-bench Lite gold patch is unavailable for smoke "
            f"instance {instance_id}; provide --gold-source with a local/cache export containing instance_id and patch"
        )
        raise ConfigError(message)
    return patch


def agent_records(records: list[BenchmarkRecord]) -> list[JsonLike]:
    return [record.agent_record() for record in records]


def assert_agent_record_safe(record: JsonLike) -> None:
    leaked = _find_sensitive_paths(record)
    if leaked:
        joined = ", ".join(leaked)
        raise ValueError(f"Agent-facing SWE-bench record leaks benchmark gold fields: {joined}")


def sanitize_instance_record(record: object) -> object:
    """Recursively strip benchmark oracle fields before agent prompt/log use.

    Removes every key in ``SENSITIVE_BENCHMARK_KEYS`` (``patch``, ``test_patch``,
    ``FAIL_TO_PASS``, ``PASS_TO_PASS``) from nested mappings, lists, and tuples so
    that no gold or hidden-oracle data reaches an agent-facing surface.
    """
    if isinstance(record, Mapping):
        mapping = cast(Mapping[object, object], record)
        cleaned: dict[str, object] = {}
        for key, value in mapping.items():
            if key in SENSITIVE_BENCHMARK_KEYS:
                continue
            cleaned[str(key)] = sanitize_instance_record(value)
        return cleaned
    if isinstance(record, list):
        list_items = cast(list[object], record)
        return [sanitize_instance_record(item) for item in list_items]
    if isinstance(record, tuple):
        tuple_items = cast(tuple[object, ...], record)
        return tuple(sanitize_instance_record(item) for item in tuple_items)
    return record


def load_task_instances(
    manifest_path: str | Path,
    split: str = "test",
    ids: Iterable[str] | None = None,
    strict: bool = True,
) -> dict[str, list[JsonLike]]:
    """Load actual SWE-bench Lite rows and convert them into agent instance records.

    When ``ids`` is ``None`` the full manifest is loaded and the result is grouped as
    ``{"main": [...], "smoke": [...]}``. When ``ids`` is provided the result is
    ``{"requested": [...]}``.

    In strict mode any requested ID that is not present in the dataset, or any ID that
    looks like a local fixture (contains ``local-`` or lacks ``__``), raises
    ``ConfigError``. In non-strict mode missing IDs are skipped with a warning.

    Every returned record is sanitized so that ``patch``, ``test_patch``,
    ``FAIL_TO_PASS``, and ``PASS_TO_PASS`` never reach the agent.
    """
    manifest = load_task_manifest(manifest_path)
    dataset_name = manifest.dataset_name
    effective_split = split or manifest.split

    if ids is None:
        groups: dict[str, tuple[str, ...]] = {
            "main": manifest.main_ids,
            "smoke": manifest.smoke_ids,
        }
    else:
        requested = tuple(ids)
        if not requested:
            raise ConfigError("swebench_instances_invalid: 'ids' was provided but empty")
        groups = {"requested": requested}

    ordered_ids = tuple(dict.fromkeys(item for group in groups.values() for item in group))
    if not ordered_ids:
        raise ConfigError("swebench_instances_invalid: no instance IDs were requested")

    if strict:
        for instance_id in ordered_ids:
            _assert_official_instance_id(instance_id)

    rows_by_id = _load_instance_rows(dataset_name, effective_split, ordered_ids)

    missing = [instance_id for instance_id in ordered_ids if instance_id not in rows_by_id]
    if missing:
        location = f"dataset {dataset_name} split {effective_split}"
        if strict:
            joined_missing = ", ".join(sorted(missing))
            message = f"swebench_instances_unavailable: {location} is missing requested instance IDs: {joined_missing}"
            raise ConfigError(message)
        for instance_id in sorted(missing):
            message = f"swebench_instances_skipped: {location} is missing requested instance ID {instance_id}"
            warnings.warn(message, stacklevel=2)

    result: dict[str, list[JsonLike]] = {}
    for group_name, group_ids in groups.items():
        records: list[JsonLike] = []
        for instance_id in group_ids:
            row = rows_by_id.get(instance_id)
            if row is None:
                continue
            records.append(_to_agent_instance_record(instance_id, row))
        result[group_name] = records
    return result


def _assert_official_instance_id(instance_id: str) -> None:
    if "local-" in instance_id or "__" not in instance_id:
        message = (
            f"swebench_instances_invalid: instance ID {instance_id!r} looks like a local fixture; "
            "official SWE-bench Lite IDs must contain '__' and must not contain 'local-'"
        )
        raise ConfigError(message)


def _load_instance_rows(
    dataset_name: str, split: str, wanted: tuple[str, ...]
) -> dict[str, Mapping[object, object]]:
    load_dataset = _import_load_dataset()
    try:
        rows = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as exc:
        raise ConfigError(
            f"swebench_instances_unavailable: could not load dataset {dataset_name} split {split}: {exc}"
        ) from exc
    wanted_set = set(wanted)
    found: dict[str, Mapping[object, object]] = {}
    for row_object in rows:
        if not isinstance(row_object, Mapping):
            continue
        row = cast(Mapping[object, object], row_object)
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or instance_id not in wanted_set:
            continue
        _ = found.setdefault(instance_id, row)
        if wanted_set <= set(found):
            break
    return found


def _to_agent_instance_record(instance_id: str, row: Mapping[object, object]) -> JsonLike:
    repo = _require_row_string(row, instance_id, "repo")
    base_commit = _require_row_string(row, instance_id, "base_commit")
    problem_statement = _require_row_string(row, instance_id, "problem_statement")
    hints_text = _optional_row_string(row, "hints_text")
    version = _optional_row_string(row, "version")
    environment_setup_commit = _optional_row_string(row, "environment_setup_commit")
    created_at = _optional_row_string(row, "created_at")
    fail_to_pass_count = len(_node_id_list(row.get("FAIL_TO_PASS")))
    pass_to_pass_count = len(_node_id_list(row.get("PASS_TO_PASS")))

    record: JsonLike = {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "hints_text": hints_text,
        "visible_test_metadata": {
            "oracle_tests_hidden": True,
            "fail_to_pass_count": fail_to_pass_count,
            "pass_to_pass_count": pass_to_pass_count,
        },
        "workspace_setup": {
            "repo": repo,
            "base_commit": base_commit,
            "environment_setup_commit": environment_setup_commit,
            "version": version,
            "created_at": created_at,
        },
        "source": "swebench_lite_official",
        "model_patch": "",
    }
    safe_record = cast(JsonLike, sanitize_instance_record(record))
    assert_agent_record_safe(safe_record)
    return safe_record


def _require_row_string(row: Mapping[object, object], instance_id: str, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"swebench_instances_invalid: dataset row {instance_id} has missing or empty field {key!r}"
        )
    return value


def _optional_row_string(row: Mapping[object, object], key: str) -> str:
    value = row.get(key)
    if isinstance(value, str):
        return value
    return ""


def _node_id_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        try:
            parsed = cast(object, json.loads(stripped))
        except json.JSONDecodeError:
            return (stripped,)
        if isinstance(parsed, list):
            parsed_items = cast(list[object], parsed)
            return tuple(str(item) for item in parsed_items)
        return (str(parsed),)
    if isinstance(value, (list, tuple)):
        sequence_items = cast(Iterable[object], value)
        return tuple(str(item) for item in sequence_items)
    return (str(value),)


def _validate_manifest_ids(smoke_ids: tuple[str, ...], main_ids: tuple[str, ...]) -> None:
    if not 1 <= len(smoke_ids) <= 3:
        raise ConfigError("Task manifest 'smoke_ids' must contain 1-3 instance IDs")
    if not 30 <= len(main_ids) <= 50:
        raise ConfigError("Task manifest 'main_ids' must contain 30-50 instance IDs")
    seen: set[str] = set()
    duplicates: list[str] = []
    for instance_id in (*smoke_ids, *main_ids):
        if instance_id in seen:
            duplicates.append(instance_id)
        seen.add(instance_id)
    if duplicates:
        raise ConfigError(f"Task manifest contains duplicate instance IDs: {', '.join(duplicates)}")


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ConfigError(f"Task manifest field '{field_name}' must be a list")
    result: list[str] = []
    values = cast(list[object], value)
    for index, item in enumerate(values):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"Task manifest field '{field_name}[{index}]' must be a non-empty string")
        result.append(item)
    return result


def _fallback_metadata(instance_id: str) -> BenchmarkMetadata:
    repo = instance_id.split("-", maxsplit=1)[0].replace("__", "/")
    return BenchmarkMetadata(
        instance_id=instance_id,
        repo=repo,
        base_commit="unknown-fixed-manifest",
        problem_statement=f"SWE-bench Lite fixed-manifest task {instance_id}.",
        hints_text="",
        fail_to_pass=(),
        pass_to_pass=(),
        source="fixed_manifest_stub_metadata",
    )


def _find_sensitive_paths(value: object, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        leaked: list[str] = []
        mapping = cast(dict[object, object], value)
        for key, item in mapping.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in SENSITIVE_BENCHMARK_KEYS:
                leaked.append(path)
            leaked.extend(_find_sensitive_paths(item, path))
        return leaked
    if isinstance(value, list):
        list_leaked: list[str] = []
        values = cast(list[object], value)
        for index, item in enumerate(values):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            list_leaked.extend(_find_sensitive_paths(item, path))
        return list_leaked
    return []


def _load_jsonl_gold_patch_source(source: Path) -> dict[str, str]:
    patches: dict[str, str] = {}
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            loaded = cast(object, json.loads(line))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in gold patch source {source} line {line_number}: {exc.msg}") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"Gold patch source {source} line {line_number} must be a JSON object")
        row = cast(dict[object, object], loaded)
        instance_id = row.get("instance_id")
        patch = row.get("patch")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ConfigError(f"Gold patch source {source} line {line_number} must contain string instance_id")
        if not isinstance(patch, str):
            raise ConfigError(f"Gold patch source {source} line {line_number} must contain string patch")
        _store_gold_patch(patches, instance_id, patch, source)
    if not patches:
        raise ConfigError(f"Gold patch source has no patch rows: {source}")
    return patches


def _import_load_dataset() -> DatasetLoader:
    try:
        datasets_module = importlib.import_module("datasets")
    except Exception as exc:
        raise ConfigError(f"gold_patch_unavailable: datasets package is unavailable: {exc}") from exc
    load_dataset = getattr(datasets_module, "load_dataset", None)
    if not callable(load_dataset):
        raise ConfigError("gold_patch_unavailable: datasets.load_dataset is unavailable")
    return cast(DatasetLoader, load_dataset)


def _store_gold_patch(patches: dict[str, str], instance_id: str, patch: str, source: Path) -> None:
    if not instance_id.strip():
        raise ConfigError(f"Gold patch source {source} contains an empty instance_id")
    if not patch.strip():
        raise ConfigError(f"Gold patch source {source} entry {instance_id} has an empty patch")
    if instance_id in patches:
        raise ConfigError(f"Gold patch source {source} contains duplicate instance_id: {instance_id}")
    patches[instance_id] = patch


_CATALOG_METADATA: dict[str, BenchmarkMetadata] = {
    "django__django-11099": BenchmarkMetadata(
        instance_id="django__django-11099",
        repo="django/django",
        base_commit="fixed-manifest-smoke",
        problem_statement="Smoke task: reproduce a Django regression from the SWE-bench Lite test split.",
        hints_text="Gold patch is reserved for harness smoke generation only.",
        fail_to_pass=("tests/model_fields/test_jsonfield.py::TestQuerying::test_key_transform",),
        pass_to_pass=(),
        source="fixed_manifest_smoke_metadata",
    ),
    "sympy__sympy-20590": BenchmarkMetadata(
        instance_id="sympy__sympy-20590",
        repo="sympy/sympy",
        base_commit="fixed-manifest-smoke",
        problem_statement="Smoke task: reproduce a SymPy regression from the SWE-bench Lite test split.",
        hints_text="Gold patch is reserved for harness smoke generation only.",
        fail_to_pass=("sympy/printing/tests/test_str.py::test_issue",),
        pass_to_pass=(),
        source="fixed_manifest_smoke_metadata",
    ),
}
