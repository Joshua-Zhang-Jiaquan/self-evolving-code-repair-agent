#!/usr/bin/env python3
"""Official SWE-bench Lite and qz offload environment preflight.

Checks the strict prerequisites for running the official SWE-bench harness
(``swebench`` import, Docker CLI + daemon, dataset row availability, disk
headroom, resource config, Qwable model source) plus qz cluster offload
readiness, then writes a machine-readable status JSON.

Strict mode (default) exits nonzero when any blocker is recorded. A qz offload
blocker is only recorded (and therefore only fails strict mode) when a stage is
classified as requiring offload and qz is unavailable. Non-strict mode always
exits 0 but still records every blocker.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repair_agent.config import (  # noqa: E402
    ConfigError,
    ConfigMap,
    load_yaml_config,
    require_string,
)
from repair_agent.logging import write_json_atomic  # noqa: E402
from repair_agent.resources import load_resource_config, resolve_resource_plan  # noqa: E402

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_SPLIT = "test"
DEFAULT_QWABLE_ID = "lordx64/Qwable-v1"
DEFAULT_OUT = Path("outputs/official_env_status.json")
DEFAULT_MIN_FREE_GB = 50.0
STRICT_DISK_FLOOR_GB = 120.0
QZ_SCHEMA_FILENAME = "train.CreateJob.schema.yaml"
QZ_SCHEMA_ACTION = "train.CreateJob"

OFFICIAL_BLOCKER_CODES = frozenset(
    {
        "swebench_package_unavailable",
        "docker_cli_unavailable",
        "docker_daemon_unreachable",
        "dataset_ids_missing",
        "dataset_access_failed",
        "disk_below_threshold",
    }
)

# JWT-like tokens (the qz auth token starts with ``eyJ``) and any ``key: value``
# secret lines must never be persisted. Used to scrub anything before storage.
_TOKEN_PATTERN = re.compile(
    r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"
    + r"|(?:token|secret|password|api[_-]?key)\s*[:=]\s*\S+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PreflightArgs:
    manifest: Path
    models_config: Path
    resources: Path
    out: Path
    strict: bool


DatasetLoader = Callable[[str, str], Iterable[ConfigMap]]


# --------------------------------------------------------------------------- #
# Individual environment checks (module-level so tests can monkeypatch them).
# --------------------------------------------------------------------------- #
def swebench_importable() -> bool:
    return importlib.util.find_spec("swebench") is not None


def docker_cli_path() -> str | None:
    return shutil.which("docker")


def docker_daemon_reachable(docker_path: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [docker_path, "info"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, exc.__class__.__name__
    if completed.returncode != 0:
        detail = _short_text(completed.stderr or completed.stdout)
        return False, detail or str(completed.returncode)
    return True, ""


def _stream_dataset(dataset_name: str, split: str) -> Iterator[ConfigMap]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    for row in dataset:
        if isinstance(row, dict):
            yield cast(ConfigMap, row)


def verify_dataset_ids(
    dataset_name: str,
    split: str,
    smoke_ids: list[str],
    main_ids: list[str],
    *,
    loader: DatasetLoader | None = None,
) -> ConfigMap:
    targets = list(smoke_ids) + list(main_ids)
    remaining = set(targets)
    found: set[str] = set()
    error: str | None = None
    source: DatasetLoader = loader if loader is not None else _stream_dataset
    try:
        for row in source(dataset_name, split):
            instance_id = row.get("instance_id")
            if isinstance(instance_id, str) and instance_id in remaining:
                found.add(instance_id)
                remaining.discard(instance_id)
                if not remaining:
                    break
    except Exception as exc:  # dataset/network boundary must not crash the preflight
        error = f"{exc.__class__.__name__}: {_mask_tokens(str(exc))[:200]}"
    return {
        "smoke_found": [i for i in smoke_ids if i in found],
        "main_found": [i for i in main_ids if i in found],
        "missing": [i for i in targets if i not in found],
        "error": error,
    }


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / float(1024**3)


def qwable_availability(model_id: str, *, cache_root: Path | None = None) -> tuple[bool, str]:
    if _qwable_cache_present(model_id, cache_root):
        return True, "cache"
    if _qwable_network_present(model_id):
        return True, "network"
    return False, "unavailable"


def _qwable_cache_present(model_id: str, cache_root: Path | None) -> bool:
    root = cache_root if cache_root is not None else Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_id.replace("/", "--")
    return (root / slug).is_dir()


def _qwable_network_present(model_id: str) -> bool:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False
    try:
        _ = HfApi().model_info(model_id)
    except Exception:  # HF/HTTP/repo errors must not crash the preflight
        return False
    return True


def qz_cli_path() -> str | None:
    return shutil.which("qz")


def qz_auth_configured(qz_path: str) -> bool:
    """Return whether ``qz config get`` succeeds. The stdout is NEVER read or
    stored because it contains the auth token; only the return code is used."""
    try:
        completed = subprocess.run(
            [qz_path, "config", "get"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def qz_capture_schema(qz_path: str, out_path: Path) -> bool:
    try:
        completed = subprocess.run(
            [qz_path, "schema", QZ_SCHEMA_ACTION, "-o", "yaml"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if completed.returncode != 0:
        return False
    content = _mask_tokens(completed.stdout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _ = out_path.write_text(content, encoding="utf-8")
    return True


def resources_worker_settings(resources_path: Path) -> tuple[dict[str, int | str], float]:
    config = load_resource_config(resources_path)
    plan = resolve_resource_plan(config, None, None)
    return plan.worker_settings, _disk_min_free_gb(config)


# --------------------------------------------------------------------------- #
# Status assembly.
# --------------------------------------------------------------------------- #
def build_status(
    *,
    dataset_name: str,
    split: str,
    smoke_ids: list[str],
    main_ids: list[str],
    qwable_model_id: str,
    resources_path: Path,
    out_path: Path,
    strict: bool,
) -> ConfigMap:
    blockers: list[ConfigMap] = []

    swebench_section = _swebench_section(blockers)
    docker_section, daemon_reachable = _docker_section(blockers)
    dataset_section = _dataset_section(dataset_name, split, smoke_ids, main_ids, blockers)
    resources_section, min_free_gb = _resources_section(resources_path, blockers)
    disk_section = _disk_section(min_free_gb, strict, blockers)
    qwable_section = _qwable_section(qwable_model_id, blockers)
    qz_section = _qz_section(out_path, daemon_reachable, blockers)

    status_value = "blocked" if blockers else "pass"
    return {
        "blockers": blockers,
        "dataset": dataset_section,
        "disk": disk_section,
        "docker": docker_section,
        "qwable": qwable_section,
        "qz": qz_section,
        "resources": resources_section,
        "status": status_value,
        "strict": strict,
        "swebench": swebench_section,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _swebench_section(blockers: list[ConfigMap]) -> ConfigMap:
    importable = swebench_importable()
    if not importable:
        blockers.append(
            {
                "code": "swebench_package_unavailable",
                "fix_command": 'pip install -e ".[swebench]"',
            }
        )
    return {"importable": importable}


def _docker_section(blockers: list[ConfigMap]) -> tuple[ConfigMap, bool]:
    cli_path = docker_cli_path()
    daemon_reachable = False
    if cli_path is None:
        blockers.append(
            {
                "code": "docker_cli_unavailable",
                "fix_command": "install Docker Engine and ensure 'docker' is on PATH",
            }
        )
    else:
        daemon_reachable, detail = docker_daemon_reachable(cli_path)
        if not daemon_reachable:
            blockers.append(
                {
                    "code": "docker_daemon_unreachable",
                    "fix_command": f"start the Docker daemon (detail: {detail or 'unreachable'})",
                }
            )
    return {"cli_path": cli_path, "daemon_reachable": daemon_reachable}, daemon_reachable


def _dataset_section(
    dataset_name: str,
    split: str,
    smoke_ids: list[str],
    main_ids: list[str],
    blockers: list[ConfigMap],
) -> ConfigMap:
    result = verify_dataset_ids(dataset_name, split, smoke_ids, main_ids)
    smoke_found = cast(list[str], result.get("smoke_found", []))
    main_found = cast(list[str], result.get("main_found", []))
    missing = cast(list[str], result.get("missing", []))
    error = result.get("error")
    if error is not None:
        blockers.append(
            {
                "code": "dataset_access_failed",
                "fix_command": (
                    f"ensure network/HF access to {dataset_name} split {split} "
                    + "or pre-cache the dataset"
                ),
            }
        )
    elif missing:
        blockers.append(
            {
                "code": "dataset_ids_missing",
                "fix_command": (
                    f"verify {len(missing)} missing instance_id(s) exist in "
                    f"{dataset_name} split {split}"
                ),
            }
        )
    return {
        "name": dataset_name,
        "split": split,
        "main_ids_found": main_found,
        "smoke_ids_found": smoke_found,
        "main_ids_count": len(main_found),
        "smoke_ids_count": len(smoke_found),
        "missing_ids": missing,
        "access_error": error,
    }


def _resources_section(
    resources_path: Path, blockers: list[ConfigMap]
) -> tuple[ConfigMap, float]:
    try:
        worker_settings, min_free_gb = resources_worker_settings(resources_path)
    except ConfigError as exc:
        blockers.append(
            {
                "code": "resources_unparseable",
                "fix_command": "fix worker/GPU/disk settings in configs/resources.yaml",
            }
        )
        return {"parseable": False, "worker_settings": {}, "error": _short_text(str(exc))}, DEFAULT_MIN_FREE_GB
    return {"parseable": True, "worker_settings": worker_settings}, min_free_gb


def _disk_section(min_free_gb: float, strict: bool, blockers: list[ConfigMap]) -> ConfigMap:
    threshold = max(min_free_gb, STRICT_DISK_FLOOR_GB) if strict else min_free_gb
    free = disk_free_gb(_disk_probe_path())
    meets = free >= threshold
    if not meets:
        blockers.append(
            {
                "code": "disk_below_threshold",
                "fix_command": f"free disk space to reach >= {threshold:.0f} GB free",
            }
        )
    return {
        "free_gb": round(free, 2),
        "threshold_gb": round(threshold, 2),
        "meets_threshold": meets,
    }


def _qwable_section(model_id: str, blockers: list[ConfigMap]) -> ConfigMap:
    available, source = qwable_availability(model_id)
    if not available:
        blockers.append(
            {
                "code": "qwable_unavailable",
                "fix_command": (
                    f"ensure HF access to {model_id} or pre-cache the model card "
                    + "(no full weight download required)"
                ),
            }
        )
    return {"source": source, "available": available, "cache_or_network": source}


def _qz_section(
    out_path: Path, daemon_reachable: bool, blockers: list[ConfigMap]
) -> ConfigMap:
    cli_path = qz_cli_path()
    auth_ok = cli_path is not None and qz_auth_configured(cli_path)
    qz_usable = cli_path is not None and auth_ok

    schema_checked = False
    schema_path: str | None = None
    if cli_path is not None:
        target = out_path.parent / "qz" / QZ_SCHEMA_FILENAME
        schema_checked = qz_capture_schema(cli_path, target)
        if schema_checked:
            schema_path = str(target)

    # Stages too heavy for local RTX 4090 execution. The official SWE-bench
    # harness requires a reachable local Docker daemon; without one, that stage
    # must be offloaded to the cluster.
    stages_requiring_offload: list[str] = []
    if not daemon_reachable:
        stages_requiring_offload.append("official_swebench_harness")
    offload_required = bool(stages_requiring_offload)

    if offload_required and not qz_usable:
        blockers.append(
            {
                "code": "qz_offload_unavailable",
                "fix_command": (
                    "install/configure the qz CLI (qz config get) or provision a "
                    + "local Docker daemon for the official harness stage"
                ),
            }
        )

    return {
        "available": qz_usable,
        "schema_checked": schema_checked,
        "dry_run_required": True,
        "schema_path": schema_path,
        "auth_configured": auth_ok,
        "cli_path": cli_path,
        "stages_requiring_offload": stages_requiring_offload,
        "offload_available": qz_usable,
    }


# --------------------------------------------------------------------------- #
# Config loading helpers.
# --------------------------------------------------------------------------- #
def load_manifest_ids(manifest_path: Path) -> tuple[str, str, list[str], list[str]]:
    config = load_yaml_config(manifest_path)
    dataset_raw = config.get("dataset_name", DEFAULT_DATASET)
    split_raw = config.get("split", DEFAULT_SPLIT)
    dataset_name = require_string(dataset_raw, "Manifest 'dataset_name' must be a string")
    split = require_string(split_raw, "Manifest 'split' must be a string")
    smoke_ids = _string_list(config.get("smoke_ids"), "smoke_ids")
    main_ids = _string_list(config.get("main_ids"), "main_ids")
    return dataset_name, split, smoke_ids, main_ids


def load_qwable_model_id(models_config_path: Path) -> str:
    config = load_yaml_config(models_config_path)
    models = config.get("models")
    if isinstance(models, dict):
        qwable = models.get("qwable")
        if isinstance(qwable, dict):
            model_id = qwable.get("id")
            if isinstance(model_id, str) and model_id.strip():
                return model_id
    return DEFAULT_QWABLE_ID


def _disk_min_free_gb(resource_config: ConfigMap) -> float:
    disk = resource_config.get("disk")
    if isinstance(disk, dict):
        value = disk.get("min_free_gb")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return DEFAULT_MIN_FREE_GB


def _disk_probe_path() -> Path:
    cwd = Path.cwd()
    return cwd if cwd.exists() else PROJECT_ROOT


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"Manifest field '{label}' must be a non-empty list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"Manifest field '{label}' must contain non-empty strings")
        result.append(item)
    return result


def _mask_tokens(text: str) -> str:
    return _TOKEN_PATTERN.sub("***REDACTED***", text)


def _short_text(value: object, limit: int = 500) -> str:
    if value is None:
        return ""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    return _mask_tokens(text.strip())[-limit:]


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight the official SWE-bench Lite harness and qz offload readiness"
    )
    _ = parser.add_argument("--manifest", required=True, help="Task manifest YAML with smoke/main IDs")
    _ = parser.add_argument("--models-config", required=True, help="Models YAML providing the Qwable id")
    _ = parser.add_argument("--resources", required=True, help="Resource YAML for worker/disk policy")
    _ = parser.add_argument("--out", default=str(DEFAULT_OUT), help="Status JSON output path")
    _ = parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable strict mode (always exit 0; strict is the default)",
    )
    parser.set_defaults(strict=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _typed_args(build_parser().parse_args(argv))
        return run_from_args(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


def run_from_args(args: PreflightArgs) -> int:
    dataset_name, split, smoke_ids, main_ids = load_manifest_ids(args.manifest)
    qwable_model_id = load_qwable_model_id(args.models_config)

    status = build_status(
        dataset_name=dataset_name,
        split=split,
        smoke_ids=smoke_ids,
        main_ids=main_ids,
        qwable_model_id=qwable_model_id,
        resources_path=args.resources,
        out_path=args.out,
        strict=args.strict,
    )
    write_json_atomic(args.out, status)

    blockers = cast(list[ConfigMap], status["blockers"])
    codes = [cast(str, b.get("code")) for b in blockers]
    print(
        f"preflight status={status['status']}; blockers={codes or 'none'}; "
        f"written to {args.out}"
    )

    if not args.strict:
        return 0
    return 1 if blockers else 0


def _typed_args(namespace: argparse.Namespace) -> PreflightArgs:
    values = cast(dict[str, object], vars(namespace))
    return PreflightArgs(
        manifest=Path(require_string(values.get("manifest"), "--manifest must be a string")),
        models_config=Path(require_string(values.get("models_config"), "--models-config must be a string")),
        resources=Path(require_string(values.get("resources"), "--resources must be a string")),
        out=Path(require_string(values.get("out"), "--out must be a string")),
        strict=bool(values.get("strict", True)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
