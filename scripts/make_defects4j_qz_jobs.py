#!/usr/bin/env python3
"""Generate qz CreateJob specs for the Defects4J pipeline.

Emits two job specs and matching dry-run previews under ``outputs/qz/``:

* ``defects4j_infer_job.json`` / ``defects4j_infer_dry_run.yaml`` --- a 4xH200 GPU
  inference job that runs the Java-capable repair agent over a pinned Defects4J id
  set and writes ``outputs/runs/d4j_baseline/predictions.jsonl``.
* ``defects4j_eval_job.json`` / ``defects4j_eval_dry_run.yaml`` --- a CPU evaluation
  job that scores those predictions through ``repair_agent.env.harness`` using the
  non-Docker Defects4J backend.

The specs follow the existing qz pattern in ``outputs/qz/`` (see
``scripts/run_gated_experiments.py`` and ``repair_agent/agent/models.py``): every
truly-unresolved id/image/spec is left as ``RESOLVE_BEFORE_SUBMISSION`` and the
required CreateJob fields are taken from ``outputs/qz/train.CreateJob.schema.yaml``.

This script NEVER submits a job. It only writes the spec JSON and renders a local
``--dry-run`` preview (which sends nothing to the server).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repair_agent.config import ConfigError, ConfigMap, require_string  # noqa: E402
from repair_agent.logging import write_json_atomic  # noqa: E402

DEFAULT_QZ_SCHEMA_PATH = Path("outputs/qz/train.CreateJob.schema.yaml")
DEFAULT_QZ_OUT_DIR = Path("outputs/qz")
QZ_RESOLVE_PLACEHOLDER = "RESOLVE_BEFORE_SUBMISSION"
QZ_CREATE_JOB_ENDPOINT = "https://qz.sii.edu.cn/api/v2/train?Action=CreateJob"

# Provided qz project id (CONTEXT). Only the project id is concrete; workspace,
# compute group, spec, and image must be resolved by an operator before submission.
QZ_PROJECT_ID = "project-632c8db8-4530-413a-ada5-df91774a7e09"

# Staged paths on the qz node. Defects4J is initialized at /tmp/opencode/defects4j;
# model weights and the HuggingFace cache are staged under the same scratch root.
DEFECTS4J_HOME = "/tmp/opencode/defects4j"
HF_HOME = "/tmp/opencode/hf_cache"
JAVA_HOME = "/usr/lib/jvm/java-11-openjdk-amd64"

# GPU inference command: run the Java-capable agent over the pinned Defects4J id set
# using the 4xH200 resource profile. Mirrors local usage.
DEFECTS4J_INFER_COMMAND = (
    "python -m repair_agent.run --defects4j "
    "--defects4j-manifest configs/defects4j_manifest.yaml "
    "--config configs/defects4j.yaml --run-id d4j_baseline "
    "--resources configs/resources.h200.yaml"
)

# CPU evaluation command: score the inference predictions with the non-Docker
# Defects4J harness backend. Mirrors local usage.
DEFECTS4J_EVAL_COMMAND = (
    "python -m repair_agent.env.harness "
    "--predictions outputs/runs/d4j_baseline/predictions.jsonl "
    "--run-id d4j_baseline_eval --max-workers 16 --strict-official "
    "--defects4j-home /tmp/opencode/defects4j "
    "--status-out outputs/harness_status_d4j_baseline.json"
)

# JWT-like bearer tokens (qz tokens start with ``eyJ``) and ``key: value`` secret
# lines must never be persisted from qz output. Used to scrub before storage.
_SECRET_PATTERN = re.compile(
    r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"
    + r"|(?:token|secret|password|api[_-]?key|authorization|bearer)\s*[:=]\s*\S+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CliArgs:
    schema_path: Path
    out_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Defects4J qz CreateJob specs (4xH200 inference + CPU eval)",
    )
    _ = parser.add_argument(
        "--schema-path",
        default=str(DEFAULT_QZ_SCHEMA_PATH),
        help="qz CreateJob schema JSON used to discover required fields",
    )
    _ = parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_QZ_OUT_DIR),
        help="Directory for the generated job/dry-run specs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _typed_args(build_parser().parse_args(argv))
        required_fields = _qz_required_fields(args.schema_path)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    infer_job = args.out_dir / "defects4j_infer_job.json"
    infer_dry = args.out_dir / "defects4j_infer_dry_run.yaml"
    infer_source = _emit_job(
        spec=_defects4j_infer_qz_spec(required_fields),
        job_path=infer_job,
        dry_run_path=infer_dry,
    )
    print(f"infer: wrote {infer_job} and {infer_dry} (dry_run_source={infer_source})")

    eval_job = args.out_dir / "defects4j_eval_job.json"
    eval_dry = args.out_dir / "defects4j_eval_dry_run.yaml"
    eval_source = _emit_job(
        spec=_defects4j_eval_qz_spec(required_fields),
        job_path=eval_job,
        dry_run_path=eval_dry,
    )
    print(f"eval: wrote {eval_job} and {eval_dry} (dry_run_source={eval_source})")
    return 0


def _emit_job(*, spec: ConfigMap, job_path: Path, dry_run_path: Path) -> str:
    write_json_atomic(job_path, spec)
    return _render_dry_run(job_path, dry_run_path)


def _qz_required_fields(schema_path: Path) -> list[str]:
    if not schema_path.is_file():
        raise ConfigError(f"qz schema not found: {schema_path}")
    try:
        loaded = cast(object, json.loads(schema_path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"qz schema is not valid JSON: {schema_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError("qz schema must decode to a JSON object")
    schema = cast(dict[str, object], loaded)
    parameters = schema.get("parameters")
    required: list[str] = []
    if isinstance(parameters, list):
        for parameter in cast(list[object], parameters):
            if not isinstance(parameter, dict):
                continue
            parameter_map = cast(dict[str, object], parameter)
            if parameter_map.get("required") is True:
                field = parameter_map.get("jsonField")
                if isinstance(field, str) and field:
                    required.append(field)
    return required


def _defects4j_infer_qz_spec(required_fields: list[str]) -> ConfigMap:
    # 4xH200 inference. The H200 node is memory-ample but weights must be staged,
    # so the device map spans all four visible GPUs (ids 0-3) via device_map_auto.
    envs = [
        {"name": "CUDA_VISIBLE_DEVICES", "value": "0,1,2,3"},
        {"name": "DEFECTS4J_HOME", "value": DEFECTS4J_HOME},
        {"name": "HF_HOME", "value": HF_HOME},
        {"name": "JAVA_HOME", "value": JAVA_HOME},
    ]
    return _qz_spec(
        name="defects4j-infer-4xh200",
        command=DEFECTS4J_INFER_COMMAND,
        envs=envs,
        required_fields=required_fields,
    )


def _defects4j_eval_qz_spec(required_fields: list[str]) -> ConfigMap:
    # Evaluation is CPU-bound (Java compile + test execution under Defects4J), so the
    # GPUs are hidden (CUDA_VISIBLE_DEVICES="") to force a CPU-only run on its spec.
    envs = [
        {"name": "CUDA_VISIBLE_DEVICES", "value": ""},
        {"name": "DEFECTS4J_HOME", "value": DEFECTS4J_HOME},
        {"name": "HF_HOME", "value": HF_HOME},
        {"name": "JAVA_HOME", "value": JAVA_HOME},
    ]
    return _qz_spec(
        name="defects4j-eval-cpu",
        command=DEFECTS4J_EVAL_COMMAND,
        envs=envs,
        required_fields=required_fields,
    )


def _qz_spec(
    *,
    name: str,
    command: str,
    envs: list[dict[str, str]],
    required_fields: list[str],
) -> ConfigMap:
    spec: ConfigMap = {
        "command": command,
        "envs": list(envs),
        "framework": "PyTorch",
        "framework_config": [
            {
                "image": QZ_RESOLVE_PLACEHOLDER,
                "image_type": QZ_RESOLVE_PLACEHOLDER,
                "instance_count": 1,
                "spec_id": QZ_RESOLVE_PLACEHOLDER,
            }
        ],
        "logic_compute_group_id": QZ_RESOLVE_PLACEHOLDER,
        "name": name,
        "project_id": QZ_PROJECT_ID,
        "workspace_id": QZ_RESOLVE_PLACEHOLDER,
    }
    for field in required_fields:
        if field not in spec:
            spec[field] = QZ_RESOLVE_PLACEHOLDER
    return spec


def _render_dry_run(job_path: Path, dry_run_path: Path) -> str:
    job_text = job_path.read_text(encoding="utf-8")
    cli_output = _qz_dry_run_via_cli(job_text)
    if cli_output is not None and cli_output.lstrip().startswith("DRY RUN: POST"):
        content, source = cli_output, "qz_cli"
    else:
        content, source = _emulated_dry_run(job_text), "emulated"
    scrubbed = _scrub_secrets(content)
    if not scrubbed.endswith("\n"):
        scrubbed += "\n"
    dry_run_path.parent.mkdir(parents=True, exist_ok=True)
    _ = dry_run_path.write_text(scrubbed, encoding="utf-8")
    return source


def _qz_dry_run_via_cli(job_text: str) -> str | None:
    qz_path = shutil.which("qz")
    if qz_path is None:
        return None
    try:
        completed = subprocess.run(
            [qz_path, "train", "CreateJob", "--data", job_text, "--dry-run", "-o", "yaml"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = completed.stdout if completed.stdout.strip() else completed.stderr
    return output if output.strip() else None


def _emulated_dry_run(job_text: str) -> str:
    return f"DRY RUN: POST {QZ_CREATE_JOB_ENDPOINT}\nBody: {job_text.rstrip()}\n"


def _scrub_secrets(text: str) -> str:
    return _SECRET_PATTERN.sub("***REDACTED***", text)


def _typed_args(namespace: argparse.Namespace) -> CliArgs:
    schema_path = require_string(
        cast(object, getattr(namespace, "schema_path")), "--schema-path must be a string"
    )
    out_dir = require_string(
        cast(object, getattr(namespace, "out_dir")), "--out-dir must be a string"
    )
    return CliArgs(schema_path=Path(schema_path), out_dir=Path(out_dir))


if __name__ == "__main__":
    raise SystemExit(main())
