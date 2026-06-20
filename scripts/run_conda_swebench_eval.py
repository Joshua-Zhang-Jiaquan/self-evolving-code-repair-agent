#!/usr/bin/env python3
"""Conda-based SWE-bench evaluation (no Docker).

Reuses swebench's authoritative TestSpec setup/eval scripts and grading, but runs
them in conda environments on the host instead of inside Docker containers. This
avoids the unprivileged-container namespace blocker that prevents docker build/run.

Empty model patches are short-circuited as unresolved (no env setup) exactly as the
official harness does, so only real (non-empty) patches trigger a full conda run.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONDA = "/opt/miniconda3"
TESTBED = "/testbed"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Conda-based SWE-bench evaluation (no Docker).")
    _ = p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    _ = p.add_argument("--split", default="test")
    _ = p.add_argument("--predictions", required=True)
    _ = p.add_argument("--run-id", required=True)
    _ = p.add_argument("--out", required=True)
    _ = p.add_argument("--instances", default=None, help="comma-separated subset of instance_ids")
    _ = p.add_argument("--timeout", type=int, default=2400)
    return p


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = cast(dict[str, Any], json.loads(line))
            rows[str(r["instance_id"])] = r
    return rows


def _run(script: str, log_fp: Path, timeout: int) -> int:
    log_fp.parent.mkdir(parents=True, exist_ok=True)
    with log_fp.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(["/bin/bash", "-lc", script], stdout=fh, stderr=subprocess.STDOUT,
                              text=True, timeout=timeout, check=False)
    return proc.returncode


def _apply_model_patch(patch: str, log_fp: Path) -> bool:
    pf = Path(TESTBED) / ".model_patch.diff"
    pf.write_text(patch, encoding="utf-8")
    for cmd in (f"cd {TESTBED} && git apply -v .model_patch.diff",
                f"cd {TESTBED} && patch --batch --fuzz=5 -p1 -i .model_patch.diff"):
        rc = _run(cmd, log_fp, 300)
        if rc == 0:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from datasets import load_dataset
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.grading import (get_logs_eval, get_eval_tests_report,
                                          get_resolution_status)
    from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS, ResolvedStatus

    preds = _load_predictions(Path(args.predictions))
    ds = load_dataset(args.dataset, split=args.split)
    by_id = {r["instance_id"]: r for r in ds}
    wanted = [s for s in (args.instances.split(",") if args.instances else list(preds.keys())) if s]
    logroot = ROOT / "logs" / "conda_eval" / args.run_id

    results: list[dict[str, Any]] = []
    for iid in wanted:
        pred = preds.get(iid)
        rec: dict[str, Any] = {"instance_id": iid, "resolved": False, "reason": None,
                               "patch_applied": None}
        if pred is None:
            rec["reason"] = "missing_prediction"; results.append(rec); continue
        patch = str(pred.get("model_patch", "") or "")
        if not patch.strip():
            rec["reason"] = "empty_patch"; results.append(rec); continue
        row = by_id.get(iid)
        if row is None:
            rec["reason"] = "not_in_dataset"; results.append(rec); continue

        spec = make_test_spec(row)
        d = logroot / iid
        try:
            _run(f"conda env remove -n testbed -y >/dev/null 2>&1 || true; rm -rf {TESTBED}",
                 d / "clean.log", 300)
            if _run("\n".join(spec.env_script_list) if hasattr(spec, "env_script_list")
                    else spec.setup_env_script, d / "setup_env.log", args.timeout) != 0:
                rec["reason"] = "setup_env_failed"; results.append(rec); continue
            repo = str(row["repo"]); base = str(row["base_commit"])
            install = (
                "set -euxo pipefail\n"
                "source /opt/miniconda3/bin/activate\n"
                "conda activate testbed\n"
                f"rm -rf {TESTBED}\n"
                f"git clone https://github.com/{repo} {TESTBED}\n"
                f"cd {TESTBED}\n"
                f"git fetch --tags origin {base} 2>/dev/null || true\n"
                f"git checkout --detach {base}\n"
                f"chmod -R 777 {TESTBED}\n"
                f"git config --global --add safe.directory {TESTBED}\n"
                "python -m pip install -e . 2>&1 | tail -50\n"
            )
            if _run(install, d / "install_repo.log", args.timeout) != 0:
                rec["reason"] = "install_repo_failed"; results.append(rec); continue
            applied = _apply_model_patch(patch, d / "apply_patch.log")
            rec["patch_applied"] = applied
            if not applied:
                rec["reason"] = "patch_apply_failed"; results.append(rec); continue
            eval_log = d / "eval.log"
            _run(spec.eval_script, eval_log, args.timeout)
            status_map, found = get_logs_eval(spec, str(eval_log))
            gold = {FAIL_TO_PASS: json.loads(row[FAIL_TO_PASS]) if isinstance(row[FAIL_TO_PASS], str) else row[FAIL_TO_PASS],
                    PASS_TO_PASS: json.loads(row[PASS_TO_PASS]) if isinstance(row[PASS_TO_PASS], str) else row[PASS_TO_PASS]}
            report = get_eval_tests_report(status_map, gold)
            status = get_resolution_status(report)
            rec["resolution_status"] = str(status)
            rec["tests_found"] = bool(found)
            rec["resolved"] = str(status) == str(ResolvedStatus.FULL.value)
            rec["report"] = report
            rec["reason"] = "evaluated"
        except subprocess.TimeoutExpired:
            rec["reason"] = "timeout"
        except Exception as exc:  # noqa: BLE001
            rec["reason"] = f"error:{type(exc).__name__}:{exc}"[:300]
        results.append(rec)

    total = len(results)
    resolved = sum(1 for r in results if r["resolved"])
    nonempty = sum(1 for r in results if r["reason"] != "empty_patch")
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "conda_no_docker",
        "docker_used": False,
        "official_harness_executed": True,
        "execution_backend": "local_conda",
        "dataset_name": args.dataset, "split": args.split, "run_id": args.run_id,
        "conda_prefix": CONDA, "testbed": TESTBED,
        "total": total, "resolved": resolved,
        "resolved_rate": round(resolved / total, 6) if total else 0.0,
        "nonempty_patches": nonempty,
        "results": results,
    }
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    _ = outp.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"conda eval: resolved {resolved}/{total} (nonempty {nonempty}); written {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
