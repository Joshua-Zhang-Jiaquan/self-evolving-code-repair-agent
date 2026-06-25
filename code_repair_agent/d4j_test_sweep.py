"""Parallel Defects4J compile/test sweep.

This runner is intentionally model-free: it checks that the Dockerized
Defects4J environment can checkout, compile, export metadata, and execute
trigger/relevant/all test scopes for the selected benchmark cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from .d4j_benchmark import BenchmarkCase, load_cases
from .defects4j import CommandResult, Defects4JCase, Defects4JClient, _clean_export_output


METADATA_PROPS = (
    "dir.src.classes",
    "dir.src.tests",
    "tests.trigger",
    "tests.relevant",
    "classes.modified",
)


@dataclass
class SweepMetrics:
    case_id: str
    project: str
    bug_id: int
    status: str
    checkout_ok: bool
    compile_ok: bool
    metadata_ok: bool
    trigger_commands: int
    trigger_passed: bool
    relevant_ran: bool
    relevant_passed: bool
    all_ran: bool
    all_passed: bool
    infrastructure_failure: bool
    wall_time_seconds: float

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def run_sweep(
    *,
    cases: List[BenchmarkCase],
    run_dir: Path,
    jobs: int,
    scopes: List[str],
    resume: bool,
    timeout: int,
) -> Dict[str, object]:
    run_dir = run_dir.resolve()
    trace_dir = run_dir / "traces"
    work_dir = run_dir / "work"
    trace_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics: List[SweepMetrics] = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [
            pool.submit(_run_case, case, run_dir, work_dir, trace_dir, scopes, resume, timeout)
            for case in cases
        ]
        for future in as_completed(futures):
            item, trace_path = future.result()
            metrics.append(item)
            print(json.dumps({"case_id": item.case_id, "status": item.status, "trace": str(trace_path)}, ensure_ascii=False), flush=True)
    metrics.sort(key=lambda item: (item.project, item.bug_id))
    _write_metrics(run_dir / "metrics.csv", metrics)
    summary = _summarize(run_dir, metrics, jobs, scopes)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_failure_report(run_dir / "failure_analysis.md", metrics)
    return summary


def _run_case(
    case: BenchmarkCase,
    run_dir: Path,
    work_dir: Path,
    trace_dir: Path,
    scopes: List[str],
    resume: bool,
    timeout: int,
) -> Tuple[SweepMetrics, Path]:
    trace_path = trace_dir / f"{case.case_id}.json"
    if resume and trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        return SweepMetrics(**trace["metrics"]), trace_path

    started = time.perf_counter()
    case_workdir = work_dir / case.case_id
    client = Defects4JClient(timeout=timeout)
    commands: List[Dict[str, object]] = []
    trace: Dict[str, object] = {"case": case.as_dict(), "commands": commands, "scopes": {}}
    checkout_ok = False
    compile_ok = False
    metadata_ok = False
    trigger_commands = 0
    trigger_passed = False
    relevant_ran = False
    relevant_passed = False
    all_ran = False
    all_passed = False
    infrastructure_failure = False
    status = "failed"

    try:
        if case_workdir.exists():
            shutil.rmtree(case_workdir)
        checkout = client.checkout(Defects4JCase(case.project, case.bug_id, case_workdir))
        trace["checkout_output_tail"] = checkout[-4000:]
        checkout_ok = True

        compile_result = client.run([client.binary, "compile"], cwd=case_workdir, check=False)
        commands.append(compile_result.as_dict())
        compile_ok = compile_result.ok
        if not compile_ok:
            infrastructure_failure = True
            status = "compile_failed"
            return _finalize(case, trace, trace_path, started, status, checkout_ok, compile_ok, metadata_ok, trigger_commands, trigger_passed, relevant_ran, relevant_passed, all_ran, all_passed, infrastructure_failure)

        metadata, metadata_results = _export_metadata(client, case_workdir)
        commands.extend(result.as_dict() for result in metadata_results)
        trace["metadata"] = metadata
        metadata_ok = all(result.ok for result in metadata_results)
        if not metadata_ok:
            infrastructure_failure = True
            status = "metadata_failed"
            return _finalize(case, trace, trace_path, started, status, checkout_ok, compile_ok, metadata_ok, trigger_commands, trigger_passed, relevant_ran, relevant_passed, all_ran, all_passed, infrastructure_failure)

        if "trigger" in scopes:
            trigger_tests = _split_metadata(metadata.get("tests.trigger", ""))
            trigger_results: List[CommandResult] = []
            for test_name in trigger_tests:
                result = client.run([client.binary, "test", "-t", test_name], cwd=case_workdir, check=False)
                trigger_results.append(_with_failing_tests(case_workdir, result))
            trigger_commands = len(trigger_results)
            trigger_passed = bool(trigger_results) and all(_d4j_test_passed(result) for result in trigger_results)
            commands.extend(result.as_dict() for result in trigger_results)
            trace["scopes"]["trigger"] = [result.as_dict() for result in trigger_results]

        if "relevant" in scopes:
            relevant_ran = True
            result = _with_failing_tests(
                case_workdir,
                client.run([client.binary, "test", "-r"], cwd=case_workdir, check=False),
            )
            relevant_passed = _d4j_test_passed(result)
            commands.append(result.as_dict())
            trace["scopes"]["relevant"] = result.as_dict()

        if "all" in scopes:
            all_ran = True
            result = _with_failing_tests(
                case_workdir,
                client.run([client.binary, "test"], cwd=case_workdir, check=False),
            )
            all_passed = _d4j_test_passed(result)
            commands.append(result.as_dict())
            trace["scopes"]["all"] = result.as_dict()

        status = "completed"
    except Exception as exc:
        infrastructure_failure = True
        status = "infrastructure_error"
        trace["error"] = str(exc)

    return _finalize(case, trace, trace_path, started, status, checkout_ok, compile_ok, metadata_ok, trigger_commands, trigger_passed, relevant_ran, relevant_passed, all_ran, all_passed, infrastructure_failure)


def _finalize(
    case: BenchmarkCase,
    trace: Dict[str, object],
    trace_path: Path,
    started: float,
    status: str,
    checkout_ok: bool,
    compile_ok: bool,
    metadata_ok: bool,
    trigger_commands: int,
    trigger_passed: bool,
    relevant_ran: bool,
    relevant_passed: bool,
    all_ran: bool,
    all_passed: bool,
    infrastructure_failure: bool,
) -> Tuple[SweepMetrics, Path]:
    metrics = SweepMetrics(
        case_id=case.case_id,
        project=case.project,
        bug_id=case.bug_id,
        status=status,
        checkout_ok=checkout_ok,
        compile_ok=compile_ok,
        metadata_ok=metadata_ok,
        trigger_commands=trigger_commands,
        trigger_passed=trigger_passed,
        relevant_ran=relevant_ran,
        relevant_passed=relevant_passed,
        all_ran=all_ran,
        all_passed=all_passed,
        infrastructure_failure=infrastructure_failure,
        wall_time_seconds=round(time.perf_counter() - started, 4),
    )
    trace["metrics"] = metrics.as_dict()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics, trace_path


def _export_metadata(client: Defects4JClient, workdir: Path) -> Tuple[Dict[str, str], List[CommandResult]]:
    metadata: Dict[str, str] = {}
    results: List[CommandResult] = []
    for prop in METADATA_PROPS:
        result = client.run([client.binary, "export", "-p", prop], cwd=workdir, check=False)
        results.append(result)
        metadata[prop] = _clean_export_output(result.output) if result.ok else ""
    return metadata, results


def _with_failing_tests(workdir: Path, result: CommandResult) -> CommandResult:
    failing_tests = workdir / "failing_tests"
    if not failing_tests.exists() or _d4j_test_passed(result):
        return result
    details = failing_tests.read_text(encoding="utf-8", errors="replace")[-6000:]
    return CommandResult(
        command=result.command,
        cwd=result.cwd,
        returncode=result.returncode,
        output=f"{result.output}\n\n[failing_tests]\n{details}",
        elapsed_seconds=result.elapsed_seconds,
    )


def _d4j_test_passed(result: CommandResult) -> bool:
    import re

    match = re.search(r"Failing tests:\s*(\d+)", result.output)
    if match:
        return int(match.group(1)) == 0
    return result.ok


def _split_metadata(text: str) -> List[str]:
    import re

    return [item.strip() for item in re.split(r"[;\n\r]+", text or "") if item.strip()]


def _write_metrics(path: Path, metrics: List[SweepMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SweepMetrics.__dataclass_fields__.keys()))
        writer.writeheader()
        for item in metrics:
            writer.writerow(item.as_dict())


def _summarize(run_dir: Path, metrics: List[SweepMetrics], jobs: int, scopes: List[str]) -> Dict[str, object]:
    total = len(metrics) or 1
    return {
        "run_dir": str(run_dir),
        "jobs": jobs,
        "scopes": scopes,
        "cases": len(metrics),
        "completed": sum(item.status == "completed" for item in metrics),
        "checkout_success_rate": round(sum(item.checkout_ok for item in metrics) / total, 4),
        "compile_success_rate": round(sum(item.compile_ok for item in metrics) / total, 4),
        "metadata_success_rate": round(sum(item.metadata_ok for item in metrics) / total, 4),
        "trigger_pass_rate": round(sum(item.trigger_passed for item in metrics) / total, 4),
        "relevant_pass_rate": round(sum(item.relevant_passed for item in metrics if item.relevant_ran) / max(1, sum(item.relevant_ran for item in metrics)), 4),
        "all_pass_rate": round(sum(item.all_passed for item in metrics if item.all_ran) / max(1, sum(item.all_ran for item in metrics)), 4),
        "infrastructure_failures": sum(item.infrastructure_failure for item in metrics),
        "wall_time_seconds": round(sum(item.wall_time_seconds for item in metrics), 4),
    }


def _write_failure_report(path: Path, metrics: List[SweepMetrics]) -> None:
    lines = ["# D4J Test Sweep Failure Analysis", ""]
    for item in metrics:
        if item.infrastructure_failure:
            lines.append(f"- `{item.case_id}` infrastructure failure: status={item.status}")
    if len(lines) == 2:
        lines.append("No infrastructure failures recorded. Test failures on buggy versions are expected before repair.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/defects4j_30.json"))
    parser.add_argument("--run-id", default=datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/d4j_test_sweeps"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--scopes", default="trigger,relevant,all")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    client = Defects4JClient(timeout=args.timeout)
    if not client.available():
        run_dir = args.out_dir / args.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "run_dir": str(run_dir),
            "status": "preflight_failed",
            "errors": ["defects4j CLI not found on PATH"],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
        raise SystemExit(2)

    scopes = [scope.strip() for scope in args.scopes.split(",") if scope.strip()]
    unknown = [scope for scope in scopes if scope not in {"trigger", "relevant", "all"}]
    if unknown:
        raise ValueError(f"unknown scopes: {unknown}")
    summary = run_sweep(
        cases=load_cases(args.config),
        run_dir=args.out_dir / args.run_id,
        jobs=args.jobs,
        scopes=scopes,
        resume=not args.no_resume,
        timeout=args.timeout,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
