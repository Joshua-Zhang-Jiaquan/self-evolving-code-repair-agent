#!/usr/bin/env python3
"""Probe local hardware and write a device inventory JSON file.

Usage:
    python scripts/probe_local_devices.py --out outputs/device_inventory.json
    python scripts/probe_local_devices.py                    # prints to stdout

Detects GPUs, CPU, RAM, and disk; recommends SWE-bench worker counts.
Uses pynvml (preferred), nvidia-smi (fallback), or marks GPUs unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _gpu_via_pynvml() -> list[dict[str, Any]]:
    """Probe GPUs via pynvml.  Returns list of GPU dicts."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name_raw = pynvml.nvmlDeviceGetName(handle)
            name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                "index": idx,
                "name": name,
                "memory_total_mb": int(mem.total) // (1024 * 1024),
                "memory_free_mb": int(mem.free) // (1024 * 1024),
                "memory_used_mb": int(mem.used) // (1024 * 1024),
                "probe_method": "pynvml",
            })
        pynvml.nvmlShutdown()
        return gpus
    except Exception as exc:
        return [{"probe_method": "pynvml", "error": str(exc)}]


def _gpu_via_nvidia_smi() -> list[dict[str, Any]]:
    """Fallback GPU detection via nvidia-smi CLI."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return [{"probe_method": "nvidia-smi", "error": result.stderr.strip()}]
        gpus = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "probe_method": "nvidia-smi",
                })
        return gpus
    except Exception as exc:
        return [{"probe_method": "nvidia-smi", "error": str(exc)}]


def _cpu_info() -> dict[str, Any]:
    """Return CPU information."""
    info: dict[str, Any] = {
        "logical_cores": os.cpu_count(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
    }
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    except Exception:
        info["physical_cores"] = None
    return info


def _memory_info() -> dict[str, Any]:
    """Return RAM information."""
    info: dict[str, Any] = {}
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["total_mb"] = int(mem.total) // (1024 * 1024)
        info["available_mb"] = int(mem.available) // (1024 * 1024)
        info["used_mb"] = int(mem.used) // (1024 * 1024)
        info["free_mb"] = int(mem.free) // (1024 * 1024)
        info["percent_used"] = mem.percent
    except Exception:
        info["error"] = "psutil unavailable"
    return info


def _disk_info() -> dict[str, Any]:
    """Return disk information for the current working directory."""
    info: dict[str, Any] = {}
    try:
        cwd = os.getcwd()
        import shutil
        usage = shutil.disk_usage(cwd)
        info["path"] = cwd
        info["total_gb"] = round(usage.total / (1024 ** 3), 2)
        info["used_gb"] = round(usage.used / (1024 ** 3), 2)
        info["free_gb"] = round(usage.free / (1024 ** 3), 2)
    except Exception:
        info["error"] = "shutil.disk_usage unavailable"
    return info


def _swebench_workers(gpu_count: int, cpu_logical: int | None, ram_available_mb: int | None) -> dict[str, Any]:
    """Recommend SWE-bench Docker max_workers based on hardware."""
    if cpu_logical is None:
        cpu_logical = os.cpu_count() or 1
    if ram_available_mb is None:
        ram_available_mb = 8192  # conservative default

    # Rule: 1 worker per 4 CPU cores, bounded by available RAM (~8 GB per worker)
    cpu_based = max(1, (cpu_logical or 1) // 4)
    ram_based = max(1, ram_available_mb // 8192)
    recommended = min(cpu_based, ram_based)
    # Cap at a reasonable number
    recommended = min(recommended, 16)

    return {
        "cpu_logical": cpu_logical,
        "ram_available_mb": ram_available_mb,
        "cpu_based_workers": cpu_based,
        "ram_based_workers": ram_based,
        "recommended_swebench_max_workers": recommended,
        "rationale": f"min(cpu_cores//4={cpu_based}, ram_gb//8={ram_based}) = {recommended}",
    }


def _detect_gpu_count(gpus: list[dict[str, Any]]) -> int:
    """Count detected GPUs, ignoring error entries."""
    count = 0
    for g in gpus:
        if "index" in g and "error" not in g:
            count += 1
    return count


def probe(args: argparse.Namespace) -> None:
    """Run the full hardware probe and write JSON output."""
    gpus = _gpu_via_pynvml()
    # If pynvml returned an error entry, try nvidia-smi fallback
    if len(gpus) == 1 and "error" in gpus[0]:
        gpus_smi = _gpu_via_nvidia_smi()
        if gpus_smi and not (len(gpus_smi) == 1 and "error" in gpus_smi[0]):
            gpus = gpus_smi
            gpus[0]["fallback_reason"] = f"pynvml failed: {gpus[0].get('error', '')}"
        else:
            gpus[0]["fallback_reason"] = "pynvml and nvidia-smi both unavailable"

    gpu_count = _detect_gpu_count(gpus)
    cpu = _cpu_info()
    memory = _memory_info()
    disk = _disk_info()
    ram_available = memory.get("available_mb", memory.get("total_mb", 8192))
    workers = _swebench_workers(gpu_count, cpu.get("logical_cores"), ram_available)

    # Build fallback info for expected vs detected GPUs
    expected_ids = args.expected_gpus or [0, 1, 2, 3]
    detected_ids = set()
    for g in gpus:
        if "index" in g and "error" not in g:
            detected_ids.add(g["index"])

    missing_gpus = set(expected_ids) - detected_ids
    fallback_reasons: list[dict[str, Any]] = []
    for mid in sorted(missing_gpus):
        fallback_reasons.append({
            "gpu_id": mid,
            "status": "missing",
            "reason": "GPU not detected by pynvml or nvidia-smi",
        })

    if gpu_count < len(expected_ids) and not found_any_gpu(gpus):
        fallback_reasons.append({
            "status": "global_fallback",
            "reason": f"No GPUs detected; running CPU-only.  pynvml error: {_first_gpu_error(gpus)}",
            "recommendation": "Fall back to CPU mode; disable GPU-dependent experiments.",
        })

    inventory: dict[str, Any] = {
        "timestamp": subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip() or "",
        "hostname": platform.node(),
        "python_version": sys.version,
        "gpus": gpus,
        "gpu_count_detected": gpu_count,
        "gpus_expected": expected_ids,
        "gpus_missing": sorted(missing_gpus),
        "cpu": cpu,
        "memory": memory,
        "disk": disk,
        "swebench_workers": workers,
        "fallback": {
            "missing_gpus": fallback_reasons,
            "all_gpus_available": len(missing_gpus) == 0,
        },
        "resource_summary": {
            "gpu_usable": gpu_count,
            "cpu_cores": cpu.get("logical_cores"),
            "ram_mb": ram_available,
            "disk_free_gb": disk.get("free_gb"),
            "swebench_max_workers": workers["recommended_swebench_max_workers"],
        },
    }

    if args.out and args.out != "-":
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Device inventory written to {out_path}")
    else:
        print(json.dumps(inventory, indent=2, ensure_ascii=False))


def _first_gpu_error(gpus: list[dict[str, Any]]) -> str:
    for g in gpus:
        if "error" in g:
            return g["error"]
    return "unknown"


def found_any_gpu(gpus: list[dict[str, Any]]) -> bool:
    for g in gpus:
        if "index" in g and "error" not in g:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe local hardware resources")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (prints to stdout if omitted)")
    parser.add_argument("--expected-gpus", type=int, nargs="*", default=None,
                        help="Expected GPU IDs (default: 0 1 2 3)")
    args = parser.parse_args()
    if args.expected_gpus is None:
        args.expected_gpus = [0, 1, 2, 3]
    probe(args)


if __name__ == "__main__":
    main()
