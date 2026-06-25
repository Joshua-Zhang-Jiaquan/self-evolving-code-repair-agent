#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

OUT_DIR="${OUT_DIR:-submission_dist}"
BUNDLE="${BUNDLE:-${OUT_DIR}/self-evolving-code-repair-agent-submission.zip}"
export OUT_DIR BUNDLE

mkdir -p "${OUT_DIR}"
rm -f "${BUNDLE}"

if rg -n "sk-[A-Za-z0-9]" README.md SUBMISSION.md reports docs artifacts/README.md scripts configs code_repair_agent tests >/tmp/submission_secret_scan.txt 2>&1; then
  echo "Refusing to create bundle: possible API key found." >&2
  cat /tmp/submission_secret_scan.txt >&2
  exit 2
fi

python3 - <<'PY'
import os
import zipfile
from pathlib import Path

root = Path.cwd()
bundle = Path(os.environ.get("BUNDLE", os.environ.get("OUT_DIR", "submission_dist") + "/self-evolving-code-repair-agent-submission.zip"))
bundle.parent.mkdir(parents=True, exist_ok=True)

include_roots = [
    "README.md",
    "SUBMISSION.md",
    "Dockerfile",
    "Dockerfile.defects4j",
    "pyproject.toml",
    "code_repair_agent",
    "configs",
    "scripts",
    "tests",
    "docs",
    "reports",
    "artifacts/README.md",
    "artifacts/proof_experiments/two_dimensional_memory-v4",
    "artifacts/runs/d4j30-self-evolved-real-v16/summary.json",
    "artifacts/runs/d4j30-self-evolved-real-v16/metrics.csv",
    "artifacts/runs/d4j30-self-evolved-real-v16/failure_analysis.md",
    "artifacts/runs/d4j30-self-evolved-real-v16/memory_before.json",
    "artifacts/runs/d4j30-self-evolved-real-v16/memory_after.json",
    "artifacts/runs/d4j30-self-evolved-real-v16/memory_snapshots",
    "artifacts/runs/d4j30-self-evolved-real-v16/traces",
    "artifacts/runs/d4j30-self-evolved-real-v16/patches",
]

skip_parts = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "work",
}
skip_suffixes = {
    ".pyc",
    ".pyo",
}

def iter_files(path: Path):
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for item in sorted(path.rglob("*")):
            rel_parts = item.relative_to(root).parts
            if any(part in skip_parts for part in rel_parts):
                continue
            if item.is_file() and item.suffix not in skip_suffixes:
                yield item

with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    seen = set()
    for entry in include_roots:
        path = root / entry
        if not path.exists():
            continue
        for file_path in iter_files(path):
            rel = file_path.relative_to(root)
            if rel in seen:
                continue
            seen.add(rel)
            zf.write(file_path, rel.as_posix())

print(bundle)
PY

echo "Created ${BUNDLE}"
