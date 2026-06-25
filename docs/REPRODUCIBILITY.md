# Reproducibility Guide

This document gives the commands needed to verify the code, rerun the benchmark, and inspect the submitted artifacts.

## 1. Local Python Tests

```bash
cd /Users/rccn/Documents/rl
python3 -m pip install -e .
PYTHONPYCACHEPREFIX=/tmp/rl_pycache python3 -m pytest -q
```

Expected current result:

```text
94 passed
```

These tests cover:

- toy repair environment and reward accounting;
- evaluation summary computation;
- Defects4J config parsing;
- safe patch application and unsafe edit rejection;
- DeepSeek response parsing and retry handling;
- self-improvement memory updates;
- candidate dedup and regression-aware feedback helpers.

## 2. Docker and Defects4J Preflight

```bash
docker info
docker build -f Dockerfile.defects4j -t code-repair-defects4j:latest .
docker run --rm code-repair-defects4j:latest defects4j info -p Lang
```

The Docker image installs Java 11, git, svn, perl, cpanm, Python 3, and Defects4J from the official GitHub repository.

## 3. Full 30-Case Benchmark

The benchmark requires DeepSeek. Set the API key only in the shell. Do not write it into files.

```bash
export DEEPSEEK_API_KEY="<set-in-shell-only>"
export DEEPSEEK_MODEL=deepseek-v4-pro
export DEEPSEEK_TIMEOUT=300
export DEFECTS4J_TIMEOUT=3600
RUN_ID=d4j30-self-evolved-repro bash scripts/run_defects4j_benchmark.sh
```

If the Docker image already exists and Docker Hub is slow or unavailable, skip the build step:

```bash
SKIP_DOCKER_BUILD=1 RUN_ID=d4j30-self-evolved-repro bash scripts/run_defects4j_benchmark.sh
```

The default case list is `configs/defects4j_30.json`:

```text
Chart 1-10
Lang 1, 3-11
Math 1-10
```

Outputs:

```text
artifacts/runs/<run_id>/summary.json
artifacts/runs/<run_id>/metrics.csv
artifacts/runs/<run_id>/memory_before.json
artifacts/runs/<run_id>/memory_after.json
artifacts/runs/<run_id>/failure_analysis.md
artifacts/runs/<run_id>/traces/*.json
artifacts/runs/<run_id>/patches/*.diff
```

## 4. Environment-Only Test Sweep

Use this when checking the Defects4J harness without spending model calls:

```bash
JOBS=4 DEFECTS4J_TIMEOUT=3600 bash scripts/run_defects4j_test_sweep.sh
```

This performs checkout, compile, metadata export, trigger tests, relevant tests, and all tests. It does not patch code or call DeepSeek.

## 5. Four-Way Memory Proof Experiments

Fast local proof, no API key required:

```bash
bash scripts/run_memory_ablation_proof.sh
```

Outputs:

```text
artifacts/proof_experiments/two_dimensional_memory/summary.json
artifacts/proof_experiments/two_dimensional_memory/metrics.csv
artifacts/proof_experiments/two_dimensional_memory/analysis.md
```

Real D4J proof, requiring DeepSeek:

```bash
export DEEPSEEK_API_KEY="<set-in-shell-only>"
export DEEPSEEK_MODEL=deepseek-v4-pro
SKIP_DOCKER_BUILD=1 bash scripts/run_d4j_memory_ablation_experiments.sh
```

The script runs four variants:

```text
none
check_only
repair_only
full
```

## 6. Verify the Best Submitted Run

```bash
python3 - <<'PY'
import csv, json, pathlib
run = pathlib.Path("artifacts/runs/d4j30-self-evolved-real-v16")
summary = json.loads((run / "summary.json").read_text())
systems = (summary.get("summary") or summary)["systems"]
print(json.dumps(systems["self_evolved"], indent=2))
rows = list(csv.DictReader((run / "metrics.csv").open()))
print("cases", len(rows))
print("failed", [row["case_id"] for row in rows if row["status"] != "solved"])
PY
```

Expected:

```text
cases = 30
failed = ['Lang-3', 'Lang-10', 'Math-1', 'Math-2', 'Math-3', 'Math-4', 'Math-6', 'Math-7']
```

## 7. Secret Hygiene Check

Before submission:

```bash
rg -n "sk-[A-Za-z0-9]|DEEPSEEK_API_KEY=.*[A-Za-z0-9]" \
  README.md SUBMISSION.md reports docs artifacts/README.md scripts configs code_repair_agent tests
```

The command should not find a real API key. It may find safe placeholders such as `<set-in-shell-only>`.
