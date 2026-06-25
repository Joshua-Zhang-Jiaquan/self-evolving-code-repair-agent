#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-code-repair-defects4j:latest}"
CONFIG="${CONFIG:-configs/defects4j_failed8_v16.json}"
RUN_PREFIX="${RUN_PREFIX:-d4j-memory-ablation-$(date -u +%Y%m%dT%H%M%SZ)}"
SYSTEMS="${SYSTEMS:-self_evolved}"
SELF_EVOLVED_ATTEMPTS="${SELF_EVOLVED_ATTEMPTS:-5}"
MAX_ATTEMPT_CAP="${MAX_ATTEMPT_CAP:-8}"
MAX_NON_PATCH_ROUNDS="${MAX_NON_PATCH_ROUNDS:-8}"
SKIP_DOCKER_BUILD="${SKIP_DOCKER_BUILD:-1}"
MEMORY_ROOT="${MEMORY_ROOT:-artifacts/proof_experiments/d4j_memory_ablation}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set. Export it in the shell; do not write it into the repo." >&2
  exit 2
fi

mkdir -p "${MEMORY_ROOT}"

for mode in none check_only repair_only full; do
  run_id="${RUN_PREFIX}-${mode}"
  memory_path="${MEMORY_ROOT}/${run_id}-memory.json"
  echo "Running ${mode}: ${run_id}" >&2
  SKIP_DOCKER_BUILD="${SKIP_DOCKER_BUILD}" \
  IMAGE_NAME="${IMAGE_NAME}" \
  RUN_ID="${run_id}" \
  FRESH_MEMORY=1 \
  bash scripts/run_defects4j_benchmark.sh \
    --memory-mode "${mode}" \
    --memory-path "${memory_path}" \
    --systems "${SYSTEMS}" \
    --self-evolved-attempts "${SELF_EVOLVED_ATTEMPTS}" \
    --max-attempt-cap "${MAX_ATTEMPT_CAP}" \
    --max-non-patch-rounds "${MAX_NON_PATCH_ROUNDS}" \
    --config "${CONFIG}"
done
