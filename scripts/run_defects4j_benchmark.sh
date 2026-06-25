#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-code-repair-defects4j:latest}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
FRESH_MEMORY="${FRESH_MEMORY:-1}"
DEEPSEEK_TIMEOUT="${DEEPSEEK_TIMEOUT:-300}"
DEFECTS4J_TIMEOUT="${DEFECTS4J_TIMEOUT:-3600}"
CODE_REPAIR_TEST_TIMEOUT="${CODE_REPAIR_TEST_TIMEOUT:-60}"
SKIP_DOCKER_BUILD="${SKIP_DOCKER_BUILD:-0}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running. Start Docker Desktop, then rerun this script." >&2
  exit 2
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set. Export it in the shell; do not write it into the repo." >&2
  exit 2
fi

if [[ "${SKIP_DOCKER_BUILD}" == "1" ]]; then
  if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "SKIP_DOCKER_BUILD=1 but image ${IMAGE_NAME} does not exist." >&2
    exit 2
  fi
else
  if ! docker build -f Dockerfile.defects4j -t "${IMAGE_NAME}" .; then
    if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
      echo "Docker build failed; reusing existing local image ${IMAGE_NAME}." >&2
    else
      exit 1
    fi
  fi
fi
BENCHMARK_ARGS=(
  python3 -m code_repair_agent.real_benchmark
  --config configs/defects4j_30.json
  --run-id "${RUN_ID}"
  --out-dir artifacts/runs
  --systems baseline,feedback,self_evolved
  --max-attempts 3
)

if [[ "${FRESH_MEMORY}" == "1" ]]; then
  BENCHMARK_ARGS+=(--fresh-memory)
fi
BENCHMARK_ARGS+=("$@")

docker run --rm \
  -e DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}" \
  -e DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}" \
  -e DEEPSEEK_TIMEOUT="${DEEPSEEK_TIMEOUT}" \
  -e DEFECTS4J_TIMEOUT="${DEFECTS4J_TIMEOUT}" \
  -e CODE_REPAIR_TEST_TIMEOUT="${CODE_REPAIR_TEST_TIMEOUT}" \
  -v "$PWD":/workspace \
  -w /workspace \
  "${IMAGE_NAME}" \
  "${BENCHMARK_ARGS[@]}"
