#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[preflight] uv is required but not found."
  echo "[preflight] Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

VENV_PATH="${VENV_PATH:-.venv}"
if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  echo "[preflight] creating virtualenv at $VENV_PATH with uv..."
  uv venv "$VENV_PATH"
fi
PYTHON_BIN="$VENV_PATH/bin/python"

BASE_URL="${BASE_URL:-}"
if [[ -z "$BASE_URL" ]]; then
  echo "[preflight] BASE_URL is required for remote vLLM validation."
  echo "Example: BASE_URL=https://<server> ./scripts/run_validate_vllm_remote_cycle2.sh"
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-mistralai/Ministral-3-3B-Instruct-2512-BF16}"
DATA_DIR="${DATA_DIR:-datasets_v2/unique_prompts_eval_hard.json}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/mistral-grpo-cycle2/validation-vllm-remote-hard.json}"

uv run --python "$PYTHON_BIN" scripts/validate_vllm_remote.py \
  --base-url "$BASE_URL" \
  --model "$MODEL_NAME" \
  --data-dir "$DATA_DIR" \
  --max-samples "$MAX_SAMPLES" \
  --save-predictions "$OUTPUT_PATH"
