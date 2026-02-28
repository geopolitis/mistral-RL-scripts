#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

MODEL_NAME="${MODEL_NAME:-mistralai/Ministral-3-3B-Instruct-2512-BF16}"
DATA_DIR="${DATA_DIR:-datasets/unique_prompts_balanced.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mistral-grpo}"
RUN_NAME="${RUN_NAME:-ministral-grpo-single-h200}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-mistral-rl}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-ministral-grpo}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-train}"
WANDB_TAGS="${WANDB_TAGS:-grpo,ministral,single-h200}"
USE_4BIT="${USE_4BIT:-0}"

ensure_wandb_installed() {
  if uv run --python "$PYTHON_BIN" -c "import wandb" >/dev/null 2>&1; then
    return 0
  fi

  echo "[preflight] wandb not found in $VENV_PATH; installing with uv..."
  uv pip install --python "$PYTHON_BIN" --upgrade wandb
}

ensure_wandb_login() {
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "[preflight] logging into W&B with WANDB_API_KEY..."
    uv run --python "$PYTHON_BIN" -m wandb login --relogin "$WANDB_API_KEY"
  fi

  if uv run --python "$PYTHON_BIN" - <<'PY'
import wandb
raise SystemExit(0 if wandb.api.api_key else 1)
PY
  then
    return 0
  fi

  echo "[preflight] W&B is not authenticated."
  echo "[preflight] Run 'uv run -m wandb login' once, or set WANDB_API_KEY before launching."
  exit 1
}

ensure_wandb_installed
ensure_wandb_login

EXTRA_FLAGS=()
if [[ "$USE_4BIT" == "1" ]]; then
  EXTRA_FLAGS+=(--use-4bit)
fi

uv run --python "$PYTHON_BIN" scripts/train_grpo_mistral.py \
  --model-name "$MODEL_NAME" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --run-name "$RUN_NAME" \
  --report-to wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-group "$WANDB_GROUP" \
  --wandb-job-type "$WANDB_JOB_TYPE" \
  --wandb-tags "$WANDB_TAGS" \
  --bf16 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-generations 2 \
  --max-prompt-length 384 \
  --max-completion-length 96 \
  --learning-rate 2e-6 \
  --num-train-epochs 1 \
  --logging-steps 10 \
  --save-steps 500 \
  --eval-steps 500 \
  "${EXTRA_FLAGS[@]}"
