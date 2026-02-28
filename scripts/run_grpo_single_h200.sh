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
USE_4BIT="${USE_4BIT:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-224}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1.5e-6}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
INIT_ADAPTER="${INIT_ADAPTER:-}"
REWARD_SHORT_WORD_THRESHOLD="${REWARD_SHORT_WORD_THRESHOLD:-20}"
REWARD_SHORT_PENALTY="${REWARD_SHORT_PENALTY:-0.15}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
ATTEMPT_FLASH_ATTN_BUILD="${ATTEMPT_FLASH_ATTN_BUILD:-1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
MAX_JOBS="${MAX_JOBS:-16}"

ensure_wandb_installed() {
  if uv run --python "$PYTHON_BIN" -c "import wandb" >/dev/null 2>&1; then
    return 0
  fi

  echo "[preflight] wandb not found in $VENV_PATH; installing with uv..."
  uv pip install --python "$PYTHON_BIN" --upgrade wandb
}

ensure_runtime_deps() {
  if uv run --python "$PYTHON_BIN" - <<'PY'
import accelerate
import datasets
import peft
import torch
import transformers
import trl
import wandb
PY
  then
    return 0
  fi

  echo "[preflight] missing runtime deps in $VENV_PATH; installing requirements-rl.txt..."
  uv pip install --python "$PYTHON_BIN" -r requirements-rl.txt
}

ensure_flash_attn_health() {
  set +e
  uv run --python "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("flash_attn") is None:
    raise SystemExit(0)

try:
    import flash_attn  # noqa: F401
except Exception as exc:  # noqa: BLE001
    print(exc)
    raise SystemExit(2)

raise SystemExit(1)
PY
  rc=$?
  set -e

  if [[ "$rc" == "2" ]]; then
    echo "[preflight] flash-attn is installed but broken; uninstalling to force sdpa fallback..."
    uv pip uninstall --python "$PYTHON_BIN" -y flash-attn >/dev/null 2>&1 || true
  fi
}

attempt_flash_attn_build() {
  if [[ "$ATTEMPT_FLASH_ATTN_BUILD" != "1" ]]; then
    return 0
  fi

  # Skip if already healthy.
  if uv run --python "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys
if importlib.util.find_spec("flash_attn") is None:
    raise SystemExit(1)
import flash_attn  # noqa: F401
raise SystemExit(0)
PY
  then
    return 0
  fi

  # Cannot build from source without nvcc/toolkit.
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "[preflight] nvcc not found; skipping flash-attn build (sdpa will be used)."
    return 0
  fi

  echo "[preflight] attempting to build/install flash-attn for current torch/cuda stack..."
  uv pip install --python "$PYTHON_BIN" -U setuptools wheel packaging ninja psutil >/dev/null
  set +e
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  MAX_JOBS="$MAX_JOBS" \
  uv pip install --python "$PYTHON_BIN" --no-build-isolation --no-cache-dir flash-attn
  rc=$?
  set -e
  if [[ "$rc" != "0" ]]; then
    echo "[preflight] flash-attn build failed; continuing with sdpa."
    uv pip uninstall --python "$PYTHON_BIN" -y flash-attn >/dev/null 2>&1 || true
  fi
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

ensure_runtime_deps
ensure_wandb_installed
ensure_wandb_login
ensure_flash_attn_health
attempt_flash_attn_build
ensure_flash_attn_health

EXTRA_FLAGS=()
if [[ "$USE_4BIT" == "1" ]]; then
  EXTRA_FLAGS+=(--use-4bit)
fi
if [[ -n "$INIT_ADAPTER" ]]; then
  EXTRA_FLAGS+=(--init-adapter "$INIT_ADAPTER")
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
  --num-generations "$NUM_GENERATIONS" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --max-completion-length "$MAX_COMPLETION_LENGTH" \
  --reward-short-word-threshold "$REWARD_SHORT_WORD_THRESHOLD" \
  --reward-short-penalty "$REWARD_SHORT_PENALTY" \
  --warmup-steps "$WARMUP_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --num-train-epochs "$NUM_TRAIN_EPOCHS" \
  --logging-steps 10 \
  --save-steps 500 \
  --eval-steps 500 \
  "${EXTRA_FLAGS[@]}"
