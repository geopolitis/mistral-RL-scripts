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

MODEL_NAME="${MODEL_NAME:-mistralai/Ministral-3-14B-Instruct-2512}"
DATA_DIR="${DATA_DIR:-datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mistral-grpo}"
RUN_NAME="${RUN_NAME:-ministral-grpo-single-h100}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-mistral-rl}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-ministral-grpo}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-train}"
WANDB_TAGS="${WANDB_TAGS:-grpo,ministral,single-h100}"

ensure_wandb_installed() {
  if python3 -c "import wandb" >/dev/null 2>&1; then
    return 0
  fi

  echo "[preflight] wandb not found; installing..."
  python3 -m pip install --upgrade wandb
}

ensure_wandb_login() {
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "[preflight] logging into W&B with WANDB_API_KEY..."
    wandb login --relogin "$WANDB_API_KEY"
  fi

  if python3 - <<'PY'
import wandb
raise SystemExit(0 if wandb.api.api_key else 1)
PY
  then
    return 0
  fi

  echo "[preflight] W&B is not authenticated."
  echo "[preflight] Run 'wandb login' once, or set WANDB_API_KEY before launching."
  exit 1
}

ensure_wandb_installed
ensure_wandb_login

python3 scripts/train_grpo_mistral.py \
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
  --use-4bit \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-generations 4 \
  --max-prompt-length 512 \
  --max-completion-length 192 \
  --learning-rate 5e-6 \
  --num-train-epochs 1 \
  --logging-steps 10 \
  --save-steps 100 \
  --eval-steps 100
