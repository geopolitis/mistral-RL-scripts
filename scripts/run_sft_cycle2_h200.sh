#!/usr/bin/env bash
set -euo pipefail

export DATA_DIR="${DATA_DIR:-datasets_v2/unique_prompts_train_v2.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/mistral-sft-cycle2}"
export RUN_NAME="${RUN_NAME:-ministral-sft-cycle2-h200}"
export WANDB_GROUP="${WANDB_GROUP:-ministral-sft-cycle2}"

exec ./scripts/run_sft_single_h200.sh
