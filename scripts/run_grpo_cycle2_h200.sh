#!/usr/bin/env bash
set -euo pipefail

export DATA_DIR="${DATA_DIR:-datasets_v2/unique_prompts_train_v2.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/mistral-grpo-cycle2}"
export RUN_NAME="${RUN_NAME:-ministral-grpo-cycle2-h200}"
export WANDB_GROUP="${WANDB_GROUP:-ministral-grpo-cycle2}"

# Cycle-2 defaults tuned for throughput/quality balance.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-384}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-192}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export REWARD_SHORT_WORD_THRESHOLD="${REWARD_SHORT_WORD_THRESHOLD:-16}"
export REWARD_SHORT_PENALTY="${REWARD_SHORT_PENALTY:-0.12}"
export WARMUP_STEPS="${WARMUP_STEPS:-80}"

exec ./scripts/run_grpo_single_h200.sh
