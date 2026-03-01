# mistral-RL-scripts

Single-GPU RL fine-tuning (GRPO) for Mistral using the deduplicated dataset `datasets/unique_prompts_balanced.json` by default (or any JSON dataset path).

Dataset summary and quality notes: [DATASET_DETAILS.md](DATASET_DETAILS.md)

<img width="6688" height="3217" alt="17723750187237317213760559468051" src="https://github.com/user-attachments/assets/4a92dd79-417f-4d0b-ba17-c6e918fae6d1" />


## 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-rl.txt
```

Optional (faster attention if it builds in your environment):

```bash
pip install --no-build-isolation flash-attn
```

## 2) Run on one H200

```bash
bash scripts/run_grpo_single_h200.sh
```

The launcher auto-loads `.env` if present (or `ENV_FILE=path/to/file`).
Priority order: explicit shell env > `.env` values > script defaults.

Example `.env`:

```bash
MODEL_NAME=mistralai/Ministral-3-3B-Instruct-2512
DATA_DIR=datasets/unique_prompts_balanced.json
OUTPUT_DIR=outputs/mistral-grpo-exp1
RUN_NAME=ministral-grpo-exp1
SEED=42
WANDB_PROJECT=mistral-rl
WANDB_ENTITY=geo-politis-n-a
WANDB_GROUP=ministral-grpo
WANDB_JOB_TYPE=train
WANDB_TAGS=grpo,ministral,single-h200,exp1
WANDB_API_KEY=your_api_key
USE_4BIT=0
```

Default base model: `mistralai/Ministral-3-3B-Instruct-2512`
Default W&B project: `mistral-rl`
Launcher preflight: installs `wandb` if missing and validates W&B authentication before training starts.
Tracking conventions: fixed seed, explicit step logging, run naming, tags/group/job_type.

If you want non-interactive auth:

```bash
export WANDB_API_KEY=your_api_key
bash scripts/run_grpo_single_h200.sh
```

Or login once manually:

```bash
python3 -m wandb login
```

## 3) Override defaults

```bash
MODEL_NAME=mistralai/Ministral-3-3B-Instruct-2512 \
DATA_DIR=datasets/unique_prompts_balanced.json \
OUTPUT_DIR=outputs/mistral-grpo-exp1 \
RUN_NAME=ministral-grpo-exp1 \
SEED=42 \
WANDB_PROJECT=mistral-rl \
WANDB_ENTITY=your_team_or_user \
WANDB_GROUP=ministral-grpo \
WANDB_JOB_TYPE=train \
WANDB_TAGS=grpo,ministral,single-h200,exp1 \
USE_4BIT=0 \
bash scripts/run_grpo_single_h200.sh
```

Set `USE_4BIT=1` only for base models that support bitsandbytes 4-bit loading.
`mistralai/Ministral-3-3B-Instruct-2512` should stay at `USE_4BIT=0`.

## 4) Run without W&B

```bash
python3 scripts/train_grpo_mistral.py \
  --model-name mistralai/Ministral-3-3B-Instruct-2512 \
  --data-dir datasets/unique_prompts_balanced.json \
  --output-dir outputs/mistral-grpo \
  --report-to none \
  --bf16
```

## 5) Direct script usage

```bash
python3 scripts/train_grpo_mistral.py \
  --model-name mistralai/Ministral-3-3B-Instruct-2512 \
  --data-dir datasets/unique_prompts_balanced.json \
  --output-dir outputs/mistral-grpo \
  --run-name ministral-grpo-single-h200 \
  --seed 42 \
  --report-to wandb \
  --wandb-project mistral-rl \
  --wandb-entity your_team_or_user \
  --wandb-group ministral-grpo \
  --wandb-job-type train \
  --wandb-tags grpo,ministral,single-h200 \
  --bf16 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-generations 4
```

## 6) Validation (infer.py)

`scripts/infer.py` runs dataset-level validation for your trained adapter and prints:
- malicious refusal rate
- benign helpfulness rate
- balanced score

Example:

```bash
python3 scripts/infer.py \
  --base-model mistralai/Ministral-3-3B-Instruct-2512 \
  --adapter-path outputs/mistral-grpo \
  --data-dir datasets/unique_prompts_balanced.json \
  --eval-split 0.02 \
  --max-samples 200 \
  --save-predictions outputs/mistral-grpo/validation.json
```

## 7) Validation with vLLM

Install vLLM (separate from `requirements-rl.txt`):

```bash
pip install vllm
```

Validate base model only:

```bash
python3 scripts/validate_vllm.py \
  --model mistralai/Ministral-3-3B-Instruct-2512 \
  --data-dir datasets/unique_prompts_balanced.json \
  --eval-split 0.02 \
  --max-samples 200 \
  --save-predictions outputs/mistral-grpo/validation-vllm-base.json
```

Validate fine-tuned LoRA adapter:

```bash
python3 scripts/validate_vllm.py \
  --model mistralai/Ministral-3-3B-Instruct-2512 \
  --adapter-path outputs/mistral-grpo \
  --data-dir datasets/unique_prompts_balanced.json \
  --eval-split 0.02 \
  --max-samples 200 \
  --save-predictions outputs/mistral-grpo/validation-vllm-lora.json
```

Validate against a remote vLLM server (OpenAI-compatible API):

```bash
export VLLM_API_KEY=your_api_key_if_needed
python3 scripts/validate_vllm_remote.py \
  --base-url http://your-vllm-server:8000 \
  --endpoint /v1/chat/completions \
  --model mistralai/Ministral-3-3B-Instruct-2512 \
  --data-dir datasets/unique_prompts_balanced.json \
  --eval-split 0.02 \
  --max-samples 200 \
  --save-predictions outputs/mistral-grpo/validation-vllm-remote.json
```

## 8) Upload to Hugging Face Hub

Set token once:

```bash
pip install huggingface_hub
export HF_TOKEN=your_hf_token
```

Upload GRPO adapter/output folder:

```bash
python3 scripts/upload_to_hf.py \
  --local-path outputs/mistral-grpo \
  --repo-id your-user/ministral-grpo-lora \
  --repo-type model \
  --private \
  --commit-message \"Upload GRPO adapter checkpoint\"
```

Upload SFT output folder:

```bash
python3 scripts/upload_to_hf.py \
  --local-path outputs/mistral-sft \
  --repo-id your-user/ministral-sft-lora \
  --repo-type model \
  --private \
  --commit-message \"Upload SFT adapter checkpoint\"
```

Upload as public repo:

```bash
python3 scripts/upload_to_hf.py \
  --local-path outputs/Ministral-3-3B-Instruct-sec \
  --repo-id llmtrace/Ministral-3-3B-Instruct-sec \
  --repo-type model \
  --public
```

## Reward behavior

The reward function is label-conditioned:
- `malicious` prompts: reward refusals, penalize unsafe/helpful attack content.
- `benign` prompts: reward normal helpful answers, penalize unnecessary refusals.

This is implemented in `scripts/train_grpo_mistral.py` and can be adjusted easily.
