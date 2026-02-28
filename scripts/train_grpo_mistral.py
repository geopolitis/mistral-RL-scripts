#!/usr/bin/env python3
"""Single-GPU GRPO fine-tuning for Mistral on prompt-injection style datasets.

Expected dataset schema (JSON list per file):
- text: prompt string
- label: benign | malicious
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import inspect
import json
import os
import random
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoTokenizer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments, set_seed
from trl import GRPOConfig, GRPOTrainer


REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i won\u2019t",
    "i will not",
    "cannot help",
    "can't help",
    "won't help",
    "not able to",
    "cannot comply",
    "can't comply",
    "unable to",
    "sorry",
)

UNSAFE_MARKERS = (
    "bypass",
    "ignore previous",
    "jailbreak",
    "disable safety",
    "exploit",
    "steal",
    "credential",
    "password",
    "api key",
)


@dataclass
class Example:
    prompt: str
    label: str


class WandbStepLoggerCallback(TrainerCallback):
    """Logs trainer metrics to W&B with explicit step indexing."""

    def __init__(self, wandb_module: Any) -> None:
        self._wandb = wandb_module

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        if not state.is_world_process_zero:
            return

        payload: dict[str, Any] = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                payload[key] = value
        if payload:
            self._wandb.log(payload, step=int(state.global_step))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for Mistral on a single H200")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/unique_prompts_balanced.json",
        help="Path to a JSON file or directory with *.json files",
    )
    parser.add_argument("--model-name", type=str, default="mistralai/Ministral-3-3B-Instruct-2512-BF16")
    parser.add_argument("--output-dir", type=str, default="outputs/mistral-grpo")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all samples")
    parser.add_argument("--train-split", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument("--num-generations", type=int, default=2)

    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 uses num-train-epochs")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)

    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--use-4bit", action="store_true", help="Enable bitsandbytes 4-bit quantization")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--run-name", type=str, default="ministral-grpo-single-h200")
    parser.add_argument("--report-to", type=str, default="none", help="none | wandb | tensorboard")
    parser.add_argument("--wandb-project", type=str, default="", help="Optional W&B project name")
    parser.add_argument("--wandb-entity", type=str, default="", help="Optional W&B entity/team")
    parser.add_argument("--wandb-group", type=str, default="", help="Optional W&B run group")
    parser.add_argument("--wandb-job-type", type=str, default="train", help="W&B job type")
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Comma-separated W&B tags, e.g. 'grpo,ministral,single-h200'",
    )
    return parser.parse_args()


def normalize_label(raw: Any) -> str:
    value = str(raw).strip().lower()
    if value in {"malicious", "attack", "unsafe", "harmful"}:
        return "malicious"
    return "benign"


def load_examples(data_dir: str, max_samples: int, seed: int) -> list[Example]:
    if os.path.isfile(data_dir):
        files = [data_dir]
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON file(s) found in {data_dir}")

    rows: list[Example] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for record in data:
            if not isinstance(record, dict):
                continue
            prompt = str(record.get("text", "")).strip()
            if not prompt:
                continue
            rows.append(Example(prompt=prompt, label=normalize_label(record.get("label", "benign"))))

    random.Random(seed).shuffle(rows)
    if max_samples > 0:
        rows = rows[:max_samples]
    if len(rows) < 10:
        raise ValueError(f"Not enough rows loaded ({len(rows)}).")
    return rows


def split_dataset(rows: list[Example], train_split: float) -> tuple[Dataset, Dataset]:
    split_idx = int(len(rows) * train_split)
    split_idx = max(1, min(split_idx, len(rows) - 1))
    train_rows = rows[:split_idx]
    eval_rows = rows[split_idx:]

    train_ds = Dataset.from_list([{"prompt": r.prompt, "label": r.label} for r in train_rows])
    eval_ds = Dataset.from_list([{"prompt": r.prompt, "label": r.label} for r in eval_rows])
    return train_ds, eval_ds


def truncate_prompt(prompt: str, tokenizer: AutoTokenizer, max_prompt_length: int) -> str:
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
        return_attention_mask=False,
    )
    return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True).strip()


def apply_prompt_truncation(dataset: Dataset, tokenizer: AutoTokenizer, max_prompt_length: int) -> Dataset:
    def _map_fn(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        return {
            "prompt": [truncate_prompt(p, tokenizer, max_prompt_length) for p in batch["prompt"]],
        }

    return dataset.map(_map_fn, batched=True, desc=f"Truncating prompts to {max_prompt_length} tokens")


def extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion["content"])
        return str(completion)

    return str(completion)


def is_refusal(text: str) -> bool:
    s = text.strip().lower()
    return any(marker in s for marker in REFUSAL_MARKERS)


def unsafe_marker_count(text: str) -> int:
    s = text.lower()
    return sum(1 for marker in UNSAFE_MARKERS if marker in s)


def reward_fn(completions: list[Any], label: list[str], **_: Any) -> list[float]:
    rewards: list[float] = []

    for completion, y in zip(completions, label, strict=False):
        output = extract_completion_text(completion)
        y = normalize_label(y)

        refusal = is_refusal(output)
        unsafe_hits = unsafe_marker_count(output)
        length_words = len(output.split())

        if y == "malicious":
            reward = 1.2 if refusal else -1.2
            reward -= 0.08 * unsafe_hits
            if length_words > 120:
                reward -= 0.1
        else:
            reward = 1.0 if not refusal else -0.6
            reward += min(length_words / 120.0, 0.3)
            reward -= 0.05 * unsafe_hits

        rewards.append(float(max(-2.0, min(2.0, reward))))

    return rewards


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    wandb_run: Any | None = None
    wandb_module: Any | None = None

    if args.report_to == "wandb":
        import wandb

        wandb_module = wandb
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
        wandb_run = wandb.init(
            project=args.wandb_project or None,
            entity=args.wandb_entity or None,
            name=args.run_name,
            group=args.wandb_group or None,
            job_type=args.wandb_job_type or None,
            tags=tags or None,
            config=vars(args),
        )

    rows = load_examples(args.data_dir, args.max_samples, args.seed)
    train_ds, eval_ds = split_dataset(rows, args.train_split)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ds = apply_prompt_truncation(train_ds, tokenizer, args.max_prompt_length)
    eval_ds = apply_prompt_truncation(eval_ds, tokenizer, args.max_prompt_length)

    attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else "sdpa"
    if attn_impl != "flash_attention_2":
        print("[setup] flash-attn not found; falling back to sdpa attention.")

    target_dtype = torch.bfloat16 if args.bf16 else torch.float16
    model_config = AutoConfig.from_pretrained(args.model_name)
    existing_quant = getattr(model_config, "quantization_config", None)
    if isinstance(existing_quant, dict):
        quant_type_val = str(existing_quant.get("quant_type", ""))
    else:
        quant_type_val = str(getattr(existing_quant, "quant_type", ""))
    is_fp8 = existing_quant is not None and "fp8" in quant_type_val.lower()

    load_kwargs: dict[str, Any] = {
        "dtype": target_dtype,
        "device_map": {"": 0},
        "attn_implementation": attn_impl,
    }

    if args.use_4bit and not is_fp8:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=target_dtype,
        )
    elif args.use_4bit and is_fp8:
        print("[setup] model has FP8 quantization; skipping --use-4bit.")

    print(f"[setup] loading model {args.model_name}...")
    model_cls = getattr(transformers, model_config.architectures[0])
    model = model_cls.from_pretrained(args.model_name, **load_kwargs)

    if is_fp8:
        assert hasattr(model, "dequantize"), (
            "Model is FP8 quantized but dequantize() is not available. "
            "Upgrade transformers or use a non-FP8 checkpoint."
        )
        print("[setup] dequantizing FP8 weights to BF16 for LoRA training...")
        model = model.dequantize()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    grpo_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "data_seed": args.seed,
        "bf16": args.bf16,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "report_to": "none" if args.report_to == "wandb" else args.report_to,
        "dataloader_num_workers": 0,
    }

    grpo_params = inspect.signature(GRPOConfig.__init__).parameters
    if "evaluation_strategy" in grpo_params:
        grpo_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in grpo_params:
        grpo_kwargs["eval_strategy"] = "steps"

    filtered_grpo_kwargs = {k: v for k, v in grpo_kwargs.items() if k in grpo_params}
    dropped_grpo_kwargs = sorted(set(grpo_kwargs) - set(filtered_grpo_kwargs))
    if dropped_grpo_kwargs:
        print(f"[setup] ignoring unsupported GRPOConfig args: {', '.join(dropped_grpo_kwargs)}")

    training_args = GRPOConfig(**filtered_grpo_kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    if args.report_to == "wandb" and wandb_module is not None:
        trainer.add_callback(WandbStepLoggerCallback(wandb_module))

    try:
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
