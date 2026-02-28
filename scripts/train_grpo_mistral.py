#!/usr/bin/env python3
"""Single-GPU GRPO fine-tuning for Mistral on prompt-injection style datasets.

Expected dataset schema (JSON list per file):
- text: prompt string
- label: benign | malicious
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments, set_seed
from trl import GRPOConfig, GRPOTrainer


REFUSAL_MARKERS = (
    "i can",
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
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for Mistral on a single H100")
    parser.add_argument("--data-dir", type=str, default="datasets", help="Directory with *.json files")
    parser.add_argument("--model-name", type=str, default="mistralai/Ministral-3-14B-Instruct-2512")
    parser.add_argument("--output-dir", type=str, default="outputs/mistral-grpo")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all samples")
    parser.add_argument("--train-split", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument("--num-generations", type=int, default=4)

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
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--run-name", type=str, default="ministral-grpo-single-h100")
    parser.add_argument("--report-to", type=str, default="none", help="none | wandb | tensorboard")
    parser.add_argument("--wandb-project", type=str, default="", help="Optional W&B project name")
    parser.add_argument("--wandb-entity", type=str, default="", help="Optional W&B entity/team")
    parser.add_argument("--wandb-group", type=str, default="", help="Optional W&B run group")
    parser.add_argument("--wandb-job-type", type=str, default="train", help="W&B job type")
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Comma-separated W&B tags, e.g. 'grpo,ministral,single-h100'",
    )
    return parser.parse_args()


def normalize_label(raw: Any) -> str:
    value = str(raw).strip().lower()
    if value in {"malicious", "attack", "unsafe", "harmful"}:
        return "malicious"
    return "benign"


def load_examples(data_dir: str, max_samples: int, seed: int) -> list[Example]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

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

    model_init_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "attn_implementation": "flash_attention_2",
        "use_cache": False,
    }

    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            )
        except Exception as exc:
            raise RuntimeError("--use-4bit requested but bitsandbytes config failed") from exc

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        report_to="none" if args.report_to == "wandb" else args.report_to,
        dataloader_num_workers=2,
        model_init_kwargs=model_init_kwargs,
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
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
