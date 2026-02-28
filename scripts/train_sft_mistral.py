#!/usr/bin/env python3
"""Stage 1: Refusal-only SFT for Mistral on prompt-injection datasets.

Trains the model to refuse malicious/unsafe prompts using completion-only
loss masking.  Benign prompts are excluded -- the base model's natural
helpfulness is preserved for GRPO Stage 2.  Run SFT BEFORE GRPO.

Expected dataset schema (JSON list per file):
- text: prompt string
- label: benign | malicious
- category: optional fine-grained category
"""

from __future__ import annotations

import argparse
import glob
import hashlib
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
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Response templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Refuse requests that are harmful, unsafe, or malicious."
)

MALICIOUS_REFUSALS: list[str] = [
    # Direct refusal
    (
        "I'm sorry, but I can't help with that request. "
        "It appears to involve potentially harmful or unsafe content. "
        "If you have a different question, I'm happy to assist."
    ),
    (
        "I cannot assist with this request as it involves content that could be harmful. "
        "Please let me know if there's something else I can help you with."
    ),
    (
        "I'm not able to help with that. This request appears to involve unsafe content. "
        "I'd be glad to help with other questions you might have."
    ),
    (
        "Sorry, but I won't be able to assist with this particular request. "
        "It seems to involve content that could cause harm. "
        "Feel free to ask me something else."
    ),
    (
        "I can't comply with this request. If you have a legitimate question "
        "or need help with something constructive, I'm here to help."
    ),
    # Explain-why-not
    (
        "I cannot help with this because it involves content that could lead to harm. "
        "My purpose is to assist with safe and constructive tasks."
    ),
    (
        "I'm sorry, but I'm unable to assist here. This request involves potentially "
        "dangerous or harmful activity that I cannot support."
    ),
    (
        "I can't provide assistance with this. The request appears to involve "
        "unsafe or malicious intent, which goes against my guidelines."
    ),
    (
        "Sorry, I will not help with that. Requests that could result in harm to "
        "people or systems are outside what I can assist with."
    ),
    (
        "I cannot comply with this request. It contains elements that could be "
        "used to cause harm, and I'm not able to support that."
    ),
    # Redirect
    (
        "I can't help with that particular request, but I'd be happy to help "
        "with something constructive instead."
    ),
    (
        "I'm unable to assist with this, but if you have a different question "
        "that doesn't involve harmful content, I'm here to help."
    ),
    (
        "Sorry, I won't be assisting with that. However, I can help you with "
        "other topics that are safe and constructive."
    ),
    (
        "I cannot provide help with this request. If you'd like to ask about "
        "something else, I'm ready to assist."
    ),
    (
        "I'm not able to help with that. I'd be glad to assist you with a "
        "different, constructive question instead."
    ),
    # Firm boundary
    (
        "I won't assist with requests that could cause harm. "
        "Please reach out with a different question."
    ),
    (
        "I can't and won't help with this type of request. "
        "It falls outside the boundaries of what I can safely assist with."
    ),
    (
        "Sorry, but this is not something I'm able to help with. "
        "I cannot support requests that may be harmful or unsafe."
    ),
    (
        "I will not provide assistance with this request. "
        "It involves content that I cannot responsibly engage with."
    ),
    (
        "I'm unable to comply with this. My guidelines prevent me from "
        "assisting with requests that could lead to harm."
    ),
    # Concise refusal
    (
        "I can't help with that. This request involves harmful content."
    ),
    (
        "Sorry, I cannot assist with this type of request."
    ),
    (
        "I won't help with that. Please ask me something else."
    ),
    (
        "I'm sorry, but I'm not able to assist with this request."
    ),
    (
        "I cannot comply with this. Let me know if I can help with something else."
    ),
]


@dataclass
class Example:
    prompt: str
    label: str
    category: str


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
        if not logs or not state.is_world_process_zero:
            return
        payload = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if payload:
            self._wandb.log(payload, step=int(state.global_step))


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT fine-tuning for Mistral (Stage 1, before GRPO)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/unique_prompts.json",
        help="Path to a JSON file or directory with *.json files",
    )
    parser.add_argument("--model-name", type=str, default="mistralai/Ministral-3-3B-Instruct-2512-BF16")
    parser.add_argument("--output-dir", type=str, default="outputs/mistral-sft")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all samples")
    parser.add_argument("--train-split", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-seq-length", type=int, default=1024)

    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 uses num-train-epochs")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--use-4bit", action="store_true", help="Enable bitsandbytes 4-bit quantization")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--run-name", type=str, default="ministral-sft-single-h200")
    parser.add_argument("--report-to", type=str, default="none", help="none | wandb | tensorboard")
    parser.add_argument("--wandb-project", type=str, default="", help="Optional W&B project name")
    parser.add_argument("--wandb-entity", type=str, default="", help="Optional W&B entity/team")
    parser.add_argument("--wandb-group", type=str, default="", help="Optional W&B run group")
    parser.add_argument("--wandb-job-type", type=str, default="train", help="W&B job type")
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Comma-separated W&B tags, e.g. 'sft,ministral,single-h200'",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading & formatting
# ---------------------------------------------------------------------------


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
            rows.append(
                Example(
                    prompt=prompt,
                    label=normalize_label(record.get("label", "benign")),
                    category=str(record.get("category", "unknown")).strip().lower(),
                )
            )

    total_loaded = len(rows)
    rows = [r for r in rows if r.label == "malicious"]
    print(f"[data] filtered to malicious-only: {len(rows)}/{total_loaded} examples kept")

    random.Random(seed).shuffle(rows)
    if max_samples > 0:
        rows = rows[:max_samples]
    if len(rows) < 10:
        raise ValueError(f"Not enough malicious rows loaded ({len(rows)}).")
    return rows


def _pick_response(prompt: str, responses: list[str]) -> str:
    """Deterministically pick a response template based on prompt hash."""
    idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(responses)
    return responses[idx]


def format_as_chat(
    examples: list[Example],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> list[dict[str, str]]:
    """Convert malicious examples to chat-template strings with refusal responses.

    Each example becomes: system prompt + user prompt + assistant refusal.
    Completion-only loss masking (via DataCollatorForCompletionOnlyLM) ensures
    gradients flow only through the assistant refusal tokens.
    """
    formatted: list[dict[str, str]] = []

    for ex in examples:
        response = _pick_response(ex.prompt, MALICIOUS_REFUSALS)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex.prompt},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted.append({"text": text})

    return formatted


def split_and_format(
    rows: list[Example],
    train_split: float,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> tuple[Dataset, Dataset]:
    split_idx = int(len(rows) * train_split)
    split_idx = max(1, min(split_idx, len(rows) - 1))

    train_rows = rows[:split_idx]
    eval_rows = rows[split_idx:]

    train_formatted = format_as_chat(train_rows, tokenizer, max_seq_length)
    eval_formatted = format_as_chat(eval_rows, tokenizer, max_seq_length)

    train_ds = Dataset.from_list(train_formatted)
    eval_ds = Dataset.from_list(eval_formatted)

    print(f"[data] train: {len(train_ds)} malicious samples, eval: {len(eval_ds)} malicious samples")

    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else "sdpa"
    if attn_impl != "flash_attention_2":
        print("[setup] flash-attn not found; falling back to sdpa attention.")

    target_dtype = torch.bfloat16 if args.bf16 else torch.float16
    model_config = AutoConfig.from_pretrained(args.model_name)

    # Check for FP8 quantization in model config
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

    # Apply LoRA
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

    return model, tokenizer


# ---------------------------------------------------------------------------
# Completion-only loss masking
# ---------------------------------------------------------------------------


def detect_response_template(tokenizer: AutoTokenizer) -> str:
    """Detect the assistant turn marker from the tokenizer's chat template.

    Compares template output with and without add_generation_prompt to isolate
    the exact string the tokenizer inserts before the assistant's first token.
    DataCollatorForCompletionOnlyLM uses this to mask all prompt tokens from loss.
    """
    prompt_messages = [
        {"role": "system", "content": "PLACEHOLDER_SYS"},
        {"role": "user", "content": "PLACEHOLDER_USR"},
    ]
    without_gen = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=False,
    )
    with_gen = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    response_template = with_gen[len(without_gen):]
    assert response_template.strip(), (
        f"Could not detect response template.\n"
        f"Without gen prompt: {without_gen!r}\n"
        f"With gen prompt: {with_gen!r}"
    )
    print(f"[setup] detected response template: {response_template!r}")
    return response_template


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # W&B setup
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

    # Load data
    rows = load_examples(args.data_dir, args.max_samples, args.seed)

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # Format data as chat conversations and split
    train_ds, eval_ds = split_and_format(rows, args.train_split, tokenizer, args.max_seq_length)

    # Build completion-only data collator
    response_template = detect_response_template(tokenizer)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Sanity check: verify the collator finds the response template in tokenized samples.
    # If it doesn't, every label is masked to -100 and training silently produces zero loss.
    _sample_encoded = tokenizer(train_ds[0]["text"], return_tensors="pt")
    _sample_batch = data_collator([{k: v.squeeze(0) for k, v in _sample_encoded.items()}])
    _non_masked = (_sample_batch["labels"][0] != -100).sum().item()
    assert _non_masked > 0, (
        f"DataCollator masked all tokens -- response template {response_template!r} "
        f"not found in tokenized text. Check tokenizer chat template compatibility."
    )
    print(f"[setup] collator sanity check passed: {_non_masked} tokens unmasked in sample")

    # Build SFTConfig
    sft_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": args.per_device_batch_size,
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
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_seq_length": args.max_seq_length,
        "dataset_text_field": "text",
        "packing": False,
        "report_to": "none" if args.report_to == "wandb" else args.report_to,
        "dataloader_num_workers": 0,
        "optim": "paged_adamw_8bit",
        "max_grad_norm": 0.3,
        "lr_scheduler_type": "cosine",
    }

    # Handle eval_strategy naming (varies by TRL version)
    sft_params = inspect.signature(SFTConfig.__init__).parameters
    if "evaluation_strategy" in sft_params:
        sft_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in sft_params:
        sft_kwargs["eval_strategy"] = "steps"

    # Filter to supported args only
    filtered_kwargs = {k: v for k, v in sft_kwargs.items() if k in sft_params}
    dropped = sorted(set(sft_kwargs) - set(filtered_kwargs))
    if dropped:
        print(f"[setup] ignoring unsupported SFTConfig args: {', '.join(dropped)}")

    training_args = SFTConfig(**filtered_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    if args.report_to == "wandb" and wandb_module is not None:
        trainer.add_callback(WandbStepLoggerCallback(wandb_module))

    try:
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[done] SFT checkpoint saved to {args.output_dir}")
        print("[done] Use this checkpoint as the base for GRPO (Stage 2).")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
