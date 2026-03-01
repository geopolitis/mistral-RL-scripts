#!/usr/bin/env python3
"""Create a W&B report for the two-stage safety fine-tuning pipeline.

Covers:
  - Stage 1: SFT (refusal-only supervised fine-tuning)
  - Stage 2: GRPO (group relative policy optimization)
  - Cross-stage comparison

Usage:
    WANDB_API_KEY=<key> python scripts/create_wandb_report.py
"""

from __future__ import annotations

import os

import wandb_workspaces.reports.v2 as wr

ENTITY = "evalonlabs"
PROJECT = "mistral-rl"

# Run IDs
SFT_RUN_ID = "vj4yv9gy"
GRPO_FROM_BASE_ID = "blnuzohg"  # GRPO without SFT init (baseline)
GRPO_FROM_BASE_FULL_ID = "cuu1tf30"  # GRPO from base, full run (748 steps)
GRPO_FROM_SFT_ID = "66hj3i0y"  # GRPO initialized from SFT adapter

# Colors
COLOR_SFT = "#2ecc71"
COLOR_GRPO_BASE = "#e74c3c"
COLOR_GRPO_BASE_FULL = "#e67e22"
COLOR_GRPO_SFT = "#3498db"


def make_sft_runset(name: str = "SFT Stage") -> wr.Runset:
    return wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name=name,
        query=SFT_RUN_ID,
        custom_run_colors={SFT_RUN_ID: COLOR_SFT},
    )


def make_grpo_from_sft_runset(name: str = "GRPO (from SFT)") -> wr.Runset:
    return wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name=name,
        query=GRPO_FROM_SFT_ID,
        custom_run_colors={GRPO_FROM_SFT_ID: COLOR_GRPO_SFT},
    )


def make_grpo_comparison_runset(name: str = "GRPO Comparison") -> wr.Runset:
    return wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name=name,
        query=f"{GRPO_FROM_BASE_FULL_ID}|{GRPO_FROM_SFT_ID}",
        custom_run_colors={
            GRPO_FROM_BASE_FULL_ID: COLOR_GRPO_BASE_FULL,
            GRPO_FROM_SFT_ID: COLOR_GRPO_SFT,
        },
    )


def make_all_runs_runset(name: str = "All Key Runs") -> wr.Runset:
    return wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name=name,
        query=f"{SFT_RUN_ID}|{GRPO_FROM_BASE_FULL_ID}|{GRPO_FROM_SFT_ID}",
        custom_run_colors={
            SFT_RUN_ID: COLOR_SFT,
            GRPO_FROM_BASE_FULL_ID: COLOR_GRPO_BASE_FULL,
            GRPO_FROM_SFT_ID: COLOR_GRPO_SFT,
        },
    )


def section_purpose() -> list:
    return [
        wr.H1(text="Purpose"),
        wr.MarkdownBlock(
            text=(
                "[LLMTrace](https://github.com/epappas/llmtrace) is an LLM security proxy that runs "
                "an [ensemble of detectors](https://github.com/epappas/llmtrace/blob/main/docs/ml/ensemble.md) "
                "(regex + DeBERTa + jailbreak heuristics) on live API traffic. It reaches 87.6% accuracy and "
                "86.9% F1 on our [benchmark](https://github.com/epappas/llmtrace/tree/main/benchmarks), "
                "but the 79.7% recall means ~20% of attacks slip through -- novel jailbreaks and prompt "
                "injections the pattern-based detectors haven't seen.\n\n"
                "This project fine-tunes Ministral-3B as a **second-level LLM-as-a-Judge** that reviews "
                "security traces asynchronously, catching what the real-time pipeline missed without "
                "adding latency to the request path. The GRPO-trained checkpoint is published at "
                "[llmtrace/Ministral-3-3B-Instruct-sec](https://huggingface.co/llmtrace/Ministral-3-3B-Instruct-sec).\n\n"
                "Design rationale in the "
                "[Mistral Judge Exploration](https://gist.github.com/epappas/d09ec1142f10a1edaf3ad12c4c427466); "
                "training scripts in [mistral-RL-scripts](https://github.com/geopolitis/mistral-RL-scripts).\n\n"
                "**The core question:** Can a fine-tuned 3.3B model reliably refuse malicious prompts "
                "it sees in traces, without becoming so paranoid that it flags legitimate traffic?\n"
            )
        ),
    ]


def section_novelty() -> list:
    return [
        wr.H1(text="Novelty"),
        wr.MarkdownBlock(
            text=(
                "Three design choices here that we haven't seen combined elsewhere:\n\n"
                "### 1. Refusal-Only SFT\n"
                "This is where most of us get burned with safety fine-tuning. Train on both malicious and "
                "benign examples during SFT, and the model starts over-refusing benign queries -- it learns "
                "\"when in doubt, refuse\" as a shortcut.\n\n"
                "We deliberately excluded all benign prompts from Stage 1. The model only sees malicious "
                "prompts paired with 24 varied refusal templates (direct, explain-why-not, redirect, firm "
                "boundary, concise). It learns *when and how to refuse* -- nothing else. The base model's "
                "natural helpfulness stays untouched.\n\n"
                "### 2. Two-Stage Pipeline (SFT then GRPO)\n"
                "We tried GRPO directly on the base model first (run `cuu1tf30`, 748 steps). The problem: "
                "the base model has no refusal prior, so the reward signal is sparse and noisy -- it doesn't "
                "know *how* to refuse, so it can't discover refusal behavior through RL alone.\n\n"
                "SFT first gives the model a strong refusal baseline. GRPO then refines the tradeoff from a "
                "better starting point. The ablation section below shows whether this actually helped.\n\n"
                "### 3. Asymmetric Reward Function\n"
                "The reward is deliberately lopsided:\n\n"
                "| Scenario | Reward |\n"
                "|----------|--------|\n"
                "| Malicious + refusal | **+1.2** |\n"
                "| Malicious + no refusal | **-1.2** |\n"
                "| Benign + helpful | **+1.0** |\n"
                "| Benign + false refusal | **-0.6** |\n"
                "| Unsafe markers in output | -0.08/hit (malicious), -0.05/hit (benign) |\n"
                "| Length bonus (benign) | up to +0.3 |\n\n"
                "**The practical implication:** the 2:1 penalty ratio on malicious (1.2 vs 0.6) encodes a "
                "safety-first prior. Missing a real threat costs twice as much as an occasional false refusal. "
                "The length bonus on benign responses prevents the model from giving terse, unhelpful answers "
                "when it correctly identifies a query as safe.\n"
            )
        ),
    ]


def section_methodology() -> list:
    return [
        wr.H1(text="Methodology"),
        wr.MarkdownBlock(
            text=(
                "We structured this as an ablation study. Four checkpoints, each isolating one variable:\n\n"
                "| Checkpoint | What It Tests |\n"
                "|------------|---------------|\n"
                "| Base Model | Ministral-3-3B-Instruct, unmodified -- the starting point |\n"
                "| SFT | Does refusal-only supervised training teach the model to refuse? |\n"
                "| GRPO (from base) | Can RL alone teach safety without SFT? |\n"
                "| GRPO (from SFT) | Does SFT give RL a meaningful head start? |\n\n"
                "The key comparison is GRPO-from-base vs GRPO-from-SFT. If the two-stage pipeline "
                "works as intended, the SFT-initialized run should converge faster and reach higher "
                "reward. If it doesn't, the SFT stage is wasted compute.\n\n"
                "### Stage 1 -- SFT (Refusal-Only)\n\n"
                "- Dataset: `unique_prompts.json`, filtered to malicious samples only (~8.5K prompts)\n"
                "- 24 refusal templates across 5 categories, deterministically assigned via md5 hash\n"
                "- Completion-only loss masking: SFTTrainer's prompt/completion columns\n"
                "- LoRA r=32, alpha=64, all linear projections (q/k/v/o/gate/up/down)\n"
                "- LR 5e-5, 1 epoch, batch 4 x 8 grad accum = 32 effective, warmup 0.05\n"
                "- 161 steps total\n\n"
                "### Stage 2 -- GRPO\n\n"
                "- Dataset: `unique_prompts_balanced.json` (malicious + benign)\n"
                "- Initialized from SFT adapter (`outputs/mistral-sft`)\n"
                "- Asymmetric reward function (see Novelty section)\n"
                "- LoRA r=32, alpha=64, QLoRA 4-bit quantization\n"
                "- LR 1.5e-6, 1 epoch, 4 generations per prompt, batch 1 x 16 grad accum = 16 effective\n\n"
                "### What We're Watching For\n\n"
                "- **Reward curve**: Is it climbing? Plateauing? The primary signal that GRPO is working.\n"
                "- **Entropy**: If this drops to near-zero, the model collapsed to a single behavior "
                "(almost always: refuse everything). This is the mode collapse red flag.\n"
                "- **Clip ratio**: High clip fractions mean policy updates are too aggressive. "
                "Should stay below ~0.2.\n"
                "- **Frac reward zero std**: If every generation in a group gets the same reward, "
                "the gradient signal is zero. Training has effectively stopped.\n"
                "- **Completion length**: Safety fine-tuning frequently causes length degeneration. "
                "Watching for collapse to very short outputs.\n"
            )
        ),
    ]


def section_overview() -> list:
    return [
        wr.H1(text="Ministral Safety Fine-Tuning Report"),
        wr.MarkdownBlock(
            text=(
                "We started this project with a hypothesis: a small, fine-tuned 3.3B model can serve as a "
                "reliable second-level safety judge for LLM traffic -- one that catches what our real-time "
                "proxy ([LLMTrace](https://github.com/epappas/llmtrace)) misses, without the latency cost "
                "of running inference on the live path.\n\n"
                "**Model:** [`mistralai/Ministral-3-3B-Instruct-2512-BF16`]"
                "(https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) (3.3B dense, BF16)\n\n"
                "| Stage | Method | Purpose | Dataset |\n"
                "|-------|--------|---------|--------|\n"
                "| 1 | SFT (Refusal-Only) | Teach refusal behavior on malicious prompts | `unique_prompts.json` (malicious only) |\n"
                "| 2 | GRPO | Refine safety-helpfulness tradeoff via RL | `unique_prompts_balanced.json` (malicious + benign) |\n\n"
                "LoRA on all linear projections: r=32, alpha=64, dropout=0.05. "
                "GRPO checkpoint: [llmtrace/Ministral-3-3B-Instruct-sec]"
                "(https://huggingface.co/llmtrace/Ministral-3-3B-Instruct-sec). "
                "Code: [mistral-RL-scripts](https://github.com/geopolitis/mistral-RL-scripts). "
                "Design rationale: [Mistral Judge Exploration]"
                "(https://gist.github.com/epappas/d09ec1142f10a1edaf3ad12c4c427466).\n"
            )
        ),
    ]


def section_sft() -> list:
    return [
        wr.H1(text="Stage 1: SFT Training"),
        wr.MarkdownBlock(
            text=(
                "SFT ran for 161 steps on ~8.5K malicious-only prompts. Each prompt was paired with one of "
                "24 refusal templates, assigned deterministically via md5 hash so it's reproducible. "
                "Loss is computed only on the completion tokens -- the prompt is masked out.\n\n"
                "The training loss curve should drop steeply in the first ~30 steps and plateau. "
                "A loss that keeps dropping toward zero would indicate memorization rather than generalization.\n\n"
                "LR: 5e-5, 1 epoch, warmup: 0.05, batch: 4 x 8 grad accum = 32 effective, max seq: 1024.\n"
            )
        ),
        wr.PanelGrid(
            runsets=[make_sft_runset()],
            panels=[
                wr.LinePlot(
                    title="Training Loss",
                    y=["loss"],
                    title_y="Loss",
                    layout=wr.Layout(x=0, y=0, w=12, h=8),
                ),
                wr.LinePlot(
                    title="Mean Token Accuracy",
                    y=["mean_token_accuracy"],
                    title_y="Accuracy",
                    layout=wr.Layout(x=12, y=0, w=12, h=8),
                ),
                wr.LinePlot(
                    title="Learning Rate Schedule",
                    y=["learning_rate"],
                    title_y="LR",
                    layout=wr.Layout(x=0, y=8, w=8, h=6),
                ),
                wr.LinePlot(
                    title="Gradient Norm",
                    y=["grad_norm"],
                    title_y="Grad Norm",
                    layout=wr.Layout(x=8, y=8, w=8, h=6),
                ),
                wr.LinePlot(
                    title="Entropy",
                    y=["entropy"],
                    title_y="Entropy",
                    layout=wr.Layout(x=16, y=8, w=8, h=6),
                ),
            ],
        ),
    ]


def section_grpo() -> list:
    return [
        wr.H1(text="Stage 2: GRPO Training"),
        wr.MarkdownBlock(
            text=(
                "GRPO refines the SFT checkpoint using the asymmetric reward function described above. "
                "This is where the model learns to balance safety with helpfulness -- SFT only taught it "
                "to refuse, GRPO teaches it *when not to*.\n\n"
                "We're watching the reward curve closely. A healthy run shows reward climbing steadily "
                "without entropy collapsing. If entropy drops to near-zero while reward plateaus high, "
                "the model collapsed to always-refuse (which scores +1.2 on every malicious prompt "
                "but -0.6 on every benign one).\n\n"
                "Init adapter: `outputs/mistral-sft` (Stage 1 checkpoint). "
                "LR: 1.5e-6, 1 epoch, 4 generations per prompt, batch: 1 x 16 grad accum = 16 effective.\n"
            )
        ),
        wr.PanelGrid(
            runsets=[make_grpo_from_sft_runset()],
            panels=[
                wr.LinePlot(
                    title="Reward (mean)",
                    y=["reward"],
                    title_y="Reward",
                    smoothing_type="gaussian",
                    smoothing_factor=0.7,
                    layout=wr.Layout(x=0, y=0, w=12, h=8),
                ),
                wr.LinePlot(
                    title="Training Loss",
                    y=["loss"],
                    title_y="Loss",
                    smoothing_type="gaussian",
                    smoothing_factor=0.5,
                    layout=wr.Layout(x=12, y=0, w=12, h=8),
                ),
                wr.LinePlot(
                    title="Entropy",
                    y=["entropy"],
                    title_y="Entropy",
                    layout=wr.Layout(x=0, y=8, w=8, h=6),
                ),
                wr.LinePlot(
                    title="Reward Std",
                    y=["reward_std"],
                    title_y="Std",
                    smoothing_type="gaussian",
                    smoothing_factor=0.5,
                    layout=wr.Layout(x=8, y=8, w=8, h=6),
                ),
                wr.LinePlot(
                    title="Gradient Norm",
                    y=["grad_norm"],
                    title_y="Grad Norm",
                    layout=wr.Layout(x=16, y=8, w=8, h=6),
                ),
                wr.LinePlot(
                    title="Clip Ratio (mean)",
                    y=["clip_ratio/region_mean"],
                    title_y="Clip Ratio",
                    layout=wr.Layout(x=0, y=14, w=8, h=6),
                ),
                wr.LinePlot(
                    title="Completion Length (mean)",
                    y=["completions/mean_length"],
                    title_y="Tokens",
                    layout=wr.Layout(x=8, y=14, w=8, h=6),
                ),
                wr.LinePlot(
                    title="Frac Reward Zero Std",
                    y=["frac_reward_zero_std"],
                    title_y="Fraction",
                    layout=wr.Layout(x=16, y=14, w=8, h=6),
                ),
            ],
        ),
    ]


def section_comparison() -> list:
    return [
        wr.H1(text="GRPO Comparison: Base vs SFT-Initialized"),
        wr.MarkdownBlock(
            text=(
                "This is the section that answers the question: **does SFT actually help GRPO?**\n\n"
                "Two GRPO runs, same reward function, same dataset (`unique_prompts_balanced.json`), "
                "different starting points:\n\n"
                "- **GRPO from base** (orange, `cuu1tf30`): RL applied directly to the unmodified base model. "
                "Ran for 748 steps.\n"
                "- **GRPO from SFT** (blue, `66hj3i0y`): RL applied to the SFT checkpoint. "
                "Still running.\n\n"
                "If the two-stage hypothesis is correct, the blue line should converge faster and reach "
                "higher reward -- the SFT stage gave it a head start on refusal behavior, so GRPO has "
                "less to learn. If the lines look the same, SFT was wasted compute.\n"
            )
        ),
        wr.PanelGrid(
            runsets=[make_grpo_comparison_runset()],
            panels=[
                wr.LinePlot(
                    title="Reward Comparison",
                    y=["reward"],
                    title_y="Reward",
                    smoothing_type="gaussian",
                    smoothing_factor=0.6,
                    layout=wr.Layout(x=0, y=0, w=12, h=8),
                ),
                wr.LinePlot(
                    title="Loss Comparison",
                    y=["loss"],
                    title_y="Loss",
                    smoothing_type="gaussian",
                    smoothing_factor=0.5,
                    layout=wr.Layout(x=12, y=0, w=12, h=8),
                ),
                wr.LinePlot(
                    title="Entropy Comparison",
                    y=["entropy"],
                    title_y="Entropy",
                    layout=wr.Layout(x=0, y=8, w=12, h=6),
                ),
                wr.LinePlot(
                    title="Reward Std Comparison",
                    y=["reward_std"],
                    title_y="Std",
                    smoothing_type="gaussian",
                    smoothing_factor=0.5,
                    layout=wr.Layout(x=12, y=8, w=12, h=6),
                ),
                wr.LinePlot(
                    title="Gradient Norm Comparison",
                    y=["grad_norm"],
                    title_y="Grad Norm",
                    layout=wr.Layout(x=0, y=14, w=12, h=6),
                ),
                wr.LinePlot(
                    title="Completion Length Comparison",
                    y=["completions/mean_length"],
                    title_y="Tokens",
                    layout=wr.Layout(x=12, y=14, w=12, h=6),
                ),
            ],
        ),
    ]


def section_summary() -> list:
    return [
        wr.H1(text="Final Summary Metrics"),
        wr.MarkdownBlock(
            text=(
                "Final numbers across all key runs. The bar plots show where each run ended up; "
                "the run comparer below shows the full config diff so nothing is hidden.\n"
            )
        ),
        wr.PanelGrid(
            runsets=[make_all_runs_runset()],
            panels=[
                wr.BarPlot(
                    title="Final Loss",
                    metrics=["loss"],
                    orientation="v",
                    layout=wr.Layout(x=0, y=0, w=8, h=8),
                ),
                wr.BarPlot(
                    title="Final Reward",
                    metrics=["reward"],
                    orientation="v",
                    layout=wr.Layout(x=8, y=0, w=8, h=8),
                ),
                wr.BarPlot(
                    title="Final Entropy",
                    metrics=["entropy"],
                    orientation="v",
                    layout=wr.Layout(x=16, y=0, w=8, h=8),
                ),
                wr.RunComparer(
                    layout=wr.Layout(x=0, y=8, w=24, h=10),
                    diff_only="split",
                ),
            ],
        ),
    ]


def section_config() -> list:
    return [
        wr.H1(text="Training Configuration Reference"),
        wr.MarkdownBlock(
            text=(
                "| Parameter | SFT | GRPO (from base) | GRPO (from SFT) |\n"
                "|-----------|-----|-------------------|------------------|\n"
                "| Learning Rate | 5e-5 | 2e-6 | 1.5e-6 |\n"
                "| Epochs | 1 | 1 | 1 |\n"
                "| Batch Size | 4 | 1 | 1 |\n"
                "| Grad Accum | 8 | 16 | 16 |\n"
                "| Effective Batch | 32 | 16 | 16 |\n"
                "| Max Seq Length | 1024 | 384+96 | 512+192 |\n"
                "| Num Generations | N/A | 8 | 4 |\n"
                "| Warmup | 0.05 | 0.03 | 0.03 |\n"
                "| LoRA r | 32 | 32 | 32 |\n"
                "| 4-bit Quant | No | No | Yes |\n"
                "| Init Adapter | None | None | outputs/mistral-sft |\n"
                "| Dataset | unique_prompts.json | unique_prompts.json | unique_prompts_balanced.json |\n"
            )
        ),
    ]


def main() -> None:
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set WANDB_API_KEY before running this script.")

    blocks = []
    blocks.extend(section_overview())
    blocks.append(wr.TableOfContents())
    blocks.append(wr.HorizontalRule())
    blocks.extend(section_purpose())
    blocks.append(wr.HorizontalRule())
    blocks.extend(section_novelty())
    blocks.append(wr.HorizontalRule())
    blocks.extend(section_methodology())
    blocks.append(wr.HorizontalRule())
    blocks.extend(section_sft())
    blocks.append(wr.HorizontalRule())
    blocks.extend(section_grpo())
    blocks.append(wr.HorizontalRule())
    blocks.extend(section_comparison())
    blocks.append(wr.HorizontalRule())
    blocks.extend(section_summary())
    blocks.append(wr.HorizontalRule())
    blocks.extend(section_config())

    report = wr.Report(
        project=PROJECT,
        entity=ENTITY,
        title="Ministral Safety Fine-Tuning: SFT + GRPO Pipeline",
        description="Two-stage safety fine-tuning for Ministral-3-3B. Stage 1: Refusal-only SFT. Stage 2: GRPO with safety-helpfulness reward.",
        blocks=blocks,
        width="readable",
    )

    report.save()
    print(f"Report created: {report.url}")


if __name__ == "__main__":
    main()
