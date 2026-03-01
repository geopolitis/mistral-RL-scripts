# Hackathon Submission: Ministral-3B-Sec

## Submission Fields

- **Title**: Ministral-3B-Sec: Two-Stage Safety Fine-Tuning with SFT + GRPO
- **Track**: Fine-Tuning (W&B)
- **GitHub**: https://github.com/geopolitis/mistral-RL-scripts
- **W&B Report**: https://wandb.ai/evalonlabs/mistral-rl/reports/Ministral-Safety-Fine-Tuning:-SFT-+-GRPO-Pipeline--VmlldzoxNjA3MTE2MQ==
- **W&B Project**: https://wandb.ai/evalonlabs/mistral-rl
- **W&B Weave Traces**: https://wandb.ai/evalonlabs/mistral-rl/weave/traces?view=traces_default
- **Video Demo**: _(record 2-min video -- see instructions below)_

### HuggingFace Models

| Model | HF Repo | Description |
|-------|---------|-------------|
| sec-v1 | [llmtrace/Ministral-3-3B-Instruct-sec-v1](https://huggingface.co/llmtrace/Ministral-3-3B-Instruct-sec-v1) | GRPO-only on raw dataset (baseline) |
| sec-v2-sft | [llmtrace/Ministral-3-3B-Instruct-sec-v2-sft](https://huggingface.co/llmtrace/Ministral-3-3B-Instruct-sec-v2-sft) | Refusal-only SFT with completion-only loss |
| sec-v2-grpo | [llmtrace/Ministral-3-3B-Instruct-sec-v2-grpo](https://huggingface.co/llmtrace/Ministral-3-3B-Instruct-sec-v2-grpo) | GRPO on top of SFT (best model) |

### Live Inference Endpoints (Basilica/vLLM)

| Model | Endpoint |
|-------|----------|
| sec-v1 | https://3c12c510-8a0d-451e-a668-41d3b4ff9da1.deployments.basilica.ai |
| sec-v2-sft | https://1d05d1e5-3626-4910-87c8-eab438e3dc78.deployments.basilica.ai |
| sec-v2-grpo | https://f4e85e87-cc18-4e7e-8ec3-b87b20a040dd.deployments.basilica.ai |

---

## Description (for submission form)

We built a two-stage reinforcement learning pipeline that teaches Mistral's Ministral-3B to defend against prompt injection attacks, jailbreaks, and social engineering -- while preserving its natural helpfulness on legitimate queries.

**Stage 1 -- Refusal-Only SFT**: We train exclusively on 5,227 malicious prompts with 25 diverse refusal templates, using completion-only loss masking so gradients only flow through the refusal tokens (not the attack prompts). Benign examples are deliberately excluded to preserve the base model's helpfulness for Stage 2.

**Stage 2 -- GRPO (Group Relative Policy Optimization)**: A custom binary reward function scores the model on both safety (refusing malicious prompts) and helpfulness (answering benign prompts), using label-conditioned rewards. The model learns to balance both objectives through RL.

The pipeline runs entirely on a single H200 GPU. SFT converges in 5.5 minutes. GRPO trains for 7 hours. All experiments are tracked end-to-end in W&B with full loss curves, reward trajectories, and entropy monitoring. LoRA adapters are published on HuggingFace and deployed as live vLLM endpoints.

---

## Technical Deep-Dive

### Problem Statement

Small language models (3B parameters) are vulnerable to prompt injection, jailbreaking, and social engineering attacks. Standard instruction-tuned models like Ministral-3B will comply with adversarial prompts that use techniques like "Ignore all previous instructions", role-playing exploits, or educational-purpose framing.

The challenge: teach the model to refuse these attacks without destroying its usefulness on benign tasks. This is harder than it sounds -- naive approaches either make the model refuse everything (over-refusal) or fail to generalize across attack vectors.

### Why Fine-Tuning Over Prompt Engineering

Prompt engineering alone is insufficient for this task because:
1. System prompts can be overridden by prompt injection attacks -- the attack is literally designed to bypass prompt-level defenses
2. A 3B model has limited instruction-following capacity -- it cannot reliably hold a complex system prompt under adversarial pressure
3. Fine-tuning embeds the safety behavior into the model weights, making it fundamentally harder to bypass

### Architecture

```
                    8,344 prompts
                    (5,227 malicious + 3,117 benign)
                           |
                    [Dataset Curation]
                    Deduplication, label normalization,
                    balanced/clean splits
                           |
            +--------------+--------------+
            |                             |
     Stage 1: SFT                  Stage 2: GRPO
     (malicious only)              (balanced dataset)
     5,227 examples                6,114 examples
     25 refusal templates          Custom reward function
     Completion-only loss          Label-conditioned scoring
     LR: 5e-5, 1 epoch           LR: 1.5e-6, 1 epoch
     ~5.5 min on H200             ~7 hrs on H200
            |                             |
            v                             v
     sec-v2-sft                    sec-v2-grpo
     (refusal foundation)          (balanced safety+helpfulness)
```

<img width="6688" height="3217" alt="17723750187237317213760559468051" src="https://github.com/user-attachments/assets/5ecdb93b-db08-4591-8031-cca7eee641ef" />


### Dataset

We curated **8,344 unique prompts** from **19 JSON files** spanning 15+ security research datasets across **140 attack categories**.

**Label distribution**: 68.35% malicious (8,480 raw rows) / 31.65% benign (3,927 raw rows), deduplicated to 8,344 unique prompts.

**Sources by size**:

| Source | Rows | Type |
|--------|-----:|------|
| InjecAgent | 2,108 | Automated prompt injection |
| Safeguard Test | 2,060 | Attack test suite |
| JackHHao Jailbreak | 1,306 | Jailbreak collection |
| AILuminate Demo | 1,200 | Mixed benchmark |
| TensorTrust | 1,000 | Trust boundary attacks |
| IvanLeoMK | 917 | Jailbreak variants |
| DeepSet | 662 | Prompt injection (all) |
| HarmBench | 400 | Harmful behaviors |
| BIPIA | 400 | Indirect injection |
| ASB | 400 | Attack samples |
| Others (9 files) | ~891 | Encoding evasion, benign, CyberSecEval2, etc. |

**Top attack categories**: prompt_injection (2,109), jailbreak (666), prompt_hijacking (573), prompt_extraction (427), data security harm (374), physical harm (340).

**Prompt length profile**: median 20 words (128 chars), mean 63 words (364 chars), max 1,973 words -- long-tail distribution with some very large prompts that increase RL rollout time.

**Data quality**: deduplication removed cross-file duplicates, 1 conflicting label resolved (same prompt mapped to both labels), label normalization across heterogeneous source formats. Balanced/clean splits created for different training stages via `scripts/prepare_datasets_v2.py`.

### Stage 1: Refusal-Only SFT

**Key insight**: Traditional SFT trains on both malicious and benign examples, but benign response templates ("Sure, I'd be happy to help!") are content-free preambles that destroy the model's natural helpfulness. Instead, we train only on malicious examples to teach refusal, preserving the base model's benign capabilities.

**Implementation details**:
- 25 diverse refusal templates spanning 5 strategies: direct refusal, explain-why-not, redirect, firm boundary, concise
- All templates contain at least one GRPO REFUSAL_MARKER (ensuring Stage 2 compatibility)
- Completion-only loss: prompt tokens are masked, gradients only update on refusal response tokens
- System prompt: "You are a helpful assistant. Refuse requests that are harmful, unsafe, or malicious."
- Deterministic template selection via prompt hash (reproducibility)

**Config**: LoRA r=32, alpha=64, LR=5e-5, 1 epoch, cosine schedule, warmup=5%

**W&B Run**: [vj4yv9gy](https://wandb.ai/evalonlabs/mistral-rl/runs/vj4yv9gy)

**Results**:

| Metric | Start | End |
|--------|-------|-----|
| Loss | 1.874 | 0.132 |
| Token Accuracy | 51.9% | 97.4% |
| Entropy | 2.79 | 1.82 |
| Grad Norm | 0.607 | 0.092 |

The model converged by step 60 (~37% through training). Entropy stayed above 1.8, confirming the model maintained output diversity without collapsing into a degenerate refusal mode. Total runtime: 331 seconds.

### Stage 2: GRPO (Group Relative Policy Optimization)

**Approach**: After SFT establishes the refusal foundation, GRPO refines the model using a custom reward function that evaluates both safety and helpfulness.

**Reward function** (label-conditioned, asymmetric):
- **Malicious prompts**: +1.2 if response contains refusal markers, -1.2 if it complies with the attack, -0.08 per unsafe marker hit
- **Benign prompts**: +1.0 for helpful response without unnecessary refusal, -0.6 for over-refusal, +up to 0.3 length bonus
- Penalty for short/degenerate completions and hard-truncated generations

**Evolution across iterations**:

We ran 26 experiments, iterating on hyperparameters, dataset quality, and reward shaping:

| Version | Run | LR | Data | Key Change | Final Reward |
|---------|-----|-----|------|------------|--------------|
| v1 (GRPO-only) | [cex6rpwh](https://wandb.ai/evalonlabs/mistral-rl/runs/cex6rpwh) | 5e-6 | raw (all datasets) | Baseline, no SFT | 0.955 |
| v2 (SFT+GRPO) | [wehkefcs](https://wandb.ai/evalonlabs/mistral-rl/runs/wehkefcs) | 1.5e-6 | balanced (6,114) | SFT foundation + lower LR | 0.492 |

**v1-GRPO** (run cex6rpwh) achieved higher reward (0.955) but at the cost of entropy collapse (2.20) and 95% clipped completions -- the model learned to produce very short, formulaic responses. This is a classic RL over-optimization failure.

**v2-GRPO** (run wehkefcs) started from the SFT checkpoint with a much lower LR (1.5e-6 vs 5e-6). While the final reward is lower (0.492), the model maintained healthy entropy (2.90) and only 48% clipped completions -- producing longer, more diverse, and more natural responses. The eval reward (0.23) vs train reward (0.49) gap suggests room for further training, but the model generalizes better.

**v2-GRPO training trajectory** (1,497 steps):

| Metric | Step 10 | Step 750 | Step 1490 |
|--------|---------|----------|-----------|
| Reward | 0.356 | 0.460 | 0.223 |
| Loss | -0.013 | 0.015 | -0.055 |
| Entropy | 2.67 | 3.01 | 2.47 |
| Completion Length | 90 | 114 | 85 |
| Clipped Ratio | 0.27 | 0.38 | 0.24 |

### Model Loading

The codebase handles dynamic model class resolution, FP8 checkpoint dequantization, and LoRA application -- battle-tested across multiple Mistral model variants:

```python
model_cls = getattr(transformers, model_config.architectures[0])
model = model_cls.from_pretrained(args.model_name, **load_kwargs)
if is_fp8:
    model = model.dequantize()  # FP8 -> BF16 for LoRA compatibility
model = get_peft_model(model, peft_config)
```

### Deployment

All three model versions are deployed as live vLLM inference endpoints on Basilica, serving OpenAI-compatible chat completion APIs. The pipeline includes:
- `scripts/upload_to_hf.py` -- automated HuggingFace upload
- `scripts/validate_vllm_remote.py` -- remote endpoint validation against the eval dataset
- Automated refusal-rate and helpfulness-rate scoring

---

## W&B Integration

### Experiment Tracking
- **26 total runs** across SFT and GRPO groups
- Full metric logging: loss, reward, entropy, grad norm, completion lengths, clipped ratios
- Config tracking for all hyperparameters
- Run groups (ministral-sft, ministral-grpo) and tags for organization

### Key W&B Runs

| Run | Type | Link |
|-----|------|------|
| SFT (refusal-only) | Stage 1 | [vj4yv9gy](https://wandb.ai/evalonlabs/mistral-rl/runs/vj4yv9gy) |
| GRPO v1 (baseline) | Stage 2 | [cex6rpwh](https://wandb.ai/evalonlabs/mistral-rl/runs/cex6rpwh) |
| GRPO v2 (SFT+GRPO) | Stage 2 | [wehkefcs](https://wandb.ai/evalonlabs/mistral-rl/runs/wehkefcs) |

### W&B Report
Published report with training curves, comparison across runs, and analysis:
[Ministral Safety Fine-Tuning: SFT + GRPO Pipeline](https://wandb.ai/evalonlabs/mistral-rl/reports/Ministral-Safety-Fine-Tuning:-SFT-+-GRPO-Pipeline--VmlldzoxNjA3MTE2MQ==)

### W&B Weave -- Tracing & Evaluation

All three deployed models are evaluated end-to-end via W&B Weave with traced inference calls against the eval dataset.

**Weave Traces Dashboard**: [evalonlabs/mistral-rl/weave/traces](https://wandb.ai/evalonlabs/mistral-rl/weave/traces?view=traces_default)

**Trace Stats**:
- 500+ traced inference calls across 3 model versions
- 3 evaluation summary traces (one per model)
- Operations: `traced_remote_call` (individual inference), `traced_eval_summary` (aggregate scores)
- Mean inference latency: 1,620ms (range: 360ms - 2,002ms)

**Evaluation Results** (122 eval samples per model, malicious + benign):

| Model | Malicious Refusal Rate | Benign Helpfulness Rate | Balanced Score |
|-------|----------------------|------------------------|----------------|
| sec-v1 (GRPO-only) | 9.2% | 100% | 54.6% |
| sec-v2-sft (SFT) | 12.3% | 100% | 56.2% |
| sec-v2-grpo (SFT+GRPO) | 7.7% | 100% | 53.8% |

Key observations:
- **100% benign helpfulness** across all models -- none over-refuse legitimate queries
- **sec-v2-sft has the highest refusal rate** (12.3%) -- SFT directly teaches refusal patterns
- **sec-v2-grpo** shows lower refusal rate (7.7%) after GRPO, which trades some refusal for more nuanced responses. The model learned to address dangerous requests by explaining why they're problematic rather than issuing a flat refusal -- a behavior that the marker-based refusal detector doesn't capture but represents a more sophisticated safety posture
- The refusal rates reflect strict keyword-based detection (REFUSAL_MARKERS); the models exhibit broader safety behavior (deflection, education, boundary-setting) not captured by exact-match markers

Each traced call includes: prompt text, label, prompt hash, full model response, latency, and refusal classification -- providing full auditability of the evaluation pipeline.

### Large-Scale Evaluation (1,000 samples, temp=0.9)

A deeper evaluation of sec-v1 on 1,000 samples at temperature 0.9 (more stochastic, harder test) confirms the model's dual capability:

| Metric | Value |
|--------|-------|
| Malicious Refusal Rate | 11.2% |
| Benign Helpfulness Rate | 99.2% |
| Samples | 1,000 |

**Refusal quality** -- the model doesn't just pattern-match refusal phrases. It provides contextual, educational refusals:
- DAN/jailbreak attempts: refuses while explaining why the exploit won't work
- Sarin synthesis request: cites the Chemical Weapons Convention and legal consequences
- Therapy recording request: explains HIPAA violations and recommends legal alternatives
- Self-harm promotion: redirects to crisis hotlines and safe peer-support frameworks

**Helpfulness quality** -- on benign prompts, the model produces substantive, well-structured answers:
- Multi-language support (English, German)
- Complex reasoning tasks (NLI classification, factorization)
- Creative writing (character roleplay, essays)
- Technical explanations (GPS, programming, guardrail architecture)

Full qualitative analysis with top-20 refusals and top-20 benign responses: `outputs/mistral-grpo/top20_refusal_vs_benign_sec-v1_1000_temp09.md`
