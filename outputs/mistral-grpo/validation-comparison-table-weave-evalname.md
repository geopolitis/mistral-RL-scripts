# Weave Eval Comparison (One Eval Per LLM)

| Alias | Weave Eval Name | Weave Run ID | Samples | Malicious Refusal Rate | Benign Helpfulness Rate | Balanced Score | Output |
|---|---|---|---:|---:|---:|---:|---|
| sec-v1 | `sec-v1_eval_2026-03-01` | `86386c36-c725-4ba6-9c86-b4a7a6308676` | 122 | 0.0923 (6/65) | 1.0000 (57/57) | 0.5462 | `outputs/mistral-grpo/validation-sec-v1-weave-evalname.json` |
| sec-v2-sft | `sec-v2-sft_eval_2026-03-01` | `475c6abb-fcdb-4283-8182-93f268ff4aa2` | 122 | 0.1231 (8/65) | 1.0000 (57/57) | 0.5615 | `outputs/mistral-grpo/validation-sec-v2-sft-weave-evalname.json` |
| sec-v2-grpo | `sec-v2-grpo_eval_2026-03-01` | `aee849c3-be55-4c9a-a9ff-a21e648f0fe8` | 122 | 0.0769 (5/65) | 1.0000 (57/57) | 0.5385 | `outputs/mistral-grpo/validation-sec-v2-grpo-weave-evalname.json` |

Weave project URL: https://wandb.ai/evalonlabs/mistral-rl/weave
