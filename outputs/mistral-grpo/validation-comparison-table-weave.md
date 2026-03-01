# Remote Validation Comparison (Weave Traced)

Settings: `datasets/unique_prompts_balanced.json`, `eval_split=0.02`, `max_samples=200`, endpoint `/v1/chat/completions`, Weave project `evalonlabs/mistral-rl`.

| Alias | Samples | Malicious Refusal Rate | Benign Helpfulness Rate | Balanced Score | Output |
|---|---:|---:|---:|---:|---|
| sec-v1 | 122 | 0.0923 (6/65) | 1.0000 (57/57) | 0.5462 | `outputs/mistral-grpo/validation-sec-v1-weave.json` |
| sec-v2-sft | 122 | 0.1077 (7/65) | 1.0000 (57/57) | 0.5538 | `outputs/mistral-grpo/validation-sec-v2-sft-weave.json` |
| sec-v2-grpo | 122 | 0.0769 (5/65) | 1.0000 (57/57) | 0.5385 | `outputs/mistral-grpo/validation-sec-v2-grpo-weave.json` |
