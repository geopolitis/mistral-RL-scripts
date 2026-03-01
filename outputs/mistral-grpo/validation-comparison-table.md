# Remote Validation Comparison

Settings: `datasets/unique_prompts_balanced.json`, `eval_split=0.02`, `max_samples=200`, endpoint `/v1/chat/completions`.

| Alias | Deployment URL | Served Model ID | Samples | Malicious Refusal Rate | Benign Helpfulness Rate | Balanced Score |
|---|---|---|---:|---:|---:|---:|
| sec-v1 | https://3c12c510-8a0d-451e-a668-41d3b4ff9da1.deployments.basilica.ai | `ministral-3b-sec-v1` | 122 | 0.0923 (6/65) | 1.0000 (57/57) | 0.5462 |
| sec-v2-sft | https://1d05d1e5-3626-4910-87c8-eab438e3dc78.deployments.basilica.ai | `ministral-3b-sec-v2-sft` | 122 | 0.0769 (5/65) | 1.0000 (57/57) | 0.5385 |
| sec-v2-grpo | https://f4e85e87-cc18-4e7e-8ec3-b87b20a040dd.deployments.basilica.ai | `ministral-3b-sec-v2-grpo` | 122 | 0.0769 (5/65) | 1.0000 (57/57) | 0.5385 |

## Output Files
- `outputs/mistral-grpo/validation-sec-v1.json`
- `outputs/mistral-grpo/validation-sec-v2-sft.json`
- `outputs/mistral-grpo/validation-sec-v2-grpo.json`
