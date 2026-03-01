# Weave Eval Comparison (temperature=0.90, sample=1000)

Settings:
- `data_dir=datasets/unique_prompts_balanced.json`
- `eval_split=0.2`
- `max_samples=1000`
- `temperature=0.9`
- Weave project: `evalonlabs/mistral-rl`

| Alias | Weave Eval Name | Run ID | Samples | Malicious Refusal Rate | Benign Helpfulness Rate | Balanced Score |
|---|---|---|---:|---:|---:|---:|
| sec-v1 | `sec-v1_eval_1000_temp09_2026-03-01` | `2572ca1e-c18a-4581-8c72-242cbbc70283` | 1000 | 0.1120 (58/518) | 0.9917 (478/482) | 0.5518 |
| sec-v2-sft | `sec-v2-sft_eval_1000_temp09_2026-03-01` | `4550b2f1-8e79-46d7-bebe-085793c71b76` | 1000 | 0.0792 (41/518) | 0.9896 (477/482) | 0.5344 |
| sec-v2-grpo | `sec-v2-grpo_eval_1000_temp09_2026-03-01` | `b0faa2b7-3b1a-44e3-a57f-4cdfc0e55110` | 1000 | 0.0965 (50/518) | 0.9938 (479/482) | 0.5452 |

Output files:
- `outputs/mistral-grpo/validation-sec-v1-weave-1000-temp09.json`
- `outputs/mistral-grpo/validation-sec-v2-sft-weave-1000-temp09.json`
- `outputs/mistral-grpo/validation-sec-v2-grpo-weave-1000-temp09.json`
