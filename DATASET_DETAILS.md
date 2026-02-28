# Dataset Details

This page summarizes the current contents of `datasets/*.json`.

## Scope

- Files scanned: `19`
- Unique prompts (`text`): `8,344`

## Labels

Unique labels:

- `benign`
- `malicious`

Label counts:

- `malicious`: `8,480` (68.35%)
- `benign`: `3,927` (31.65%)

## Categories

- Unique categories: `140`
- Taxonomy note: categories mix naming styles (for example `prompt_injection` and `Physical Harm`), so normalization may be useful before analytics.

Top 10 categories by row count:

1. `benign` - 3,244
2. `prompt_injection` - 2,109
3. `jailbreak` - 666
4. `prompt_hijacking` - 573
5. `Others` - 510
6. `prompt_extraction` - 427
7. `Data Security Harm` - 374
8. `Physical Data` - 374
9. `Physical Harm` - 340
10. `over_defense` - 339

## Prompt Length Profile

- Word count: min `1`, median `20`, mean `62.82`, max `1,973`
- Character count: min `7`, median `128`, mean `363.81`, max `11,977`

This indicates a long-tail distribution with some very large prompts that can increase RL rollout time.

## Data Quality Notes

- Prompts with conflicting labels (same `text` mapped to both benign and malicious): `1`
- Example conflicting prompt starts with:
  - `I want you to act as a storyteller...`

## Per-file Row Counts

| File | Rows |
|---|---:|
| `datasets/ailuminate_demo.json` | 1,200 |
| `datasets/asb_attacks.json` | 400 |
| `datasets/benign_samples.json` | 110 |
| `datasets/bipia_indirect.json` | 400 |
| `datasets/cyberseceval2_pi.json` | 251 |
| `datasets/deepset_all.json` | 662 |
| `datasets/deepset_v2.json` | 355 |
| `datasets/encoding_evasion.json` | 24 |
| `datasets/harmbench_behaviors.json` | 400 |
| `datasets/hpi_attack_approx.json` | 55 |
| `datasets/injecagent_attacks.json` | 2,108 |
| `datasets/injection_samples.json` | 110 |
| `datasets/ivanleomk_all.json` | 917 |
| `datasets/ivanleomk_v2.json` | 610 |
| `datasets/jackhhao_jailbreak.json` | 1,306 |
| `datasets/notinject_samples.json` | 339 |
| `datasets/safeguard_test.json` | 2,060 |
| `datasets/tensor_trust_attacks.json` | 1,000 |
| `datasets/transfer_attack_samples.json` | 100 |

## Why This Matters for Training

- Class imbalance (`~68/32`) can bias policy behavior.
- Duplicate prompts can overweight repeated patterns.
- Very long prompts/completions increase rollout latency and total training time.

