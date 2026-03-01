# Interim Eval Report (Temperature 0.9)

This report is from completed runs with `samples=200` per model.
Full-dataset runs (`max-samples=6114`) are currently still in progress.

| Model | Samples | Malicious | Successful Refusals | Malicious Refusal Rate | Benign | Benign Helpful | Benign Helpfulness Rate | Balanced Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sec-v1 | 200 | 95 | 10 | 0.1053 | 105 | 104 | 0.9905 | 0.5479 |
| sec-v2-sft | 200 | 95 | 5 | 0.0526 | 105 | 104 | 0.9905 | 0.5216 |
| sec-v2-grpo | 200 | 95 | 12 | 0.1263 | 105 | 104 | 0.9905 | 0.5584 |