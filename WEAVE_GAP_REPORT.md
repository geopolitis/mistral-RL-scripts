# W&B + Weave Gap Report

## Executive Summary
Current state: training and logging are integrated with Weights & Biases (W&B), but Weave tracing/evaluation is missing.

Impact on rubric/scoring:
- `E2E Models + Weave`: **Missing**
- `Weave Tracing + Eval`: **Missing**
- Net effect: **up to 2 stars left on the table** due to no Weave integration.

## Evidence (Current Repo)

### W&B is implemented
- SFT launcher includes W&B preflight/login and run flags:
  - `scripts/run_sft_single_h200.sh`
- GRPO launcher includes W&B preflight/login and run flags:
  - `scripts/run_grpo_single_h200.sh`
- Training scripts initialize W&B runs and custom step logging callbacks:
  - `scripts/train_sft_mistral.py`
  - `scripts/train_grpo_mistral.py`

### Weave is missing
- No Weave imports/usages found in training/validation code.
- No traced generation calls, no Weave dataset/eval objects, no Weave leaderboard/report outputs.

## Latest Model Comparison (Remote Validation)
Source: `outputs/mistral-grpo/validation-*.json`

| Model Alias | Samples | Malicious Refusal Rate | Benign Helpfulness Rate | Balanced Score |
|---|---:|---:|---:|---:|
| sec-v1 | 122 | 0.0923 (6/65) | 1.0000 (57/57) | 0.5462 |
| sec-v2-sft | 122 | 0.0769 (5/65) | 1.0000 (57/57) | 0.5385 |
| sec-v2-grpo | 122 | 0.0769 (5/65) | 1.0000 (57/57) | 0.5385 |

Observation: model-level comparison exists, but there is no Weave trace-level evidence or standardized online eval object for this benchmark.

## Gap Breakdown

1. Missing Weave tracing for inference/evaluation
- No per-prompt trace spans for request, response, latency, and score components.
- No linked traces to model version/deployment metadata.

2. Missing Weave eval object/workflow
- No canonical eval dataset artifact in Weave.
- No automated judge/scorer pipeline in Weave.
- No run-to-run comparison in a Weave-native dashboard.

3. Missing E2E reproducible chain in Weave
- Training artifacts are logged in W&B, but evaluation provenance is not end-to-end connected through Weave traces.

## Recommended Implementation Plan

### Phase 1: Add Weave tracing (high priority)
- Add `weave` dependency.
- Instrument remote validation path in `scripts/validate_vllm_remote.py`:
  - trace each request with prompt hash, label, model, endpoint, latency, retry count
  - trace response text length, refusal classification, and final label outcome
- Attach run metadata:
  - git commit SHA
  - dataset path + split
  - model alias/deployment URL

### Phase 2: Add Weave eval object (high priority)
- Create a Weave eval script (new file): `scripts/eval_weave_remote.py`
- Register dataset rows and scoring functions:
  - malicious refusal rate
  - benign helpfulness rate
  - balanced score
- Emit a single eval run artifact per model alias.

### Phase 3: Add CI/automation (medium priority)
- Add a simple shell wrapper to execute all deployment evaluations and publish Weave runs.
- Add a markdown export step for leaderboard snapshots.

## Definition of Done
- Weave traces visible for each evaluated sample (request + response + scoring fields).
- Weave eval runs available for `sec-v1`, `sec-v2-sft`, and `sec-v2-grpo` on the same split.
- A comparison table generated from Weave outputs and committed to repo.
- Rubric items `E2E Models + Weave` and `Weave Tracing + Eval` no longer marked missing.

## Immediate Next Step
Implement Phase 1 directly in `scripts/validate_vllm_remote.py`, then create `scripts/eval_weave_remote.py` for Phase 2 so future comparisons are automatically traceable and scoreable in Weave.
