# Training Report Review

Short answer: the run is technically working, but optimization quality is weak and expensive right now.

## What Looks Okay

- Setup is stable (model loads, LoRA trainable params ~1.72%, steps progressing).
- Reward briefly improves early (`-0.44 -> -0.04`), so learning signal exists.

## Main Problems In The Logs

1. Reward is still mostly negative, including eval (`eval_reward ~ -0.29` then `-0.27`), and unstable.
2. `completions/clipped_ratio` is very high (`~0.8–0.9`) which means generations hit max length constantly (wasted compute + noisy reward).
3. `max_prompt_length` is ignored (`[setup] ignoring unsupported GRPOConfig args: max_prompt_length`) so long prompts are likely unbounded.
4. `clip_ratio/*` is all zero every log, suggesting policy updates may be weak/uninformative.
5. Training is slow (`~10.5s/step`, eval ~14 min) because rollout settings are heavy.

## Highest-Impact Fixes (In Order)

1. Fix reward heuristic bug: remove `"i can"` from refusal markers (it penalizes benign helpful outputs).
2. Reduce rollout cost:
   - `num_generations: 8 -> 2`
   - `max_completion_length: 384/192 -> 96` (or `128`)
3. Lower LR for stability:
   - `5e-6 -> 1e-6 or 2e-6`
4. Enforce prompt truncation in preprocessing (since config arg is ignored) by truncating tokenized prompt before training.
5. Run shorter debug runs first (`max_steps 200-400`) and track:
   - malicious refusal rate
   - benign helpfulness rate
   - eval reward trend

## Suggested Next Step

Patch the training script with these changes first, then run a short controlled experiment to verify reward trend and latency before full training.
