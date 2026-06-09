# Target-Aware Beta Thompson Sampling

This branch supports an optional target-aware Thompson sampler for competence-difficulty matching. It maintains a Beta posterior for each training example's binary final-answer success probability and prefers examples whose sampled probability is close to a target such as `0.5`.

Enable it by adding these arguments to the existing Android GUI GRPO command:

```bash
data.sampler_type=beta_thompson \
data.beta_ts_target_success=0.5 \
data.beta_ts_temperature=0.15 \
data.beta_ts_prior_alpha=1.0 \
data.beta_ts_prior_beta=1.0 \
data.beta_ts_uniform_mix=0.1 \
algorithm.online_filtering=false
```

The posterior update compares the generated `<answer>...</answer>` value with `ground_truth`. It deliberately ignores dense `overall` reward and LLM-judge ranking, so the Beta posterior remains interpretable as task success probability.

`beta_ts_temperature` controls how strongly sampling concentrates around the target. `beta_ts_uniform_mix` preserves exploration and prevents early posterior noise from permanently starving tasks. When enabled, the training dataloader uses `num_workers=0` so each posterior update can affect the next batch instead of being hidden behind prefetched batches.

The implementation records these training metrics:

- `beta_ts/batch_success_rate`
- `beta_ts/selected_posterior_mean`
- `beta_ts/posterior_mean`
- `beta_ts/posterior_min`
- `beta_ts/posterior_max`

For the first controlled comparison, keep all other settings fixed and compare `data.sampler_type=random` against `data.sampler_type=beta_thompson`. Use at least four rollouts per prompt when possible; `n=3` only observes success rates in `{0, 1/3, 2/3, 1}`.

The sampler posterior is indexed by the filtered dataset index. The `IndexedDataset` wrapper also exposes a stable `task_id` field when the source dataset contains one, but changing the dataset ordering or filtering between resumed runs invalidates the index-to-task mapping.
