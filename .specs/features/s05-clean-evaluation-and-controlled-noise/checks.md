# S05: Clean Evaluation and Controlled Noise Perturbation - Checks

Profile: light
Plan: `.specs/features/s05-clean-evaluation-and-controlled-noise/plan.md`

## Checks

### S1: Controlled Noise Injection Engine · 2 files · 18 KB · ~4k

**C1** - `docs/noise_evaluation_protocol.md` documents nominal rates 0.0, 0.02, 0.05, 0.10, 0.15, uniform 4-base resampling, and mathematical proof of expected effective rate $3\epsilon/4$ (AC 1)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_noise_protocol_documentation_and_proof`

**C2** - `viral_cnn.eval.noise.inject_noise` applies Bernoulli selection and uniform 4-base resampling across {0, 1, 2, 3} (AC 2)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_inject_noise_bernoulli_and_uniform_resampling`

**C3** - `viral_cnn.eval.noise.inject_noise` preserves padding and ambiguous tokens (<0) without applying noise perturbations (AC 3)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_inject_noise_preserves_padding_and_ambiguous_tokens`

**C4** - `viral_cnn.eval.noise.verify_effective_noise_rate` verifies that empirical substitution rate converges to $(3/4)\epsilon$ within $\pm 0.005$ tolerance (AC 4)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_empirical_noise_rate_converges_to_three_quarters_epsilon`

### S2: Comprehensive Metrics and Paired Comparisons · 2 files · 22 KB · ~6k

**C5** - `viral_cnn.eval.metrics.compute_classification_metrics` computes Top-1, Top-5, Macro-F1, and Weighted-F1 across declared classes (AC 5)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_classification_metrics_accuracy_and_f1`

**C6** - `viral_cnn.eval.metrics.compute_species_f1_bands` partitions per-species F1 into 4 continuous, gapless bands ([0.0, 0.50), [0.50, 0.80), [0.80, 0.95), [0.95, 1.00]) conserving total species count (AC 6)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_species_f1_bands_continuous_and_exhaustive`

**C7** - `viral_cnn.eval.metrics.compute_paired_comparison` calculates pairwise percentage-point difference $\Delta_{pp}$ and paired contingency counts (AC 7)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_paired_comparison_percentage_points_and_contingency`

**C8** - `docs/noise_evaluation_protocol.md` establishes that fragment-level bootstrap intervals measure query sampling and do not represent genome or training seed uncertainty (AC 8)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_noise_protocol_fragment_bootstrap_interpretation`

### S3: Evaluation Runner and CLI Interface · 3 files · 24 KB · ~6k

**C9** - CLI `scripts/run_evaluation.py` executes evaluation run, emits query prediction records, and outputs JSON metrics summary with exit code 0 (AC 9)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_evaluation_cli_exit_zero_and_prediction_manifest`

**C10** - CLI `scripts/run_evaluation.py` exits with code 2 on unsupported noise rates, missing files, or invalid model variants (AC 10)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_evaluation_cli_invalid_arguments_exit_code_2`

**C11** - Evaluation routines execute deterministically on synthetic fixtures without GPU hardware or external weight downloads (AC 11)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluation_pipeline_deterministic_synthetic_execution`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Nominal noise rates evaluated (5) | `noise_00` C2 · `noise_02` C4 · `noise_05` C4 · `noise_10` C4 · `noise_15` C4 | - |
| F1 performance band intervals (4) | `band_0_to_50` C6 · `band_50_to_80` C6 · `band_80_to_95` C6 · `band_95_to_100` C6 | - |
| Paired comparison contingency outcomes (4) | `both_correct` C7 · `only_a_correct` C7 · `only_b_correct` C7 · `both_wrong` C7 | - |
| Evaluation metric types calculated (4) | `top1_accuracy` C5 · `top5_accuracy` C5 · `macro_f1` C5 · `weighted_f1` C5 | - |
| CLI execution status codes (2) | `0` C9 · `2` C10 | - |

- Claims naming a status code or response shape: C9, C10 - each has a proof crossing the process boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C10
- failure modes: C10
- idempotency: C2
- authorization: n/a - local scientific CLI tool with no network or authorization boundary
- concurrency: n/a - deterministic batch processing on single machine with no concurrent state mutation
- data lifecycle: C1
- dependency failure: C11
- state transitions: n/a - stateless data transformation and metric computation
- observability: C9

## Handoff

- S1 = 4k in docs and noise engine; S2 = 6k in metrics engine and tests; S3 = 6k in evaluation runner, CLI, and fixtures. Total = ~16k tokens, well under the 150k budget - one builder

- **Mechanism:** one builder
