# S11 Strict Matched-Control Policies checks

Profile: light
Plan: `.specs/features/s11-strict-matched-control-policies/plan.md`

## Intent

In `src/viral_cnn/interpretability/motifs.py:335-370`, matched-control selection silently fell back to relaxed GC ($\pm 1$ base) or arbitrary valid candidates without recording match quality, obscuring compositional control relaxation. This feature introduces the `MatchedControl` dataclass, records formal match quality (`exact`, `relaxed_1`, `unmatched_fallback`, `unavailable`), supports strict-mode matching (`quality == 'exact'`), and computes matching coverage metrics.

10 checks in 1 slice · 4 one-way doors · 0 open, of which 0 block

## Checks

### S1 - Strict matched-control policy and quality tracking · 2 files · 70 KB · ~18k

**C1** - `sample_matched_control_region` returns `MatchedControl` with `quality="exact"` and `distance=0` for exact GC match (AC 1)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_gc_exact_control_has_identical_gc_count -k test_gc_exact_control_has_identical_gc_count`

**C2** - `sample_matched_control_region` returns `MatchedControl` with `quality="relaxed_1"` and `distance=1` when only $\pm 1$ GC candidate exists (AC 2)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_gc_relaxed_control_is_labeled_relaxed -k test_gc_relaxed_control_is_labeled_relaxed`

**C3** - `sample_matched_control_region` returns `MatchedControl` with `start=None`, `quality="unavailable"`, and `distance=-1` when no candidate satisfies GC distance $\le 1$ (AC 3)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_gc_unavailable_returns_unavailable -k test_gc_unavailable_returns_unavailable`

**C4** - `sample_matched_control_region` returns `MatchedControl` with `quality="exact"` and `distance=0` for exact homopolymer run length in complexity matching (AC 4)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_complexity_exact_control_matches_run_length -k test_complexity_exact_control_matches_run_length`

**C5** - `sample_matched_control_region` does not label fallback candidates as `exact` (labels `relaxed_1` or `unmatched_fallback`) (AC 5)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_complexity_fallback_is_not_labeled_exact -k test_complexity_fallback_is_not_labeled_exact`

**C6** - `sample_matched_control_region` rejects any window overlapping any index in `motif_starts` within $[start, start+K)$ (AC 6)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_control_never_overlaps_target_motif -k test_control_never_overlaps_target_motif`

**C7** - `sample_matched_control_region` rejects any window containing invalid bases or padding ($< 0$ or $\ge 4$) (AC 7)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_ambiguous_bases_are_not_selected_as_control -k test_ambiguous_bases_are_not_selected_as_control`

**C8** - `evaluate_motif_perturbation` records `control_match_quality`, `control_match_distance`, and `control_policy_requested` in every emitted record (AC 8)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_motif_control_artifact_records_match_quality -k test_motif_control_artifact_records_match_quality`

**C9** - `evaluate_motif_perturbation` with `strict_matching=True` skips occurrences with non-exact controls (AC 9)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_strict_matching_skips_non_exact_controls -k test_strict_matching_skips_non_exact_controls`

**C10** - `summarize_matched_control_coverage` computes `exact_match_rate` and quality breakdown counts (AC 10)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_summarize_matched_control_coverage_metrics -k test_summarize_matched_control_coverage_metrics`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| gc match quality (3) | `exact` C1 · `relaxed_1` C2 · `unavailable` C3 | - |
| complexity match quality (3) | `exact` C4 · `relaxed_1` C5 · `unmatched_fallback` C5 | - |
| candidate filters (2) | `overlap_rejection` C6 · `invalid_base_rejection` C7 | - |
| provenance fields (3) | `control_match_quality` C8 · `control_match_distance` C8 · `control_policy_requested` C8 | - |
| strict mode enforcement (2) | `exact_evaluated` C8 · `non_exact_skipped` C9 | - |
| coverage summary metrics (2) | `exact_match_rate` C10 · `quality_breakdown_counts` C10 | - |

- No check claims more than the single case its proof exercises

## Swept

- validation : C1, C2, C4, C7
- failure modes : C3
- idempotency : C10
- authorization : n/a - internal module logic without user authentication
- concurrency : n/a - stateless tensor operations without shared mutable state
- data lifecycle : n/a - in-memory tensors and immutable CSV exports
- dependency failure : n/a - standard PyTorch operations with no external network services
- state transitions : n/a - pure functional transformations without state machine
- observability : C8, C10

## Handoff

Files touched:
- `src/viral_cnn/interpretability/motifs.py` (60 KB)
- `tests/test_s11_strict_matched_control_policies.py` (10 KB)

Total estimate: ~70 KB · ~18k tokens.
Budget: 150k tokens. Estimate is well within budget: one builder.
