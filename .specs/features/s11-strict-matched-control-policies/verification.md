# S11 Strict Matched-Control Policies verification

**Verdict**: PASS
**Profile**: light
**Diff range**: e88d74c..0c08e43
**Round**: 1 - full
**Verifier**: independent-subagent (author != verifier)
**Commit**: 0c08e43715b08bd17ff15111450e51145b84e053

## Verdict

PASS

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `sample_matched_control_region` returns `MatchedControl` with `quality="exact"` and `distance=0` for exact GC match (AC 1) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_gc_exact_control_has_identical_gc_count -k test_gc_exact_control_has_identical_gc_count` exit 0 | `tests/test_s11_strict_matched_control_policies.py:47` - `self.assertEqual(ctrl.quality, "exact")` | PASS |
| C2 | `sample_matched_control_region` returns `MatchedControl` with `quality="relaxed_1"` and `distance=1` when only $\pm 1$ GC candidate exists (AC 2) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_gc_relaxed_control_is_labeled_relaxed -k test_gc_relaxed_control_is_labeled_relaxed` exit 0 | `tests/test_s11_strict_matched_control_policies.py:72` - `self.assertEqual(ctrl.quality, "relaxed_1")` | PASS |
| C3 | `sample_matched_control_region` returns `MatchedControl` with `start=None`, `quality="unavailable"`, and `distance=-1` when no candidate satisfies GC distance $\le 1$ (AC 3) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_gc_unavailable_returns_unavailable -k test_gc_unavailable_returns_unavailable` exit 0 | `tests/test_s11_strict_matched_control_policies.py:94` - `self.assertEqual(ctrl.quality, "unavailable")` | PASS |
| C4 | `sample_matched_control_region` returns `MatchedControl` with `quality="exact"` and `distance=0` for exact homopolymer run length in complexity matching (AC 4) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_complexity_exact_control_matches_run_length -k test_complexity_exact_control_matches_run_length` exit 0 | `tests/test_s11_strict_matched_control_policies.py:114` - `self.assertEqual(ctrl.quality, "exact")` | PASS |
| C5 | `sample_matched_control_region` does not label fallback candidates as `exact` (labels `relaxed_1` or `unmatched_fallback`) (AC 5) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_complexity_fallback_is_not_labeled_exact -k test_complexity_fallback_is_not_labeled_exact` exit 0 | `tests/test_s11_strict_matched_control_policies.py:136` - `self.assertNotEqual(ctrl.quality, "exact")` | PASS |
| C6 | `sample_matched_control_region` rejects any window overlapping any index in `motif_starts` within $[start, start+K)$ (AC 6) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_control_never_overlaps_target_motif -k test_control_never_overlaps_target_motif` exit 0 | `tests/test_s11_strict_matched_control_policies.py:159` - `self.assertFalse(overlap, f"Control at {ctrl.start} overlaps motif at {m}")` | PASS |
| C7 | `sample_matched_control_region` rejects any window containing invalid bases or padding ($< 0$ or $\ge 4$) (AC 7) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_ambiguous_bases_are_not_selected_as_control -k test_ambiguous_bases_are_not_selected_as_control` exit 0 | `tests/test_s11_strict_matched_control_policies.py:180` - `self.assertTrue(bool(((window >= 0) & (window < 4)).all().item()))` | PASS |
| C8 | `evaluate_motif_perturbation` records `control_match_quality`, `control_match_distance`, and `control_policy_requested` in every emitted record (AC 8) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_motif_control_artifact_records_match_quality -k test_motif_control_artifact_records_match_quality` exit 0 | `tests/test_s11_strict_matched_control_policies.py:204` - `self.assertEqual(df["control_match_quality"].iloc[0], "exact")` | PASS |
| C9 | `evaluate_motif_perturbation` with `strict_matching=True` skips occurrences with non-exact controls (AC 9) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_strict_matching_skips_non_exact_controls -k test_strict_matching_skips_non_exact_controls` exit 0 | `tests/test_s11_strict_matched_control_policies.py:241` - `self.assertEqual(len(df_strict), 0)` | PASS |
| C10 | `summarize_matched_control_coverage` computes `exact_match_rate` and quality breakdown counts (AC 10) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s11_strict_matched_control_policies.TestS11StrictMatchedControlPolicies.test_summarize_matched_control_coverage_metrics -k test_summarize_matched_control_coverage_metrics` exit 0 | `tests/test_s11_strict_matched_control_policies.py:258` - `self.assertAlmostEqual(row["exact_match_rate"], 0.5)` | PASS |

## Gate

`REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest -v tests/test_s11_strict_matched_control_policies.py` - 10 passed, 0 failed
