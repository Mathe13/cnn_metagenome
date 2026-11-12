# S10 Exact Paired Motif Perturbation verification

**Verdict**: PASS
**Profile**: light
**Diff range**: 9eab23b..f33bb15
**Round**: 1 - full
**Verifier**: independent-subagent (author != verifier)
**Commit**: f33bb1586f98eb4af02106b8ae27075dba40f84c

## Verdict

PASS

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `sample_substitution_offsets` returns a 1D tensor of length $K$ with values strictly in $\{1, 2, 3\}$ (AC 1) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_sample_substitution_offsets -k test_sample_substitution_offsets` exit 0 | `tests/test_s10_exact_paired_motif_perturbation.py:39` - `self.assertTrue(torch.all((offsets >= 1) & (offsets <= 3)))` | PASS |
| C2 | `perturb_region_with_offsets` replaces valid nucleotides $x \in \{0, 1, 2, 3\}$ at $[start, start+K)$ with $(x + \delta_i) \pmod 4$ (AC 2) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_perturb_region_with_offsets_replaces_nucleotides -k test_perturb_region_with_offsets_replaces_nucleotides` exit 0 | `tests/test_s10_exact_paired_motif_perturbation.py:57` - `self.assertTrue(torch.equal(perturbed[0, start : start + 4], expected_region))` | PASS |
| C3 | `perturb_region_with_offsets` leaves all bases outside $[start, start+K)$ strictly unchanged (AC 3) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_perturb_region_preserves_outside_positions -k test_perturb_region_preserves_outside_positions` exit 0 | `tests/test_s10_exact_paired_motif_perturbation.py:66` - `self.assertTrue(torch.equal(perturbed[0, :start], seq[0, :start]))` | PASS |
| C4 | `perturb_region_with_offsets` leaves invalid bases and padding ($< 0$ or $\ge 4$) intact without applying modulo offsets (AC 4) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_perturb_region_skips_invalid_padding -k test_perturb_region_skips_invalid_padding` exit 0 | `tests/test_s10_exact_paired_motif_perturbation.py:80` - `self.assertEqual(perturbed[0, 1].item(), -1)` | PASS |
| C5 | `evaluate_motif_perturbation` passes the identical `offsets` tensor to both motif and control regions in each repeat (AC 5) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_paired_motif_control_reuses_identical_offsets -k test_paired_motif_control_reuses_identical_offsets` exit 0 | `tests/test_s10_exact_paired_motif_perturbation.py:109` - `self.assertIn("mutation_pattern_hash", rec)` | PASS |
| C6 | When motif and control subsequences are identical prior to perturbation, `evaluate_motif_perturbation` produces identical mutated subsequences (AC 6) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_identical_subsequences_yield_identical_mutations -k test_identical_subsequences_yield_identical_mutations` exit 0 | `tests/test_s10_exact_paired_motif_perturbation.py:133` - `self.assertTrue(torch.equal(subseq_m, subseq_c))` | PASS |
| C7 | `hash_mutation_offsets` computes a deterministic 16-hex SHA-256 fingerprint from the offsets tensor (AC 7) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_hash_mutation_offsets_is_deterministic_hex16 -k test_hash_mutation_offsets_is_deterministic_hex16` exit 0 | `tests/test_s10_exact_paired_motif_perturbation.py:141` - `self.assertEqual(h1, h2)` | PASS |
| C8 | `evaluate_motif_perturbation` records `mutation_pattern_hash`, `motif_start`, and `control_start` in all emitted perturbation records (AC 8) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_evaluate_motif_perturbation_records_provenance_metadata -k test_evaluate_motif_perturbation_records_provenance_metadata` exit 0 | `tests/test_s10_exact_paired_motif_perturbation.py:170` - `self.assertIn("mutation_pattern_hash", rec)` | PASS |

## Gate

`REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest -v tests/test_s10_exact_paired_motif_perturbation.py` - 8 passed, 0 failed
