# S10 Exact Paired Motif Perturbation checks

Profile: light
Plan: `.specs/features/s10-exact-paired-motif-perturbation/plan.md`

## Intent

In `src/viral_cnn/interpretability/motifs.py:449-450`, motif and matched control regions independently advanced the RNG stream, causing different nucleotide substitutions to be applied to the motif and control in paired repeats. This feature ensures that both regions receive the exact same substitution offset vector $\delta_i \in \{1, 2, 3\}$, isolating sequence context from mutation randomness, and records mutation pattern provenance hashes.

8 checks in 1 slice · 3 one-way doors · 0 open, of which 0 block

## Checks

### S1 - Exact paired motif and control perturbation mechanics · 2 files · 63 KB · ~16k

**C1** - `sample_substitution_offsets` returns a 1D tensor of length $K$ with values strictly in $\{1, 2, 3\}$ (AC 1)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_sample_substitution_offsets -k test_sample_substitution_offsets`

**C2** - `perturb_region_with_offsets` replaces valid nucleotides $x \in \{0, 1, 2, 3\}$ at $[start, start+K)$ with $(x + \delta_i) \pmod 4$ (AC 2)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_perturb_region_with_offsets_replaces_nucleotides -k test_perturb_region_with_offsets_replaces_nucleotides`

**C3** - `perturb_region_with_offsets` leaves all bases outside $[start, start+K)$ strictly unchanged (AC 3)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_perturb_region_preserves_outside_positions -k test_perturb_region_preserves_outside_positions`

**C4** - `perturb_region_with_offsets` leaves invalid bases and padding ($< 0$ or $\ge 4$) intact without applying modulo offsets (AC 4)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_perturb_region_skips_invalid_padding -k test_perturb_region_skips_invalid_padding`

**C5** - `evaluate_motif_perturbation` passes the identical `offsets` tensor to both motif and control regions in each repeat (AC 5)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_paired_motif_control_reuses_identical_offsets -k test_paired_motif_control_reuses_identical_offsets`

**C6** - When motif and control subsequences are identical prior to perturbation, `evaluate_motif_perturbation` produces identical mutated subsequences (AC 6)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_identical_subsequences_yield_identical_mutations -k test_identical_subsequences_yield_identical_mutations`

**C7** - `hash_mutation_offsets` computes a deterministic 16-hex SHA-256 fingerprint from the offsets tensor (AC 7)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_hash_mutation_offsets_is_deterministic_hex16 -k test_hash_mutation_offsets_is_deterministic_hex16`

**C8** - `evaluate_motif_perturbation` records `mutation_pattern_hash`, `motif_start`, and `control_start` in all emitted perturbation records (AC 8)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s10_exact_paired_motif_perturbation.TestS10ExactPairedMotifPerturbation.test_evaluate_motif_perturbation_records_provenance_metadata -k test_evaluate_motif_perturbation_records_provenance_metadata`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| offsets values (3) | `1` C1 · `2` C1 · `3` C1 | - |
| perturbation boundaries (2) | `inside` C2 · `outside` C3 | - |
| nucleotide validity (2) | `valid_base` C2 · `invalid_or_padding` C4 | - |
| pairing targets (2) | `motif` C5 · `control` C5 | - |
| paired subsequence identity (2) | `identical_regions` C6 · `differing_regions` C5 | - |
| provenance fields (3) | `mutation_pattern_hash` C8 · `motif_start` C8 · `control_start` C8 | - |
| hash format (2) | `hex_length_16` C7 · `sha256_deterministic` C7 | - |

- No check claims more than the single case its proof exercises

## Swept

- validation : C1, C2, C4
- failure modes : C4
- idempotency : C7
- authorization : n/a - internal module logic without user authentication
- concurrency : n/a - stateless tensor operations without shared mutable state
- data lifecycle : n/a - in-memory tensors and immutable CSV exports
- dependency failure : n/a - standard PyTorch operations with no external network services
- state transitions : n/a - pure functional transformations without state machine
- observability : C8

## Handoff

Files touched:
- `src/viral_cnn/interpretability/motifs.py` (55 KB)
- `tests/test_s10_exact_paired_motif_perturbation.py` (8 KB)

Total estimate: ~63 KB · ~16k tokens.
Budget: 150k tokens. Estimate is well within budget: one builder.
