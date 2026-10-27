# n07-long-sequence-voter-notebooks verification

**Verdict**: PASS
**Profile**: light
**Diff range**: cd20e1f..HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `01_ncbi_download_and_prepare_database.ipynb` contains valid JSON, zero legacy imports, and imports exclusively from `viral_cnn.*` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c1_notebook_01_imports_and_decoupling tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:28` - `self.assertNotIn("short.utils", code)` | PASS |
| C2 | `01_ncbi_download_and_prepare_database.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions, exporting `long_database_preparation_audit.csv` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c2_notebook_01_execution_smoke_test tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:50` - `self.assertLess(duration, 60.0)` | PASS |
| C3 | Sequence preparation audit quantifies training reference overlap via `quantify_training_overlap` and records giant vs small virus read disparity (500 vs 50 reads) | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c3_overlap_quantification_and_sampling_bias tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:66` - `self.assertIn(500, reads_counts)` | PASS |
| C4 | Historical notebooks `long/download_from_ncbi.ipynb` and `long/prepare database.ipynb` remain strictly unmodified | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c4_historical_download_and_prep_notebooks_preserved tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:83` - `self.assertEqual(res.stdout.strip(), "")` | PASS |
| C5 | `02_cnn_long_voter_inference.ipynb` contains valid JSON, zero legacy imports, and imports exclusively from `viral_cnn.*` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c5_notebook_02_imports_and_decoupling tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:99` - `self.assertNotIn("short.utils", code)` | PASS |
| C6 | `02_cnn_long_voter_inference.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions, verifying window logit equivalence via `verify_window_logit_equivalence` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c6_notebook_02_execution_smoke_test tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:121` - `self.assertLess(duration, 60.0)` | PASS |
| C7 | Sliding-window consensus evaluates pre-softmax mean logits and majority voting across clean and noise levels $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$, comparing `handle_short='skip'` vs `handle_short='pad'` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c7_sliding_window_consensus_and_short_handling tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:138` - `self.assertFalse(short_row["skip_mode_handled"])` | PASS |
| C8 | Historical notebook `long/cnn-long-voter.ipynb` remains strictly unmodified (`git status --porcelain long/cnn-long-voter.ipynb` empty) | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c8_historical_voter_notebook_preserved tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:153` - `self.assertEqual(res.stdout.strip(), "")` | PASS |
| C9 | `03_cnn_long_voter_compare.ipynb` contains valid JSON, zero legacy imports, and imports exclusively from `viral_cnn.*` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c9_notebook_03_imports_and_decoupling tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:169` - `self.assertNotIn("short.utils", code)` | PASS |
| C10 | `03_cnn_long_voter_compare.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions, reproducing Table 12 and exporting `table_12_long_voter_comparison.csv` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c10_notebook_03_execution_smoke_test tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:191` - `self.assertLess(duration, 60.0)` | PASS |
| C11 | Error stratification partitions performance across contig length bins, ambiguity rates, and reference presence (`seen_genome` vs `unseen_genome`), diagnosing ConvNeXt collapse | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c11_error_stratification_and_collapse_diagnosis tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:210` - `self.assertAlmostEqual(resnet_acc, 0.7269, places=3)` | PASS |
| C12 | Historical notebook `long/cnn-long-voter-compare.ipynb` remains strictly unmodified (`git status --porcelain long/cnn-long-voter-compare.ipynb` empty) | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c12_historical_compare_notebook_preserved tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:228` - `self.assertEqual(res.stdout.strip(), "")` | PASS |
| C13 | `04_long_motifs_and_gc_analysis.ipynb` contains valid JSON, zero legacy imports, and imports exclusively from `viral_cnn.*` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c13_notebook_04_imports_and_decoupling tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:244` - `self.assertNotIn("short.utils", code)` | PASS |
| C14 | `04_long_motifs_and_gc_analysis.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions, generating `long_contig_motif_summary.csv` and `figure_long_motifs_gc.png` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c14_notebook_04_execution_smoke_test tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:270` - `self.assertTrue(summary_csv.exists())` | PASS |
| C15 | Historical notebook `long/long_loss_function.ipynb` remains strictly unmodified (`git status --porcelain long/long_loss_function.ipynb` empty) | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c15_historical_long_loss_notebook_preserved tests/test_n07_long_sequence_voter.py` exit 0 | `tests/test_n07_long_sequence_voter.py:281` - `self.assertEqual(res.stdout.strip(), "")` | PASS |

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Reproduction notebooks in long (4) | `01_ncbi_download_and_prepare_database.ipynb` C1, C2 · `02_cnn_long_voter_inference.ipynb` C5, C6 · `03_cnn_long_voter_compare.ipynb` C9, C10 · `04_long_motifs_and_gc_analysis.ipynb` C13, C14 | - |
| Voter consensus aggregation modes (2) | `mean_logits` C7, C10 · `majority_vote` C7, C10 | - |
| Short contig handling policies (2) | `handle_short='skip'` C7 · `handle_short='pad'` C7 | - |
| Error stratification dimensions (3) | `length_bins` C11 · `ambiguity_bins` C11 · `reference_genome_presence (seen vs unseen)` C11 | - |
| Historical notebooks in long preserved (5) | `long/download_from_ncbi.ipynb` C4 · `long/prepare database.ipynb` C4 · `long/cnn-long-voter.ipynb` C8 · `long/cnn-long-voter-compare.ipynb` C12 · `long/long_loss_function.ipynb` C15 | - |

## Gate

`MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py` - 15 passed, 0 failed
