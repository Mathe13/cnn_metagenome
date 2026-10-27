# n05-short-motifs-and-triplets-notebooks verification

**Verdict**: PASS
**Profile**: light
**Diff range**: d012ac9..HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `07_first_layer_motifs_and_perturbation.ipynb` contains valid JSON, zero imports of `short/utils/*`, and imports exclusively from `viral_cnn.*` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c1_notebook_07_imports_and_decoupling tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:28` - `self.assertNotIn("short.utils", code)` | PASS |
| C2 | `07_first_layer_motifs_and_perturbation.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c2_notebook_07_execution_smoke_test tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:50` - `self.assertLess(duration, 60.0)` | PASS |
| C3 | Figure 3 panel (e) links to checkpoint `convnext_noise_species` distinct from panel (b) `resnet_noise_species` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c3_notebook_07_panel_e_checkpoint_attribution tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:65` - `self.assertNotEqual(ckpt_map["convnext_noise_species"], ckpt_map["resnet_noise_species"])` | PASS |
| C4 | Paired motif perturbation evaluates both uniform random control and GC-matched control (`abs(ΔGC) = 0`) on same sequences without overlap | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c4_notebook_07_paired_gc_controls tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:77` - `self.assertIn("mean_specific_drop_uniform", df.columns)` | PASS |
| C5 | Motif perturbation summary reports paired Wilcoxon signed-rank test statistics and Benjamini-Hochberg FDR $q$-values for both control policies | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c5_notebook_07_wilcoxon_and_fdr tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:88` - `self.assertIn("p_fdr_gc", df.columns)` | PASS |
| C6 | Symmetric Jaccard co-occurrence matrices exported to Parquet under `results_reproduction/motifs/` and Table 8 / Fig 3 artifacts generated | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c6_notebook_07_artifacts_and_jaccard_export tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:108` - `np.testing.assert_allclose(mat, mat.T, atol=1e-6)` | PASS |
| C7 | Historical notebook `short/full_loss_function.ipynb` remains strictly unmodified (`git status --porcelain short/full_loss_function.ipynb` empty) | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c7_historical_full_loss_notebook_preserved tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:118` - `self.assertEqual(res.stdout.strip(), "")` | PASS |
| C8 | `08_triplet_composition_and_cooccurrence.ipynb` contains valid JSON, zero imports of `short/utils/*`, and imports exclusively from `viral_cnn.*` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c8_notebook_08_imports_and_decoupling tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:134` - `self.assertNotIn("short.utils", code)` | PASS |
| C9 | `08_triplet_composition_and_cooccurrence.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c9_notebook_08_execution_smoke_test tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:156` - `self.assertLess(duration, 60.0)` | PASS |
| C10 | Triplet composition and exploratory RSCU evaluated across frame 0, 3-frames, and 6-frames with Benjamini-Hochberg FDR correction and unannotated metagenomic warning | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c10_notebook_08_rscu_and_fdr tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:169` - `self.assertIn("rscu_frame0", df.columns)` | PASS |
| C11 | Translation boundary signals (`ATG` and `TAA`/`TAG`/`TGA`) audited across learned motifs against genomic background frequencies | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c11_notebook_08_translation_boundary_signals tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:186` - `self.assertIn("start_atg", signals)` | PASS |
| C12 | Top-200 co-occurrence pairs consumed from notebook 07 Jaccard artifact, tested for orientation symmetry $(A, B)$ vs $(B, A)$ and evaluated against permutation null model | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c12_notebook_08_cooccurrence_symmetry_and_null_model tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:201` - `self.assertIn("delta_r_orientation", df.columns)` | PASS |
| C13 | Inter-architecture co-occurrence correlations report Pearson $r$ and Spearman $\rho$ with two-tailed $p$-values and FDR correction | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c13_notebook_08_inter_architecture_correlations tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:212` - `self.assertGreater(r, 0.9)` | PASS |
| C14 | Historical notebook `short/codon_usage.ipynb` remains strictly unmodified (`git status --porcelain short/codon_usage.ipynb` empty) | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c14_historical_codon_usage_notebook_preserved tests/test_n05_motifs_and_triplets.py` exit 0 | `tests/test_n05_motifs_and_triplets.py:226` - `self.assertEqual(res.stdout.strip(), "")` | PASS |

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Reproduction notebooks (2) | `07_first_layer_motifs_and_perturbation.ipynb` C1, C2 · `08_triplet_composition_and_cooccurrence.ipynb` C8, C9 | - |
| Control policies audited (2) | `uniform_random` C4 · `gc_matched` C4, C5 | - |
| Figure 3 panel variants (5) | `resnet50_species` (a) C3, C6 · `resnet_noise_species` (b) C3, C6 · `convnext_zero_species` (c) C6 · `convnext_transfer_species` (d) C6 · `convnext_noise_species` (e) C3, C6 | - |
| Reading frame modes evaluated (3) | `frame0` C10 · `forward_3frames` C10 · `six_frames` C10 | - |
| Translation boundary codons audited (2) | `start_codons (ATG)` C11 · `stop_codons (TAA, TAG, TGA)` C11 | - |
| Co-occurrence pair evaluations (3) | `top_200_edges` C12 · `orientation_symmetry` C12 · `permutation_null_model` C12, C13 | - |
| Historical notebooks preserved (2) | `short/full_loss_function.ipynb` C7 · `short/codon_usage.ipynb` C14 | - |

## Gate

`MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest tests/test_n05_motifs_and_triplets.py` - 14 passed, 0 failed
