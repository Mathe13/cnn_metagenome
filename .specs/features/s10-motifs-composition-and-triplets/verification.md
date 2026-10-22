# S10: Motifs, Compositional Controls, and Triplet Analysis - Verification

**Verdict**: PASS
**Profile**: light
**Diff range**: 08afec0..c84cb22
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `viral_cnn.interpretability.motifs.extract_first_layer_motifs` extracts consensus 7-mers using `consensus_rule="max_positive"` over first-layer convolution weights (AC 1, AC 2) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:86` - `self.assertIn("CCCCCCC", motifs); self.assertEqual(motifs["CCCCCCC"], 1)` | PASS |
| C2 | `viral_cnn.interpretability.motifs.extract_first_layer_motifs` extracts consensus 7-mers using `consensus_rule="max_abs"` over first-layer convolution weights (AC 3) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:108` - `self.assertIn("TGGGGGG", motifs_abs); self.assertIn("AGGGGGG", motifs_pos)` | PASS |
| C3 | `viral_cnn.interpretability.motifs.scan_motif_occurrences` constructs 1D spatial convolution kernels and returns exact 0-based match coordinates across 4-channel sequences (AC 4) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:121` - `self.assertEqual(matches["CCCCCCC"][0], [4, 15])` | PASS |
| C4 | `viral_cnn.interpretability.motifs.scan_motif_occurrences` excludes positions with padding or ambiguous bases encoded as null vectors or negative indices (AC 5) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:129` - `self.assertEqual(len(matches["CCCCCCC"][0]), 0)` | PASS |
| C5 | `viral_cnn.interpretability.motifs.perturb_sequence_mask` substitutes target motif positions with different nucleotides using random offsets $\in \{1, 2, 3\}$ (AC 6) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:141` - `self.assertNotEqual(seq[i].item(), perturbed[i].item()); self.assertTrue(0 <= perturbed[i].item() < 4)` | PASS |
| C6 | `viral_cnn.interpretability.motifs.sample_matched_control_region` with `control_policy="uniform_random"` samples non-overlapping 7-bp valid windows from the same sequence (AC 8) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:165` - `self.assertTrue(start + 7 <= 10 or start >= 17)` | PASS |
| C7 | `viral_cnn.interpretability.motifs.evaluate_motif_perturbation` computes paired probability drops ($\Delta p_{\text{motif}}$, $\Delta p_{\text{random}}$, $\Delta p_{\text{specific}}$) across $N_{\text{repeats}} = 5$ repetitions with fixed seed 42 (AC 7, AC 9) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:193` - `self.assertEqual(records["n_repeats"].iloc[0], 5); self.assertIn("specific_probability_drop", records.columns)` | PASS |
| C8 | `viral_cnn.interpretability.motifs.summarize_motif_control` performs paired one-sided Wilcoxon signed-rank tests and Benjamini-Hochberg FDR adjustment at $\alpha = 0.05$ on motifs with $N_{\text{sequences}} \ge 5$ (AC 10, AC 11, AC 12) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:215` - `self.assertTrue(summary["significant_fdr"].iloc[0]); self.assertLess(summary["p_fdr"].iloc[0], 0.05)` | PASS |
| C9 | `viral_cnn.interpretability.motifs.sample_matched_control_region` with `control_policy="gc_matched"` samples non-overlapping control windows with identical GC count ($\Delta\text{GC}=0$, fallback $\le 1$) (AC 13, AC 14) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:241` - `self.assertEqual(start, 10)` | PASS |
| C10 | `viral_cnn.interpretability.motifs.sample_matched_control_region` with `control_policy="complexity_matched"` samples non-overlapping control windows matching maximum homopolymer run length (AC 15) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:262` - `self.assertEqual(start, 10)` | PASS |
| C11 | `viral_cnn.interpretability.motifs.evaluate_motif_perturbation` under `control_policy="gc_matched"` computes GC-controlled specific drops ($\Delta p_{\text{specific\_GC}}$) and Wilcoxon FDR statistics (AC 16) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:283` - `self.assertIn("specific_probability_drop", records.columns); self.assertFalse(records.empty)` | PASS |
| C12 | `viral_cnn.interpretability.motifs.audit_motif_composition_robustness` compares historical uniform drops vs GC-matched drops and flags compositionally confounded homopolymers (AC 17, AC 18) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:307` - `self.assertTrue(audit["is_confounded"].iloc[0]); self.assertEqual(audit["verdict"].iloc[0], "compositionally_confounded")` | PASS |
| C13 | `viral_cnn.interpretability.motifs.compute_jaccard_cooccurrence` constructs exact symmetric pairwise Jaccard co-occurrence matrices across test sequence presence (AC 19) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:326` - `np.testing.assert_allclose(jaccard.values, jaccard.values.T); self.assertAlmostEqual(jaccard.loc["MOTIF_A", "MOTIF_B"], 1.0 / 3.0, places=4)` | PASS |
| C14 | `viral_cnn.interpretability.motifs.analyze_cooccurrence_pairs` audits orientation dependency between canonical $(u, v)$ and reversed $(v, u)$ top-200 edges, computing absolute difference $\Delta r$ (AC 20) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:343` - `self.assertGreaterEqual(res["delta_r_orientation"], 0.0); self.assertIn("r_canonical", res)` | PASS |
| C15 | `viral_cnn.interpretability.motifs.analyze_cooccurrence_pairs` computes symmetric intraclass correlation (ICC) and duplicate bidirectional pair correlation over all $2K$ ordered pairs $[(x_i, y_i), (y_i, x_i)]$ (AC 21) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:357` - `self.assertTrue(-1.0 <= res["icc_symmetric"] <= 1.0); self.assertTrue(-1.0 <= res["r_bidirectional"] <= 1.0)` | PASS |
| C16 | `viral_cnn.interpretability.motifs.analyze_cooccurrence_pairs` evaluates statistical significance via a node-permutation null model (1,000 permutations) and computes empirical $p_{\text{perm}}$ (AC 22, AC 23) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:371` - `self.assertTrue(0.0 <= res["p_perm"] <= 1.0); self.assertIn("p_perm", res)` | PASS |
| C17 | `viral_cnn.interpretability.motifs.compute_exploratory_triplet_rscu` computes triplet RSCU across `frame0`, `forward_3frames`, and `six_frames` modes (AC 24, AC 25) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:386` - `self.assertEqual(len(rscu_f0["rscu"]), 64); self.assertAlmostEqual(rscu_f0["rscu"]["ATG"], 1.0)` | PASS |
| C18 | `viral_cnn.interpretability.motifs.compute_exploratory_triplet_rscu` computes 0th-order multinomial base-composition null expectations $P(b_1 b_2 b_3) = P(b_1)P(b_2)P(b_3)$ (AC 26) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:398` - `self.assertAlmostEqual(null_probs["AAA"], 0.125, places=3); self.assertAlmostEqual(null_probs["TTT"], 0.0, places=3)` | PASS |
| C19 | `viral_cnn.interpretability.motifs.audit_translation_boundary_signals` tests ATG start and TAA/TAG/TGA stop frequencies in motifs against mononucleotide composition expectations (AC 27) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:409` - `self.assertIn("start_atg", report); self.assertIn("observed_count", report["start_atg"])` | PASS |
| C20 | `viral_cnn.interpretability.motifs.compute_exploratory_triplet_rscu` designates all metric outputs as `ExploratoryTripletComposition` and issues warnings when reading frame is unverified (AC 28) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:422` - `self.assertEqual(res["metric_label"], "ExploratoryTripletComposition"); self.assertFalse(res["verified_biological_frame"])` | PASS |
| C21 | CLI `scripts/run_motif_analysis.py` executes motif analysis pipeline and outputs `motif_summary.parquet`, `cooccurrence_synergy.json`, and `run_manifest.json` with exit code 0 (AC 29) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:436` - `self.assertEqual(exit_code, 0); self.assertTrue(os.path.exists(os.path.join(out_dir, "motif_summary.parquet")))` | PASS |
| C22 | CLI `scripts/run_motif_analysis.py` exits with code 2 on invalid arguments, missing input files, or unsupported control policies (AC 29) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:446` - `self.assertEqual(exit_code >> 8, 2)` | PASS |
| C23 | `viral_cnn.interpretability.motifs.generate_motif_audit_figures` renders multi-panel figures attributing each panel to verified model checkpoints (AC 30) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:464` - `self.assertTrue(os.path.exists(out_path)); self.assertGreater(os.path.getsize(out_path), 1000)` | PASS |
| C24 | `docs/motifs_and_triplet_composition_audit.md` documents comparison of historical tables/figures vs GC/symmetry controls and text revision recommendations (AC 31, AC 32) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:474` - `self.assertIn("GCMatchedControl", content); self.assertIn("ExploratoryTripletComposition", content)` | PASS |
| C25 | Motif analysis pipeline executes deterministically on synthetic fixtures without GPU hardware or external checkpoint downloads (AC 1, AC 6, AC 13) | `.venv/bin/python3 -m unittest -v tests/test_motifs.py` exit 0 | `tests/test_motifs.py:491` - `self.assertEqual(s1, s2); self.assertEqual(len(motifs), 2)` | PASS |

## Coverage

| Set (size) | Recomputed from | Member -> proof | Unproven |
| --- | --- | --- | --- |
| Consensus extraction rules evaluated (2) | `src/viral_cnn/interpretability/motifs.py` (`extract_first_layer_motifs`) | `max_positive` C1 · `max_abs` C2 | - |
| Paired control sampling policies audited (3) | `src/viral_cnn/interpretability/motifs.py` (`sample_matched_control_region`) | `uniform_random` C6 · `gc_matched` C9 · `complexity_matched` C10 | - |
| Triplet frame analysis modes audited (3) | `src/viral_cnn/interpretability/motifs.py` (`compute_exploratory_triplet_rscu`) | `frame0` C17 · `forward_3frames` C17 · `six_frames` C17 | - |
| Co-occurrence correlation metrics evaluated (3) | `src/viral_cnn/interpretability/motifs.py` (`analyze_cooccurrence_pairs`) | `canonical_upper_triangle` C14 · `reversed_order` C14 · `symmetric_intraclass` C15 | - |
| Translation boundary signal codons evaluated (4) | `src/viral_cnn/interpretability/motifs.py` (`audit_translation_boundary_signals`) | `start_atg` C19 · `stop_taa` C19 · `stop_tag` C19 · `stop_tga` C19 | - |
| CLI execution status codes (2) | `scripts/run_motif_analysis.py` | `0` C21 · `2` C22 | - |

## Swept dimensions

- validation: C22 (CLI `scripts/run_motif_analysis.py` exits with code 2 on invalid arguments, missing input files, or unsupported control policies)
- failure modes: C4 (`scan_motif_occurrences` excludes sequences with padding or ambiguous indices, yielding no false positive matches)
- idempotency: C25 (pipeline and control region sampling run deterministically with identical random seed across repeated runs)
- authorization: n/a - local research scripts with no network authentication or multi-tenant boundary
- concurrency: n/a - sequential batch evaluation with thread-safe localized RNG and no shared concurrent write states
- data lifecycle: n/a - read-only evaluation over immutable checkpoints and sequence datasets without database lifecycle or TTL
- dependency failure: C4 (robust handling of missing channels, padded sequences, and null vectors)
- state transitions: C12 (`audit_motif_composition_robustness` classifies motifs losing significance under GC controls as `compositionally_confounded`)
- observability: C21 (CLI logs execution progress and writes parquet summaries, JSON metrics, and run manifest)

## Gate

`.venv/bin/python3 -m unittest -v tests/test_motifs.py` - 25 passed, 0 failed
