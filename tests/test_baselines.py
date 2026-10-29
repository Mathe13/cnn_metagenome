"""Unit and contract tests for S07: Kraken2 and BLASTN baselines under unified protocol."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from viral_cnn.baselines.taxonomy import parse_kraken_output, parse_blast_output
from viral_cnn.baselines.overlap import audit_exact_overlap, canonical_sequence_hash
from viral_cnn.baselines.paired import (
    wilson_score_interval,
    evaluate_predictions,
    paired_discordance,
    bootstrap_species_difference,
)
from viral_cnn.baselines.long_eval import (
    integrate_long_voter_predictions,
    evaluate_long_baselines,
)
from viral_cnn.baselines.kmer_baseline import KmerCentroidClassifier


class TestBaselineParsingAndTaxonomy(unittest.TestCase):
    """Checks C1-C6: Kraken2 and BLASTN raw output parsers and taxonomy conversion."""

    def test_parse_kraken_output_classified_species_mapping(self):
        """C1: Map 'C' status with taxid in [2, 13684] to zero-based class label taxid - 2."""
        lines = [
            "C\tq1\t2\t512\t0:478\n",      # taxid 2 -> label 0
            "C\tq2\t102\t512\t0:478\n",    # taxid 102 -> label 100
            "C\tq3\t13684\t512\t0:478\n",  # taxid 13684 -> label 13682
        ]
        res = parse_kraken_output(lines, valid_query_ids=["q1", "q2", "q3"])
        self.assertEqual(res["q1"], 0)
        self.assertEqual(res["q2"], 100)
        self.assertEqual(res["q3"], 13682)

    def test_parse_kraken_output_unclassified_and_root_abstention(self):
        """C2: Map unclassified ('U') and root (taxid 1) to label -1 (abstention)."""
        lines = [
            "U\tq1\t0\t512\t0:478\n",      # unclassified -> -1
            "C\tq2\t1\t512\t0:478\n",      # root taxid 1 -> -1
            "C\tq3\t99999\t512\t0:478\n",  # out-of-range -> -1
        ]
        res = parse_kraken_output(lines, valid_query_ids=["q1", "q2", "q3"])
        self.assertEqual(res["q1"], -1)
        self.assertEqual(res["q2"], -1)
        self.assertEqual(res["q3"], -1)

    def test_parse_kraken_output_rejects_duplicate_or_unknown_queries(self):
        """C3: Reject duplicate query IDs or queries absent from expected manifest."""
        duplicate_lines = [
            "C\tq1\t2\t512\t0:478\n",
            "C\tq1\t3\t512\t0:478\n",
        ]
        with self.assertRaises(ValueError):
            parse_kraken_output(duplicate_lines, valid_query_ids=["q1"])

        unknown_lines = [
            "C\tq1\t2\t512\t0:478\n",
            "C\tq_unknown\t3\t512\t0:478\n",
        ]
        with self.assertRaises(ValueError):
            parse_kraken_output(unknown_lines, valid_query_ids=["q1"])

    def test_parse_blast_output_bitscore_aggregation_and_top_candidates(self):
        """C4: Aggregate BLAST hits by species label with max bitscore and rank top candidates."""
        lines = [
            # q1 has two hits for species 10 (scores 500 and 600) and one for species 20 (score 700)
            "q1\tsub1_label_10\t500.0\t1e-50\t99.0\t512\n",
            "q1\tsub2_label_10\t600.0\t1e-60\t99.5\t512\n",
            "q1\tsub3_label_20\t700.0\t1e-70\t100.0\t512\n",
            "q1\tsub4_label_5\t400.0\t1e-40\t95.0\t512\n",
        ]
        res = parse_blast_output(lines, valid_query_ids=["q1"], max_targets=3)
        self.assertIn("q1", res)
        # Ordered by bitscore descending: species 20 (700), species 10 (600), species 5 (400)
        self.assertEqual(res["q1"], [20, 10, 5])

    def test_parse_blast_output_unaligned_abstention(self):
        """C5: Queries with no alignments in BLAST output map to empty candidate list."""
        lines = [
            "q1\tsub1_label_10\t500.0\t1e-50\t99.0\t512\n",
        ]
        res = parse_blast_output(lines, valid_query_ids=["q1", "q_no_hit"], max_targets=5)
        self.assertEqual(res["q1"], [10])
        self.assertEqual(res["q_no_hit"], [])

    def test_parse_blast_output_rejects_invalid_subject_format(self):
        """C6: Reject subject ID lacking '_label_{int}' suffix with ValueError."""
        lines = [
            "q1\tsub1_missing_label\t500.0\t1e-50\t99.0\t512\n",
        ]
        with self.assertRaises(ValueError):
            parse_blast_output(lines, valid_query_ids=["q1"])


class TestExactOverlapAndEvaluator(unittest.TestCase):
    """Checks C7-C12: Exact sequence overlap audit and unified metric evaluator."""

    def test_audit_exact_overlap_partitioning_strata(self):
        """C7: Partition queries into all, overlap, and no_exact_overlap strata via SHA256."""
        queries = [
            ("q1", "ACGT" * 128, 0),  # Matches training
            ("q2", "AAAA" * 128, 1),  # Not in training
            ("q3", "CCCC" * 128, 2),  # Matches training twice
        ]
        training = [
            "ACGT" * 128,
            "CCCC" * 128,
            "CCCC" * 128,
        ]
        audit_df, masks = audit_exact_overlap(queries, training)
        self.assertEqual(len(audit_df), 3)
        self.assertEqual(audit_df.loc[0, "train_count"], 1)
        self.assertEqual(audit_df.loc[1, "train_count"], 0)
        self.assertEqual(audit_df.loc[2, "train_count"], 2)
        self.assertTrue(masks["overlap"][0])
        self.assertFalse(masks["overlap"][1])
        self.assertTrue(masks["no_exact_overlap"][1])

    def test_audit_exact_overlap_handles_duplicate_sequence_columns(self):
        """C1: Extract scalar string sequences from the first matching column when DataFrame has duplicate 'sequence' columns."""
        seq_match = "ACGT" * 128
        seq_novel = "TTTT" * 128
        df_multi = pd.DataFrame({
            "sequence_padded": [seq_match, seq_novel],
            "sequence": ["RAW_UNPADDED_1", "RAW_UNPADDED_2"],
            "label": [0, 1],
            "query_id": ["q1", "q2"],
        })
        # Simulate .rename(columns={"sequence_padded": "sequence"}) producing duplicate 'sequence' columns
        df_dup = df_multi.rename(columns={"sequence_padded": "sequence"})
        self.assertEqual(list(df_dup.columns), ["sequence", "sequence", "label", "query_id"])

        audit_df, masks = audit_exact_overlap(df_dup, [seq_match])
        self.assertEqual(len(audit_df), 2)
        self.assertEqual(list(audit_df["query_id"]), ["q1", "q2"])
        self.assertEqual(int(audit_df.loc[0, "train_count"]), 1)
        self.assertEqual(int(audit_df.loc[1, "train_count"]), 0)
        self.assertEqual(list(masks["overlap"]), [True, False])
        self.assertEqual(list(masks["no_exact_overlap"]), [False, True])

    def test_audit_exact_overlap_resolves_canonical_sequence_column_names(self):
        """C2: Resolve sequence column in priority order ('sequence_padded', 'padded_Representation', 'sequence')."""
        seq_a = "ACGT" * 128
        seq_b = "GGGG" * 128
        for col_name in ("sequence_padded", "padded_Representation", "sequence"):
            df_col = pd.DataFrame({
                "query_id": ["q1", "q2"],
                col_name: [seq_a, seq_b],
                "label": [0, 1],
            })
            audit_df, masks = audit_exact_overlap(df_col, [seq_a])
            self.assertEqual(len(audit_df), 2)
            self.assertEqual(int(audit_df.loc[0, "train_count"]), 1)
            self.assertEqual(int(audit_df.loc[1, "train_count"]), 0)
            self.assertEqual(list(masks["overlap"]), [True, False])

        # Priority order check when both 'sequence_padded' and 'sequence' exist without renaming
        df_both = pd.DataFrame({
            "query_id": ["q1"],
            "sequence": [seq_b],
            "sequence_padded": [seq_a],
            "label": [0],
        })
        audit_both, masks_both = audit_exact_overlap(df_both, [seq_a])
        self.assertEqual(int(audit_both.loc[0, "train_count"]), 1)
        self.assertEqual(list(masks_both["overlap"]), [True])

    def test_notebook_09_unique_sequence_column_construction(self):
        """C3: Notebook 09 builds df_test_queries with unique ['query_id', 'sequence', 'label'] columns without duplicate rename."""
        import json

        nb_path = PROJECT_ROOT / "notebooks" / "reproduction" / "short" / "09_benchmark_cnns_kraken2_blast_overlap.ipynb"
        nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
        cell_sources = "\n".join("".join(c.get("source", [])) for c in nb_data["cells"] if c.get("cell_type") == "code")
        self.assertNotIn("rename(columns={col: \"sequence\"})", cell_sources)

        # Verify that none of the saved cell outputs contain an error traceback
        error_outputs = [
            out.get("ename")
            for c in nb_data["cells"]
            for out in c.get("outputs", [])
            if out.get("output_type") == "error"
        ]
        self.assertEqual(error_outputs, [])

        # Simulate multi-column Parquet sample returned by read_parquet_sample when both sequence_padded and sequence exist
        df_raw_test = pd.DataFrame({
            "sequence_padded": ["ACGT" * 128, "CCCC" * 128],
            "sequence": ["ACGT", "CCCC"],
            "label": [0, 1],
        })
        col = (
            "sequence_padded"
            if "sequence_padded" in df_raw_test.columns
            else ("padded_Representation" if "padded_Representation" in df_raw_test.columns else "sequence")
        )
        df_test_queries = pd.DataFrame({
            "query_id": [f"test_q_{i:05d}" for i in range(len(df_raw_test))],
            "sequence": df_raw_test[col].tolist(),
            "label": df_raw_test["label"].astype(int).tolist(),
        })
        self.assertEqual(list(df_test_queries.columns), ["query_id", "sequence", "label"])
        self.assertIsInstance(df_test_queries["sequence"], pd.Series)
        audit_df, masks = audit_exact_overlap(df_test_queries, ["ACGT" * 128])
        self.assertEqual(list(masks["overlap"]), [True, False])

    def test_evaluate_predictions_abstention_counted_in_denominator(self):
        """C8: Abstentions (pred == -1) count as errors in the total denominator N."""
        y_true = [0, 1, 2, 3]
        # q0 correct, q1 abstained (-1), q2 wrong (0), q3 correct (3)
        y_pred = [0, -1, 0, 3]
        res = evaluate_predictions(y_true, y_pred)
        # Accuracy = 2/4 = 0.50
        self.assertEqual(res["n"], 4)
        self.assertAlmostEqual(res["accuracy"], 0.50, places=4)

    def test_evaluate_predictions_wilson_confidence_intervals(self):
        """C9: Closed-form Wilson score confidence intervals for Top-1 accuracy."""
        low, high = wilson_score_interval(successes=80, n=100)
        # Expected around 0.71 to 0.86
        self.assertGreater(low, 0.70)
        self.assertLess(high, 0.88)
        self.assertLess(low, 0.80)
        self.assertGreater(high, 0.80)

    def test_evaluate_predictions_coverage_and_conditional_accuracy(self):
        """C10: Compute coverage fraction and conditional accuracy on covered queries."""
        y_true = [0, 1, 2, 3]
        y_pred = [0, -1, -1, 3]  # 2 covered, both correct
        res = evaluate_predictions(y_true, y_pred)
        self.assertAlmostEqual(res["coverage"], 0.50, places=4)
        self.assertAlmostEqual(res["conditional_accuracy"], 1.00, places=4)

    def test_evaluate_predictions_top5_accuracy(self):
        """C11: Compute Top-5 accuracy when candidate list is provided."""
        y_true = [10, 20, 30]
        y_pred = [10, 99, 99]
        top_cands = [
            [10, 11, 12],       # true 10 is at rank 1
            [21, 22, 20, 23],   # true 20 is at rank 3
            [31, 32, 33, 34],   # true 30 not in top candidates
        ]
        res = evaluate_predictions(y_true, y_pred, top_candidates=top_cands)
        # Top-5 hit count = 2 / 3 = 0.6667
        self.assertAlmostEqual(res["top5"], 2.0 / 3.0, places=4)

    def test_evaluate_predictions_macro_f1_with_zero_division(self):
        """C12: Compute Macro-F1 across true classes with zero division mapped to 0.0."""
        y_true = [0, 1, 2]
        y_pred = [0, 0, 0]  # only class 0 predicted
        res = evaluate_predictions(y_true, y_pred)
        self.assertGreaterEqual(res["macro_f1"], 0.0)
        self.assertLessEqual(res["macro_f1"], 1.0)


class TestPairedDiscordanceAndBootstrap(unittest.TestCase):
    """Checks C13-C15: Paired discordance and cluster bootstrap."""

    def test_paired_discordance_rejects_mismatched_query_sets(self):
        """C13: Reject comparisons between mismatched query counts with ValueError."""
        with self.assertRaises(ValueError):
            paired_discordance([0, 1], [0, 1, 2], [0, 1])

    def test_paired_discordance_contingency_counts_and_percentage(self):
        """C14: Compute 2x2 contingency table (both_correct, only_a, only_b, both_wrong)."""
        y_true = [0, 1, 2, 3]
        pred_a = [0, 1, 9, 9]  # A correct on 0, 1
        pred_b = [0, 9, 2, 9]  # B correct on 0, 2
        res = paired_discordance(y_true, pred_a, pred_b)
        self.assertEqual(res["both_correct"], 1)   # query 0
        self.assertEqual(res["only_a_correct"], 1) # query 1
        self.assertEqual(res["only_b_correct"], 1) # query 2
        self.assertEqual(res["both_wrong"], 1)     # query 3
        self.assertEqual(res["discordant_queries"], 2)
        self.assertAlmostEqual(res["discordant_pct"], 50.0, places=2)

    def test_bootstrap_species_difference_quantile_intervals(self):
        """C15: Cluster bootstrap resampling true species over 2,000 repeats with seed 42."""
        labels = [0, 0, 1, 1, 2, 2]
        diffs = [1.0, 1.0, 0.0, 0.0, -1.0, -1.0]  # Mean diff = 0.0
        ci_low, ci_high = bootstrap_species_difference(diffs, labels, repeats=2000, seed=42)
        # CI should bracket 0.0
        self.assertLessEqual(ci_low, 5.0)
        self.assertGreaterEqual(ci_high, -5.0)


class TestLongReadsBaselineProtocol(unittest.TestCase):
    """Checks C16-C19: Long-sequence baseline evaluation and voter integration."""

    def test_integrate_long_voter_predictions_schema_unification(self):
        """C16: Ingest LongSequenceVoter predictions into unified baseline mapping."""
        voter_df = pd.DataFrame({
            "query_id": ["read1", "read2"],
            "pred_mean_logits": [10, 20],
            "pred_majority_vote": [10, 25],
        })
        integrated = integrate_long_voter_predictions(voter_df)
        self.assertIn("Voter_mean_logits", integrated)
        self.assertIn("Voter_majority_vote", integrated)
        self.assertEqual(integrated["Voter_mean_logits"]["read1"], 10)
        self.assertEqual(integrated["Voter_majority_vote"]["read2"], 25)

    def test_evaluate_long_baselines_reference_genome_stratification(self):
        """C17: Stratify long reads across reference genome presence (seen_genome vs unseen)."""
        truth_df = pd.DataFrame({
            "query_id": ["r1", "r2", "r3", "r4"],
            "label": [0, 0, 1, 1],
            "seen_genome": [True, True, False, False],
            "length": [1000, 3000, 6000, 2500],
        })
        preds = {
            "Kraken2": {"r1": 0, "r2": 0, "r3": -1, "r4": 1},
            "BLASTN": {"r1": 0, "r2": 0, "r3": 1, "r4": 1},
        }
        metrics_df, _ = evaluate_long_baselines(truth_df, preds)
        strata = set(metrics_df["stratum"].unique())
        self.assertIn("all", strata)
        self.assertIn("seen_genome", strata)
        self.assertIn("unseen_genome", strata)

        # On seen_genome, Kraken2 has 100% accuracy
        k_seen = metrics_df[(metrics_df["method"] == "Kraken2") & (metrics_df["stratum"] == "seen_genome")]
        self.assertAlmostEqual(k_seen["accuracy"].values[0], 1.0, places=4)

    def test_evaluate_long_baselines_length_bin_stratification(self):
        """C18: Stratify long reads across length bins [512, 2000), [2000, 5000), >=5000 bp."""
        truth_df = pd.DataFrame({
            "query_id": ["r1", "r2", "r3"],
            "label": [0, 1, 2],
            "length": [1000, 3000, 7000],
        })
        preds = {"Voter": {"r1": 0, "r2": 1, "r3": 2}}
        metrics_df, _ = evaluate_long_baselines(truth_df, preds)
        strata = set(metrics_df["stratum"].unique())
        self.assertIn("len_512_to_2000", strata)
        self.assertIn("len_2000_to_5000", strata)
        self.assertIn("len_ge_5000", strata)

    def test_long_reads_paired_discordance_and_bootstrap_intervals(self):
        """C19: Paired comparison and species-cluster bootstrap on long reads."""
        truth_df = pd.DataFrame({
            "query_id": ["r1", "r2", "r3", "r4"],
            "label": [0, 0, 1, 1],
            "length": [1000, 2000, 3000, 4000],
        })
        preds = {
            "Voter": {"r1": 0, "r2": 0, "r3": 1, "r4": 99},
            "Kraken2": {"r1": 0, "r2": 0, "r3": -1, "r4": 1},
        }
        _, paired_df = evaluate_long_baselines(truth_df, preds)
        self.assertGreater(len(paired_df), 0)
        self.assertIn("ci95_low_pp", paired_df.columns)
        self.assertIn("ci95_high_pp", paired_df.columns)


class TestCLIRunnerAndKmerBaseline(unittest.TestCase):
    """Checks C20-C22: CLI runner, protocol documentation, and k-mer baseline."""

    def test_run_baseline_comparison_cli_short_and_long_regimes(self):
        """C20: CLI scripts/run_baseline_comparison.py executes and writes CSVs and manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            truth_csv = tmppath / "truth.csv"
            truth_df = pd.DataFrame({
                "query_id": ["q1", "q2", "q3"],
                "label": [0, 1, 2],
                "seen_genome": [True, False, True],
                "length": [600, 2500, 8000],
            })
            truth_df.to_csv(truth_csv, index=False)

            kraken_out = tmppath / "kraken.out"
            kraken_out.write_text("C\tq1\t2\t512\t0:478\nU\tq2\t0\t512\t0:478\nC\tq3\t4\t512\t0:478\n")

            out_dir = tmppath / "results"

            cmd = [
                ".venv/bin/python3",
                "scripts/run_baseline_comparison.py",
                "--regime", "short",
                "--truth", str(truth_csv),
                "--predictions", f"Kraken2={kraken_out}",
                "--output-dir", str(out_dir),
                "--bootstrap-repeats", "100",
                "--seed", "42",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI error: {res.stderr}")
            self.assertTrue((out_dir / "metrics_by_stratum.csv").exists())
            self.assertTrue((out_dir / "comparison_manifest.json").exists())

    def test_baseline_protocol_documentation_scientific_sections(self):
        """C21: Documentation docs/baseline_benchmark_protocol.md covers required scientific sections."""
        doc_path = Path("docs/baseline_benchmark_protocol.md")
        self.assertTrue(doc_path.exists())
        text = doc_path.read_text(encoding="utf-8")
        self.assertIn("95.89%", text)
        self.assertIn("Abstention and Denominator Rules", text)
        self.assertIn("Kraken2 Flat Taxonomy Mapping", text)
        self.assertIn("seen_genome", text)
        self.assertIn("Species-Cluster Bootstrap", text)

    def test_kmer_centroid_classifier_fit_predict_under_noise(self):
        """C22: KmerCentroidClassifier trains on k-mer counts and classifies sequences."""
        train_seqs = [
            "ACGTACGT" * 64,  # class 0
            "TGCATGCA" * 64,  # class 1
        ]
        train_labels = [0, 1]
        clf = KmerCentroidClassifier(k=4)
        clf.fit(train_seqs, train_labels)

        # Test on identical sequence
        preds = clf.predict(train_seqs)
        self.assertEqual(preds[0], 0)
        self.assertEqual(preds[1], 1)

        # Test on sequence with 5% synthetic substitutions
        noisy_seq = list("ACGTACGT" * 64)
        for i in range(0, len(noisy_seq), 20):
            noisy_seq[i] = "T"
        noisy_pred = clf.predict(["".join(noisy_seq)])
        self.assertEqual(noisy_pred[0], 0)


if __name__ == "__main__":
    unittest.main()
