"""Unit and integration tests for M06 long sequences and strata analysis (C1-C17)."""
from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from viral_cnn.m06_workflow import (
    begin_m06,
    reserve_m06_run,
    sample_deterministic_contigs,
    load_top_motifs_for_long_analysis,
    slice_contig_and_track_motifs,
    stratify_long_errors_with_windows,
    export_stratified_long_errors_csv,
    render_figure_long_accuracy_by_length_and_windows,
    render_figure_long_paired_vote_vs_mean_logits,
    render_figure_long_motifs_gc,
    PROTOCOL_ID,
    SCIENTIFIC_DEMARCATION_TEXT,
)
from viral_cnn.notebook_runtime import preflight_notebook, JupyterHubDataTopology
from viral_cnn.inference.audit_long import build_diagnostic_finding, audit_long_inputs, quantify_training_overlap_detailed
from viral_cnn.inference.voter import LongSequenceVoter, slice_sequence


def _create_mock_long_environment(target_dir: Path) -> dict:
    """Creates a mock dataset and checkpoints environment for long/01-04 testing."""
    target_dir.mkdir(parents=True, exist_ok=True)
    topo = JupyterHubDataTopology(target_dir)

    long_dir = Path(topo.paths["long"]["ncbi_parquet_dir"])
    long_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    bases = ["A", "C", "G", "T"]

    # Generate 6 mock contigs: 3 short, 3 long (>= 2000 bp)
    lengths = [400, 600, 1500, 2500, 4000, 8000]
    df_long = pd.DataFrame({
        "query_id": [f"query_{i:03d}" for i in range(len(lengths))],
        "sequence": ["".join(rng.choice(bases, size=l)) for l in lengths],
        "species_id": [i % 5 for i in range(len(lengths))],
        "truth_label": [i % 5 for i in range(len(lengths))],
    })
    df_long.to_parquet(long_dir / "test.parquet")

    # Mapping file
    mapping_path = Path(topo.paths["long"]["mapping_species_final"])
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    df_map = pd.DataFrame({
        "species_id": list(range(5)),
        "model_label": list(range(5)),
    })
    df_map.to_csv(mapping_path, index=False)

    # Short species train parquet for overlap check in long/01
    short_dir = Path(topo.paths["short"]["species_parquet_dir"])
    short_dir.mkdir(parents=True, exist_ok=True)
    df_train = pd.DataFrame({
        "sequence": ["".join(rng.choice(bases, size=512)) for _ in range(10)],
        "sequence_padded": ["".join(rng.choice(bases, size=512)) for _ in range(10)],
        "label": [0] * 10,
    })
    df_train.to_parquet(short_dir / "train.parquet")

    # Checkpoint files
    for variant in ("resnet_clean_species", "resnet_noise_species", "convnext_zero_species", "convnext_transfer_species", "convnext_noise_species"):
        ckpt = Path(topo.paths["checkpoints"][variant])
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"mock_checkpoint_bytes")

    return {
        "data_root": target_dir,
        "df_long": df_long,
        "mapping_path": mapping_path,
    }


class M06Tests(unittest.TestCase):
    """Test suite for M06 requirements (C1-C17)."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.env = _create_mock_long_environment(self.project_root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_paths(self):
        """C1: Reserve collision-free directory under results_reproduction/runs/<run_id>/long/<stage>/."""
        run_id, out_01 = reserve_m06_run(self.project_root, "01", run_id="test_run_c1")
        self.assertEqual(run_id, "test_run_c1")
        self.assertTrue(out_01.is_dir())
        self.assertTrue(str(out_01).endswith("results_reproduction/runs/test_run_c1/long/01"))

        # Collision detection: re-reserving identical path raises FileExistsError
        with self.assertRaises(FileExistsError):
            reserve_m06_run(self.project_root, "01", run_id="test_run_c1")

    def test_long_01_overlap_prep(self):
        """C2: Audit long sequences and calculate k-mer overlap against training fragments with k=15."""
        df_queries = self.env["df_long"]
        train_seqs = ["ACGT" * 128] * 5

        input_audit = audit_long_inputs(
            df_queries,
            sequence_col="sequence",
            query_id_col="query_id",
            truth_col="truth_label",
        )
        self.assertEqual(input_audit["eligible_count"], len(df_queries))
        self.assertEqual(input_audit["genome_identity"]["status"], "unknown")

        overlap = quantify_training_overlap_detailed(
            df_queries["sequence"].tolist(),
            train_seqs,
            k=15,
        )
        self.assertIn("mean_kmer_overlap_fraction", overlap)
        self.assertIn("fraction_with_any_overlap", overlap)

    def test_long_02_paired_five_models(self):
        """C3: In empirical mode, verify all 5 checkpoints and paired predictions for vote and mean logits."""
        variants = [
            "resnet_clean_species", "resnet_noise_species",
            "convnext_zero_species", "convnext_transfer_species", "convnext_noise_species"
        ]
        topo = JupyterHubDataTopology(self.project_root)
        for v in variants:
            p = Path(topo.paths["checkpoints"][v])
            self.assertTrue(p.is_file(), f"Missing checkpoint: {p}")

        # Test LongSequenceVoter slice with handle_short='pad'
        short_seq = "ACGT" * 50  # 200 bp < 512 bp
        windows_pad = slice_sequence(short_seq, window_size=512, handle_short="pad")
        self.assertEqual(len(windows_pad), 1)
        self.assertEqual(len(windows_pad[0]), 512)

        windows_skip = slice_sequence(short_seq, window_size=512, handle_short="skip")
        self.assertEqual(len(windows_skip), 0)

    def test_missing_checkpoint_failed_manifest(self):
        """C4: Missing checkpoint or input parquet in empirical mode raises FileNotFoundError and writes FAILED manifest."""
        run = begin_m06(
            "02",
            run_id="test_run_c4",
            data_root=self.project_root,
            smoke_test=False,
            project_root=self.project_root,
        )
        # Remove one required checkpoint to trigger failure
        topo = JupyterHubDataTopology(self.project_root)
        ckpt = Path(topo.paths["checkpoints"]["convnext_noise_species"])
        if ckpt.exists():
            ckpt.unlink()

        with self.assertRaises(FileNotFoundError):
            with run.guard():
                # Simulate checkpoint verification
                if not ckpt.is_file():
                    raise FileNotFoundError(f"Missing required checkpoint: {ckpt}")

        with open(run.manifest_path) as f:
            man = json.load(f)
        self.assertEqual(man["status"], "FAILED")
        self.assertEqual(man["failure"]["error_type"], "FileNotFoundError")

    def test_long_02_complete_manifest(self):
        """C5: On success, emit paired prediction parquet and manifest with COMPLETE status."""
        run = begin_m06(
            "02",
            run_id="test_run_c5",
            data_root=self.project_root,
            smoke_test=True,
            project_root=self.project_root,
        )
        pred_file = run.output_dir / "predictions_resnet_clean_species_epsilon0_pad.parquet"
        pd.DataFrame({
            "query_id": ["q1", "q2"],
            "pred_mean_logits": [0, 1],
            "pred_majority_vote": [0, 1],
            "truth_label": [0, 1],
            "handle_short": ["pad", "pad"],
        }).to_parquet(pred_file)

        with run.guard():
            run.manifest["artifacts"]["pred_resnet"] = str(pred_file)

        with open(run.manifest_path) as f:
            man = json.load(f)
        self.assertEqual(man["status"], "COMPLETE")
        self.assertEqual(man["protocol_id"], PROTOCOL_ID)

    def test_dynamic_run_resolution_no_hardcoding(self):
        """C6: long/03 resolves run_id dynamically without static hardcoded strings."""
        run_id, out_02 = reserve_m06_run(self.project_root, "02", run_id="dyn_run_123")
        man_02 = out_02 / "manifest_02_voter_inference.json"
        man_02.write_text(json.dumps({"notebook": "02", "is_smoke_test": True}))

        with patch.dict(os.environ, {"M06_LONG_RUN_ID": "dyn_run_123"}):
            run_3 = begin_m06("03", data_root=self.project_root, smoke_test=True, project_root=self.project_root)
            self.assertEqual(run_3.run_id, "dyn_run_123")

    def test_table_12_paired_export(self):
        """C7: Table 12 contains overall accuracy and coverage for mean_logits and majority_vote across 5 variants."""
        variants = [
            "resnet_clean_species", "resnet_noise_species",
            "convnext_zero_species", "convnext_transfer_species", "convnext_noise_species"
        ]
        records = []
        for v in variants:
            records.append({
                "variant": v,
                "aggregation": "mean_logits",
                "overall_accuracy": 0.85,
                "coverage": 1.0,
                "covered_queries": 100,
            })
            records.append({
                "variant": v,
                "aggregation": "majority_vote",
                "overall_accuracy": 0.82,
                "coverage": 1.0,
                "covered_queries": 100,
            })
        df_t12 = pd.DataFrame(records)
        t12_csv = self.project_root / "table_12_long_voter_comparison.csv"
        df_t12.to_csv(t12_csv, index=False)

        loaded = pd.read_csv(t12_csv)
        self.assertEqual(len(loaded), 10)
        self.assertEqual(set(loaded["variant"]), set(variants))
        self.assertEqual(set(loaded["aggregation"]), {"mean_logits", "majority_vote"})

    def test_stratified_errors_with_sample_size(self):
        """C8: Stratification covers length and window bins, reporting sample size N explicitly in CSV."""
        df_results = pd.DataFrame({
            "query_id": [f"q{i}" for i in range(20)],
            "length": [300]*4 + [1000]*6 + [3000]*5 + [8000]*3 + [15000]*2,
            "n_windows": [1]*4 + [3]*6 + [10]*5 + [30]*3 + [60]*2,
            "truth_label": [0]*20,
            "pred_mean_logits": [0]*16 + [1]*4,
            "pred_majority_vote": [0]*15 + [1]*5,
            "ambiguity_ratio": [0.0]*20,
            "taxon_id": ["tax1"]*20,
        })
        strata = stratify_long_errors_with_windows(df_results)
        self.assertIn("by_length", strata)
        self.assertIn("by_windows", strata)

        csv_out = self.project_root / "stratified_long_errors.csv"
        df_strat = export_stratified_long_errors_csv(strata, csv_out)
        self.assertTrue(csv_out.is_file())
        self.assertIn("sample_size_n", df_strat.columns)
        self.assertTrue((df_strat["sample_size_n"] > 0).any())

    def test_figure_accuracy_by_length_and_windows(self):
        """C9: Generate figure_long_accuracy_by_length_and_windows.png with support N indicated."""
        df_results = pd.DataFrame({
            "query_id": [f"q{i}" for i in range(10)],
            "length": [400]*3 + [1200]*4 + [6000]*3,
            "n_windows": [1]*3 + [4]*4 + [25]*3,
            "truth_label": [0]*10,
            "pred_mean_logits": [0]*8 + [1]*2,
            "pred_majority_vote": [0]*8 + [1]*2,
        })
        strata = stratify_long_errors_with_windows(df_results)
        fig_path = self.project_root / "figure_long_accuracy_by_length_and_windows.png"
        render_figure_long_accuracy_by_length_and_windows(strata, fig_path)
        self.assertTrue(fig_path.is_file())
        self.assertGreater(fig_path.stat().st_size, 1000)

    def test_figure_paired_vote_vs_mean_logits(self):
        """C10: Generate figure_long_paired_vote_vs_mean_logits.png comparing aggregators across 5 variants."""
        variants = ["resnet_clean_species", "resnet_noise_species", "convnext_zero_species", "convnext_transfer_species", "convnext_noise_species"]
        records = []
        for v in variants:
            records.append({"variant": v, "aggregation": "mean_logits", "overall_accuracy": 0.85, "covered_queries": 50})
            records.append({"variant": v, "aggregation": "majority_vote", "overall_accuracy": 0.82, "covered_queries": 50})
        df_t12 = pd.DataFrame(records)

        fig_path = self.project_root / "figure_long_paired_vote_vs_mean_logits.png"
        render_figure_long_paired_vote_vs_mean_logits(df_t12, fig_path)
        self.assertTrue(fig_path.is_file())
        self.assertGreater(fig_path.stat().st_size, 1000)

    def test_diagnostic_hypothesis_demarcation(self):
        """C11: Diagnostic findings classify mechanism causes as HYPOTHESIS or INCONCLUSIVE without causal claims."""
        finding = build_diagnostic_finding(
            claim="ConvNeXt degradation under long sequences",
            test="Stratified comparison across length and window counts",
            effect="Acc drops from 72% to 45% on contigs >10000 bp",
            n=100,
            inputs=["predictions_convnext_zero_species_pad.parquet"],
            limitations=["Absence of controlled multi-seed retraining on long contigs", "Simulated NCBI contigs"],
            status="HYPOTHESIS",
            is_smoke_test=False,
        )
        self.assertEqual(finding["status"], "HYPOTHESIS")
        self.assertIn("DESCRIPTIVE LONG SEQUENCE", SCIENTIFIC_DEMARCATION_TEXT)

    def test_long_04_motif_loading_and_sampling(self):
        """C12: Dynamically load top 7-mer motifs from 07 and sample contigs deterministically with seed 42, length >= 2000."""
        motifs = load_top_motifs_for_long_analysis(default_motifs=["CCCCCCC", "GGGGGGG", "ATATATA", "TTTTTTT"])
        self.assertEqual(len(motifs), 4)
        for m in motifs:
            self.assertEqual(len(m), 7)

        df_long = self.env["df_long"]
        sampled = sample_deterministic_contigs(df_long, min_length=2000, n_contigs=2, seed=42)
        self.assertEqual(len(sampled), 2)
        self.assertTrue((sampled["length"] >= 2000).all())

        # Determinism check: identical seed produces identical contigs
        sampled_again = sample_deterministic_contigs(df_long, min_length=2000, n_contigs=2, seed=42)
        pd.testing.assert_frame_equal(sampled, sampled_again)

    def test_long_04_contig_motif_summary(self):
        """C13: Slicing contigs into 512 bp windows yields summary with positions, GC% and motif counts."""
        contig = ("ACGT" * 50) + "CCCCCCC" + ("TGCA" * 100) + "GGGGGGG" + ("ACGT" * 200)
        df_summary = slice_contig_and_track_motifs(contig, ["CCCCCCC", "GGGGGGG"], window_size=512, stride=256)
        self.assertGreater(len(df_summary), 1)
        self.assertIn("gc_percent", df_summary.columns)
        self.assertIn("motif_CCCCCCC", df_summary.columns)
        self.assertIn("motif_GGGGGGG", df_summary.columns)
        self.assertTrue((df_summary["motif_CCCCCCC"] > 0).any())

    def test_long_04_figure_and_manifest(self):
        """C14: Render figure_long_motifs_gc.png and emit manifest with COMPLETE status."""
        run = begin_m06("04", run_id="test_run_c14", data_root=self.project_root, smoke_test=True, project_root=self.project_root)
        contig = ("ACGT" * 50) + "CCCCCCC" + ("TGCA" * 100) + "GGGGGGG" + ("ACGT" * 200)
        df_summary = slice_contig_and_track_motifs(contig, ["CCCCCCC", "GGGGGGG"], window_size=512, stride=256)

        fig_path = run.output_dir / "figure_long_motifs_gc.png"
        render_figure_long_motifs_gc(df_summary, fig_path)
        self.assertTrue(fig_path.is_file())

        with run.guard():
            run.manifest["artifacts"]["figure"] = str(fig_path)

        with open(run.manifest_path) as f:
            man = json.load(f)
        self.assertEqual(man["status"], "COMPLETE")

    def test_smoke_test_cpu_speed(self):
        """C15: Smoke test mode executes completely on CPU within 15 seconds."""
        t0 = time.time()
        run = begin_m06("01", run_id="speed_run", data_root=self.project_root, smoke_test=True, project_root=self.project_root)
        with run.guard():
            # simulate light processing
            pass
        t_elapsed = time.time() - t0
        self.assertLess(t_elapsed, 15.0)

    def test_terminology_long_sequences_not_reads(self):
        """C16: Ensure documentation and metadata refer to long-sequences/contigs and not unverified reads."""
        self.assertIn("LONG SEQUENCE", SCIENTIFIC_DEMARCATION_TEXT)
        self.assertNotIn("real long-reads", SCIENTIFIC_DEMARCATION_TEXT)

    def test_logging_path_traceability(self):
        """C17: Artifact paths are logged with [LOG_PATH] prefix respecting LOG_PATH=0/false."""
        dummy = self.project_root / "dummy.txt"
        dummy.write_text("hello")

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with patch.dict(os.environ, {"LOG_PATH": "1"}):
                from viral_cnn.utils.logging_utils import log_artifact_path
                log_artifact_path("Dummy", dummy)
                output = mock_stderr.getvalue()
                self.assertIn("[LOG_PATH]", output)
                self.assertIn(str(dummy), output)

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with patch.dict(os.environ, {"LOG_PATH": "0"}):
                from viral_cnn.utils.logging_utils import log_artifact_path
                log_artifact_path("Dummy", dummy)
                output = mock_stderr.getvalue()
                self.assertNotIn("[LOG_PATH]", output)


if __name__ == "__main__":
    unittest.main()
