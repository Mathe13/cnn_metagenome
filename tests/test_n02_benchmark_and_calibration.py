"""
Unit and integration tests for N02: Short-Read Clean Benchmark, Noise Robustness,
Hierarchical Entropy, and Calibration Notebooks.
Covers checks C1 through C18.
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from viral_cnn.eval.calibration import TaxonomyEntropyAnalyzer
from viral_cnn.eval.confusion import compare_confused_species_identity
from viral_cnn.eval.metrics import (
    compute_species_f1_bands,
    compute_species_specialist_comparison,
)
from viral_cnn.eval.runner import evaluate_dataloader
from viral_cnn.models.builder import build_model
from viral_cnn.models.registry import build_registered_model
from viral_cnn.training.profiler import profile_inference_model


class TestN02BenchmarkAndCalibration(unittest.TestCase):
    """Test suite covering checks C1 through C18 for N02."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cpu")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Slice S1: Faixas de F1, Comparação Especialista e Profiling
    # -------------------------------------------------------------------------

    def test_c1_species_f1_bands_default_thresholds(self):
        """C1: compute_species_f1_bands with default thresholds."""
        y_true = [0, 0, 1, 1, 2, 2, 3, 3]
        y_pred = [0, 0, 1, 0, 2, 2, 0, 0]  # classes 0, 1, 2, 3
        num_classes = 4

        res = compute_species_f1_bands(y_true, y_pred, num_classes=num_classes)
        bands = res["bands"]

        expected_keys = {"band_0_to_50", "band_50_to_80", "band_80_to_95", "band_95_to_100"}
        self.assertEqual(set(bands.keys()), expected_keys)

        total_count = sum(b["count"] for b in bands.values())
        self.assertEqual(total_count, num_classes)

        total_percent = sum(b["percent"] for b in bands.values())
        self.assertAlmostEqual(total_percent, 100.0, places=1)

    def test_c2_species_f1_bands_table4_thresholds_and_confidence(self):
        """C2: compute_species_f1_bands with Table 4 thresholds and confidences."""
        y_true = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        y_pred = [0, 0, 1, 1, 2, 0, 3, 0, 0, 0]
        confidences = [0.95, 0.90, 0.85, 0.88, 0.70, 0.65, 0.60, 0.55, 0.40, 0.45]
        num_classes = 5

        thresholds = (0.0, 0.50, 0.80, 0.90, 1.0)
        res = compute_species_f1_bands(
            y_true,
            y_pred,
            num_classes=num_classes,
            thresholds=thresholds,
            confidences=confidences,
        )
        bands = res["bands"]

        expected_keys = {"band_0_to_50", "band_50_to_80", "band_80_to_90", "band_90_to_100"}
        self.assertEqual(set(bands.keys()), expected_keys)

        total_count = sum(b["count"] for b in bands.values())
        self.assertEqual(total_count, num_classes)

        for k, b in bands.items():
            self.assertIn("mean_confidence", b)
            self.assertGreaterEqual(b["mean_confidence"], 0.0)
            self.assertLessEqual(b["mean_confidence"], 1.0)

    def test_c3_species_f1_bands_invalid_thresholds_raise_value_error(self):
        """C3: compute_species_f1_bands raises ValueError on invalid thresholds."""
        y_true = [0, 1]
        y_pred = [0, 1]

        # Not starting at 0.0
        with self.assertRaises(ValueError):
            compute_species_f1_bands(y_true, y_pred, 2, thresholds=(0.1, 0.5, 1.0))

        # Not ending at 1.0
        with self.assertRaises(ValueError):
            compute_species_f1_bands(y_true, y_pred, 2, thresholds=(0.0, 0.5, 0.9))

        # Not strictly increasing
        with self.assertRaises(ValueError):
            compute_species_f1_bands(y_true, y_pred, 2, thresholds=(0.0, 0.5, 0.5, 1.0))

        # Too few thresholds
        with self.assertRaises(ValueError):
            compute_species_f1_bands(y_true, y_pred, 2, thresholds=(0.0,))

    def test_c4_species_specialist_comparison_partition(self):
        """C4: compute_species_specialist_comparison partition conservation."""
        # 4 species:
        # species 0: resnet 2/2, convnext 1/2 -> resnet_superior
        # species 1: resnet 1/2, convnext 2/2 -> convnext_superior
        # species 2: resnet 1/2, convnext 1/2 -> ties
        # species 3: resnet 0/2, convnext 0/2 -> both_zero
        y_true = [0, 0, 1, 1, 2, 2, 3, 3]
        preds_resnet = [0, 0, 1, 0, 2, 0, 0, 0]
        preds_convnext = [0, 1, 1, 1, 2, 0, 1, 1]
        confs = [0.9] * len(y_true)

        res = compute_species_specialist_comparison(
            y_true=y_true,
            preds_resnet=preds_resnet,
            preds_convnext=preds_convnext,
            confs_resnet=confs,
            confs_convnext=confs,
            num_classes=4,
        )

        self.assertEqual(res["resnet_superior"], 1)
        self.assertEqual(res["convnext_superior"], 1)
        self.assertEqual(res["ties"], 1)
        self.assertEqual(res["both_zero"], 1)
        self.assertEqual(res["total_evaluated_species"], 4)

    def test_c5_profile_inference_model_metrics(self):
        """C5: profile_inference_model returns required metrics."""
        model = build_model("resnet50_1d", num_classes=5)
        prof = profile_inference_model(
            model=model,
            input_size=(4, 512),
            device=self.device,
            num_warmup=1,
            num_runs=3,
        )

        required_keys = {
            "backbone_params",
            "head_params",
            "total_params",
            "trainable_params",
            "size_mb",
            "gflops",
            "avg_latency_ms",
            "std_latency_ms",
            "throughput_seq_sec",
        }
        for k in required_keys:
            self.assertIn(k, prof)
            self.assertGreater(prof[k], 0)

    def test_c6_evaluate_dataloader_noise_sweep(self):
        """C6: evaluate_dataloader over noise sweep with rate verification."""
        model = build_model("resnet50_1d", num_classes=5)
        inputs = torch.randint(0, 4, (32, 512), dtype=torch.long)
        labels = torch.randint(0, 5, (32,), dtype=torch.long)
        ds = TensorDataset(inputs, labels)
        loader = DataLoader(ds, batch_size=16)

        noise_rates = [0.0, 0.05, 0.10]
        results = evaluate_dataloader(
            model=model,
            dataloader=loader,
            device=self.device,
            noise_rates=noise_rates,
            seed=42,
            num_classes=5,
            verify_noise=True,
        )

        self.assertEqual(set(results.keys()), set(noise_rates))
        for rate, res in results.items():
            self.assertEqual(res.noise_rate, rate)
            self.assertIsNotNone(res.metrics.top1_accuracy)
            self.assertIsNotNone(res.f1_bands)
            if rate > 0.0:
                expected = 0.75 * rate
                self.assertAlmostEqual(res.effective_noise_rate, expected, delta=0.03)

    # -------------------------------------------------------------------------
    # Slice S2: TaxonomyEntropyAnalyzer
    # -------------------------------------------------------------------------

    def test_c7_taxonomy_entropy_analyzer_hook_registration_and_cleanup(self):
        """C7: TaxonomyEntropyAnalyzer hook registration and removal."""
        model = build_model("resnet50_1d", num_classes=5)
        species_to_genus = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)

        analyzer = TaxonomyEntropyAnalyzer(model, species_to_genus, device=self.device)
        self.assertIsNotNone(analyzer.hook_handle)

        # Trigger forward to check activation capture
        x = torch.randint(0, 4, (2, 512), dtype=torch.long)
        _ = analyzer.analyze_batch(x)
        self.assertIn("pre_classifier", analyzer.activations)
        self.assertEqual(analyzer.activations["pre_classifier"].shape[0], 2)

        # Remove hook
        analyzer.remove_hook()
        self.assertIsNone(analyzer.hook_handle)

        # Test context manager cleanup
        with TaxonomyEntropyAnalyzer(model, species_to_genus, device=self.device) as ctx_analyzer:
            self.assertIsNotNone(ctx_analyzer.hook_handle)
        self.assertIsNone(ctx_analyzer.hook_handle)

    def test_c8_taxonomy_entropy_analyzer_validation_errors(self):
        """C8: TaxonomyEntropyAnalyzer raises on missing fc or negative indices."""
        model_no_fc = nn.Sequential(nn.Conv1d(4, 16, 3), nn.AdaptiveAvgPool1d(1))
        species_to_genus = torch.tensor([0, 1, 2], dtype=torch.long)

        # Missing fc
        with self.assertRaises(AttributeError):
            TaxonomyEntropyAnalyzer(model_no_fc, species_to_genus, device=self.device)

        # Negative indices in mapping
        valid_model = build_model("resnet50_1d", num_classes=3)
        invalid_mapping = torch.tensor([0, -1, 1], dtype=torch.long)
        with self.assertRaises(ValueError):
            TaxonomyEntropyAnalyzer(valid_model, invalid_mapping, device=self.device)

    def test_c9_taxonomy_entropy_analyzer_analyze_batch_bounds(self):
        """C9: analyze_batch outputs within theoretical bounds."""
        num_species = 8
        num_genus = 3
        model = build_model("resnet50_1d", num_classes=num_species)
        species_to_genus = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.long)

        with TaxonomyEntropyAnalyzer(model, species_to_genus, device=self.device) as analyzer:
            x = torch.randint(0, 4, (4, 512), dtype=torch.long)
            out = analyzer.analyze_batch(x)

            for key in ["h_species", "h_genus", "h_activation", "hierarchical_entropy_gap"]:
                self.assertIn(key, out)
                self.assertIsInstance(out[key], np.ndarray)
                self.assertEqual(len(out[key]), 4)

            # Bounds
            self.assertTrue((out["h_species"] <= math.log(num_species) + 1e-5).all())
            self.assertTrue((out["h_genus"] <= math.log(num_genus) + 1e-5).all())
            self.assertTrue((out["hierarchical_entropy_gap"] >= -1e-5).all())
            self.assertTrue((out["h_activation"] >= 0.0).all())
            self.assertTrue((out["h_activation"] <= 1.0 + 1e-5).all())

    def test_c10_taxonomy_entropy_analyzer_dataloader_aggregation(self):
        """C10: analyze_dataloader aggregates sample-weighted averages."""
        num_species = 5
        model = build_model("resnet50_1d", num_classes=num_species)
        species_to_genus = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)

        x = torch.randint(0, 4, (16, 512), dtype=torch.long)
        y = torch.randint(0, num_species, (16,), dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        with TaxonomyEntropyAnalyzer(model, species_to_genus, device=self.device) as analyzer:
            summary = analyzer.analyze_dataloader(loader)

            self.assertEqual(summary["n_samples"], 16)
            for k in ["mean_h_species", "mean_h_genus", "mean_h_activation", "mean_hierarchical_gap"]:
                self.assertIn(k, summary)
                self.assertIsInstance(summary[k], float)

    # -------------------------------------------------------------------------
    # Slice S3: Notebook 01_benchmark_clean_and_noise.ipynb
    # -------------------------------------------------------------------------

    def test_c11_notebook_01_imports_and_architecture_decoupling(self):
        """C11: Notebook 01 imports viral_cnn without short/utils/*."""
        nb_path = Path("notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb")
        self.assertTrue(nb_path.exists(), f"Notebook {nb_path} must exist.")

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_text = ""
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                code_text += "".join(cell.get("source", [])) + "\n"

        self.assertNotIn("short.utils", code_text)
        self.assertNotIn("from utils", code_text)
        self.assertNotIn("class ResNet50_1D", code_text)
        self.assertNotIn("class HybridConvNeXt", code_text)
        self.assertIn("viral_cnn", code_text)

    def test_c12_notebook_01_execution_smoke_test(self):
        """C12: Notebook 01 executes without error under SMOKE_TEST=1."""
        nb_path = Path("notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        global_ns = {"__name__": "__main__"}
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                try:
                    exec(code, global_ns)
                except Exception as e:
                    self.fail(f"Cell {idx} of 01_benchmark_clean_and_noise.ipynb raised exception: {e}")

    def test_c13_notebook_01_artifacts_generated(self):
        """C13: Notebook 01 outputs structured artifacts."""
        out_dir = Path("results_reproduction/n02_benchmark")
        expected_files = [
            "benchmark_clean_noise_summary.csv",
            "table2_profiling.csv",
            "table4_f1_bands.csv",
            "specialist_comparison.csv",
            "specialist_comparison_eps010.csv",
            "manifest.json",
        ]
        for fname in expected_files:
            fpath = out_dir / fname
            self.assertTrue(fpath.exists(), f"Expected artifact {fname} not found in {out_dir}")
            self.assertGreater(fpath.stat().st_size, 0)

    # -------------------------------------------------------------------------
    # Slice S4: Notebook 02 e Identidade de Espécies
    # -------------------------------------------------------------------------

    def test_c14_compare_confused_species_identity_offline(self):
        """C14: compare_confused_species_identity executes offline."""
        seq1 = "ACGTACGT" * 16
        seq2 = "ACGAACGT" * 16

        res = compare_confused_species_identity(seq1, seq2, use_ncbi_online=False)

        self.assertIn("identity", res)
        self.assertIn("alignment_score", res)
        self.assertIn("aligned_seq1", res)
        self.assertIn("aligned_seq2", res)
        self.assertGreater(res["identity"], 0.0)
        self.assertLessEqual(res["identity"], 100.0)

        # Test with token tensor
        tokens1 = torch.tensor([0, 1, 2, 3] * 8, dtype=torch.long)
        tokens2 = torch.tensor([0, 1, 2, 3] * 8, dtype=torch.long)
        res_tokens = compare_confused_species_identity(tokens1, tokens2)
        self.assertEqual(res_tokens["identity"], 100.0)

    def test_c15_notebook_02_imports_and_decoupling(self):
        """C15: Notebook 02 imports viral_cnn without short/utils/*."""
        nb_path = Path("notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb")
        self.assertTrue(nb_path.exists(), f"Notebook {nb_path} must exist.")

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_text = ""
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                code_text += "".join(cell.get("source", [])) + "\n"

        self.assertNotIn("short.utils", code_text)
        self.assertNotIn("from utils", code_text)
        self.assertIn("viral_cnn", code_text)

    def test_c16_notebook_02_execution_smoke_test(self):
        """C16: Notebook 02 executes without error under SMOKE_TEST=1."""
        nb_path = Path("notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        global_ns = {"__name__": "__main__"}
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                try:
                    exec(code, global_ns)
                except Exception as e:
                    self.fail(f"Cell {idx} of 02_analyze_confidence_and_calibration.ipynb raised exception: {e}")

    def test_c17_notebook_02_artifacts_generated(self):
        """C17: Notebook 02 outputs structured artifacts."""
        out_dir = Path("results_reproduction/n02_calibration")
        expected_files = [
            "confused_pairs_identity.csv",
            "table7_hierarchical_entropy.csv",
            "calibration_audit_noise.csv",
            "manifest.json",
        ]
        for fname in expected_files:
            fpath = out_dir / fname
            self.assertTrue(fpath.exists(), f"Expected artifact {fname} not found in {out_dir}")
            self.assertGreater(fpath.stat().st_size, 0)

    def test_c18_historical_directories_preservation(self):
        """C18: Historical short/ and long/ directories remain strictly untouched."""
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain", "short/", "long/"],
            text=True,
        ).strip()
        self.assertEqual(
            git_status,
            "",
            f"Historical directories short/ or long/ have modified files:\n{git_status}",
        )

    # -------------------------------------------------------------------------
    # Spec R01: Parameter Reconciliation, Fail-Fast and Benchmark Tracking
    # -------------------------------------------------------------------------

    def test_r01_resnet_parameter_decomposition(self):
        """C1: profile_inference_model decomposes ResNet50_1D parameters for Species and Genus."""
        # Species (13683)
        model_species = build_model("resnet50_1d", num_classes=13683)
        prof_sp = profile_inference_model(model_species, input_size=(2, 512), device=self.device, num_warmup=1, num_runs=1)
        self.assertEqual(prof_sp["backbone_params"], 15955584)
        self.assertEqual(prof_sp["head_params"], 28036467)
        self.assertEqual(prof_sp["total_params"], 43992051)
        self.assertEqual(prof_sp["backbone_params"] + prof_sp["head_params"], prof_sp["total_params"])

        # Genus (2702)
        model_genus = build_model("resnet50_1d", num_classes=2702)
        prof_gen = profile_inference_model(model_genus, input_size=(2, 512), device=self.device, num_warmup=1, num_runs=1)
        self.assertEqual(prof_gen["backbone_params"], 15955584)
        self.assertEqual(prof_gen["head_params"], 5536398)
        self.assertEqual(prof_gen["total_params"], 21491982)
        self.assertEqual(prof_gen["backbone_params"] + prof_gen["head_params"], prof_gen["total_params"])

    def test_r01_convnext_parameter_decomposition(self):
        """C2: profile_inference_model decomposes ConvNeXt_Tiny_1D parameters for Species and Genus."""
        # Species (13683)
        model_species = build_model("convnext_tiny_1d", num_classes=13683)
        prof_sp = profile_inference_model(model_species, input_size=(2, 512), device=self.device, num_warmup=1, num_runs=1)
        self.assertEqual(prof_sp["backbone_params"], 135977344)
        self.assertEqual(prof_sp["head_params"], 28036467)
        self.assertEqual(prof_sp["total_params"], 164013811)
        self.assertEqual(prof_sp["backbone_params"] + prof_sp["head_params"], prof_sp["total_params"])

        # Genus (2702)
        model_genus = build_model("convnext_tiny_1d", num_classes=2702)
        prof_gen = profile_inference_model(model_genus, input_size=(2, 512), device=self.device, num_warmup=1, num_runs=1)
        self.assertEqual(prof_gen["backbone_params"], 135977344)
        self.assertEqual(prof_gen["head_params"], 5536398)
        self.assertEqual(prof_gen["total_params"], 141513742)
        self.assertEqual(prof_gen["backbone_params"] + prof_gen["head_params"], prof_gen["total_params"])

    def test_r01_gitignore_selective_tracking(self):
        """C3: git check-ignore allows lightweight records and blocks binary weights/parquets under results_reproduction/."""
        # Ignored files
        ignored_files = [
            "results_reproduction/checkpoints/resnet50_noise/finetune_noise_last.pt",
            "results_reproduction/checkpoints/convnext_noise/finetune_noise_1.pth",
            "results_reproduction/taxonomy/main_taxonomy_hybrid_CLEANSED.parquet",
            "results_reproduction/index.sqlite",
            "results_reproduction/cache.db",
        ]
        for f in ignored_files:
            ret = subprocess.run(["git", "check-ignore", f], capture_output=True)
            self.assertEqual(ret.returncode, 0, f"Expected {f} to be ignored by .gitignore")

        # Tracked files (not ignored)
        tracked_files = [
            "results_reproduction/n02_benchmark/benchmark_clean_noise_summary.csv",
            "results_reproduction/n02_benchmark/manifest.json",
            "results_reproduction/manifest.yaml",
            "results_reproduction/figure.png",
        ]
        for f in tracked_files:
            ret = subprocess.run(["git", "check-ignore", f], capture_output=True)
            self.assertNotEqual(ret.returncode, 0, f"Expected {f} to be tracked (not ignored) by .gitignore")

    def test_r01_notebook_01_fail_fast_missing_checkpoint(self):
        """C4: Notebook 01 cell 3 raises FileNotFoundError if is_smoke_test is False and a checkpoint is missing."""
        nb_path = Path("notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        from unittest.mock import MagicMock
        mock_ctx = MagicMock()
        mock_ctx.is_smoke_test = False
        mock_ctx.topology.paths = {
            "checkpoints": {
                "resnet_clean_species": "/nonexistent/checkpoint/path.pt",
            }
        }
        mock_ctx.device = "cpu"

        cell_3_code = "".join(nb["cells"][3]["source"])
        ns = {
            "ctx": mock_ctx,
            "Path": Path,
            "species_variants": ["resnet_clean_species"],
            "genus_variants": [],
            "build_registered_model": build_registered_model,
            "profile_inference_model": profile_inference_model,
            "free_gpu_memory": lambda m: None,
            "pd": pd,
            "output_dir": Path(self.temp_dir),
        }
        with self.assertRaises(FileNotFoundError):
            exec(cell_3_code, ns)

    def test_r01_notebook_01_fail_fast_sample_count(self):
        """C5: Notebook 01 cell 4 raises RuntimeError if is_smoke_test is False and n_samples diverges."""
        nb_path = Path("notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        from unittest.mock import MagicMock
        mock_ctx = MagicMock()
        mock_ctx.is_smoke_test = False
        mock_ctx.topology.paths = {
            "checkpoints": {v: None for v in [
                "resnet_clean_species", "resnet_noise_species", "convnext_zero_species",
                "convnext_transfer_species", "convnext_noise_species", "resnet_genus", "convnext_genus"
            ]}
        }
        mock_ctx.device = "cpu"

        class MockMetrics:
            top1_accuracy = 0.5
            top5_accuracy = 0.8
            macro_f1 = 0.5
            weighted_f1 = 0.5
            cross_entropy_loss = 2.0
            n_samples = 100  # Diverges from 1706270

        class MockResult:
            effective_noise_rate = 0.0
            metrics = MockMetrics()
            f1_bands = {"bands": {}}
            y_true_arr = None
            y_pred_arr = None
            confs_arr = None
            predictions = []

        mock_evals = {0.0: MockResult()}

        cell_4_code = "".join(nb["cells"][4]["source"])
        ns = {
            "ctx": mock_ctx,
            "Path": Path,
            "species_variants": ["resnet_clean_species"],
            "genus_variants": [],
            "build_registered_model": lambda *a, **kw: nn.Linear(10, 5),
            "evaluate_dataloader": lambda *a, **kw: mock_evals,
            "test_loader_species": None,
            "test_loader_genus": None,
            "free_gpu_memory": lambda m: None,
            "pd": pd,
            "output_dir": Path(self.temp_dir),
        }
        with self.assertRaises(RuntimeError) as ctx_err:
            exec(cell_4_code, ns)
        self.assertIn("Species sample count mismatch", str(ctx_err.exception))

    def test_r01_notebook_01_prints_tables_and_manifest(self):
        """C6: Notebook 01 executes in smoke mode, prints tables to stdout, and logs 7 checkpoints in manifest."""
        import io
        import sys

        nb_path = Path("notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        captured_stdout = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_stdout

        global_ns = {"__name__": "__main__"}
        try:
            for cell in nb["cells"]:
                if cell.get("cell_type") == "code":
                    exec("".join(cell.get("source", [])), global_ns)
        finally:
            sys.stdout = old_stdout

        output_text = captured_stdout.getvalue()
        self.assertIn("Profiling table saved.", output_text)
        self.assertIn("Benchmark summary saved.", output_text)
        self.assertIn("Table 4 F1 bands saved.", output_text)
        self.assertIn("Specialist Comparison at epsilon = 0.05", output_text)
        self.assertIn("Specialist Comparison at epsilon = 0.10", output_text)
        self.assertIn("Manifest successfully written.", output_text)

        out_dir = Path("results_reproduction/n02_benchmark")
        self.assertTrue((out_dir / "specialist_comparison.csv").exists())
        self.assertTrue((out_dir / "specialist_comparison_eps010.csv").exists())

        manifest_path = out_dir / "manifest.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        self.assertIn("checkpoints", manifest_data)
        expected_models = [
            "resnet_clean_species",
            "resnet_noise_species",
            "convnext_zero_species",
            "convnext_transfer_species",
            "convnext_noise_species",
            "resnet_genus",
            "convnext_genus",
        ]
        for m in expected_models:
            self.assertIn(m, manifest_data["checkpoints"])


if __name__ == "__main__":
    unittest.main()

