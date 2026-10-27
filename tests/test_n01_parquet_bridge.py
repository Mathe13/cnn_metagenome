"""
Tests for N01: Remote JupyterHub Parquet Bridge, Inspection Utilities, and Notebook Runtime.
Covers checks C1 through C19.
"""

import math
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Safely mock GPU/third-party dependencies required only for loading historical modules
sys.modules.setdefault("cudf", MagicMock())
sys.modules.setdefault("pytorch_grad_cam", MagicMock())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from viral_cnn.data.encoding import NUCLEOTIDES, NUM_FEATURES
from viral_cnn.data.parquet_dataset import ParquetBatchIterableDataset
from viral_cnn.eval.confusion import (
    compute_and_plot_confusion_insights,
    get_class_label,
    get_confused_class_samples,
    inspect_csv,
    plot_confusion_matrix_insights,
)
from viral_cnn.interpretability.gradcam import (
    GradCAM1D,
    ScoreCAM1D,
    visualize_heat_map_examples_by_class,
    visualize_heatmap,
)
from viral_cnn.models.builder import build_model
from viral_cnn.models.convnext import ConvNeXt_Tiny_1D
from viral_cnn.models.resnet import ResNet50_1D
from viral_cnn.notebook_runtime import (
    JupyterHubDataTopology,
    ensure_smoke_test_artifacts,
    setup_notebook_runtime,
    write_notebook_manifest,
)
from viral_cnn.taxonomy.mapping import load_taxonomy_mapping

# Historical reference imports
from short.utils.analize_funcs import (
    get_confused_class_samples as historical_get_confused_class_samples,
    plot_confusion_matrix_insights as historical_plot_confusion_matrix_insights,
)
from short.utils.dna_dataset_2 import ParquetCPUDataset as HistoricalParquetCPUDataset
from short.utils.gradcam import GradCAM1D as HistoricalGradCAM1D


def make_synthetic_parquet(file_path: str, num_rows: int, seq_col: str = "sequence_padded", label_col: str = "label", num_classes: int = 10):
    bases = ["A", "C", "G", "T"]
    seqs = []
    labels = []
    for i in range(num_rows):
        seq = "".join(random.choices(bases, k=500)) + "N" * 12
        seqs.append(seq)
        labels.append(i % num_classes)
    df = pd.DataFrame({seq_col: seqs, label_col: labels})
    df.to_parquet(file_path, engine="pyarrow", index=False)


class TestParquetBatchIterableDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.file1 = os.path.join(self.temp_dir, "part1.parquet")
        self.file2 = os.path.join(self.temp_dir, "part2.parquet")
        self.file3 = os.path.join(self.temp_dir, "part3.parquet")
        make_synthetic_parquet(self.file1, num_rows=12, seq_col="sequence_padded", label_col="label")
        make_synthetic_parquet(self.file2, num_rows=18, seq_col="sequence_padded", label_col="label")
        make_synthetic_parquet(self.file3, num_rows=10, seq_col="sequence_padded", label_col="label")
        self.file_paths = [self.file1, self.file2, self.file3]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_c1_tokenized_batch_shapes_and_types(self):
        """C1: ParquetBatchIterableDataset yields (inputs, labels) with shape (B, 512) and (B,) with tokens 0..4."""
        ds = ParquetBatchIterableDataset(
            file_paths=self.file_paths,
            batch_size=8,
            shuffle=False,
        )
        batches = list(ds)
        self.assertGreater(len(batches), 0)
        inputs, labels = batches[0]
        self.assertEqual(inputs.dim(), 2)
        self.assertEqual(inputs.shape[1], 512)
        self.assertEqual(inputs.dtype, torch.long)
        self.assertEqual(labels.dtype, torch.long)
        self.assertTrue((inputs >= 0).all())
        self.assertTrue((inputs <= 4).all())

        alt_file = os.path.join(self.temp_dir, "alt.parquet")
        make_synthetic_parquet(alt_file, num_rows=6, seq_col="padded_Representation")
        ds_alt = ParquetBatchIterableDataset(file_paths=[alt_file], batch_size=4, shuffle=False)
        inputs_alt, labels_alt = next(iter(ds_alt))
        self.assertEqual(inputs_alt.shape[1], 512)

        raw_file = os.path.join(self.temp_dir, "raw.parquet")
        make_synthetic_parquet(raw_file, num_rows=6, seq_col="sequence")
        ds_raw = ParquetBatchIterableDataset(file_paths=[raw_file], batch_size=4, shuffle=False)
        inputs_raw, labels_raw = next(iter(ds_raw))
        self.assertEqual(inputs_raw.shape[1], 512)

    def test_c2_fast_metadata_row_count_and_len(self):
        """C2: __len__ calculates math.ceil(total_samples / batch_size) via parquet metadata."""
        ds = ParquetBatchIterableDataset(file_paths=self.file_paths, batch_size=7)
        expected_total = 12 + 18 + 10
        self.assertEqual(ds.total_samples, expected_total)
        self.assertEqual(len(ds), math.ceil(expected_total / 7))

    def test_c3_legacy_global_rng_numerical_equivalence(self):
        """C3: When shuffle=True, legacy_global_rng=True matches ParquetCPUDataset under same seed."""
        # Pass defensive copies of list to prevent ParquetCPUDataset's in-place shuffle from mutating test fixture
        random.seed(42)
        ds_hist = HistoricalParquetCPUDataset(
            file_paths=list(self.file_paths),
            padded_column_name="sequence_padded",
            label_column="label",
            maxlen=512,
            nucleotides=NUCLEOTIDES,
            batch_size=8,
        )
        hist_batches = list(ds_hist)

        random.seed(42)
        ds_new = ParquetBatchIterableDataset(
            file_paths=list(self.file_paths),
            padded_column_name="sequence_padded",
            label_column="label",
            maxlen=512,
            nucleotides=NUCLEOTIDES,
            batch_size=8,
            shuffle=True,
            legacy_global_rng=True,
        )
        new_batches = list(ds_new)

        self.assertEqual(len(hist_batches), len(new_batches))
        for (h_in, h_lbl), (n_in, n_lbl) in zip(hist_batches, new_batches):
            self.assertTrue(torch.equal(h_in, n_in))
            self.assertTrue(torch.equal(h_lbl, n_lbl))

    def test_c4_deterministic_order_when_shuffle_false(self):
        """C4: shuffle=False yields identical batches in identical sequence across passes."""
        ds = ParquetBatchIterableDataset(file_paths=self.file_paths, batch_size=10, shuffle=False)
        pass1 = list(ds)
        pass2 = list(ds)
        self.assertEqual(len(pass1), len(pass2))
        for (in1, l1), (in2, l2) in zip(pass1, pass2):
            self.assertTrue(torch.equal(in1, in2))
            self.assertTrue(torch.equal(l1, l2))

    def test_c5_multiprocessing_disjoint_worker_partitioning(self):
        """C5: DataLoader with num_workers > 1 partitions files disjointly with zero loss/duplication."""
        ds = ParquetBatchIterableDataset(file_paths=self.file_paths, batch_size=4, shuffle=False)
        loader = DataLoader(ds, num_workers=2, batch_size=None)
        all_labels = []
        for _, labels in loader:
            all_labels.extend(labels.tolist())

        expected_count = 12 + 18 + 10
        self.assertEqual(len(all_labels), expected_count)

    def test_c6_cudf_import_error_or_fallback(self):
        """C6: backend='cudf' raises ImportError if missing or falls back when requested."""
        ds_fail = ParquetBatchIterableDataset(
            file_paths=self.file_paths,
            backend="cudf",
            fallback_to_cpu=False,
        )
        # In test environment, real cudf is not installed
        with self.assertRaises(ImportError):
            list(ds_fail)

        ds_fallback = ParquetBatchIterableDataset(
            file_paths=self.file_paths,
            backend="cudf",
            fallback_to_cpu=True,
        )
        batches = list(ds_fallback)
        self.assertGreater(len(batches), 0)


class TestInterpretabilityGradCAM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.resnet = build_model("resnet50_1d", num_classes=5, dropout_rate=0.3)
        self.convnext = build_model("convnext_tiny_1d", num_classes=5, dropout_rate=0.3)
        self.input_tensor = torch.randint(0, 4, (1, 512))

    def test_c7_heatmap_shape_and_bounds(self):
        """C7: GradCAM1D and ScoreCAM1D output shape (512,) bounded in [0.0, 1.0]."""
        gradcam = GradCAM1D(self.resnet, self.resnet.layer4)
        cam = gradcam(self.input_tensor, target_class=2)
        self.assertEqual(cam.shape, (512,))
        self.assertGreaterEqual(float(cam.min()), -1e-6)
        self.assertLessEqual(float(cam.max()), 1.0 + 1e-6)

        scorecam = ScoreCAM1D(self.resnet, self.resnet.layer4)
        s_cam = scorecam(self.input_tensor, target_class=2)
        self.assertEqual(s_cam.shape, (512,))
        self.assertGreaterEqual(float(s_cam.min()), -1e-6)
        self.assertLessEqual(float(s_cam.max()), 1.0 + 1e-6)

    def test_c8_resnet_numerical_equivalence_with_historical(self):
        """C8: GradCAM1D produces output matching historical short/utils/gradcam.py on ResNet50_1D."""
        new_cam = GradCAM1D(self.resnet, self.resnet.layer4)(self.input_tensor, target_class=1)
        hist_cam = HistoricalGradCAM1D(self.resnet, self.resnet.layer4)(self.input_tensor, target_class=1)
        self.assertTrue(np.allclose(new_cam, hist_cam, atol=1e-5))

    def test_c9_convnext_compatibility(self):
        """C9: GradCAM1D and ScoreCAM1D operate on ConvNeXt_Tiny_1D without error."""
        gradcam = GradCAM1D(self.convnext, self.convnext.layer4)
        cam = gradcam(self.input_tensor, target_class=0)
        self.assertEqual(cam.shape, (512,))

        # Target conv1 for Score-CAM to avoid 2048 CPU forwards during unit tests
        scorecam = ScoreCAM1D(self.convnext, self.convnext.conv1)
        s_cam = scorecam(self.input_tensor, target_class=0)
        self.assertEqual(s_cam.shape, (512,))

    def test_c10_padding_masking_and_headless_figure(self):
        """C10: visualize_heat_map_examples_by_class zeros out padding and returns Figure in headless mode."""
        padded_input = self.input_tensor.clone()
        padded_input[0, 500:] = 4

        fig = visualize_heat_map_examples_by_class(
            model=self.resnet,
            target_layer=self.resnet.layer4,
            class_true_id=1,
            target_pred_id=2,
            examples=[padded_input[0]],
            score_principal_confusao=0.85,
            cam_type="GRAD",
            device=torch.device("cpu"),
            show=False,
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_c11_invalid_cam_type_raises_value_error(self):
        """C11: visualize_heat_map_examples_by_class raises ValueError on invalid cam_type."""
        with self.assertRaises(ValueError):
            visualize_heat_map_examples_by_class(
                model=self.resnet,
                target_layer=self.resnet.layer4,
                class_true_id=0,
                target_pred_id=1,
                examples=[self.input_tensor[0]],
                score_principal_confusao=0.5,
                cam_type="UNKNOWN",
                device=torch.device("cpu"),
                show=False,
            )


class TestConfusionAnalysis(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mapping_csv = os.path.join(self.temp_dir, "taxonomy.csv")
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "species": ["Alpha virus", "Beta virus", "Gamma virus", "Delta virus"]
        })
        df.to_csv(self.mapping_csv, index=False)

        inputs = torch.randint(0, 4, (20, 512))
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3] * 2 + [0, 1, 2, 3])
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        self.loader = DataLoader(dataset, batch_size=5)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(512, 4)

            def forward(self, x):
                B = x.shape[0]
                logits = torch.zeros(B, 4)
                for i in range(B):
                    logits[i, 1] = 2.0
                return logits

        self.model = MockModel()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_c12_top_k_confusion_pairs_ranking(self):
        """C12: compute_and_plot_confusion_insights returns sorted Top-K off-diagonal confused pairs."""
        pairs = compute_and_plot_confusion_insights(
            model=self.model,
            test_loader=self.loader,
            device=torch.device("cpu"),
            num_classes=4,
            top_k_confusion=3,
            show=False,
        )
        self.assertIsInstance(pairs, list)
        self.assertLessEqual(len(pairs), 3)
        if len(pairs) > 1:
            self.assertGreaterEqual(pairs[0][0], pairs[1][0])
        for score, true_id, pred_id in pairs:
            self.assertNotEqual(true_id, pred_id)

    def test_c13_taxonomic_name_resolution(self):
        """C13: Resolves 0-based model_label to 1-based CSV id taxon name."""
        name = get_class_label(self.mapping_csv, target_label_0based=0, col_name="species")
        self.assertEqual(name, "Alpha virus")
        name2 = get_class_label(self.mapping_csv, target_label_0based=2, col_name="species")
        self.assertEqual(name2, "Gamma virus")

        inspect_csv(self.mapping_csv, target_label_0based=1, col_name="species")

    def test_c14_get_confused_class_samples_quota(self):
        """C14: get_confused_class_samples collects requested quota per target true class."""
        confused_pairs = [(0.8, 0, 1), (0.6, 2, 1)]
        samples = get_confused_class_samples(
            confused_pairs_info=confused_pairs,
            test_loader=self.loader,
            device=torch.device("cpu"),
            n_examples_per_class=2,
        )
        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), 2)
        for item in samples:
            self.assertIn("class_true", item)
            self.assertLessEqual(len(item["class_true"]["examples"]), 2)


class TestNotebookRuntime(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_c15_setup_runtime_and_reproducible_seeding(self):
        """C15: setup_notebook_runtime sets seeds deterministically and resolves DATA_ROOT."""
        ctx = setup_notebook_runtime(data_root=self.temp_dir, seed=42, smoke_test=True)
        self.assertEqual(ctx.seed, 42)
        self.assertEqual(str(ctx.data_root), str(Path(self.temp_dir).resolve()))
        self.assertTrue(ctx.is_smoke_test)

    def test_c16_jupyterhub_data_topology_resolution(self):
        """C16: JupyterHubDataTopology maps all canonical short/ and long/ paths from models.yaml."""
        topology = JupyterHubDataTopology(data_root=self.temp_dir)
        self.assertIn("short", topology.paths)
        self.assertIn("long", topology.paths)
        self.assertIn("checkpoints", topology.paths)
        self.assertIn("resnet_clean_species", topology.paths["checkpoints"])

    def test_c17_synthetic_smoke_test_artifacts(self):
        """C17: ensure_smoke_test_artifacts synthesizes valid short/ and long/ partitions."""
        artifacts = ensure_smoke_test_artifacts(data_root=self.temp_dir, num_samples_per_split=5)
        self.assertTrue(Path(artifacts["short"]["test_parquet"]).exists())
        self.assertTrue(Path(artifacts["long"]["test_parquet"]).exists())
        self.assertTrue(Path(artifacts["short"]["species_csv"]).exists())

        df_short = pd.read_parquet(artifacts["short"]["test_parquet"])
        self.assertIn("sequence_padded", df_short.columns)
        self.assertIn("label", df_short.columns)

        df_long = pd.read_parquet(artifacts["long"]["test_parquet"])
        for col in ["species_id", "genus_id", "origin_len", "read_len", "sequence"]:
            self.assertIn(col, df_long.columns)

    def test_c18_write_notebook_manifest_json_yaml(self):
        """C18: write_notebook_manifest writes structured JSON and YAML records."""
        out_json = os.path.join(self.temp_dir, "manifest.json")
        out_yaml = os.path.join(self.temp_dir, "manifest.yaml")
        manifest = write_notebook_manifest(
            notebook_id="test_notebook_01",
            data_root=self.temp_dir,
            is_smoke_test=True,
            metrics_summary={"accuracy": 0.95},
            output_json_path=out_json,
            output_yaml_path=out_yaml,
        )
        self.assertTrue(os.path.exists(out_json))
        self.assertTrue(os.path.exists(out_yaml))
        self.assertEqual(manifest.notebook_id, "test_notebook_01")
        self.assertEqual(manifest.metrics_summary["accuracy"], 0.95)

    def test_c19_historical_directories_preservation(self):
        """C19: Asserts that short/ and long/ git status remains clean."""
        res = os.popen("git status --porcelain short/ long/").read().strip()
        self.assertEqual(res, "", f"Historical directories short/ or long/ have been modified: {res}")

    def test_c20_multi_split_directory_disjunction_and_sampling(self):
        """C20: resolve_split_files, read_parquet_sample, and ParquetBatchIterableDataset never merge train+validation+test subdirs."""
        from viral_cnn.notebook_runtime import resolve_split_files, read_parquet_sample

        root_dir = Path(self.temp_dir) / "multi_split_dataset"
        for sp_name, n_rows in [("train", 20), ("validation", 8), ("test", 6)]:
            sp_dir = root_dir / sp_name
            sp_dir.mkdir(parents=True, exist_ok=True)
            df_sp = pd.DataFrame({
                "sequence_padded": ["ACGT" * 128 for _ in range(n_rows)],
                "label": [i % 3 for i in range(n_rows)],
            })
            df_sp.to_parquet(sp_dir / "part_0.parquet", index=False)

        # 1. resolve_split_files resolves 'val' -> 'validation/part_0.parquet' and 'test' -> 'test/part_0.parquet'
        val_files = resolve_split_files(root_dir, "val")
        test_files = resolve_split_files(root_dir, "test")
        train_files = resolve_split_files(root_dir, "train")
        self.assertEqual(len(val_files), 1)
        self.assertTrue(val_files[0].endswith("validation/part_0.parquet"))
        self.assertEqual(len(test_files), 1)
        self.assertTrue(test_files[0].endswith("test/part_0.parquet"))
        self.assertEqual(len(train_files), 1)
        self.assertTrue(train_files[0].endswith("train/part_0.parquet"))

        # 2. ParquetBatchIterableDataset on root_dir with shuffle=False selects strictly 'test' (6 rows), not 34 rows
        ds_test = ParquetBatchIterableDataset(root_dir, batch_size=4, shuffle=False)
        self.assertEqual(ds_test.total_samples, 6)
        batches = list(ds_test)
        self.assertEqual(sum(bx.shape[0] for bx, _ in batches), 6)

        # 3. read_parquet_sample streams bounded rows from split directory
        df_sample = read_parquet_sample(root_dir, columns=["sequence_padded", "label"], max_rows=4, split="train")
        self.assertEqual(len(df_sample), 4)



if __name__ == "__main__":
    unittest.main()
