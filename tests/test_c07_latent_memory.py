"""Proofs for C07's batched, RAM-budgeted latent evaluation contract."""

import json
import hashlib
import os
import tempfile
import unittest
import ast
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from viral_cnn.eval.latent import (
    EmbeddingArtifact,
    LatentMemoryBudget,
    LinearProbeClassifier,
    extract_latent_embeddings,
    load_embedding_artifact,
    save_embedding_artifact,
)


def make_artifact(split, rows, features=None, labels=None):
    features = np.asarray(features if features is not None else np.arange(len(rows) * 4).reshape(len(rows), 4), dtype=np.float32)
    labels = np.asarray(labels if labels is not None else np.arange(len(rows)) % 2, dtype=np.int64)
    return EmbeddingArtifact(features, labels, np.asarray(rows, dtype=str), {
        "verified": True, "split": split, "variant": "tiny",
        "partitions": [f"{split}.parquet"], "partition_sha256": [f"{split}-hash"],
        "checkpoint": "fixture.pt", "checkpoint_sha256": "checkpoint-hash",
        "class_order": [0, 1], "encoding": {"channels": ["A", "C", "G", "T"]},
        "sampling": {"rule": "all", "seed": 42}, "mode": "synthetic",
        "extractor_code_sha256": "code-hash",
    })


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 4, bias=False)
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(self.projection(x))


class Dataset:
    def __init__(self, count):
        self.total_samples = count


class BatchOnlyArray:
    """Array view that rejects full materialization while allowing bounded row slices."""
    def __init__(self, array, maximum_batch):
        self.array = array
        self.maximum_batch = maximum_batch
        self.shape = array.shape
        self.dtype = array.dtype
        self.nbytes = array.nbytes
        self.batch_sizes = []

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        result = self.array[index]
        if isinstance(index, slice):
            size = len(result)
            self.batch_sizes.append(size)
            if size > self.maximum_batch:
                raise AssertionError(f"full-array materialization attempted: {size} rows")
        return result


class LatentMemoryTests(unittest.TestCase):
    def test_budget_defaults_to_available_ram_fraction_and_validates_overrides(self):
        with patch.object(LatentMemoryBudget, "available_ram_bytes", return_value=10_000_000):
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(LatentMemoryBudget.from_environment().budget_bytes, 3_000_000)
            with patch.dict(os.environ, {"LATENT_RAM_BUDGET_MB": "2"}, clear=True):
                self.assertEqual(LatentMemoryBudget.from_environment().budget_bytes, 2 * 1024 * 1024)
            with patch.dict(os.environ, {"LATENT_RAM_BUDGET_MB": "0"}, clear=True):
                self.assertEqual(LatentMemoryBudget.from_environment().budget_bytes, 0)
            for value in ("-1", "1.5", "bad"):
                with patch.dict(os.environ, {"LATENT_RAM_BUDGET_MB": value}, clear=True):
                    with self.assertRaises(ValueError):
                        LatentMemoryBudget.from_environment()

    def test_spill_roundtrip_exclusive_paths_and_owned_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root / "keep.txt"
            other.write_text("keep")
            left = LatentMemoryBudget(16, root / "spill", observe_system_memory=False)
            right = LatentMemoryBudget(0, root / "spill", observe_system_memory=False)
            a = left.allocate((2,), np.int64, "small")
            a[:] = [3, 4]
            b = left.allocate((3,), np.float32, "large")
            b[:] = [1.5, 2.5, 3.5]
            c = right.allocate((3,), np.float32, "large")
            c[:] = [6, 7, 8]
            np.testing.assert_array_equal(a, [3, 4])
            np.testing.assert_array_equal(b, [1.5, 2.5, 3.5])
            np.testing.assert_array_equal(c, [6, 7, 8])
            self.assertEqual(a.dtype, np.dtype(np.int64))
            self.assertEqual(a.shape, (2,))
            self.assertEqual(b.dtype, np.dtype(np.float32))
            self.assertEqual(b.shape, (3,))
            self.assertEqual(c.dtype, np.dtype(np.float32))
            self.assertEqual(c.shape, (3,))
            self.assertEqual(left.spill_count, 1)
            self.assertNotEqual(left.spill_paths, right.spill_paths)
            left.close()
            self.assertTrue(other.exists())
            self.assertTrue(right.spill_paths[0].exists())
            right.close()

    def test_extraction_batches_preserve_order_alignment_and_spill(self):
        count = 1031
        model = TinyModel().eval()
        tokens = torch.arange(count * 8, dtype=torch.float32).reshape(count, 8)
        labels = torch.arange(count) % 3
        row_ids = [f"partition:{i}" for i in range(count)]
        observed = []
        handle = model.register_forward_pre_hook(lambda _m, args: observed.append(len(args[0])))
        batches = [(tokens[i:i + 256], labels[i:i + 256], row_ids[i:i + 256]) for i in range(0, count, 256)]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                manager = LatentMemoryBudget(0, tmp, observe_system_memory=False)
                features, actual_labels, actual_ids, predictions = extract_latent_embeddings(
                    model, batches, device="cpu", return_identities=True,
                    return_predictions=True, memory_manager=manager,
                )
                self.assertTrue(all(size <= 256 for size in observed))
                self.assertEqual(observed, [256, 256, 256, 256, 7])
                np.testing.assert_array_equal(actual_labels, labels.numpy())
                np.testing.assert_array_equal(actual_ids, np.asarray(row_ids))
                with torch.inference_mode():
                    expected_features = model.projection(tokens).numpy()
                    expected_predictions = model(tokens).argmax(1).numpy()
                np.testing.assert_allclose(features, expected_features)
                np.testing.assert_array_equal(predictions, expected_predictions)
                self.assertGreater(manager.spill_count, 0)
                manager.close()
        finally:
            handle.remove()

    def test_embedding_artifact_streaming_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(42)
            item = make_artifact("test", [f"p:{i}" for i in range(1031)],
                                 rng.normal(size=(1031, 4)).astype(np.float32))
            expected_features = item.features.copy()
            guarded_features = BatchOnlyArray(item.features, 128)
            item.features = guarded_features
            path = save_embedding_artifact(item, Path(tmp) / "features", batch_size=128)
            self.assertEqual(guarded_features.batch_sizes, [128] * 8 + [7])
            expected = dict(item.manifest)
            expected.pop("verified")
            manager = LatentMemoryBudget(0, Path(tmp) / "spill", observe_system_memory=False)
            loaded = load_embedding_artifact(path, expected, memory_manager=manager, batch_size=128)
            np.testing.assert_array_equal(loaded.features, expected_features)
            np.testing.assert_array_equal(loaded.labels, item.labels)
            np.testing.assert_array_equal(loaded.row_ids, item.row_ids)
            self.assertEqual(loaded.features.dtype, item.features.dtype)
            self.assertEqual(loaded.features.shape, item.features.shape)
            self.assertEqual(loaded.labels.dtype, item.labels.dtype)
            self.assertEqual(loaded.row_ids.dtype, np.dtype("U256"))
            self.assertIsInstance(loaded.features, np.memmap)
            self.assertIsInstance(loaded.labels, np.memmap)
            self.assertIsInstance(loaded.row_ids, np.memmap)
            self.assertEqual(manager.ram_bytes, 0)
            self.assertEqual(manager.spill_bytes, loaded.features.nbytes + loaded.labels.nbytes + loaded.row_ids.nbytes)
            self.assertEqual(loaded.manifest["artifact_sha256"], item.manifest.get("artifact_sha256", loaded.manifest["artifact_sha256"]))
            self.assertGreater(manager.spill_count, 0)
            manager.close()

    def test_probe_validation_batches_and_preserves_global_selection(self):
        torch.manual_seed(2)
        rng = np.random.default_rng(3)
        train = make_artifact("train", [f"tr:{i}" for i in range(80)], rng.normal(size=(80, 4)))
        validation = make_artifact("validation", [f"va:{i}" for i in range(1031)], rng.normal(size=(1031, 4)))
        probe = LinearProbeClassifier(4, 2)
        sizes = []
        handle = probe.fc.register_forward_pre_hook(lambda _m, args: sizes.append(len(args[0])))
        try:
            result = probe.fit_select_and_seal(train, validation, epochs=1, batch_size=256)
        finally:
            handle.remove()
        self.assertLessEqual(max(sizes), 256)
        self.assertEqual(result["selection_rule"], "maximum validation Accuracy; earliest epoch wins ties")
        self.assertEqual(result["selected_epoch"], min(
            row["epoch"] for row in result["selection_history"]
            if row["validation_accuracy"] == max(x["validation_accuracy"] for x in result["selection_history"])
        ))
        self.assertEqual(len(result["selection_history"]), 1)
        with torch.inference_mode():
            direct_accuracy = float((probe(torch.from_numpy(validation.features)).argmax(1).numpy() == validation.labels).mean())
        self.assertAlmostEqual(result["selection_history"][0]["validation_accuracy"], direct_accuracy)
        self.assertEqual(result["state"], "SEALED")
        self.assertEqual(probe._audit_state, "SEALED")
        self.assertEqual(result["weights_sha256"], probe._sealed_sha256)
        self.assertEqual(len(result["weights_sha256"]), 64)

    def test_probe_test_evaluation_batches_preserve_metrics(self):
        torch.manual_seed(5)
        rng = np.random.default_rng(6)
        train = make_artifact("train", [f"tr:{i}" for i in range(40)], rng.normal(size=(40, 4)))
        validation = make_artifact("validation", [f"va:{i}" for i in range(24)], rng.normal(size=(24, 4)))
        test_x = rng.normal(size=(1031, 4)).astype(np.float32)
        test_y = np.arange(1031, dtype=np.int64) % 2
        test = make_artifact("test", [f"te:{i}" for i in range(1031)], test_x, test_y)
        probe = LinearProbeClassifier(4, 2)
        probe.fit_select_and_seal(train, validation, epochs=1, batch_size=8)
        sizes = []
        handle = probe.fc.register_forward_pre_hook(lambda _m, args: sizes.append(len(args[0])))
        try:
            actual = probe.evaluate_artifact(test, batch_size=256)
        finally:
            handle.remove()
        with torch.inference_mode():
            expected_pred = probe(torch.from_numpy(test_x)).argmax(1).numpy()
        expected_acc = float(np.mean(expected_pred == test_y))
        tp = np.bincount(test_y[test_y == expected_pred], minlength=2)
        predicted = np.bincount(expected_pred, minlength=2)
        support = np.bincount(test_y, minlength=2)
        expected_f1 = float(np.mean(2 * tp[support > 0] / (predicted[support > 0] + support[support > 0])))
        self.assertLessEqual(max(sizes), 256)
        self.assertEqual(actual["n_test_samples"], 1031)
        self.assertEqual(actual["supported_classes"], 2)
        self.assertAlmostEqual(actual["top1_accuracy"], expected_acc)
        self.assertAlmostEqual(actual["macro_f1"], expected_f1)

    def test_logits_are_spill_managed_and_notebook_keeps_top1(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb"
        notebook = json.loads(notebook_path.read_text())
        source = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
        self.assertIn("return_predictions=True", source)
        self.assertIn("native_test_predictions", source)
        self.assertNotIn("return_identities=True, return_logits=True", source)
        self.assertIn('"Test_Row_IDs_SHA256": eval_res["row_ids_sha256"]', source)
        model = TinyModel().eval()
        x = torch.randn(11, 8)
        y = torch.arange(11) % 3
        with tempfile.TemporaryDirectory() as tmp:
            manager = LatentMemoryBudget(0, tmp, observe_system_memory=False)
            result = extract_latent_embeddings(model, [(x, y)], return_logits=True, memory_manager=manager)
            self.assertEqual(result[2].shape, (11, 3))
            self.assertIsInstance(result[2], np.memmap)
            manager.close()
        test = make_artifact("test", [f"te:{i}" for i in range(8)], labels=np.arange(8) % 2)
        train = make_artifact("train", [f"tr:{i}" for i in range(8)])
        validation = make_artifact("validation", [f"va:{i}" for i in range(8)])
        probe = LinearProbeClassifier(4, 2)
        probe.fit_select_and_seal(train, validation, epochs=1, batch_size=4)
        evaluation = probe.evaluate_artifact(test, include_row_ids=False)
        self.assertEqual(evaluation["row_ids_sha256"], hashlib.sha256(test.row_ids.tobytes()).hexdigest())
        self.assertNotIn("row_ids", evaluation)

    def test_invalid_budget_and_spill_failure_do_not_publish_partial_artifact(self):
        with self.assertRaises(ValueError):
            LatentMemoryBudget(-1, ".")
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "file"
            blocker.write_text("not a directory")
            with self.assertRaises(OSError) as raised:
                LatentMemoryBudget(0, blocker / "spill").allocate((10,), np.float32, "failure")
            self.assertEqual(list(Path(tmp).glob("*.parquet")), [])
            self.assertIn(str(blocker / "spill"), str(raised.exception))
            partial = Path(tmp) / "partial.parquet"
            partial.write_bytes(b"partial")
            expected = dict(make_artifact("test", ["test:0"]).manifest)
            expected.pop("verified")
            with self.assertRaisesRegex(ValueError, "incomplete.*manifest.*partial.parquet"):
                load_embedding_artifact(partial, expected)

    def test_notebook_configures_and_records_memory_budget(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb"
        notebook = json.loads(notebook_path.read_text())
        source = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
        code = "\n".join("".join(c.get("source", [])) for c in notebook["cells"] if c.get("cell_type") == "code")
        tree = ast.parse(code)
        batch_assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                                and any(isinstance(target, ast.Name) and target.id == "latent_eval_batch_size"
                                        for target in node.targets))
        namespace = {"os": os}
        with patch.dict(os.environ, {}, clear=True):
            exec(compile(ast.Module(body=[batch_assignment], type_ignores=[]), "notebook", "exec"), namespace)
        self.assertEqual(namespace["latent_eval_batch_size"], 256)
        with patch.dict(os.environ, {"LATENT_EVAL_BATCH_SIZE": "64"}, clear=True):
            exec(compile(ast.Module(body=[batch_assignment], type_ignores=[]), "notebook", "exec"), namespace)
        self.assertEqual(namespace["latent_eval_batch_size"], 64)
        with patch.object(LatentMemoryBudget, "available_ram_bytes", return_value=10_000_000):
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(LatentMemoryBudget.from_environment().budget_bytes, 3_000_000)
            with patch.dict(os.environ, {"LATENT_RAM_BUDGET_MB": "2"}, clear=True):
                self.assertEqual(LatentMemoryBudget.from_environment().budget_bytes, 2 * 1024 * 1024)
        manifest_call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                             and isinstance(node.func, ast.Name) and node.func.id == "write_notebook_manifest")
        config = next(keyword.value for keyword in manifest_call.keywords if keyword.arg == "configuration")
        self.assertIsInstance(config, ast.Dict)
        config_values = {key.value: value for key, value in zip(config.keys, config.values)
                         if isinstance(key, ast.Constant)}
        self.assertEqual(ast.unparse(config_values["latent_eval_batch_size"]), "latent_eval_batch_size")
        self.assertEqual(ast.unparse(config_values["latent_ram_budget_mb"]), "latent_ram_budget_mb")
        self.assertEqual(ast.unparse(config_values["latent_spill_dir"]), "str(latent_spill_dir.resolve())")
        self.assertEqual(ast.unparse(config_values["latent_storage"]), "latent_memory.report()")
        self.assertTrue(any(isinstance(node, ast.Call) and ast.unparse(node.func) == "LatentMemoryBudget.from_environment"
                            for node in ast.walk(tree)))
        self.assertIn("[LATENT_MEMORY] batch_size=", code)


if __name__ == "__main__":
    unittest.main()
