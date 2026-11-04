"""Focused scientific contract tests for the C03 paired benchmark protocol."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viral_cnn.baselines.overlap import audit_exact_overlap, canonical_sequence_hash, load_prepared_baseline_benchmark
from viral_cnn.baselines.preparation import _noisy_sequence, prepare_queries, write_method_condition_manifest
from viral_cnn.baselines.paired import evaluate_predictions


class TestC03Contracts(unittest.TestCase):
    def test_overlap_is_forward_reverse_complement_canonical_and_64bp_is_sequence(self):
        forward = "AACCGT" * 10 + "AACC"
        reverse_complement = forward.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        frame = pd.DataFrame({"query_id": ["q1", "q2"], "sequence": [forward, "A" * 64], "label": [0, 1]})
        audit, masks = audit_exact_overlap(frame, [reverse_complement])
        self.assertEqual(audit["train_count"].tolist(), [1, 0])
        self.assertEqual(masks["all"].sum(), masks["overlap"].sum() + masks["no_exact_overlap"].sum())
        self.assertEqual(canonical_sequence_hash(forward), canonical_sequence_hash(reverse_complement))

    def test_overlap_loader_rejects_unknown_boolean_and_missing_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pd.DataFrame({"query_id": ["q1"], "label": [0], "has_overlap": ["not-a-bool"]}).to_csv(root / "exact_overlap.csv", index=False)
            with self.assertRaisesRegex(ValueError, "has_overlap"):
                load_prepared_baseline_benchmark(root, methods=[], noise_levels=[])
            pd.DataFrame({"query_id": ["q1"], "label": [0], "has_overlap": ["false"], "train_count": [None]}).to_csv(root / "exact_overlap.csv", index=False)
            with self.assertRaisesRegex(ValueError, "train_count"):
                load_prepared_baseline_benchmark(root, methods=[], noise_levels=[])

    def test_noise_is_replacement_with_original_base_eligible(self):
        import numpy as np
        sequence = "ACGT" * 64
        noisy, eligible, changed = _noisy_sequence(sequence, 1.0, np.random.default_rng(42))
        self.assertEqual(eligible, 256)
        self.assertEqual(len(noisy), len(sequence))
        self.assertTrue(set(noisy) <= set("ACGT"))
        self.assertLessEqual(changed, eligible)

    def test_query_export_is_repeatable_and_records_per_rate_content(self):
        import json
        import numpy as np
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mapping = pd.DataFrame({"species": [f"species_{i}" for i in range(13683)], "id": np.arange(1, 13684)})
            mapping_path = root / "species_name_ids.csv"
            mapping.to_csv(mapping_path, index=False)
            sequence = "ACGT" * 128
            train_path, test_path = root / "train.csv", root / "test.csv"
            pd.DataFrame({"label": [0], "sequence": [sequence]}).to_csv(train_path, index=False)
            pd.DataFrame({"query_id": ["q1"], "label": [0], "sequence": [sequence]}).to_csv(test_path, index=False)
            first = prepare_queries(train_path, test_path, root / "out_a", class_order_path=mapping_path)
            second = prepare_queries(train_path, test_path, root / "out_b", class_order_path=mapping_path)
            self.assertEqual(
                [item["query_content_sha256"] for item in first["conditions"]],
                [item["query_content_sha256"] for item in second["conditions"]],
            )
            self.assertEqual(first["selected_queries"], 1)
            audit = pd.read_csv(root / "out_a" / "exact_overlap.csv")
            self.assertEqual(audit.loc[0, "has_overlap"], True)
            qmanifest = json.loads((root / "out_a" / "query_manifest_noise_05.json").read_text())
            self.assertEqual(qmanifest["noise_protocol"], "resample_including_original_v1")
            self.assertEqual(qmanifest["class_order_sha256"], first["class_order_sha256"])

    def test_method_manifest_cannot_claim_complete_without_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            qmanifest = root / "query.json"
            qmanifest.write_text("{}")
            with self.assertRaisesRegex(ValueError, "existing output"):
                write_method_condition_manifest(
                    root / "manifest.json", method="BLASTN", noise_rate=0.0,
                    query_manifest=qmanifest, class_order_sha256="a" * 64,
                    reference={"path": "train.fasta", "sha256": "b" * 64},
                    tool_version="blastn 1", command=["blastn"], abstention_policy="empty hits => -1",
                    completion="COMPLETE", is_smoke_test=False,
                )

    def test_metrics_reject_misaligned_top5_and_keep_denominator(self):
        with self.assertRaisesRegex(ValueError, "top_candidates"):
            evaluate_predictions([0, 1], [0, -1], top_candidates=[[0]])
        metrics = evaluate_predictions([0, 1], [0, -1], top_candidates=[[0], []])
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["n_covered"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["top5"], 0.5)


if __name__ == "__main__":
    unittest.main()
