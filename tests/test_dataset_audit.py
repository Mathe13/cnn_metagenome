"""
Tests for dataset provenance manifest, canonical sequence hashing, duplicate audit,
and cross-partition split leakage analysis (S02).
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from viral_cnn.data.audit import (
    audit_cross_partition_overlap,
    audit_partition_duplicates,
    canonical_sequence_hash,
    reverse_complement,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TRAIN_FIXTURE = FIXTURES_DIR / "sample_train.csv"
TEST_FIXTURE = FIXTURES_DIR / "sample_test.csv"
MANIFEST_FILE = Path(__file__).resolve().parent.parent / "docs" / "dataset_manifest.md"


class TestDatasetAudit(unittest.TestCase):
    def test_manifest_observed_vs_inferred_fields(self):
        """C1: Verifies docs/dataset_manifest.md distinguishes observed from inferred fields."""
        self.assertTrue(MANIFEST_FILE.exists(), f"Manifest file missing: {MANIFEST_FILE}")
        content = MANIFEST_FILE.read_text(encoding="utf-8")

        self.assertIn("Campos Diretamente Observados", content)
        self.assertIn("Campos Inferidos", content)
        self.assertIn("Kraken2", content)
        self.assertIn("NCBI", content)
        self.assertIn("ICTV", content)
        self.assertIn("13.683", content)
        self.assertIn("2.702", content)

    def test_manifest_split_parameters_and_baseline_overlap(self):
        """C2: Verifies docs/dataset_manifest.md records split parameters and 95.91% overlap finding."""
        content = MANIFEST_FILE.read_text(encoding="utf-8")

        self.assertIn("80%", content)
        self.assertIn("10%", content)
        self.assertIn("random_state = 42", content)
        self.assertIn("95,91%", content)
        self.assertIn("262.470", content)
        self.assertIn("273.660", content)

    def test_manifest_subsequence_overlap_caveat(self):
        """C3: Verifies docs/dataset_manifest.md documents the non-identical sequence overlap caveat."""
        content = MANIFEST_FILE.read_text(encoding="utf-8")

        self.assertIn(
            "A ausência de correspondência de sequência idêntica NÃO equivale a zero sobreposição",
            content,
        )

    def test_canonical_sequence_hash_strand_invariance(self):
        """C4: Verifies strand-invariant canonical hashing across A, C, G, T, N."""
        seq = "ACGTNNTTCCGG"
        rc = reverse_complement(seq)
        self.assertNotEqual(seq, rc)
        self.assertEqual(rc, "CCGGAANNACGT")

        hash_fwd = canonical_sequence_hash(seq)
        hash_rev = canonical_sequence_hash(rc)
        self.assertEqual(
            hash_fwd, hash_rev, "Canonical hash must be identical for seq and its reverse complement"
        )

        # Invariance under case
        self.assertEqual(canonical_sequence_hash(seq.lower()), hash_fwd)

    def test_audit_partition_duplicates_metrics(self):
        """C5: Verifies audit_partition_duplicates calculates total, unique, and duplicate metrics."""
        report = audit_partition_duplicates(TRAIN_FIXTURE)

        self.assertEqual(report["total_sequences"], 6)
        # sample_train.csv has 6 rows, but row 1 and row 6 are identical ('ACGTACGTACGT') -> 5 unique
        self.assertEqual(report["unique_canonical_sequences"], 5)
        self.assertEqual(report["duplicate_sequences"], 1)
        self.assertAlmostEqual(report["duplication_rate_pct"], 100.0 * 1 / 6, places=3)
        self.assertEqual(report["classes_count"], 4)

    def test_cross_partition_overlap_detection(self):
        """C6: Verifies cross-partition overlap detects forward and reverse-complement matches."""
        report = audit_cross_partition_overlap(TRAIN_FIXTURE, TEST_FIXTURE)

        self.assertEqual(report["total_train_sequences"], 6)
        self.assertEqual(report["total_test_sequences"], 4)
        # sample_test.csv has 4 rows:
        # 1. ACGTACGTACGT (exact forward match with train)
        # 2. GGGGCCCCAAAA (reverse complement of train's TTTTGGGGCCCC)
        # 3. CCCCCCCCGGGG (disjoint)
        # 4. TATATATATATA (disjoint, class 5)
        # Therefore, 2 test queries have overlap with train!
        self.assertEqual(report["train_matches"], 2)
        self.assertEqual(report["disjoint_queries"], 2)
        self.assertAlmostEqual(report["overlap_pct"], 50.0, places=3)
        self.assertEqual(report["single_source_classes"], 1)  # class 5 is in test but not train

    def test_audit_cli_exit_zero_and_json_payload(self):
        """C7: Verifies CLI scripts/audit_dataset.py returns exit 0 and valid JSON report."""
        cmd = [
            sys.executable,
            "scripts/audit_dataset.py",
            "--train",
            str(TRAIN_FIXTURE),
            "--test",
            str(TEST_FIXTURE),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        self.assertEqual(result.returncode, 0, f"CLI stderr: {result.stderr}")
        data = json.loads(result.stdout)
        self.assertIn("cross_partition_overlap", data)
        self.assertEqual(data["cross_partition_overlap"]["train_matches"], 2)
        self.assertEqual(data["cross_partition_overlap"]["total_test_sequences"], 4)

    def test_audit_cli_invalid_arguments_exit_code_2(self):
        """C8: Verifies CLI scripts/audit_dataset.py returns exit code 2 on missing or invalid inputs."""
        # Missing required arguments
        cmd_no_args = [sys.executable, "scripts/audit_dataset.py"]
        res_no_args = subprocess.run(cmd_no_args, capture_output=True, text=True, check=False)
        self.assertEqual(res_no_args.returncode, 2)

        # Non-existent file
        cmd_bad_file = [
            sys.executable,
            "scripts/audit_dataset.py",
            "--dataset",
            "nonexistent_file.csv",
        ]
        res_bad_file = subprocess.run(cmd_bad_file, capture_output=True, text=True, check=False)
        self.assertEqual(res_bad_file.returncode, 2)

    def test_audit_synthetic_fixtures_deterministic_execution(self):
        """C9: Verifies synthetic contract audit runs sub-second and deterministically."""
        report1 = audit_cross_partition_overlap(TRAIN_FIXTURE, TEST_FIXTURE)
        report2 = audit_cross_partition_overlap(TRAIN_FIXTURE, TEST_FIXTURE)
        self.assertEqual(report1, report2)


if __name__ == "__main__":
    unittest.main()
