"""Contract tests for S13: Dataset overlap scientific contract and manuscript reconciliation.

Verifies C1 through C8 from .specs/features/s13-dataset-overlap-contract/checks.md.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from viral_cnn.data.overlap_contract import (
    audit_manuscript_terminology,
    build_dataset_overlap_audit,
    write_dataset_overlap_audit_artifact,
)
from viral_cnn.reporting.manuscript import (
    build_manuscript_variants,
    compile_manuscript,
    generate_manuscript_content,
)


class TestS13DatasetOverlapContract(unittest.TestCase):
    """Test suite verifying S13 scientific contract and manuscript reconciliation."""

    # C1: Canonical schema
    def test_build_dataset_overlap_audit_schema(self):
        """C1: build_dataset_overlap_audit returns dict matching canonical S13 schema."""
        train_seqs = ["ACGTACGT", "GGGGCCCC"]
        test_seqs = ["ACGTACGT", "TTTTAAAA"]

        audit = build_dataset_overlap_audit(train_seqs, test_seqs)

        self.assertIsInstance(audit, dict)
        self.assertEqual(audit["split_unit"], "fragment")
        self.assertTrue(audit["row_disjoint"])
        self.assertFalse(audit["genome_disjoint"])

        self.assertIn("sequence_identity_audit", audit)
        sia = audit["sequence_identity_audit"]
        self.assertTrue(sia["canonicalizes_reverse_complement"])
        self.assertEqual(sia["n_queries"], 2)
        self.assertEqual(sia["n_exact_or_rc_overlap"], 1)
        self.assertEqual(sia["overlap_fraction"], 0.5)

    # C2: Reverse complement canonicalization
    def test_canonical_reverse_complement_matching(self):
        """C2: sequence_identity_audit canonicalizes RC matching forward and RC equally."""
        # Forward train sequence: "AAACCCGGG"
        # Reverse complement:     "CCCGGGTTT"
        train_seqs = ["AAACCCGGG"]
        test_rc = ["CCCGGGTTT"]

        # With canonicalize_rc=True, match must be detected
        audit_rc = build_dataset_overlap_audit(train_seqs, test_rc, canonicalize_rc=True)
        self.assertEqual(audit_rc["sequence_identity_audit"]["n_exact_or_rc_overlap"], 1)
        self.assertEqual(audit_rc["sequence_identity_audit"]["overlap_fraction"], 1.0)

        # With canonicalize_rc=False, forward-only hash fails to match
        audit_no_rc = build_dataset_overlap_audit(train_seqs, test_rc, canonicalize_rc=False)
        self.assertEqual(audit_no_rc["sequence_identity_audit"]["n_exact_or_rc_overlap"], 0)
        self.assertEqual(audit_no_rc["sequence_identity_audit"]["overlap_fraction"], 0.0)

    # C3: JSON artifact serialization
    def test_write_dataset_overlap_audit_artifact(self):
        """C3: write_dataset_overlap_audit_artifact writes valid JSON matching canonical format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "audit" / "dataset_overlap_audit.json"
            audit_data = {
                "split_unit": "fragment",
                "row_disjoint": True,
                "genome_disjoint": False,
                "sequence_identity_audit": {
                    "canonicalizes_reverse_complement": True,
                    "n_queries": 273660,
                    "n_exact_or_rc_overlap": 262470,
                    "overlap_fraction": 0.9591,
                },
            }

            written = write_dataset_overlap_audit_artifact(audit_data, out_path)
            self.assertTrue(written.is_file())

            loaded = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(loaded, audit_data)
            self.assertEqual(loaded["sequence_identity_audit"]["overlap_fraction"], 0.9591)

    # C4: Detection of sequence disjointness claims
    def test_audit_manuscript_detects_sequence_disjointness_claims(self):
        """C4: audit_manuscript_terminology reports violation on claims of sequence-level disjointness."""
        bad_text_1 = "The 80/10/10 split ensures fragment sequences are disjunct across partitions."
        res_1 = audit_manuscript_terminology(bad_text_1)
        self.assertFalse(res_1["passed"])
        self.assertTrue(any("fragment sequences are disjunct" in v["matched"].lower() for v in res_1["violations"]))

        bad_text_2 = "While no identical 512-bp sequence fragment appears across partitions, genomes overlap."
        res_2 = audit_manuscript_terminology(bad_text_2)
        self.assertFalse(res_2["passed"])
        self.assertTrue(any("no identical 512-bp sequence fragment appears" in v["matched"].lower() for v in res_2["violations"]))

    # C5: Detection of unqualified unseen-genome claims
    def test_audit_manuscript_detects_unqualified_unseen_genome_claims(self):
        """C5: audit_manuscript_terminology reports violation on unqualified unseen-genome claims."""
        bad_text = "The model demonstrates high accuracy and strong unseen-genome generalization."
        res = audit_manuscript_terminology(bad_text)
        self.assertFalse(res["passed"])
        self.assertTrue(any("unseen-genome generalization" in v["matched"].lower() for v in res["violations"]))

        # Contrast/negation is permitted and not flagged as an unqualified claim
        good_text = "Evaluation measures fragment classification rather than unseen-genome generalization."
        res_good = audit_manuscript_terminology(good_text)
        unseen_violations = [v for v in res_good["violations"] if "unseen-genome generalization" in v["matched"].lower()]
        self.assertEqual(len(unseen_violations), 0)

    # C6: Reconciled manuscript passes audit
    def test_reconciled_manuscript_passes_terminology_audit(self):
        """C6: generate_manuscript_content passes audit_manuscript_terminology in review and submission modes."""
        for review_mode in (False, True):
            content = generate_manuscript_content(review_mode=review_mode)
            report = audit_manuscript_terminology(content)
            self.assertTrue(report["passed"], f"Review mode {review_mode} failed: {report}")
            self.assertEqual(len(report["violations"]), 0)
            self.assertEqual(len(report["missing_required"]), 0)

    # C7: Reconciled manuscript contains required terminology
    def test_reconciled_manuscript_contains_required_terminology(self):
        """C7: generate_manuscript_content contains all three required scientific terms."""
        content = generate_manuscript_content(review_mode=False)
        self.assertIn("fragment-level closed-set classification", content)
        self.assertIn("known-reference distribution", content)
        self.assertIn("no-exact-overlap subset", content)

    # C8: Reconciled manuscript compiles cleanly
    def test_reconciled_manuscript_compiles_cleanly(self):
        """C8: compile_manuscript compiles reconciled manuscript_submission.tex with exit code 0."""
        paper_dir = Path("paper")
        build_manuscript_variants(paper_dir)
        submission_tex = paper_dir / "manuscript_submission.tex"
        self.assertTrue(submission_tex.exists())

        retcode = compile_manuscript(submission_tex)
        self.assertEqual(retcode, 0, "LaTeX compilation must succeed with return code 0")
        pdf_path = paper_dir / "manuscript_submission.pdf"
        self.assertTrue(pdf_path.exists(), "Output PDF must exist")
        self.assertGreater(pdf_path.stat().st_size, 1000, "PDF must not be empty")


if __name__ == "__main__":
    unittest.main()
