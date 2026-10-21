# S02: Dataset Provenance and Fragment Split Audit - verification

**Verdict**: PASS
**Profile**: light
**Diff range**: 3ae1a73..HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `docs/dataset_manifest.md` documents dataset provenance distinguishing observed fields from inferred/reconstructed fields across Kraken2, NCBI, and ICTV (AC 1) | `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_observed_vs_inferred_fields` exit 0 | `tests/test_dataset_audit.py:36` - `self.assertIn("Campos Diretamente Observados", content)` · `tests/test_dataset_audit.py:37` - `self.assertIn("Campos Inferidos", content)` | PASS |
| C2 | `docs/dataset_manifest.md` records 80/10/10 stratified fragment split parameters (`seed=42`) and the 95.91% (262,470 / 273,660) baseline exact overlap finding (AC 2) | `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_split_parameters_and_baseline_overlap` exit 0 | `tests/test_dataset_audit.py:51` - `self.assertIn("95,91%", content)` · `tests/test_dataset_audit.py:52` - `self.assertIn("262.470", content)` | PASS |
| C3 | `docs/dataset_manifest.md` documents non-zero sequence overlap risks from tiled strides and high-similarity sub-sequences when identical matches are absent (AC 3) | `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_subsequence_overlap_caveat` exit 0 | `tests/test_dataset_audit.py:59` - `self.assertIn("A ausência de correspondência de sequência idêntica NÃO equivale a zero sobreposição", content)` | PASS |
| C4 | `viral_cnn.data.audit.canonical_sequence_hash` computes strand-invariant SHA-256 digest on `min(seq, reverse_complement(seq))` across A, C, G, T, N (AC 4, AC 6) | `.venv/bin/python3 -m unittest discover -s tests -k test_canonical_sequence_hash_strand_invariance` exit 0 | `tests/test_dataset_audit.py:73` - `self.assertEqual(hash_fwd, hash_rev, "Canonical hash must be identical for seq and its reverse complement")` | PASS |
| C5 | `viral_cnn.data.audit.audit_partition_duplicates` measures total sequences, unique canonical sequences, and exact duplication percentage (AC 5) | `.venv/bin/python3 -m unittest discover -s tests -k test_audit_partition_duplicates_metrics` exit 0 | `tests/test_dataset_audit.py:84` - `self.assertEqual(report["total_sequences"], 6)` · `tests/test_dataset_audit.py:86` - `self.assertEqual(report["unique_canonical_sequences"], 5)` · `tests/test_dataset_audit.py:88` - `self.assertAlmostEqual(report["duplication_rate_pct"], 100.0 * 1 / 6, places=3)` | PASS |
| C6 | `viral_cnn.data.audit.audit_cross_partition_overlap` identifies forward and reverse complement duplicates across split partitions (AC 7, AC 10) | `.venv/bin/python3 -m unittest discover -s tests -k test_cross_partition_overlap_detection` exit 0 | `tests/test_dataset_audit.py:103` - `self.assertEqual(report["train_matches"], 2)` · `tests/test_dataset_audit.py:105` - `self.assertAlmostEqual(report["overlap_pct"], 50.0, places=3)` | PASS |
| C7 | CLI `scripts/audit_dataset.py` executes partition audit and emits structured JSON report with total, unique, and overlap counts with exit code 0 (AC 8, AC 10) | `.venv/bin/python3 -m unittest discover -s tests -k test_audit_cli_exit_zero_and_json_payload` exit 0 | `tests/test_dataset_audit.py:120` - `self.assertEqual(result.returncode, 0, f"CLI stderr: {result.stderr}")` · `tests/test_dataset_audit.py:122` - `self.assertIn("cross_partition_overlap", data)` | PASS |
| C8 | CLI `scripts/audit_dataset.py` validates arguments and exits with code 2 on missing files or unsupported file formats (AC 9) | `.venv/bin/python3 -m unittest discover -s tests -k test_audit_cli_invalid_arguments_exit_code_2` exit 0 | `tests/test_dataset_audit.py:131` - `self.assertEqual(res_no_args.returncode, 2)` · `tests/test_dataset_audit.py:141` - `self.assertEqual(res_bad_file.returncode, 2)` | PASS |
| C9 | Audit routines and CLI verify deterministic calculations against synthetic fixtures without scanning multi-gigabyte files (AC 11) | `.venv/bin/python3 -m unittest discover -s tests -k test_audit_synthetic_fixtures_deterministic_execution` exit 0 | `tests/test_dataset_audit.py:147` - `self.assertEqual(report1, report2)` | PASS |

## Gate

`.venv/bin/python3 -m unittest discover -s tests` - 19 passed, 0 failed

