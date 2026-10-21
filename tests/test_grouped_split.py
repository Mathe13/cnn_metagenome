"""
Tests for genome-disjoint evaluation feasibility, grouped splitting engine,
and zero-leakage partition manifests (S03).
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from viral_cnn.data.grouped_split import (
    build_genome_disjoint_split,
    load_records_from_file,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GENOME_FIXTURE = FIXTURES_DIR / "sample_genomes.csv"
FEASIBILITY_DOC = Path(__file__).resolve().parent.parent / "docs" / "genome_split_feasibility.md"


class TestGroupedSplit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_feasibility_manifest_two_genome_requirement(self):
        """C1: Verifies docs/genome_split_feasibility.md documents the >= 2 genome requirement."""
        self.assertTrue(FEASIBILITY_DOC.exists(), f"Feasibility doc missing: {FEASIBILITY_DOC}")
        content = FEASIBILITY_DOC.read_text(encoding="utf-8")

        self.assertIn("no mínimo 2 genomas de referência independentes", content)
        self.assertIn("closed-set", content.lower())
        self.assertIn("zero-leakage", content)

    def test_feasibility_manifest_single_genome_population_shift(self):
        """C2: Verifies docs/genome_split_feasibility.md documents single-genome infeasibility & population shift."""
        content = FEASIBILITY_DOC.read_text(encoding="utf-8")

        self.assertIn("Inviabilidade de Classes com Genoma Único", content)
        self.assertIn("train_only", content)
        self.assertIn("mudança drástica de população avaliada", content)

    def test_feasibility_manifest_protocol_comparability_rules(self):
        """C3: Verifies docs/genome_split_feasibility.md documents strict protocol comparability rules."""
        content = FEASIBILITY_DOC.read_text(encoding="utf-8")

        self.assertIn("Regra Permanente de Comparabilidade", content)
        self.assertIn("Nunca confronte a acurácia do split agrupado", content)
        self.assertIn("13.683 espécies", content)

    def test_grouped_split_all_fragments_in_single_partition(self):
        """C4: Verifies all sequence fragments of any single genome reside strictly in one partition."""
        records = load_records_from_file(GENOME_FIXTURE)
        result = build_genome_disjoint_split(records, test_size=0.4, random_state=42)

        partitions = result["partitions"]
        genome_to_partitions: dict[str, set[str]] = {}

        for part_name, recs in partitions.items():
            for r in recs:
                gid = r["genome_id"]
                genome_to_partitions.setdefault(gid, set()).add(part_name)

        for gid, part_set in genome_to_partitions.items():
            self.assertEqual(
                len(part_set),
                1,
                f"Genome {gid} found in multiple partitions: {part_set}",
            )

    def test_grouped_split_zero_genome_leakage(self):
        """C5: Verifies zero genome accession overlap between train, val, and test partitions."""
        records = load_records_from_file(GENOME_FIXTURE)
        result = build_genome_disjoint_split(records, test_size=0.33, val_size=0.0, random_state=42)

        train_genomes = set(r["genome_id"] for r in result["partitions"]["train"])
        test_genomes = set(r["genome_id"] for r in result["partitions"]["test"])

        self.assertTrue(train_genomes.isdisjoint(test_genomes))
        self.assertEqual(result["metrics"]["genome_overlap_count"], 0)

    def test_grouped_split_single_genome_isolation(self):
        """C6: Verifies single-genome species are isolated to train_only or excluded."""
        records = load_records_from_file(GENOME_FIXTURE)

        # Policy train_only:
        res_train_only = build_genome_disjoint_split(
            records, single_genome_policy="train_only", random_state=42
        )
        # Species 3 (ACC_006) and Species 4 (ACC_007) have only 1 genome
        test_classes_train_only = set(r["species_id"] for r in res_train_only["partitions"]["test"])
        self.assertNotIn("3", test_classes_train_only)
        self.assertNotIn("4", test_classes_train_only)

        # Policy exclude:
        res_exclude = build_genome_disjoint_split(
            records, single_genome_policy="exclude", random_state=42
        )
        all_classes_exclude = set(
            r["species_id"]
            for part in res_exclude["partitions"].values()
            for r in part
        )
        self.assertNotIn("3", all_classes_exclude)
        self.assertNotIn("4", all_classes_exclude)

    def test_grouped_split_taxonomic_coverage_metrics(self):
        """C7: Verifies calculation of coverage metrics (eligible, infeasible, fragments)."""
        records = load_records_from_file(GENOME_FIXTURE)
        result = build_genome_disjoint_split(records, min_genomes_per_class=2, random_state=42)
        metrics = result["metrics"]

        # Fixture has 4 species: Species 1 (3 genomes), Species 2 (2 genomes), Species 3 (1), Species 4 (1)
        self.assertEqual(metrics["total_species"], 4)
        self.assertEqual(metrics["eligible_species_count"], 2)
        self.assertEqual(metrics["infeasible_species_count"], 2)
        self.assertAlmostEqual(metrics["taxonomic_coverage_pct"], 50.0, places=3)
        self.assertEqual(metrics["total_fragments"], 14)

    def test_build_grouped_split_cli_exit_zero_and_manifests(self):
        """C8: Verifies CLI scripts/build_grouped_split.py generates manifests with exit 0."""
        out_dir = Path(self.temp_dir) / "split_out"
        cmd = [
            sys.executable,
            "scripts/build_grouped_split.py",
            "--input",
            str(GENOME_FIXTURE),
            "--output-dir",
            str(out_dir),
            "--test-size",
            "0.3",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)

        self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")
        self.assertTrue((out_dir / "train_manifest.csv").exists())
        self.assertTrue((out_dir / "test_manifest.csv").exists())
        self.assertTrue((out_dir / "split_summary.json").exists())

        data = json.loads(res.stdout)
        self.assertEqual(data["metrics"]["genome_overlap_count"], 0)
        self.assertEqual(data["metrics"]["eligible_species_count"], 2)

    def test_build_grouped_split_cli_invalid_arguments_exit_code_2(self):
        """C9: Verifies CLI scripts/build_grouped_split.py exits with code 2 on bad arguments."""
        # Missing file
        cmd_bad_file = [
            sys.executable,
            "scripts/build_grouped_split.py",
            "--input",
            "nonexistent_file.csv",
            "--output-dir",
            self.temp_dir,
        ]
        res1 = subprocess.run(cmd_bad_file, capture_output=True, text=True, check=False)
        self.assertEqual(res1.returncode, 2)

        # Invalid test_size
        cmd_bad_test_size = [
            sys.executable,
            "scripts/build_grouped_split.py",
            "--input",
            str(GENOME_FIXTURE),
            "--output-dir",
            self.temp_dir,
            "--test-size",
            "1.5",
        ]
        res2 = subprocess.run(cmd_bad_test_size, capture_output=True, text=True, check=False)
        self.assertEqual(res2.returncode, 2)

    def test_grouped_split_deterministic_synthetic_execution(self):
        """C10: Verifies grouped split runs deterministically and sub-second."""
        records = load_records_from_file(GENOME_FIXTURE)
        res1 = build_genome_disjoint_split(records, random_state=42)
        res2 = build_genome_disjoint_split(records, random_state=42)

        self.assertEqual(res1["metrics"], res2["metrics"])
        self.assertEqual(len(res1["partitions"]["train"]), len(res2["partitions"]["train"]))
        self.assertEqual(len(res1["partitions"]["test"]), len(res2["partitions"]["test"]))


if __name__ == "__main__":
    unittest.main()
