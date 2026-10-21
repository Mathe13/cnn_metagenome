"""
Contract tests for repository inventory and historical notebook integrity (C1, C2, C3).
"""

import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_NOTEBOOKS = [
    "short/benchmark.ipynb",
    "short/finetune_resnet50_species.ipynb",
    "short/finetune_convnext_species.ipynb",
    "short/analize_convnext.ipynb",
    "short/ICTV_latent_space-bkp.ipynb",
    "short/full_loss_function.ipynb",
    "short/codon_usage.ipynb",
    "short/ICTV_improve_kraken_database.ipynb",
    "short/improve_kraken_database-Copy1.ipynb",
    "short/benchmark_cnns_kraken2_blast_overlap.ipynb",
    "long/cnn-long-voter.ipynb",
    "long/cnn-long-voter-compare.ipynb",
    "long/long_loss_function.ipynb",
    "long/download_from_ncbi.ipynb",
    "long/prepare database.ipynb",
]


class TestInventory(unittest.TestCase):
    def test_historical_notebooks_intact(self):
        """C1: Verifies all 15 historical notebooks exist and are valid JSON notebooks."""
        self.assertEqual(len(EXPECTED_NOTEBOOKS), 15)
        for nb_rel_path in EXPECTED_NOTEBOOKS:
            nb_path = PROJECT_ROOT / nb_rel_path
            self.assertTrue(nb_path.exists(), f"Notebook missing: {nb_rel_path}")
            self.assertTrue(nb_path.is_file(), f"Notebook path is not a file: {nb_rel_path}")
            with open(nb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("cells", data, f"Notebook missing 'cells' key: {nb_rel_path}")
            self.assertGreater(len(data["cells"]), 0, f"Notebook has 0 cells: {nb_rel_path}")

    def test_notebook_paper_mappings(self):
        """C2: Verifies docs/inventory.md exists and documents mappings to manuscript tables/figures."""
        inv_path = PROJECT_ROOT / "docs" / "inventory.md"
        self.assertTrue(inv_path.exists(), "docs/inventory.md does not exist")
        content = inv_path.read_text(encoding="utf-8")

        # Must cite paper tables and figures
        for table_num in range(1, 13):
            self.assertIn(f"Tabela {table_num}", content, f"docs/inventory.md missing Tabela {table_num}")
        for fig_num in range(1, 6):
            self.assertTrue(
                f"Fig. {fig_num}" in content or f"Figura {fig_num}" in content,
                f"docs/inventory.md missing Fig. {fig_num}",
            )

        # Must list all 15 notebooks
        for nb_rel_path in EXPECTED_NOTEBOOKS:
            nb_name = Path(nb_rel_path).name
            self.assertIn(nb_name, content, f"docs/inventory.md does not mention notebook {nb_name}")

    def test_missing_checkpoints_documented(self):
        """C3: Verifies missing checkpoints and Parquet splits are documented as blocking/remote."""
        inv_path = PROJECT_ROOT / "docs" / "inventory.md"
        self.assertTrue(inv_path.exists())
        content = inv_path.read_text(encoding="utf-8")

        # Checkpoints cited in historical notebooks
        expected_missing = [
            "best_resnet_classifier__fast_phase4.pt",
            "best_convnext_phase3_restarts_v2.pt",
            "best_convnext_classifier_fast_phase4_2_restarts.pt",
            "finetune_noise_7.pt",
            "finetune_noise_14.pt",
            "processed_kraken_species_splited_parquet",
        ]
        for item in expected_missing:
            self.assertIn(item, content, f"Missing item {item} not documented in docs/inventory.md")

        self.assertIn("Registro de Bloqueio", content)


if __name__ == "__main__":
    unittest.main()

