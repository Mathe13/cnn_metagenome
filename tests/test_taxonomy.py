"""
Contract tests for taxonomic mapping and 1-based to 0-based label alignment (C4).
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from viral_cnn.taxonomy.mapping import (
    taxonomic_id_to_model_label,
    model_label_to_taxonomic_id,
    load_taxonomy_mapping,
)


class TestTaxonomyMapping(unittest.TestCase):
    def test_label_zero_based_offset(self):
        """C4: Verifies taxonomic ID to model label conversion satisfies model_label = id - 1."""
        # Species rank bounds
        self.assertEqual(taxonomic_id_to_model_label(1), 0)
        self.assertEqual(taxonomic_id_to_model_label(13683), 13682)
        self.assertEqual(model_label_to_taxonomic_id(0), 1)
        self.assertEqual(model_label_to_taxonomic_id(13682), 13683)

        # Genus rank bounds
        self.assertEqual(taxonomic_id_to_model_label(2702), 2701)
        self.assertEqual(model_label_to_taxonomic_id(2701), 2702)

        # Invalid bounds checks
        with self.assertRaises(ValueError):
            taxonomic_id_to_model_label(0)
        with self.assertRaises(ValueError):
            model_label_to_taxonomic_id(-1)

        # Verify against real CSV files
        species_csv = PROJECT_ROOT / "dataset_kraken" / "species_name_ids.csv"
        self.assertTrue(species_csv.exists())
        species_map = load_taxonomy_mapping(species_csv, rank_column="species")
        self.assertEqual(len(species_map), 13683)
        self.assertIn(0, species_map)
        self.assertIn(13682, species_map)
        self.assertEqual(species_map[0], "Alphafusarivirus rutstroemiae")

        genus_csv = PROJECT_ROOT / "dataset_kraken" / "genus_name_ids.csv"
        self.assertTrue(genus_csv.exists())
        genus_map = load_taxonomy_mapping(genus_csv, rank_column="genus")
        self.assertEqual(len(genus_map), 2702)
        self.assertIn(0, genus_map)
        self.assertIn(2701, genus_map)
        self.assertEqual(genus_map[0], "Alphafusarivirus")


if __name__ == "__main__":
    unittest.main()

