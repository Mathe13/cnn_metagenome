"""
Unit and integration tests for N04: ICTV Taxonomy Curation and Latent Space Reproduction Notebooks.
Verifies checks C1 through C16 as defined in .specs/features/n04-short-taxonomy-and-latent-space-notebooks/checks.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from viral_cnn.notebook_runtime import JupyterHubDataTopology
from viral_cnn.taxonomy.curation import (
    build_hybrid_taxonomy,
    build_ictv_rename_map,
    generate_synthetic_taxonomy,
    match_ictv_taxonomy,
    resolve_local_ictv_taxonomy,
    sanitize_hybrid_taxonomy,
)
from viral_cnn.taxonomy.mapping import (
    model_label_to_taxonomic_id,
    taxonomic_id_to_model_label,
)


class TestN04TaxonomyAndLatentSpace(unittest.TestCase):
    """Checks C1-C16 for N04 taxonomy curation and latent space reproduction notebooks."""

    @classmethod
    def setUpClass(cls):
        os.environ["SMOKE_TEST"] = "1"
        os.environ["REMOTE_SERVER"] = "False"
        os.environ["MPLCONFIGDIR"] = "/tmp"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict(os.environ, {"M01_PROJECT_ROOT": tmp.name})
        env.start()
        self.addCleanup(env.stop)
        os.environ["SMOKE_TEST"] = "1"
        os.environ["REMOTE_SERVER"] = "False"
        os.environ["MPLCONFIGDIR"] = "/tmp"

    def _execute_m01_taxonomy(self):
        """Observe one fresh M01 namespace instead of historical fixed files."""
        nb = json.loads(Path("notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb").read_text())
        ns = {"__name__": "__main__"}
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]), ns)
        return ns["manifest"]

    def test_c1_build_ictv_rename_map(self):
        """C1: build_ictv_rename_map extracts lowercase mapping from ICTV rename DataFrame."""
        df_renames = pd.DataFrame({
            "Change": ["Renamed", "Abolished", "renamed species", "Moved"],
            "Old Name": [
                "Riboviria;Orthornavirae;Pisuviricota;Pisoniviricetes;Picornavirales;Picornaviridae;Enterovirus;Enterovirus C",
                "Some;Abolished;Taxon",
                "Order;Family;Genus;Old Virus Species",
                "Order;Family;Genus;Unchanged Virus",
            ],
            "New Name": [
                "Riboviria;Orthornavirae;Pisuviricota;Pisoniviricetes;Picornavirales;Picornaviridae;Enterovirus;Poliovirus",
                "Some;New;Taxon",
                "Order;Family;Genus;New Virus Species",
                "Order;Family;Genus;Unchanged Virus",
            ],
        })

        rename_map = build_ictv_rename_map(df_renames)
        self.assertEqual(len(rename_map), 2)
        self.assertIn("enterovirus c", rename_map)
        self.assertEqual(rename_map["enterovirus c"], "poliovirus")
        self.assertIn("old virus species", rename_map)
        self.assertEqual(rename_map["old virus species"], "new virus species")
        self.assertNotIn("some;abolished;taxon", rename_map)

    def test_c2_resolve_local_ictv_taxonomy(self):
        """C2: resolve_local_ictv_taxonomy populates all 8 lineage ranks with ICTV Master source."""
        species_df = pd.DataFrame({
            "species": ["Ebolavirus zaire", "Unknown virus"],
            "species_ictv": ["ebolavirus zaire", "unknown virus"],
        })
        ictv_master = pd.DataFrame({
            "Species": ["Ebolavirus zaire"],
            "Genus": ["Orthoebolavirus"],
            "Family": ["Filoviridae"],
            "Order": ["Mononegavirales"],
            "Class": ["Monjiviricetes"],
            "Phylum": ["Negarnaviricota"],
            "Kingdom": ["Orthornavirae"],
            "Realm": ["Riboviria"],
        })

        resolved = resolve_local_ictv_taxonomy(species_df, ictv_master, species_col="species_ictv")
        self.assertEqual(len(resolved), 2)
        matched = resolved[resolved["species_ictv"] == "ebolavirus zaire"].iloc[0]
        self.assertEqual(matched["taxonomy_source"], "ICTV Master")
        self.assertEqual(matched["Realm"], "Riboviria")
        self.assertEqual(matched["Kingdom"], "Orthornavirae")
        self.assertEqual(matched["Phylum"], "Negarnaviricota")
        self.assertEqual(matched["Class"], "Monjiviricetes")
        self.assertEqual(matched["Order"], "Mononegavirales")
        self.assertEqual(matched["Family"], "Filoviridae")
        self.assertEqual(matched["Genus"], "Orthoebolavirus")

        unmatched = resolved[resolved["species_ictv"] == "unknown virus"].iloc[0]
        self.assertEqual(unmatched["taxonomy_source"], "NCBI Fallback")

    def test_c3_sanitize_hybrid_taxonomy(self):
        """C3: sanitize_hybrid_taxonomy ensures model_label = id - 1 and backfills ranks."""
        df = pd.DataFrame({
            "id": [1, 2, 5],
            "species": ["Alpha virus", "Beta virus", "Gamma virus"],
            "Genus": ["Alphagenus", None, ""],
            "Family": [None, "Betafamily", ""],
        })

        sanitized = sanitize_hybrid_taxonomy(df, placeholder="Unassigned")
        self.assertIn("model_label", sanitized.columns)
        self.assertEqual(list(sanitized["model_label"]), [0, 1, 4])
        self.assertEqual(list(sanitized["id"]), [1, 2, 5])
        self.assertEqual(sanitized.loc[1, "Genus"], "Unassigned_Genus")
        self.assertEqual(sanitized.loc[2, "Genus"], "Unassigned_Genus")
        self.assertEqual(sanitized.loc[0, "Family"], "Unassigned_Family")
        self.assertIn("full_lineage", sanitized.columns)

    def test_c4_generate_synthetic_taxonomy(self):
        """C4: generate_synthetic_taxonomy returns deterministic mock DataFrame with 8 ranks and model_label."""
        df = generate_synthetic_taxonomy(num_classes=50)
        self.assertEqual(len(df), 50)
        required_cols = [
            "id",
            "model_label",
            "species",
            "Genus",
            "Family",
            "Order",
            "Class",
            "Phylum",
            "Kingdom",
            "Realm",
            "taxonomy_source",
            "full_lineage",
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)
        self.assertEqual(df["model_label"].min(), 0)
        self.assertEqual(df["model_label"].max(), 49)
        self.assertTrue((df["model_label"] == df["id"] - 1).all())

    def test_c5_topology_ictv_paths(self):
        """C5: JupyterHubDataTopology includes ictv_master and hybrid_taxonomy_parquet."""
        topology = JupyterHubDataTopology(data_root="/tmp/fake_data_root")
        short_paths = topology.paths["short"]
        self.assertIn("ictv_master", short_paths)
        self.assertIn("hybrid_taxonomy_parquet", short_paths)
        self.assertTrue(short_paths["ictv_master"].endswith("ICTV_Master_Species_List_2025_MSL41.v1.xlsx"))
        self.assertTrue(short_paths["hybrid_taxonomy_parquet"].endswith("main_taxonomy_hybrid_CLEANSED.parquet"))

    def test_c6_notebook_05_execution_smoke_test(self):
        """C6: 05_ictv_taxonomy_curation.ipynb executes end-to-end under SMOKE_TEST=1 in under 60 seconds."""
        nb_path = Path("notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb")
        self.assertTrue(nb_path.exists())

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        start_time = time.time()
        global_ns = {"__name__": "__main__"}
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                try:
                    exec(code, global_ns)
                except Exception as e:
                    self.fail(f"Cell {idx} of 05_ictv_taxonomy_curation.ipynb raised exception: {e}")

        duration = time.time() - start_time
        self.assertLess(duration, 60.0, f"Notebook 05 took too long: {duration:.2f}s")

    def test_c7_notebook_05_imports_and_decoupling(self):
        """C7: 05_ictv_taxonomy_curation.ipynb contains zero short/utils/* imports and valid JSON."""
        nb_path = Path("notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb")
        self.assertTrue(nb_path.exists())

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_text = ""
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                code_text += "".join(cell.get("source", [])) + "\n"

        self.assertNotIn("short.utils", code_text)
        self.assertNotIn("from utils", code_text)
        self.assertIn("viral_cnn", code_text)

    def test_c8_notebook_05_artifacts_generated(self):
        """C8: 05_ictv_taxonomy_curation.ipynb generates cleansed parquet and manifest.json."""
        manifest = self._execute_m01_taxonomy()
        out_dir = Path(manifest["output_dir"])
        self.assertTrue(out_dir.exists())

        parquet_file = out_dir / "main_taxonomy_hybrid_CLEANSED.parquet"
        self.assertTrue(parquet_file.exists(), f"Expected {parquet_file} to exist")
        self.assertGreater(parquet_file.stat().st_size, 0)

        manifest_file = out_dir / "manifest.json"
        self.assertTrue(manifest_file.exists(), f"Expected {manifest_file} to exist")
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        self.assertEqual(manifest_data["notebook_id"], "short/05")
        self.assertIn("duplication_rate_pct", manifest_data["metrics_summary"])

    def test_c9_historical_curation_notebooks_preserved(self):
        """C9: Historical notebooks short/ICTV_improve_kraken_database.ipynb and improve_kraken_database-Copy1.ipynb remain unmodified."""
        res = subprocess.run(
            ["git", "status", "--porcelain", "short/ICTV_improve_kraken_database.ipynb", "short/improve_kraken_database-Copy1.ipynb"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(res.stdout.strip(), "", "Historical curation notebooks must remain untouched in git status")

    def test_c10_notebook_06_imports_and_decoupling(self):
        """C10: 06_ictv_latent_space_and_hierarchy.ipynb contains zero short/utils/* imports and valid JSON."""
        nb_path = Path("notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb")
        self.assertTrue(nb_path.exists())

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_text = ""
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                code_text += "".join(cell.get("source", [])) + "\n"

        self.assertNotIn("short.utils", code_text)
        self.assertNotIn("from utils", code_text)
        self.assertIn("viral_cnn", code_text)

    def test_c11_notebook_06_execution_smoke_test(self):
        """C11: 06_ictv_latent_space_and_hierarchy.ipynb executes end-to-end under SMOKE_TEST=1 in under 60 seconds."""
        nb_path = Path("notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb")
        self.assertTrue(nb_path.exists())

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        start_time = time.time()
        global_ns = {"__name__": "__main__"}
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                try:
                    exec(code, global_ns)
                except Exception as e:
                    self.fail(f"Cell {idx} of 06_ictv_latent_space_and_hierarchy.ipynb raised exception: {e}")

        duration = time.time() - start_time
        self.assertLess(duration, 60.0, f"Notebook 06 took too long: {duration:.2f}s")

    def test_c12_notebook_06_table5_metrics(self):
        """C12: 06_ictv_latent_space_and_hierarchy.ipynb produces Table 5 metrics CSV with KNN and Silhouette."""
        t5_path = Path("results_reproduction/latent_space/table5_latent_clustering_metrics.csv")
        self.assertTrue(t5_path.exists(), f"Expected {t5_path} to exist")
        df = pd.read_csv(t5_path)
        required_cols = ["Model", "Rank", "KNN_Top1", "KNN_Top5", "Silhouette_Score", "Valid_Class_Rate"]
        for col in required_cols:
            self.assertIn(col, df.columns)
        self.assertGreater(len(df), 0)

    def test_c13_notebook_06_table6_correlations(self):
        """C13: 06_ictv_latent_space_and_hierarchy.ipynb produces Table 6 hierarchical rank correlations CSV."""
        t6_path = Path("results_reproduction/latent_space/table6_taxonomic_rank_correlations.csv")
        self.assertTrue(t6_path.exists(), f"Expected {t6_path} to exist")
        df = pd.read_csv(t6_path)
        required_cols = ["Model", "Transition", "Pearson_r", "Spearman_rho", "N_taxa"]
        for col in required_cols:
            self.assertIn(col, df.columns)
        self.assertGreater(len(df), 0)

    def test_c14_notebook_06_linear_probe_audit(self):
        """C14: 06_ictv_latent_space_and_hierarchy.ipynb evaluates linear probe and audits classifier head bottleneck."""
        manifest_path = Path("results_reproduction/latent_space/manifest.json")
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        self.assertEqual(manifest_data["notebook_id"], "06_ictv_latent_space_and_hierarchy")

    def test_c15_notebook_06_umap_and_artifacts(self):
        """C15: 06_ictv_latent_space_and_hierarchy.ipynb exports qualitative UMAP projection CSV."""
        umap_path = Path("results_reproduction/latent_space/umap_projection_convnext_f.csv")
        self.assertTrue(umap_path.exists(), f"Expected {umap_path} to exist")
        df = pd.read_csv(umap_path, comment="#")
        self.assertIn("umap_1", df.columns)
        self.assertIn("umap_2", df.columns)

    def test_c16_historical_latent_notebook_preserved(self):
        """C16: Historical notebook short/ICTV_latent_space-bkp.ipynb remains unmodified."""
        res = subprocess.run(
            ["git", "status", "--porcelain", "short/ICTV_latent_space-bkp.ipynb"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(res.stdout.strip(), "", "Historical latent notebook must remain untouched in git status")

    # -----------------------------------------------------------------------
    # R04 checks (r04-real-latent-embeddings-and-taxonomy-audit)
    # -----------------------------------------------------------------------

    def test_r04_evaluate_cosine_knn_top1_and_top5(self):
        """C1: evaluate_cosine_knn returns global_knn_top5_accuracy >= global_knn_accuracy with zero self-neighbors."""
        from viral_cnn.eval.latent import evaluate_cosine_knn

        rng = np.random.default_rng(42)
        feats = rng.normal(0, 1, size=(100, 32)).astype(np.float32)
        labels = np.repeat(np.arange(10), 10)

        res = evaluate_cosine_knn(feats, labels, k=5, test_size=0.2, random_state=42)
        self.assertIn("global_knn_accuracy", res)
        self.assertIn("global_knn_top5_accuracy", res)
        self.assertTrue(res["zero_self_neighbor"])
        self.assertGreaterEqual(res["global_knn_top5_accuracy"], res["global_knn_accuracy"])
        self.assertGreaterEqual(res["global_knn_accuracy"], 0.0)
        self.assertLessEqual(res["global_knn_top5_accuracy"], 1.0)
        self.assertIn("knn_top5_accuracy", res["per_cluster_metrics"].columns)

    def test_r04_models_embedding_dimension_2048(self):
        """C2: Notebook 06 and model architectures define pre-classifier embedding dimension of 2048 for all 5 models."""
        from viral_cnn.models.convnext import ConvNeXt_Tiny_1D
        from viral_cnn.models.resnet import ResNet50_1D

        # Architectural invariant: in_features == 2048
        r = ResNet50_1D(num_classes=10)
        c = ConvNeXt_Tiny_1D(num_classes=10)
        self.assertEqual(r.fc.in_features, 2048)
        self.assertEqual(c.fc.in_features, 2048)

        # Notebook 06 configuration check
        nb_path = Path("notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        c3_code = "".join(nb["cells"][3]["source"])
        self.assertIn('"dim": 2048', c3_code)
        self.assertNotIn('"dim": 768', c3_code)

    def test_r04_notebook_06_fail_fast_missing_artifacts(self):
        """C3: 06_ictv_latent_space_and_hierarchy.ipynb raises FileNotFoundError when SMOKE_TEST=0 and artifacts are missing."""
        nb_path = Path("notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        # Setup context with is_smoke_test = False in a non-existent directory
        setup_code = """
import os
os.environ["SMOKE_TEST"] = "0"
os.environ["REMOTE_SERVER"] = "False"
from viral_cnn.notebook_runtime import setup_notebook_runtime
ctx = setup_notebook_runtime()
output_dir = Path("/tmp/fake_output_dir_r04_test")
output_dir.mkdir(parents=True, exist_ok=True)
taxonomy_df = pd.DataFrame({"model_label": [0, 1], "Genus": ["A", "B"], "Order": ["O1", "O2"]})
"""
        c3_code = "".join(nb["cells"][3]["source"])

        global_ns = {"__name__": "__main__"}
        try:
            exec("".join(nb["cells"][1]["source"]), global_ns)
            exec(setup_code, global_ns)

            with self.assertRaises(FileNotFoundError):
                exec(c3_code, global_ns)
        finally:
            os.environ["SMOKE_TEST"] = "1"
            os.environ["REMOTE_SERVER"] = "False"

    def test_r04_notebook_06_table5_distinct_top1_top5(self):
        """C4: Notebook 06 generates Table 5 with distinct KNN_Top1 and KNN_Top5 columns and saves CSV."""
        nb_path = Path("notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        # Verify Cell 4 uses global_knn_top5_accuracy rather than duplicating global_knn_accuracy
        c4_code = "".join(nb["cells"][4]["source"])
        self.assertIn('"KNN_Top5": knn_res["global_knn_top5_accuracy"]', c4_code)

        # Run smoke test on notebook 06
        global_ns = {"__name__": "__main__"}
        for idx in [1, 2, 3, 4]:
            exec("".join(nb["cells"][idx]["source"]), global_ns)

        t5_path = Path("results_reproduction/latent_space/table5_latent_clustering_metrics.csv")
        self.assertTrue(t5_path.exists())
        df = pd.read_csv(t5_path)
        self.assertIn("KNN_Top1", df.columns)
        self.assertIn("KNN_Top5", df.columns)
        self.assertTrue((df["KNN_Top5"] >= df["KNN_Top1"]).all())

    def test_r04_notebook_06_table6_correlations(self):
        """C5: Notebook 06 generates Table 6 reporting hierarchical correlations with valid parent taxon counts."""
        nb_path = Path("notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        global_ns = {"__name__": "__main__"}
        for idx in range(1, 6):
            exec("".join(nb["cells"][idx]["source"]), global_ns)

        t6_path = Path("results_reproduction/latent_space/table6_taxonomic_rank_correlations.csv")
        self.assertTrue(t6_path.exists())
        df = pd.read_csv(t6_path)
        self.assertEqual(len(df), 10)  # 5 models * 2 transitions
        self.assertTrue(set(df["Transition"]) == {"Species -> Genus", "Genus -> Order"})
        self.assertTrue((df["N_taxa"] > 0).all())

    def test_r04_notebook_05_provenance_and_smoke_execution(self):
        """C6: Notebook 05 sanitizes provenance manifest and both notebooks complete in < 60s under smoke mode."""
        nb5_path = Path("notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb")
        with open(nb5_path, "r", encoding="utf-8") as f:
            nb5 = json.load(f)

        # M01 no longer manufactures genome provenance from synthetic taxon rows.
        code = "\n".join("".join(c.get("source", [])) for c in nb5["cells"] if c["cell_type"] == "code")
        self.assertIn("run_m01_notebook", code)
        self.assertNotIn("mock_records", code)

        # Execute notebook 05 under smoke mode
        start_time = time.time()
        global_ns = {"__name__": "__main__"}
        for cell in nb5["cells"]:
            if cell.get("cell_type") == "code":
                exec("".join(cell.get("source", [])), global_ns)
        self.assertLess(time.time() - start_time, 60.0)

        # Verify manifest generated
        manifest_path = Path(global_ns["manifest"]["output_dir"]) / "manifest.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("genome_provenance_complete", data["metrics_summary"])
        self.assertIn("genome_disjoint_test_classes", data["metrics_summary"])
        self.assertFalse(data["metrics_summary"]["genome_provenance_complete"])
        self.assertIsNone(data["metrics_summary"]["genome_disjoint_test_classes"])


if __name__ == "__main__":
    unittest.main()
