"""Unit and integration tests for M03 latent space and visualizations (C1-C16)."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from viral_cnn.m03_workflow import begin_m03, M03Run, PROTOCOL_ID, DISCLAIMER_TEXT
from viral_cnn.notebook_runtime import preflight_notebook, JupyterHubDataTopology
from viral_cnn.paired_umap import (
    MODEL_NAMES,
    ProjectionConfig,
    select_paired_sample,
    run_paired_umap,
    _selective_panel,
)
from viral_cnn.latent_plotting import (
    load_saved_comparison,
    render_comparison,
    render_quantitative_summary,
)
from viral_cnn.latent_resume import LatentResumeCache


def _create_mock_latent_source(target_dir: Path, n_samples: int = 20) -> dict:
    """Create a mock valid latent source output directory for testing."""
    target_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    row_ids = [f"seq_{i:04d}" for i in range(n_samples)]
    labels = [i % 4 for i in range(n_samples)]
    genera = [f"Genus_{i % 3}" for i in range(n_samples)]

    colors = {"Genus_0": "#1f77b4", "Genus_1": "#ff7f0e", "Genus_2": "#2ca02c"}
    projections = {}

    for name in MODEL_NAMES:
        df = pd.DataFrame({
            "umap_1": np.linspace(0, 1, n_samples) + (0.1 if "convnext" in name else 0.0),
            "umap_2": np.linspace(1, 0, n_samples),
            "label": labels,
            "row_id": row_ids,
            "Genus": genera,
            "projection_method": "PCA",
            "is_qualitative_only": True,
        })
        csv_filename = "umap_projection_convnext_f.csv" if name == "convnext_noise_species" else f"umap_projection_{name}.csv"
        csv_path = target_dir / csv_filename
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("# QUALITATIVE ONLY\n# Do not infer phylogenetic distance\n")
            df.to_csv(f, index=False)
        frames[name] = df
        import hashlib
        projections[name] = {
            "path": str(csv_path.resolve()),
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        }

    selection_df = pd.DataFrame({"row_id": row_ids, "label": labels, "Genus": genera})
    selection_path = target_dir / "umap_sample_selection.csv"
    selection_df.to_csv(selection_path, index=False)
    import hashlib
    selection_sha = hashlib.sha256(json.dumps(selection_df.to_dict(orient="records"), sort_keys=True).encode()).hexdigest()

    meta = {
        "protocol": "c10_paired_umap_v1",
        "status": "COMPLETE",
        "projection_method": "PCA",
        "parameters": {"n_components": 2},
        "selection": {
            "selected_count": n_samples,
            "selected_genera": ["Genus_0", "Genus_1", "Genus_2"],
            "configuration": {"max_samples": n_samples, "max_genera": 3, "seed": 42},
        },
        "selection_sha256": selection_sha,
        "sample_file": {"path": str(selection_path), "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest()},
        "projections": projections,
        "genus_colors": colors,
        "is_qualitative_only": True,
    }
    (target_dir / "umap_comparison.json").write_text(json.dumps(meta, indent=2) + "\n")
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "notebook_id": "short/06",
        "run_id": "mock_run_06",
        "is_smoke_test": False,
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"meta": meta, "frames": frames, "dir": target_dir}


class M03Tests(unittest.TestCase):
    """Test suite for M03 checks C1 through C16."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_paths(self):
        """C1: Resolver caminhos e parâmetros a partir de config/env sem caminhos fixos de usuário."""
        with patch.dict(os.environ, {"DATA_ROOT": str(self.root), "SMOKE_TEST": "1"}):
            run06 = begin_m03("short/06", project_root=self.root)
            self.assertEqual(run06.ctx.data_root, self.root / run06.output_dir.relative_to(self.root) / "fixtures")
            self.assertTrue(run06.output_dir.is_dir())
            self.assertIn("short/06", str(run06.output_dir))

            run06b = begin_m03("short/06b", project_root=self.root)
            self.assertTrue(run06b.output_dir.is_dir())
            self.assertIn("short/06b", str(run06b.output_dir))
            self.assertNotEqual(run06.output_dir, run06b.output_dir)

    def test_preflight_06b(self):
        """C2: preflight_notebook("short/06b") valida os requisitos e reporta READY ou BLOCKED."""
        # Case 1: no LATENT_SOURCE_DIR and smoke_test=False -> BLOCKED
        with patch.dict(os.environ, {}, clear=True):
            report = preflight_notebook("short/06b", self.root, smoke_test=False)
            self.assertEqual(report["status"], "BLOCKED")

        # Case 2: valid source directory -> READY
        source_dir = self.root / "mock_latent"
        _create_mock_latent_source(source_dir)
        with patch.dict(os.environ, {"LATENT_SOURCE_DIR": str(source_dir)}):
            report = preflight_notebook("short/06b", self.root, smoke_test=False)
            self.assertEqual(report["status"], "READY")

        # Case 3: smoke_test=True -> READY
        with patch.dict(os.environ, {}, clear=True):
            report_smoke = preflight_notebook("short/06b", self.root, smoke_test=True)
            self.assertEqual(report_smoke["status"], "READY")

    def test_collision_rejection(self):
        """C3: Rejeitar destino que já contenha produtos ou colisão com plano M00."""
        run = begin_m03("short/06", project_root=self.root, settings={"is_smoke_test": True})
        # Simulate an existing file in output dir
        (run.output_dir / "existing_artifact.txt").write_text("already exists")

        # Attempt to run with an M00 plan pointing to non-empty directory
        plan = {
            "protocol_id": "m00_run_plan_v1",
            "data_root": str(self.root),
            "is_smoke_test": True,
            "parent_run_id": "p0",
            "seed": 42,
            "stages": [{
                "notebook_id": "short/06",
                "action": "run",
                "run_id": run.run_id,
                "parent_run_id": "p0",
                "output_dir": str(run.output_dir),
                "manifest": str(run.output_dir / "manifest.json"),
            }],
        }
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(plan))

        with self.assertRaises((FileExistsError, ValueError)):
            begin_m03("short/06", project_root=self.root, run_plan=plan_path)

    def test_failed_manifest(self):
        """C4: Falha por exceção registra manifesto atômico FAILED e nunca COMPLETE."""
        run = begin_m03("short/06", project_root=self.root, settings={"is_smoke_test": True})
        with self.assertRaises(RuntimeError):
            with run.guard():
                raise RuntimeError("Simulated crash in feature extraction")

        manifest = json.loads(run.manifest_path.read_text())
        self.assertEqual(manifest["status"], "FAILED")
        self.assertIn("Simulated crash", manifest.get("error", ""))
        self.assertNotEqual(manifest["status"], "COMPLETE")

    def test_complete_manifest(self):
        """C5: Conclusão com sucesso publica manifesto individual COMPLETE com protocolo e hashes."""
        run = begin_m03("short/06", project_root=self.root, settings={"is_smoke_test": True})
        dummy_in = self.root / "input.csv"
        dummy_in.write_text("dummy,input")
        dummy_out = run.output_dir / "output.csv"
        dummy_out.write_text("dummy,output")

        manifest = run.complete(input_files=[dummy_in])
        self.assertEqual(manifest["status"], "COMPLETE")
        self.assertEqual(manifest["protocol_id"], PROTOCOL_ID)
        self.assertEqual(manifest["seed"], 42)
        self.assertTrue(manifest["is_smoke_test"])
        self.assertEqual(len(manifest["inputs"]), 1)
        self.assertEqual(len(manifest["outputs"]), 1)
        self.assertEqual(manifest["outputs"][0]["relative_path"], "output.csv")

    def test_c09_reuse_validation(self):
        """C6: Modo reuse valida integridade dos embeddings via SHA-256 e C09."""
        runs_dir = self.root / "results_reproduction" / "runs"
        old_run = runs_dir / "old_run" / "06_ictv_latent_space_and_hierarchy"
        new_run = runs_dir / "new_run" / "06_ictv_latent_space_and_hierarchy"
        old_run.mkdir(parents=True, exist_ok=True)
        new_run.mkdir(parents=True, exist_ok=True)

        producer = LatentResumeCache(old_run, mode="rerun")
        identity = {"seed": 42, "stage": "knn"}
        producer.run_stage("knn", identity, lambda: {"accuracy": 0.85})

        consumer = LatentResumeCache(new_run, mode="reuse", reuse_run_id="old_run")
        reused = consumer.run_stage("knn", identity, lambda: self.fail("recalculated"))
        self.assertEqual(reused, {"accuracy": 0.85})

        # Corrupt file in old_run -> consumer detects mismatch
        for f in old_run.glob("stage_knn_*.pt"):
            f.write_bytes(b"corrupted binary payload")
        another_run = runs_dir / "another_run" / "06_ictv_latent_space_and_hierarchy"
        another_run.mkdir(parents=True, exist_ok=True)
        consumer_corrupt = LatentResumeCache(another_run, mode="reuse", reuse_run_id="old_run")
        # Reuse should detect corrupted file and recompute via compute callback
        recomputed = consumer_corrupt.run_stage("knn", identity, lambda: {"accuracy": 0.99, "recomputed": True})
        self.assertEqual(recomputed.get("accuracy"), 0.99)

    def test_reject_synthetic_in_empirical(self):
        """C7: Modo empírico (SMOKE_TEST=0) rejeita geração ou dependência de fixture sintética."""
        run = begin_m03("short/06", project_root=self.root, settings={"is_smoke_test": False, "data_root": str(self.root)})
        with self.assertRaises(ValueError):
            run.prepare_smoke()

    def test_original_space_metrics(self):
        """C8: KNN, Silhouette e Sonda operam no espaço de alta dimensão sem redução por UMAP."""
        from viral_cnn.eval.latent import evaluate_cosine_knn, compute_cosine_silhouette_fast
        rng = np.random.default_rng(42)
        dim = 128
        n_samples = 60
        features = rng.normal(size=(n_samples, dim)).astype(np.float32)
        labels = np.array([0] * 30 + [1] * 30, dtype=np.int64)

        # High-dim KNN cosine
        res = evaluate_cosine_knn(features, labels, k=5, random_state=42)
        self.assertIn("global_knn_accuracy", res)
        acc = res["global_knn_accuracy"]
        self.assertIsInstance(acc, float)
        self.assertTrue(0.0 <= acc <= 1.0)

        # High-dim Silhouette
        sil, _ = compute_cosine_silhouette_fast(features, labels)
        self.assertIsInstance(sil, float)
        self.assertTrue(-1.0 <= sil <= 1.0)

    def test_quantitative_summary_artifacts(self):
        """C9: Gerar latent_quantitative_summary.csv e .png confrontando os cinco modelos."""
        out_dir = self.root / "quant_summary"
        out_dir.mkdir(parents=True, exist_ok=True)
        t5_records = [
            {"Model": "ResNet50", "Rank": "Species", "KNN_Accuracy": 0.85, "Silhouette": 0.12},
            {"Model": "ResNet50", "Rank": "Genus", "KNN_Accuracy": 0.92, "Silhouette": 0.18},
            {"Model": "ConvNeXt_F", "Rank": "Species", "KNN_Accuracy": 0.88, "Silhouette": 0.15},
            {"Model": "ConvNeXt_F", "Rank": "Genus", "KNN_Accuracy": 0.94, "Silhouette": 0.20},
        ]
        probe_records = [
            {"Model": "ResNet50", "Linear_Probe_Test_Acc": 0.84, "Native_Head_Accuracy": 0.85},
            {"Model": "ConvNeXt_F", "Linear_Probe_Test_Acc": 0.87, "Native_Head_Accuracy": 0.88},
        ]
        csv_p, png_p = render_quantitative_summary(t5_records, probe_records, out_dir)
        self.assertTrue(csv_p.is_file())
        self.assertTrue(png_p.is_file())

        df = pd.read_csv(csv_p)
        self.assertIn("Model", df.columns)
        self.assertIn("KNN_Species_Acc", df.columns)
        self.assertEqual(len(df), 2)

    def test_paired_selection_identity(self):
        """C10: Amostragem pareada C10 seleciona as mesmas sequências em todos os modelos."""
        n_samples = 40
        labels_base = np.array([i % 4 for i in range(n_samples)], dtype=np.int64)
        row_ids_base = np.array([f"seq_{i:04d}" for i in range(n_samples)], dtype=object)
        species_to_genus = {0: "Alpha", 1: "Beta", 2: "Gamma", 3: "Delta"}

        labels = {m: labels_base.copy() for m in MODEL_NAMES}
        row_ids = {m: row_ids_base.copy() for m in MODEL_NAMES}

        sample = select_paired_sample(labels, row_ids, species_to_genus, ProjectionConfig(max_samples=16, max_genera=4, seed=42))
        self.assertEqual(len(sample.frame), 16)
        self.assertTrue(sample.frame["row_id"].is_unique)

        # Every model's mapped indices must yield the identical selection
        for name in MODEL_NAMES:
            pos = sample.indices[name]
            self.assertTrue(np.array_equal(row_ids[name][pos], sample.frame["row_id"].to_numpy()))
            self.assertTrue(np.array_equal(labels[name][pos], sample.frame["label"].to_numpy()))

    def test_exported_selection_csvs(self):
        """C11: Exportar selected_genera.csv e sampled_query_ids.csv com listas completas e status de destaque."""
        out_dir = self.root / "umap_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        resume = LatentResumeCache(out_dir, mode="rerun")

        n_samples = 30
        labels_base = np.array([i % 3 for i in range(n_samples)], dtype=np.int64)
        row_ids_base = np.array([f"seq_{i:04d}" for i in range(n_samples)], dtype=object)
        species_to_genus = {0: "Alpha", 1: "Beta", 2: "Gamma"}

        model_specs = [{"name": m, "display": m, "test_artifact_sha256": "sha_" + m} for m in MODEL_NAMES]
        features = {m: np.random.randn(n_samples, 16).astype(np.float32) for m in MODEL_NAMES}
        labels = {m: labels_base for m in MODEL_NAMES}
        row_ids = {m: row_ids_base for m in MODEL_NAMES}

        res = run_paired_umap(
            model_specs, features, labels, row_ids, species_to_genus,
            resume, {"env": "test"}, config=ProjectionConfig(max_samples=15, max_genera=3, seed=42)
        )

        gen_csv = out_dir / "selected_genera.csv"
        qry_csv = out_dir / "sampled_query_ids.csv"
        self.assertTrue(gen_csv.is_file())
        self.assertTrue(qry_csv.is_file())

        df_gen = pd.read_csv(gen_csv)
        self.assertIn("Genus", df_gen.columns)
        self.assertIn("is_highlighted", df_gen.columns)
        self.assertIn("sample_count", df_gen.columns)

        df_qry = pd.read_csv(qry_csv)
        self.assertIn("query_id", df_qry.columns)
        self.assertIn("label", df_qry.columns)
        self.assertIn("Genus", df_qry.columns)
        self.assertIn("is_highlighted", df_qry.columns)

    def test_selective_panel_rendering(self):
        """C12: Painel seletivo destaca gêneros escolhidos e renderiza fundo neutro."""
        out_dir = self.root / "sel_panel"
        out_dir.mkdir(parents=True, exist_ok=True)
        panel_path = out_dir / "umap_selective_panel.png"

        frames = {}
        n = 20
        for name in MODEL_NAMES:
            frames[name] = pd.DataFrame({
                "umap_1": np.random.randn(n),
                "umap_2": np.random.randn(n),
                "Genus": ["Genus_A"] * 10 + ["Genus_B"] * 10,
            })
        names = {m: m for m in MODEL_NAMES}
        colors = {"Genus_A": "#ff0000", "Genus_B": "#00ff00"}

        _selective_panel(frames, names, "PCA", panel_path, {"Genus_A"}, colors)
        self.assertTrue(panel_path.is_file())
        self.assertGreater(panel_path.stat().st_size, 0)

    def test_qualitative_disclaimers(self):
        """C13: Gráficos e arquivos UMAP/PCA incluem aviso proeminente de caráter qualitativo."""
        self.assertIn("QUALITATIVE ONLY", DISCLAIMER_TEXT)
        self.assertIn("phylogenetic", DISCLAIMER_TEXT)

        # Check in paired_umap DISCLAIMER header
        from viral_cnn.paired_umap import DISCLAIMER
        self.assertIn("QUALITATIVE ONLY", DISCLAIMER)

    def test_06b_regeneration_preserves_coordinates(self):
        """C14: 06b regenera painel completo, seletivo e HTML preservando estritamente as coordenadas salvas."""
        mock = _create_mock_latent_source(self.root / "src_06")
        out_06b = self.root / "out_06b"
        out_06b.mkdir(parents=True, exist_ok=True)

        comparison, inputs = load_saved_comparison(mock["dir"], out_06b)
        rendering = render_comparison(comparison, out_06b)

        # Verify regenerated files exist
        self.assertTrue((out_06b / "umap_comparison.png").is_file())
        self.assertTrue((out_06b / "umap_selective_panel.png").is_file())
        self.assertTrue((out_06b / "umap_comparison.html").is_file())
        self.assertTrue((out_06b / "selected_genera.csv").is_file())
        self.assertTrue((out_06b / "sampled_query_ids.csv").is_file())

        # Verify coordinates are unchanged
        orig_df = mock["frames"][MODEL_NAMES[0]]
        reloaded_df = pd.read_csv(out_06b / f"umap_projection_{MODEL_NAMES[0]}.csv", comment="#")
        np.testing.assert_allclose(orig_df["umap_1"].to_numpy(), reloaded_df["umap_1"].to_numpy())
        np.testing.assert_allclose(orig_df["umap_2"].to_numpy(), reloaded_df["umap_2"].to_numpy())

    def test_logging_path(self):
        """C15: Artefatos lidos ou gravados emitem log com prefixo [LOG_PATH], respeitando LOG_PATH=0."""
        from viral_cnn.utils.logging_utils import log_artifact_path

        # With LOG_PATH=1 (default)
        with patch.dict(os.environ, {"LOG_PATH": "1"}):
            with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
                log_artifact_path("Test artifact", self.root / "file.txt")
                self.assertIn("[LOG_PATH]", mock_stderr.getvalue())

        # With LOG_PATH=0
        with patch.dict(os.environ, {"LOG_PATH": "0"}):
            with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
                log_artifact_path("Test artifact", self.root / "file.txt")
                self.assertNotIn("[LOG_PATH]", mock_stderr.getvalue())

    def test_documentation_and_limits(self):
        """C16: docs/m03_latent_space_visualizations.md documenta comandos, preflight e limites qualitativos."""
        doc_path = Path("docs/m03_latent_space_visualizations.md")
        self.assertTrue(doc_path.is_file(), "m03_latent_space_visualizations.md documentation missing")
        content = doc_path.read_text()
        self.assertIn("preflight_notebook", content)
        self.assertIn("short/06b", content)
        self.assertIn("m03_latent_audit_v1", content)
        self.assertIn("qualitativa", content.lower())
        self.assertIn("filogenética", content.lower())


if __name__ == "__main__":
    unittest.main()
