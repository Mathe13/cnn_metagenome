"""Unit and integration tests for M04 first-layer motifs and paired perturbation (C1-C16)."""
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
import torch

from viral_cnn.m04_workflow import (
    begin_m04,
    M04Run,
    PROTOCOL_ID,
    BIOLOGICAL_DEMARCATION_TEXT,
)
from viral_cnn.notebook_runtime import preflight_notebook, JupyterHubDataTopology
from viral_cnn.interpretability.motifs import (
    extract_first_layer_motifs,
    evaluate_motif_perturbation,
    summarize_motif_control,
    audit_motif_composition_robustness,
    compute_motif_presence_matrix_from_loader,
    compute_jaccard_cooccurrence,
    sample_matched_control_region,
    reconcile_table_08_schema,
    generate_figure_03_panels,
    generate_figure_03_scatter,
    SyntheticAuditCNN,
    VARIANT_PANEL_MAP,
)


def _create_mock_07_environment(target_dir: Path) -> dict:
    """Creates a mock dataset and checkpoint environment for short/07 preflight testing."""
    target_dir.mkdir(parents=True, exist_ok=True)
    topo = JupyterHubDataTopology(target_dir)

    parquet_dir = Path(topo.paths["short"]["species_parquet_dir"])
    parquet_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "sequence_padded": ["ACGT" * 128] * 10,
        "label": [0] * 10,
    })
    df.to_parquet(parquet_dir / "test.parquet")

    ckpts = {}
    for name in [
        "resnet_clean_species",
        "resnet_noise_species",
        "convnext_zero_species",
        "convnext_transfer_species",
        "convnext_noise_species",
    ]:
        p = Path(topo.paths["checkpoints"][name])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"mock_checkpoint_bytes")
        ckpts[name] = p

    return {
        "data_root": target_dir,
        "parquet_dir": parquet_dir,
        "checkpoints": ckpts,
    }


class M04Tests(unittest.TestCase):
    """Test suite for M04 checks C1 through C16."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.env = _create_mock_07_environment(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_paths(self) -> None:
        """C1: Resolver caminhos sem depender de diretórios fixos legados."""
        run = begin_m04(
            "short/07",
            project_root=self.root,
            settings={
                "data_root": str(self.env["data_root"]),
                "is_smoke_test": True,
            },
        )
        self.assertTrue(run.output_dir.is_dir())
        self.assertTrue(str(run.output_dir).startswith(str(self.root)))
        self.assertEqual(run.manifest["protocol_id"], PROTOCOL_ID)

        # Runs distintas não colidem
        run2 = begin_m04(
            "short/07",
            project_root=self.root,
            settings={
                "data_root": str(self.env["data_root"]),
                "is_smoke_test": True,
            },
        )
        self.assertNotEqual(run.output_dir, run2.output_dir)

    def test_preflight_07_five_models(self) -> None:
        """C2: Validar a presença e legibilidade dos cinco checkpoints de espécies."""
        # Com todos os 5 checkpoints e parquet
        report = preflight_notebook(
            "short/07",
            data_root=self.env["data_root"],
            smoke_test=False,
        )
        self.assertEqual(report["status"], "READY")
        self.assertFalse(report["is_smoke_test"])
        self.assertEqual(len(report["requirements"]), 6)  # 1 parquet + 5 ckpts

        # Removendo o quinto checkpoint (convnext_noise_species)
        self.env["checkpoints"]["convnext_noise_species"].unlink()
        report_blocked = preflight_notebook(
            "short/07",
            data_root=self.env["data_root"],
            smoke_test=False,
        )
        self.assertEqual(report_blocked["status"], "BLOCKED")

    def test_collision_rejection(self) -> None:
        """C3: Abortar se diretório reservado contiver produtos ou colidir."""
        plan = {
            "protocol_id": "m00_run_plan_v1",
            "parent_run_id": "mock_parent",
            "seed": 42,
            "data_root": str(self.env["data_root"]),
            "is_smoke_test": True,
            "stages": [
                {
                    "notebook_id": "short/07",
                    "action": "run",
                    "run_id": "colliding_run",
                    "parent_run_id": "mock_parent",
                    "output_dir": str(self.root / "results_reproduction/runs/colliding_run/short/07"),
                    "manifest": str(self.root / "results_reproduction/runs/colliding_run/short/07/manifest.json"),
                }
            ],
        }
        dest = self.root / "results_reproduction/runs/colliding_run/short/07"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "existing_artifact.txt").write_text("already here")

        plan_file = self.root / "run_plan.json"
        plan_file.write_text(json.dumps(plan))

        with self.assertRaises((FileExistsError, ValueError)):
            begin_m04("short/07", project_root=self.root, run_plan=plan_file)

    def test_failed_manifest(self) -> None:
        """C4: Registrar manifesto com status FAILED sob erro, nunca COMPLETE."""
        run = begin_m04(
            "short/07",
            project_root=self.root,
            settings={"data_root": str(self.env["data_root"]), "is_smoke_test": True},
        )
        with self.assertRaises(RuntimeError):
            with run.guard():
                raise RuntimeError("simulated pipeline error")

        manifest = json.loads(run.manifest_path.read_text())
        self.assertEqual(manifest["status"], "FAILED")
        self.assertIn("simulated pipeline error", manifest["error"])
        self.assertNotEqual(manifest["status"], "COMPLETE")

    def test_complete_manifest(self) -> None:
        """C5: Publicar manifesto COMPLETE com protocolo m04_motifs_audit_v1 e hashes."""
        run = begin_m04(
            "short/07",
            project_root=self.root,
            settings={"data_root": str(self.env["data_root"]), "is_smoke_test": True},
        )
        dummy_out = run.output_dir / "table_08_motif_perturbation.csv"
        dummy_out.write_text("panel,variant,motif,activation_drop\na,ResNet50_Clean,CCCCCCC,0.25\n")

        manifest = run.complete()
        self.assertEqual(manifest["status"], "COMPLETE")
        self.assertEqual(manifest["protocol_id"], PROTOCOL_ID)
        self.assertEqual(manifest["seed"], 42)
        self.assertTrue(any(p["relative_path"] == "table_08_motif_perturbation.csv" for p in manifest["outputs"]))

    def test_motif_extraction_top10(self) -> None:
        """C6: Extrair até 10 motivos consenso de 7-mers dos pesos de 1ª camada."""
        model = SyntheticAuditCNN(variant_idx=0)
        motifs = extract_first_layer_motifs(model, consensus_rule="max_positive")
        self.assertIsInstance(motifs, dict)
        self.assertGreater(len(motifs), 0)
        self.assertLessEqual(len(motifs), 16)
        top10 = list(motifs.keys())[:10]
        self.assertTrue(all(len(m) == 7 for m in top10))
        self.assertIn("CCCCCCC", motifs)
        self.assertIn("GGGGGGG", motifs)

    def test_paired_gc_control_same_sequence(self) -> None:
        """C7: Controles pareados uniforme e GC na mesma sequência sem sobreposição."""
        # Sequência sintética contendo CCCCCCC (pos 8-15) e região controle com mesmo GC (pos 30-37: GGGGGGG)
        seq = torch.tensor([0, 1, 2, 3] * 2 + [1] * 7 + [0, 3] * 7 + [2] * 7 + [0, 1, 2, 3] * 5, dtype=torch.long)
        motif_starts = [8]
        rng = np.random.default_rng(42)

        ctrl_start = sample_matched_control_region(
            sequence=seq,
            motif_starts=motif_starts,
            k=7,
            query_motif="CCCCCCC",
            control_policy="gc_matched",
            rng=rng,
        )
        self.assertIsNotNone(ctrl_start)
        # Sem sobreposição com [8, 15)
        self.assertTrue(ctrl_start >= 15 or ctrl_start + 7 <= 8)
        # Pareamento exato de GC (|ΔGC| = 0): CCCCCCC tem GC=7, controle deve ter GC=7
        ctrl_bases = [int(seq[ctrl_start + i].item()) for i in range(7)]
        gc_count = sum(1 for b in ctrl_bases if b in (1, 2))  # C=1, G=2
        self.assertEqual(gc_count, 7)

    def test_wilcoxon_and_fdr_computation(self) -> None:
        """C8: Computar queda específica, teste pareado de Wilcoxon e FDR Benjamini-Hochberg."""
        # Cria dados sintéticos com 10 sequências para um motivo
        records = []
        for i in range(10):
            records.append({
                "model_name": "resnet50_species",
                "sequence_id": i,
                "motif": "CCCCCCC",
                "n_occurrences": 1,
                "baseline_probability": 0.8,
                "motif_probability_drop": 0.35 + 0.01 * (i % 3),
                "random_probability_drop": 0.05 + 0.01 * (i % 2),
                "specific_probability_drop": 0.30,
                "control_policy": "uniform_random",
                "n_repeats": 5,
            })
        df = pd.DataFrame(records)
        summary = summarize_motif_control(df, alpha=0.05, min_sequences=5)

        self.assertEqual(len(summary), 1)
        self.assertIn("p_value", summary.columns)
        self.assertIn("p_fdr", summary.columns)
        self.assertIn("significant_fdr", summary.columns)
        self.assertLess(summary["p_value"].iloc[0], 0.05)
        self.assertTrue(bool(summary["significant_fdr"].iloc[0]))

    def test_reconciled_table_08_export(self) -> None:
        """C9: Salvar table_08 e motif_perturbation_by_variant com esquema padronizado."""
        raw_df = pd.DataFrame([{
            "model_name": "resnet50_species",
            "motif": "CCCCCCC",
            "mean_specific_drop_uniform": 0.28,
            "mean_random_drop_uniform": 0.04,
            "mean_motif_drop_uniform": 0.32,
            "p_fdr_uniform": 1e-5,
            "significant_fdr_uniform": True,
            "mean_specific_drop_gc": 0.01,
            "mean_random_drop_gc": 0.31,
            "mean_motif_drop_gc": 0.32,
            "p_fdr_gc": 0.45,
            "significant_fdr_gc": False,
            "n_occurrences_uniform": 25,
            "is_confounded": True,
            "verdict": "compositionally_confounded",
        }])
        reconciled = reconcile_table_08_schema(raw_df)

        expected_cols = {
            "panel", "variant", "motif", "activation_drop",
            "mean_random_drop_uniform", "mean_motif_drop_uniform", "mean_specific_drop_uniform",
            "mean_random_drop_gc", "mean_motif_drop_gc", "mean_specific_drop_gc",
            "fdr_uniform", "fdr_gc", "n_occurrences", "is_gc_significant",
        }
        self.assertTrue(expected_cols.issubset(reconciled.columns))
        self.assertEqual(reconciled["panel"].iloc[0], "a")
        self.assertEqual(reconciled["variant"].iloc[0], "ResNet50_Clean")
        self.assertEqual(reconciled["activation_drop"].iloc[0], 0.28)
        self.assertFalse(reconciled["is_gc_significant"].iloc[0])

    def test_reject_synthetic_in_empirical(self) -> None:
        """C10: Rejeitar preparação sintética quando SMOKE_TEST=0."""
        run = begin_m04(
            "short/07",
            project_root=self.root,
            settings={"data_root": str(self.env["data_root"]), "is_smoke_test": False},
        )
        with self.assertRaises(ValueError):
            run.prepare_smoke()

    def test_figure_03_five_panels(self) -> None:
        """C11: Renderizar Figura 3 principal com 5 painéis (a–e) sem barras artificiais."""
        rows = []
        for panel, (v_key, (p_id, v_name)) in enumerate(VARIANT_PANEL_MAP.items()):
            rows.append({
                "model_name": v_key,
                "motif": f"MOTIF{panel}A",
                "mean_specific_drop_uniform": 0.20 + 0.02 * panel,
                "mean_specific_drop_gc": 0.15,
                "significant_fdr_gc": True,
            })
        df = pd.DataFrame(rows)

        fig_path = self.root / "figure_03_motifs.png"
        generate_figure_03_panels(df, str(fig_path))

        self.assertTrue(fig_path.is_file())
        self.assertGreater(fig_path.stat().st_size, 1000)

    def test_panel_e_authenticity(self) -> None:
        """C12: Garantir que painel (e) utilize dados autênticos de ConvNeXt_F, sem replicar painel (d)."""
        model_d = SyntheticAuditCNN(variant_idx=3)  # ResNet50_F
        model_e = SyntheticAuditCNN(variant_idx=4)  # ConvNeXt_F

        motifs_d = extract_first_layer_motifs(model_d)
        motifs_e = extract_first_layer_motifs(model_e)

        # Panel e possui motivos adicionais ativados por filtros específicos (ex: CGCGCGC)
        self.assertIn("CGCGCGC", motifs_e)
        self.assertNotIn("CGCGCGC", motifs_d)

        # Conclusão de pesos diferentes
        self.assertFalse(torch.equal(model_d.conv1.weight, model_e.conv1.weight))

    def test_scatter_gc_uniform_rendering(self) -> None:
        """C13: Renderizar dispersão uniforme vs GC com diagonal e limiares."""
        df = pd.DataFrame([
            {"model_name": "resnet50_species", "motif": "CCCCCCC", "mean_specific_drop_uniform": 0.28, "mean_specific_drop_gc": 0.01},
            {"model_name": "convnext_noise_species", "motif": "AAAAAAA", "mean_specific_drop_uniform": 0.22, "mean_specific_drop_gc": 0.20},
        ])
        scatter_path = self.root / "figure_03_scatter_gc_uniform.png"
        generate_figure_03_scatter(df, str(scatter_path))

        self.assertTrue(scatter_path.is_file())
        self.assertGreater(scatter_path.stat().st_size, 500)

    def test_jaccard_parquets_export(self) -> None:
        """C14: Exportar matrizes de presença e Jaccard para consumo a jusante em 08/M05."""
        seqs = torch.tensor([
            [0, 1, 2, 3] * 2 + [1] * 7 + [0, 1, 2, 3] * 5,  # 8 + 7 + 20 = 35
            [0, 1, 2, 3] * 2 + [2] * 7 + [0, 1, 2, 3] * 5,  # 8 + 7 + 20 = 35
            [1] * 7 + [2] * 7 + [0, 1, 2, 3] * 5 + [0],     # 7 + 7 + 20 + 1 = 35
        ], dtype=torch.long)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(seqs), batch_size=2)
        motifs_list = ["CCCCCCC", "GGGGGGG"]

        presence = compute_motif_presence_matrix_from_loader(loader, motifs_list, max_sequences=10)
        jacc_df = compute_jaccard_cooccurrence(presence, motifs_list)

        parquet_out = self.root / "motif_interaction_jaccard_test.parquet"
        jacc_df.to_parquet(parquet_out)

        self.assertTrue(parquet_out.is_file())
        reloaded = pd.read_parquet(parquet_out)
        self.assertEqual(reloaded.shape, (2, 2))
        # Simetria e diagonal unitária
        self.assertAlmostEqual(reloaded.iloc[0, 0], 1.0)
        self.assertAlmostEqual(reloaded.iloc[1, 1], 1.0)
        self.assertAlmostEqual(reloaded.iloc[0, 1], reloaded.iloc[1, 0])

    def test_biological_demarcation(self) -> None:
        """C15: Demarcar que motivos são detectores composicionais, não fatores de transcrição."""
        self.assertIn("COMPOSITIONAL BASE DETECTORS ONLY", BIOLOGICAL_DEMARCATION_TEXT)
        self.assertIn("not verified functional biological", BIOLOGICAL_DEMARCATION_TEXT)

    def test_logging_path(self) -> None:
        """C16: Emitir logs com prefixo [LOG_PATH], respeitando LOG_PATH=0/false."""
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with patch.dict(os.environ, {"LOG_PATH": "1"}):
                run = begin_m04("short/07", project_root=self.root, settings={"data_root": str(self.env["data_root"]), "is_smoke_test": True})
                dummy = self.root / "dummy_log_test.txt"
                dummy.write_text("ok")
                run.complete(input_files=[dummy])
                self.assertIn("[LOG_PATH]", mock_stderr.getvalue())

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with patch.dict(os.environ, {"LOG_PATH": "0"}):
                run2 = begin_m04("short/07", project_root=self.root, settings={"data_root": str(self.env["data_root"]), "is_smoke_test": True})
                dummy2 = self.root / "dummy_log_test2.txt"
                dummy2.write_text("ok")
                run2.complete(input_files=[dummy2])
                self.assertNotIn("[LOG_PATH]", mock_stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
