"""Unit and integration tests for M05 exploratory triplet composition and cooccurrence (C1-C17)."""
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

from viral_cnn.m05_workflow import (
    begin_m05,
    M05Run,
    sample_deterministic_test_sequences,
    generate_network_edge_table,
    render_figure_04_rscu,
    render_figure_05_network,
    render_figure_05_cooccurrence_synergy,
    render_figure_05_permutation_null,
    export_reconciled_table_11b_diagnostics,
    PROTOCOL_ID,
    BIOLOGICAL_DEMARCATION_TEXT,
)
from viral_cnn.notebook_runtime import preflight_notebook, JupyterHubDataTopology
from viral_cnn.interpretability.motifs import (
    compute_exploratory_triplet_rscu,
    audit_translation_boundary_signals,
    analyze_cooccurrence_pairs,
    compute_jaccard_cooccurrence,
)


def _create_mock_08_environment(target_dir: Path) -> dict:
    """Creates a mock dataset and 07 output environment for short/08 testing."""
    target_dir.mkdir(parents=True, exist_ok=True)
    topo = JupyterHubDataTopology(target_dir)

    parquet_dir = Path(topo.paths["short"]["species_parquet_dir"])
    parquet_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "sequence_padded": ["ACGT" * 128] * 10,
        "label": [0] * 10,
    })
    df.to_parquet(parquet_dir / "test.parquet")

    motifs_07_dir = target_dir / "results_reproduction" / "motifs"
    motifs_07_dir.mkdir(parents=True, exist_ok=True)

    manifest_07 = {
        "protocol_id": "m04_motifs_audit_v1",
        "status": "COMPLETE",
        "is_smoke_test": False,
    }
    with open(motifs_07_dir / "manifest_07_motifs.json", "w") as f:
        json.dump(manifest_07, f)

    motifs_df = pd.DataFrame({
        "panel": ["a", "a", "b", "b"],
        "variant": ["resnet50_species", "resnet50_species", "convnext_zero_species", "convnext_zero_species"],
        "motif": ["CCCCCCC", "GGGGGGG", "CCCCCCC", "ATATATA"],
    })
    motifs_df.to_parquet(motifs_07_dir / "motif_summary.parquet")

    variants = [
        "resnet50_species",
        "resnet_noise_species",
        "convnext_zero_species",
        "convnext_transfer_species",
        "convnext_noise_species",
    ]
    motifs_list = ["CCCCCCC", "GGGGGGG", "ATATATA", "TTTTTTT", "AAAAAAA"]
    for v in variants:
        jaccard_data = np.array([
            [1.0, 0.4, 0.2, 0.1, 0.0],
            [0.4, 1.0, 0.3, 0.0, 0.2],
            [0.2, 0.3, 1.0, 0.5, 0.1],
            [0.1, 0.0, 0.5, 1.0, 0.4],
            [0.0, 0.2, 0.1, 0.4, 1.0],
        ])
        jacc_df = pd.DataFrame(jaccard_data, index=motifs_list, columns=motifs_list)
        jacc_df.to_parquet(motifs_07_dir / f"motif_interaction_jaccard_{v}.parquet")

    return {
        "data_root": target_dir,
        "parquet_dir": parquet_dir,
        "motifs_07_dir": motifs_07_dir,
    }


class M05Tests(unittest.TestCase):
    """Test suite for M05 checks C1 through C17."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.env = _create_mock_08_environment(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_paths(self) -> None:
        """C1: Resolver caminhos sem depender de diretórios fixos legados."""
        run = begin_m05(
            "short/08",
            project_root=self.root,
            settings={
                "data_root": str(self.env["data_root"]),
                "is_smoke_test": True,
            },
        )
        self.assertTrue(run.output_dir.is_dir())
        self.assertTrue(str(run.output_dir).startswith(str(self.root)))
        self.assertEqual(run.manifest["protocol_id"], PROTOCOL_ID)

        # Runs distintas recebem diretórios distintos
        run2 = begin_m05(
            "short/08",
            project_root=self.root,
            settings={
                "data_root": str(self.env["data_root"]),
                "is_smoke_test": True,
            },
        )
        self.assertNotEqual(run.output_dir, run2.output_dir)

    def test_preflight_08_inputs(self) -> None:
        """C2: Validar a presença e legibilidade do manifesto e artefatos de 07."""
        with patch.dict(os.environ, {"M05_MOTIFS_07_DIR": str(self.env["motifs_07_dir"])}):
            report = preflight_notebook("short/08", self.env["data_root"], smoke_test=False)
            self.assertEqual(report["status"], "READY")

            # Remove um dos arquivos de Jaccard obrigatórios
            missing_file = self.env["motifs_07_dir"] / "motif_interaction_jaccard_resnet50_species.parquet"
            missing_file.unlink()

            report_blocked = preflight_notebook("short/08", self.env["data_root"], smoke_test=False)
            self.assertEqual(report_blocked["status"], "BLOCKED")

    def test_collision_rejection(self) -> None:
        """C3: Abortar se diretório reservado já contiver produtos ou colidir."""
        dest = self.root / "results_reproduction" / "runs" / "m05-collision-test" / "short" / "08"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "prior_artifact.txt").write_text("prior")

        with self.assertRaises((FileExistsError, ValueError)):
            begin_m05(
                "short/08",
                project_root=self.root,
                settings={
                    "run_id": "m05-collision-test",
                    "data_root": str(self.env["data_root"]),
                    "is_smoke_test": True,
                },
            )

    def test_failed_manifest(self) -> None:
        """C4: Emitir manifesto atômico FAILED sob erro inesperado."""
        run = begin_m05(
            "short/08",
            project_root=self.root,
            settings={
                "data_root": str(self.env["data_root"]),
                "is_smoke_test": True,
            },
        )
        with self.assertRaises(RuntimeError):
            with run.guard():
                raise RuntimeError("Erro inesperado para teste de falha atômica")

        with open(run.manifest_path) as f:
            manifest = json.load(f)

        self.assertEqual(manifest["status"], "FAILED")
        self.assertIn("Erro inesperado para teste de falha atômica", manifest["error"])
        self.assertNotEqual(manifest["status"], "COMPLETE")

    def test_complete_manifest(self) -> None:
        """C5: Publicar manifesto COMPLETE com protocolo m05_triplets_audit_v1 e produtos."""
        run = begin_m05(
            "short/08",
            project_root=self.root,
            settings={
                "data_root": str(self.env["data_root"]),
                "is_smoke_test": True,
            },
        )
        dummy_product = run.output_dir / "table_09_exploratory_rscu.csv"
        dummy_product.write_text("codon,rscu\nATG,1.2\n")

        manifest = run.complete(input_files=[self.env["parquet_dir"] / "test.parquet"])
        self.assertEqual(manifest["status"], "COMPLETE")
        self.assertEqual(manifest["protocol_id"], PROTOCOL_ID)
        self.assertEqual(manifest["seed"], 42)
        self.assertTrue(len(manifest["outputs"]) >= 1)
        self.assertTrue(len(manifest["inputs"]) >= 1)

    def test_deterministic_sampling(self) -> None:
        """C6: Amostragem determinística com semente 42 fixa."""
        seqs_1 = sample_deterministic_test_sequences(
            self.env["parquet_dir"],
            max_rows=5,
            seed=42,
            is_smoke_test=False,
        )
        seqs_2 = sample_deterministic_test_sequences(
            self.env["parquet_dir"],
            max_rows=5,
            seed=42,
            is_smoke_test=False,
        )
        self.assertEqual(seqs_1, seqs_2)
        self.assertEqual(len(seqs_1), 5)

    def test_rscu_three_frames_export(self) -> None:
        """C7: Cálculo dos 64 códons sob 3 modos de leitura e exportação da Tabela 9."""
        dna_sequences = ["ACGT" * 128] * 10
        rscu_f0 = compute_exploratory_triplet_rscu(dna_sequences, frame_mode="frame0")
        rscu_3f = compute_exploratory_triplet_rscu(dna_sequences, frame_mode="forward_3frames")
        rscu_6f = compute_exploratory_triplet_rscu(dna_sequences, frame_mode="six_frames")

        self.assertEqual(len(rscu_f0["rscu"]), 64)
        self.assertEqual(len(rscu_3f["rscu"]), 64)
        self.assertEqual(len(rscu_6f["rscu"]), 64)

        t9_records = []
        for codon, val_f0 in rscu_f0["rscu"].items():
            t9_records.append({
                "codon": codon,
                "rscu_frame0": val_f0,
                "rscu_forward_3frames": rscu_3f["rscu"][codon],
                "rscu_six_frames": rscu_6f["rscu"][codon],
                "multinomial_null": rscu_f0["multinomial_null_probabilities"][codon],
            })
        df_t9 = pd.DataFrame(t9_records)
        out_csv = self.root / "table_09_exploratory_rscu.csv"
        df_t9.to_csv(out_csv, index=False)

        self.assertTrue(out_csv.is_file())
        reloaded = pd.read_csv(out_csv)
        self.assertEqual(len(reloaded), 64)
        self.assertIn("multinomial_null", reloaded.columns)

    def test_figure_04_profile_rendering(self) -> None:
        """C8: Renderização da Figura 4 de perfil de tripletos com linha unitária."""
        dna_sequences = ["ACGT" * 128] * 10
        rscu_f0 = compute_exploratory_triplet_rscu(dna_sequences, frame_mode="frame0")
        rscu_3f = compute_exploratory_triplet_rscu(dna_sequences, frame_mode="forward_3frames")
        rscu_6f = compute_exploratory_triplet_rscu(dna_sequences, frame_mode="six_frames")

        fig4_path = self.root / "figure_04_rscu_profile.png"
        render_figure_04_rscu(rscu_f0, rscu_3f, rscu_6f, fig4_path)

        self.assertTrue(fig4_path.is_file())
        self.assertGreater(fig4_path.stat().st_size, 1000)

    def test_boundary_signals_audit(self) -> None:
        """C9: Ocorrências de ATG e STOP contra modelo de composição (Tabela 10)."""
        sample_motifs = ["ATGAAAA", "TAACCCC", "TAGGGGG", "TGATTTT", "CCCCCCC"]
        base_freqs = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
        audit = audit_translation_boundary_signals(sample_motifs, base_freqs)

        t10_records = []
        for key, data in audit.items():
            t10_records.append({
                "signal": key,
                "codon": data["codon"],
                "observed_count": data["observed_count"],
                "expected_count": data["expected_count"],
                "fold_enrichment": data["fold_enrichment"],
                "null_probability": data["null_probability"],
            })
        df_t10 = pd.DataFrame(t10_records)
        out_csv = self.root / "table_10_translation_boundary_signals.csv"
        df_t10.to_csv(out_csv, index=False)

        self.assertTrue(out_csv.is_file())
        self.assertIn("start_atg", [r["signal"].lower() for r in t10_records])
        self.assertEqual(len(df_t10), 4)

    def test_reject_synthetic_in_empirical(self) -> None:
        """C10: Rejeitar preparação de fixtures sintéticas sob SMOKE_TEST=0."""
        run = begin_m05(
            "short/08",
            project_root=self.root,
            settings={
                "data_root": str(self.env["data_root"]),
                "is_smoke_test": False,
                "motifs_07_dir": str(self.env["motifs_07_dir"]),
            },
        )
        with self.assertRaises(ValueError):
            run.prepare_smoke()

    def test_cooccurrence_symmetric_synergy(self) -> None:
        """C11: Análise de coocorrência simétrica, ICC e teste de permutação de nós (Tabela 11)."""
        motifs = ["CCCCCCC", "GGGGGGG", "ATATATA", "TTTTTTT", "AAAAAAA"]
        jaccard_data = np.array([
            [1.0, 0.4, 0.2, 0.1, 0.0],
            [0.4, 1.0, 0.3, 0.0, 0.2],
            [0.2, 0.3, 1.0, 0.5, 0.1],
            [0.1, 0.0, 0.5, 1.0, 0.4],
            [0.0, 0.2, 0.1, 0.4, 1.0],
        ])
        jacc_df = pd.DataFrame(jaccard_data, index=motifs, columns=motifs)
        rscu_map = {"CCCCCCC": 1.2, "GGGGGGG": 0.8, "ATATATA": 1.1, "TTTTTTT": 0.9, "AAAAAAA": 1.0}

        results = analyze_cooccurrence_pairs(
            jaccard_df=jacc_df,
            rscu_map=rscu_map,
            top_k=200,
            n_permutations=100,
            seed=42,
        )
        self.assertEqual(results["status"], "VALID")
        self.assertGreater(results["n_valid_edges"], 0)
        self.assertIn("r_bidirectional", results)
        self.assertIn("icc_symmetric", results)
        self.assertIn("p_perm", results)
        self.assertTrue(np.isfinite(results["p_perm"]))

    def test_orientation_invariance(self) -> None:
        """C12: Invariância exata à orientação das arestas (delta_r_orientation = 0)."""
        motifs = ["CCCCCCC", "GGGGGGG", "ATATATA", "TTTTTTT"]
        jaccard_data = np.array([
            [1.0, 0.5, 0.3, 0.2],
            [0.5, 1.0, 0.4, 0.1],
            [0.3, 0.4, 1.0, 0.6],
            [0.2, 0.1, 0.6, 1.0],
        ])
        jacc_df = pd.DataFrame(jaccard_data, index=motifs, columns=motifs)
        rscu_map = {"CCCCCCC": 1.3, "GGGGGGG": 0.7, "ATATATA": 1.2, "TTTTTTT": 0.8}

        results = analyze_cooccurrence_pairs(jacc_df, rscu_map, top_k=200, n_permutations=50, seed=42)
        self.assertEqual(results["delta_r_orientation"], 0.0)
        self.assertEqual(results["r_canonical"], results["r_reversed"])

    def test_edge_table_c06_compatibility(self) -> None:
        """C13: Tabela de arestas com coordenadas x1/y1/x2/y2 compatível com C06."""
        motifs = ["CCCCCCC", "GGGGGGG", "ATATATA"]
        jaccard_data = np.array([
            [1.0, 0.4, 0.2],
            [0.4, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ])
        jacc_df = pd.DataFrame(jaccard_data, index=motifs, columns=motifs)
        edge_table = generate_network_edge_table(jacc_df, top_k=200, seed=42)

        expected_cols = {"x1", "y1", "x2", "y2", "motif_a", "motif_b", "jaccard"}
        self.assertTrue(expected_cols.issubset(edge_table.columns))
        self.assertEqual(len(edge_table), 3)

        out_path = self.root / "edge_table.csv"
        edge_table.to_csv(out_path, index=False)
        self.assertTrue(out_path.is_file())

    def test_figure_05_network_and_null_rendering(self) -> None:
        """C14: Renderização das figuras de rede, sinergia e nulo de permutação."""
        motifs = ["CCCCCCC", "GGGGGGG", "ATATATA"]
        jaccard_data = np.array([
            [1.0, 0.4, 0.2],
            [0.4, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ])
        jacc_df = pd.DataFrame(jaccard_data, index=motifs, columns=motifs)
        edge_table = generate_network_edge_table(jacc_df, top_k=200, seed=42)

        net_fig = self.root / "figure_05_network.png"
        render_figure_05_network(edge_table, net_fig)
        self.assertTrue(net_fig.is_file())
        self.assertGreater(net_fig.stat().st_size, 500)

        results = {
            "status": "VALID",
            "r_bidirectional": 0.12,
            "icc_symmetric": 0.11,
            "p_perm": 0.4695,
            "null_distribution": np.random.normal(0, 0.1, 100).tolist(),
        }
        synergy_fig = self.root / "figure_05_cooccurrence_synergy.png"
        render_figure_05_cooccurrence_synergy(results, synergy_fig)
        self.assertTrue(synergy_fig.is_file())
        self.assertGreater(synergy_fig.stat().st_size, 500)

        null_fig = self.root / "figure_05_permutation_null.png"
        render_figure_05_permutation_null(results, null_fig)
        self.assertTrue(null_fig.is_file())
        self.assertGreater(null_fig.stat().st_size, 500)

    def test_inter_variant_diagnostics(self) -> None:
        """C15: Diagnóstico descritivo inter-modelos sem falso teste inferencial."""
        motifs = ["CCCCCCC", "GGGGGGG", "ATATATA"]
        mat1 = pd.DataFrame(np.eye(3), index=motifs, columns=motifs)
        mat2 = pd.DataFrame(np.eye(3), index=motifs, columns=motifs)
        matrices = {
            "resnet50_species": mat1,
            "resnet_noise_species": mat2,
        }
        diag_path = self.root / "table_11b_jaccard_variant_diagnostics.csv"
        df_diag = export_reconciled_table_11b_diagnostics(matrices, diag_path)

        self.assertTrue(diag_path.is_file())
        row = df_diag.iloc[0]
        self.assertEqual(row["status"], "DESCRIPTIVE_SHARED_INPUT_CHECK")
        self.assertEqual(row["n_common_motifs"], 3)
        self.assertTrue(bool(row["off_diagonal_identical"]))

    def test_exploratory_demarcation(self) -> None:
        """C16: Demarcação formal de composição exploratória de tripletos."""
        self.assertIn("EXPLORATORY TRIPLET COMPOSITION ONLY", BIOLOGICAL_DEMARCATION_TEXT)
        self.assertIn("lack verified biological open reading frames", BIOLOGICAL_DEMARCATION_TEXT)

    def test_logging_path(self) -> None:
        """C17: Rastreabilidade com prefixo [LOG_PATH], respeitando LOG_PATH=0/false."""
        dummy_file = self.root / "dummy.txt"
        dummy_file.write_text("sample")

        with patch("sys.stderr", new=io.StringIO()) as mock_stderr:
            with patch.dict(os.environ, {"LOG_PATH": "true"}):
                run = begin_m05(
                    "short/08",
                    project_root=self.root,
                    settings={
                        "data_root": str(self.env["data_root"]),
                        "is_smoke_test": True,
                    },
                )
                run.complete(input_files=[dummy_file])
            self.assertIn("[LOG_PATH]", mock_stderr.getvalue())

        with patch("sys.stderr", new=io.StringIO()) as mock_stderr:
            with patch.dict(os.environ, {"LOG_PATH": "0"}):
                run2 = begin_m05(
                    "short/08",
                    project_root=self.root,
                    settings={
                        "data_root": str(self.env["data_root"]),
                        "is_smoke_test": True,
                    },
                )
                run2.complete(input_files=[dummy_file])
            self.assertNotIn("[LOG_PATH]", mock_stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
