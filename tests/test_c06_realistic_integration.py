"""Realistic end-to-end integration and negative test suite for C06 publication pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from viral_cnn.reporting.publication import (
    ITEM_IDS,
    PublicationValidationError,
    assemble_publication,
    build_candidate_publication_config,
    load_publication_config,
    sha256_file,
    validate_selection,
)
from viral_cnn.reporting.transformations import (
    FIVE_VARIANTS_SPECIES,
    execute_publication_transformation,
)


class TestC06RealisticIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / ".git").mkdir()
        (self.root / "configs").mkdir()
        (self.root / "raw_stages").mkdir()
        (self.root / "derived").mkdir()

        self.dataset = self.root / "raw_stages" / "dataset.csv"
        self.dataset.write_text("dataset_identity_v1\n", encoding="utf-8")
        self.class_order = self.root / "raw_stages" / "class_order.csv"
        self.class_order.write_text("class_order_v1\n", encoding="utf-8")
        self.checkpoint = self.root / "raw_stages" / "model.pt"
        self.checkpoint.write_bytes(b"checkpoint-weights")

        self.panel_checkpoints = {}
        for var in FIVE_VARIANTS_SPECIES:
            ckpt_p = self.root / "raw_stages" / f"{var}.pt"
            ckpt_p.write_bytes(f"weights:{var}".encode("utf-8"))
            self.panel_checkpoints[var] = sha256_file(ckpt_p)

        self.identity = {
            "dataset_sha256": sha256_file(self.dataset),
            "population_id": "closed_set_viral_v1",
            "split_id": "stratified_80_10_10",
            "seed": 42,
            "taxonomy_version": "ncbi_ictv_2024",
            "class_order_sha256": sha256_file(self.class_order),
            "checkpoint_sha256s": {"fixture_model": sha256_file(self.checkpoint)},
            "model_variants": ["fixture_model"],
            "noise_conditions": [0.0, 0.02, 0.05, 0.10, 0.15],
            "metric_definition": "top1_accuracy_over_all_queries",
        }

        self._create_raw_notebook_outputs()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_parent_manifest(self, artifact_path: Path, protocol: str, item_id: str) -> Path:
        manifest_p = artifact_path.with_name(f"{artifact_path.stem}.manifest.json")
        identity = dict(self.identity)
        if item_id == "F3":
            identity["checkpoint_sha256s"] = self.panel_checkpoints
            identity["model_variants"] = list(FIVE_VARIANTS_SPECIES)

        data = {
            "is_smoke_test": False,
            "protocol": protocol,
            "status": "COMPLETE",
            "artifact_sha256": sha256_file(artifact_path),
            "n": 1706270,
            "support": 1706270,
            "seed": 42,
            "split_id": "stratified_80_10_10",
            "evidence_kind": "empirical",
            "scientific_identity": identity,
            "inputs": [
                {"path": str(self.dataset), "sha256": sha256_file(self.dataset)},
                {"path": str(self.class_order), "sha256": sha256_file(self.class_order)},
            ],
        }
        if item_id == "F3":
            data["panels"] = ["a", "b", "c", "d", "e"]
            data["panel_variants"] = {
                "a": "ResNet50_Clean",
                "b": "ConvNeXt_Z",
                "c": "ConvNeXt_T",
                "d": "ResNet50_F",
                "e": "ConvNeXt_F",
            }
        if item_id.startswith("F"):
            data["selection"] = {
                "ids": ["Species_A", "Species_B"],
                "levels": ["Species"],
                "sampling": "deterministic",
                "seed": 42,
                "filters": {"noise_levels": [0.0]},
            }
        manifest_p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return manifest_p

    def _create_raw_notebook_outputs(self):
        """Create authentic raw notebook formats for T1-T12 and F1-F5."""
        self.raw_files: dict[str, Path] = {}
        self.parent_manifests: dict[str, Path] = {}

        # 1. short/01
        dir_01 = self.root / "raw_stages" / "short" / "01"
        dir_01.mkdir(parents=True, exist_ok=True)

        t1_raw = dir_01 / "table1_metrics.csv"
        pd.DataFrame({
            "model": ["ResNet50_Clean", "ConvNeXt_Z"],
            "taxonomic_level": ["Species", "Species"],
            "backbone_m": [15.96, 28.16],
            "fc_params_m": [28.03, 28.03],
            "parameters_m": [43.99, 56.19],
            "accuracy": [84.88, 85.12],
            "top_5": [94.10, 94.50],
            "macro_f1": [78.20, 78.90],
            "latency_ms": [1.45, 2.10],
        }).to_csv(t1_raw, index=False)
        self.raw_files["T1"] = t1_raw
        self.parent_manifests["T1"] = self._write_parent_manifest(t1_raw, "short_01_metrics_v1", "T1")

        t2_raw = dir_01 / "table2_profiling.csv"
        pd.DataFrame({
            "model": ["ResNet50_Clean", "ConvNeXt_Z"],
            "backbone_m": [15.96, 28.16],
            "fc_params_m": [28.03, 28.03],
            "parameters_m": [43.99, 56.19],
            "peak_vram_mb": [1850.5, 2100.2],
            "step_time_ms": [12.4, 15.8],
            "throughput_seq_s": [320.5, 250.1],
            "gpu_hours": [4.2, 5.8],
        }).to_csv(t2_raw, index=False)
        self.raw_files["T2"] = t2_raw
        self.parent_manifests["T2"] = self._write_parent_manifest(t2_raw, "short_01_profiling_v1", "T2")

        t4_raw = dir_01 / "table4_f1_bands.csv"
        pd.DataFrame({
            "f1_band": ["[0.0-0.2]", "[0.2-0.4]", "[0.4-0.6]", "[0.6-0.8]", "[0.8-1.0]"],
            "count_resnet_clean": [120, 250, 480, 1800, 11033],
            "pct_resnet_clean": [0.88, 1.83, 3.51, 13.15, 80.63],
        }).to_csv(t4_raw, index=False)
        self.raw_files["T4"] = t4_raw
        self.parent_manifests["T4"] = self._write_parent_manifest(t4_raw, "short_01_f1_bands_v1", "T4")

        # 2. short/02
        dir_02 = self.root / "raw_stages" / "short" / "02"
        dir_02.mkdir(parents=True, exist_ok=True)

        t7_raw = dir_02 / "table7_hierarchical_entropy.csv"
        pd.DataFrame({
            "model_name": ["ResNet50_Clean", "ConvNeXt_Z"],
            "mean_h_species": [1.45, 1.38],
            "mean_h_genus": [0.82, 0.79],
            "mean_hierarchical_gap": [0.63, 0.59],
            "mean_h_activation": [2.10, 2.05],
        }).to_csv(t7_raw, index=False)
        self.raw_files["T7"] = t7_raw
        self.parent_manifests["T7"] = self._write_parent_manifest(t7_raw, "short_02_entropy_v1", "T7")

        # 3. short/06 & short/06b
        dir_06 = self.root / "raw_stages" / "short" / "06"
        dir_06.mkdir(parents=True, exist_ok=True)

        t5_raw = dir_06 / "table5_latent_clustering_metrics.csv"
        pd.DataFrame({
            "Model": ["ResNet50_Clean", "ConvNeXt_Z"],
            "Rank": ["Species", "Species"],
            "KNN_Accuracy": [82.50, 83.10],
            "Silhouette_Score": [0.42, 0.45],
            "Silhouette_Valid_Taxa_Pct": [98.5, 99.1],
            "Linear_Probe_Accuracy": [83.80, 84.20],
        }).to_csv(t5_raw, index=False)
        self.raw_files["T5"] = t5_raw
        self.parent_manifests["T5"] = self._write_parent_manifest(t5_raw, "short_06_latent_v1", "T5")

        t6_raw = dir_06 / "table6_taxonomic_rank_correlations.csv"
        pd.DataFrame({
            "Model": ["ResNet50_Clean", "ConvNeXt_Z"],
            "Transition": ["Species -> Genus", "Species -> Genus"],
            "Pearson_r": [0.85, 0.88],
            "Spearman_rho": [0.82, 0.86],
            "N_taxa": [2702, 2702],
        }).to_csv(t6_raw, index=False)
        self.raw_files["T6"] = t6_raw
        self.parent_manifests["T6"] = self._write_parent_manifest(t6_raw, "short_06_correlations_v1", "T6")

        dir_06b = self.root / "raw_stages" / "short" / "06b"
        dir_06b.mkdir(parents=True, exist_ok=True)

        f2_raw = dir_06b / "fig2_umap_projection.csv"
        f2_content = (
            "# Paired UMAP 2D Latent Projections\n"
            "# Seed: 42, Metric: cosine\n"
            "# Disclaimer: Qualitative visualization only; metric analysis in original space.\n"
            "fragment_id,model,genus,umap_1,umap_2\n"
            "frag_01,ResNet50_Clean,Alphavirus,1.25,3.40\n"
            "frag_02,ConvNeXt_Z,Alphavirus,1.15,3.25\n"
        )
        f2_raw.write_text(f2_content, encoding="utf-8")
        self.raw_files["F2"] = f2_raw
        self.parent_manifests["F2"] = self._write_parent_manifest(f2_raw, "short_06b_umap_v1", "F2")

        # 4. short/07
        dir_07 = self.root / "raw_stages" / "short" / "07"
        dir_07.mkdir(parents=True, exist_ok=True)

        t8_raw = dir_07 / "motif_perturbation_by_variant.csv"
        pd.DataFrame({
            "variant": ["ResNet50_Clean", "ConvNeXt_Z"],
            "motif": ["TATAAAG", "GCGCGCG"],
            "clean_prob": [0.95, 0.92],
            "mutated_prob": [0.53, 0.54],
            "activation_drop": [0.42, 0.38],
            "control_uniform_drop": [0.15, 0.12],
            "control_gc_drop": [0.18, 0.14],
            "delta_drop_gc": [0.24, 0.24],
            "p_value_gc": [0.001, 0.001],
            "p_adj_gc": [0.005, 0.005],
            "sample_size": [500, 500],
        }).to_csv(t8_raw, index=False)
        self.raw_files["T8"] = t8_raw
        self.parent_manifests["T8"] = self._write_parent_manifest(t8_raw, "short_07_motifs_v1", "T8")

        f3_raw = dir_07 / "fig3_motif_perturbation.csv"
        pd.DataFrame([
            {"panel": "a", "variant": "ResNet50_Clean", "motif": "TATAAAG", "activation_drop": 0.42},
            {"panel": "b", "variant": "ConvNeXt_Z", "motif": "GCGCGCG", "activation_drop": 0.38},
            {"panel": "c", "variant": "ConvNeXt_T", "motif": "CGCTAAG", "activation_drop": 0.35},
            {"panel": "d", "variant": "ResNet50_F", "motif": "ATATAAG", "activation_drop": 0.40},
            {"panel": "e", "variant": "ConvNeXt_F", "motif": "CCGGAAG", "activation_drop": 0.37},
        ]).to_csv(f3_raw, index=False)
        self.raw_files["F3"] = f3_raw
        self.parent_manifests["F3"] = self._write_parent_manifest(f3_raw, "short_07_perturbation_v1", "F3")

        # 5. short/08
        dir_08 = self.root / "raw_stages" / "short" / "08"
        dir_08.mkdir(parents=True, exist_ok=True)

        t9_raw = dir_08 / "table_09_exploratory_rscu.csv"
        pd.DataFrame({
            "codon": ["AAA", "AAC"],
            "rscu_frame0": [1.12, 0.88],
            "rscu_forward_3frames": [1.05, 0.92],
            "rscu_six_frames": [1.02, 0.95],
            "multinomial_null": [1.00, 1.00],
        }).to_csv(t9_raw, index=False)
        self.raw_files["T9"] = t9_raw
        self.parent_manifests["T9"] = self._write_parent_manifest(t9_raw, "short_08_rscu_v1", "T9")

        t10_raw = dir_08 / "table_10_translation_boundary_signals.csv"
        pd.DataFrame({
            "signal": ["START", "STOP"],
            "codon": ["ATG", "TAA"],
            "observed_count": [450, 310],
            "expected_count": [300, 280],
            "fold_enrichment": [1.50, 1.11],
            "null_probability": [0.001, 0.02],
        }).to_csv(t10_raw, index=False)
        self.raw_files["T10"] = t10_raw
        self.parent_manifests["T10"] = self._write_parent_manifest(t10_raw, "short_08_signals_v1", "T10")

        t11_raw = dir_08 / "table11_cooccurrence_synergy.csv"
        pd.DataFrame({
            "status": ["COMPLETE"],
            "motif_score": ["rscu_synergy"],
            "n_valid_edges": [150],
            "n_valid_permutations": [1000],
            "r_canonical": [0.45],
            "r_reversed": [0.42],
            "delta_r_orientation": [0.03],
            "icc_symmetric": [0.72],
            "r_bidirectional": [0.44],
            "p_permutation_null": [0.002],
        }).to_csv(t11_raw, index=False)
        self.raw_files["T11"] = t11_raw
        self.parent_manifests["T11"] = self._write_parent_manifest(t11_raw, "short_08_synergy_v1", "T11")

        f4_raw = dir_08 / "fig4_rscu_profile.csv"
        pd.DataFrame({
            "codon": ["AAA", "AAC", "AAG"],
            "rscu_frame0": [1.12, 0.88, 1.05],
            "rscu_forward_3frames": [1.05, 0.92, 1.01],
            "rscu_six_frames": [1.02, 0.95, 1.00],
        }).to_csv(f4_raw, index=False)
        self.raw_files["F4"] = f4_raw
        self.parent_manifests["F4"] = self._write_parent_manifest(f4_raw, "short_08_fig_rscu_v1", "F4")

        f5_raw = dir_08 / "edge_table.csv"
        pd.DataFrame({
            "x1": [0.1, 0.5],
            "y1": [0.2, 0.4],
            "x2": [0.3, 0.7],
            "y2": [0.4, 0.6],
            "weight": [0.85, 0.60],
        }).to_csv(f5_raw, index=False)
        self.raw_files["F5"] = f5_raw
        self.parent_manifests["F5"] = self._write_parent_manifest(f5_raw, "short_08_edges_v1", "F5")

        # 6. short/09
        dir_09 = self.root / "raw_stages" / "short" / "09"
        dir_09.mkdir(parents=True, exist_ok=True)

        t3_raw = dir_09 / "table3_noise_degradation.csv"
        pd.DataFrame({
            "noise_rate": [0.0, 0.05],
            "unconditional_accuracy": [84.88, 72.10],
            "conditional_accuracy": [84.88, 72.10],
            "coverage": [100.0, 100.0],
            "denominator_n": [1706270, 1706270],
        }).to_csv(t3_raw, index=False)
        self.raw_files["T3"] = t3_raw
        self.parent_manifests["T3"] = self._write_parent_manifest(t3_raw, "short_09_noise_v1", "T3")

        f1_raw = dir_09 / "fig1_noise_robustness.csv"
        pd.DataFrame({
            "noise_rate": [0.0, 0.02, 0.05, 0.10, 0.15],
            "unconditional_accuracy": [84.88, 80.20, 72.10, 60.50, 48.30],
        }).to_csv(f1_raw, index=False)
        self.raw_files["F1"] = f1_raw
        self.parent_manifests["F1"] = self._write_parent_manifest(f1_raw, "short_09_fig_robustness_v1", "F1")

        # 7. long/03
        dir_long03 = self.root / "raw_stages" / "long" / "03"
        dir_long03.mkdir(parents=True, exist_ok=True)

        t12_raw = dir_long03 / "table_12_long_voter_comparison.csv"
        pd.DataFrame({
            "variant": ["ResNet50_Clean", "ResNet50_Clean", "ConvNeXt_Z", "ConvNeXt_Z"],
            "aggregation": ["majority_vote", "mean_logits", "majority_vote", "mean_logits"],
            "accuracy": [72.69, 74.15, 68.40, 70.10],
            "n_sequences": [1500, 1500, 1500, 1500],
            "coverage": [100.0, 100.0, 100.0, 100.0],
        }).to_csv(t12_raw, index=False)
        self.raw_files["T12"] = t12_raw
        self.parent_manifests["T12"] = self._write_parent_manifest(t12_raw, "long_03_consensus_v1", "T12")

    def _create_and_transform_all_derived(self) -> dict[str, Any]:
        """Execute publication transformations for all 17 items into self.root / 'derived'."""
        out_dir = self.root / "derived"
        items_cfg = []

        for item_id in ITEM_IDS:
            raw_path = self.raw_files[item_id]
            parent_manifest_path = self.parent_manifests[item_id]

            derived_artifact, derived_manifest = execute_publication_transformation(
                item_id=item_id,
                source_artifact_path=raw_path,
                output_dir=out_dir,
                parent_manifest_path=parent_manifest_path,
            )
            derived_sha256 = sha256_file(derived_artifact)

            plot = {"x": "x", "y": "y", "title": item_id}
            if item_id == "F1":
                plot = {"x": "epsilon", "y": "unconditional_accuracy", "title": "Figure 1"}
            elif item_id == "F2":
                plot = {
                    "kind": "scatter",
                    "x": "umap_1",
                    "y": "umap_2",
                    "title": "Figure 2",
                    "interpretation": "qualitative visualization only",
                }
            elif item_id == "F3":
                plot = {
                    "kind": "panel_grid",
                    "x": "motif",
                    "y": "activation_drop",
                    "title": "Figure 3",
                }
            elif item_id == "F4":
                plot = {
                    "x": "triplet",
                    "y": "rscu",
                    "title": "Figure 4",
                    "interpretation": "exploratory triplet composition",
                }
            elif item_id == "F5":
                plot = {
                    "kind": "network_edges",
                    "x1": "x1",
                    "y1": "y1",
                    "x2": "x2",
                    "y2": "y2",
                    "title": "Figure 5",
                }

            expected = {
                "scientific_identity": dict(self.identity),
            }
            if item_id.startswith("T"):
                df_der = pd.read_csv(derived_artifact)
                expected["columns"] = list(df_der.columns)
            if item_id == "F3":
                expected["scientific_identity"]["checkpoint_sha256s"] = self.panel_checkpoints
                expected["scientific_identity"]["model_variants"] = list(FIVE_VARIANTS_SPECIES)
                expected["panels"] = ["a", "b", "c", "d", "e"]
                expected["panel_variants"] = {
                    "a": "ResNet50_Clean",
                    "b": "ConvNeXt_Z",
                    "c": "ConvNeXt_T",
                    "d": "ResNet50_F",
                    "e": "ConvNeXt_F",
                }

            derived_data = json.loads(derived_manifest.read_text(encoding="utf-8"))
            items_cfg.append({
                "item_id": item_id,
                "artifact": str(derived_artifact),
                "manifest": str(derived_manifest),
                "sha256": derived_sha256,
                "protocol": derived_data.get("protocol", "c06_strict_v1"),
                "transformation": derived_data.get("transformation", f"transform_{item_id.lower()}"),
                "expected": expected,
                "output": f"{item_id.lower()}.csv",
                "plot": plot,
            })

        cfg_data = {
            "publication_protocol": "c06_strict_v1",
            "schema_version": 1,
            "run_id": "realistic_c06_run",
            "historical_source": {},
            "items": items_cfg,
            "_project_root": str(self.root),
        }
        cfg_path = self.root / "configs" / "publication_realistic.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_data, sort_keys=False), encoding="utf-8")
        return {"config_path": cfg_path, "config_data": cfg_data}

    # S1 - Candidate configuration from run_plan
    def test_candidate_config_maps_stages(self):
        """C1: build_candidate_publication_config maps stages and calculates artifact SHA-256."""
        run_plan_data = {
            "protocol_id": "m00_run_plan_v1",
            "run_id": "test_run_42",
            "parent_run_id": "test_run_42",
            "stages": [
                {"notebook_id": "short/01", "output_dir": str(self.root / "raw_stages" / "short" / "01")},
                {"notebook_id": "short/02", "output_dir": str(self.root / "raw_stages" / "short" / "02")},
                {"notebook_id": "short/06", "output_dir": str(self.root / "raw_stages" / "short" / "06")},
                {"notebook_id": "short/06b", "output_dir": str(self.root / "raw_stages" / "short" / "06b")},
                {"notebook_id": "short/07", "output_dir": str(self.root / "raw_stages" / "short" / "07")},
                {"notebook_id": "short/08", "output_dir": str(self.root / "raw_stages" / "short" / "08")},
                {"notebook_id": "short/09", "output_dir": str(self.root / "raw_stages" / "short" / "09")},
                {"notebook_id": "long/03", "output_dir": str(self.root / "raw_stages" / "long" / "03")},
            ],
        }
        plan_path = self.root / "run_plan.json"
        plan_path.write_text(json.dumps(run_plan_data, indent=2), encoding="utf-8")

        base_cfg_path = self.root / "configs" / "base_template.yaml"
        base_items = [{"item_id": iid, "artifact": "", "sha256": ""} for iid in ITEM_IDS]
        base_cfg_path.write_text(yaml.safe_dump({"items": base_items}, sort_keys=False), encoding="utf-8")

        result = build_candidate_publication_config(plan_path, base_cfg_path, project_root=self.root)
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["counts"]["missing"], 0)
        self.assertEqual(result["counts"]["proposed"], 17)

        # Confirm SHA256 matches actual file
        cand_t1 = next(item for item in result["candidate_config"]["items"] if item["item_id"] == "T1")
        self.assertEqual(cand_t1["sha256"], sha256_file(self.raw_files["T1"]))

    def test_candidate_config_emits_diff_report(self):
        """C2: build_candidate_publication_config computes structured diff between base and candidate."""
        run_plan_data = {
            "protocol_id": "m00_run_plan_v1",
            "run_id": "test_diff_run",
            "stages": [
                {"notebook_id": "short/01", "output_dir": str(self.root / "raw_stages" / "short" / "01")},
                # Other stages omitted to test partial/missing diff
            ],
        }
        plan_path = self.root / "run_plan_partial.json"
        plan_path.write_text(json.dumps(run_plan_data, indent=2), encoding="utf-8")

        base_cfg_path = self.root / "configs" / "base_partial.yaml"
        base_items = [{"item_id": iid, "artifact": "", "sha256": ""} for iid in ITEM_IDS]
        base_cfg_path.write_text(yaml.safe_dump({"items": base_items}, sort_keys=False), encoding="utf-8")

        result = build_candidate_publication_config(plan_path, base_cfg_path, project_root=self.root)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertTrue(result["counts"]["missing"] > 0)
        self.assertEqual(result["counts"]["proposed"], 3)  # T1, T2, T4 in short/01

        diff = result["diff"]
        t1_diff = next(d for d in diff if d["item_id"] == "T1")
        self.assertEqual(t1_diff["status"], "PROPOSED")
        t3_diff = next(d for d in diff if d["item_id"] == "T3")
        self.assertEqual(t3_diff["status"], "MISSING")

    def test_candidate_config_cli(self):
        """C3: CLI invocation of scripts/build_candidate_publication_config.py produces valid YAML and exit code."""
        run_plan_data = {
            "protocol_id": "m00_run_plan_v1",
            "run_id": "test_cli_run",
            "stages": [
                {"notebook_id": "short/01", "output_dir": str(self.root / "raw_stages" / "short" / "01")},
            ],
        }
        plan_path = self.root / "run_plan_cli.json"
        plan_path.write_text(json.dumps(run_plan_data, indent=2), encoding="utf-8")

        base_cfg_path = self.root / "configs" / "base_cli.yaml"
        base_items = [{"item_id": iid, "artifact": "", "sha256": ""} for iid in ITEM_IDS]
        base_cfg_path.write_text(yaml.safe_dump({"items": base_items}, sort_keys=False), encoding="utf-8")

        out_cand_yaml = self.root / "configs" / "candidate_cli.yaml"
        cmd = [
            sys.executable,
            "scripts/build_candidate_publication_config.py",
            "--run-plan", str(plan_path),
            "--base-config", str(base_cfg_path),
            "--output", str(out_cand_yaml),
            "--project-root", str(self.root),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertTrue(out_cand_yaml.is_file())
        self.assertIn("Candidate C06 selection status", proc.stdout)

    # S2 - Realistic end-to-end integration
    def test_realistic_raw_transformations_emit_derived(self):
        """C4: Raw notebook outputs transformed via execute_publication_transformation emit verified derived artifacts."""
        suite = self._create_and_transform_all_derived()
        config = load_publication_config(suite["config_path"])
        config["_project_root"] = str(self.root)

        rows = validate_selection(config, strict_empirical=True)
        self.assertEqual(len(rows), 17)
        for row in rows:
            self.assertEqual(row["status"], "DERIVED_VERIFIED")
            self.assertTrue(Path(row["artifact"]).is_file())
            self.assertTrue(Path(row["manifest"]).is_file())

    def test_realistic_assembly_all_17_items(self):
        """C5: assemble_publication(strict_empirical=True) accepts all 17 items and produces full package."""
        suite = self._create_and_transform_all_derived()
        out_pkg = self.root / "assembly_pkg"

        paper = self.root / "paper"
        paper.mkdir(parents=True, exist_ok=True)
        table_blocks = "\n".join(
            r"\begin{table}[h!]\centering\caption{Table}\begin{tabular}{cc}a&b\end{tabular}\end{table}"
            for _ in range(12)
        )
        figure_blocks = "\n".join(
            rf"\begin{{figure}}[h!]\centering\includegraphics{{figures/fig_{i}.png}}\end{{figure}}"
            for i in range(1, 6)
        )
        tex = (
            r"\documentclass{article}" + "\n" + r"\begin{document}" + "\n"
            + r"\begin{abstract}\textbf{Results}: old.\par\vspace{0.5em}\noindent\textbf{Conclusions}: old.\end{abstract}" + "\n"
            + r"\section{Methods}" + "\nOld.\n" + r"\section{Results}" + "\n"
            + table_blocks + "\n" + figure_blocks + "\n" + r"\section{Discussion}" + "\nOld.\n"
            + r"\section{Conclusions}" + "\nOld.\n" + r"\subsection*{Availability of data and materials}" + "\n"
            + r"\end{document}"
        )
        (paper / "review.tex").write_text(tex + "\n" + "\n".join(r"\notavdois{n}" for _ in range(56)), encoding="utf-8")
        (paper / "submission.tex").write_text(tex, encoding="utf-8")
        (paper / "references.bib").write_text("", encoding="utf-8")

        suite["config_data"]["historical_source"] = {
            "manuscript_review": "paper/review.tex",
            "manuscript_submission": "paper/submission.tex",
        }
        suite["config_data"]["c05_status"] = "NOT_RUN"
        Path(suite["config_path"]).write_text(yaml.safe_dump(suite["config_data"], sort_keys=False), encoding="utf-8")

        from unittest.mock import patch
        real_run = subprocess.run
        def fake_run(command, **kwargs):
            if Path(command[0]).name != "pdflatex":
                return real_run(command, **kwargs)
            out_arg = next(arg.split("=", 1)[1] for arg in command if arg.startswith("-output-directory="))
            stem = Path(command[-1]).stem
            Path(out_arg, f"{stem}.pdf").write_bytes(b"%PDF-fixture")
            return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

        with patch("viral_cnn.reporting.publication._git_state", return_value=("fixture-commit", True)), \
             patch("viral_cnn.reporting.publication.subprocess.run", side_effect=fake_run):
            result = assemble_publication(
                suite["config_path"],
                out_pkg,
                strict_empirical=True,
                diagnostic_only=False,
            )
        assembly_err = Path(result["outputs"]["assembly_errors"]).read_text() if "assembly_errors" in result.get("outputs", {}) else ""
        self.assertEqual(result["status"], "ASSEMBLED_REVIEW_REQUIRED", msg=assembly_err)
        self.assertEqual(result["gates"]["empirical"], "PASS")
        self.assertEqual(result["gates"]["local"], "PASS")
        self.assertEqual(len(result["rows"]), 17)
        self.assertTrue((out_pkg / "evidence_matrix.csv").is_file())
        self.assertTrue((out_pkg / "publication_manifest.json").is_file())

    def test_inter_notebook_dependencies_chain(self):
        """C6: Inter-notebook dependency chain 07->08, C03->09, 06->F2, long/01->03->T12 is strictly preserved."""
        suite = self._create_and_transform_all_derived()
        config = load_publication_config(suite["config_path"])
        config["_project_root"] = str(self.root)

        # Validate that F2 derived manifest tracks parent from 06b
        f2_item = next(item for item in config["items"] if item["item_id"] == "F2")
        f2_manifest = json.loads(Path(f2_item["manifest"]).read_text(encoding="utf-8"))
        self.assertIn("short/06b", f2_manifest["derivation"]["parent_manifest"]["path"])

        # Validate that T11 derived manifest tracks parent from 08
        t11_item = next(item for item in config["items"] if item["item_id"] == "T11")
        t11_manifest = json.loads(Path(t11_item["manifest"]).read_text(encoding="utf-8"))
        self.assertIn("short/08", t11_manifest["derivation"]["parent_manifest"]["path"])

        # Validate that T12 tracks long/03
        t12_item = next(item for item in config["items"] if item["item_id"] == "T12")
        t12_manifest = json.loads(Path(t12_item["manifest"]).read_text(encoding="utf-8"))
        self.assertIn("long/03", t12_manifest["derivation"]["parent_manifest"]["path"])

    # S3 - Negative tests
    def test_negative_tampered_hash(self):
        """C7: Tampering with artifact SHA-256 triggers PublicationValidationError under strict mode."""
        suite = self._create_and_transform_all_derived()
        config = load_publication_config(suite["config_path"])
        config["_project_root"] = str(self.root)

        # Tamper T1 artifact file
        t1_item = next(item for item in config["items"] if item["item_id"] == "T1")
        Path(t1_item["artifact"]).write_text("corrupted content", encoding="utf-8")

        with self.assertRaises(PublicationValidationError) as ctx:
            validate_selection(config, strict_empirical=True)
        self.assertIn("sha256 mismatch", str(ctx.exception).lower())

    def test_negative_scientific_identity_mismatch(self):
        """C8: Divergence in split_id or class_order produces INVALID status."""
        suite = self._create_and_transform_all_derived()
        config = load_publication_config(suite["config_path"])
        config["_project_root"] = str(self.root)

        # Alter split_id in expected config
        t1_item = next(item for item in config["items"] if item["item_id"] == "T1")
        t1_item["expected"]["scientific_identity"]["split_id"] = "different_split_99"

        rows = validate_selection(config, strict_empirical=False)
        t1_row = next(r for r in rows if r["item_id"] == "T1")
        self.assertEqual(t1_row["status"], "INVALID")
        self.assertIn("split_id", t1_row["diagnostic"])

    def test_negative_synthetic_dependency(self):
        """C9: Transitive dependency on synthetic artifact (is_smoke_test=True) causes rejection."""
        # Mark parent manifest of T7 as smoke test
        t7_manifest_path = self.parent_manifests["T7"]
        m_data = json.loads(t7_manifest_path.read_text(encoding="utf-8"))
        m_data["is_smoke_test"] = True
        t7_manifest_path.write_text(json.dumps(m_data), encoding="utf-8")

        suite = self._create_and_transform_all_derived()
        config = load_publication_config(suite["config_path"])
        config["_project_root"] = str(self.root)

        with self.assertRaises(PublicationValidationError) as ctx:
            validate_selection(config, strict_empirical=True)
        self.assertIn("synthetic", str(ctx.exception).lower())

    def test_negative_data_integrity_violations(self):
        """C10: Zeros fabrications and incomplete T12 aggregations trigger PublicationValidationError."""
        # 1. Zero-fill in T2 profiling
        bad_t2_raw = self.root / "bad_t2.csv"
        pd.DataFrame({
            "model": ["ResNet50_Clean"],
            "backbone_m": [15.96],
            "fc_params_m": [28.03],
            "parameters_m": [43.99],
            "peak_vram_mb": [0.0],  # Forbidden zero-fill
            "step_time_ms": [15.2],
            "throughput_seq_s": [280.0],
            "gpu_hours": [3.5],
        }).to_csv(bad_t2_raw, index=False)

        with self.assertRaises(PublicationValidationError) as ctx:
            execute_publication_transformation(
                item_id="T2",
                source_artifact_path=bad_t2_raw,
                output_dir=self.root / "bad_derived",
                parent_manifest_path=self.parent_manifests["T2"],
            )
        self.assertIn("zero-fill detected", str(ctx.exception).lower())

        # 2. Incomplete T12 aggregations (missing mean_logits)
        bad_t12_raw = self.root / "bad_t12.csv"
        pd.DataFrame({
            "variant": ["ResNet50_Clean"],
            "aggregation": ["majority_vote"],  # missing mean_logits
            "accuracy": [72.69],
            "n_sequences": [1500],
            "coverage": [100.0],
        }).to_csv(bad_t12_raw, index=False)

        with self.assertRaises(PublicationValidationError) as ctx:
            execute_publication_transformation(
                item_id="T12",
                source_artifact_path=bad_t12_raw,
                output_dir=self.root / "bad_derived",
                parent_manifest_path=self.parent_manifests["T12"],
            )
        self.assertIn("missing required aggregation mode", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
