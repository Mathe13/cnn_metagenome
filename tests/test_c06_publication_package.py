"""Offline contracts for C06 selection, evidence, and isolated publication assembly."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from viral_cnn.notebook_runtime import reserve_run_directory
from viral_cnn.reporting.publication import (
    ITEM_IDS,
    PublicationValidationError,
    assemble_publication,
    compile_manuscript_package,
    evidence_matrix,
    load_publication_config,
    reconcile_value,
    render_figures,
    render_manuscript_variants,
    render_tables,
    sha256_file,
    validate_selection,
    write_editorial_review,
    write_publication_manifest,
)


class C06Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / ".git").mkdir()
        (self.root / "configs").mkdir()
        (self.root / "evidence").mkdir()
        source = self.root / "evidence" / "source.txt"
        source.write_text("source fixture\n", encoding="utf-8")
        dataset = self.root / "evidence" / "dataset.csv"
        dataset.write_text("fixture dataset identity\n", encoding="utf-8")
        class_order = self.root / "evidence" / "class_order.csv"
        class_order.write_text("fixture class order\n", encoding="utf-8")
        checkpoint = self.root / "evidence" / "fixture_model.pt"
        checkpoint.write_bytes(b"fixture-checkpoint")
        panel_variants = {"a": "ResNet50_Clean", "b": "ConvNeXt_Z", "c": "ConvNeXt_T",
                          "d": "ResNet50_F", "e": "ConvNeXt_F"}
        panel_checkpoints = {}
        panel_checkpoint_records = []
        for variant in panel_variants.values():
            path = self.root / "evidence" / f"{variant}.pt"
            path.write_bytes(f"checkpoint:{variant}".encode())
            panel_checkpoints[variant] = sha256_file(path)
            panel_checkpoint_records.append({"variant": variant, "path": str(path), "sha256": panel_checkpoints[variant]})
        self.identity = {
            "dataset_sha256": sha256_file(dataset), "population_id": "fixture-population-v1",
            "split_id": "fixture-split", "seed": 42,
            "taxonomy_version": "taxonomy-fixture-v1", "class_order_sha256": sha256_file(class_order),
            "checkpoint_sha256s": {"fixture_model": sha256_file(checkpoint)},
            "model_variants": ["fixture_model"], "noise_conditions": [0.0],
            "metric_definition": "top1_accuracy_over_all_queries",
        }
        self.items = []
        for item_id in ITEM_IDS:
            artifact = self.root / "evidence" / f"{item_id}.csv"
            if item_id == "F3":
                frame = pd.DataFrame([{"panel": panel, "variant": variant, "x": 1, "y": 0.25}
                                      for panel, variant in panel_variants.items()])
            elif item_id == "F5":
                frame = pd.DataFrame({"x1": [0.0, 0.1], "y1": [0.0, 0.1], "x2": [1.0, 0.9], "y2": [1.0, 0.9]})
            else:
                frame = pd.DataFrame({"x": [0, 1], "y": [0.25, 0.5], "accuracy": [50.0, 55.0],
                                      "denominator_n": [10, 10], "support": [5, 5]})
            frame.to_csv(artifact, index=False)
            protocol = "c01_motifs_v1" if item_id in {"T8", "T9", "T10", "T11", "F3", "F4", "F5"} else "c06_fixture_v1"
            transformation = f"transform_{item_id.lower()}"
            expected = {"scientific_identity": dict(self.identity)}
            if item_id.startswith("T"):
                expected["columns"] = ["x", "y"]
            if item_id == "F3":
                expected["scientific_identity"] = {**self.identity,
                    "checkpoint_sha256s": panel_checkpoints,
                    "model_variants": list(panel_variants.values())}
                expected["panel_variants"] = panel_variants
                expected["panels"] = ["a", "b", "c", "d", "e"]
            manifest_path = self.root / "evidence" / f"{item_id}.manifest.json"
            manifest = {
                "is_smoke_test": False, "protocol": protocol, "status": "COMPLETE",
                "transformation": transformation, "artifact_sha256": sha256_file(artifact),
                "inputs": [{"path": str(source), "sha256": sha256_file(source)},
                           {"path": str(dataset), "sha256": sha256_file(dataset)},
                           {"path": str(class_order), "sha256": sha256_file(class_order)}],
                "seed": 42, "split_id": "fixture-split", "n": 10, "evidence_kind": "empirical",
                "scientific_identity": dict(self.identity),
                "support": 5,
                "checkpoints": panel_checkpoint_records if item_id == "F3" else [
                    {"variant": "fixture_model", "path": str(checkpoint), "sha256": sha256_file(checkpoint)}
                ],
            }
            if item_id.startswith("F"):
                manifest["selection"] = {"ids": ["fixture-1", "fixture-2"], "levels": ["Species"],
                                          "sampling": "all fixture rows", "seed": 42, "filters": []}
            if item_id == "F3":
                manifest["panel_variants"] = panel_variants
                manifest["panels"] = ["a", "b", "c", "d", "e"]
                manifest["scientific_identity"] = expected["scientific_identity"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            plot = {"x": "x", "y": "y", "title": item_id}
            if item_id == "F2":
                plot["kind"] = "scatter"
                plot["interpretation"] = "qualitative visualization only"
            if item_id == "F4":
                plot["interpretation"] = "exploratory triplet composition"
            if item_id == "F3":
                plot["kind"] = "panel_grid"
            if item_id == "F5":
                plot = {"kind": "network_edges", "x1": "x1", "y1": "y1", "x2": "x2", "y2": "y2", "title": item_id}
            self.items.append({"item_id": item_id, "artifact": str(artifact), "manifest": str(manifest_path),
                               "sha256": sha256_file(artifact), "protocol": protocol,
                               "transformation": transformation, "expected": expected,
                               "output": f"{item_id}.csv",
                               "plot": plot})
        self.config_path = self.root / "configs" / "c06.yaml"
        self.config_data = {
            "publication_protocol": "c06_strict_v1", "schema_version": 1, "run_id": "fixture",
            "historical_source": {}, "items": self.items,
        }
        self.write_config()
        self.config = load_publication_config(self.config_path)
        # Fixtures live under /tmp, which may itself contain a source checkout.
        self.config["_project_root"] = str(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def write_config(self):
        self.config_path.write_text(yaml.safe_dump(self.config_data, sort_keys=False), encoding="utf-8")

    def rewrite_manifest(self, item_id: str, **updates):
        item = next(item for item in self.items if item["item_id"] == item_id)
        path = Path(item["manifest"])
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update(updates)
        path.write_text(json.dumps(value), encoding="utf-8")
        return value, path


class TestEvidenceSelection(C06Fixture):
    def test_index_covers_all_publication_items(self):
        self.assertEqual(set(item["item_id"] for item in self.config["items"]), set(ITEM_IDS))

    def test_selected_input_validation_reports_missing_corrupt_and_unproven_inputs(self):
        self.items[0]["artifact"] = ""
        self.items[1]["sha256"] = "0" * 64
        self.write_config()
        rows = validate_selection(load_publication_config(self.config_path), strict_empirical=False)
        self.assertIn("missing", rows[0]["diagnostic"])
        self.assertIn("sha256 mismatch", rows[1]["diagnostic"])
        self.assertEqual(rows[0]["status"], "MISSING")
        self.assertEqual(rows[1]["status"], "INVALID")
        for item_id, update, expected in (
            ("T3", {"is_smoke_test": False}, "manifest unreadable"),
            ("T4", {"is_smoke_test": None}, "is_smoke_test=false"),
            ("T5", {"protocol": "wrong-protocol"}, "protocol mismatch"),
            ("T8", {"evidence_kind": None}, "evidence_kind must be empirical or structural"),
        ):
            item = next(item for item in self.items if item["item_id"] == item_id)
            path = Path(item["manifest"])
            if item_id == "T3":
                path.write_text("{invalid json", encoding="utf-8")
            else:
                self.rewrite_manifest(item_id, **update)
            self.write_config()
            row = next(row for row in validate_selection(load_publication_config(self.config_path), strict_empirical=False)
                       if row["item_id"] == item_id)
            self.assertEqual(row["status"], "INVALID")
            self.assertIn(item_id, row["diagnostic"])
            self.assertIn(str(path), row["diagnostic"])
            self.assertIn(expected, row["diagnostic"])
        artifact_item = next(item for item in self.items if item["item_id"] == "T9")
        Path(artifact_item["artifact"]).write_bytes(b"\xff\xfe")
        artifact_item["sha256"] = sha256_file(Path(artifact_item["artifact"]))
        self.rewrite_manifest("T9", artifact_sha256=artifact_item["sha256"])
        self.write_config()
        unreadable = next(row for row in validate_selection(load_publication_config(self.config_path), strict_empirical=False)
                          if row["item_id"] == "T9")
        self.assertEqual(unreadable["status"], "INVALID")
        self.assertIn("artifact unreadable", unreadable["diagnostic"])
        self.assertIn(str(artifact_item["artifact"]), unreadable["diagnostic"])

    def test_scientific_identity_and_composition_must_match(self):
        self.items[0]["expected"]["scientific_identity"]["class_order_sha256"] = "expected-order"
        self.write_config()
        identity = dict(self.identity)
        identity["class_order_sha256"] = "different-order"
        self.rewrite_manifest("T1", scientific_identity=identity)
        row = validate_selection(load_publication_config(self.config_path), strict_empirical=False)[0]
        self.assertIn("class_order_sha256", row["diagnostic"])
        mismatch_cases = {
            "dataset_sha256": "different-dataset", "population_id": "different-population", "split_id": "different-split",
            "seed": 7, "noise_conditions": [0.02, 0.05],
            "model_variants": ["different_model"],
            "checkpoint_sha256s": {"fixture_model": "different-checkpoint"},
        }
        for field, expected in mismatch_cases.items():
            item = next(item for item in self.items if item["item_id"] == "T6")
            item["expected"]["scientific_identity"] = {**self.identity, field: expected}
            self.write_config()
            row = next(row for row in validate_selection(load_publication_config(self.config_path), strict_empirical=False)
                       if row["item_id"] == "T6")
            self.assertEqual(row["status"], "INVALID", field)
            self.assertIn(field, row["diagnostic"], field)
        self.items[5]["expected"]["scientific_identity"] = dict(self.identity)

    def test_strict_gate_fails_closed_per_required_item(self):
        f5_artifact = self.items[-1]["artifact"]
        self.items[-1]["artifact"] = ""
        self.write_config()
        with self.assertRaisesRegex(PublicationValidationError, "F5"):
            validate_selection(load_publication_config(self.config_path), strict_empirical=True)
        self.items[-1]["artifact"] = f5_artifact
        invalid_cases = (
            ("T3", {"sha256": "0" * 64}, None),
            ("T4", {}, {"is_smoke_test": True}),
            ("T5", {}, {"status": "PARTIAL"}),
            ("T6", {}, {"inputs": []}),
        )
        for item_id, item_updates, manifest_updates in invalid_cases:
            item = next(item for item in self.items if item["item_id"] == item_id)
            item.update(item_updates)
            if manifest_updates:
                self.rewrite_manifest(item_id, **manifest_updates)
            self.write_config()
            with self.subTest(item_id=item_id), self.assertRaisesRegex(PublicationValidationError, item_id):
                validate_selection(load_publication_config(self.config_path), strict_empirical=True)
        child = self.root / "evidence" / "strict-transitive.json"
        child.write_text(json.dumps({"is_smoke_test": True}), encoding="utf-8")
        t7, _ = self.rewrite_manifest("T7")
        t7["inputs"].append({"path": str(child), "sha256": sha256_file(child)})
        Path(next(item for item in self.items if item["item_id"] == "T7")["manifest"]).write_text(json.dumps(t7), encoding="utf-8")
        self.write_config()
        with self.assertRaisesRegex(PublicationValidationError, "synthetic dependency"):
            validate_selection(load_publication_config(self.config_path), strict_empirical=True)

    def test_table_11_requires_architecture_correlation_semantics(self):
        item = next(item for item in self.items if item["item_id"] == "T11")
        item["transformation"] = "table_11_cooccurrence_synergy"
        self.write_config()
        rows = validate_selection(load_publication_config(self.config_path), strict_empirical=False)
        t11 = next(row for row in rows if row["item_id"] == "T11")
        self.assertIn("transformation mismatch", t11["diagnostic"])

    def test_only_selected_and_transitive_inputs_affect_admissibility(self):
        unselected = self.root / "evidence" / "unselected-smoke.json"
        unselected.write_text('{"is_smoke_test": true}', encoding="utf-8")
        rows = validate_selection(self.config, strict_empirical=False)
        self.assertTrue(all(row["status"] in {"EMPIRICAL_VERIFIED", "DERIVED_VERIFIED"} for row in rows))
        smoke = self.root / "evidence" / "selected-dependency.json"
        smoke.write_text('{"is_smoke_test": true}', encoding="utf-8")
        item = next(item for item in self.items if item["item_id"] == "T1")
        value, _ = self.rewrite_manifest("T1")
        value["inputs"].append({"path": str(smoke), "sha256": sha256_file(smoke)})
        Path(item["manifest"]).write_text(json.dumps(value), encoding="utf-8")
        rows = validate_selection(self.config, strict_empirical=False)
        self.assertEqual(rows[0]["status"], "SYNTHETIC")
        self.assertIn("synthetic dependency", rows[0]["diagnostic"])
        blocked = assemble_publication(self.config_path, self.root / "transitive-blocked", strict_empirical=True)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertEqual(blocked["gates"]["empirical"], "BLOCKED")
        self.assertTrue((self.root / "transitive-blocked" / "evidence_matrix.csv").is_file())

    def test_partial_selection_is_diagnostic_and_never_complete(self):
        self.items[2]["artifact"] = ""
        self.write_config()
        out = self.root / "run"
        result = assemble_publication(self.config_path, out, strict_empirical=True, diagnostic_only=True)
        self.assertEqual(result["status"], "BLOCKED")
        matrix = pd.read_csv(result["matrix"])
        self.assertEqual(len(matrix), 17)
        self.assertEqual(matrix.loc[matrix.item_id == "T3", "status"].iloc[0], "MISSING")
        self.assertFalse((out / "tables").exists())


class TestPublicationTables(C06Fixture):
    def test_tables_are_derived_and_historical_values_are_separate(self):
        outputs = render_tables(self.config, self.root / "tables", validate_selection(self.config))
        self.assertEqual(sum(key.startswith("T") and key.endswith(":csv") for key in outputs), 12)
        self.assertEqual(sum(key.startswith("T") and key.endswith(":markdown") for key in outputs), 12)
        self.assertEqual(sum(key.startswith("T") and key.endswith(":latex") for key in outputs), 12)
        self.assertTrue(all(outputs[f"T{i}:markdown"].is_file() and outputs[f"T{i}:latex"].is_file() for i in range(1, 13)))
        generated = pd.read_csv(outputs["T1:csv"])
        original = pd.read_csv(Path(self.items[0]["artifact"]))
        pd.testing.assert_frame_equal(generated, original)
        matrix = evidence_matrix(self.config, validate_selection(self.config))
        self.assertTrue((matrix.historical_status == "HISTORICAL_UNREPRODUCED").all())
        report = pd.read_csv(self.root / "tables" / "reconciliation_report.csv")
        self.assertIn("HISTORICAL_UNREPRODUCED", set(report.status))

    def test_structural_counts_are_separated_from_unmeasured_costs(self):
        t2 = next(item for item in self.items if item["item_id"] == "T2")
        pd.DataFrame({"model": ["ResNet50"], "parameters_m": [43.99], "backbone_m": [15.96]}).to_csv(t2["artifact"], index=False)
        t2["sha256"] = sha256_file(Path(t2["artifact"]))
        t2["expected"]["columns"] = ["model", "parameters_m", "backbone_m"]
        structural_identity = dict(t2["expected"]["scientific_identity"])
        structural_identity["checkpoint_sha256s"] = "n/a - structural parameter count does not load a checkpoint"
        t2["expected"]["scientific_identity"] = structural_identity
        self.rewrite_manifest("T2", artifact_sha256=t2["sha256"], evidence_kind="structural",
                              scientific_identity=structural_identity, checkpoints=[])
        self.write_config()
        source_path = self.root / "evidence" / "source.txt"
        config_hash = sha256_file(self.config_path)
        manifest, manifest_path = self.rewrite_manifest("T2")
        manifest["inputs"].append({"path": str(self.config_path), "sha256": config_hash})
        manifest["derivation"] = {
            "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "config": {"path": str(self.config_path), "sha256": config_hash},
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        rows = validate_selection(load_publication_config(self.config_path), strict_empirical=False)
        row = rows[1]
        self.assertEqual(row["status"], "DERIVED_VERIFIED")
        rendered = render_tables(load_publication_config(self.config_path), self.root / "table-output", rows)
        frame = pd.read_csv(rendered["T2:csv"])
        self.assertNotIn("latency_ms", frame.columns)
        self.assertNotIn("gflops", frame.columns)
        manifest, manifest_path = self.rewrite_manifest("T2", n_not_applicable_reason="parameter count is structural, not a sample metric")
        manifest.pop("n", None)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        rows = validate_selection(load_publication_config(self.config_path), strict_empirical=False)
        structural_row = rows[1]
        self.assertIsNone(structural_row["n"])
        self.assertIn("structural, not a sample metric", structural_row["n_not_applicable_reason"])
        self.assertEqual(set(manifest["derivation"]), {"source", "config"})

    def test_metrics_preserve_estimands_support_and_protocol_strata(self):
        self.rewrite_manifest("T3", estimand="top1_accuracy", metric_unit="percent",
                              unsupported_class_rule="include all declared classes; zero score without predictions",
                              protocol_stratum="fixture-protocol", ci_population="fixture CI queries",
                              calibration_population="fixture calibration validation split")
        rows = validate_selection(self.config)
        matrix = evidence_matrix(self.config, rows)
        self.assertEqual(matrix.loc[matrix.item_id == "T3", "n"].iloc[0], 10)
        self.assertEqual(matrix.loc[matrix.item_id == "T3", "support"].iloc[0], 5)
        self.assertEqual(matrix.loc[matrix.item_id == "T3", "denominator"].iloc[0], 10)
        frame = pd.read_csv(self.items[2]["artifact"])
        self.assertEqual(int(frame.denominator_n.iloc[0]), 10)
        manifest = json.loads(Path(self.items[2]["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["estimand"], "top1_accuracy")
        self.assertEqual(manifest["metric_unit"], "percent")
        self.assertEqual(manifest["unsupported_class_rule"], "include all declared classes; zero score without predictions")
        self.assertEqual(manifest["protocol_stratum"], "fixture-protocol")
        self.assertEqual(manifest["ci_population"], "fixture CI queries")
        self.assertEqual(manifest["calibration_population"], "fixture calibration validation split")

    def test_discrepancies_use_compatible_units_and_keep_full_precision(self):
        pp = reconcile_value(84.881, 84.88, observed_unit="%", historical_unit="%",
                             observed_protocol="p1", historical_protocol="p1")
        self.assertEqual(pp["difference"], "0.001")
        self.assertEqual(pp["unit"], "pp")
        incompatible = reconcile_value(80, 81, observed_unit="%", historical_unit="%",
                                       observed_protocol="p1", historical_protocol="p2")
        self.assertIsNone(incompatible["difference"])
        self.assertEqual(incompatible["status"], "INCOMPATIBLE")


class TestPublicationFigures(C06Fixture):
    def test_figures_render_from_validated_inputs_without_fallbacks(self):
        outputs = render_figures(self.config, self.root / "figures", validate_selection(self.config))
        self.assertEqual({key for key in outputs if key.startswith("F") and not key.startswith("F3:panel_")},
                         {f"F{i}" for i in range(1, 6)})
        self.assertEqual(sum(key.startswith("F3:panel_") for key in outputs), 5)
        rendered = json.loads(outputs["figure_render_manifest"].read_text(encoding="utf-8"))
        for item_id in (f"F{i}" for i in range(1, 6)):
            png = outputs[item_id]
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(png.with_suffix(".pdf").read_bytes().startswith(b"%PDF-"))
            item = next(item for item in self.config["items"] if item["item_id"] == item_id)
            self.assertTrue(item["plot"], f"{item_id} render parameters must be selected in config")
            provenance = rendered["rendered_figures"][item_id]
            self.assertEqual(provenance["render_source"], "selected_artifact")
            self.assertEqual(provenance["source_path"], item["artifact"])
            self.assertEqual(provenance["source_sha256"], item["sha256"])
            self.assertEqual(provenance["plot_parameters"], item["plot"])

    def test_figure_3_panel_identity_uses_variant_provenance(self):
        item = next(item for item in self.items if item["item_id"] == "F3")
        valid_f3 = next(row for row in validate_selection(self.config) if row["item_id"] == "F3")
        self.assertEqual(valid_f3["panel_variants"]["d"], "ResNet50_F")
        self.assertEqual(valid_f3["panel_variants"]["e"], "ConvNeXt_F")
        self.assertIn("ResNet50_F", valid_f3["checkpoint_sha256s"])
        self.assertIn("ConvNeXt_F", valid_f3["checkpoint_sha256s"])
        self.rewrite_manifest("F3", panel_variants={"d": "ResNet50", "e": "ConvNeXt_F"})
        self.write_config()
        rows = validate_selection(load_publication_config(self.config_path), strict_empirical=False)
        self.assertIn("panel_variants", next(row for row in rows if row["item_id"] == "F3")["diagnostic"])

    def test_qualitative_and_statistical_selections_are_recorded(self):
        rows = validate_selection(self.config)
        for row in rows:
            if row["item_id"].startswith("F"):
                self.assertEqual(row["status"], "EMPIRICAL_VERIFIED")
        manifest = json.loads(Path(next(item for item in self.items if item["item_id"] == "F2")["manifest"]).read_text())
        self.assertEqual(set(manifest["selection"]), {"ids", "levels", "sampling", "seed", "filters"})
        self.assertEqual(manifest["selection"], {
            "ids": ["fixture-1", "fixture-2"], "levels": ["Species"],
            "sampling": "all fixture rows", "seed": 42, "filters": [],
        })
        self.assertEqual(next(item for item in self.items if item["item_id"] == "F2")["plot"]["interpretation"], "qualitative visualization only")
        self.assertEqual(next(item for item in self.items if item["item_id"] == "F4")["plot"]["interpretation"], "exploratory triplet composition")

    def test_render_and_case_sensitive_path_failures_block_completion(self):
        from viral_cnn.reporting.figures import verify_latex_figure_paths
        tex = self.root / "case.tex"
        (self.root / "Fig.png").write_bytes(b"PNG")
        tex.write_text(r"\includegraphics{fig.png}", encoding="utf-8")
        matches, mismatches = verify_latex_figure_paths(tex, self.root)
        self.assertFalse(matches)
        self.assertEqual(mismatches[0]["status"], "case_mismatch")
        bad_item = next(item for item in self.config["items"] if item["item_id"] == "F1")
        bad_item["plot"] = {"x": "absent", "y": "y"}
        with self.assertRaisesRegex(PublicationValidationError, "x/y columns absent"):
            render_figures(self.config, self.root / "bad-figures", validate_selection(self.config))
        self.write_config()
        result = assemble_publication(self.config_path, self.root / "failed-render", strict_empirical=True)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["gates"]["local"], "BLOCKED")
        self.assertTrue(Path(result["outputs"]["assembly_errors"]).is_file())


class TestPublicationManuscript(C06Fixture):
    def setUp(self):
        super().setUp()
        review = self.root / "paper" / "review.tex"
        submission = self.root / "paper" / "submission.tex"
        review.parent.mkdir()
        tables = "\n".join(r"\begin{table}[h!]\centering\caption{Table}\begin{tabular}{cc}a&b\end{tabular}\end{table}" for _ in range(12))
        figures = "\n".join(r"\begin{figure}[h!]\centering\includegraphics{figures/fig_1.png}\end{figure}" for _ in range(5))
        base = (
            r"\documentclass{article}" + "\n" + r"\begin{document}" + "\n"
            + r"\begin{abstract}\textbf{Results}: old numeric result.\par\vspace{0.5em}\noindent\textbf{Conclusions}: old text.\end{abstract}" + "\n"
            + r"\section{Methods}" + "\nOld method.\n" + r"\section{Results}" + "\n" + tables + "\n" + figures + "\n"
            + r"\section{Discussion}" + "\nOld claim 99%.\n" + r"\section{Conclusions}" + "\nOld conclusion.\n"
            + r"\subsection*{Availability of data and materials}" + "\nOpen.\n" + r"\end{document}"
        )
        review.write_text(base + "\n" + "\n".join(r"\notavdois{note}" for _ in range(56)), encoding="utf-8")
        submission.write_text(base, encoding="utf-8")
        self.config_data["historical_source"] = {"manuscript_review": "paper/review.tex", "manuscript_submission": "paper/submission.tex"}
        self.write_config()
        self.config = load_publication_config(self.config_path)
        self.config["_project_root"] = str(self.root)

    def test_both_variants_consume_the_same_package_results(self):
        render_tables(self.config, self.root / "tables", validate_selection(self.config))
        render_figures(self.config, self.root / "figures", validate_selection(self.config))
        manuscripts = render_manuscript_variants(self.config, self.root)
        for path in manuscripts.values():
            content = path.read_text(encoding="utf-8")
            self.assertEqual(content.count(r"\input{tables/table_"), 12)
            self.assertNotIn("old numeric result", content)
            self.assertNotIn("Old claim 99%", content)
        review = manuscripts["review"].read_text(encoding="utf-8")
        submission = manuscripts["submission"].read_text(encoding="utf-8")
        review_body = re.sub(r"\\section\*\{Editorial review notes\}.*(?=\\end\{document\})", "", review, flags=re.DOTALL)
        self.assertEqual(review_body.strip(), submission.strip())
        self.assertNotIn(r"\begin{tabular}", review_body)
        self.assertEqual(
            re.findall(r"\\input\{tables/table_\d+\.tex\}", review_body),
            re.findall(r"\\input\{tables/table_\d+\.tex\}", submission),
        )

    def test_methods_and_results_follow_observed_protocols(self):
        self.rewrite_manifest("T3", ci_population="fixture CI queries",
                              calibration_population="fixture calibration validation split")
        paths = render_manuscript_variants(self.config, self.root)
        content = paths["submission"].read_text(encoding="utf-8")
        for phrase in ("closed-set", "not a genome-disjoint test", r"3\epsilon/4", "cosine distance", "UMAP is used only",
                       "Triplet composition summaries are exploratory", "confidence-interval/calibration populations"):
            self.assertIn(phrase.lower(), content.lower())
        for selected in ("fixture_model=", "fixture CI queries", "fixture calibration validation split"):
            self.assertIn(selected, content)

    def test_inconclusive_evidence_limits_genome_and_causal_claims(self):
        content = render_manuscript_variants(self.config, self.root)["submission"].read_text(encoding="utf-8")
        abstract, rest = content.split(r"\end{abstract}", 1)
        discussion = rest.split(r"\section{Discussion}", 1)[1].split(r"\section{Conclusions}", 1)[0]
        conclusions = rest.split(r"\section{Conclusions}", 1)[1].split(r"\subsection*{Availability", 1)[0]
        self.assertIn("does not establish performance on unseen genomes", abstract)
        self.assertIn("remain unresolved rather than being inferred", discussion)
        self.assertIn("does not establish performance on unseen genomes", conclusions)
        self.assertNotIn("due to length bias", abstract + discussion + conclusions)

    def test_optional_c05_state_limits_claims_without_blocking_package(self):
        self.config_data["c05_status"] = "NOT_RUN"
        self.write_config()
        config = load_publication_config(self.config_path)
        config["_project_root"] = str(self.root)
        content = render_manuscript_variants(config, self.root)["review"].read_text(encoding="utf-8")
        self.assertIn("C05 scratch comparison is NOT_RUN", content)
        self.assertTrue(all(row["status"] == "EMPIRICAL_VERIFIED" for row in validate_selection(self.config)))
        out = self.root / "c05-not-run-package"
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
            result = assemble_publication(self.config_path, out, strict_empirical=True, command="fixture assembly")
        self.assertEqual(result["status"], "ASSEMBLED_REVIEW_REQUIRED")
        self.assertEqual(result["gates"]["empirical"], "PASS")
        self.assertIn("C05 scratch comparison is NOT_RUN",
                      (out / "manuscript_submission.tex").read_text(encoding="utf-8"))

    def test_historical_values_remain_unreproduced_and_noncomplete(self):
        matrix = evidence_matrix(self.config, validate_selection(self.config))
        self.assertTrue((matrix.historical_status == "HISTORICAL_UNREPRODUCED").all())
        self.assertTrue((matrix.claim_role == "computed").all())
        result = assemble_publication(self.config_path, self.root / "historical-diagnostic",
                                      strict_empirical=False, diagnostic_only=True)
        self.assertEqual(result["status"], "DIAGNOSTIC")
        self.assertEqual(json.loads(Path(result["manifest"]).read_text())["gate_states"]["empirical"], "PASS")

    def test_review_notes_are_preserved_and_submission_is_clean(self):
        paths = render_manuscript_variants(self.config, self.root)
        review = paths["review"].read_text(encoding="utf-8")
        submission = paths["submission"].read_text(encoding="utf-8")
        self.assertEqual(review.count(r"\notavdois{"), 56)
        self.assertNotIn(r"\notavdois{", submission)

    def test_editorial_review_status_is_explicit_and_independent_of_compile(self):
        path = write_editorial_review(self.root)
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual({report[k]["status"] for k in ("references", "affiliations", "wording")}, {"PENDING"})
        self.assertFalse(report["compilation_is_approval"])


class TestPackageIsolation(C06Fixture):
    def test_cli_and_notebook_use_isolated_run_namespace(self):
        first = reserve_run_directory(self.root, "run-a", "publication")
        self.assertEqual(first, self.root / "results_reproduction" / "runs" / "run-a" / "publication")
        with self.assertRaises(FileExistsError):
            reserve_run_directory(self.root, "run-a", "publication")
        self.items[0]["artifact"] = ""
        self.write_config()
        from scripts import compile_paper, generate_paper_figures, generate_paper_tables
        for module in (generate_paper_tables, generate_paper_figures, compile_paper):
            self.assertIs(module.assemble_publication, assemble_publication)
            def reserve(_root, run_id, _interface):
                target = self.root / "cli-runs" / run_id / "publication"
                target.mkdir(parents=True)
                return target
            with patch.object(module, "reserve_run_directory", side_effect=reserve), \
                 patch("sys.argv", [module.__file__, "--mode", "strict", "--config", str(self.config_path), "--run-id", module.__name__.split(".")[-1]]):
                self.assertEqual(module.main(), 2)
            target = self.root / "cli-runs" / module.__name__.split(".")[-1] / "publication"
            self.assertTrue((target / "evidence_matrix.csv").is_file())

    def test_master_notebook_bootstraps_src_without_heavy_side_effects(self):
        notebook = Path(__file__).resolve().parents[1] / "notebooks" / "reproduction" / "00_regenerate_all_paper_artifacts.ipynb"
        data = json.loads(notebook.read_text(encoding="utf-8"))
        first = "".join(data["cells"][0]["source"])
        self.assertLess(first.index("sys.path.insert"), first.index("from viral_cnn"))
        source = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])
        self.assertNotIn("torch.load", source)
        self.assertNotIn("requests.get", source.lower())
        self.assertNotIn("urlopen(", source.lower())
        self.assertNotIn("train(", source)
        self.assertIn("reserve_run_directory", source)
        self.assertIn("assemble_publication", source)

    def test_compile_artifacts_are_isolated_and_historical_sources_unchanged(self):
        from viral_cnn.reporting.publication import validate_manuscript_paths
        tex = self.root / "manuscript_submission.tex"
        tex.write_text("\\documentclass{article}\n\\input{tables/table_1.tex}\n\\includegraphics{figures/fig_1.png}", encoding="utf-8")
        (self.root / "tables").mkdir()
        (self.root / "tables" / "table_1.tex").write_text("table", encoding="utf-8")
        (self.root / "figures").mkdir()
        (self.root / "figures" / "fig_1.png").write_bytes(b"PNG")
        self.assertEqual(validate_manuscript_paths({"submission": tex}, self.root), {})
        bad_tex = self.root / "bad_submission.tex"
        bad_tex.write_text("\\includegraphics{figures/Fig_1.png}", encoding="utf-8")
        self.assertIn("submission", validate_manuscript_paths({"submission": bad_tex}, self.root))
        old_hash = sha256_file(tex)
        paths = {"submission": tex}
        real_run = subprocess.run
        def fake_run(command, **kwargs):
            if Path(command[0]).name != "pdflatex":
                return real_run(command, **kwargs)
            out_arg = next(arg.split("=", 1)[1] for arg in command if arg.startswith("-output-directory="))
            stem = Path(command[-1]).stem
            out = Path(out_arg)
            (out / f"{stem}.pdf").write_bytes(b"%PDF-fixture")
            return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
        with patch("viral_cnn.reporting.publication.subprocess.run", side_effect=fake_run):
            result = compile_manuscript_package(paths, self.root / "compiled")
        self.assertEqual(result["submission"]["returncode"], 0)
        self.assertTrue(Path(result["submission"]["pdf"]).is_file())
        self.assertTrue(Path(result["submission"]["log"]).is_file())
        self.assertEqual(sha256_file(tex), old_hash)

    def test_complete_fixture_assembly_renders_and_compiles_in_reserved_package(self):
        paper = self.root / "paper"
        paper.mkdir()
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
        review = paper / "review.tex"
        submission = paper / "submission.tex"
        review.write_text(tex + "\n" + "\n".join(r"\notavdois{n}" for _ in range(56)), encoding="utf-8")
        submission.write_text(tex, encoding="utf-8")
        (paper / "references.bib").write_text("", encoding="utf-8")
        self.config_data["historical_source"] = {
            "manuscript_review": "paper/review.tex", "manuscript_submission": "paper/submission.tex"
        }
        self.config_data["c05_status"] = "NOT_RUN"
        self.write_config()
        run = reserve_run_directory(self.root, "complete-fixture", "publication")

        real_run = subprocess.run
        def fake_run(command, **kwargs):
            if Path(command[0]).name != "pdflatex":
                return real_run(command, **kwargs)
            output_arg = next((arg.split("=", 1)[1] for arg in command if arg.startswith("-output-directory=")), None)
            if output_arg is None:
                raise AssertionError(f"unexpected compiler command: {command!r}")
            output = Path(output_arg)
            stem = Path(command[-1]).stem
            (output / f"{stem}.pdf").write_bytes(b"%PDF-fixture")
            return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

        with patch("viral_cnn.reporting.publication._git_state", return_value=("fixture-commit", True)), \
             patch("viral_cnn.reporting.publication.subprocess.run", side_effect=fake_run):
            result = assemble_publication(self.config_path, run, strict_empirical=True)
        assembly_error = Path(result["outputs"]["assembly_errors"]).read_text(encoding="utf-8") if "assembly_errors" in result.get("outputs", {}) else ""
        self.assertEqual(result["status"], "ASSEMBLED_REVIEW_REQUIRED", msg=assembly_error)
        self.assertEqual(result["gates"]["local"], "PASS")
        self.assertEqual(len(result["rows"]), 17)
        review_text = (run / "manuscript_review.tex").read_text(encoding="utf-8")
        self.assertIn("C05 scratch comparison is NOT_RUN", review_text)
        for mode in ("review", "submission"):
            self.assertTrue((run / mode / f"manuscript_{mode}.pdf").is_file())
            self.assertTrue((run / mode / "compile.log").is_file())


class TestPublicationManifest(C06Fixture):
    def test_manifest_records_inputs_outputs_versions_and_gates(self):
        output = self.root / "manifest-run"
        output.mkdir()
        artifact = output / "matrix.csv"
        artifact.write_text("item_id,status\nT1,MISSING\n", encoding="utf-8")
        manifest_path = write_publication_manifest(
            self.config, output, validate_selection(self.config), {"matrix": artifact},
            command="assemble --mode diagnostic", interface="cli",
            gate_states={"local": "PASS", "empirical": "BLOCKED", "scientific": "PENDING", "editorial": "PENDING"},
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["git_commit"])
        self.assertIn("command", manifest)
        self.assertEqual(manifest["interface"], "cli")
        self.assertIn("configuration_sha256", manifest)
        self.assertTrue(manifest["git_dirty"])
        self.assertTrue(manifest["inputs"])
        self.assertIn("historical_sources", manifest)
        self.assertEqual(manifest["run_id"], "manifest-run")
        self.assertIn("pandas", manifest["versions"])
        self.assertEqual(manifest["outputs"]["matrix"]["sha256"], sha256_file(artifact))
        self.assertEqual(manifest["gate_states"]["empirical"], "BLOCKED")
        self.assertIn("local", manifest["gate_states"])
        self.assertIn("scientific", manifest["gate_states"])
        self.assertIn("editorial", manifest["gate_states"])
        self.assertEqual(manifest["gate_states"], {
            "local": "PASS", "empirical": "BLOCKED", "scientific": "PENDING", "editorial": "PENDING",
        })

    def test_synthetic_run_and_final_report_do_not_claim_reproduction(self):
        self.rewrite_manifest("T1", is_smoke_test=True)
        self.write_config()
        out = self.root / "diagnostic-run"
        result = assemble_publication(self.config_path, out, strict_empirical=False, diagnostic_only=True,
                                      command="python scripts/assemble_publication.py --mode diagnostic --run-id fixture")
        self.assertEqual(result["status"], "DIAGNOSTIC")
        matrix = pd.read_csv(result["matrix"])
        self.assertEqual(matrix.loc[matrix.item_id == "T1", "status"].iloc[0], "SYNTHETIC")
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["gate_states"]["empirical"], "BLOCKED")
        self.assertIn("command", manifest)
        self.assertEqual(manifest["interface"], "cli")
        self.assertTrue(manifest["inputs"])
        self.assertIn("evidence_matrix", manifest["outputs"])
        self.assertEqual(len(manifest["inputs"]), 17)
        self.assertTrue(any(row["diagnostic"] for row in manifest["inputs"]))
        self.assertTrue(all("manifest_sha256" in row for row in manifest["inputs"]))
        self.assertTrue(all("source_inputs" in row for row in manifest["inputs"]))
        self.assertTrue(any(all("sha256" in item for item in row["source_inputs"])
                            for row in manifest["inputs"] if row["source_inputs"]))
        status = Path(result["outputs"]["publication_status_report"]).read_text(encoding="utf-8")
        self.assertIn("synthetic", status.lower())
        self.assertIn("--mode diagnostic", status)
        self.assertIn(str(result["manifest"]), status)
        self.assertIn("Numeric discrepancies were not computed", status)
        self.assertIn("does not reproduce the article", status)
        self.assertNotIn("tables", result.get("outputs", {}))


if __name__ == "__main__":
    unittest.main()
