"""Offline contracts for C06 selection, evidence, and isolated publication assembly."""

from __future__ import annotations

import hashlib
import json
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
            "dataset_sha256": sha256_file(dataset), "split_id": "fixture-split", "seed": 42,
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

    def test_scientific_identity_and_composition_must_match(self):
        self.items[0]["expected"]["scientific_identity"]["class_order_sha256"] = "expected-order"
        self.write_config()
        identity = dict(self.identity)
        identity["class_order_sha256"] = "different-order"
        self.rewrite_manifest("T1", scientific_identity=identity)
        row = validate_selection(load_publication_config(self.config_path), strict_empirical=False)[0]
        self.assertIn("class_order_sha256", row["diagnostic"])

    def test_strict_gate_fails_closed_per_required_item(self):
        self.items[-1]["artifact"] = ""
        self.write_config()
        with self.assertRaisesRegex(PublicationValidationError, "F5"):
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
        rows = validate_selection(load_publication_config(self.config_path), strict_empirical=False)
        row = rows[1]
        self.assertEqual(row["status"], "DERIVED_VERIFIED")
        rendered = render_tables(load_publication_config(self.config_path), self.root / "table-output", rows)
        frame = pd.read_csv(rendered["T2:csv"])
        self.assertNotIn("latency_ms", frame.columns)
        self.assertNotIn("gflops", frame.columns)

    def test_metrics_preserve_estimands_support_and_protocol_strata(self):
        rows = validate_selection(self.config)
        matrix = evidence_matrix(self.config, rows)
        self.assertEqual(matrix.loc[matrix.item_id == "T3", "n"].iloc[0], 10)
        self.assertEqual(matrix.loc[matrix.item_id == "T3", "support"].iloc[0], 5)
        self.assertEqual(matrix.loc[matrix.item_id == "T3", "denominator"].iloc[0], 10)
        frame = pd.read_csv(self.items[2]["artifact"])
        self.assertEqual(int(frame.denominator_n.iloc[0]), 10)

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
        self.assertEqual({key for key in outputs if not key.startswith("F3:panel_")}, {f"F{i}" for i in range(1, 6)})
        self.assertEqual(sum(key.startswith("F3:panel_") for key in outputs), 5)
        for item_id in (f"F{i}" for i in range(1, 6)):
            png = outputs[item_id]
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(png.with_suffix(".pdf").read_bytes().startswith(b"%PDF-"))

    def test_figure_3_panel_identity_uses_variant_provenance(self):
        item = next(item for item in self.items if item["item_id"] == "F3")
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

    def test_methods_and_results_follow_observed_protocols(self):
        paths = render_manuscript_variants(self.config, self.root)
        content = paths["submission"].read_text(encoding="utf-8")
        for phrase in ("closed-set", "not a genome-disjoint test", r"3\epsilon/4", "cosine distance", "UMAP is used only"):
            self.assertIn(phrase, content)

    def test_inconclusive_evidence_limits_genome_and_causal_claims(self):
        content = render_manuscript_variants(self.config, self.root)["submission"].read_text(encoding="utf-8")
        self.assertIn("does not establish performance on unseen genomes", content)
        self.assertIn("remain unresolved rather than being inferred", content)
        self.assertNotIn("due to length bias", content)

    def test_optional_c05_state_limits_claims_without_blocking_package(self):
        self.config_data["c05_status"] = "NOT_RUN"
        self.write_config()
        config = load_publication_config(self.config_path)
        config["_project_root"] = str(self.root)
        content = render_manuscript_variants(config, self.root)["review"].read_text(encoding="utf-8")
        self.assertIn("C05 scratch comparison is NOT_RUN", content)
        self.assertTrue(all(row["status"] == "EMPIRICAL_VERIFIED" for row in validate_selection(self.config)))

    def test_historical_values_remain_unreproduced_and_noncomplete(self):
        matrix = evidence_matrix(self.config, validate_selection(self.config))
        self.assertTrue((matrix.historical_status == "HISTORICAL_UNREPRODUCED").all())
        self.assertTrue((matrix.claim_role == "computed").all())

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

    def test_compile_artifacts_are_isolated_and_historical_sources_unchanged(self):
        tex = self.root / "manuscript_submission.tex"
        tex.write_text("\\documentclass{article}", encoding="utf-8")
        old_hash = sha256_file(tex)
        paths = {"submission": tex}
        def fake_run(command, **kwargs):
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
        self.write_config()
        run = reserve_run_directory(self.root, "complete-fixture", "publication")

        def fake_run(command, **kwargs):
            output = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("-output-directory=")))
            stem = Path(command[-1]).stem
            (output / f"{stem}.pdf").write_bytes(b"%PDF-fixture")
            return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

        with patch("viral_cnn.reporting.publication._git_state", return_value=("fixture-commit", True)), \
             patch("viral_cnn.reporting.publication.subprocess.run", side_effect=fake_run):
            result = assemble_publication(self.config_path, run, strict_empirical=True)
        self.assertEqual(result["status"], "ASSEMBLED_REVIEW_REQUIRED")
        self.assertEqual(result["gates"]["local"], "PASS")
        self.assertEqual(len(result["rows"]), 17)
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
        self.assertEqual(manifest["run_id"], "manifest-run")
        self.assertIn("pandas", manifest["versions"])
        self.assertEqual(manifest["outputs"]["matrix"]["sha256"], sha256_file(artifact))
        self.assertEqual(manifest["gate_states"]["empirical"], "BLOCKED")

    def test_synthetic_run_and_final_report_do_not_claim_reproduction(self):
        self.rewrite_manifest("T1", is_smoke_test=True)
        self.write_config()
        out = self.root / "diagnostic-run"
        result = assemble_publication(self.config_path, out, strict_empirical=False, diagnostic_only=True)
        self.assertEqual(result["status"], "DIAGNOSTIC")
        matrix = pd.read_csv(result["matrix"])
        self.assertEqual(matrix.loc[matrix.item_id == "T1", "status"].iloc[0], "SYNTHETIC")
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["gate_states"]["empirical"], "BLOCKED")
        self.assertNotIn("tables", result.get("outputs", {}))


if __name__ == "__main__":
    unittest.main()
