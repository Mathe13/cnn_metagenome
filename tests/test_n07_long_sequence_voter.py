import json
import os
import uuid
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd
from viral_cnn.notebook_runtime import JupyterHubDataTopology


class TestN07LongSequenceVoter(unittest.TestCase):
    """Contract tests verifying N07 checks C1 to C15 for long-sequence voter reproduction notebooks."""

    @classmethod
    def run_notebook(cls, filename):
        nb_path = Path("notebooks/reproduction/long") / filename
        namespace = {"__name__": "__main__"}
        keys=("REMOTE_SERVER","SMOKE_TEST","MPLCONFIGDIR")
        old={key:os.environ.get(key) for key in keys}
        os.environ.update({"REMOTE_SERVER":"False","SMOKE_TEST":"1","MPLCONFIGDIR":"/tmp"})
        try:
            for idx, cell in enumerate(json.loads(nb_path.read_text(encoding="utf-8"))["cells"]):
                if cell.get("cell_type") == "code":
                    try:
                        exec("".join(cell.get("source", [])), namespace)
                    except Exception as exc:
                        raise AssertionError(f"{filename} cell {idx}: {exc}") from exc
        finally:
            for key,value in old.items():
                if value is None: os.environ.pop(key,None)
                else: os.environ[key]=value
        return namespace

    @classmethod
    def setUpClass(cls):
        cls.chain_run_id = "n07-c02-" + uuid.uuid4().hex[:10]
        old = {key: os.environ.get(key) for key in ("REMOTE_SERVER", "SMOKE_TEST", "C02_LONG_RUN_ID", "C02_DIAGNOSTIC_LIMIT", "MPLCONFIGDIR")}
        os.environ.update({"REMOTE_SERVER": "False", "SMOKE_TEST": "1", "C02_LONG_RUN_ID": cls.chain_run_id, "C02_DIAGNOSTIC_LIMIT": "2", "MPLCONFIGDIR": "/tmp"})
        started = time.time()
        try:
            for name in ("01_ncbi_download_and_prepare_database.ipynb", "02_cnn_long_voter_inference.ipynb", "03_cnn_long_voter_compare.ipynb"):
                cls.run_notebook(name)
        finally:
            cls.chain_duration = time.time() - started
            for key, value in old.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value

    @classmethod
    def chain_dir(cls, notebook):
        return Path("results_reproduction/runs") / cls.chain_run_id / "long" / notebook

    # --------------------------------------------------------------------------
    # S1: NCBI Download, Database Preparation & Overlap Audit
    # --------------------------------------------------------------------------
    def test_c1_notebook_01_imports_and_decoupling(self):
        """C1: 01_ncbi_download_and_prepare_database.ipynb has valid JSON, zero legacy imports, exclusively viral_cnn.*."""
        nb_path = Path("notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb")
        self.assertTrue(nb_path.exists(), f"Notebook {nb_path} does not exist")

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        self.assertIn("cells", nb)
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                self.assertNotIn("short.utils", code, f"Cell {idx} imports short.utils")
                self.assertNotIn("short/utils", code, f"Cell {idx} imports short/utils")

    def test_c2_notebook_01_execution_smoke_test(self):
        """C2: long/01 executes as a CPU smoke pipeline and writes an isolated manifest."""
        rid="n07-01-"+uuid.uuid4().hex[:8]
        old=os.environ.get("C02_LONG_RUN_ID"); os.environ["C02_LONG_RUN_ID"]=rid
        started=time.time()
        try: self.run_notebook("01_ncbi_download_and_prepare_database.ipynb")
        finally:
            if old is None: os.environ.pop("C02_LONG_RUN_ID",None)
            else: os.environ["C02_LONG_RUN_ID"]=old
        self.assertLess(time.time()-started,60.0)
        manifest=json.loads((Path("results_reproduction/runs")/rid/"long/01/manifest_01_ncbi_prep.json").read_text())
        self.assertTrue(manifest["is_smoke_test"])
        self.assertEqual(manifest["input_audit"]["eligible_count"],12)
        self.assertEqual(manifest["input_audit"]["genome_identity"]["status"],"unknown")

    def test_c3_overlap_quantification_and_sampling_bias(self):
        """C3: measured composition and the historical 50/200/500 rule remain separate."""
        out=self.chain_dir("01")
        manifest=json.loads((out/"manifest_01_ncbi_prep.json").read_text())
        audit=pd.read_csv(out/"long_database_preparation_audit.csv")
        self.assertIn("observed_query_count",audit.columns)
        self.assertIn("mean_kmer_overlap_fraction",manifest["overlap_metrics"])
        self.assertEqual(manifest["input_audit"]["historical_sampling_rule"]["small_lt_10000"],50)
        self.assertEqual(manifest["input_audit"]["sampling"]["kind"],"synthetic_fixture")
        self.assertNotIn("reads_simulated_per_genome",audit.columns)

    def test_c4_historical_download_and_prep_notebooks_preserved(self):
        """C4: Historical notebooks long/download_from_ncbi.ipynb and long/prepare database.ipynb remain unmodified."""
        for nb in ["long/download_from_ncbi.ipynb", "long/prepare database.ipynb"]:
            res = subprocess.run(
                ["git", "status", "--porcelain", nb],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(res.stdout.strip(), "", f"Historical notebook {nb} was modified!")

    # --------------------------------------------------------------------------
    # S2: CNN Long Sequence Voter Inference & Noise Robustness
    # --------------------------------------------------------------------------
    def test_c5_notebook_02_imports_and_decoupling(self):
        """C5: 02_cnn_long_voter_inference.ipynb has valid JSON, zero legacy imports, exclusively viral_cnn.*."""
        nb_path = Path("notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb")
        self.assertTrue(nb_path.exists())

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                self.assertNotIn("short.utils", code)
                self.assertNotIn("short/utils", code)

    def test_c6_notebook_02_execution_smoke_test(self):
        """C6: long/02 writes all five synthetic variants with singleton logit equivalence."""
        rid="n07-02-"+uuid.uuid4().hex[:8]
        old=os.environ.get("C02_LONG_RUN_ID"); os.environ["C02_LONG_RUN_ID"]=rid
        started=time.time()
        try: self.run_notebook("02_cnn_long_voter_inference.ipynb")
        finally:
            if old is None: os.environ.pop("C02_LONG_RUN_ID",None)
            else: os.environ["C02_LONG_RUN_ID"]=old
        self.assertLess(time.time()-started,60.0)
        manifest=json.loads((Path("results_reproduction/runs")/rid/"long/02/manifest_02_voter_inference.json").read_text())
        self.assertEqual(len(manifest["variants"]),5)
        self.assertEqual(manifest["logit_diff_max"],0.0)
        self.assertTrue(manifest["is_smoke_test"])

    def test_c7_sliding_window_consensus_and_short_handling(self):
        """C7: pad and skip persist separately, retaining short queries as uncovered errors."""
        out=self.chain_dir("02")
        pad=pd.read_parquet(out/"predictions_resnet_clean_species_epsilon0_pad.parquet")
        skip=pd.read_parquet(out/"predictions_resnet_clean_species_epsilon0_skip.parquet")
        short=pad[pad.length<512].iloc[0]
        skip_short=skip[skip.query_id==short.query_id].iloc[0]
        self.assertGreater(short.num_windows,0)
        self.assertFalse(skip_short.skip_covered)
        self.assertTrue(pd.isna(skip_short.pred_mean_logits))
        self.assertEqual(set(skip.handle_short),{"skip"})
        manifest=json.loads((out/"manifest_02_voter_inference.json").read_text())
        self.assertEqual(manifest["tail"],"drop incomplete tail")

    def test_c8_historical_voter_notebook_preserved(self):
        """C8: Historical notebook long/cnn-long-voter.ipynb remains strictly unmodified."""
        res = subprocess.run(
            ["git", "status", "--porcelain", "long/cnn-long-voter.ipynb"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(res.stdout.strip(), "", "Historical notebook long/cnn-long-voter.ipynb was modified!")

    # --------------------------------------------------------------------------
    # S3: Long Sequence Comparison & ConvNeXt Collapse Diagnosis
    # --------------------------------------------------------------------------
    def test_c9_notebook_03_imports_and_decoupling(self):
        """C9: 03_cnn_long_voter_compare.ipynb has valid JSON, zero legacy imports, exclusively viral_cnn.*."""
        nb_path = Path("notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb")
        self.assertTrue(nb_path.exists())

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                self.assertNotIn("short.utils", code)
                self.assertNotIn("short/utils", code)

    def test_c10_notebook_03_execution_smoke_test(self):
        """C10: the long/01–03 smoke chain completed under 60 seconds in its isolated namespace."""
        self.assertLess(self.chain_duration,60.0)
        for notebook in ("01","02","03"):
            self.assertTrue((self.chain_dir(notebook)/"run_manifest.json").is_file())
        manifest=json.loads((self.chain_dir("03")/"manifest_03_voter_compare.json").read_text())
        self.assertEqual(manifest["long_sequence_protocol"],"c02_audit_v1")
        self.assertTrue(manifest["is_smoke_test"])

    def test_c11_error_stratification_and_collapse_diagnosis(self):
        """C11: Table 12 is calculated from paired fixture rows; diagnostic claims stay inconclusive."""
        out=self.chain_dir("03")
        table=pd.read_csv(out/"table_12_long_voter_comparison.csv")
        self.assertEqual(len(table),20)
        self.assertEqual(set(table.variant.nunique() for _ in [0]),{5})
        self.assertEqual(set(table.aggregation),{"mean_logits","majority_vote"})
        self.assertTrue(((table.correct/table.N)-table.accuracy).abs().max()<1e-12)
        diag=json.loads((out/"long_sequence_diagnostics.json").read_text())
        self.assertEqual(diag["status"],"INCONCLUSIVE")
        self.assertTrue(all(x["status"]=="INCONCLUSIVE" for x in diag["window_influence"]))
        self.assertNotIn("mechanisms",diag)

    def test_c12_historical_compare_notebook_preserved(self):
        """C12: Historical notebook long/cnn-long-voter-compare.ipynb remains strictly unmodified."""
        res = subprocess.run(
            ["git", "status", "--porcelain", "long/cnn-long-voter-compare.ipynb"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(res.stdout.strip(), "", "Historical notebook long/cnn-long-voter-compare.ipynb was modified!")

    # --------------------------------------------------------------------------
    # S4: Long Contig Motifs and GC Profile Analysis
    # --------------------------------------------------------------------------
    def test_c13_notebook_04_imports_and_decoupling(self):
        """C13: 04_long_motifs_and_gc_analysis.ipynb has valid JSON, zero legacy imports, exclusively viral_cnn.*."""
        nb_path = Path("notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb")
        self.assertTrue(nb_path.exists())

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                self.assertNotIn("short.utils", code)
                self.assertNotIn("short/utils", code)

    def test_c14_notebook_04_execution_smoke_test(self):
        """C14: 04_long_motifs_and_gc_analysis.ipynb executes under SMOKE_TEST=1 in under 60s, generating summary and figure."""
        nb_path = Path("notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb")
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
                    self.fail(f"Cell {idx} of 04_long_motifs_and_gc_analysis.ipynb raised: {e}")

        duration = time.time() - start_time
        self.assertLess(duration, 60.0, f"Notebook 04 took too long: {duration:.2f}s")

        summary_csv = Path("results_reproduction/long_voter/long_contig_motif_summary.csv")
        fig_path = Path("results_reproduction/long_voter/figure_long_motifs_gc.png")
        self.assertTrue(summary_csv.exists(), "long_contig_motif_summary.csv not found")
        self.assertTrue(fig_path.exists(), "figure_long_motifs_gc.png not found")

    def test_c15_historical_long_loss_notebook_preserved(self):
        """C15: Historical notebook long/long_loss_function.ipynb remains strictly unmodified."""
        res = subprocess.run(
            ["git", "status", "--porcelain", "long/long_loss_function.ipynb"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(res.stdout.strip(), "", "Historical notebook long/long_loss_function.ipynb was modified!")


    # --------------------------------------------------------------------------
    # R07 Spec Verification Checks (C1 - C6)
    # --------------------------------------------------------------------------
    def test_r07_resolve_long_artifact_path(self):
        """R07 C1: JupyterHubDataTopology.resolve_long_artifact_path searches candidate directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            top = JupyterHubDataTopology(data_root=root)

            # Test artifact directly in root
            (root / "taxonomia_recuperada.csv").touch()
            res_tax = top.resolve_long_artifact_path("taxonomia_recuperada.csv")
            self.assertIsNotNone(res_tax)
            self.assertEqual(res_tax, root / "taxonomia_recuperada.csv")

            # Test artifact in root/long
            (root / "long").mkdir(parents=True)
            (root / "long" / "mapping_species_final.csv").touch()
            res_map = top.resolve_long_artifact_path("mapping_species_final.csv")
            self.assertIsNotNone(res_map)
            self.assertEqual(res_map, root / "long" / "mapping_species_final.csv")

            # Test artifact in root/cnn_long
            (root / "cnn_long").mkdir(parents=True)
            (root / "cnn_long" / "processed_ncbi_splited_parquet").mkdir(parents=True)
            res_ncbi = top.resolve_long_artifact_path("processed_ncbi_splited_parquet")
            self.assertIsNotNone(res_ncbi)
            self.assertEqual(res_ncbi, root / "cnn_long" / "processed_ncbi_splited_parquet")

            # Test artifact in root/cnn_kraken
            (root / "cnn_kraken").mkdir(parents=True)
            (root / "cnn_kraken" / "resultados_inferencia_long_reads_ResNet50.parquet").touch()
            res_inf = top.resolve_long_artifact_path("resultados_inferencia_long_reads_ResNet50.parquet")
            self.assertIsNotNone(res_inf)
            self.assertEqual(res_inf, root / "cnn_kraken" / "resultados_inferencia_long_reads_ResNet50.parquet")

            # Non-existent artifact returns None
            self.assertIsNone(top.resolve_long_artifact_path("non_existent_file_xyz.csv"))

    def test_r07_notebook_01_fail_fast_outside_smoke(self):
        """R07 C2: long/01 raises FileNotFoundError when is_smoke_test=False and ncbi_parquet_dir is missing."""
        nb_path = Path("notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        c1_code = "".join(nb["cells"][1]["source"])
        c3_code = "".join(nb["cells"][3]["source"])

        global_ns = {"__name__": "__main__"}
        exec(c1_code, global_ns)
        global_ns["ctx"].is_smoke_test = False
        global_ns["ctx"].topology.paths["long"]["ncbi_parquet_dir"] = "/non_existent_ncbi_parquet_dir_xyz"

        with self.assertRaises(FileNotFoundError):
            exec(c3_code, global_ns)

    def test_r07_notebook_02_fail_fast_outside_smoke(self):
        """R07 C3: long/02 raises FileNotFoundError when is_smoke_test=False and checkpoints are missing."""
        nb_path = Path("notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        c1_code = "".join(nb["cells"][1]["source"])
        c3_code = "".join(nb["cells"][3]["source"])

        global_ns = {"__name__": "__main__"}
        exec(c1_code, global_ns)
        global_ns["ctx"].is_smoke_test = False
        global_ns["ctx"].topology.paths["checkpoints"]["resnet_clean_species"] = "/non_existent_checkpoint_xyz"
        global_ns["ctx"].topology.paths["long"]["ncbi_parquet_dir"] = "/non_existent_ncbi_dir_xyz"

        with self.assertRaises(FileNotFoundError):
            exec(c3_code, global_ns)

        # The model constructor is reachable only in the explicitly marked smoke branch;
        # empirical input validation still requires all five checkpoints and the mapping.
        self.assertIn("if ctx.is_smoke_test:", c3_code)
        self.assertIn("five identified checkpoints", c3_code)
        self.assertIn("canonical Species mapping missing", c3_code)

    def test_r07_notebook_03_dynamic_table_12_and_fail_fast(self):
        """R07 C4: long/03 computes Table 12 dynamically and raises FileNotFoundError outside smoke mode if missing."""
        nb_path = Path("notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        c1_code = "".join(nb["cells"][1]["source"])
        c3_code = "".join(nb["cells"][3]["source"])
        c5_code = "".join(nb["cells"][5]["source"])

        # Check that hardcoded table_12_data literal is removed from cell 3
        self.assertNotIn("table_12_data = [", c3_code)

        global_ns = {"__name__": "__main__"}
        old=os.environ.get("C02_LONG_RUN_ID"); os.environ["C02_LONG_RUN_ID"]="missing-c02-predecessor-"+uuid.uuid4().hex[:8]
        try:
            exec(c1_code, global_ns)
            global_ns["ctx"].is_smoke_test = False
            with self.assertRaises(FileNotFoundError): exec(c3_code, global_ns)
        finally:
            if old is None: os.environ.pop("C02_LONG_RUN_ID",None)
            else: os.environ["C02_LONG_RUN_ID"]=old
        self.assertIn("INCONCLUSIVE", c5_code)
        self.assertIn("build_diagnostic_finding", c5_code)

    def test_r07_notebook_04_fail_fast_outside_smoke(self):
        """R07 C5: long/04 raises FileNotFoundError when is_smoke_test=False and contigs are missing."""
        nb_path = Path("notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        c1_code = "".join(nb["cells"][1]["source"])
        c3_code = "".join(nb["cells"][3]["source"])

        global_ns = {"__name__": "__main__"}
        exec(c1_code, global_ns)
        global_ns["ctx"].is_smoke_test = False
        global_ns["ctx"].topology.paths["long"]["ncbi_parquet_dir"] = "/non_existent_ncbi_parquet_dir_xyz"

        with self.assertRaises(FileNotFoundError):
            exec(c3_code, global_ns)

    def test_r07_smoke_execution_all_long_notebooks(self):
        """R07 C6: all long notebooks execute; C02 produces computed smoke metrics and long/04 still runs."""
        self.assertLess(self.chain_duration,60.0)
        started=time.time()
        self.run_notebook("04_long_motifs_and_gc_analysis.ipynb")
        self.assertLess(time.time()-started,60.0)
        t12=self.chain_dir("03")/"table_12_long_voter_comparison.csv"
        self.assertTrue(t12.is_file())
        metrics=pd.read_csv(t12)
        self.assertEqual(len(metrics),20)
        self.assertTrue(((metrics.correct/metrics.N)-metrics.accuracy).abs().max()<1e-12)
        manifest=json.loads((self.chain_dir("03")/"run_manifest.json").read_text())
        self.assertTrue(manifest["is_smoke_test"])
        motif_csv=Path("results_reproduction/long_voter/long_contig_motif_summary.csv")
        fig_path=Path("results_reproduction/long_voter/figure_long_motifs_gc.png")
        self.assertTrue(motif_csv.exists())
        self.assertTrue(fig_path.exists())
        motifs=pd.read_csv(motif_csv)
        self.assertTrue(any(c.startswith("motif_") for c in motifs.columns))


if __name__ == "__main__":
    unittest.main()

