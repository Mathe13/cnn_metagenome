# r07-real-long-sequence-voter-and-ncbi-audit verification

**Verdict**: PASS
**Profile**: light
**Diff range**: 3c9225a..e18d58f
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `JupyterHubDataTopology.resolve_long_artifact_path` searches candidate directories (`root`, `root/long`, `root/cnn_long`, `root/cnn_kraken`, `root.parent/long`, and cwd) and returns the resolved existing `Path` | `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py -k test_r07_resolve_long_artifact_path` exit 0 | `tests/test_n07_long_sequence_voter.py:290-325` - `self.assertEqual(res_tax, root / "taxonomia_recuperada.csv")`, `self.assertEqual(res_map, root / "long" / "mapping_species_final.csv")`, `self.assertEqual(res_ncbi, root / "cnn_long" / "processed_ncbi_splited_parquet")`, `self.assertEqual(res_inf, root / "cnn_kraken" / "resultados_inferencia_long_reads_ResNet50.parquet")`, `self.assertIsNone(top.resolve_long_artifact_path("non_existent_file_xyz.csv"))` | PASS |
| C2 | `notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb` raises `FileNotFoundError` in Cell 3 when `is_smoke_test == False` and `ncbi_parquet_dir` does not exist on disk, without generating synthetic sequences | `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py -k test_r07_notebook_01_fail_fast_outside_smoke` exit 0 | `tests/test_n07_long_sequence_voter.py:326-342` - `with self.assertRaises(FileNotFoundError): exec(c3_code, global_ns)` | PASS |
| C3 | `notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb` raises `FileNotFoundError` in Cell 3 when `is_smoke_test == False` and real checkpoints or parquets are missing, never using `MockVoterModel` outside smoke mode | `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py -k test_r07_notebook_02_fail_fast_outside_smoke` exit 0 | `tests/test_n07_long_sequence_voter.py:343-366` - `with self.assertRaises(FileNotFoundError): exec(c3_code, global_ns)`, `self.assertNotIn("MockVoterModel", non_smoke_branch)` | PASS |
| C4 | `notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb` derives Table 12 dynamically from evaluated prediction DataFrames and raises `FileNotFoundError` when `is_smoke_test == False` and prediction files are missing, without using `table_12_data` literal or `rng.randint` | `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py -k test_r07_notebook_03_dynamic_table_12_and_fail_fast` exit 0 | `tests/test_n07_long_sequence_voter.py:367-390` - `self.assertNotIn("table_12_data = [", c3_code)`, `with self.assertRaises(FileNotFoundError): exec(c3_code, global_ns)`, `with self.assertRaises(FileNotFoundError): exec(c5_code, global_ns)` | PASS |
| C5 | `notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb` raises `FileNotFoundError` in Cell 3 when `is_smoke_test == False` and real contigs are missing, without generating random nucleotide strings | `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py -k test_r07_notebook_04_fail_fast_outside_smoke` exit 0 | `tests/test_n07_long_sequence_voter.py:391-407` - `with self.assertRaises(FileNotFoundError): exec(c3_code, global_ns)` | PASS |
| C6 | Notebooks `long/01` to `long/04` execute end-to-end within 60 seconds each under `is_smoke_test == True`, producing Table 12, error stratification, and all required manifests and figures | `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py -k test_r07_smoke_execution_all_long_notebooks` exit 0 | `tests/test_n07_long_sequence_voter.py:408-452` - `self.assertLess(elapsed, 60.0, ...)`, `self.assertAlmostEqual(resnet_acc, 0.7269, places=3)`, `self.assertAlmostEqual(convnext_t_acc, 0.0349, places=3)`, `self.assertGreater(total_motifs, 0, ...)` | PASS |

## Gate

`PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py -k "test_r07_"` - 6 passed, 0 failed
