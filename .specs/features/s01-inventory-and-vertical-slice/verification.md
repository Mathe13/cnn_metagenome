# S01: Inventory, Organization, and Vertical Slice Extraction - verification

**Verdict**: PASS
**Profile**: light
**Diff range**: 989f781..HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | All 15 historical notebooks in `short/` and `long/` remain present and unmodified from baseline inventory (AC 1) | `.venv/bin/python3 -m unittest discover -s tests -k test_historical_notebooks_intact` exit 0 | `tests/test_inventory.py:33` - `self.assertEqual(len(EXPECTED_NOTEBOOKS), 15)` | PASS |
| C2 | `docs/inventory.md` documents all 15 notebooks and maps each to manuscript Tables 1-12 and Figures 1-5 (AC 2) | `.venv/bin/python3 -m unittest discover -s tests -k test_notebook_paper_mappings` exit 0 | `tests/test_inventory.py:46` - `self.assertTrue(inv_path.exists(), "docs/inventory.md does not exist")` | PASS |
| C3 | Missing `.pt` checkpoints and Parquet splits are documented as remote blocking preconditions in `docs/inventory.md` (AC 3) | `.venv/bin/python3 -m unittest discover -s tests -k test_missing_checkpoints_documented` exit 0 | `tests/test_inventory.py:79` - `self.assertIn(item, content, f"Missing item {item} not documented in docs/inventory.md")` | PASS |
| C4 | Taxonomic CSV IDs map to zero-based class indices via `model_label = id - 1` across species and genus (AC 4) | `.venv/bin/python3 -m unittest discover -s tests -k test_label_zero_based_offset` exit 0 | `tests/test_taxonomy.py:25` - `self.assertEqual(taxonomic_id_to_model_label(1), 0)` | PASS |
| C5 | 512 bp spatial one-hot encoding strictly maps A=0, C=1, G=2, T=3 and produces null vector `[0.0, 0.0, 0.0, 0.0]` for padding and ambiguous bases (AC 5) | `.venv/bin/python3 -m unittest discover -s tests -k test_spatial_one_hot_strict_channels_and_null_pad` exit 0 | `tests/test_encoding.py:31` - `self.assertEqual(NUCLEOTIDE_MAPPING, {"A": 0, "C": 1, "G": 2, "T": 3})` | PASS |
| C6 | Extracted `viral_cnn.data.encoding` produces identical numerical output tensors to historical `short/utils/models.py:OneHotEmbedding` (AC 8) | `.venv/bin/python3 -m unittest discover -s tests -k test_parity_with_historical_one_hot_embedding` exit 0 | `tests/test_encoding.py:84` - `self.assertTrue(torch.equal(hist_output, new_output))` | PASS |
| C7 | CLI `scripts/run_inference.py` executes forward pass on ResNet50-1D and ConvNeXt-1D returning exit code 0 and valid JSON output (AC 6) | `.venv/bin/python3 -m unittest discover -s tests -k test_clean_inference_exit_zero_and_json` exit 0 | `tests/test_cli_inference.py:36` - `self.assertEqual(res.returncode, 0)` | PASS |
| C8 | CLI `scripts/run_inference.py` rejects invalid nucleotide inputs and unsupported model types with exit code 2 (AC 7) | `.venv/bin/python3 -m unittest discover -s tests -k test_invalid_inputs_exit_code_2` exit 0 | `tests/test_cli_inference.py:66` - `self.assertEqual(res1.returncode, 2)` | PASS |

## Gate

`.venv/bin/python3 -m unittest discover -s tests` - 10 passed, 0 failed
