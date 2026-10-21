# S03: Genome-Disjoint Evaluation Feasibility - verification

**Verdict**: PASS
**Profile**: light
**Diff range**: c6272f1..HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `docs/genome_split_feasibility.md` documents mathematical and biological criteria proving closed-set genome-disjoint evaluation requires $\ge 2$ independent reference genomes per species (AC 1) | `.venv/bin/python3 -m unittest discover -s tests -k test_feasibility_manifest_two_genome_requirement` exit 0 | `tests/test_grouped_split.py:41` - `self.assertIn("no mínimo 2 genomas de referência independentes", content)` | PASS |
| C2 | `docs/genome_split_feasibility.md` documents single-genome species infeasibility for disjoint testing and records the resulting population shift relative to 13,683 species (AC 2) | `.venv/bin/python3 -m unittest discover -s tests -k test_feasibility_manifest_single_genome_population_shift` exit 0 | `tests/test_grouped_split.py:49` - `self.assertIn("Inviabilidade de Classes com Genoma Único", content)` | PASS |
| C3 | `docs/genome_split_feasibility.md` establishes protocol comparability rules prohibiting uncontextualized accuracy comparisons between grouped and fragment splits (AC 3) | `.venv/bin/python3 -m unittest discover -s tests -k test_feasibility_manifest_protocol_comparability_rules` exit 0 | `tests/test_grouped_split.py:57` - `self.assertIn("Regra Permanente de Comparabilidade", content)` | PASS |
| C4 | `viral_cnn.data.grouped_split.build_genome_disjoint_split` partitions fragments so all fragments of any accession reside strictly in one partition (AC 4) | `.venv/bin/python3 -m unittest discover -s tests -k test_grouped_split_all_fragments_in_single_partition` exit 0 | `tests/test_grouped_split.py:75` - `self.assertEqual(len(part_set), 1)` | PASS |
| C5 | `viral_cnn.data.grouped_split.build_genome_disjoint_split` guarantees zero genome accession overlap between train and test (`set(train_genomes) ∩ set(test_genomes) == ∅`) (AC 5) | `.venv/bin/python3 -m unittest discover -s tests -k test_grouped_split_zero_genome_leakage` exit 0 | `tests/test_grouped_split.py:89` - `self.assertTrue(train_genomes.isdisjoint(test_genomes))` | PASS |
| C6 | `viral_cnn.data.grouped_split.build_genome_disjoint_split` isolates single-genome species as infeasible for test evaluation and assigns them to an explicit train_only or excluded category (AC 6) | `.venv/bin/python3 -m unittest discover -s tests -k test_grouped_split_single_genome_isolation` exit 0 | `tests/test_grouped_split.py:102` - `self.assertNotIn("3", test_classes_train_only)` | PASS |
| C7 | `viral_cnn.data.grouped_split.build_genome_disjoint_split` calculates taxonomic coverage metrics including eligible species, infeasible species, and fragment counts (AC 7) | `.venv/bin/python3 -m unittest discover -s tests -k test_grouped_split_taxonomic_coverage_metrics` exit 0 | `tests/test_grouped_split.py:125` - `self.assertEqual(metrics["eligible_species_count"], 2)` | PASS |
| C8 | CLI `scripts/build_grouped_split.py` generates disjoint partition manifests and structured JSON feasibility report with exit code 0 (AC 8) | `.venv/bin/python3 -m unittest discover -s tests -k test_build_grouped_split_cli_exit_zero_and_manifests` exit 0 | `tests/test_grouped_split.py:145` - `self.assertEqual(res.returncode, 0)` | PASS |
| C9 | CLI `scripts/build_grouped_split.py` validates arguments and exits with code 2 on missing files, missing genome columns, or invalid split fractions (AC 9) | `.venv/bin/python3 -m unittest discover -s tests -k test_build_grouped_split_cli_invalid_arguments_exit_code_2` exit 0 | `tests/test_grouped_split.py:166` - `self.assertEqual(res1.returncode, 2)` | PASS |
| C10 | Grouped split routines verify deterministic partitioning and zero-leakage contracts against synthetic fixtures without requiring external downloads (AC 10) | `.venv/bin/python3 -m unittest discover -s tests -k test_grouped_split_deterministic_synthetic_execution` exit 0 | `tests/test_grouped_split.py:188` - `self.assertEqual(res1["metrics"], res2["metrics"])` | PASS |

## Gate

`.venv/bin/python3 -m unittest discover -s tests` - 29 passed, 0 failed
