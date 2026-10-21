# S03: Genome-Disjoint Evaluation Feasibility - Checks

Profile: light
Plan: `.specs/features/s03-genome-disjoint-evaluation-feasibility/plan.md`

## Checks

### S1: Genome-Disjoint Feasibility Analysis and Manifest · 1 file · 12 KB · ~3k

**C1** - `docs/genome_split_feasibility.md` documents mathematical and biological criteria proving closed-set genome-disjoint evaluation requires $\ge 2$ independent reference genomes per species (AC 1)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_feasibility_manifest_two_genome_requirement`

**C2** - `docs/genome_split_feasibility.md` documents single-genome species infeasibility for disjoint testing and records the resulting population shift relative to 13,683 species (AC 2)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_feasibility_manifest_single_genome_population_shift`

**C3** - `docs/genome_split_feasibility.md` establishes protocol comparability rules prohibiting uncontextualized accuracy comparisons between grouped and fragment splits (AC 3)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_feasibility_manifest_protocol_comparability_rules`

### S2: Grouped Split Partitioning Engine · 2 files · 22 KB · ~6k

**C4** - `viral_cnn.data.grouped_split.build_genome_disjoint_split` partitions fragments so all fragments of any accession reside strictly in one partition (AC 4)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_grouped_split_all_fragments_in_single_partition`

**C5** - `viral_cnn.data.grouped_split.build_genome_disjoint_split` guarantees zero genome accession overlap between train and test (`set(train_genomes) ∩ set(test_genomes) == ∅`) (AC 5)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_grouped_split_zero_genome_leakage`

**C6** - `viral_cnn.data.grouped_split.build_genome_disjoint_split` isolates single-genome species as infeasible for test evaluation and assigns them to an explicit train_only or excluded category (AC 6)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_grouped_split_single_genome_isolation`

**C7** - `viral_cnn.data.grouped_split.build_genome_disjoint_split` calculates taxonomic coverage metrics including eligible species, infeasible species, and fragment counts (AC 7)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_grouped_split_taxonomic_coverage_metrics`

### S3: Grouped Split CLI and Plan · 3 files · 18 KB · ~5k

**C8** - CLI `scripts/build_grouped_split.py` generates disjoint partition manifests and structured JSON feasibility report with exit code 0 (AC 8)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_build_grouped_split_cli_exit_zero_and_manifests`

**C9** - CLI `scripts/build_grouped_split.py` validates arguments and exits with code 2 on missing files, missing genome columns, or invalid split fractions (AC 9)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_build_grouped_split_cli_invalid_arguments_exit_code_2`

**C10** - Grouped split routines verify deterministic partitioning and zero-leakage contracts against synthetic fixtures without requiring external downloads (AC 10)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_grouped_split_deterministic_synthetic_execution`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Genome accession partition invariants (2) | `train_accessions_disjoint` C5 · `test_accessions_disjoint` C5 | - |
| Species feasibility classifications (2) | `multi_genome_eligible` C6 · `single_genome_infeasible` C6 | - |
| Partition coverage metrics reported (3) | `eligible_species_count` C7 · `infeasible_species_count` C7 · `retained_fragments_count` C7 | - |
| CLI exit status codes (2) | `0` C8 · `2` C9 | - |
| Feasibility documentation sections audited (3) | `mathematical_criteria` C1 · `population_shift` C2 · `comparability_rules` C3 | - |

- Claims naming a status code or response shape: C8, C9 - each has a proof crossing the process boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C9
- failure modes: C9
- idempotency: C5
- authorization: n/a - local scientific CLI tool with no network or authorization boundary
- concurrency: n/a - deterministic batch processing on single machine with no concurrent state mutation
- data lifecycle: C1
- dependency failure: C10
- state transitions: n/a - stateless data transformation producing partition manifest files
- observability: C8

## Handoff

- S1 = 3k in docs; S2 = 6k in src/viral_cnn/data/grouped_split.py and tests; S3 = 5k in scripts and fixtures. Total = ~14k tokens, well under the 150k budget - one builder

- **Mechanism:** one builder
