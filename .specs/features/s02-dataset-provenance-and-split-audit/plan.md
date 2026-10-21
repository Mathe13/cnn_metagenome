# S02: Dataset Provenance and Fragment Split Audit

## Problem

The viral classification dataset comprises 17,062,701 short fragments in `dataset_kraken/species_seqs.csv` (5.0 GB), 8,168,145 fragments in `genus_seqs.csv` (2.4 GB), 1,273,788 fragments in `family_seqs.csv` (401 MB), and 33,023 full/contig sequences downloaded via NCBI Entrez in `long/download_from_ncbi.ipynb`.
However, the historical preprocessed parquet splits (`processed_kraken_species_splited_parquet/`) are missing from disk, and the exact relationship between fragmented short sequences and their source genomes is undocumented in the codebase.
Crucially, the historical short-sequence models were trained on an 80/10/10 stratified split by **fragment** (`train_test_split(..., random_state=42)`), rather than by genome.
As discovered in `short/benchmark_cnns_kraken2_blast_overlap.ipynb`, when test queries are audited against training fragments, **95.91%** of sampled test queries (262,470 of 273,660 queries across 13,683 species) have an identical sequence or reverse-complement match in the training set.
Without an explicit, auditable provenance manifest and a dedicated split audit engine, downstream evaluations cannot verify data leakage, distinguish observed from inferred reference fields, or assess true generalizability beyond identical fragments.

When this ships, the complete provenance of Kraken2, NCBI, and ICTV datasets is formally codified in `docs/dataset_manifest.md` (distinguishing observed from inferred fields), a strand-invariant canonical hashing module (`src/viral_cnn/data/audit.py`) provides fast duplicate and reverse-complement detection, and a CLI tool (`scripts/audit_dataset.py`) computes partition counts, exact duplication rates, and train-test leakage metrics with proof-backed deterministic contract tests.

## Flow

Reuses `src/viral_cnn/taxonomy/mapping.py:load_taxonomy_mapping` while preserving existing files in place.

1. dataset CSV or partition files -> `viral_cnn.data.audit` (new, door 1) - streams sequences, applies canonical hashing `min(seq, rc)`, and tracks class occurrence frequencies
2. `viral_cnn.data.audit` -> `viral_cnn.data.split_auditor` (new, door 2) - computes set intersections between partitions, measures exact sequence duplicates and cross-split leakage
3. `scripts/audit_dataset.py` (new, no door - placement per conventions) - CLI interface parsing arguments, invoking audit routines, and formatting reports
4. out: structured JSON audit report and `docs/dataset_manifest.md` summary (door 3)

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `CanonicalSequenceHash` - strand-invariant SHA-256 digest of `min(seq, rc)` used to detect exact fragment matches across partitions, lives in `src/viral_cnn/data/audit.py` |
| domain | new term: `FragmentSplitLeakage` - quantification of test set fragments whose exact sequence or reverse complement was observed during training due to fragment-level rather than genome-level splitting |
| stored data | nothing to migrate: historical CSVs in `dataset_kraken/` remain read-only; new provenance documentation created in `docs/dataset_manifest.md` and audit output artifacts in `results_reports/` or stdout |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| canonical sequence identity hash | `hashlib.sha256(min(seq, reverse_complement(seq)).encode()).hexdigest()` | naive string equality or forward-strand only hashing - rejected because DNA sequences from opposite strands represent the identical genomic locus and biological entity, causing false negatives in overlap detection |
| bounded-memory streaming partition audit | `src/viral_cnn/data/audit.py` streaming chunks into 64-bit integer hash sets or SQLite index | loading all 17,062,701 raw sequence strings into memory at once - rejected because a 5.0 GB raw CSV requires over 30 GB of Python heap, risking Out-Of-Memory crashes on standard workstations |
| CLI audit report schema | stdout outputs JSON containing `summary` (`total_sequences`, `unique_sequences`, `classes_count`), `overlap` (`train_matches`, `overlap_pct`), and `taxonomic_leakage` (`single_source_classes`) | free-form terminal log text - rejected because machine-readable scripts and verification tests require deterministic JSON parsing |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Dataset Provenance and Reference Manifest (P1)

Complete documentation and versioning of dataset references, taxonomic mappings, and split methodology.

**Acceptance Criteria**

1. The system SHALL document the exact dataset provenance for Kraken2, NCBI, and ICTV in `docs/dataset_manifest.md`, explicitly separating observed fields (from `dataset_kraken/*_name_ids.csv` and sequence CSVs) from inferred/reconstructed fields (NCBI Entrez accessions and contig lengths).
2. The system SHALL record in `docs/dataset_manifest.md` the fragment-based 80/10/10 stratified split parameters (random_state=42), the class distribution (13,683 Species, 2,702 Genera), and the empirical baseline finding that 95.91% (262,470 of 273,660) of sampled test queries in `short/benchmark_cnns_kraken2_blast_overlap.ipynb` have identical or reverse-complement sequence matches in the training set.
3. The system SHALL document in `docs/dataset_manifest.md` that lack of identical matches between partitions does not imply zero sequence overlap, noting tile strides and high-similarity sub-sequences as potential leakage sources.

**Independent test:** run provenance verification script asserting that `docs/dataset_manifest.md` exists, contains required taxonomic headers (13683 Species, 2702 Genera), records the 95.91% baseline overlap finding, and distinguishes observed from inferred fields.

### S2: Canonical Sequence Hashing and Duplicate Audit Engine (P1)

Strand-invariant hashing and partition duplicate detection module.

**Acceptance Criteria**

4. WHEN given a DNA nucleotide string containing characters A, C, G, T, or N THEN the system SHALL compute a strand-invariant canonical hash using `SHA-256(min(seq, reverse_complement(seq)))`.
5. WHEN evaluating duplicate sequences within a dataset partition THEN the system SHALL report the total sequence count, the unique canonical sequence count, and the exact duplication rate as a percentage.
6. The system SHALL produce identical canonical hashes for any sequence and its exact reverse complement across all test inputs.
7. The system SHALL identify reverse complement matches between different rows even when forward strings differ.

**Independent test:** run unit tests in `tests/test_dataset_audit.py` verifying canonical hash invariance across test sequences and accurate duplicate counting on synthetic partitions.

### S3: Split Overlap and Leakage CLI (P1)

Executable CLI tool auditing sequence overlap and taxonomic leakage between partitions.

**Acceptance Criteria**

8. WHEN executing `scripts/audit_dataset.py --dataset <path>` or `scripts/audit_dataset.py --train <path> --test <path>` THEN the system SHALL return exit code 0 and emit a structured JSON payload containing total counts, unique counts, class counts, cross-partition sequence overlaps, and leakage percentage.
9. IF an invalid file path or unsupported format is passed to `scripts/audit_dataset.py` THEN the system SHALL exit with code 2 and emit an informative error message to standard error.
10. WHEN evaluating cross-partition overlap between train and test partitions THEN the system SHALL identify identical sequence matches, reverse-complement matches, and quantify classes having samples restricted to a single partition.
11. The system SHALL verify all audit calculation routines against deterministic synthetic test fixtures without requiring multi-gigabyte disk scans during unit tests.

**Independent test:** run `python3 scripts/audit_dataset.py --train tests/fixtures/sample_train.csv --test tests/fixtures/sample_test.csv` and assert exit code 0 with valid JSON report matching expected overlap values.

## Out of scope

| Excluded | Why |
| --- | --- |
| Full model retraining on grouped splits | S02 focuses on dataset provenance and fragment split audit; grouped genome-level splits and retraining belong to S03 and S04 |
| Real-time downloading from NCBI Entrez | Network calls during test suites violate sandboxing and repeatability; NCBI provenance is codified from existing notebooks and local artifacts |
| In-place modification of historical CSVs | Invariant AD-004 requires preserving `dataset_kraken/` as read-only historical records |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Default sequence identity hashing | Strand-invariant SHA-256 on `min(seq, rc)` | Matches the exact hash methodology proven in `short/benchmark_cnns_kraken2_blast_overlap.ipynb` | y |
| Standard split proportions for audit baseline | 80% train, 10% test, 10% validation stratified by class | Matches documented splits in `long/prepare database.ipynb` and manuscript methods | y |
| Unit test execution mode | Deterministic synthetic sequence fixtures | Full 5.0 GB CSV scan takes several minutes; automated CI/unit tests require sub-second execution | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| CLI `scripts/audit_dataset.py` | CLI flags and options (`--dataset`, `--train`, `--test`, `--val`, `--sample-size`, `--output`) | AC 8 |
| CLI `scripts/audit_dataset.py` | output format and structured JSON payload | AC 8 |
| CLI `scripts/audit_dataset.py` | error handling and exit codes (exit 0 on success, exit 2 on error) | AC 9 |
| screen | n/a - this feature exposes no user-facing GUI; all interaction occurs via CLI and Python API |

## Sources

- Manuscript: "Spatial one-hot nucleotide representations of DNA enable robust and interpretable taxonomic classification with 1D convolutional networks" (Biodata Mining / submitted)
- Project rule: `AGENTS.md` (Scientific invariants: Closed-set, splits por fragmentos, ruído, integridade)
- Roadmap: `ROADMAP_SPECS_ARTIGO.md` (Spec S02: Proveniência do dataset e auditoria do split por fragmento)

