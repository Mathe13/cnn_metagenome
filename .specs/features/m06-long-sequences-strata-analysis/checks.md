# M06 checks

Profile: light
Plan: `.specs/features/m06-long-sequences-strata-analysis/plan.md` — aprovado pelo usuário (“sim”).

17 checks em três slices; cinco portas; nenhuma questão bloqueante.

## Checks

### S1

**C1** — The system SHALL reserve collision-free, isolated directories under `results_reproduction/runs/<run_id>/long/<stage>/` for stages `long/01`, `long/02`, `long/03`, and `long/04` under protocol `m06_long_strata_audit_v1`. (AC 1).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_paths -k test_paths`

**C2** — WHEN `long/01` executes THEN the system SHALL audit input long sequences and calculate k-mer overlap against training fragments with $k=15$, saving `long_database_preparation_audit.csv` and `manifest_01_ncbi_prep.json`. (AC 2).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_long_01_overlap_prep -k test_long_01_overlap_prep`

**C3** — WHEN `long/02` executes in empirical mode THEN the system SHALL verify that all five Species checkpoints exist and evaluate predictions for all five variants on identical long queries, computing both `mean_logits` and `majority_vote` with `handle_short="pad"`. (AC 3).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_long_02_paired_five_models -k test_long_02_paired_five_models`

**C4** — IF any required input checkpoint or parquet is missing in empirical mode THEN the system SHALL raise FileNotFoundError and write an atomic failure manifest with status `FAILED`. (AC 4).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_missing_checkpoint_failed_manifest -k test_missing_checkpoint_failed_manifest`

**C5** — WHEN `long/02` finishes successfully THEN the system SHALL write paired prediction parquet files and emit `manifest_02_voter_inference.json` and `manifest.json` with status `COMPLETE`. (AC 5).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_long_02_complete_manifest -k test_long_02_complete_manifest`

### S2

**C6** — WHEN `long/03` executes THEN the system SHALL dynamically resolve the `run_id` of `long/02` without hardcoded run IDs and verify query identity and class ordering alignment across all five models before computing Table 12. (AC 6).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_dynamic_run_resolution_no_hardcoding -k test_dynamic_run_resolution_no_hardcoding`

**C7** — WHEN Table 12 is computed THEN the system SHALL export `table_12_long_voter_comparison.csv` containing overall accuracy and coverage for both `mean_logits` and `majority_vote` across all five model variants. (AC 7).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_table_12_paired_export -k test_table_12_paired_export`

**C8** — WHEN `long/03` performs stratification THEN the system SHALL compute prediction error and accuracy across five length bins (`[0, 512)`, `[512, 2000)`, `[2000, 5000)`, `[5000, 10001)`, `[10001, inf)`) and window count bins, exporting `stratified_long_errors.csv` with the sample count $N$ explicitly reported for each stratum. (AC 8).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_stratified_errors_with_sample_size -k test_stratified_errors_with_sample_size`

**C9** — WHEN `long/03` visualizes strata THEN the system SHALL generate `figure_long_accuracy_by_length_and_windows.png` displaying accuracy across length and window strata with $N$ indicated. (AC 9).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_figure_accuracy_by_length_and_windows -k test_figure_accuracy_by_length_and_windows`

**C10** — WHEN `long/03` visualizes aggregators THEN the system SHALL generate `figure_long_paired_vote_vs_mean_logits.png` displaying paired delta accuracy and agreement between voting and mean logits. (AC 9).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_figure_paired_vote_vs_mean_logits -k test_figure_paired_vote_vs_mean_logits`

**C11** — WHEN `long/03` evaluates mechanisms for ConvNeXt accuracy drop THEN the system SHALL categorize all mechanism findings as `HYPOTHESIS` or `INCONCLUSIVE` in `long_voter_diagnosis.json`, explicitly disclaiming causal architectural claims without controlled multi-seed retraining. (AC 10).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_diagnostic_hypothesis_demarcation -k test_diagnostic_hypothesis_demarcation`

### S3

**C12** — WHEN `long/04` executes THEN the system SHALL dynamically load top learned 7-mer motifs from stage `short/07` (M04) or a verified fallback, and sample contigs deterministically with seed 42 and length threshold $\ge 2000$ bp rather than taking the first file row. (AC 11).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_long_04_motif_loading_and_sampling -k test_long_04_motif_loading_and_sampling`

**C13** — WHEN `long/04` processes sampled contigs THEN the system SHALL slice sequences into 512 bp windows (stride 256 bp) and export `long_contig_motif_summary.csv` containing window positions, GC content percentage, and learned motif counts. (AC 12).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_long_04_contig_motif_summary -k test_long_04_contig_motif_summary`

**C14** — WHEN `long/04` finishes THEN the system SHALL render `figure_long_motifs_gc.png` showing the dual-panel profile of GC percentage and motif occurrences, and emit `manifest_04_long_motifs.json` and `manifest.json` with status `COMPLETE`. (AC 13).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_long_04_figure_and_manifest -k test_long_04_figure_and_manifest`

**C15** — WHILE executing in smoke test mode (`SMOKE_TEST=1`) THEN all long stages SHALL execute completely on CPU fixtures within 15 seconds, setting `is_smoke_test: true` in all generated manifests. (AC 14).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_smoke_test_cpu_speed -k test_smoke_test_cpu_speed`

**C16** — WHEN reporting and documenting long-sequence results THEN the system SHALL refer to evaluated sequences as *long-sequences* or *contigs*, reserving *long-reads* exclusively for authentic sequencer read data. (AC 15).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_terminology_long_sequences_not_reads -k test_terminology_long_sequences_not_reads`

**C17** — WHEN artifacts are loaded or persisted across all M06 workflows THEN the system SHALL emit console and log messages with prefix `[LOG_PATH]` indicating the absolute path, respecting `LOG_PATH=0/false`. (AC 16).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m06_long_sequences_strata.M06Tests.test_logging_path_traceability -k test_logging_path_traceability`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Assemblies (4) | long/01 C2 · long/02 C5 · long/03 C7 · long/04 C14 | - |
| Manifest states (2) | COMPLETE C5 · FAILED C4 | - |
| Model variants (5) | resnet_clean_species C3 · resnet_noise_species C3 · convnext_zero_species C3 · convnext_transfer_species C3 · convnext_noise_species C3 | - |
| Aggregation methods (2) | mean_logits C7 · majority_vote C7 | - |
| Doors (5) | lifecycle C1 · paired-5-models C3 · dynamic-02-resolution C6 · stratification-and-figures C8 · dynamic-motifs-and-sampling C12 | - |
| Entities (5) | database_prep_audit C2 · paired_predictions C5 · table_12 C7 · stratified_errors C8 · contig_motif_summary C13 | - |
| Figures (3) | figure_long_accuracy_by_length_and_windows C9 · figure_long_paired_vote_vs_mean_logits C10 · figure_long_motifs_gc C14 | - |

## Test policy

| Target | Policy |
| --- | --- |
| m06_workflow | unit: resolução de entradas, isolamento de saídas sob runs/, reserva de diretórios, captura de falhas e manifestos atômicos |
| long_01_prep | unit: quantificação de overlap de k-mers e emissão do manifesto de preparação |
| long_02_voter | unit: inferência pareada com LongSequenceVoter sobre 5 variantes Species computando mean_logits e majority_vote com handle_short='pad' |
| long_03_compare | unit: resolução dinâmica de 02, Tabela 12 pareada, estratos por comprimento e janelas com N, figuras de agregação e diagnóstico hipótese |
| long_04_motifs | unit: carregamento dinâmico de 7-mers de 07, amostragem determinística de contigs >=2000 bp, resumo de janelas e figura de perfil |
| notebooks_long | integration: execução de ponta a ponta sobre fixtures determinísticas em CPU |

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C2, C3, C4, C12
- failure modes: C4, emissão de manifesto FAILED sob ausência de arquivos empíricos
- idempotency: C1, C5, C7, C14
- authorization: n/a — execução local sob permissões de arquivo do sistema operacional
- concurrency: C1, reserva atômica de diretório exclusivo impedindo colisões
- data lifecycle: C2, C5, C7, C8, C13
- dependency failure: C4, C6, C12
- state transitions: C4, C5, C14
- observability: C1, C2, C5, C7, C8, C9, C10, C11, C13, C14, C16, C17

## Handoff

S1–S3 = ~42k total, dentro do orçamento de 150k. Mecanismo: um builder.
