# M04 checks

Profile: light
Plan: `.specs/features/m04-motifs-and-first-layer-perturbation/plan.md` — aprovado pelo usuário (“prossiga”).

16 checks em três slices; cinco portas; nenhuma questão bloqueante.

## Checks

### S1

**C1** — WHEN executa 07 THEN o fluxo SHALL resolver os caminhos de entrada e saída a partir do plano M00 ou variáveis de ambiente sem depender de diretórios fixos legados. (AC 1).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_paths -k test_paths`

**C2** — WHEN `preflight_notebook("short/07", ...)` é executado no modo empírico (`SMOKE_TEST=0`) THEN o runtime SHALL validar a presença e legibilidade dos cinco checkpoints de espécies e reportar `READY` apenas se todos existirem, ou `BLOCKED` se qualquer um faltar. (AC 2).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_preflight_07_five_models -k test_preflight_07_five_models`

**C3** — IF o diretório reservado de destino já contiver produtos ou colidir com run existente THEN o fluxo SHALL abortar antes de modificar qualquer arquivo. (AC 3).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_collision_rejection -k test_collision_rejection`

**C4** — WHEN uma execução de 07 falha por erro inesperado THEN o fluxo SHALL emitir manifesto atômico com status `FAILED` e detalhe da falha, nunca `COMPLETE`. (AC 4).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_failed_manifest -k test_failed_manifest`

**C5** — WHEN a etapa 07 conclui com sucesso THEN o fluxo SHALL publicar manifesto `COMPLETE` contendo protocolo `m04_motifs_audit_v1`, `is_smoke_test`, semente 42, hashes das entradas e lista completa de artefatos gerados. (AC 5).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_complete_manifest -k test_complete_manifest`

### S2

**C6** — WHEN executa a extração de motivos de 1ª camada THEN o extrator SHALL analisar os pesos convolucionais de entrada e extrair até 10 motivos consenso de 7 bp para cada um dos cinco modelos. (AC 6).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_motif_extraction_top10 -k test_motif_extraction_top10`

**C7** — WHEN avalia a perturbação de cada motivo THEN o fluxo SHALL selecionar controles nulos (uniforme aleatório e pareado por GC com |ΔGC|=0) obrigatoriamente na mesma sequência hospedeira do motivo sem sobreposição. (AC 7).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_paired_gc_control_same_sequence -k test_paired_gc_control_same_sequence`

**C8** — WHEN calcula o impacto da perturbação THEN o fluxo SHALL computar a queda específica de ativação, o teste bicaudal de Wilcoxon pareado e a correção para múltiplos testes por FDR Benjamini-Hochberg. (AC 8).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_wilcoxon_and_fdr_computation -k test_wilcoxon_and_fdr_computation`

**C9** — WHEN exporta os resultados tabulares de perturbação THEN o fluxo SHALL salvar `table_08_motif_perturbation.csv` e `motif_perturbation_by_variant.csv` contendo esquema de colunas padronizado com quedas uniformes e GC, FDRs e contagens de ocorrências. (AC 9).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_reconciled_table_08_export -k test_reconciled_table_08_export`

**C10** — WHEN executa 07 no modo empírico (`SMOKE_TEST=0`) THEN o fluxo SHALL rejeitar checkpoints ou fixtures marcados como sintéticos (`is_smoke_test=true`). (AC 10).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_reject_synthetic_in_empirical -k test_reject_synthetic_in_empirical`

### S3

**C11** — WHEN renderiza a Figura 3 científica principal THEN o renderizador SHALL produzir painel com exatamente cinco subgráficos (a: ResNet50 Clean, b: ConvNeXt_Z, c: ConvNeXt_T, d: ResNet50_F, e: ConvNeXt_F) exibindo quedas de ativação e significância por GC, sem barras de status de checkpoint. (AC 11).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_figure_03_five_panels -k test_figure_03_five_panels`

**C12** — The renderizador SHALL garantir que o painel (e) utilize os dados e ativações genuínos de ConvNeXt_F (`convnext_noise_species`), nunca clonando bytes ou dados do painel (d). (AC 12).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_panel_e_authenticity -k test_panel_e_authenticity`

**C13** — WHEN renderiza a figura de sensibilidade composicional THEN o renderizador SHALL exportar gráfico de dispersão confrontando quedas sob controle uniforme versus GC com indicação de linha de identidade e limiares de robustez. (AC 13).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_scatter_gc_uniform_rendering -k test_scatter_gc_uniform_rendering`

**C14** — WHEN calcula a coocorrência entre motivos THEN o fluxo SHALL exportar matrizes de presença e matrizes Jaccard em Parquet (`motif_interaction_jaccard_<variant>.parquet`) no namespace da execução para consumo por 08 / M05. (AC 14).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_jaccard_parquets_export -k test_jaccard_parquets_export`

**C15** — WHEN documenta ou sumariza os motivos de 1ª camada THEN o texto e os metadados SHALL demarcar explicitamente que os motivos representam detectores de composição de bases nucleotídicas e não sítios funcionais biológicos comprovados de fatores de transcrição nem códons biológicos funcionais. (AC 15).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_biological_demarcation -k test_biological_demarcation`

**C16** — WHEN artefatos são lidos ou gravados nos fluxos M04 THEN o código SHALL emitir logs com prefixo `[LOG_PATH]` contendo o caminho do arquivo, respeitando `LOG_PATH=0/false`. (AC 16).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m04_motifs_and_first_layer_perturbation.M04Tests.test_logging_path -k test_logging_path`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Assemblies (1) | short/07 C5 | - |
| Manifest states (2) | COMPLETE C5 · FAILED C4 | - |
| Control policies (2) | uniform_random C7 · gc_matched C7 | - |
| Model variants (5) | resnet50_species C6 · resnet_noise_species C6 · convnext_zero_species C6 · convnext_transfer_species C6 · convnext_noise_species C12 | - |
| Figures (2) | figure_03_motifs C11 · figure_03_scatter_gc_uniform C13 | - |
| Preflight states (2) | READY C2 · BLOCKED C2 | - |
| Doors (5) | lifecycle C5 · 5-variant-preflight C2 · paired-gc-control C7 · reconciled-table-08 C9 · scientific-f3-panels C11 | - |
| Entities (4) | model C6 · motif C6 · perturbation_sample C8 · cooccurrence_matrix C14 | - |

## Test policy

| Target | Policy |
| --- | --- |
| m04_workflow | unit: resolução de entradas, isolamento de saídas, captura de falhas e manifesto atômico |
| preflight_07 | unit: validação da presença física e legibilidade dos cinco checkpoints de espécies |
| paired_perturbation | unit: amostragem pareada de controle uniforme e GC na mesma sequência com |ΔGC|=0 |
| figure_03 | unit: renderização de cinco subpainéis (a–e) e autenticidade estrita do painel (e) de ConvNeXt_F |
| notebook_07 | integration: execução de ponta a ponta sobre fixtures e exportação das matrizes Jaccard e T8 |

## Swept

- validation: C2, C7, C10
- failure modes: C3, C4
- idempotency: C1, C3, C5
- authorization: n/a — execução local sob permissões de arquivo do sistema operacional
- concurrency: C3, reserva atômica de diretório exclusivo impedindo colisões
- data lifecycle: C5, C9, C14
- dependency failure: C2, C4, C10
- state transitions: C4, C5
- observability: C5, C9, C11, C13, C15, C16

## Handoff

S1–S3 = ~36k total, dentro do orçamento de 150k. Mecanismo: um builder.
