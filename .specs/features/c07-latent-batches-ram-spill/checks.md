# C07 — Checks: avaliação latente em lotes com RAM e spill

Profile: light
Plan: `.specs/features/c07-latent-batches-ram-spill/plan.md`

## Checks

### S1 — armazenamento limitado, extração em lotes e sonda

**C1** — O orçamento padrão é 30% da RAM disponível observada na criação da execução; override em MiB prevalece, zero seleciona spill, e valores negativos/não inteiros são rejeitados.
Proof: `.venv/bin/python3 -m unittest tests.test_c07_latent_memory -k test_budget_defaults_to_available_ram_fraction_and_validates_overrides`

**C2** — Ao exceder o orçamento de resultados gerenciados, arrays spillados em diretório gravável preservam dtype, forma, conteúdo e ordem; diretórios-base iguais geram arquivos distintos; fechar um store remove somente seus próprios temporários.
Proof: `.venv/bin/python3 -m unittest tests.test_c07_latent_memory -k test_spill_roundtrip_exclusive_paths_and_owned_cleanup`

**C3** — A extração de 1.031 linhas em lotes de 256 não chama o modelo com batch maior que 256 e retorna features, labels e row IDs alinhados à ordem observada, inclusive o último lote de 7.
Proof: `.venv/bin/python3 -m unittest tests.test_c07_latent_memory -k test_extraction_batches_preserve_order_alignment_and_spill`

**C4** — Salvar e carregar um artefato de 1.031 linhas por blocos preserva todas as features, labels, row IDs e o manifesto validado, sem requerer uma segunda cópia integral em RAM.
Proof: `.venv/bin/python3 -m unittest tests.test_c07_latent_memory -k test_embedding_artifact_streaming_roundtrip`

**C5** — A seleção da sonda valida em lotes de no máximo 256, calcula Accuracy global sobre todas as linhas e mantém seleção pela maior Accuracy com empate resolvido pela primeira época e selagem do hash.
Proof: `.venv/bin/python3 -m unittest tests.test_c07_latent_memory -k test_probe_validation_batches_and_preserves_global_selection`

**C6** — Avaliação da sonda processa o teste em lotes de no máximo 256 e suas Accuracy, Macro-F1, N e suporte de classes coincidem com a referência calculada sobre o mesmo conjunto.
Proof: `.venv/bin/python3 -m unittest tests.test_c07_latent_memory -k test_probe_test_evaluation_batches_preserve_metrics`

**C7** — Extração que pede logits mantém retorno explícito com orçamento RAM/spill; o caminho do notebook para a cabeça nativa retorna Top-1 por linha, sem guardar a matriz completa de logits.
Proof: `.venv/bin/python3 -m unittest tests.test_c07_latent_memory -k test_logits_are_spill_managed_and_notebook_keeps_top1`

**C8** — Configuração inválida e erro de escrita no spill falham com o caminho da operação, e nenhum artefato Parquet/manifesto incompleto é aceito como resultado completo.
Proof: `.venv/bin/python3 -m unittest tests.test_c07_latent_memory -k test_invalid_budget_and_spill_failure_do_not_publish_partial_artifact`

**C9** — O notebook 06 usa lote 256, calcula 30% de RAM disponível quando `LATENT_RAM_BUDGET_MB` não está definido, aceita override e registra orçamento efetivo, batch e caminhos de spill no manifesto/log.
Proof: `.venv/bin/python3 -m unittest tests.test_c07_latent_memory -k test_notebook_configures_and_records_memory_budget`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Landing door 1 — política compartilhada (4) | batch default 256 -> C3,C5,C6,C9 · RAM default 30% -> C1,C9 · override/zero -> C1,C9 · spill default exclusivo por execução -> C2,C8,C9 | - |
| Landing door 2 — identidade compacta de saída (1) | campo `Test_Row_IDs_SHA256` -> C7,C9 | - |
| Relations — execução/arrays (3) | orçamento e stores vivos -> C1,C2 · feature/label/row ID alinhados -> C3,C4 · artefato Parquet/sidecar -> C4,C8 | - |
| Superfícies Python (4) | sucesso -> C2–C7 · configuração inválida -> C1,C8 · escrita impossível -> C8 · logits explicitamente solicitados -> C7 | - |
| Superfície notebook 06 (2) | configuração padrão e override -> C9 · erro de spill -> C8,C9 | - |
| Bordas de lote do fixture (4) | 1.031 total -> C3,C4,C5,C6 · primeiro lote 256 -> C3,C5,C6 · quatro lotes completos -> C3,C5,C6 · último lote 7 -> C3,C5,C6 | - |
| Caminhos do store (3) | RAM sob orçamento -> C2 · spill por orçamento -> C2,C3,C7 · cleanup sem apagar arquivo alheio -> C2 | - |

Claims: C1–C9 correspondem aos critérios 1–19 do plano; C1 cobre 13, C2 cobre 3–6 e 15–16, C3–C4 cobrem 6–7, C5–C6 cobrem 1 e 8–10, C7 cobre 11–12 e C9 cobre 2 e 17–19. C8 cobre 14.

## Swept

- validation: C1,C3,C8,C9
- failure modes: C8
- idempotency: C2,C4,C8
- authorization: n/a — pipeline local sem serviço externo ou usuário autenticado
- concurrency: C2 — instâncias de armazenamento recebem caminhos exclusivos
- data lifecycle: C2,C4,C8 — ownership de temporários e publicação completa
- dependency failure: C8 — diretório de spill indisponível
- state transitions: C2,C5 — RAM→spill e sonda unfitted→sealed
- observability: C1,C2,C8,C9 — orçamento, batch e paths efetivos

## Out of scope

- KNN, Silhouette e UMAP em armazenamento externo — continuam com seus próprios algoritmos e limites; o plano não altera essas análises.
- Prova empírica integral no cluster — requer acesso aos dados, checkpoints e servidor remotos.

## Handoff

- **Slices**: 1 (`S1`)
- **Checks**: 9 (`C1`–`C9`)
- **Files in play**: 6 (`src/viral_cnn/eval/latent.py`, `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb`, `tests/test_c01_latent_audit.py`, `tests/test_c07_latent_memory.py`, `plan.md`, este `checks.md`); handoff operacional em `.specs/STATE.md`.
- **Estimated load**: 6 arquivos × ~5k tokens + provas/testes × ~7k + ~25k implementação/revisão = ~62k tokens < 150k -> **single builder pass**.
- **Mechanism**: single builder; a estimativa fica abaixo do orçamento do perfil `light`.
- **Completion**: C1–C9 implementados; provas sintéticas passaram e a verificação independente concluiu PASS na rodada 2; validação empírica remota permanece pendente.
