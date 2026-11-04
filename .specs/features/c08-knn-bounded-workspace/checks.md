# C08 — Checks: KNN exato com workspace limitado

Profile: light
Plan: `.specs/features/c08-knn-bounded-workspace/plan.md`

13 checks em 2 slices · 2 one-way doors · 0 questões abertas.

## Checks

### S1 — KNN exato com workspace limitado

**C1** — Sem overrides, notebook 06 usa workspace KNN de 1024 MiB e lote de consultas de 256, passa `ctx.device` e registra os três valores efetivos.
Proof: `.venv/bin/python3 -m unittest tests.test_c08_knn_workspace -k test_notebook_knn_defaults_overrides_and_device`

**C2** — Workspace ou lote não inteiros positivos são rejeitados antes de `torch.mm`; workspace menor que o necessário para uma consulta e cinco referências falha com valor solicitado e mínimo.
Proof: `.venv/bin/python3 -m unittest tests.test_c08_knn_workspace -k test_invalid_workspace_and_too_small_budget_fail_before_matmul`

**C3** — KNN em blocos preserva split estratificado 80/20 com semente 42, `k=5`, L2/cosseno, voto majoritário e zero auto-vizinhos, inclusive quando a referência cabe em múltiplos blocos.
Proof: `.venv/bin/python3 -m unittest tests.test_c08_knn_workspace -k test_tiled_knn_keeps_scientific_protocol`

**C4** — Para 1.031 linhas, features de referência/consulta são reunidas somente por blocos, a matriz de similaridades e tensores explícitos ficam sob `workspace_mb`, e o último lote de consultas pode ser incompleto.
Proof: `.venv/bin/python3 -m unittest tests.test_c08_knn_workspace -k test_workspace_bounds_matmul_and_never_gathers_full_features`

**C5** — Para consultas com empates exatos, blocos de referência diferentes produzem os mesmos cinco índices ordenados por similaridade decrescente e menor índice original; sem empates, produzem o Top-5 de matriz completa.
Proof: `.venv/bin/python3 -m unittest tests.test_c08_knn_workspace -k test_tiled_topk_matches_dense_and_breaks_ties_by_original_index`

**C6** — Accuracy, Top-5, N e métricas por classe do KNN em blocos coincidem com a referência densa no mesmo conjunto sem empates, dentro de `1e-5` para similaridades.
Proof: `.venv/bin/python3 -m unittest tests.test_c08_knn_workspace -k test_tiled_knn_global_metrics_match_dense_reference`

**C7** — Species, Genus e Order usam os mesmos índices elegíveis e rótulos atuais sem criar uma cópia integral das features por nível; Silhouette amostra antes de reunir as features.
Proof: `.venv/bin/python3 -m unittest tests.test_c08_knn_workspace -k test_rank_selection_and_silhouette_sample_before_feature_gather`

### S2 — Auditoria compacta e processos seguintes

**C8** — Cada auditoria concluída produz JSON `c08_knn_audit_v1` sem listas integrais de IDs e Parquet com uma linha por consulta e cinco índices de vizinhos reconstruíveis pelo artefato test identificado.
Proof: `.venv/bin/python3 -m unittest tests.test_c08_knn_workspace -k test_compact_knn_audit_roundtrip_and_artifact_binding`

**C9** — Resumo KNN registra hashes ordenados de referências, consultas e vizinhos, N, auto-vizinhos zero, dimensões de blocos, dispositivo e protocolo `c08_tiled_cosine_v1`.
Proof: `.venv/bin/python3 -m unittest tests.test_c08_knn_workspace -k test_knn_audit_metadata_hashes_and_counts`

**C10** — Erro na escrita de Parquet/JSON elimina temporários próprios e não publica auditoria completa nem manifesto de execução concluída.
Proof: `.venv/bin/python3 -m unittest tests.test_c08_knn_workspace -k test_audit_write_failure_is_atomic`

**C11** — Chamadas Python legadas continuam retornando `reference_ids`, `query_ids`, `neighbor_indices` e `neighbor_ids`; o notebook usa modo compacto e não serializa listas integrais de IDs.
Proof: `.venv/bin/python3 -m unittest tests.test_c08_knn_workspace -k test_legacy_knn_details_and_notebook_compact_mode`

**C12** — Notebook 06 conserva amostras Silhouette de no máximo 5.000 e UMAP de no máximo 100, registra amostras reais e classes excluídas no manifesto.
Proof: `.venv/bin/python3 -m unittest tests.test_c08_knn_workspace -k test_notebook_silhouette_umap_limits_and_manifest`

**C13** — Relatório de verificação separa testes sintéticos CPU, validação CUDA/dados remotos pendente e reprodução científica não realizada.
Proof: `python3 .agent/skills/tlc-spec-lean/scripts/validate_verification.py c08-knn-bounded-workspace`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Landing door 1 — protocolo (5) | cosseno/L2 C3 · split 80/20/42 C3 · k=5 C3 · voto C3 · desempate por índice C5 | - |
| Landing door 2 — auditoria (5) | JSON versionado C8 · IDs por hashes C9 · Parquet de índices C8 · vínculo ao test C8 · publicação atômica C10 | - |
| Superfície Python (4) | sucesso C3-C9,C11 · ValueError C2 · OSError C10 · RuntimeError C2 | - |
| Superfície notebook (2) | conclusão/manifesto C1,C8,C12 · erro sem auditoria completa C10 | - |
| Níveis (3) | Species C7 · Genus C7 · Order C7 | - |
| Processos posteriores (2) | Silhouette 5000 C7,C12 · UMAP 100 C12 | - |
| Bordas de blocos (4) | referência em um bloco C5 · referência em múltiplos blocos C3,C5 · consulta em lote cheio C4 · último lote incompleto C4 | - |
| Auditoria por avaliação (3) | referências C9 · consultas C8,C9 · vizinhos C8,C9 | - |

Claims: C1 cobre critérios 1–2; C2 cobre 3 e 8; C3 cobre 4; C4 cobre 5; C5 cobre 6; C6 cobre 7; C7 cobre 9; C8 cobre 10; C9 cobre 11; C10 cobre 12; C11 cobre 14–15; C12 cobre 13; C13 cobre 16.

## Swept

- validation: C2,C4
- failure modes: C2,C10
- idempotency: C8,C10 — execução distinta e publicação sem promover parcial
- authorization: n/a — pipeline local sem usuário autenticado ou serviço remoto
- concurrency: C5,C8,C10 — ordem determinística e caminhos por execução
- data lifecycle: C8,C10 — Parquet persistente, temporários removidos
- dependency failure: C10 — erro local de disco; rede não participa
- state transitions: C8,C10 — auditoria em construção → completa somente após publicação
- observability: C1,C9,C12,C13

## Handoff

- **Slices**: 2 (`S1`, `S2`), em um builder porque compartilham a mesma passagem KNN.
- **Files in play**: `src/viral_cnn/eval/latent.py`, notebook 06, `tests/test_c08_knn_workspace.py`, testes C01/latentes existentes, `plan.md`, este `checks.md`, `.specs/STATE.md` e relatório do Verifier.
- **Estimated load**: módulo/notebook/testes existentes ~125 KiB ÷ 4 ≈ 32k tokens + testes novos ~15k + implementação/revisão ~45k + artefatos ~8k = ~100k tokens < 150k.
- **Mechanism**: one builder; perfil `light`; Verifier independente após o último commit da implementação.
