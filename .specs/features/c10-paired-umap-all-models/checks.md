# C10 checks

Profile: light
Plan: `.specs/features/c10-paired-umap-all-models/plan.md` — aprovado pelo usuário (“sim”).

17 checks em duas slices, duas portas, nenhuma questão bloqueante.

## Checks

### S1 — seleção e projeções pareadas

**C1** — A primeira célula usa limites 100/10 e rejeita limites não inteiros ou abaixo de 3/1 antes das projeções (AC 1–2).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_configuration_boundary -k test_configuration_boundary`

**C2** — A seleção registra a exclusão de linhagem ausente sem alterar labels/IDs de entrada (AC 3).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_coverage_and_balanced_quotas -k test_coverage_and_balanced_quotas`

**C3** — A seleção usa seed 42, no máximo 100 IDs sem reposição e dez gêneros, redistribuindo quotas quando um gênero tem suporte pequeno (AC 4; door 1).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_coverage_and_balanced_quotas -k test_coverage_and_balanced_quotas`

**C4** — Reordenar as linhas dos cinco modelos preserva os IDs selecionados e alinha as features corretas (AC 5).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_permuted_models_project_identical_rows -k test_permuted_models_project_identical_rows`

**C5** — ID ausente, ID duplicado e label divergente são recusados antes do primeiro ajuste (AC 6).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_identity_errors_before_projection -k test_identity_errors_before_projection`

**C6** — Cinco projeções usam exatamente a mesma sequência de IDs/labels (AC 7).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_permuted_models_project_identical_rows -k test_permuted_models_project_identical_rows`

**C7** — Cada ajuste usa cosseno, duas dimensões, min_dist 0,1, seed 42 e vizinhos min(15,N−1), inclusive para amostra menor que 16 (AC 8).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_projection_parameters_and_small_sample -k test_projection_parameters_and_small_sample`

**C8** — Menos de três fragmentos selecionados causa ValueError e nenhum manifesto completo (AC 9).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_insufficient_sample -k test_insufficient_sample`

**C9** — A projeção obtém somente as features selecionadas; não materializa o array integral nem altera KNN/Silhouette (AC 10).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_selected_feature_access_and_quantitative_contracts -k test_selected_feature_access_and_quantitative_contracts`

### S2 — saídas, método e cache

**C10** — Cinco CSVs e seleção comum preservam IDs/labels/Genus/método, com hashes, parâmetros e contagens nos metadados (AC 11; door 2).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_outputs_provenance_and_panel -k test_outputs_provenance_and_panel`

**C11** — O painel contém cinco variantes, cores/legenda comuns e aviso de interpretação qualitativa e eixos independentes (AC 12).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_outputs_provenance_and_panel -k test_outputs_provenance_and_panel`

**C12** — Sem UMAP, o fallback se identifica como PCA nos CSVs, metadados e títulos/eixos do gráfico (AC 13).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_pca_fallback_is_explicit -k test_pca_fallback_is_explicit`

**C13** — Reuse restaura as cinco variantes sem ajuste, validando a seleção e suas dependências (AC 14).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_resume_and_variant_corruption -k test_resume_and_variant_corruption`

**C14** — Corromper um CSV invalida somente a projeção correspondente, preservando os bytes de todas as outras fontes (AC 15).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_resume_and_variant_corruption -k test_resume_and_variant_corruption`

**C15** — Falha de escrita impede manifesto completo e permite retomada das variantes já concluídas (AC 16).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.PairedUMAPTests.test_interrupted_publication -k test_interrupted_publication`

**C16** — O smoke do notebook gera cinco projeções/painel, registra o protocolo no manifesto e retoma sem ajustes novos (AC 7, 11, 14).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.NotebookPairedUMAPTests.test_notebook_smoke_resume -k test_notebook_smoke_resume`

**C17** — A documentação distingue primeiras 100 linhas/C10, invalidação C09 pelo hash global e validação CPU sintética da empírica (AC 17).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c10_paired_umap.NotebookPairedUMAPTests.test_documented_limits -k test_documented_limits`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Variantes (5) | resnet_clean_species C6 · resnet_noise_species C6 · convnext_zero_species C6 · convnext_transfer_species C6 · convnext_noise_species C6 | - |
| Configuração (4) | padrão C1 · override C1 · não-inteiro C1 · abaixo-limite C1 | - |
| Erros de comparação (4) | insuficiente C8 · ausente C5 · duplicado C5 · label-divergente C5 | - |
| Retomada (4) | reutilizado C13 · descartado C14 · recalculado C14 · falha-operacional C15 | - |
| Métodos (2) | UMAP C7 · PCA C12 | - |
| Saídas (4) | CSV-por-variante C10 · seleção C10 · metadados C10 · painel C11 | - |
| Entidades (4) | comparação C10 · seleção C3 · projeção C6 · checkpoint C13 | - |
| Portas (2) | população-pareada C3 · saídas-cache C10 | - |
| Assemblies (1) | primeira-célula-notebook C1 | - |

## Swept

- validation: C1, C5, C8
- failure modes: C14–C15
- idempotency: C3–C4, C13–C15
- authorization: n/a — notebook e arquivos locais sem serviço externo
- concurrency: C14–C15; namespaces exclusivos existentes mantidos
- data lifecycle: C10, C13–C15
- dependency failure: C12, C15
- state transitions: C13–C15
- observability: C1–C2, C10, C12, C17

## Handoff

Um builder; Verifier independente somente após commit. Arquivos existentes relevantes ~158 kB / 4 = 39,5k tokens; módulo/testes/documentação novos estimados em 44 kB / 4 = 11k; total ~50,5k < orçamento padrão 150k. Sem divisão da implementação.

O teste C08 que exige literalmente `umap_feats[:100]` será atualizado para a seleção pareada aprovada, preservando sua asserção de limite padrão 100 e os controles Silhouette. Essa mudança acompanha a nova metodologia identificada, sem remover teste ou enfraquecer o limite.

- **Boundary:** C1–C17 implementados; bateria latente ampliada com 93 testes verdes (inclui 14 C10), antes do commit de implementação e do Verifier.
- **Settled mid-build:** a camada C10 ajusta UMAP explicitamente para propagar falhas de dependência durante fit; reutiliza a função histórica para PCA quando UMAP não é importável. Essa decisão evita rotular o fallback silencioso histórico como UMAP e foi registrada em Flow antes do código. Ausência local de IPython só desativa exibição inline; o PNG continua sendo salvo.
- **Abandoned:** nenhuma mudança em `eval/latent.py`, no amostrador KNN/Silhouette ou em notebooks/manuscrito históricos; não houve migração permissiva de caches antigos. As provas apenas foram fortalecidas com invalidação por seleção/hash/backend e interrupção entre CSVs, sem reduzir asserções.
