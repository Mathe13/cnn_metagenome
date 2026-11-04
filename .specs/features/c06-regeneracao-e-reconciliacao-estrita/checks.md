# C06 checks

Profile: light
Plan: `.specs/features/c06-regeneracao-e-reconciliacao-estrita/plan.md`

27 checks em 4 slices · 3 portas unidirecionais · 0 questões abertas, 0 bloqueadoras

## Checks

### S1 — Seleção e portão de admissibilidade · ~45 KB · ~11k

**C1** — A configuração C06 indexa explicitamente Tabelas 1–12 e Figuras 1–5, e cada item declara entradas, transformação, protocolo e condições esperadas (AC 1–2)
Proof: `PYTHONPATH=src python3 -m unittest -k test_index_covers_all_publication_items tests/test_c06_publication_package.py`

**C2** — A validação rejeita arquivo ausente, ilegível, hash divergente ou manifesto sem modo/protocolo exigido, identificando item, campo e caminho (AC 2–3)
Proof: `PYTHONPATH=src python3 -m unittest -k test_selected_input_validation_reports_missing_corrupt_and_unproven_inputs tests/test_c06_publication_package.py`

**C3** — A validação rejeita incompatibilidade de ordem de classes, checkpoint/variante, população ou composição de modelos e condições de ruído (AC 2)
Proof: `PYTHONPATH=src python3 -m unittest -k test_scientific_identity_and_composition_must_match tests/test_c06_publication_package.py`

**C4** — `strict_empirical=True` encerra com diagnóstico e não produz pacote completo quando uma entrada obrigatória está ausente, corrompida, sintética, incompleta ou sem proveniência (AC 3, 6)
Proof: `PYTHONPATH=src python3 -m unittest -k test_strict_gate_fails_closed_per_required_item tests/test_c06_publication_package.py`

**C5** — A semântica esperada da Tabela 11 rejeita artefato de coocorrência como substituto da correlação entre arquiteturas (AC 4)
Proof: `PYTHONPATH=src python3 -m unittest -k test_table_11_requires_architecture_correlation_semantics tests/test_c06_publication_package.py`

**C6** — Manifesto sintético não selecionado não altera a admissibilidade; dependência sintética transitiva de entrada selecionada bloqueia o pacote (AC 5)
Proof: `PYTHONPATH=src python3 -m unittest -k test_only_selected_and_transitive_inputs_affect_admissibility tests/test_c06_publication_package.py`

**C7** — Seleção parcialmente elegível preserva diagnóstico das lacunas e mantém o estado do pacote incompleto, sem publicar estado empírico integral (AC 6)
Proof: `PYTHONPATH=src python3 -m unittest -k test_partial_selection_is_diagnostic_and_never_complete tests/test_c06_publication_package.py`

### S2 — Derivação de tabelas e figuras · ~55 KB · ~14k

**C8** — As 12 tabelas estritas são derivadas das entradas validadas e exportadas em CSV, Markdown e LaTeX; referências históricas ficam separadas (AC 7)
Proof: `PYTHONPATH=src python3 -m unittest -k test_tables_are_derived_and_historical_values_are_separate tests/test_c06_publication_package.py`

**C9** — Contagens estruturais registram código/configuração e N não aplicável com motivo, sem preencher latência, throughput ou FLOPs ausentes por estimativa (AC 8)
Proof: `PYTHONPATH=src python3 -m unittest -k test_structural_counts_are_separated_from_unmeasured_costs tests/test_c06_publication_package.py`

**C10** — Métricas e intervalos preservam unidade, estimando, denominador, suporte, classes sem suporte e estratos de protocolo; populações incompatíveis não são combinadas (AC 9)
Proof: `PYTHONPATH=src python3 -m unittest -k test_metrics_preserve_estimands_support_and_protocol_strata tests/test_c06_publication_package.py`

**C11** — Reconciliação calcula diferenças sem arredondamento intermediário, usa pp apenas para percentuais compatíveis e explica comparações indisponíveis entre populações incompatíveis (AC 10)
Proof: `PYTHONPATH=src python3 -m unittest -k test_discrepancies_use_compatible_units_and_keep_full_precision tests/test_c06_publication_package.py`

**C12** — As Figuras 1–5 são renderizadas como PNG e PDF a partir de dados validados e parâmetros persistidos, sem imagem extraída nem padrão procedural (AC 11)
Proof: `PYTHONPATH=src python3 -m unittest -k test_figures_render_from_validated_inputs_without_fallbacks tests/test_c06_publication_package.py`

**C13** — Painéis 3(d) e 3(e) apontam respectivamente para manifestos/checkpoints ResNet50_F e ConvNeXt_F, e identidade é validada por proveniência (AC 12)
Proof: `PYTHONPATH=src python3 -m unittest -k test_figure_3_panel_identity_uses_variant_provenance tests/test_c06_publication_package.py`

**C14** — Seleções qualitativas/estatísticas registram IDs, níveis, amostragem, semente e filtros; UMAP permanece visual e composição de tripletos exploratória (AC 13)
Proof: `PYTHONPATH=src python3 -m unittest -k test_qualitative_and_statistical_selections_are_recorded tests/test_c06_publication_package.py`

**C15** — Falha de transformação, renderização ou caminho com caixa divergente registra erro e impede estado completo, sem placeholder ou fallback histórico (AC 14)
Proof: `PYTHONPATH=src python3 -m unittest -k test_render_and_case_sensitive_path_failures_block_completion tests/test_c06_publication_package.py`

### S3 — Manuscrito e interpretação · ~50 KB · ~12.5k

**C16** — Variantes review e submission consomem as mesmas tabelas, figuras e parâmetros do pacote, sem tabelas numéricas independentes no template (AC 15)
Proof: `PYTHONPATH=src python3 -m unittest -k test_both_variants_consume_the_same_package_results tests/test_c06_publication_package.py`

**C17** — Methods/Results descrevem split por fragmentos, ruído efetivo `3ε/4`, variantes/seleção observadas, KNN em alta dimensão com cosseno e populações de IC/calibração (AC 16)
Proof: `PYTHONPATH=src python3 -m unittest -k test_methods_and_results_follow_observed_protocols tests/test_c06_publication_package.py`

**C18** — Diagnóstico inconclusivo sobre origem genômica ou queda em sequências longas qualifica Abstract, Discussion e Conclusions sem declarar mecanismo causal ou genomas vistos/não vistos (AC 17)
Proof: `PYTHONPATH=src python3 -m unittest -k test_inconclusive_evidence_limits_genome_and_causal_claims tests/test_c06_publication_package.py`

**C19** — C05 `NOT_RUN`, `PILOT` ou `PARTIAL` limita conclusões às configurações avaliadas e não impede pacote válido sem dependência C05 obrigatória (AC 18)
Proof: `PYTHONPATH=src python3 -m unittest -k test_optional_c05_state_limits_claims_without_blocking_package tests/test_c06_publication_package.py`

**C20** — Rascunho não estrito rotula números históricos sem reprodução como `HISTORICAL_UNREPRODUCED` e não promove o pacote a empírico completo (AC 19)
Proof: `PYTHONPATH=src python3 -m unittest -k test_historical_values_remain_unreproduced_and_noncomplete tests/test_c06_publication_package.py`

**C21** — Review conserva as 56 notas editoriais, submission exclui notas/macros e ambas mantêm os mesmos resultados e limitações científicas (AC 20)
Proof: `PYTHONPATH=src python3 -m unittest -k test_review_notes_are_preserved_and_submission_is_clean tests/test_c06_publication_package.py`

**C22** — Relatório editorial lista referências, afiliações e redação como revisados ou pendentes, sem inferir aprovação a partir da compilação (AC 21)
Proof: `PYTHONPATH=src python3 -m unittest -k test_editorial_review_status_is_explicit_and_independent_of_compile tests/test_c06_publication_package.py`

### S4 — Montagem isolada, manifesto e limites de reprodução · ~55 KB · ~14k

**C23** — CLI e notebook mestre usam a mesma seleção/portão e namespace C00 exclusivo, não gravam nas entradas e não sobrescrevem fontes/PDFs históricos (AC 22)
Proof: `PYTHONPATH=src python3 -m unittest -k test_cli_and_notebook_use_isolated_run_namespace tests/test_c06_publication_package.py`

**C24** — Primeira célula do notebook resolve `src` a partir da raiz do repositório antes de importar `viral_cnn`, e notebook não dispara treino, download ou inferência (AC 23)
Proof: `PYTHONPATH=src python3 -m unittest -k test_master_notebook_bootstraps_src_without_heavy_side_effects tests/test_c06_publication_package.py`

**C25** — Compilação retém PDF e logs na saída da execução, verifica referências e preserva os hashes das fontes históricas anteriores (AC 24)
Proof: `PYTHONPATH=src python3 -m unittest -k test_compile_artifacts_are_isolated_and_historical_sources_unchanged tests/test_c06_publication_package.py`
Proof: `PYTHONPATH=src python3 -m unittest -k test_complete_fixture_assembly_renders_and_compiles_in_reserved_package tests/test_c06_publication_package.py`

**C26** — Manifesto C06 registra comando/interface, commit e estado sujo, configuração, fontes e hashes, versões, saídas e estados dos portões local, empírico, científico e editorial (AC 25)
Proof: `PYTHONPATH=src python3 -m unittest -k test_manifest_records_inputs_outputs_versions_and_gates tests/test_c06_publication_package.py`

**C27** — Montagem com fixtures marca evidência como sintética e mantém portão empírico bloqueado; relatório distingue implementação validada, regeneração empírica e prontidão editorial e informa comandos, manifestos, diferenças e dependências indisponíveis (AC 26–27)
Proof: `PYTHONPATH=src python3 -m unittest -k test_synthetic_run_and_final_report_do_not_claim_reproduction tests/test_c06_publication_package.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Landing doors (3) | índice/configuração explícitos C1 · matriz/status de evidência C2,C4,C7,C26 · notebook mestre C23,C24 | - |
| Relations (0) | n/a - plano declara que não há mudança de cardinalidade/entidades persistidas | - |
| Publication items (17) | Tabela 1 C1,C8-C11 · Tabela 2 C1,C8-C11 · Tabela 3 C1,C8,C10-C11 · Tabela 4 C1,C8,C10-C11 · Tabela 5 C1,C8,C10-C11 · Tabela 6 C1,C8,C10-C11 · Tabela 7 C1,C8,C10-C11 · Tabela 8 C1,C8,C10-C11 · Tabela 9 C1,C8,C10-C11 · Tabela 10 C1,C8,C10-C11 · Tabela 11 C1,C5,C8,C10-C11 · Tabela 12 C1,C8,C10-C11 · Figura 1 C1,C12,C14-C15 · Figura 2 C1,C12,C14-C15 · Figura 3 C1,C12-C15 · Figura 4 C1,C12,C14-C15 · Figura 5 C1,C12,C14-C15 | - |
| Evidence statuses (6) | `EMPIRICAL_VERIFIED` C2,C4,C26 · `DERIVED_VERIFIED` C9,C26 · `HISTORICAL_UNREPRODUCED` C20 · `SYNTHETIC` C27 · `MISSING` C2,C4,C7 · `INVALID` C2-C5,C15 | - |
| Editorial statuses (2) | review with 56 notes C21-C22 · submission without notes/macros C21-C22 | - |
| Package gates (4) | local C23-C25 · empirical C4,C6-C7,C27 · scientific C3,C5,C10,C13-C20 · editorial C22,C26-C27 | - |
| CLI/notebook surfaces (4) | `scripts/generate_paper_tables.py` C23 · `scripts/generate_paper_figures.py` C23 · `scripts/compile_paper.py` C25 · notebook mestre C23-C24 | - |
| HTTP/API routes (0) | n/a - pipeline local, sem serviço HTTP (Surface) | - |

- Itens da matriz T1–T12/F1–F5 enumerados individualmente: 17/17 cobertos.
- Portas de Landing: 3/3 cobertas; Relations não introduz entidades persistidas.
- Não há teste de política adicional no perfil `light`; os contratos locais serão verificados por testes nomeados acima.

## Swept

- validation: C1-C5,C8-C15,C26 (schema, identidade, semântica, unidades, suporte e caminhos)
- failure modes: C2,C4-C7,C15,C25,C27 (ausência/corrupção, parcialidade, renderização e compilação)
- idempotency: C23 (cada montagem reserva namespace exclusivo e não substitui fontes)
- authorization: n/a - execução local sem serviço remoto, identidade de usuário ou rate limit
- concurrency: C23 (reserva exclusiva por execução; montagem só publica pacote completo validado)
- data lifecycle: C20,C23,C25-C27 (histórico preservado, outputs isolados, hashes e estado de pacote)
- dependency failure: C2,C4,C15,C25,C27 (entradas, transformações, renderização e compilador ausentes/falhos)
- state transitions: C4,C7,C19-C20,C26-C27 (incompleto, bloqueado, histórico, sintético e gates)
- observability: C2,C7,C10-C11,C14-C15,C22,C26-C27 (diagnósticos, discrepâncias, proveniência e status)

## Handoff

- S1 ~11k + S2 ~14k + S3 ~12.5k + S4 ~14k = ~51.5k tokens; escopo previsto de reporting, CLIs, notebook, manifesto, documentação e testes: ~240 KB / 4 ≈ 60k tokens, abaixo do budget `light` de 150k.
- Mecanismo: um builder; não há corte de superfície necessário.
- **Boundary:** implementação C06 concluída; 27 seletores nomeados passaram individualmente e os 28 testes C06 passaram em conjunto. Montagem empírica continua bloqueada até selecionar dados/checkpoints e manifestos canônicos.
- **Settled mid-build:** manifesto registra run_id do namespace C00. A fonte histórica disponível contém 56 chamadas `\\notavdois`, conforme o critério aprovado; a submission tem zero chamadas.
- **Abandoned:** nenhum.
