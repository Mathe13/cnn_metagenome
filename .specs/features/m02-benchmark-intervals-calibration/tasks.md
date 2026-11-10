# M02 Tasks

## Execution Protocol

Implementar com o skill `tlc-spec-driven` v4.1.0 já localizado e lido nesta sessão. Seguir seu ciclo Execute, gates, commit por tarefa e Verifier independente. Caminho do skill: `/home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/.agents/skills/tlc-spec-driven/`. Não substituir por tlc-spec-lean.

**Design**: `.specs/features/m02-benchmark-intervals-calibration/design.md`
**Status**: In Progress
**Approved**: 2026-11-09 - user: "aprovado"
**Build**: none - repositório sem comando de build, linter ou typecheck configurado; sintaxe/imports e contratos são exercitados por unittest.
**Run m02**: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_m02_benchmark_intervals_calibration`
**Regression command**: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_c03_baselines tests.test_c04_uncertainty tests.test_calibration tests.test_m00_run_plan tests.test_c00_preflight_manifest tests.test_notebook_runtime`

A especificação foi aprovada em 09/11/2026: user: "prossiga". Tarefas e design A aprovados em 09/11/2026: user: "aprovado". Provas M02 abaixo são seletores planejados, ainda sem execução; nenhuma saída verde é afirmada.

## Test Coverage Matrix

> Gerado das diretrizes AGENTS.md da raiz e src/viral_cnn/AGENTS.md, seção Testing do README e amostra de seis suítes: C03, C04, calibration, M00, M01 e N02. unittest, TemporaryDirectory, valores esperados literais, patch controlado e execução de células são padrões encontrados. Não há manifesto de build/lint/typecheck configurado. Testes novos limitam-se aos contratos científicos e de rastreabilidade necessários ao M02.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| --- | --- | --- | --- | --- |
| Validação C03 e preflight | integration | Pacote completo de 35 condições; casos negativos de identidade, hashes, modo, classes, referência e conclusão. Preflight e notebook devem alcançar o mesmo guard. | tests/test_m02_benchmark_intervals_calibration.py | Run m02, seletores C4–C8/C24 |
| Ciclo e resolução de artefatos | integration | Três etapas, standalone e plano, colisão, falha e invariância dos bytes das fontes. | tests/test_m02_benchmark_intervals_calibration.py | Run m02, seletores C2/C3/C9–C12/C25/C26 |
| Métricas, deltas e figuras | unit | Valores literais, 105/150 células, abstenção, vazio/espécie única, unidades, N e dados que reproduzem figuras. | tests/test_m02_benchmark_intervals_calibration.py | Run m02, seletores C18–C23 |
| Notebooks e calibração | integration | Executar células em namespace temporário, modelos mínimos sem checkpoints reais; fit/selo, suporte, GradCAM e manifestos. | tests/test_m02_benchmark_intervals_calibration.py | Run m02, seletores C1/C9–C17/C26 |
| Documentação e configuração | none | Revisão contra contratos implementados, comandos e schemas carregados pelo teste de integração. | docs/ e configs/ | Build gate e revisão documental |

Novos testes M02 forçam REMOTE_SERVER=False e limitam threads de CPU quando necessário. Toda fixture em TemporaryDirectory; nenhuma chamada de rede, treinamento completo ou instalação automática. Entradas fixture podem reproduzir o schema empírico para exercitar guards, mas não são resultados científicos empíricos.

## Gate Check Commands

> Gerado dos comandos unittest documentados no repositório. Sem nova ferramenta de qualidade.

| Gate Level | When to Use | Command |
| --- | --- | --- |
| Quick | Tarefas unitárias | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_m02_benchmark_intervals_calibration.M02Tests.test_<seletor_da_tarefa>` para os seletores da tarefa; nomes concretos estão em Checks. |
| Full | Integração de componente/notebook | Run m02 selecionando os testes já entregues e os adicionados na tarefa; nenhum teste é adiado para tarefa posterior. |
| Build | Config/docs e encerramento | `git diff --check`; Run regressions e Run m02 uma vez no HEAD final via check_proofs.py. |

Suítes existentes completas são executadas no encerramento; gates por tarefa selecionam os testes relevantes da tarefa. O mínimo planejado é 34 métodos M02, incluindo os 26 seletores de Checks e oito exercícios adicionais de integração/componentes. Subtests distinguem etapas, modos, fontes e mutações; o número final real será registrado sem excluir provas aprovadas. Regressões N02/N06 que executam notebooks entram somente com sandbox temporário confirmado, sem executar o legado contra DATA_ROOT compartilhado.

## Execution Plan

Oito tarefas, execução inline sequencial. Nenhum batch worker necessário. Verifier independente obrigatório após o último commit e provas; não há UI/capturas vinculantes nem screens agent.

```
T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8
```

### Phase 1: Foundation

```
T1 → T2
```

### Phase 2: Scientific outputs

```
T3 → T4
```

### Phase 3: Notebook adoption and documentation

```
T5 → T6 → T7 → T8
```

A ordem entre fases é estrita. Além da ordem, dependências de implementação são mostradas abaixo; todas apontam para tarefas anteriores.

## Task Breakdown

### T1: Inspeção especializada do pacote C03

**What**: Expor a inspeção das 35 condições e conectá-la aos preflights C00/M00 existentes, com config C03 coerente.
**Where**: `src/viral_cnn/baselines/overlap.py`
**Depends on**: None
**Reuses**: load_prepared_baseline_benchmark e verificações atuais de conteúdo, referência e checkpoint.
**Requirement**: M02-01, M02-02, M02-03

**Done when**:

- [x] Fixture estrutural completa aceita; ausência de build/vínculo, condição incompleta, hash inválido, queries divergentes, classes e política incompatíveis rejeitadas antes de métricas.
- [x] Provas C4–C8, C24 implementadas com a tarefa; nenhum teste exigido é adiado.
- [x] Mínimo de 8 métodos novos de teste ou equivalentes subtests explicitamente registrados.
- [x] Gate Full passa; contagem real e resultado registrados antes do commit.

**Tests**: integration
**Gate**: Full
**Commit**: `feat(m02): validate complete C03 packages before benchmark`

**Result**: 8 testes M02 passaram em 12,685 s; git diff --check passou. Pacote é fixture estrutural, sem inferência.

### T2: Ciclo de execução M02

**What**: Criar o componente de resolução e reserva M02 com publicação atômica COMPLETE/FAILED e fontes smoke isoladas.
**Where**: `src/viral_cnn/m02_workflow.py`
**Depends on**: T1
**Reuses**: reserve_run_directory, file_identity, load_run_plan, LOG_PATH e padrão de claim M01.
**Requirement**: M02-01, M02-04

**Done when**:

- [x] Entradas explícitas/plano identificadas; conflito rejeitado; duas execuções disjuntas; colisão e falha preservam sentinelas; manifesto tem hashes reais, lineage e modo.
- [x] Provas C2, C3, C9–C12, C25, C26 implementadas com a tarefa; nenhum teste exigido é adiado.
- [x] Mínimo de 7 métodos novos de teste ou equivalentes subtests explicitamente registrados.
- [x] Gate Full passa; contagem real e resultado registrados antes do commit.

**Tests**: integration
**Gate**: Full
**Commit**: `feat(m02): isolate benchmark run lifecycle and inputs`

**Result**: 8 testes de ciclo/entrada passaram em 2,443 s. Snapshot de fontes e falha sem fixtures confirmados.

### T3: Resumo e curvas dos sete métodos

**What**: Criar o componente que exporta métricas das 105 células e renderiza curvas de acurácia/cobertura a partir dos dados persistidos.
**Where**: `src/viral_cnn/baselines/benchmark_plotting.py`
**Depends on**: T2
**Reuses**: evaluate_predictions e wilson_score_interval; estilo matplotlib Agg do projeto.
**Requirement**: M02-07, M02-09

**Done when**:

- [x] 105 células mantidas; abstenção conserva N; sete métodos e três estratos visíveis; N/IC Wilson identificados; CSV e PNG/PDF válidos.
- [x] Provas C18–C20, C23 implementadas com a tarefa; nenhum teste exigido é adiado.
- [x] Mínimo de 4 métodos novos de teste ou equivalentes subtests explicitamente registrados.
- [x] Gate Quick passa; contagem real e resultado registrados antes do commit.

**Tests**: unit
**Gate**: Quick
**Commit**: `feat(m02): render seven-method accuracy and coverage curves`

**Result**: 4 testes de resumo/curvas passaram; 105 células, abstenção, N/Wilson e PNG/PDF derivados do CSV verificados.

### T4: Diferenças pareadas CNN–ferramenta

**What**: Acrescentar o componente de deltas dos dez pares CNN–BLASTN/Kraken2, tabelas e figura derivada.
**Where**: `src/viral_cnn/baselines/benchmark_plotting.py`
**Depends on**: T3
**Reuses**: bootstrap_species_difference e funções de renderização da tarefa T3.
**Requirement**: M02-08, M02-09

**Done when**:

- [x] 150 células preservadas; diferenças em pontos percentuais com bootstrap por espécie; vazio/espécie única têm IC indisponível; sinal e pareamento conferidos contra valores literais.
- [x] Provas C21–C23 implementadas com a tarefa; nenhum teste exigido é adiado.
- [x] Mínimo de 3 métodos novos de teste ou equivalentes subtests explicitamente registrados.
- [x] Gate Quick passa; contagem real e resultado registrados antes do commit.

**Tests**: unit
**Gate**: Quick
**Commit**: `feat(m02): add paired species bootstrap difference panels`

**Result**: 3 testes passaram: 150 deltas pareados, ICs indisponíveis e figura regenerada de CSV.

### T5: Notebook 01 no ciclo M02

**What**: Adaptar o notebook 01 ao ciclo M02 preservando profiling, métricas, IC C04 e faixas F1.
**Where**: `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb`
**Depends on**: T4
**Reuses**: Workflow T2, evaluate_dataloader, bootstrap_classification_intervals e evidence C04.
**Requirement**: M02-04, M02-05

**Done when**:

- [x] Células executadas sobre fixtures temporárias com modelos mínimos; destinos M00/standalone isolados; protocolo C04, semente/repetições e escopo de ruído persistidos.
- [x] Provas C9–C13, C17, C26 implementadas com a tarefa; nenhum teste exigido é adiado.
- [x] Mínimo de 3 métodos novos de teste ou equivalentes subtests explicitamente registrados.
- [x] Gate Full passa; contagem real e resultado registrados antes do commit.

**Tests**: integration
**Gate**: Full
**Commit**: `feat(m02): adopt isolated C04 benchmark notebook runs`

**Result**: 4 testes passaram: notebook 01 inteiro com backbone mínimo, 35 evidências C04, IC 2000/42/95%, falha isolada e reserva/split M00.

### T6: Notebook 02 com calibração e GradCAM delimitado

**What**: Adaptar o notebook 02 ao ciclo M02, restringir GradCAM aleatório ao smoke e tornar N explícito na figura de confiabilidade.
**Where**: `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb`
**Depends on**: T5
**Reuses**: Workflow T2; TemperatureScaler selado e save_reliability_comparison; ajuste mínimo do renderizador existente pertence ao mesmo componente.
**Requirement**: M02-04, M02-05, M02-06

**Done when**:

- [x] Fixtures percorrem células; temperatura ajustada só em validation limpa e selada; fit em test/identidade sobreposta falha; N e suporte antes/depois conferidos; caminho empírico não chama GradCAM aleatório.
- [x] Provas C9–C12, C14–C17, C26 implementadas com a tarefa; nenhum teste exigido é adiado.
- [x] Mínimo de 5 métodos novos de teste ou equivalentes subtests explicitamente registrados.
- [x] Gate Full passa; contagem real e resultado registrados antes do commit.

**Tests**: integration
**Gate**: Full
**Commit**: `fix(m02): constrain GradCAM smoke and preserve sealed calibration`

**Result**: 5 testes passaram em 12,546 s; notebook 02 inteiro, validação limpa/SEALED, quatro diagramas, N e suporte, GradCAM omitido no modo empírico. Corrigida leitura de batches com IDs nas rotinas de confusão. Fixtures têm 64 amostras e identidade de validação distinta; gates de completude C04 explícitos.

### T7: Notebook 09 consumidor C03

**What**: Remover overrides históricos e conectar validação, ciclo M02 e tabelas/figuras completas sem inferência de métodos.
**Where**: `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb`
**Depends on**: T6
**Reuses**: Componentes T1–T4, métricas/discordância C03 e fixture smoke explícita.
**Requirement**: M02-01, M02-02, M02-03, M02-04, M02-07, M02-08, M02-09

**Done when**:

- [x] Execução completa do notebook em namespace temporário aceita fixture válida e bloqueia pacote inválido; ambiente preservado; nenhuma chamada a ferramentas/modelos; CSVs/figuras e hashes no manifesto.
- [x] Provas C1–C12, C17–C26 implementadas com a tarefa; nenhum teste exigido é adiado.
- [x] Mínimo de 4 métodos novos de teste ou equivalentes subtests explicitamente registrados.
- [x] Gate Full passa; contagem real e resultado registrados antes do commit.

**Tests**: integration
**Gate**: Full
**Commit**: `feat(m02): consume validated paired benchmark in notebook 09`

**Result**: 6 testes de adoção/guards passaram: consumo completo sem torch.load/build, smoke preserva ambiente e 40 queries, entrada inválida/ausente falha e segunda realização de ruído é rejeitada com hashes coerentes. Removidos overrides históricos; 105 métricas/150 deltas e três figuras derivadas.

### T8: Guia de execução e limites M02

**What**: Documentar configuração, preflight especializado, execução por notebook, produtos e limites científicos, com links README e atualização de STATE.
**Where**: `docs/m02_benchmark_intervals_calibration.md`
**Depends on**: T7
**Reuses**: Convenções documentais M00/M01; config standalone M02 e referências de plano entregues com o componente.
**Requirement**: M02-01, M02-04, M02-05, M02-09

**Done when**:

- [ ] Comandos exatos e arquivos/manifests descritos; dependências canônicas ausentes declaradas; distingue uma realização de ruído, unidades de IC e ausência de reprodução empírica local.
- [ ] Provas Revisão documental e manifesto; critérios científicos cobertos nas tarefas anteriores. implementadas com a tarefa; nenhum teste exigido é adiado.
- [ ] Revisão dos comandos e limites científicos concluída.
- [ ] Gate Build passa; contagem real e resultado registrados antes do commit.

**Tests**: none
**Gate**: Build
**Commit**: `docs(m02): document benchmark execution and evidence limits`

## Checks

**C1** - 09 conserva DATA_ROOT e SMOKE_TEST do ambiente (M02-01, C03 AC 1).
Proof: run m02 | test_notebook09_preserves_environment (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C2** - Fonte C03 por ambiente/plano registra caminho e SHA-256 (M02-01, C03 AC 2).
Proof: run m02 | test_source_paths_and_hashes (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C3** - Overrides de fonte divergentes do plano falham antes de métricas (M02-01, C03 AC 3).
Proof: run m02 | test_conflicting_plan_source (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C4** - Preflight e avaliação rejeitam preparação/build/condição ausente, incompleta ou com hash inválido nas 35 condições (M02-02, C03 AC 4).
Proof: run m02 | test_c03_completeness_matrix (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C5** - IDs, sequência e rótulos distintos na mesma taxa são rejeitados (M02-02, C03 AC 5).
Proof: run m02 | test_c03_query_content_identity (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C6** - Classes, referência e abstenção incompatíveis com C03 são rejeitadas (M02-02, C03 AC 6).
Proof: run m02 | test_c03_classes_reference_abstention (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C7** - Dependência sintética é rejeitada em modo empírico (M02-02, C03 AC 7).
Proof: run m02 | test_c03_synthetic_dependency (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C8** - 09 consome C03 completo sem executar CNN/Kraken2/BLASTN (M02-03, C03 AC 8).
Proof: run m02 | test_notebook09_consumes_without_inference (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C9** - 01/02/09 reservam destinos exclusivos e adotam run_id/parent M00 (M02-04, Execução AC 1).
Proof: run m02 | test_three_notebook_reservations (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C10** - Colisão/entrada ausente não sobrescreve fonte nem publica COMPLETE (M02-04, Execução AC 2).
Proof: run m02 | test_collision_and_missing_inputs (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C11** - Manifestos têm protocolo, modo, população/split, seed, commit/dirty, configuração e identidades/hashes reais (M02-04, Execução AC 3).
Proof: run m02 | test_manifest_identity_and_lineage (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C12** - Falha após reserva registra FAILED nos três fluxos (M02-04, Execução AC 4).
Proof: run m02 | test_failed_run_manifest (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C13** - IC C04 identifica fragmento estratificado/suporte fixo, 95%, seed 42 e repetições (M02-05, Execução AC 5).
Proof: run m02 | test_c04_interval_contract (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C14** - Temperatura usa validação limpa identificada e selo antes de test; test/identidade sobreposta rejeitados (M02-05, Execução AC 6).
Proof: run m02 | test_validation_temperature_seal (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C15** - Confiabilidade exibe N, conserva suporte e vincula ECE/Brier/NLL às entradas (M02-05, Execução AC 7).
Proof: run m02 | test_reliability_support_and_n (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C16** - 02 empírico omite GradCAM aleatório e registra omissão; smoke conserva demonstração (M02-06, Execução AC 8).
Proof: run m02 | test_gradcam_smoke_only (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C17** - Três fluxos registram uma realização por taxa e ausência de variabilidade entre treinamentos (M02-05, Execução AC 9).
Proof: run m02 | test_uncertainty_scope (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C18** - Resumo contém sete métodos × cinco taxas × três estratos e três métricas separadas (M02-07, Curvas AC 1).
Proof: run m02 | test_full_metric_grid (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C19** - Abstenção -1 conta como erro global e permanece em N (M02-07, Curvas AC 2).
Proof: run m02 | test_abstention_denominators (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C20** - Curvas mostram os sete métodos em três estratos com N e Wilson identificado (M02-07, Curvas AC 3).
Proof: run m02 | test_accuracy_coverage_panels (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C21** - Dez pares CNN–ferramenta × cinco taxas × três estratos usam pp, espécie, 2000 repetições e seed 42 (M02-08, Curvas AC 4).
Proof: run m02 | test_paired_difference_grid (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C22** - Estrato vazio/espécie única declara motivo de IC indisponível sem limites inventados (M02-08, Curvas AC 5).
Proof: run m02 | test_paired_unavailable_intervals (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C23** - Figuras derivam de CSV e PNG/PDF têm vínculos e hashes no manifesto (M02-09, Curvas AC 6).
Proof: run m02 | test_csv_figure_regeneration_and_hashes (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C24** - COMPLETE com produto alterado é recusado pelo hash (M02-02, Edge Cases AC 1).
Proof: run m02 | test_complete_with_corrupted_product (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C25** - Reuso de reserva preserva todos os bytes existentes (M02-04, Edge Cases AC 2).
Proof: run m02 | test_reservation_preserves_bytes (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

**C26** - Falta empírica falha sem gerar fixture na fonte compartilhada (M02-04, Edge Cases AC 3).
Proof: run m02 | test_empirical_failure_never_creates_fixtures (tests.test_m02_benchmark_intervals_calibration.M02Tests) ... ok

## Granularity and Co-location

Cada tarefa entrega um componente ou uma adaptação de notebook. Hooks preflight/config estritamente necessários acompanham o componente que ativam; testes são escritos no mesmo commit. T8 é uma entrega documental coesa, com README/STATE apenas como referências de navegação e handoff.

| Task | Layer | Matrix requires | Tests | Dependency / diagram |
| --- | --- | --- | --- | --- |
| T1 | Validação e preflight C03 | integration | integration | None |
| T2 | Ciclo e resolução | integration | integration | T1 → T2 |
| T3 | Curvas | unit | unit | Fim Phase 1 precede T3 |
| T4 | Deltas | unit | unit | T3 → T4 |
| T5 | Notebook 01 | integration | integration | Fim Phase 2 precede T5 |
| T6 | Notebook 02 | integration | integration | T5 → T6 |
| T7 | Notebook 09 | integration | integration | T6 → T7 |
| T8 | Documentação | none | none | T7 → T8 |

T3 também depende de T2 e T5 de T4 pelo encadeamento estrito entre fases. As três dependências entre fases não são paralelismo implícito.

## Handoff

**Settled mid-build:** Spec aprovada: user: "prossiga", 2026-11-09. Design A e tarefas aprovados: user: "aprovado", 2026-11-09.
**Dispatched:** none. Após último commit: registrar HEAD, disparar Verifier independente e atualizar retorno antes de qualquer novo dispatch.
**Walked:** ainda não executado. Walk de pesquisa é execução local de células/fixtures e inspeção das tabelas/figuras, sem UI web.
**Next:** executar T1 após gate --execute.
**Limits:** sem preflight remoto, checkpoints canônicos ou número do artigo reproduzido. Parquets afetados no incidente M01 permanecem inadmissíveis.

