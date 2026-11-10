# M02 Benchmark, Intervals and Calibration Specification

**Approved**: 2026-11-09 - user: "prossiga"

## Problem Statement

O notebook 09 sobrescreve DATA_ROOT, SMOKE_TEST e C03_PREPARED_DIR com uma execução histórica. Sua figura omite duas variantes e não apresenta cobertura nem intervalos pareados. Os notebooks 01/02 já implementam C04, mas o GradCAM de 02 usa entradas aleatórias também no modo empírico.

M02 prepara 01/02/09 para a próxima rodada identificada, preservando C03/C04 e os resultados históricos. Não executa inferência pesada nem reproduz números do artigo nesta máquina.

## Goals

- [ ] Validar as 35 condições C03 antes de calcular resultados de 09.
- [ ] Integrar 01/02/09 aos destinos e dependências explícitos M00.
- [ ] Preservar ICs e calibração C04 com evidências de seleção na validação.
- [ ] Gerar curvas de acurácia/cobertura e diferenças pareadas rastreáveis.
- [ ] Impedir que GradCAM aleatório seja produzido como análise empírica.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Executar CNN, Kraken2 ou BLASTN para produzir as condições de 09 | São pré-requisitos C03; 09 consome resultados completos. |
| Treinamento, novos checkpoints e múltiplas realizações de ruído | Alteram orçamento e experimento; não fazem parte de M02. |
| Corrigir manuscrito ou promover figuras para identificadores F | Reconciliação posterior M08/C06. |
| Refazer curadoria M01, embeddings ou motivos | Outras etapas do roadmap. |
| Restaurar Parquets locais sobrescritos na M01 | Conteúdo anterior não demonstrável; selecionar fontes canônicas no servidor. |
| Interface web, serviços externos e instalação do skill no projeto | Trabalho local de reprodução; skill solicitado é lido no caminho encontrado. |

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Escopo | M02 integral do roadmap de 09/11/2026 | Pedido explícito de implementar M02; inclui as duas figuras propostas. | y |
| GradCAM | Executar a demonstração aleatória somente em smoke | Alternativa expressamente permitida no roadmap; dispensa introduzir seleção biológica nova. | y |
| Protocolos estatísticos | Preservar C03/C04 existentes | AD-021/022 e contratos comuns; Wilson individual, bootstrap estratificado de fragmentos C04 e bootstrap pareado por espécie C03. | y |
| Artefatos remotos | Verificação local sintética e verificação empírica separadas | Manifesto/STATE registram ausência dos artefatos canônicos do cluster. | y |
| Protocolo do skill | tlc-spec-driven v4.1.0 encontrado em outro projeto local | Não há instalação desse nome aqui; o repositório contém tlc-spec-lean, que não substitui silenciosamente o pedido. | y |

**Open questions:** none. Especificação aprovada pelo usuário em 09/11/2026 com “prossiga”. Tarefas e design aprovados em 09/11/2026 com “aprovado”.

## User Stories

### P1: C03 pronto e pareado antes do consumo

**User Story**: Como pesquisador, quero detectar condições ausentes ou incompatíveis antes de avaliar o benchmark.

**Acceptance Criteria**:

1. WHEN 09 inicia THEN o sistema SHALL preservar os valores DATA_ROOT e SMOKE_TEST fornecidos pelo ambiente.
2. WHEN C03_PREPARED_DIR ou a dependência C03 do plano M00 identifica a entrada THEN o sistema SHALL registrar o caminho resolvido e SHA-256 dos manifestos consumidos.
3. IF ambiente e plano selecionam fontes C03 diferentes THEN o sistema SHALL interromper antes de publicar métricas.
4. IF preparação, construção das bases ou qualquer uma das 35 condições de sete métodos por cinco taxas estiver ausente, incompleta ou com hash inválido THEN o sistema SHALL bloquear o preflight de 09 e impedir a avaliação empírica.
5. IF query_id, sequência materializada ou rótulo diferirem entre métodos na mesma taxa THEN o sistema SHALL rejeitar o benchmark.
6. IF ordem de classes, referência ou política de abstenção diferir do contrato C03 THEN o sistema SHALL rejeitar o benchmark.
7. IF uma dependência C03 for sintética no modo empírico THEN o sistema SHALL rejeitar seu consumo.
8. The sistema SHALL consumir os resultados de inferência C03 em 09 sem executar CNN, Kraken2 ou BLASTN.

**Independent Test**: Montar uma fixture temporária completa das 35 condições, aceitá-la estruturalmente e rejeitar mutações de manifesto, hash, sequência, rótulo, classe, abstenção e modo.

### P1: Execução isolada e evidência estatística preservada

**User Story**: Como pesquisador, quero rerodar 01/02/09 sem sobrescrever resultados e identificar o significado dos intervalos.

**Acceptance Criteria**:

1. WHEN 01, 02 ou 09 inicia uma execução THEN o sistema SHALL reservar um destino exclusivo em results_reproduction/runs com run_id e parent_run_id do plano quando presente.
2. IF o destino já estiver ocupado ou uma entrada empírica obrigatória faltar THEN o sistema SHALL interromper sem sobrescrever a fonte ou publicar COMPLETE.
3. WHEN a execução termina THEN o sistema SHALL publicar manifesto com protocolo, modo, população/split, semente, Git commit/dirty, configuração, identidades de entradas/checkpoints e hashes dos produtos.
4. IF a execução falhar após reservar o destino THEN o sistema SHALL registrar FAILED nesse namespace.
5. WHEN 01 publica ICs C04 THEN o sistema SHALL identificar bootstrap de fragmentos estratificado por táxon com suporte fixo, confiança 95%, semente 42 e número de repetições.
6. WHEN 02 ajusta temperatura THEN o sistema SHALL usar somente validação limpa identificada e selar o parâmetro antes da avaliação de teste.
7. WHEN 02 publica diagramas de confiabilidade THEN o sistema SHALL exibir N e preservar a igualdade de suporte antes/depois com ECE, Brier e NLL vinculados às entradas.
8. WHILE 02 executa em modo empírico o sistema SHALL omitir a demonstração GradCAM com torch.randint e registrar essa omissão.
9. WHEN 01/02/09 descrevem os intervalos THEN o sistema SHALL declarar uma realização de ruído por taxa e ausência de estimativa de variabilidade entre treinamentos.

**Independent Test**: Executar os caminhos de notebook em fixtures isoladas, observar duas reservas distintas, colisão/falha e ausência de GradCAM aleatório empírico; testar rejeição de temperatura ajustada em teste ou com identidade sobreposta.

### P1: Curvas e diferenças com denominador explícito

**User Story**: Como pesquisador, quero comparar sete métodos por ruído e estrato sem confundir cobertura com acurácia.

**Acceptance Criteria**:

1. WHEN 09 agrega resultados THEN o sistema SHALL exportar acurácia global, cobertura e acurácia condicional separadas para os sete métodos, cinco taxas e três estratos all/overlap/no_exact_overlap.
2. WHEN um método abstém com label -1 THEN o sistema SHALL contar a query como erro na acurácia global e mantê-la no denominador N.
3. WHEN 09 gera curvas THEN o sistema SHALL mostrar acurácia e cobertura dos sete métodos nos três estratos com N e ICs individuais identificados como Wilson quando disponíveis.
4. WHEN 09 gera diferenças pareadas THEN o sistema SHALL exportar e representar os dez pares entre cinco CNNs e BLASTN/Kraken2 nas cinco taxas e três estratos em pontos percentuais com bootstrap por espécie, 2.000 repetições e semente 42.
5. IF um estrato estiver vazio ou contiver menos de duas espécies para bootstrap THEN o sistema SHALL declarar o IC pareado indisponível com motivo sem inventar limites.
6. WHEN figuras são geradas THEN o sistema SHALL salvar CSV de origem e PNG/PDF derivados com vínculos e hashes no manifesto da execução.

**Independent Test**: Usar previsões literais incluindo abstenção, estrato vazio e espécie única; verificar N, cobertura, acurácia condicional, sinal/unidade dos deltas e regeneração dos painéis dos CSVs.

## Edge Cases

- IF um manifesto declarar COMPLETE com produto divergente THEN o sistema SHALL rejeitar a entrada pelo hash.
- IF uma reserva for reutilizada THEN o sistema SHALL preservar todos os bytes existentes.
- IF faltar dado empírico THEN o sistema SHALL falhar sem criar fixture no diretório da fonte.

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| M02-01 | C03, critérios 1–3: configuração e fonte explícita | Specify | Pending |
| M02-02 | C03, critérios 4–7: 35 condições e identidade | Specify | Pending |
| M02-03 | C03, critério 8: consumo sem inferência | Specify | Pending |
| M02-04 | Execução, critérios 1–4: isolamento e manifestos | Specify | Pending |
| M02-05 | Execução, critérios 5–7 e 9: IC/calibração | Specify | Pending |
| M02-06 | Execução, critério 8: GradCAM smoke | Specify | Pending |
| M02-07 | Curvas, critérios 1–3: métricas e cobertura | Specify | Pending |
| M02-08 | Curvas, critérios 4–5: diferenças pareadas | Specify | Pending |
| M02-09 | Curvas, critério 6: produtos rastreáveis | Specify | Pending |

**Coverage:** 9 requisitos; decomposição em tarefas e um check por critério após aprovação.

## Implicit Requirements Sweep

| Dimension | Resolution |
| --- | --- |
| Input validation & bounds | C03 critérios 4–7; Curvas critério 5. |
| Failure / partial failure | Execução critérios 2 e 4. |
| Idempotency / retry | Nova reserva por tentativa; destinos ocupados falham. |
| Auth boundaries & rate limits | N/A: arquivos locais autorizados; sem serviço remoto. |
| Concurrency / ordering | Reserva exclusiva M00; validação C03 precede métricas; temperatura selada antes de teste. |
| Data lifecycle / expiry | Fontes somente leitura e produtos em novo namespace; sem exclusão/expiração. |
| Observability | LOG_PATH padrão ligado; caminhos/hashes no preflight e manifesto. |
| External-dependency failure | Dependência ausente/incompleta bloqueia; nenhum download automático. |
| State-transition integrity | COMPLETE apenas com produtos identificados; falha registra FAILED. |

## Differences from the source

Nenhuma alteração metodológica ao roadmap. Novas figuras permanecem produtos M02 sem promoção automática ao artigo. Não há captura visual vinculante.

## Sources

- Pedido: “usando tlc-spec-driven vamos implementar m02”.
- ROADMAP_MELHORIAS_REPRODUCTION_2026-11-09.md, contratos comuns e seção M02.
- AGENTS.md e src/viral_cnn/AGENTS.md, invariantes científicos e contratos de código.
- .specs/STATE.md, AD-021/022/028/029 e incidente M01.
- docs/dataset_manifest.md; configs/baseline_c03.yaml; configs/models.yaml; configs/reproduction_m00.yaml.
- notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb, 02_analyze_confidence_and_calibration.ipynb e 09_benchmark_cnns_kraken2_blast_overlap.ipynb, inspecionados sem execução.
- src/viral_cnn/baselines/overlap.py, preparation.py e paired.py: validação C03, Wilson e bootstrap por espécie já existentes.
- src/viral_cnn/eval/uncertainty.py e calibration.py: bootstrap C04 e TemperatureScaler com proveniência e selo.
- tests/test_c03_baselines.py e tests/test_c04_uncertainty.py: contratos existentes a preservar.
- /home/matheus/Documents/mestrado/aulas/RL/seminario/mcts-hnm/.agents/skills/tlc-spec-driven/SKILL.md, versão 4.1.0: aprovação de spec e tarefas antes de execução.

## Evidence Chain and Impact

Fragmentos canônicos e CSV Species/Genus → split/curadoria M01 → A/C/G/T 512 bp → checkpoints do registro → inferência C04 de 01/02 ou condições C03 já calculadas → métricas/IC/calibração → novos CSVs/figuras M02. C03 e C04 são populações distintas; não se fundem seus denominadores. Identidade genômica permanece unknown sem accession por fragmento.

Mudanças previstas: notebooks de reprodução 01/02/09, preflight/runtime/plano somente onde necessário, módulos eval/baselines, config C03/M02, testes e documentação. Históricos short/ e long/, checkpoints e saídas anteriores ficam preservados. Novos CSVs e figuras podem mudar os produtos apresentados no benchmark; pesos, arquitetura, codificação, split, taxonomia e estimandos publicados permanecem identificados.

O carregador C03 já valida extensamente manifestos e hashes; a implementação deve reutilizá-lo. 09 atualmente usa o alias resnet50_species enquanto a configuração declara resnet_clean_species: a compatibilidade deve ser explícita, sem substituir checkpoint por dimensão de cabeça. A restauração dos arquivos canônicos não foi demonstrada. Nenhum resultado empírico pode ser certificado nesta etapa local.

## Success Criteria

- [ ] Um check executável por critério e revisão independente após implementação.
- [ ] Testes locais offline forçam REMOTE_SERVER=False e usam diretórios temporários sem tocar nos Parquets da raiz.
- [ ] Comandos exatos, arquivos alterados, artefatos ausentes e divergências ficam registrados em verification.md.
- [ ] Preflight remoto e reprodução integral permanecem explicitamente pendentes até execução no servidor.
