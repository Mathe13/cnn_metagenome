# C00 — Preflight empírico e manifestos

## Problem

Quem executa os notebooks de reprodução ainda precisa descobrir entradas ausentes durante o processamento e distinguir manualmente saídas sintéticas de evidência empírica. Diretórios fixos permitem sobrescrever resultados e misturar execuções. O README orienta copiar `.env`, mas `setup_notebook_runtime` lê o ambiente do processo e não carrega esse arquivo.

Inspeção na base `4cbb0c27041be3c019b85b73bb4a4f88b201d419`, em 2026-11-03:

- `setup_notebook_runtime` já respeita `SMOKE_TEST=0`; sem modo explícito, o padrão remoto pode escolher smoke quando falta o diretório Species. A existência do diretório não comprova entradas completas.
- `short/06` usa taxonomia sintética quando falta seu arquivo, independentemente do modo; os notebooks usam caminhos fixos em `results_reproduction/`.
- `NotebookRunManifest` registra versões, caminhos e métricas, mas não identidade por execução, hashes, saídas nem qualidade da evidência. Alguns notebooks escrevem manifestos próprios.
- O manifesto versionado de `n02_benchmark` declara `is_smoke_test: true`. Os caminhos de checkpoints registrados nele não comprovam carregamento desses pesos.
- `tests/test_manuscript_compilation.py` e `tests/test_r08_reconciliation_and_tables.py` compilam em `paper/`, podendo regravar o PDF versionado. Esse PDF já está modificado na árvore de trabalho; a alteração preexistente deve ser preservada.
- Não foi encontrado arquivo de dependências versionadas. Versões registradas em saídas antigas não estabelecem o ambiente instalado ou o ambiente histórico.

Com C00, o pesquisador poderá inspecionar pré-requisitos por notebook antes de executar e obter um relatório `READY`/`BLOCKED`. Cada execução terá saídas próprias e evidência identificada. `READY` significará prontidão técnica dos requisitos inspecionados, sem certificar reprodução científica.

## Flow

Reutiliza `viral_cnn.notebook_runtime`, sua topologia e coleta de versões, `viral_cnn.models.registry` e a resolução de entradas de `viral_cnn.baselines.overlap`, evitando outro registro de modelos ou outra topologia.

1. Ambiente do processo e seleção de notebook → `viral_cnn.notebook_runtime` (exists): resolve raiz, modo e pré-requisitos específicos; registra caminhos com o mecanismo `LOG_PATH` existente.
2. Caminhos resolvidos → `viral_cnn.notebook_runtime` (exists): inspeciona existência, legibilidade e metadados disponíveis; emite relatório por requisito e decisão de prontidão, sem instanciar modelos ou percorrer sequências.
3. Contexto aprovado → notebooks em `notebooks/reproduction/` (exists): reservam saída exclusiva por execução (door 1) e consomem apenas entradas identificadas; artefatos sintéticos ficam dentro dessa execução.
4. Execução → `write_notebook_manifest` em `viral_cnn.notebook_runtime` (exists): persiste manifesto comum (door 2), incluindo configuração efetiva, entradas, checkpoints realmente usados, saídas e métricas observadas.
5. Manifesto → consumidor empírico no runtime (exists): rejeita modo sintético ou identidade obrigatória ausente antes de admitir a execução como candidata empírica. Completude científica de tabelas será tratada em C06.
6. Fontes do manuscrito → `scripts/compile_paper.py` e seus testes (exists): verificam compilação com saída temporária, preservando o PDF versionado.

## Impact

| Front | What changes |
| --- | --- |
| domain | `is_smoke_test`: os notebooks continuam distinguindo síntese e execução empírica; intenção empírica não será convertida em síntese por falta de entradas. O runtime e os ramos de carregamento dos notebooks consomem essa decisão. |
| domain | Novo `run_id`: identifica uma execução e seu namespace de saídas; a mesma cadeia pode apontar explicitamente para artefatos de execuções anteriores. |
| domain | Novo relatório de preflight: `READY` cobre apenas requisitos técnicos declarados; não comprova qualidade de pesos, independência genômica ou reprodução dos números. |
| stored data | Sem migração dos resultados existentes. Manifestos antigos permanecem históricos; não recebem hashes ou proveniência retroativa inventada. Novas saídas são isoladas. |
| scientific | Pode alterar admissibilidade, localização e rastreabilidade de CSVs, figuras, caches e checkpoints gerados; não muda arquitetura, codificação A/C/G/T, ruído, split ou definição de métrica. |
| environment | Acrescenta dependências diretas versionadas para o caminho CPU validado e instruções separadas para kernel/GPU remoto. Dask/cuDF/CUDA ausentes são registrados, sem versões presumidas. |
| tests | Smoke usa `REMOTE_SERVER=False`, fixtures temporárias e CPU. Compilação de LaTeX deixa de escrever em `paper/`. |

## Relations

- Uma execução possui um relatório de preflight e um manifesto final comum.
- Uma execução possui várias entradas e saídas; uma entrada pode ser saída de outra execução explicitamente identificada.
- Um checkpoint efetivamente carregado pertence a uma variante e nível taxonômico identificados pelo registro existente.
- Uma execução possui um diretório exclusivo; outra execução não pode assumir esse diretório (door 1).
- Evidência observada, inferida e desconhecida permanece distinguível no manifesto (door 2).

## Surface

None - não há API HTTP nem serviço externo. A interface local de inspeção e os notebooks são descritos em Flow e Observable; formato persistido em Landing.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1. Namespace persistido de execução | `results_reproduction/runs/<run_id>/<notebook_id>/`; reserva exclusiva do diretório do notebook, com falha se já existir; dependências de outra execução exigem caminho explícito | Reutilizar silenciosamente os diretórios fixos atuais: mistura caches, CSVs e pesos entre execuções. Mesmo `run_id` pode agrupar notebooks distintos, sem reexecutar sobre a mesma saída. |
| 2. Manifesto comum versionado | `schema_version: 1`, `run_id`, `notebook_id`, `is_smoke_test`, `git_commit`, `configuration`, `seed`, `library_versions`, `inputs`, `checkpoints`, `outputs`, `metrics_summary`, `evidence`; qualidade de evidência: `observed`, `inferred`, `unknown`; identidade de checkpoint carregado: `path` e `sha256` | Aceitar um caminho esperado como peso carregado ou admitir manifesto legado incompleto como empírico: não comprova a origem dos resultados. |

Nothing else in this change is hard to reverse. As duas portas deverão ser registradas como decisões de projeto ao implementar o plano aprovado.

## Criteria

### S1: Prontidão explícita antes de processar dados (P1)

**Acceptance Criteria**

1. WHEN o preflight é solicitado para um notebook THEN o sistema SHALL produzir relatório com caminho resolvido e estado de cada requisito obrigatório, retornando `READY` somente quando todos forem satisfeitos.
2. IF uma entrada obrigatória está ausente, ilegível ou tem metadados incompatíveis THEN o sistema SHALL retornar `BLOCKED` enumerando os requisitos que falharam antes de gerar métricas, tabelas ou checkpoints.
3. IF a execução empírica foi solicitada por `SMOKE_TEST=0` ou `smoke_test=False` THEN o sistema SHALL impedir geração ou substituição automática por entradas sintéticas.
4. WHEN o preflight inspeciona Parquet e checkpoints THEN o sistema SHALL limitar-se a caminhos e metadados sem iterar linhas, executar inferência, carregar pesos, baixar bases ou iniciar treinamento.
5. The preflight SHALL declarar pré-requisitos para os 15 notebooks atuais: short 01–11 e long 01–04, incluindo entradas preparadas por método/ruído no short/09 e resultados por variante exigidos pelo long/03.

**Independent test:** inspecionar fixture mínima válida e fixture com várias entradas ausentes; confirmar relatório completo e ausência de produtos científicos nos dois casos.

### S2: Saídas e proveniência separadas por execução (P1)

**Acceptance Criteria**

6. WHEN um notebook inicia uma execução THEN o sistema SHALL reservar exclusivamente seu diretório sob `runs/<run_id>/<notebook_id>/` antes de escrever produtos.
7. IF esse diretório já existe THEN o sistema SHALL recusar a execução sem alterar seus arquivos, inclusive sob duas tentativas concorrentes de reserva.
8. WHILE uma execução é sintética o sistema SHALL gravar fixtures somente dentro do namespace dessa execução ou de diretório temporário explicitamente fornecido, preservando os dados de entrada existentes.
9. WHEN uma execução termina THEN o sistema SHALL escrever manifesto de versão 1 com comando/notebook, commit e estado sujo do Git, configuração efetiva, semente, modo, versões instaladas, identidades de entradas e saídas e métricas efetivamente calculadas.
10. WHEN um checkpoint é efetivamente usado THEN o manifesto SHALL registrar variante, nível taxonômico, caminho e SHA-256 do arquivo carregado.
11. WHEN contagens de dados ou identidade por hash não foram medidas THEN o manifesto SHALL registrar seu estado `unknown`, distinguindo-as de valores `observed` e `inferred`.
12. IF um manifesto é sintético, legado sem identidade obrigatória ou incompleto THEN o portão empírico SHALL rejeitá-lo com os campos que motivaram a rejeição.
13. WHEN dependências entre notebooks são resolvidas THEN o sistema SHALL registrar os caminhos e manifestos de origem selecionados sem buscar implicitamente resultados de outra execução.
14. IF uma execução falha depois da reserva THEN o sistema SHALL registrar o estado de falha sem apresentar o manifesto parcial como execução concluída.

**Independent test:** executar uma cadeia sintética pequena em diretório temporário; tentar repetir o identificador e submeter seu manifesto ao portão empírico; confirmar rejeições e preservação dos primeiros arquivos.

### S3: Ambiente verificável e execução local sem alterar o artigo (P1)

**Acceptance Criteria**

15. WHEN o ambiente é inspecionado THEN o sistema SHALL registrar Python, PyTorch, CUDA, NumPy, pandas, pyarrow, Dask e cuDF com versão instalada ou estado explícito de indisponibilidade.
16. The documentation SHALL fornecer comandos para instalar o ambiente CPU versionado, registrar o kernel Jupyter e exportar as variáveis antes de iniciar o servidor, esclarecendo que copiar `.env` não altera um kernel existente.
17. WHEN os testes de compilação existentes são executados THEN o sistema SHALL gerar PDF e auxiliares em diretório temporário sem alterar o conteúdo de `paper/manuscript_submission.pdf`.
18. WHEN a verificação local de C00 é executada THEN o relatório SHALL distinguir prova sintética, inspeção de existência dos dados reais e execução empírica, deixando esta última pendente se o armazenamento remoto não estiver acessível.

**Independent test:** rodar a inspeção de ambiente e um smoke offline, comparar hashes do PDF antes/depois da compilação isolada e inspecionar a raiz real configurada sem processar o conjunto.

## Out of scope

| Excluded | Why |
| --- | --- |
| Corrigir treino da sonda, caches de embeddings e protocolo KNN | C01; C00 somente impede entrada sintética silenciosa e registra origem. |
| Demonstrar mecanismos causais nos contigs e completar Tabela 12 | C02; preflight exige entradas declaradas, sem validar as conclusões científicas. |
| Executar Kraken2/BLASTN ou produzir consultas e predições ausentes | C03; C00 lista essas entradas e bloqueia sua ausência. |
| ICs, calibração completa, retreinamento e alteração dos números do artigo | C04–C06; C00 não constitui reprodução dos resultados. |
| Reconstruir accessions ou certificar independência genômica | Não há proveniência por fragmento demonstrada; taxonomia não resolve essa lacuna. |
| Provisionar GPU, instalar CUDA no servidor ou acessar JupyterHub sem conexão disponível | Exige ambiente remoto real; entregar instruções verificáveis e registrar a limitação. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Perfil da skill | `light`, padrão declarado pela skill; casos negativos do roadmap continuam obrigatórios | O repositório não declara perfil específico; não importar o perfil de features históricas | n |
| Seleção inicial | Implementar somente C00 | Primeira dependência bloqueadora do roadmap | n |
| Compatibilidade histórica | Preservar leitura histórica; proibir promoção automática de manifestos legados a empíricos | Permite consulta sem fabricar proveniência | n |
| Limite de hashes no preflight | Não calcular hash integral de checkpoints ou Parquets volumosos durante inspeção; identidade completa dos pesos é medida ao uso | Preflight deve ser leve e não confundir nome de arquivo com identidade comprovada | n |
| Dependências remotas | Usar ambiente CPU local medido para a instalação versionada; registrar o ambiente GPU separadamente quando acessível | Não há evidência para fixar versões históricas CUDA/Dask/cuDF | n |
| Modo sem flags | Remoto continua padrão; ausência de entradas retorna bloqueio para o notebook selecionado; síntese requer modo local ou smoke explícito | Compatibiliza AD-017 com a regra empírica do roadmap; atualiza a descrição do fallback ao implementar | n |

**Open questions:** none - decisões de implementação têm os defaults acima; acesso aos dados remotos condiciona a prova empírica, sem impedir a implementação local.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Inspeção local | Sucesso e ausência/ilegibilidade de entradas | AC 1–2; estados `READY` e `BLOCKED` |
| Inspeção local | Trabalho pesado e dependência de rede | AC 4; inspeção sem processamento de linhas ou downloads |
| Notebooks | Estado vazio e origem das entradas | AC 2–3 e 5; lista de requisitos antes dos produtos |
| Notebooks | Reexecução e concorrência | AC 6–7; reserva exclusiva, sem sobrescrita |
| Notebooks | Fixtures, saídas e dependências de outra execução | AC 8 e 13 |
| Manifesto | Sucesso, falha parcial e evidência ausente | AC 9–14 |
| Console e log | Caminhos de entrada carregados | existing - `log_artifact_path`, contrato `LOG_PATH` do AGENTS.md |
| Ambiente/kernel | Configuração efetiva e versões | AC 15–16 |
| Testes de compilação | Preservação dos PDFs existentes | AC 17 |
| Relatório de verificação | Limite entre inspeção, síntese e execução real | AC 18 |
| Todas as superfícies locais | Autorização, rate limiting, paginação e compatibilidade HTTP | n/a - não há serviço HTTP ou credenciais novas |
| Interface visual | Loading, layout e navegação | n/a - sem tela ou fluxo visual novo |

As nove dimensões foram consideradas: validação (AC 1–5), falha parcial (AC 14), repetição (AC 7), autorização (n/a — execução local), concorrência (AC 7), ciclo de dados (AC 6–13), dependência externa (AC 2 e 15), transições (AC 1–2 e 14) e observabilidade (AC 9–11 e 15). A associação a provas será feita em `checks.md` depois da revisão deste plano.

## Sources

- `ROADMAP_CORRECOES_REPRODUCAO.md`, seção C00 e regras comuns — escopo, manifesto mínimo, portões local e remoto.
- `AGENTS.md`, `src/viral_cnn/AGENTS.md` e `.specs/STATE.md` — contratos científicos, topologia, logging, preservação de artefatos e limites da evidência.
- Specs N01, N02, N07 e R08; runtime, configs, notebooks e testes citados em Problem — comportamento existente e distinção entre verificação sintética histórica e execução empírica pendente.
