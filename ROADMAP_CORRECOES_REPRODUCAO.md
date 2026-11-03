# Roadmap de correções para os notebooks de reprodução

**Base de inspeção:** `4cbb0c27041be3c019b85b73bb4a4f88b201d419` (2026-11-03).  
**Objetivo:** obter evidência rastreável para responder às notas de revisão do artigo, distinguindo execução sintética, execução empírica e limitação que os dados não permitem resolver.  
**Uso:** criar uma feature independente por etapa em `.specs/features/`, seguindo o fluxo do `tlc-spec-driven` (`PLAN → CHECKS → BUILD → VERIFY`). Antes de começar, ler `AGENTS.md`, `.specs/STATE.md` e as specs anteriores relacionadas. Os IDs abaixo são novos e não substituem as specs `S`, `N` ou `R` já concluídas.

## Ponto de partida e regras para todas as specs

- Preservar os notebooks históricos `short/` e `long/`, checkpoints originais, partições Parquet e resultados publicados. Usar saídas em uma pasta exclusiva por execução; nunca tomar um artefato sintético ou um número copiado do manuscrito como evidência empírica.
- Todo caminho empírico deve declarar entradas obrigatórias e falhar claramente quando elas não existirem. Desabilitar fallback automático para `SMOKE_TEST` quando a execução for explicitamente solicitada com `SMOKE_TEST=0`.
- Manifesto mínimo por execução: commit, notebook/comando, modo sintético ou empírico, configuração, semente, dados/split e contagens, versões de bibliotecas, checkpoint e hash, arquivos de saída e métricas. Distinguir campos **observados**, **inferidos** e **desconhecidos**.
- Para cada spec, escrever `plan.md` com escopo, dependências e impacto científico; `checks.md` com provas observáveis e caso negativo; implementar somente depois; escrever `verification.md` com comandos executados, resultados e limitações. O portão final inclui um teste sintético pequeno **e**, quando disponível, uma validação explícita com arquivos reais do JupyterHub. Teste sintético verde não conclui a etapa empírica.
- Verificar a execução a partir da raiz do repositório no kernel correto. Conferir `REMOTE_SERVER=true`, `SMOKE_TEST=0`, `DATA_ROOT=/home/mlima/workspace` (ou o caminho real), e conferir no manifesto que `is_smoke_test` é `false`.
- Não usar a ausência de `genome_id` como motivo para inventar accessions, reconstruir genomas a partir do de-para taxonômico ou afirmar generalização a genomas inéditos. O resultado atual da auditoria exige declarar essa limitação.

## Sequência das specs

| Ordem | Feature sugerida | Prioridade | Depende de | Entrega verificável |
| --- | --- | --- | --- | --- |
| 0 | `c00-preflight-empirico-e-manifestos` | Bloqueadora | — | Ambiente e entradas verificados; saídas isoladas; nenhum resultado sintético aceito como empírico. |
| 1 | `c01-sonda-latente-sem-vazamento` | Bloqueadora | C00 | Notebook 06 usa treino/validação para ajustar a sonda; teste permanece intocado. |
| 2 | `c02-auditoria-empirica-sequencias-longas` | Bloqueadora | C00 | Auditoria com suporte real; Tabela 12 completa; diagnósticos apresentados como hipóteses ou demonstrados. |
| 3 | `c03-baselines-pareados-artefatos-reais` | Alta | C00 | Notebook 09 consome predições reais alinhadas e comprova cobertura, denominadores e proveniência. |
| 4 | `c04-incerteza-calibracao-e-figuras` | Alta | C00, C03 para comparações pareadas | ICs das métricas principais; diagramas de confiabilidade dos modelos e níveis de ruído. |
| 5 | `c05-finetune-e-comparacao-controlada` | Condicional | C00 | Fine-tuning auditável; comparação causal somente se treinamento pareado real for executado. |
| 6 | `c06-regeneracao-e-reconciliacao-estrita` | Bloqueadora para submissão | C01–C04; C05 se executada | Tabelas/figuras reais ligadas aos manifestos, discrepâncias explicadas e afirmações revisadas. |

### C00 — Preflight empírico, ambiente e isolamento das saídas

**Superfícies:** `src/viral_cnn/notebook_runtime.py`, `README.md`, `env.example`, notebooks em `notebooks/reproduction/`, scripts auxiliares. O README manda copiar `.env`, mas o runtime lê variáveis do processo; não presumir que o kernel carregou esse arquivo. Não há arquivo único de dependências versionadas.

**PLAN:** definir topologia dos dados/checkpoints, versão de Python e pacotes, ordem de execução e diretório de saída por `run_id`. Registrar os pré-requisitos específicos de cada notebook, incluindo as predições preparadas para o 09 e resultados de inferência para `long/03`.

**CHECKS:** `SMOKE_TEST=0` com dados ausentes falha antes de gerar tabela; manifesto sintético não passa pelo portão empírico; execução posterior não sobrescreve CSV/PDF/checkpoint de execução anterior; um preflight encontra ou lista precisamente entradas faltantes. O teste de compilação do manuscrito atualmente regrava um PDF versionado: isolar sua saída.

**BUILD:** configurar dependências reproduzíveis e kernel; implementar preflight e manifesto único; separar resultados por execução; documentar como exportar as variáveis para o JupyterHub. Não baixar bases nem iniciar treinamento nessa etapa.

**VERIFY:** relatório com caminhos reais, versões, status de cada entrada e decisão `READY`/`BLOCKED`; incluir uma execução sintética e uma verificação de existência em `DATA_ROOT` sem processar milhões de fragmentos.

### C01 — Corrigir o vazamento na sonda linear e auditar o KNN

**Superfícies:** `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb`, `src/viral_cnn/eval/latent.py`, testes de espaço latente, caches `features_*.parquet`.

**Problema observado:** o notebook extrai embeddings do **teste** e usa os primeiros 80% para `LinearProbeClassifier.fit(..., split_origin="train")`. O rótulo passado ao método não altera a origem dos dados. O KNN atual separa índices, não genomas; cache sem identidade de entrada/checkpoint pode reaproveitar embeddings obsoletos.

**Aceitação:** extrair embeddings de treino e validação para ajustar/selecionar a sonda, congelar seus parâmetros e avaliar uma vez no teste; registrar contagens e hashes dos splits e checkpoints. Teste negativo deve detectar qualquer ajuste sobre embeddings do teste. Manter KNN por fragmento com referência/consulta disjuntas e sem auto-vizinhança, registrando número de classes válidas; oferecer variante por genoma **somente** se IDs por fragmento existirem. Invalidar cache quando mudar entrada, split, checkpoint ou código. Caso contrário, declarar que a análise não testa independência genômica.

### C02 — Substituir diagnóstico pré-escrito das sequências longas por auditoria

**Superfícies:** `notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb`, `02_cnn_long_voter_inference.ipynb`, `03_cnn_long_voter_compare.ipynb`, `src/viral_cnn/inference/audit_long.py` e `src/viral_cnn/baselines/long_eval.py`.

**Problemas observados:** `long/01` amostra 50 contigs e 200 fragmentos de treino para o overlap de k-mers e escreve a disparidade 500 vs. 50 como linha fixa; `long/03` grava explicações de *logit pulling*, viés de amostragem e calibração como `VERIFIED_AND_DOCUMENTED` sem testes correspondentes. A Tabela 12 aceita pelo menos um arquivo de modelo, mesmo que faltem as demais variantes. Conferir a compatibilidade entre o nome da saída do `long/02` e os nomes esperados pelo `long/03`.

**Aceitação:** processar o universo elegível ou reportar claramente uma amostra planejada, com denominador e critérios; contar comprimentos e pesos de amostragem observados; identificar cinco modelos esperados e exigir arquivo, checkpoint, rótulo e `query_id` alinhados para cada um. Comparar voto majoritário e média dos logits pré-softmax, distribuição dos logits, erros por comprimento/ambiguidade/táxon e mapeamento de classes. Cada mecanismo só pode receber status `SUPPORTED` mediante teste e efeito observável; caso contrário, `HYPOTHESIS` ou `INCONCLUSIVE`. Se o genoma de origem for desconhecido, não preencher `seen_genome` por hipótese. Não chamar contigs de leituras experimentais de ONT.

### C03 — Fechar o benchmark real de Kraken2/BLASTN

**Superfícies:** notebook `short/09_benchmark_cnns_kraken2_blast_overlap.ipynb`, `src/viral_cnn/baselines/overlap.py`, `paired.py`, script de preparação, documentação de entradas.

**Problema observado:** fora do modo sintético, o 09 exige `exact_overlap.csv`/`truth.parquet` e predições já preparadas para CNNs, Kraken2 e BLASTN em cada nível de ruído; ele não executa Kraken2/BLASTN. O roadmap deve decidir se esses arquivos reais já existem ou se uma etapa de geração reproduzível precisa ser implementada.

**Aceitação:** manifesto por método contendo versão/base de referência, taxonomia, corrupção, consulta, rótulo e regra de abstenção; chaves `query_id` únicas e alinhadas nas comparações; `all = overlap + no_exact_overlap`; ausência de previsão contada como erro na acurácia global, com cobertura e acurácia condicional separadas. Não publicar ICs de diferenças pareadas gerados por fixtures de 40 consultas. Se a base da ferramenta incluir o genoma da consulta, registrar esse cenário e não inferir generalização.

### C04 — Intervalos de confiança e calibração com saídas publicáveis

**Superfícies:** `short/01_benchmark_clean_and_noise.ipynb`, `short/02_analyze_confidence_and_calibration.ipynb`, `src/viral_cnn/eval/`, `src/viral_cnn/reporting/`.

**Problema observado:** o 09 tem ICs para diferenças selecionadas; o 01 ainda não exporta ICs para todas as acurácias e Macro/Weighted-F1 do artigo. O 02 calcula ECE, Brier, NLL e temperatura, mas não chama a exportação dos diagramas de confiabilidade.

**Aceitação:** guardar predições/rótulos ou contagens suficientes para reamostragem auditável sem matriz densa de probabilidades `(N, C)`; gerar IC de 95% para as métricas e estratos que aparecerão no texto, com unidade de reamostragem, semente, repetições e denominador explícitos. Reamostragem por espécie pode medir incerteza condicional à composição observada; **não** a variabilidade de treinamento entre sementes. Exportar CSV e figuras de confiabilidade para cinco modelos × cinco níveis de ruído; ajustar temperatura apenas com validação. Relatar calibração antes/depois, suporte e eventuais falhas de carregamento de checkpoint.

### C05 — Fine-tuning e comparação controlada: decisão condicionada ao prazo

**Superfícies:** `short/03_finetune_resnet50_species.ipynb`, `short/04_finetune_convnext_species.ipynb`, `scripts/run_controlled_training.py`, `configs/controlled/baseline_pair.yaml`.

**Situação:** o notebook 04 contém somente 3 das 5 células de código executadas no arquivo versionado; isso não prova que esteja quebrado, mas não demonstra fine-tuning completo. O script controlado usa fixtures sintéticas quando não acha o diretório esperado e lança `NotImplementedError` no caminho de Parquet real. Os fine-tunings históricos têm configurações e inicializações diferentes das de uma comparação causal.

**Trilha mínima:** validar 03/04 no JupyterHub sem sobrescrever checkpoints históricos, conferir o checkpoint de origem, épocas, seleção, precisão e métricas; manter conclusões como comparação **entre configurações avaliadas** e declarar execução única por configuração.

**Trilha opcional, se houver tempo/GPU:** criar loaders reais no script controlado; executar ResNet50/ConvNeXt em condições pareadas e múltiplas sementes, registrando orçamento, parâmetros, logs e comparação no mesmo teste. Testes com fixtures comprovam apenas a mecânica. Só depois dessa execução discutir um possível efeito da arquitetura; se não ocorrer, encerrar a spec com conclusão causal explicitamente indisponível.

### C06 — Regeneração estrita, matriz de evidências e atualização do artigo

**Superfícies:** `src/viral_cnn/reporting/tables.py`, `figures.py`, `scripts/generate_paper_tables.py`, `generate_paper_figures.py`, `paper/`, `README.md`; criar o notebook mestre `notebooks/reproduction/00_regenerate_all_paper_artifacts.ipynb` previsto no roadmap anterior, se ainda for a interface desejada.

**Problema observado:** o relatório de reconciliação versionado cita como `DYNAMIC_ARTIFACT_LOADED` arquivos das Tabelas 5, 6 e 8–12 que não estão presentes no clone. Os CSVs versionados do benchmark e da calibração têm manifestos `is_smoke_test: true`. A geração não pode passar no modo empírico com tabela de fallback, arquivo inexistente, manifesto sintético ou conjunto incompleto de modelos.

**Aceitação:** para cada Tabela 1–12 e Figura 1–5, produzir uma linha `afirmação no artigo → entrada/manifesto → transformação → saída → N → status`. `strict_empirical=True` exige entradas reais, gera valores a partir delas e falha para qualquer lacuna; referências históricas são exibidas em coluna separada, nunca usadas para completar valores ausentes. Exigir nomes de figuras corretos em Linux e painel (e) ligado ao checkpoint ConvNeXt_F. Publicar divergências em pontos percentuais e sua explicação. Atualizar Methods, Results, Discussion e limitações conforme a evidência obtida; se proveniência genômica ou causa da queda em contigs seguirem desconhecidas, reduzir as afirmações no resumo e na conclusão. Referências, afiliações e redação editorial devem ser revisadas no manuscrito, não consideradas resolvidas pela execução dos notebooks.

## Portões de execução e definição de pronto

1. **Portão local:** contrato científico e caso negativo passam em testes pequenos; notebooks sintéticos executam sem produzir artefatos com status empírico.
2. **Portão remoto:** preflight C00 passa; cada notebook relevante executa de ponta a ponta no JupyterHub com `SMOKE_TEST=0`; entradas e saídas têm manifesto e contagens; checkpoints e taxonomia são os corretos.
3. **Portão científico:** C01 e C02 não contêm vazamento ou mecanismo causal presumido; C03 tem comparações alinhadas; C04 tem ICs/calibração; C06 rejeita dados sintéticos ou incompletos. C05 opcional decide apenas o nível de afirmação causal.
4. **Portão de escrita:** cada número relevante da versão de submissão vem de uma saída empírica ou é claramente rotulado como resultado histórico ainda não reproduzido; limitações de split por fragmento, referência genômica e dados ausentes estão explícitas.

### Prompt inicial para cada feature

> Use `tlc-spec-driven` e leia `AGENTS.md`, `.specs/STATE.md` e `ROADMAP_CORRECOES_REPRODUCAO.md`. Implemente somente a feature **CXX**. Primeiro investigue o código atual e os artefatos relacionados e escreva `plan.md` e `checks.md` em `.specs/features/<slug>/`. Preserve os notebooks históricos e as saídas de experimentos anteriores. Inclua pelo menos um caso negativo que impeça transformar dados sintéticos, dados do teste usados no treino, artefatos incompletos ou afirmações sem teste em evidência empírica. Execute verificações locais sem treinamento pesado e, quando houver acesso ao JupyterHub, registre separadamente a execução real e seus manifestos em `verification.md`. Se um dado de origem não existir, deixe a limitação explícita; não o infira do mapeamento taxonômico.

**Não é condição para começar C01–C04:** recuperar accessions ausentes ou iniciar retreinamento. **Não é condição para concluir C06:** ter experimento pareado C05, desde que o artigo limite suas conclusões às configurações efetivamente avaliadas.
