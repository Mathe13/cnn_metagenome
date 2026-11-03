# C01 — Sonda linear sem vazamento e auditoria do KNN

## Problem

O notebook `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb` declara uma sonda independente em treino/validação, mas extrai apenas embeddings do teste e usa os primeiros 80% para `LinearProbeClassifier.fit(..., split_origin="train")`. Alterar essa string não altera a origem dos exemplos. O resultado não mede uma sonda ajustada independentemente do teste.

Evidências da inspeção de 2026-11-03, sobre HEAD `4cbb0c27041be3c019b85b73bb4a4f88b201d419` e alterações locais existentes:

- Célula 3: apenas `test_loader` alimenta `extract_latent_embeddings`. Célula 6: `models_to_evaluate[:2]` seleciona somente as duas ResNet50; não há auditoria de sonda para ConvNeXt. Não há seleção por validação.
- `LinearProbeClassifier.fit` rejeita apenas `split_origin == "test"`; aceita arrays do teste renomeados como treino. O teste histórico C22 verifica essa string; a prova de N04/C14 verifica presença de manifesto, sem comprovar origem dos exemplos.
- O cache `features_<modelo>.parquet` é aceito pela existência do arquivo. Não identifica split, dados de origem, pesos ou código de extração; treino e teste não podem compartilhar esse nome com segurança.
- `evaluate_cosine_knn` separa índices por classe com semente 42 e normaliza em L2, mas retorna somente uma flag de ausência de auto-vizinhança. Classes unitárias entram somente na referência; quando todas são unitárias, há fallback não estratificado. `k` é reduzido silenciosamente quando a referência tem menos de cinco exemplos.
- `KNN_Top5` significa presença do rótulo verdadeiro entre os cinco vizinhos, conforme R04; não é Top-5 de logits. `Valid_Class_Rate` na Tabela 5 recebe `retained_sample_rate`, uma fração de amostras, não de classes.
- A introdução do notebook ainda diz ConvNeXt 768-d, embora o código atual e o manuscrito descrevam o modelo híbrido com entrada de `fc` de 2048 dimensões. O texto do artigo também chama os Silhouettes de euclidianos, enquanto o método/código usam cosseno; a reconciliação editorial pertence a C06.
- Existem três Parquets locais de 16 linhas cada, compatíveis com fixtures. O manifesto local de espaço latente declara `is_smoke_test: true`. Os cinco checkpoints canônicos Species não existem nos caminhos locais inspecionados e `/home/mlima/workspace` está ausente. Esses arquivos não demonstram disponibilidade do experimento empírico.
- A verificação de C00 registra integração parcial: APIs de reserva/manifestações existem no runtime, mas os notebooks ainda usam saídas fixas. C01 não deve assumir que C00 está concluída.

Com esta feature, a sonda será ajustada no treino, selecionada na validação e congelada antes de uma avaliação final no teste. A auditoria distinguirá essa avaliação do KNN que usa referências do próprio teste e registrará as identidades necessárias para verificar a separação operacional dos exemplos.

## Flow

Reutiliza a extração pré-`fc`, o KNN, o Silhouette e a sonda em `viral_cnn.eval.latent`, o registro de variantes e a infraestrutura já adicionada de C00; não cria outra arquitetura, split ou topologia.

1. Configuração e entradas → notebook 06 e `viral_cnn.notebook_runtime` (exists): preflight específico de treino/validação/teste, taxonomia e cinco variantes Species; reserva saída exclusiva e identifica dependência taxonômica de outra execução.
2. Partições explícitas → `viral_cnn.data.parquet_dataset.ParquetBatchIterableDataset` (exists): lotes sem embaralhamento para extração, com identidade de linha vinculada à partição e ordem de leitura observada.
3. Modelo registrado e partições train/validation → `extract_latent_embeddings` em `viral_cnn.eval.latent` (exists): captura entrada real de `fc`, associa rótulos e identidades, e persiste apenas os dois artefatos de ajuste/seleção com proveniência verificável (door 1).
4. Artefatos train/validation → `LinearProbeClassifier` em `viral_cnn.eval.latent` (exists): valida proveniência, ajusta no treino e seleciona pesos pela validação; sela o hash antes de abrir o split test (door 2).
5. Modelo registrado e partição test, após a sonda selada → `extract_latent_embeddings` (exists): extrai e persiste o artefato test e os logits nativos, sem expor seus exemplos ao ajuste/seleção.
6. Sonda selada e artefato test → `LinearProbeClassifier.evaluate_artifact` e notebook 06 (exists): avaliam a sonda e a cabeça nativa sobre os mesmos exemplos do teste e exportam contagens, métricas e identidades da avaliação final.
7. Artefato test → `evaluate_cosine_knn`, `evaluate_silhouette_with_audit` e `calculate_taxonomic_correlations` (exists): produzem Tabelas 5/6 com partições e suportes auditados; UMAP permanece uma saída qualitativa separada.
8. Produtos e trilha de seleção → `write_notebook_manifest` (exists): registra configuração, hashes, suportes, saídas e limites de evidência no namespace C00, sem substituir resultados históricos.

## Impact

| Front | What changes |
| --- | --- |
| domain | `Linear_Probe_Test_Acc` passa a medir teste independente do ajuste da sonda; números anteriores ficam históricos e sem essa certificação. |
| domain | `split_origin` deixa de ser prova suficiente no fluxo empírico; a origem vem do artefato de embeddings associado às entradas de extração. |
| domain | `zero_self_neighbor` passa a ter evidência de identidades disjuntas e vizinhos observados. Isso não implica ausência de sequências idênticas ou de genomas compartilhados. |
| stored data | Novos caches separados por variante/split e identidade; caches antigos sem metadados são preservados e recusados para reutilização empírica. Nenhum backfill de proveniência. |
| outputs | Tabelas 5/6, relatório de sonda, CSV de predições, suporte por classe, projeção qualitativa e manifesto do 06 podem mudar; nomes históricos de métricas continuam identificados por seu significado. C06 decide a incorporação no manuscrito. |
| runtime | Completa a integração de C00 necessária ao notebook 06 e seus pré-requisitos; não migra os outros 14 notebooks nesta feature. |
| callers | Testes de `latent.py`, notebook 06 e uso sintético da sonda em `scripts/run_latent_evaluation.py` devem acompanhar o contrato; não ampliar o CLI para um novo pipeline empírico. |
| scientific | Congela backbone/checkpoint e mantém classes/logits, codificação, split publicado e ruído. Sonda corrigida recebe identificador próprio para distinguir o protocolo inválido anterior. |

## Relations

- Uma execução possui artefatos de embeddings para cada variante e cada um dos três splits.
- Cada artefato corresponde a uma lista identificada de partições, a um checkpoint, ao pré-processamento e à versão do extrator (door 1).
- Cada linha de embeddings possui um rótulo e uma identidade operacional de exemplo; a ordem preserva a associação entre eles.
- Uma sonda selecionada pertence a uma variante e aos artefatos train/validation que a determinaram (door 2).
- Uma avaliação final pertence a uma sonda selada e ao artefato test; referência/consulta do KNN pertencem a uma análise distinta.
- IDs de genoma são opcionais e nunca derivados de nome taxonômico ou identidade operacional de linha.

## Surface

None - não há API HTTP ou serviço externo. O contrato local de artefatos e sonda está em Landing; o notebook e seus estados estão em Observable.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1. Identidade persistida dos embeddings | Cache `features_<variant>_<split>_<fingerprint>.parquet` acompanhado de manifesto versionado; fingerprint vincula `split`, partições e SHA-256, checkpoint e SHA-256, ordem de classes, configuração de codificação, modo sintético/empírico e digest do código relevante; identidade de linha: digest da partição + ordinal da linha | Cache por nome de modelo ou por caminho/mtime: pode aceitar dados ou pesos substituídos sem mudar o nome. IDs de linha não são hashes de sequência, pois fragmentos idênticos podem existir no split publicado. |
| 2. Protocolo de sonda persistido | `probe_protocol: c01_train_val_frozen_v1`; estados `UNFITTED -> FITTED -> SELECTED -> SEALED -> TEST_EVALUATED`; manifesto de seleção vincula artefatos train/validation e hash dos pesos selados; teste não participa de fit/seleção | Confiar em `split_origin="train"` ou ajustar depois de consultar o teste: não permite auditar a origem ou a independência do ajuste. |

Nothing else in this change is hard to reverse. Ao implementar, registrar as portas aprovadas em `.specs/STATE.md`; não reescrever decisões científicas anteriores silenciosamente.

## Criteria

### S1: Embeddings rastreáveis dos três splits (P1)

**Acceptance Criteria**

1. WHEN o notebook 06 inicia uma execução empírica THEN o sistema SHALL exigir partições explícitas e não vazias de train, validation e test, taxonomia alinhada e checkpoints das cinco variantes Species antes de produzir métricas.
2. IF falta uma dessas entradas ou a taxonomia não cobre um rótulo utilizado THEN o sistema SHALL interromper a execução com o caminho/rótulo responsável, sem substituição sintética ou atribuição silenciosa ao táxon zero.
3. WHEN embeddings são extraídos THEN o sistema SHALL preservar a correspondência entre vetor pré-`fc`, rótulo e identidade de linha usando a dimensão observada de `model.fc.in_features`.
4. IF os splits compartilham partição ou identidade operacional de linha THEN o fluxo da sonda SHALL rejeitar a configuração antes de qualquer atualização de parâmetros.
5. WHEN um cache é consultado THEN o sistema SHALL reutilizá-lo somente se sua identidade corresponder integralmente às entradas, split, checkpoint, ordem de classes, configuração, modo e código relevantes.
6. IF um cache está incompleto, sem manifesto, corrompido ou com identidade divergente THEN o sistema SHALL recusá-lo e registrar a causa antes de recalcular em uma nova saída, preservando o arquivo anterior.
7. WHEN a extração utiliza amostragem ou limite de linhas THEN o manifesto SHALL registrar regra, semente, população disponível e contagem efetivamente extraída por split.

**Independent test:** extrair vetores de um modelo pequeno com três fixtures distintas; conferir valores contra a entrada de `fc`, alinhamento por lote e recusa de alias de partição e cache de outro split/checkpoint.

### S2: Ajuste e seleção sem consultar o teste (P1)

**Acceptance Criteria**

8. WHEN o fluxo auditado ajusta a sonda THEN o sistema SHALL aceitar somente embeddings com proveniência train verificada, mantendo o backbone sem atualizações.
9. IF embeddings do teste são fornecidos ao ajuste ou seleção, mesmo acompanhados de `split_origin="train"`, THEN o sistema SHALL rejeitá-los antes da primeira atualização da sonda.
10. WHEN a sonda é selecionada THEN o sistema SHALL escolher a época de maior Accuracy na validação, resolvendo empate pela primeira época e registrando os valores por época e a regra aplicada.
11. WHEN os pesos selecionados são selados THEN o sistema SHALL registrar o hash desses pesos antes de carregar o artefato de teste para a avaliação da sonda.
12. WHILE a sonda está selada o sistema SHALL recusar fit ou seleção posteriores no mesmo fluxo de execução.
13. WHEN a avaliação final é solicitada THEN o sistema SHALL avaliar uma única vez por sonda selada no fluxo auditado, preservando o hash dos pesos e o vínculo aos exemplos do teste.
14. WHEN o notebook exporta a auditoria da sonda THEN o sistema SHALL incluir ResNet50, ResNet50_F, ConvNeXt_Z, ConvNeXt_T e ConvNeXt_F com Accuracy, Macro-F1, N e suporte de classes.
15. WHEN a cabeça nativa é comparada à sonda THEN o sistema SHALL usar os mesmos exemplos identificados do teste e apresentar o KNN test-reference/query como protocolo separado, sem afirmar gargalo causal da cabeça.

**Independent test:** usar artefato test com origem falsamente nomeada train e confirmar ausência de atualização; verificar seleção por validação numa trajetória conhecida e impossibilidade de refit após selagem. Trocar somente os dados do teste não pode alterar os pesos selecionados.

### S3: KNN por fragmentos com denominadores verificáveis (P1)

**Acceptance Criteria**

16. WHEN KNN é avaliado em Species, Genus e Order THEN o sistema SHALL usar os embeddings originais do teste, normalização L2, cosseno, semente 42 e referência/consulta estratificada nominal 80/20, registrando as proporções efetivas e identidades de ambos os conjuntos.
17. WHEN vizinhos são recuperados THEN a auditoria SHALL registrar zero interseção entre identidades de referência e consulta e zero ocorrência da própria identidade entre os vizinhos de cada consulta.
18. IF não há consultas estratificadas elegíveis ou menos de cinco referências THEN o resultado empírico de 5-NN SHALL ser marcado indisponível com motivo e suporte, sem fallback aleatório não estratificado ou redução silenciosa de k.
19. WHEN métricas KNN são exportadas THEN o sistema SHALL registrar classes totais, classes com consulta, classes somente na referência, N de consultas, k solicitado/efetivo e a definição de Top-5 como presença do rótulo entre vizinhos.
20. IF IDs de genoma por fragmento não estão disponíveis THEN o relatório SHALL declarar `genome_independence: unknown` e não executar a variante por genoma nem preencher accessions por heurística.
21. WHERE IDs de genoma válidos por fragmento são fornecidos THEN a variante existente por genoma SHALL produzir resultado separado com interseção de genomas nula, população elegível e exclusões por espécie explícitas.
22. WHEN Tabelas 5/6 são exportadas THEN o sistema SHALL distinguir fração de classes válidas de fração de amostras retidas no Silhouette e manter suportes de correlação KNN/Silhouette separados.

**Independent test:** verificar os vizinhos efetivamente usados em fixture com IDs; casos negativos com todas as classes unitárias, menos de cinco referências e ausência de IDs genômicos. Excluir uma classe do Silhouette não pode eliminá-la do suporte válido do KNN.

### S4: Evidência local e empírica separadas (P1)

**Acceptance Criteria**

23. WHEN o notebook produz resultados THEN o sistema SHALL salvar caches, pesos da sonda, predições, auditorias e manifesto em namespace exclusivo por execução integrado às APIs C00, registrando hashes dos splits/checkpoints e configuração efetiva.
24. WHILE o notebook executa em modo smoke o sistema SHALL usar fixtures identificadas por split em diretório temporário ou exclusivo e registrar `is_smoke_test: true`, sem admissão empírica.
25. WHEN C01 é verificada THEN o relatório SHALL separar testes locais, execução real e divergências históricas, mantendo a validação empírica pendente se os dados/checkpoints não estiverem acessíveis.

**Independent test:** executar o 06 com fixtures CPU e casos negativos em namespace temporário; verificar manifesto, origens e arquivos. A validação remota requer execução explícita com `SMOKE_TEST=0` e artefatos reais, sem declarar os CSVs locais atuais como reprodução.

## Out of scope

| Excluded | Why |
| --- | --- |
| Recuperar accessions ou redesenhar o split publicado | Ausência de IDs não impede a correção da sonda; disjunção operacional não comprova disjunção de sequência/genoma. |
| Retreinar backbones ou comparar arquiteturas causalmente | C05; C01 ajusta somente sondas sobre representações congeladas. |
| Migrar todos os notebooks para C00 | Aqui será integrada somente a cadeia de entrada/saída do 06; pendências gerais de C00 permanecem explícitas. |
| Igualar os valores do manuscrito ou reescrever suas conclusões | C06 conciliará resultados e texto; não ajustar métricas aos números históricos. |
| Trocar distância, projeção, arquitetura ou ordem das classes | Contratos científicos preservados; UMAP não alimenta KNN, sonda ou Silhouette. |
| Ampliar o CLI sintético para inferência empírica completa | Atualizar consumidores da API da sonda é necessário; novo pipeline de CLI não é. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Perfil da skill | `light`; casos negativos do roadmap continuam obrigatórios | Não há perfil declarado em AGENTS.md | n |
| Dependência de C00 | Planejar C01 agora; construir sobre APIs presentes e exigir fechamento dos requisitos C00 do 06 antes da validação empírica | O relatório C00 declara implementação parcial, não uma cadeia certificada | n |
| Ajuste da sonda | AdamW, lr 1e-3, weight_decay 1e-4, batch 256, 10 épocas, semente 42; smoke reduz épocas | Preserva os defaults atuais e acrescenta seleção em validação; registrar tudo na execução | n |
| Amostragem da sonda | Sem truncamento ou subamostragem empírica implícitos; qualquer orçamento reduzido deve ser configuração identificada e relatada | Não apresentar piloto como avaliação do universo completo | n |
| Macro-F1 da sonda | Manter a média sobre classes com suporte real no conjunto avaliado, regra atual de `evaluate`, com denominador explícito | Evita mudança silenciosa da métrica; não equiparar automaticamente ao Macro-F1 do artigo | n |
| Arredondamento 80/20 | Preservar a regra atual por classe elegível, reportando proporções e singletons; eliminar somente fallback sem estratificação | Mantém comparabilidade onde há suporte e torna casos degenerados indisponíveis | n |
| Compatibilidade de fit | Arrays sem proveniência podem permanecer apenas como uso mecânico legado/sintético explicitamente não certificado; o caminho auditado exige artefatos verificados | Permite ajustar consumidores existentes sem atribuir origem a arrays arbitrários | n |
| Limite da proteção de origem | Verificar a trilha gerada pelo extrator, IDs e arquivos, sem prometer detectar metadados adulterados deliberadamente | A prova protege contra troca acidental de split, não contra falsificação externa de artefatos | n |
| Ordem de extração | Capturar identidade de linha junto ao vetor; não inferir ordem por contagem após execução multiworker | A associação observada evita erros de alinhamento entre lotes | n |

**Open questions:** none - defaults acima deixam o plano implementável; disponibilidade remota condiciona a evidência empírica, não a escrita do plano.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Notebook 06 | Entradas ausentes, taxonomia incompleta e alias de split | AC 1–4; erro explícito antes do ajuste |
| Cache | Hit, obsoleto, legado e gravação interrompida | AC 5–6; reutilização por identidade e recusa com motivo |
| Extração | Dimensão, ordem, limites e suporte | AC 3 e 7 |
| Sonda | Origem inválida, seleção e selagem | AC 8–13 |
| Sonda | Resultados por variante e comparabilidade | AC 14–15 |
| KNN | Sucesso, conjunto vazio e referência insuficiente | AC 16–19 |
| KNN por genoma | IDs ausentes ou variante habilitada | AC 20–21 |
| Tabelas 5/6 | Suporte de classes e amostras | AC 22 |
| Execução | Repetição, saídas isoladas e modo sintético | AC 23–24; existing - reserva exclusiva em C00 |
| Console e log | Caminhos carregados de datasets, caches e pesos | existing - `log_artifact_path` exigido por AGENTS.md; incluir nos novos carregamentos |
| Verificação | Dados reais indisponíveis e divergências | AC 25 |
| Todas as superfícies | Autorização, rate limiting e paginação HTTP | n/a - pipeline local sem serviço HTTP |
| Interface visual | Loading, navegação e layout | n/a - sem tela nova; UMAP permanece artefato qualitativo existente |

As nove dimensões foram consideradas: validação (AC 1–5, 8–9), falha parcial (AC 6), repetição (AC 13 e 23), autorização (n/a — uso local), concorrência (reserva exclusiva C00 e publicação de cache completo), ciclo de dados (AC 5–6 e 23), dependência externa (AC 1–2 e 25), transições (AC 10–13) e observabilidade (AC 7, 19 e 23). A associação a provas será escrita em `checks.md` após revisão do plano.

## Sources

- `ROADMAP_CORRECOES_REPRODUCAO.md`, C01, dependência C00 e portões — delimita correção da sonda, KNN e invalidação de cache.
- `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` e specs S08/N04/R04 — contratos científicos, implementação anterior e preservação histórica.
- Notebook 06, `src/viral_cnn/eval/latent.py`, seus testes, notebook histórico `short/ICTV_latent_space-bkp.ipynb`, manifesto local e verificação C00 — evidência inspecionada; resultados locais sintéticos não certificam o artigo.
