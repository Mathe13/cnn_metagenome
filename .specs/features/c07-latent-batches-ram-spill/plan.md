# C07 — Avaliação latente em lotes com RAM e spill para disco

## Problem

O notebook de reprodução short/06 falha durante a seleção da sonda linear de ResNet50. O log remoto fornecido em 2026-11-04 registra 1.706.271 exemplos de validação e uma tentativa de alocar 86,97 GiB, com 80,57 GiB livres. `fit_select_and_seal` envia todos os embeddings de validação à GPU e calcula todos os logits simultaneamente. O parâmetro `batch_size=256` controla somente o treinamento.

O mesmo padrão aparece em `LinearProbeClassifier.evaluate`. Além disso, a extração acumula listas e as concatena em RAM; o notebook guarda logits nativos completos de cinco variantes, embora a comparação posterior use apenas Top-1. Avaliar em lotes sem limitar o armazenamento no host deslocaria o problema para a RAM.

O resultado pretendido é executar seleção e avaliação completas com memória de dispositivo proporcional ao lote, armazenamento dos resultados em RAM sob orçamento explícito e spill para disco quando necessário, mantendo identidades e protocolo científico.

## Flow

Reutiliza as APIs de extração, artefatos e sonda em `viral_cnn.eval.latent`, os logs em `viral_cnn.utils.logging_utils` e o namespace de execução já usado pelo notebook 06.

1. Notebook 06 (exists) recebe tamanho dos lotes, orçamento de RAM opcional e diretório de spill; se o orçamento não foi informado, calcula 30% da RAM disponível no início da execução; compartilha esse limite durante a execução e registra o valor calculado sob o namespace atual (door 1).
2. Partições e modelo registrado → `extract_latent_embeddings` em `viral_cnn.eval.latent` (exists): extrai em lotes, associa features/rótulos/identidades e entrega arrays residentes em RAM ou mapeados em disco, sem concatenar uma segunda cópia integral (door 1).
3. Artefatos → `save_embedding_artifact` e `load_embedding_artifact` (exists): escrevem/leem em blocos preservando Parquet, sidecar e verificações de identidade existentes, sem materializar todo o conteúdo novamente.
4. Artefatos train/validation → `LinearProbeClassifier.fit_select_and_seal` (exists): conserva treinamento, valida em lotes transferidos individualmente, acumula acertos no dispositivo e seleciona/sela pesos com a regra atual.
5. Após selagem, partições test → `extract_latent_embeddings` e notebook 06 (exists): guardam Top-1 nativo em vez de todos os logits; solicitação explícita de logits por outros consumidores continua atendida com armazenamento limitado pelo orçamento.
6. Artefato test → `LinearProbeClassifier.evaluate_artifact`/`evaluate` (exists): avalia em lotes e calcula Accuracy/Macro-F1 globais com os contadores por classe; resultados por exemplo solicitados usam o mesmo armazenamento RAM/spill.
7. Notebook 06 (exists) exporta resultados e configuração efetiva, libera artefatos que não serão mais usados e encerra o armazenamento temporário sem invalidar arrays ainda em uso; falhas de disco deixam erro explícito, sem promover artefato parcial a completo.

## Impact

| Front | What changes |
| --- | --- |
| domain | `batch_size` passa a limitar também cada forward de validação da sonda; avaliação recebe limite equivalente sem quebrar chamadas existentes. |
| domain | Resultado em RAM pode ter backing em disco; valores, dtype, ordem de linhas e classes mantêm o significado atual. |
| domain | O notebook 06 conserva Top-1 nativo, suficiente para sua comparação atual, em vez de cinco matrizes de logits completas. |
| outputs | `linear_probe_audit.csv` passa a identificar as linhas por SHA-256 ordenado (`Test_Row_IDs_SHA256`) em vez de serializar milhões de IDs; contagem e ordem continuam vinculadas à avaliação selada. |
| stored data | Parquets/sidecars históricos são preservados; spill é temporário e não substitui artefato científico completo. Não haverá migração, sobrescrita ou revalidação retrospectiva de caches antigos. |
| results | A seleção da sonda, seus CSVs/manifestos e a comparação de cabeça nativa são afetados operacionalmente; equivalência numérica será verificada em fixtures. Pequenas diferenças de kernels GPU entre tamanhos de lote exigem comparação remota, sem promessa de igualdade bit a bit. |
| scope | Arquivos previstos: módulo latente e utilitário de armazenamento conforme convenções, notebook 06, testes dos contratos e documentação de execução. Notebooks históricos ficam preservados. |

## Relations

Uma execução possui um orçamento de armazenamento compartilhado e zero ou mais arquivos de spill exclusivos. Cada resultado preserva a correspondência um a um entre linha, rótulo e identidade. O armazenamento temporário não altera a entidade científica `EmbeddingArtifact` nem seu vínculo a split/checkpoint.

## Surface

None - não há rota HTTP; as superfícies são chamadas Python e artefatos locais, sem códigos HTTP aplicáveis.

A API de extração recebe opções adicionais de orçamento/armazenamento e retorno Top-1; entrega arrays compatíveis com os consumidores existentes. Resultados possíveis: sucesso, `ValueError` ou `OSError`.

A API de avaliação da sonda recebe tamanho de lote opcional e arrays em RAM ou disco; entrega métricas com o significado atual. Resultados possíveis: sucesso, `ValueError` ou `RuntimeError`.

O notebook recebe `LATENT_EVAL_BATCH_SIZE=256`, `LATENT_RAM_BUDGET_MB` opcional (default 30% da RAM disponível) e `LATENT_SPILL_DIR` opcional; entrega os resultados existentes e configuração efetiva registrada. Resultados possíveis: concluído ou erro explícito. Estes conjuntos de resultados serão enumerados em Coverage nas provas.

O CSV de auditoria da sonda exporta a identidade das linhas como o campo `Test_Row_IDs_SHA256`, junto com N, variante e métricas; a lista integral de IDs permanece no artefato test versionado e pode ser incluída explicitamente pelas chamadas Python compatíveis.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — Política de armazenamento consumida pelo notebook | `LATENT_EVAL_BATCH_SIZE=256`; `LATENT_RAM_BUDGET_MB` em MiB, default dinâmico igual a 30% da RAM disponível quando a execução começa; `LATENT_SPILL_DIR` ausente resolve para `output_dir / "spill"`; orçamento calculado compartilhado entre resultados ainda vivos da execução; zero MiB força disco | Orçamento independente por array: multiplicaria a ocupação ao manter cinco variantes. Spill somente após OOM: o sistema operacional pode encerrar o processo antes de uma exceção Python. Valor fixo de 1024 MiB: ignora a capacidade do host e a preferência agora definida pelo usuário. |
| 2 — Identidade compacta na auditoria CSV da sonda | Campo `Test_Row_IDs_SHA256`: SHA-256 da sequência ordenada de IDs, com N mantido como coluna separada | Serializar a lista completa por variante: cria cópias grandes em Python/JSON e torna o CSV impraticável; o artefato test conserva os IDs completos e o caminho Python mantém inclusão explícita opcional. |

Nenhuma dependência externa nova será adicionada. O layout interno do spill é temporário, reversível e não constitui formato de cache público.

## Criteria

**Acceptance Criteria**

1. WHEN a sonda valida ou avalia N exemplos THEN o sistema SHALL executar cada forward com no máximo `batch_size` exemplos, incluindo o último lote incompleto.
2. WHEN o notebook executa sem overrides THEN o sistema SHALL configurar avaliação em lotes de 256 e calcular o orçamento compartilhado como 30% da RAM disponível, em MiB, antes da primeira alocação gerenciada.
3. WHILE resultados gerenciados cabem no orçamento THEN o sistema SHALL armazená-los em RAM.
4. WHEN uma nova alocação de resultados ultrapassaria o orçamento compartilhado THEN o sistema SHALL usar armazenamento em disco sem exceder o orçamento de arrays residentes gerenciados.
5. WHEN a RAM disponível sinaliza pressão ou uma alocação em RAM gera `MemoryError` THEN o sistema SHALL usar spill antes de continuar a acumulação dos resultados.
6. WHEN o armazenamento muda de RAM para disco THEN o sistema SHALL preservar valores, dtype e ordem das linhas sem perda, repetição ou desalinhamento de rótulos e identidades.
7. WHEN a extração é finalizada ou um artefato é persistido/recarregado THEN o sistema SHALL evitar concatenação/materialização de uma segunda cópia integral dos resultados gerenciados em RAM.
8. WHEN a validação calcula Accuracy THEN o sistema SHALL dividir a soma global de acertos pelo número total de exemplos, sem média não ponderada das accuracies dos lotes.
9. WHEN o teste calcula Macro-F1 THEN o sistema SHALL manter a regra atual da sonda: média por classes com suporte real, a partir das contagens globais de todos os lotes.
10. WHEN a sonda seleciona os pesos THEN o sistema SHALL manter maior Accuracy de validação com primeira época como desempate, selagem/hash e avaliação única do teste previstos por AD-019.
11. WHEN o notebook compara a cabeça nativa à sonda THEN o sistema SHALL usar Top-1 dos mesmos exemplos identificados sem armazenar a matriz completa de logits nativos.
12. WHERE um consumidor solicita explicitamente logits completos THEN o sistema SHALL manter o retorno existente e aplicar o orçamento RAM/spill ao armazenamento desses logits.
13. IF tamanho do lote não é inteiro positivo ou orçamento de RAM é negativo THEN o sistema SHALL rejeitar a configuração antes da primeira inferência.
14. IF disco está cheio ou diretório de spill não pode ser escrito THEN o sistema SHALL propagar erro com o caminho envolvido sem publicar resultado científico parcial como completo.
15. WHEN duas instâncias usam o mesmo diretório-base de spill THEN o sistema SHALL criar arquivos exclusivos sem sobrescrever os da outra instância.
16. WHEN um armazenamento é encerrado explicitamente THEN o sistema SHALL remover somente seus arquivos temporários após liberar os respectivos arrays; arquivos científicos persistidos permanecem intactos.
17. WHEN ocorre spill THEN o sistema SHALL registrar seu caminho por `log_artifact_path`, respeitando `LOG_PATH`, e registrar orçamento, lote e uso de disco na configuração/auditoria da execução.
18. WHILE métricas da sonda são acumuladas por lote THEN o sistema SHALL manter contadores no dispositivo sem `.item()` ou transferência desses contadores para CPU até terminar a avaliação.
19. WHEN a implementação é verificada THEN o relatório SHALL distinguir provas sintéticas CPU, validação remota pendente e resultados científicos não reproduzidos.

## Out of scope

| Excluded | Why |
| --- | --- |
| Mudar split, classes, backbone, treinamento da sonda, ruído ou definição de métrica | A correção é de uso de memória e preserva o protocolo C01. |
| Reimplementar KNN, Silhouette ou UMAP como algoritmos externos em disco | Seus custos e possíveis cópias internas não são resolvidos pelo armazenamento dos resultados de inferência. |
| Garantir teto de RSS de todo o processo ou eliminar qualquer OOM | O orçamento cobre os arrays gerenciados; Python, bibliotecas, workers, conjuntos de IDs e memória de kernels têm custos adicionais. |
| Retomada automática após queda de processo ou reaproveitamento automático de caches históricos | Requer protocolo próprio de recuperação/proveniência; esta feature encerra seus temporários e preserva os artefatos já salvos. |
| Executar treinamento empírico ou ajustar resultados ao artigo | Dados e checkpoints remotos não estão disponíveis nesta sessão. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Perfil da skill | `light` | Não há outro perfil declarado; verificador independente obrigatório e testes negativos explícitos. | n |
| Orçamento automático de RAM | 30% da RAM disponível no começo da execução; override por `LATENT_RAM_BUDGET_MB` | Preferência definida pelo usuário em 2026-11-04; o orçamento é compartilhado e registrado. | y — user requested |
| Pressão de memória durante execução | Fazer spill preventivamente quando a RAM disponível cair abaixo do orçamento calculado; se a medição dinâmica não estiver disponível, respeitar o limite de 30% calculado no início e acionar spill quando o orçamento gerenciado for atingido | Evita depender de exceção após esgotamento do host; a medição é preventiva e não garante capacidade contra concorrência externa. | n |
| Transferências necessárias dos resultados | Features e previsões destinadas ao host podem ser copiadas por lote; contadores das métricas permanecem no dispositivo até o final | A extração exige armazenamento no host; a regra AD-018 impede sincronização desnecessária das métricas, não a saída solicitada pelo usuário. | n |
| Resultados já salvos | Preservar; não editar sidecars nem alterar hashes para forçar reutilização | O módulo atual identifica o código extrator; alteração exige validar a proveniência antes de qualquer reutilização. | n |

**Open questions:** none - o usuário aprovou a implementação do plano revisado em 2026-11-04. Dados/checkpoints reais bloqueiam somente a validação empírica.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| APIs Python | Entradas, retornos e limites | Criteria 1–7, 12–13; chamadas existentes continuam válidas. |
| APIs Python | Erros e acesso | Criteria 13–14; existing - guardas de split/selagem; n/a - autenticação HTTP e rate limits, pois são APIs locais. |
| Notebook 06 | Configuração e defaults | Criterion 2; door 1. |
| Notebook 06 | Vazio, falha, ordem e comparação | Criteria 6, 8–11, 14; existing - extração vazia no fluxo auditado é recusada. |
| Spill | Orçamento, pressão, duplicatas e ciclo de vida | Criteria 3–7 e 14–16. |
| Artefatos científicos | Formato, publicação e preservação | Criteria 7, 10 e 14; existing - identidade/hash e recusa de sobrescrita. |
| Console e manifesto | Caminhos e configuração efetiva | Criterion 17. |
| Documento de execução | Português, parâmetros e limites da verificação | Criterion 19; registrar comando de teste e execução remota no relatório final. |
| Interface visual | Loading, layout e confirmação destrutiva | n/a - não há tela nova nem operação destrutiva sobre dados do usuário. |

As nove dimensões foram examinadas: validação (13), falha parcial (14), repetição (10 e 15), autorização (n/a — execução local), concorrência/ordem (6 e 15), ciclo dos dados (7 e 16), dependências externas (14; sem rede), transições (4–5 e 10) e observabilidade (17 e 19). As provas e a associação por dimensão serão derivadas em checks.md somente após a revisão humana deste plano.

## Sources

- Pedido do usuário de 2026-11-04 e log anexado: avaliação em batches com resultados em RAM e spill quando necessário; OOM de 86,97 GiB na validação de ResNet50.
- `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` (AD-012, AD-017, AD-018 e AD-019) e plano C01: contratos científicos e de execução.
- `src/viral_cnn/eval/latent.py` e `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb`: cadeia atual de extração → seleção → teste → métricas.
