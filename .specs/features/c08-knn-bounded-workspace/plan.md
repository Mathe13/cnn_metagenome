# C08 — KNN exato com memória temporária limitada

## Problem

O notebook de reprodução short/06 já limita a extração e a avaliação da sonda por lotes, mas a avaliação KNN ainda copia toda a referência e todas as consultas para tensores próprios e calcula, por lote de consultas, uma matriz de similaridades contra a referência inteira. Com milhões de fragmentos, esse custo cresce com o número de referências e escapa ao orçamento de arrays do C07. A auditoria atual também converte listas completas de IDs de referências, consultas e vizinhos em um único JSON. Não há medição empírica de pico desta etapa no cluster nesta sessão.

No código atual, `evaluate_cosine_knn` recebe `device="cpu"` por padrão e o notebook 06 não informa `device`; portanto, a etapa KNN do notebook roda na **CPU** até que a chamada seja alterada explicitamente. A VRAM remota de aproximadamente 96 GiB não limita o consumo de RAM dessa chamada. O usuário deve conseguir executar KNN exato com uso temporário previsível, auditar os vizinhos e distinguir qualquer mudança de protocolo dos resultados históricos.

## Flow

Reutiliza `evaluate_cosine_knn` e os artefatos de embeddings versionados do notebook 06; o C07 continua responsável pelo armazenamento dos embeddings extraídos.

1. Notebook 06 (exists) lê `LATENT_KNN_WORKSPACE_MB` e `LATENT_KNN_QUERY_BATCH_SIZE`, valida os valores, passa `ctx.device` e registra dispositivo, limites e dimensões efetivas no manifesto da execução.
2. `evaluate_cosine_knn` em `viral_cnn.eval.latent` (exists) mantém a partição estratificada de referências/consultas, a ordem de IDs e o KNN cosseno no espaço original; lê os embeddings por blocos de índices sem formar cópias integrais de features de referência/consulta.
3. `evaluate_cosine_knn` (exists) normaliza blocos de referências e consultas, calcula similaridades em blocos que cabem no workspace e conserva apenas os cinco melhores candidatos por consulta; acumula Top-1, Top-5, métricas por classe e contagens de auditoria globais.
4. Notebook 06 (exists) recebe resultados compactos e persiste os índices de consulta/vizinhos em Parquet por variante e nível taxonômico (door 2); o JSON de auditoria armazena hashes, contagens, protocolo e vínculos aos Parquets/test embeddings (door 2).
5. `evaluate_silhouette_with_audit` (exists) e `export_umap_projection` (exists) mantêm as amostras atuais de 5.000 e 100 linhas no notebook; o notebook registra seus tamanhos efetivos sem usar o limite KNN como regra científica para essas amostras.

## Impact

| Front | What changes |
| --- | --- |
| domain | O KNN continua exato, com `k=5`, normalização L2, cosseno, split estratificado 80/20 e semente 42; empates de similaridade passam a usar o menor índice original de referência, registrado como protocolo `c08_tiled_cosine_v1` (door 1). |
| domain | O notebook usa `ctx.device` para KNN: CUDA no cluster quando disponível, CPU no smoke local. Resultados de pontos sem empate serão comparados à implementação atual com tolerância numérica; empates podem alterar a escolha de vizinhos e exigem auditoria separada. |
| outputs | `table5_latent_clustering_metrics.csv` e Tabela 6 podem mudar somente por empates ou diferenças numéricas do backend; os resultados históricos não serão sobrescritos e cada nova execução terá seu próprio `run_id` e protocolo. |
| stored data | `knn_identity_audit.json` deixa de embutir milhões de IDs e passa a referenciar Parquets de vizinhos por variante/rank, com hashes e contagens (door 2); os artefatos de embeddings test continuam a fonte versionada dos IDs completos. |
| resources | `LATENT_KNN_WORKSPACE_MB` limita apenas blocos explícitos de cálculo KNN, separadamente do orçamento C07 para arrays persistentes. O RSS total, overhead de PyTorch, caches de CUDA, índices/IDs e custos de Silhouette/UMAP não recebem teto rígido. |

## Relations

Uma execução C08 tem um protocolo KNN identificado, cinco variantes e até três níveis por variante. Cada avaliação de nível possui uma partição disjunta de referência/consulta, um resumo de auditoria e, quando há consultas suficientes, um Parquet imutável com um registro por consulta e exatamente `k=5` índices de referência na ordem de vizinhança. Esses índices apontam para linhas do artefato test identificado por hash; o JSON guarda o hash do Parquet e as contagens. Nenhum arquivo histórico é migrado.

## Surface

None - não há rota HTTP nem códigos de status. As superfícies são chamadas Python, variáveis de ambiente do notebook e artefatos locais.

`evaluate_cosine_knn(...)` recebe features/labels/IDs, `workspace_mb`, `query_batch_size`, limite opcional `reference_batch_size` para controlar blocos, `device` e saída opcional de auditoria; devolve as métricas e contagens atuais, detalhes completos por padrão aos consumidores existentes ou modo compacto ao notebook. Resultados possíveis: sucesso, `ValueError` para configuração/identidade inválida, `OSError` para escrita e `RuntimeError` para bloco que não cabe.

O notebook 06 recebe `LATENT_KNN_WORKSPACE_MB` e `LATENT_KNN_QUERY_BATCH_SIZE` opcionais e `ctx.device`; gera Tabelas 5/6 e auditoria JSON/Parquet identificada no manifesto. Resultados possíveis: conclusão com manifesto ou erro explícito sem auditoria completa. Esses resultados serão enumerados em `Coverage` nas provas.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — protocolo de desempate e execução | `c08_tiled_cosine_v1`: Top-5 por similaridade cosseno decrescente e, em igualdade exata, menor índice original de referência; `k=5`, split estratificado 80/20, semente 42 e voto majoritário atual registrados no manifesto. | Depender da ordem de empates de `torch.topk` por bloco: a escolha poderia variar com o tamanho do bloco, sem trilha de interpretação. |
| 2 — auditoria de vizinhos escalável | `knn_identity_audit.json` versão `c08_knn_audit_v1` com variante, rank, N, hashes ordenados de IDs e caminho/hash do Parquet de vizinhos; Parquet por variante/rank contém índice original da consulta e os cinco índices originais das referências, vinculados ao hash do artefato test. | Manter listas completas de IDs no JSON: a serialização cresce com milhões de referências/consultas e duplica identidades já presentes no artefato test. Guardar só contagens: perderia a possibilidade de reconstruir a vizinhança. |

Nenhuma dependência externa nova. O protocolo e o formato de auditoria são identificadores novos; caches e saídas C01/C07 permanecem históricos.

## Criteria

### S1: KNN exato com workspace limitado (P1)

**Acceptance Criteria**

1. WHEN o notebook 06 executa sem overrides THEN o sistema SHALL usar `LATENT_KNN_WORKSPACE_MB=1024` MiB e `LATENT_KNN_QUERY_BATCH_SIZE=256` e registrar ambos como configuração efetiva.
2. WHEN o notebook 06 inicia KNN THEN o sistema SHALL passar `ctx.device` à avaliação e registrar o dispositivo efetivo.
3. IF workspace ou lote de consultas não forem inteiros positivos THEN o sistema SHALL rejeitar a configuração antes da primeira multiplicação de matrizes.
4. WHEN KNN avalia uma partição THEN o sistema SHALL manter `k=5`, split estratificado de referência/consulta 80/20 com semente 42, normalização L2, cosseno, voto majoritário e ausência de auto-vizinho.
5. WHILE calcula similaridades THEN o sistema SHALL manter cada bloco explícito de referências, consultas, matriz de similaridades e candidatos dentro de `workspace_mb`, sem transferir ou copiar todas as features de referência/consulta de uma vez.
6. WHEN o número de referências ultrapassa o bloco calculado THEN o sistema SHALL combinar o Top-5 de todos os blocos sem omitir referências e desempatar similaridades iguais pelo menor índice original de referência.
7. WHEN a avaliação termina THEN o sistema SHALL calcular Accuracy, Top-5, denominadores e métricas por classe sobre todas as consultas, equivalentes ao KNN de matriz completa para distâncias sem empate dentro da tolerância numérica declarada.
8. IF nem cinco referências e uma consulta cabem no workspace configurado THEN o sistema SHALL falhar antes da inferência com mensagem contendo tamanho solicitado e mínimo calculado, sem reduzir `k` ou subamostrar consultas.
9. WHEN o notebook avalia Species, Genus e Order THEN o sistema SHALL evitar cópias integrais de features filtradas por rank, preservando a seleção de linhas e os mapeamentos taxonômicos atuais.

**Independent test:** fixtures com referências maiores que um bloco, lotes finais incompletos, empates exatos e monitor de formas de tensores comparam o resultado a uma referência de matriz completa.

### S2: auditoria e controles dos processos seguintes (P1)

**Acceptance Criteria**

10. WHEN o notebook grava auditoria KNN THEN o sistema SHALL publicar JSON `c08_knn_audit_v1` sem listas integrais de IDs e um Parquet de índices por variante/rank com exatamente uma linha por consulta e cinco vizinhos, ambos ligados ao hash do artefato test.
11. WHEN uma auditoria KNN é publicada THEN o sistema SHALL registrar hashes ordenados de referências, consultas e vizinhos, N de cada conjunto, contagem de auto-vizinhos zero, dimensões efetivas dos blocos, dispositivo e protocolo.
12. IF a escrita de Parquet/JSON falha THEN o sistema SHALL remover temporários próprios e impedir que a auditoria incompleta seja reportada como concluída no manifesto da execução.
13. WHEN o notebook executa Silhouette e UMAP THEN o sistema SHALL manter os limites existentes de 5.000 e 100 linhas, respectivamente, e registrar amostra efetiva e classes retidas/excluídas no manifesto.
14. WHEN o modo Python legado solicita detalhes de vizinhos THEN o sistema SHALL continuar retornando `reference_ids`, `query_ids`, `neighbor_indices` e `neighbor_ids` com seus significados atuais.
15. WHEN o notebook executa KNN THEN o sistema SHALL usar o modo compacto sem materializar listas integrais de IDs no JSON.
16. WHEN a feature é verificada localmente THEN o relatório SHALL distinguir equivalência sintética CPU, validação CUDA/dados reais pendente e ausência de reprodução científica do artigo.

**Independent test:** smoke do notebook 06 lê o JSON e os Parquets de auditoria sintéticos, reconstrói os IDs a partir do artefato test e confirma hash, N, exclusão de auto-vizinho e amostras Silhouette/UMAP.

## Out of scope

| Excluded | Why |
| --- | --- |
| KNN aproximado, subamostragem de referências/consultas ou mudança de `k` | Alterariam o resultado científico para resolver memória. |
| Reescrever Silhouette/UMAP em algoritmo externo ou controlar o RSS total do processo | Suas amostras já são pequenas no notebook; um teto total exigiria controlar bibliotecas e workers fora deste fluxo. |
| Retreinar modelos ou alterar embeddings, split publicado, arquitetura e nomes de variantes | A mudança é na execução e auditoria do KNN; diferenças empíricas exigem comparação com artefatos reais. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Skill disponível | Usar `tlc-spec-lean`, única variante `tlc-spec` instalada neste workspace | O pedido menciona `tlc-spec`; busca local e catálogo encontraram apenas `tlc-spec-lean`, já usado no C07. | n |
| Perfil de verificação | `light` | O repositório não declara perfil; é o padrão da skill. Empates e publicação parcial exigem provas negativas mesmo nesse perfil. | n |
| Workspace KNN | 1024 MiB separados do orçamento de arrays C07 | Limita a matriz temporária no CPU padrão atual e permite override no cluster; não promete teto de RSS. | n |
| Equivalência numérica entre CPU e CUDA | Comparar métricas e vizinhos sem empates com tolerância de similaridade `1e-5`; registrar empates | Kernels diferentes podem arredondar na última casa; empate exato tem regra nova e protocolo identificado. | n |

**Open questions:** none — defaults reversíveis e decisões de formato/protocolo estão explícitos para revisão deste plano.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| API Python KNN | valores padrão, forma dos retornos e detalhes legados | Criteria 1, 7, 14 |
| API Python KNN | validação de limites e erro de workspace insuficiente | Criteria 3, 8 |
| API Python KNN | identidade, ordem, empates e ausência de auto-vizinho | Criteria 4, 6, 9, 11 |
| Notebook 06 | variáveis de ambiente, valores efetivos, dispositivo e log | Criteria 1, 2, 11, 13 |
| Notebook 06 | saída parcial ou erro de disco | Criterion 12 |
| Auditoria JSON/Parquet | formato, tamanho, vínculo e reconstrução | Criteria 10, 11, 12, 15 |
| Notebook 06 | autorização, rate limit e confirmação destrutiva | n/a — execução local sem serviço ou operação sobre artefatos históricos |

As nove dimensões foram consideradas: validação (3, 8), falha parcial (12), repetição (10, 12 e nomes por execução), autorização (n/a — execução local), concorrência e ordem (6, 10–11), ciclo de dados (10, 12), dependências externas (12; sem rede), transições (10, 12) e observabilidade (1–2, 11, 13, 16). Elas serão associadas às provas em `checks.md` após a revisão deste plano.

## Sources

- Pedido do usuário nesta conversa: controles de memória para KNN e demais processos com `tlc-spec`; preferência anterior por batch e spill quando necessário.
- `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` (AD-012, AD-017, AD-018 e AD-019) e `docs/latent_space_audit.md`: invariantes científicos, proveniência e limites de afirmação.
- `src/viral_cnn/eval/latent.py`, notebook `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb`, testes C01/C07 e artefatos históricos: comportamento executado e contratos atuais.
