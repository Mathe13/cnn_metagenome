# M01 — Dados, taxonomia e split identificados

## Problem

Os notebooks de reprodução 10, 11 e 05 escrevem em pastas fixas e não vinculam suas entradas e saídas a uma rodada M00. O 05 usa um caminho pessoal para a planilha ICTV e gera taxonomia sintética quando a planilha falta em modo empírico. Seu manifesto registra o Parquet produzido como entrada, sem demonstrar a origem da curadoria. O pesquisador pode consumir uma taxonomia incompleta ou sintética como se fosse a reconstrução do híbrido histórico.

O saneamento atual corrige `model_label` e preenche ranks ausentes silenciosamente. A auditoria 11 verifica IDs e nomes originais, mas não verifica integralmente a correspondência de `model_label` nem a disponibilidade real de Genus/Order. A auditoria 10 já remove padding, mantém todas as linhas no denominador e usa índice SQLite limitado em disco; faltam identificação de execução, contagens de validation e separação explícita de overlap direto/RC.

Esta entrega cobre apenas M01 do roadmap de 09/11/2026. A população continua sendo o split histórico por fragmentos, closed-set; uma auditoria não cria independência genômica nem reproduz métricas do artigo.

## Flow

Reutilizar o runtime e as reservas M00, a normalização/índice SQLite de `data.retraining_audit`, a auditoria de `data.taxonomy_split_audit` e a curadoria de `taxonomy.curation`, sem reimplementar o split ou modificar notebooks históricos.

1. Configuração/ambiente e, quando selecionado, plano M00 entram em `m01_workflow`, que reutiliza `notebook_runtime` e `run_planning` (exists): resolver entradas explícitas, modo, IDs e destino exclusivo para cada notebook (door 1).
2. Fragmentos train/validation/test entram em `data.retraining_audit` (exists): inspecionar esquema, contar classes/qualidade/comprimentos por split e comparar test contra train por sequência direta e reverso complementar, persistindo decisão e CSVs identificados (door 2).
3. Taxonomia existente, CSV original e decisão/CSV selecionados de 10 entram em `data.taxonomy_split_audit` (exists): validar identidade e contagens, reportar ranks ausentes e estratificar overlap, preservando a referência como somente leitura (door 3).
4. Modo explícito de 05 e entradas entram em `taxonomy.curation` (exists): reconstruir a partir da planilha real ou auditar o Parquet existente; apenas o modo smoke explícito permite gerar taxonomia sintética. Validar IDs e ranks antes de publicar nova curadoria (door 3).
5. Notebooks 10/11/05 (exists) apresentam os resultados dos módulos, publicam CSVs e manifestos individuais pelo runtime (exists) e documentam o contrato para consumidores 02/06, sem executá-los.

## Impact

| Front | What changes |
| --- | --- |
| arquivos previstos | Três notebooks de reprodução 10/11/05; módulos `data/retraining_audit.py`, `data/taxonomy_split_audit.py`, `taxonomy/curation.py`; auxiliares de runtime/planejamento necessários à integração M00; configuração M01, configuração M00, testes e documentação. |
| saídas | Cada análise escreve em namespace reservado de `results_reproduction/runs/`, sem migrar ou sobrescrever saídas históricas. Fontes externas são somente leitura. |
| números afetados | Novos relatórios de contagem/qualidade por split/classe e overlap train→test; estratificações por Genus/Family/Order e cobertura de ranks. Nenhuma Accuracy, Top-5, F1, tabela ou figura publicada é recalculada. |
| cadeia científica | Dados/split → auditoria de sequências e rótulos → taxonomia validada → insumos futuros de métricas hierárquicas. Checkpoints, inferência, codificação e métricas permanecem fora desta execução. |
| compatibilidade | Preservar funções legadas quando possível; novo fluxo M01 estrito valida antes de saneamento. Testes N04 que esperam pasta fixa precisam verificar o novo namespace, mantendo suas garantias científicas. |
| referência versus curadoria | Auditar híbrido existente não equivale a reconstruí-lo; os dois modos recebem protocolos diferentes. `NCBI Fallback` histórico não comprova consulta NCBI nesta execução. |

## Relations

Uma rodada possui três análises identificadas, cada uma com um manifesto e seus produtos. A análise 11 consome exatamente uma decisão e um CSV por espécie da mesma execução 10. A análise 05 registra uma origem de curadoria ou uma referência existente, sem substituir a fonte. Cada classe preserva a associação original entre espécie e índice de saída. A validade de cada rank é independente da proveniência genômica; táxons não identificam genomas de origem.

## Surface

None - nenhuma API HTTP ou serviço novo. Interfaces locais dos três notebooks:

- `short/10`: recebe configuração/ambiente de splits, CSV original, limite piloto e diretório temporário, com plano M00 opcional; produz `decision.json`, `per_species.csv`, CSVs de contagens/qualidade por split e `manifest.json`. Manifesto `COMPLETE/FAILED`; decisão científica preserva estados atuais, incluindo `PROVENANCE_REQUIRED` e `INCONCLUSIVE_PARTIAL_TEST`.
- `short/11`: recebe taxonomia existente, CSV original e decisão/CSV explícitos de 10, com plano M00 opcional; produz `taxonomy_audit.json`, `per_taxon.csv`, `per_genus.csv`, `per_family.csv`, `per_order.csv`, relatório de ranks e `manifest.json`. Manifesto `COMPLETE/FAILED`; integridade e adequação hierárquica são resultados separados da conclusão da auditoria.
- `short/05`: recebe modo `curate/audit_existing`, planilha/CSV ou Parquet selecionado e decisão 10 opcional identificada, com plano M00 opcional; produz relatório JSON, CSV de validação/cobertura e `manifest.json`. Parquet novo apenas em `curate` válido ou smoke identificado. Manifesto `COMPLETE/FAILED`; falha de entrada gera exceção e nunca manifesto COMPLETE.

Precedência: seleção explícita do plano M00, quando fornecido, fixa os caminhos e rejeita overrides conflitantes; sem plano, ambiente sobre configuração, seguido da topologia existente. Caminhos relativos de entrada usam DATA_ROOT, saídas usam a raiz do repositório. Preservar os nomes de ambiente já disponíveis (`TAXONOMY_PATH`, `ORIGINAL_MAPPING_PATH`, `OVERLAP_DIR`); overrides de saída ficam sujeitos à reserva exclusiva. Nenhuma descoberta por arquivo mais recente.

Sem plano, o runtime gera um run_id exclusivo. Com plano, adotar o destino reservado para a etapa, recusando uma segunda execução não vazia; o vínculo parent_run_id acompanha o manifesto. O 05 assume `curate`; `audit_existing` exige seleção explícita, sem fallback automático. O modo remoto/empírico continua padrão conforme AD-017; testes locais forçam REMOTE_SERVER=False e selecionam fixtures explicitamente. Piloto limita test e é rotulado sem estimar prevalência do teste completo.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — manifesto individual e adoção M00 | `schema_version=1`, `protocol_id`, `notebook_id=short/10\|short/11\|short/05`, `run_id`, `parent_run_id`, `is_smoke_test`, `status=COMPLETE\|FAILED`, configuração, semente 42, commit/dirty, versões, identidades de entradas/saídas. Publicar COMPLETE por temporário/rename somente após todos os produtos. Reservar novo destino com o runtime; consumir reserva M00 vazia sem sobrescrever produtos. | Pastas fixas e retomada implícita: misturam produtos históricos/novos. Manifestar apenas o Parquet de saída como entrada: perde a proveniência. |
| 2 — população e overlap explícitos | `protocol_id=m01_fragment_split_audit_v1`; contagens integrais train/validation/test, por classe, qualidade e comprimentos; taxa principal `100 * exact_test_matches_train / test_rows`; contagens `direct_matches`, `reverse_complement_matches`, `both_matches`, união sem dupla contagem. `scope=full_test` ou piloto explícito. | Reutilizar percentual do benchmark C03 ou excluir linhas vazias do denominador: população diferente. Comparar todos os pares de splits: custo adicional fora do overlap train→test pedido. |
| 3 — modos de taxonomia e ranks | `m01_taxonomy_split_audit_v1`, `m01_ictv_curate_v1` e `m01_existing_taxonomy_audit_v1` distintos. Validação estrita de `model_label = id - 1`, IDs únicos/conjunto igual ao CSV e nomes originais preservados; `hierarchy_policy=reject_missing` no produto para 02/06. Nulos, strings vazias e placeholders Unassigned/Unclassified são reportados como não resolvidos; não são inventados nem fundidos em táxon biológico. Auditoria pode concluir com `hierarchy_ready=false`, sem publicar Parquet validado. | Reparar índice silenciosamente, preencher Genus/Order com táxon fictício ou remover classes: altera o significado dos logits/população. Promover híbrido auditado a curadoria MSL: origem não demonstrada. |

Hash SHA-256 integral para CSVs, planilha, taxonomia, decisão, manifestos e produtos consumidos. Para fragmentos volumosos, registrar inventário de arquivos/tamanhos e identidade limitada M00 com ausência de hash explícita, sem afirmar identidade criptográfica completa. Campos sem evidência permanecem unknown. Adicionar ao STATE a decisão que formaliza esses contratos depois da aprovação.

## Criteria

### S1: Entradas e execução isoladas (P1)

**Acceptance Criteria**

1. WHEN executa 10, 11 ou 05 THEN o fluxo SHALL resolver e registrar os caminhos selecionados segundo a precedência definida em Surface.
2. WHEN executa duas análises sem plano THEN o runtime SHALL produzir destinos distintos sem alterar fontes ou saídas históricas.
3. WHEN executa uma etapa selecionada do plano M00 THEN o fluxo SHALL usar seus run_id, parent_run_id e destino reservado vazio.
4. IF destino contém produtos, há override conflitante ou falta entrada empírica obrigatória THEN o fluxo SHALL falhar antes de publicar manifesto COMPLETE.
5. WHEN conclui cada análise THEN o fluxo SHALL publicar manifesto individual com protocolo, modo, configuração, versões e identidades das entradas e produtos reais.

**Independent test:** reservas temporárias, plano M00 mínimo, sentinelas de referências, duas execuções e tentativa repetida, override conflitante e entrada ausente. Conferir hashes antes/depois.

### S2: Contagens e overlap no split correto (P1)

**Acceptance Criteria**

6. WHEN audita splits completos THEN a análise 10 SHALL registrar N por split/classe, distribuição de comprimentos e linhas vazias após remover padding para train/validation/test.
7. WHEN calcula overlap de test contra train THEN a análise 10 SHALL registrar correspondências diretas, RC, interseção e união sem dupla contagem.
8. WHEN calcula a taxa principal THEN a análise 10 SHALL usar todas as linhas de test no denominador, inclusive vazias e sem A/C/G/T, reportando o subconjunto compatível separadamente.
9. IF faltam IDs genômicos confiáveis por fragmento em auditoria integral THEN a decisão 10 SHALL manter `PROVENANCE_REQUIRED`.
10. IF há limite de linhas de test THEN a decisão 10 SHALL registrar `INCONCLUSIVE_PARTIAL_TEST` e o limite aplicado.
11. WHEN emite o relatório 10 THEN o fluxo SHALL identificar overlap de fragmentos train→test como população distinta das consultas C03.

**Independent test:** fixture com padding, vazio, N-only, IUPAC, >512 bp, correspondência só direta, só RC e ambas; contagens esperadas à mão e classes ausentes de um split. Verificar execução em lotes sem carregar o dataset completo.

### S3: Taxonomia existente e identidade dos logits (P1)

**Acceptance Criteria**

12. WHEN audita o híbrido THEN a análise 11 SHALL detectar IDs duplicados/ausentes/extras, model_label divergente de id-1 e nomes originais alterados contra o CSV selecionado.
13. IF decisão/CSV de 10 têm totais incompatíveis, identidade divergente ou escopo piloto THEN a análise 11 SHALL rejeitar a estratificação como auditoria integral.
14. WHEN faltam Genus/Order ou existem placeholders THEN 11/05 SHALL registrar labels afetados e `hierarchy_ready=false`, sem preencher ranks ou remover classes silenciosamente.
15. WHEN estratifica overlap por rank THEN a análise 11 SHALL conservar o denominador completo e reportar separadamente os rótulos com rank não resolvido.
16. WHEN audita o Parquet existente em 05 THEN o fluxo SHALL emitir `m01_existing_taxonomy_audit_v1`, preservando a fonte e sem afirmar reconstrução pela planilha MSL.

**Independent test:** CSV original e Parquet mínimo com label trocado, duplicatas, ID extra, nome alterado, rank nulo/placeholder e decisão adulterada. Reconciliar soma por grupos com N integral.

### S4: Curadoria empírica sem fallback sintético (P1)

**Acceptance Criteria**

17. IF falta a planilha ICTV no modo empírico `curate` THEN o notebook 05 SHALL levantar FileNotFoundError sem produzir Parquet sintético ou manifesto COMPLETE.
18. IF faltam abas/colunas obrigatórias na planilha ou o CSV de classes é inválido THEN o fluxo 05 SHALL rejeitar a curadoria antes de publicar Parquet validado.
19. WHEN conclui `curate` com identidade de classes e Genus/Order válidos THEN 05 SHALL publicar novo Parquet sob `m01_ictv_curate_v1` com a planilha/CSV e eventual fallback NCBI local identificados.
20. WHILE executa smoke explicitamente THEN o fluxo 05 SHALL identificar seus produtos/manifestações como sintéticos, mantendo essa origem em dependências de 11.
21. WHEN falta evidência de accession por fragmento THEN o relatório 05 SHALL manter proveniência genômica desconhecida, inclusive quando a taxonomia estiver válida.

**Independent test:** executar células de 05 com runtime temporário em modo empírico e planilha ausente; planilha mínima válida/inválida; impedir rede e geração sintética no caso empírico; executar smoke offline separado.

### S5: Uso documentado e limites da evidência (P1)

**Acceptance Criteria**

22. WHEN carrega artefatos persistentes nos fluxos M01 THEN o código SHALL registrar seus caminhos por LOG_PATH, respeitando LOG_PATH=0/false.
23. WHEN documenta M01 THEN a documentação SHALL fornecer comandos/parâmetros exatos para 10→11→05, caminhos dos manifestos e distinção entre smoke local, preflight remoto e validação empírica.

**Independent test:** capturar logs de fixtures e executar validação local dos notebooks/configuração. Preflight do cluster e execução integral serão evidências separadas, não inferidas dos testes locais.

## Out of scope

| Excluded | Why |
| --- | --- |
| Retreinar, alterar split ou recuperar accessions por nome de espécie | M01 audita a evidência existente; nova população exige novo experimento. |
| Recalcular C03, métricas/embeddings 02/06 ou tabelas do artigo | Consumidores posteriores dependem de seleção e validação empíricas próprias. |
| Refazer consultas NCBI ou instalar bases | Sem rede automática; fallback só se fornecido como artefato local identificado. |
| Busca por quase duplicatas ou overlap de todos os pares de splits | E01 é outra análise; M01 registra contagens dos três splits e overlap test contra train. |
| Corrigir algoritmo histórico de fuzzy matching e saneamento global | O fluxo novo exige validação estrita e registra limitações; mudar heurísticas taxonômicas requer reconciliação própria. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Perfil | light | Padrão da skill; verificação independente após implementação. | n |
| Planilha ausente nesta máquina | Auditar referência existente somente quando explicitamente selecionada | A ausência local não demonstra ausência no cluster; evita fallback implícito. | n |
| Fragmentos volumosos | Inventário de identidade limitada M00, hash integral apenas quando disponível/configurado | Evita leitura extra irrestrita; identidade incompleta permanece explícita. | n |

**Open questions:** none — defaults acima para revisão do plano. A aprovação autoriza a implementação e commits locais conforme a skill; não autoriza processamento integral no cluster.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Notebooks 10/11/05 | Entradas, defaults, precedência e resolução remota/local | Surface; Criteria 1, 3–4, 17–20 |
| Produtos por execução | Nomes, agrupamento por run/etapa, origem e tratamento de colisão | Landing 1–3; Criteria 2–5 |
| Relatórios de split | Contagem, piloto, denominador e ordenação por split/label | Criteria 6–11 |
| Taxonomia e estratificações | Identidade, ranks ausentes, proveniência e dependências inválidas | Criteria 12–21 |
| Erros | Exceção com entrada/causa, nenhum COMPLETE após falha | Surface; Criteria 4, 13, 17–18 |
| Progresso e log | Caminhos e resumo persistido, sem saída por fragmento | Criteria 5, 22 |
| Documentação | Parâmetros/comandos, manifestos e evidência ainda não executada | Criterion 23 |
| Tela/API | Estados visuais, autenticação e rate limit | n/a — notebooks locais sob permissões do sistema, sem API/serviço novo |
| Ação destrutiva | Confirmação de exclusão/migração | n/a — referências somente leitura e destinos exclusivos |

Nove dimensões para derivar em checks: validação (4, 12–14, 18); falha parcial (4–5, 17–18); idempotência/duplicatas (2–4); autorização/rate limits (n/a — execução local sem serviço); concorrência/ordenação (reserva exclusiva existente e dependência identificada 3, 13); ciclo de dados (2, 5, 16); falha de dependência externa (4, 13, 17); transições (5, 17–20); observabilidade (5, 22–23).

## Sources

- Pedido do usuário: “usando tlc-spec-lean vamos implementar m01”; `ROADMAP_MELHORIAS_REPRODUCTION_2026-11-09.md`, contratos comuns/M01; `.specs/STATE.md`, AD-001–028 e M00 verificada.
- `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `docs/dataset_manifest.md`; notebook histórico `short/ICTV_improve_kraken_database.ipynb` (MSL41.v1 e saneamento histórico preservados). `notebooks/original/` não existe nesta árvore; não criar cópia nem mover históricos.
- Inspeção dos três notebooks e módulos citados: CSV Species com 13.683 linhas, SHA-256 `51002670cf805a448c01b6d2c65c3dfb4038e020277f238c0a4719155c216517`; CSV Genus com 2.702 linhas, SHA-256 `14e0859f43dc29401f2aca957d615c910801037d61df7ec9b88060fbbe56fb8d`. Planilha `short/ICTV_Master_Species_List_2025_MSL41.v1.xlsx` e diretório `cnn_kraken/data_taxonomy_parquet/` ausentes localmente. Há Parquets locais train/test/val, cuja origem canônica não foi demonstrada; sua presença não comprova disponibilidade dos splits reais. Checkpoints não são necessários a M01 e não foram carregados. Divergências de títulos e dropout entre README/configs/artigo permanecem fora deste escopo.
