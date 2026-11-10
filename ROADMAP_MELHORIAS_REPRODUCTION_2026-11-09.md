# Roadmap pré-execução dos notebooks de `reproduction`

**Base inspecionada:** `cnn_metagenome`, commit `1285292` em 09/11/2026; notebooks e código locais, inventário `c06_candidate_inventory_v2.csv` e prévias dos artefatos executados no JupyterHub. **Limite:** os Parquets volumosos, checkpoints e saídas completas do cluster não estão montados nesta cópia. Valores observados em prévias servem para identificar divergências, não para certificar novos resultados.

## Objetivo e decisão de escopo

Preparar uma nova execução empírica integral, rastreável e imutável; gerar as evidências do artigo e aproveitar a rodada para testar hipóteses adicionais que respondam à revisão. Manter os notebooks históricos, os resultados anteriores e os checkpoints intactos. A análise é **closed-set com split por fragmentos**. Sem IDs confiáveis de genoma por fragmento, não anunciar teste em genomas inéditos nem transformar o de-para taxonômico em proveniência genômica.

Implementar as fases **M00–M07 antes da execução integral**. A fase **M08** conclui a reconciliação após os resultados. As extensões E01–E04 são condicionadas à evidência e ao orçamento computacional. Não repetir `03/04` de fine-tuning por padrão: avaliá-los apenas se a pergunta científica exigir pesos novos e houver plano de treinamento com orçamento e linhagem.

### Diagnóstico que orienta o plano

| Área | Já existe | Pendência específica antes da nova rodada |
| --- | --- | --- |
| Execução/manifestos | `reserve_run_directory`, preflight, C06 estrito; `01/02/06/06b/09` e `long/01–03` já isolam boa parte das saídas | `short/05/07/08/10/11` e `long/04` ainda usam caminhos fixos; manifestos 07/08 não satisfazem C06; a ordem e as versões das dependências 07→08 não são fixadas. |
| Taxonomia e split | Auditorias 10/11 apontam ~95,94% de overlap exato/RC e proveniência genômica ausente | `short/05` tem fallback sintético se falta a planilha ICTV no modo empírico; necessidade de apontar arquivo de origem, política de gêneros ausentes e denominadores. |
| Benchmark | C03 export, bases, manifests e avaliações pareadas têm implementação | A primeira célula do `short/09` força `DATA_ROOT`, `SMOKE_TEST=0` e um `C03_PREPARED_DIR` de execução anterior. Validar as 35 condições (7 métodos × 5 ruídos) e a identidade das queries antes do notebook. |
| Interpretação | `short/07` gera T8 e uma figura com dois painéis | C06 pede F3 de cinco modelos; o painel de barras mede presença de checkpoints, não efeito. |
| Tripletos/coocorrência | `short/08` gera T9/T10 e teste de permutação; a prévia de F5 usa 37 arestas, p≈0,46953 | Fundo é formado pelas primeiras 1.000 sequências lidas; T11 atual é correlação entre arquiteturas, mas a tabela do notebook é coocorrência de um modelo. Correlações legadas r=1 decorrem de matrizes idênticas/degeneradas. C06 espera uma rede para F5; o notebook gera barras. |
| Figuras/resultados históricos | `paper/` e geradores de tabelas/figuras contêm versões anteriores | `paper/manuscript_submission.tex` afirma T11 significativa com números fixos em `reporting/tables.py`; `paper/tables/table_11.csv` contém outra medida (r=0, p=1). Não promover nenhuma dessas versões por nome ou data. |

## Contratos comuns a todas as fases

1. **Identidade:** cada execução recebe `run_id`, `population_id`, `split_id`, Git commit e sinalização de árvore modificada; cada produto guarda protocolo, transformação, código/configuração, entradas e hashes, versão da taxonomia, ordem das classes, checkpoints e SHA-256, semente, limites de amostra, contagens elegíveis/excluídas e status. `unknown` permanece `unknown` quando necessário. Registrar o caminho absoluto no log e o vínculo relativo dentro do manifesto.
2. **Saídas:** escrever em `results_reproduction/runs/<run_id>/<etapa>/`; nunca usar a pasta fixa de uma execução antiga como destino. Escrita temporária seguida de substituição atômica apenas *dentro do novo namespace*; falhas ficam `FAILED`, saídas parciais não viram `COMPLETE`. Resolver dependências pelo manifesto explícito do run anterior, não por “último arquivo” ou relógio do sistema.
3. **Modos:** `SMOKE_TEST=1` usa fixture e marca todas as saídas como sintéticas. `SMOKE_TEST=0` exige todos os arquivos reais, sem fallback. Pré-visualizar e testar esquema/cabeçalhos antes de processar milhões de fragmentos. C06 rejeita qualquer dependência sintética transitiva.
4. **Métricas:** manter 512 bp, alfabeto A/C/G/T, ordem de classes, reamostragem de base inclusive a original, denominador, políticas de abstenção e a regra de ajuste de temperatura em validação. Expansões que mudarem a população, a quantidade de ruído ou o estimando devem ter **novo protocolo e arquivo**, preservando o resultado principal comparável ao histórico.
5. **Publicação:** CSV/Parquet de dados é a origem regenerável da figura; PNG/PDF é derivado. Manifestações por artefato para T1–T12 e F1–F5 só depois de verificar seus hashes e dependências reais. Uma figura de exploração pode ir ao suplemento sem ocupar à força um identificador F.

## Fases de implementação antes de reexecutar

### M00 — Congelar referência e criar plano executável do run

**Arquivos:** `src/viral_cnn/notebook_runtime.py`, `scripts/preflight_notebook.py`, um preflight/orquestrador leve em `scripts/`, README e configs. **Dependências:** nenhuma. **Esforço estimado:** 5–8 h.

- Inventariar o estado de `paper/`, `results_reproduction/` e os caminhos externos **sem copiar binários volumosos**; criar `reference_inventory.csv` com caminho, tamanho, hash quando viável e status de origem. Manter a cópia histórica somente para comparação.
- Gerar `run_plan.json` com os IDs de run de cada etapa, dependências 07→08, 06→06b, C03→09 e long/01→02→03→04, recursos esperados, e `skip` justificado para 03/04 quando usar checkpoints existentes. O preflight deverá fazer inspeção de esquema/metadados e enumerar entradas faltantes por notebook.
- Padronizar `run_id` por notebook e vínculo `parent_run_id`; distinguir reservas novas de retomada verificada (06/06b). Permitir selecionar runs anteriores **somente** como entrada identificada e imutável.
- **Aceitação:** dois runs de teste não se sobrescrevem; falta de arquivo empírico interrompe antes de gerar resultado; o plano informa exatamente qual manifesto C03/07/06 será lido. Teste de hash/semântica em fixture pequena e preflight no servidor antes do processamento completo.

### M01 — Dados, taxonomia e split (`10`, `11`, `05`)

**Arquivos:** `short/10_decisao_retreinamento_por_genoma.ipynb`, `short/11_auditoria_taxonomia_ictv_split.ipynb`, `short/05_ictv_taxonomy_curation.ipynb`, `src/viral_cnn/data/{retraining_audit,taxonomy_split_audit}.py`, `src/viral_cnn/taxonomy/curation.py`. **Depende de:** M00. **Esforço:** 6–10 h.

- Tirar 10/11/05 das pastas fixas; passar caminhos de entrada por configuração ou ambiente, principalmente taxonomia ICTV sob `cnn_kraken/data_taxonomy_parquet/`, o CSV original de classes e decisão 10. Separar *taxonomia existente de referência* de *nova curadoria*: não tratar o híbrido existente como se tivesse sido recriado a partir da planilha MSL sem comprovação.
- Corrigir em 05 o `if ctx.is_smoke_test or not ictv_master_path.exists()` para falhar quando falta a planilha em modo empírico; se a planilha realmente não estiver disponível, executar o caminho de **auditoria do Parquet existente** com protocolo próprio, sem afirmar que o notebook o reconstruiu. Verificar `Genus`/`Order` ausentes e definir exclusão/mapeamento explícito antes do 06; nunca preenchê-los silenciosamente.
- Registrar N por split/classe, sequências vazias após padding, comprimentos, overlap direto e reverso complementar; distinguir overlap entre fragmentos do teste de overlap do benchmark C03. Fazer checagem de IDs de classes, duplicações e rótulos alterados.
- **Aceitação:** manifestos e CSVs individuais para as três análises; `PROVENANCE_REQUIRED` segue válido quando não há genome ID; taxa corrigida usa denominador completo; teste de planilha ausente em modo empírico falha e não produz Parquet “real”.

### M02 — Benchmark, intervalos e calibração (`01`, `02`, `09`)

**Arquivos:** `short/{01,02,09}_*.ipynb`, `configs/baseline_c03.yaml`, `src/viral_cnn/eval/`, `src/viral_cnn/baselines/`, scripts C03. **Depende de:** M00/M01; C03 completo antes do 09. **Esforço:** 7–12 h, sem incluir inferência pesada.

- Remover a célula que injeta três variáveis em 09. Ler `C03_PREPARED_DIR` do ambiente ou plano do run; imprimir caminho e hashes; exigir `preparation_manifest.json`, `database_build_manifest.json` e os manifestos **COMPLETE** de 35 condições, com mesmas `query_id`/sequência/rótulo por taxa, mesma ordem de classes e política de abstenção. A etapa 09 não gera os resultados de Kraken2/BLASTN nem CNN: essas execuções são pré-requisitos.
- Preservar os ICs, a calibração ECE/Brier/NLL e os diagramas de confiabilidade já implementados em 01/02; testar que as figuras contêm N e que temperatura foi selecionada apenas na validação. Explicitar se IC é por bootstrap de fragmentos/espécies e se há **uma** realização de ruído ou várias; não interpretar IC de teste como variabilidade entre treinamentos.
- **Correção separada:** a célula GradCAM em 02 usa `torch.randint` mesmo no modo empírico; movê-la para smoke ou selecionar fragmentos reais identificados, registrando `query_id`, classe e checkpoint. Não usar essa visualização aleatória como evidência biológica.
- **Nova figura valiosa:** curvas de acurácia e cobertura por ruído para os sete métodos, com três painéis `all`, `overlap`, `no_exact_overlap`, N e IC pareado quando disponível. Outra figura compacta: diferença pareada CNN–BLASTN/Kraken2 por ruído e estrato com IC; evita superpor curvas demais. Dados tabulares devem preservar os métodos não mostrados por legibilidade.
- **Aceitação:** notebook 09 sem caminho de execução codificado; preflight detecta manifesto ausente/incompleto e IDs discordantes; saídas distinguem acurácia global, cobertura e acurácia condicional; testes pequenos para abstenção e igualdade de queries.

### M03 — Espaço latente e visualizações úteis (`06`, `06b`)

**Arquivos:** `short/{06,06b}_*.ipynb`, `src/viral_cnn/{paired_umap,latent_plotting,latent_resume}.py`. **Depende de:** M01 e checkpoints identificados. **Esforço:** 4–7 h.

- Preservar correções de sonda linear, KNN com memória limitada, cache de embeddings e UMAP pareado; validar em preflight os hashes treino/validação/teste, taxonomia, a seleção determinística por gênero e o denominador de cada métrica. Registrar se o 06b reutiliza embeddings do 06 ou os recalcula. Usar mesmo conjunto de fragmentos nos cinco modelos, projetando cada espaço separadamente.
- **Manter** o painel UMAP completo atual e o HTML explorável. **Adicionar** um painel de gêneros selecionados por regra prévia (ex.: quota mínima e estratos de suporte/erro, semente fixa), destacando apenas alguns gêneros e cinza para os demais. Exportar `selected_genera.csv`, `sampled_query_ids.csv`, contagens por gênero, método UMAP/PCA e parâmetros; não escolher gêneros pela aparência dos gráficos.
- Se houver valor real, acrescentar figura **quantitativa no espaço original** com KNN, Silhouette, cobertura de classes válidas e sonda linear; UMAP permanece qualitativo e seus eixos não são alinhados entre modelos.
- **Aceitação:** painel completo e seletivo usam IDs alinhados; figuras se regeneram dos CSVs; execução com gêneros faltantes falha/declara exclusão, não inventa taxonomia. Nenhuma afirmação de filogenia ou separação global sai do UMAP.

### M04 — Motivos e F3 (`07`)

**Arquivos:** `short/07_first_layer_motifs_and_perturbation.ipynb`, `src/viral_cnn/interpretability/motifs.py`, geradores de figura. **Depende de:** M00/M01 e cinco checkpoints. **Esforço:** 7–12 h.

- Criar diretório exclusivo para 07, registrar checkpoint e hash de **cada** variante, seleção dos até 10 motivos/variante, critérios de ocorrência, N elegível/ocorrências e teste emparelhado; guardar resultados por sequência/ocorrência (ou resumos suficientes para IC) antes de agregar. Não substituir um controle GC por um controle de posição diferente sem novo protocolo.
- Exportar `motif_perturbation_by_variant.csv` com `panel`, `variant`, `motif`, `mean_random_drop_uniform`, `mean_motif_drop_uniform`, `mean_specific_drop_uniform`, equivalentes GC, FDR, suportes e intervalos adequados à unidade de pareamento. Reconciliar explicitamente o nome das colunas esperado por C06; hoje o renderizador exige `panel/variant/motif/activation_drop` e a T8 contém colunas diferentes.
- **F3 principal:** cinco painéis, um por modelo, com efeito do motivo versus controle aleatório da mesma sequência; cor/forma para significativo sob controle GC e indicação de N. **Figura adicional:** dispersão uniforme × GC já produzida pelo 07, agora por variante e com diagonal e limiares explicados. O gráfico de barras `status=1` dos checkpoints sai da figura científica; a identidade fica no manifesto/legenda.
- **Aceitação:** painel (e) usa ConvNeXt_F verificado por hash e apresenta seus próprios pontos; troca de checkpoint invalida artefato; dados tabulares reproduzem todas as posições da F3; nenhum motivo é descrito como sítio funcional sem evidência independente.

### M05 — Tripletos, T11 e F5 (`08`)

**Arquivos:** `short/08_triplet_composition_and_cooccurrence.ipynb`, `src/viral_cnn/interpretability/motifs.py`, C06 e manuscrito na M08. **Depende de:** M04 (07 identificado) e M01. **Esforço:** 7–12 h.

- Consumir `motif_summary.parquet` e matrizes Jaccard **do manifesto 07 selecionado**, não da pasta fixa. Substituir as primeiras 1.000 linhas do teste por amostragem determinística de arquivos/fragmentos com cobertura conhecida (ou censo em streaming se barato); fixar seleção e declarar que fundo e motivos vêm de populações distintas, quando for o caso. Reportar sensibilidade às 3 molduras, 6 molduras e composição de bases. Rotular o resultado como composição exploratória de tripletos, pois não há ORF verificada.
- **T11 — decisão obrigatória antes de codificar:** (A) retirar a antiga inferência entre arquiteturas e usar uma tabela descritiva de overlap de motivos/matrizes, com título e transformação novos; ou (B) definir uma nova medida entre modelos baseada em observações independentes, unidades válidas, nulos por permutação e FDR antes de calcular. Não publicar `table_11b_jaccard_variant_correlations.csv`: pares com matrizes idênticas ou sem variância não suportam r=1, p=0. `table_11_cooccurrence_synergy.csv` responde a outra pergunta e não preenche T11 atual.
- **F5 — decisão vinculada à T11:** preferir gráfico pequeno com correlação bidirecional, ICC, n de arestas e p por permutação, rotulado exploratório. A prévia observada (37 arestas válidas; r≈0,1204; p≈0,4695) não sustenta “sinergia” significativa. Se o manuscrito exigir rede, exportar `edge_table.csv` com IDs de motivos, Jaccard e coordenadas *derivadas e documentadas*; a tabela resumo sozinha não satisfaz `network_edges`. Manter valor de p no texto/legenda, sem tratar arestas dependentes como observações independentes.
- **Nova visualização útil:** distribuição de escores dos nós e das arestas versus permutações (nulo empírico), com estatística observada marcada; ajuda a ver que o valor de p inclui a topologia e o grau, sem prometer causalidade.
- **Aceitação:** 08 falha se o 07 selecionado estiver incompleto; `n_valid_edges` e total elegível são declarados; teste reproduz a regra simétrica e mostra que a inversão das pontas não altera a estatística; T11 e F5, suas legendas e `publication_c06.yaml` contam a mesma história.

### M06 — Sequências longas e análise de estratos (`long/01–04`)

**Arquivos:** `notebooks/reproduction/long/{01,02,03,04}_*.ipynb`, `src/viral_cnn/inference/`. **Depende de:** M00/M01 e cinco checkpoints. **Esforço:** 5–9 h.

- Manter o protocolo C02 já implementado para 01–03: cada previsão tem `query_id`, contig/fragmento, estratégia de agregação, N e qualidade da correspondência de classes. Conferir seleção de checkpoint e causa de exclusões antes de comparar vote e mean logits. Estatísticas por comprimento, GC, número de janelas e táxon devem indicar suporte e a seleção de contigs; mecanismos propostos para queda de ConvNeXt permanecem hipóteses se não testados.
- Atualizar long/04: hoje escolhe o **primeiro contig** e quatro motivos fixos; amostrar exemplares por critérios predefinidos, usar motivos obtidos de 07 e salvar IDs, origem, posições das janelas e contagens. Não vender um caso ilustrativo como tendência global. Tratar comprimento inferior a 512 e bases ambíguas conforme o mesmo protocolo 01–03.
- **Figuras adicionais:** acurácia/erro por faixa de comprimento e nº de janelas com N; efeito pareado de voto versus média de logits por variante. Perfil GC/motivos por contig como suplemento, com seleção documentada. Falar em *sequências longas/contigs*, salvo evidência de leituras reais de sequenciador.
- **Aceitação:** T12 usa população pareada nos cinco modelos ou registra impossibilidade; análise por estrato não infere causa sozinha; 04 não depende do “primeiro arquivo” para escolher um exemplo empírico.

### M07 — Contratos de publicação e validação local (antes do cluster)

**Arquivos:** `src/viral_cnn/reporting/{publication,tables,figures}.py`, `configs/publication_c06.yaml`, `00_regenerate_all_paper_artifacts.ipynb`, testes científicos focados. **Depende de:** decisões M04/M05. **Esforço:** 6–10 h.

- Criar transformações específicas para as fontes realmente produzidas (T1–T12, F1–F5) em vez de associar um nome semelhante à saída. C06 hoje começa com todos os artefatos, manifestos, protocolos, hashes e identidade vazios; preencher a seleção **após a execução empírica**, mas testar desde já as transformações e rejeições com fixtures.
- Alinhar F3 a cinco painéis e F5 à escolha de gráfico; o renderizador C06 consome CSV/Parquet e **não aceita PNG do notebook como fonte**. Estabelecer contratos de colunas, unidades, esquema de status, N e seleção por figura. Para cada CSV de publicação emitir manifesto derivado que vincule sua entrada 07/08/09 ou anterior e o código de transformação sem simular proveniência inexistente.
- Criar diagnóstico em lote (`T1–T12`, `F1–F5`) que exiba `OK | FALTA | SCHEMA_MISMATCH | IDENTITY_MISMATCH | SCIENTIFIC_MISMATCH`, com fonte proposta e recomendação; modo estrito falha claramente. Manter o histórico e as constantes do manuscrito apenas na coluna *histórico*, nunca como fallback.
- **Aceitação:** testes negativos recusam artefato sintético, hash diferente, fonte incompleta, F3 sem cinco modelos, T11 rotulada como correlação entre arquiteturas com CSV de coocorrência, F5 de barras quando configurada como rede; geração das figuras a partir de tabelas pequenas testadas e figuras legíveis.

## Extensões com valor científico, sob condição

| Extensão | Valor e pergunta | Gatilho/limite | Produto |
| --- | --- | --- | --- |
| **E01 — quase duplicatas entre splits** | Até que ponto resultados variam com semelhança de sequências que não são correspondências exatas? | Só após M01. Usar amostra estratificada, método explícito de k-mers/MinHash/alinhamento, reverse complement e limiar pré-fixado; custo de busca medido antes de escalar. **Não** chamar isso de split por genoma. | Curva de acurácia ou overlap por faixa de similaridade, N, taxa de cobertura e limitação da busca. |
| **E02 — realizações independentes de corrupção** | A curva de robustez depende da realização do ruído? | Reutilizar mesmas queries/checkpoints; rodar poucas sementes pré-definidas para cada ε; novo ID de protocolo; execução cara só se o benchmark principal estiver estável. | Média e dispersão entre realizações de corrupção, separadas dos ICs por reamostragem de fragments e da variabilidade de treino. |
| **E03 — baseline de composição real** | Um classificador simples de k-mers explica parte do ganho de motifs/arquitetura? | `short/09` implementa k-mer apenas para smoke; treinar fora de smoke **somente** sobre dados de treino, com limite de memória e as mesmas queries/estratos; não fazer avaliação sobre o próprio treino. | Comparação pareada e curva de cobertura/acurácia por ruído com mesmo N; suplemento se o custo exceder o ganho. |
| **E04 — sensibilidade das visualizações** | Os achados qualitativos 06b/F3/F5 mudam com amostra ou seleção? | Após figuras principais; repetir quotas/sementes de amostragem sem usar nova projeção para inferir desempenho; reusar embeddings/caches válidos. | Diagnóstico de estabilidade e critérios de seleção publicados, sem multiplicar painéis no texto principal. |

## Ordem de execução após implementar M00–M07

| Etapa | Executar/selecionar | Portão antes de avançar |
| --- | --- | --- |
| 0 | Preflight M00; verificar hardware, ambiente, caminhos reais, versões, dados/checkpoints, espaço e planilha ICTV | `READY`; plano e namespaces reservados; nenhuma configuração sintética ou caminho antigo hardcoded. |
| 1 | `short/10` → `short/11` → `short/05` (ou auditoria do híbrido existente) | Proveniência genômica declarada, split/labels reconciliados, `Genus/Order` válidos para 06/02; sem recriação ICTV fictícia. |
| 2 | `short/01` → `short/02`; em paralelo lógico ao fluxo latente, C03 `export` → `build-databases` → 35 predições/manifests → `short/09` | N, ordem de classes, ruído e consultas verificados; temperatura provém da validação; 09 só inicia com C03 completo. |
| 3 | `short/06` → `short/06b` | Embeddings e figuras reutilizados somente após hash/identidade; métricas quantitativas independentes do UMAP. |
| 4 | `short/07` → `short/08` | Cinco checkpoints, CSVs e F3; 08 usa exatamente a saída selecionada de 07; T11/F5 têm decisão metodológica registrada. |
| 5 | `long/01` → `long/02` → `long/03` → `long/04` | Cinco previsões alinhadas, manifestos COMPLETE e diagnósticos rotulados; download 01 só quando dados ainda não estiverem disponíveis de modo verificável. |
| 6 | `00_regenerate_all_paper_artifacts` em diagnóstico → selecionar fontes verificadas em C06 → montagem estrita → reconciliação do manuscrito | Matriz dos 17 itens sem substituição silenciosa; histórico e novo resultado lado a lado; texto/captions coerentes. |

## M08 — Concluir artigo e rastrear diferenças (após os dados)

**Arquivos:** `paper/manuscript_submission.tex`, `paper/manuscript_review.tex`, C06, `docs/` e tabela de reconciliação. **Depende de:** execução acima. **Esforço:** 6–12 h, mais revisão científica.

- Para cada T/F, registrar `resultado histórico → resultado novo → diferença → N → protocolo → fonte → explicação ou status INCONCLUSIVE`. Atualizar também título, resumo e limitações quando o nível de evidência mudar. T11 e F5 exigem troca de afirmação; F3 exige legenda dos cinco checkpoints e do controle GC.
- Conferir tabelas de contagem, percentuais e arredondamento; não alterar número manualmente para coincidir com PDF. Gerar pacote PDF/TeX em diretório novo e inspecionar páginas/legendas. A montagem estrita falha se qualquer item requerido não tiver evidência; uma seção suplementar explicitamente incompleta pode existir sem ser rotulada como pacote concluído.

## Priorização e tempo

**Caminho crítico de código:** M00 → M01 → M04 → M05 → M07; M02 precisa estar concluída antes do 09, e M03/M06 podem avançar com os respectivos insumos prontos. Os tempos de M00–M07 somam aproximadamente **47–80 horas de implementação/revisão**, sem contar inferência no cluster, geração de bases ou treinamento. Com 8–12 h semanais, reservar várias semanas; reduzir escopo deixando E01–E04 e retreinamento para uma segunda rodada. É uma estimativa de esforço, não garantia de duração de execução.

**Primeiro pacote curto a implementar:** M00 + as duas correções de segurança de 05/09 + isolamento de saídas 07/08/10/11/long04 + contratos de entrada 07→08. Assim a reexecução deixa de misturar evidência nova e histórica. O pacote seguinte resolve F3/T11/F5 e C06 antes de gastar o processamento integral.

### Prompt reutilizável para uma spec

> Leia `AGENTS.md`, `.specs/STATE.md`, este roadmap e a implementação atual. Implemente apenas **MXX**, mantendo saídas antigas intocadas. Antes de codificar, registre em `plan.md` os arquivos, a população científica e as dependências; em `checks.md`, os contratos e ao menos um caso negativo. Implemente no módulo `src/` e mantenha o notebook como interface documentada. Em `verification.md`, separe testes locais sintéticos, preflight remoto e validação empírica real, com caminhos e manifests. Atualize apenas as alegações sustentadas pelos artefatos verificados.
