# Plano de correção antes da próxima execução empírica

**Repositório:** [Mathe13/cnn_metagenome](https://github.com/Mathe13/cnn_metagenome)  
**Base examinada:** `main` em `f4277239e1d2d2ca162d50932696d9d7bb3d8ff1`, 15/11/2026.  
**Objetivo:** fazer as saídas dos notebooks de reprodução alimentarem T1–T12 e F1–F5 sem perda de proveniência, valores inventados ou interpretações científicas além das evidências. A execução empírica completa vem depois das correções e dos testes de integração locais.

## Resultado esperado

1. Cada item de publicação aponta para um CSV/Parquet produzido ou transformado, um manifesto e uma identidade científica verificável.
2. Um teste integrado gera saídas realistas de todas as etapas, aplica as transformações e executa o validador C06 estrito. Itens intencionalmente pendentes são relatados como pendentes; nenhum deles é aceito por uma exceção silenciosa.
3. Tabelas e figuras do artigo usam apenas números observados e suporte experimental com denominadores explícitos.
4. Uma nova rodada empírica cria um `run_id` próprio, preserva os resultados anteriores e termina com uma matriz de evidências auditável.

## Regras para implementação com IDEs agênticas

- Criar **uma spec por pacote** abaixo, com requisitos observáveis, exemplos de entrada e saída, plano de arquivos, testes positivos e negativos. Implementar em ordem, com commits pequenos. Não usar vários agentes editando o mesmo módulo simultaneamente.
- Congelar o commit de partida. Antes de editar, conferir o código atual, `AGENTS.md`, as specs M00–M07 e a documentação do protocolo C06; se `main` avançou, reavaliar os trechos afetados.
- Preferir adaptadores explícitos entre a saída científica do notebook e o esquema de publicação. Não renomear colunas em todos os notebooks só para satisfazer C06 sem revisar o significado das métricas.
- Não inserir `0`, `unknown` ou `N=len(df)` como substitutos de medições ou denominadores. `unknown` é legítimo para accession/genoma não recuperável, desde que impeça afirmações de independência genômica.
- Não usar arquivos históricos como fallback. Um item que não satisfaça o contrato deve falhar com erro específico e diagnóstico acionável.
- Testes locais usam fixtures pequenas; `SMOKE_TEST` nunca produz evidência empírica. Não disparar treinamento novo por padrão. Preservar `results_reproduction/runs/<run_id>` e manifestos anteriores.
- Ao fechar cada spec, registrar: comando de teste, saída, arquivos alterados, exemplo real de esquema consumido, pendências científicas e impacto em `configs/publication_c06.yaml`.

## Ordem de dependências

| Spec | Prioridade | Entrega | Depende de |
|---|---|---|---|
| C01 | Bloqueante | Matriz executável de contratos T/F | — |
| C02 | Bloqueante | Proveniência de derivados aceita por C06 | C01 |
| C03 | Bloqueante | Adaptadores T1–T10, T12 e F1/F2/F4 | C01–C02 |
| C04 | Bloqueante científico | Decisão e contrato de T11 | C01–C02 |
| C05 | Alta | Verificação final de F3/F5 e legendas | C02–C04 |
| C06 | Bloqueante | Teste integrado de ponta a ponta e configuração selecionável | C01–C05 |
| C07 | Operacional | Reexecução empírica, curadoria e pacote final | C06 |

### C01 — Inventário e contratos de dados

**Problema:** propostas de fontes e colunas em `configs/reproduction_m07.yaml` não correspondem a diversas saídas dos notebooks; os testes atuais fornecem tabelas sintéticas já no formato esperado.

**Implementar:**

1. Criar uma matriz versionada para T1–T12/F1–F5 com: notebook produtor, caminho relativo no run, manifesto pai, colunas e unidades **realmente exportadas**, cardinalidade, definição do denominador, transformação, colunas publicadas e destino C06.
2. Fazer o preflight ler os cabeçalhos das saídas reais disponíveis ou fixtures extraídas dos exportadores. Diferenciar `arquivo ausente`, `esquema incompatível`, `unidade ambígua` e `proveniência incompleta`.
3. Corrigir caminhos propostos comprovadamente errados: T2 é exportada por `short/01`, T7 por `short/02`, F2 precisa de coordenadas de projeção (não `selected_genera.csv`). Selecionar uma projeção identificada ou produzir um CSV combinado com identidade de modelo e gênero.
4. Registrar a conversão `noise_rate → epsilon` (T3/F1), `Transition → level_pair` (T6), `codon → triplet` e a escolha explícita de uma série RSCU (T9/F4). T10 mantém `signal` e `codon`; não o apresentar como uso de códons em ORFs confirmadas.

**Aceitação:** uma fixture com o esquema exportado por cada notebook passa no adaptador apropriado; cada coluna sem mapeamento falha com nome da fonte, item e coluna. F2 contém coordenadas, IDs dos pontos, modelo, gênero, método UMAP/PCA e aviso qualitativo; comentários iniciais do CSV são lidos corretamente ou removidos por exportação estruturada adicional.

**Arquivos candidatos:** `configs/reproduction_m07.yaml`, `src/viral_cnn/reporting/transformations.py`, `src/viral_cnn/reporting/publication.py`, `src/viral_cnn/paired_umap.py`, `tests/test_m07_publication_contracts.py` e um documento de contratos em `docs/`.

### C02 — Manifestos derivados, hashes e suporte experimental

**Problema:** `emit_derived_manifest()` grava `evidence_kind: derived`; `validate_selection()` aceita apenas `empirical` e `structural`. `n` e `support` recebem o número de linhas da tabela transformada. O manifesto derivado registra o manifesto pai, mas não o arquivo fonte transformado.

**Implementar:**

1. Definir formalmente `derived` no esquema C06: transformação determinística de evidência empírica identificada, com validação transitiva de todos os pais. Manter `structural` reservado à derivação estrutural que já tem regras próprias.
2. Registrar caminho e SHA-256 do CSV/Parquet fonte, do manifesto pai, do código/configuração da transformação e do arquivo gerado. Validar existência, hash, status, modo não sintético e identidade de todos os pais; recusar ciclos ou caminhos fora do run selecionado quando contrariem o plano.
3. Separar `output_rows`, `n_queries`, `n_fragments`, `n_taxa` e `denominator_n` conforme o item. Propagar ou calcular o suporte a partir de campos auditados do pai. Quando não aplicável, exigir motivo específico; nunca inferir `N` do tamanho de uma tabela de resumo.
4. Herdar identidade científica apenas após verificar sua presença e compatibilidade: dataset, split, classe, taxonomia, checkpoints, variantes, condições de ruído, métrica e semente. Para figuras, registrar seleção de IDs, níveis, amostragem, filtros e semente.

**Aceitação:** um artefato de T1 criado por `execute_publication_transformation()` chega a `DERIVED_VERIFIED` no C06; adulteração de fonte, manifesto ou pai resulta em `INVALID`; pai sintético resulta em bloqueio; `N` permanece o número de observações avaliadas mesmo quando o CSV final tem uma linha. Nenhuma validação é afrouxada para passar uma fixture.

**Arquivos candidatos:** `src/viral_cnn/reporting/{transformations,publication}.py`, esquema/config C06, testes M07/C06.

### C03 — Adaptadores de tabelas e figuras básicas

**Problema:** T1/T2 preenchem medições faltantes com `0.0`; T12 pede colunas agregadas enquanto `long/03` exporta linhas por variante e agregação; outros itens têm nomes ou unidades diferentes.

**Implementar:**

- **T1:** exigir origem verificável para parâmetros, acurácia, top-5, macro-F1 e latência; quando uma medida for indisponível, usar estado explícito e ajustar a apresentação. Definir se percentuais são `0–1` ou `0–100` e verificar limites.
- **T2:** exigir perfis medidos para VRAM, tempo, throughput e horas GPU; remover preenchimento numérico. Distinguir `step_time_ms` de latência de inferência.
- **T3/F1:** converter `noise_rate` sem perder método, estrato, cobertura e `N`; definir métricas com e sem abstenção.
- **T6:** converter transição taxonômica e manter tipo de correlação e número de táxons.
- **T9/F4:** mapear `codon` para rótulo de **tripleto exploratório**; selecionar frame 0, três frames ou seis frames sem confundir os valores; manter as três séries disponíveis em tabela ou suplemento.
- **T10:** preservar ATG/STOP como contagem exploratória em motivos, sem afirmar função de início/fim de tradução.
- **T12:** pivotar as linhas `variant × aggregation` de `build_c02_table12()` para uma linha por variante e modo (`pad`/`skip`), respeitando `N`, cobertura, acurácia incondicional e exclusões. Testar ausência e duplicação de cada agregador.
- **F2:** alimentar o gráfico a partir da projeção identificada; usar gênero como cor/destaque se presente, com seleção e parâmetros declarados. Eixos de ajustes independentes não permitem comparar distâncias entre painéis.

**Aceitação:** fixtures com os cabeçalhos atuais de `short/01`, `02`, `06`, `08`, `09` e `long/03` geram as colunas finais; falta de VRAM/latência ou dado de pivot produz erro ou indisponibilidade explícita, nunca zero; figuras têm dados rastreáveis ao CSV escolhido. Revisar T4/T5/T7/T8 contra o inventário C01 antes de marcá-las prontas.

### C04 — Decisão científica sobre T11

**Problema:** `short/08` produz coocorrência e associação exploratória entre motivo e escore; isso não equivale automaticamente à correlação **entre arquiteturas**. O transformador só exige o teste de permutação se a coluna estiver presente, e `expected.columns: []` bloqueia C06.

**Decisão padrão proposta:** T11 descreve a associação de coocorrência/orientação efetivamente calculada em `short/08`, com `n_valid_edges`, `n_valid_permutations`, correlações, ICC, delta por orientação e `p_permutation_null`. Renomear título, rótulos e legenda para não sugerir correlação entre modelos. Os diagnósticos de Jaccard entre variantes continuam descritivos e separados. Se a pergunta científica exigir correlação entre arquiteturas, abrir uma spec distinta com pares independentes, diversidade não degenerada, nulo de permutação apropriado e correção por múltiplos testes.

**Aceitação:** T11 tem lista não vazia de colunas esperadas, interpretação coerente no CSV, legenda e manuscrito; `p=0` originado de matrizes idênticas/sem variância é rejeitado; ausência do teste necessário falha. A escolha científica fica documentada antes da geração do artigo.

### C05 — Figuras F3/F5, seleção e legibilidade

**Implementar:**

1. **F3:** verificar os cinco checkpoints e os painéis a–e, comparabilidade da perturbação pareada, controles uniforme e GC pareado, unidade do eixo, suporte por motivo e status estatístico. Não substituir resultado ausente por barra ou queda zero.
2. **F5:** usar `edge_table.csv` com IDs, pesos, coordenadas, filtro/top-k, seed e layout. Informar a análise exploratória e o resultado do teste de permutação na legenda; o grafo não deve sugerir significância de cada aresta.
3. Revisar renderização de F1–F5 em tamanho de coluna, nomes de variantes, legenda, contraste e aviso qualitativo de F2; guardar captura de inspeção ou checklist de figura associado ao run.

**Aceitação:** teste negativo para checkpoint ausente em F3 e coordenadas ausentes em F5; teste de montagem com dados estruturados e inspeção visual das cinco figuras. Legendas e tabelas usam o mesmo protocolo e população selecionados.

### C06 — Integração realista e seleção C06

**Implementar:**

1. Criar um teste de integração pequeno que **chame os exportadores/adaptadores reais** com dados determinísticos, depois `execute_publication_transformation()` e `validate_selection()`/`assemble_publication()` em modo estrito. Fixtures devem imitar formatos de saída dos notebooks, não o formato já pronto do artigo.
2. Exercer os 17 itens e as dependências 07→08, C03→09, 06→F2 e long/01→03→T12. Para T11, usar a decisão C04; se ainda pendente, o teste de pacote deve reportar o bloqueio, não marcar sucesso total.
3. Incluir testes negativos para hash adulterado, split/seed divergente, classe/ordem trocada, checkpoint ausente, dependência sintética, CSV com comentários, coluna faltante, zero fabricado, N de resumo, seleção de figura incompleta e par de agregações T12 incompleto.
4. Preparar um comando para construir uma **configuração candidata** C06 a partir de um `run_plan.json` explícito. Ele propõe caminhos/hashes e imprime diferenças; a seleção final exige revisão humana dos itens científicos, sem escolher automaticamente o último timestamp.

**Aceitação:** testes M00–M07 e C06 passam; `python scripts/assemble_publication.py --mode diagnostic` com a configuração candidata distingue os itens completos dos faltantes; após seleção de evidência empírica, o modo estrito aceita todos os 17 itens. O notebook `00_regenerate_all_paper_artifacts.ipynb` aponta para a configuração selecionada e reproduz as mesmas decisões.

### C07 — Nova rodada empírica e reconciliação

1. Congelar commit, configuração, inventário de dados e hashes de checkpoints. Criar `run_id` novo; executar preflight M00. Preservar o achado de overlap exato/RC e a **proveniência genômica desconhecida** como limitações explícitas.
2. Confirmar previamente as 35 condições esperadas do C03; gerar/validar seus manifestos e somente então executar `short/09`. Não iniciar a etapa de publicação a partir de um benchmark parcial.
3. Executar `short/01–02`, `05–08`, `09–11` e `long/01–04` conforme as dependências do plano. `03/04` de treinamento apenas se houver decisão metodológica separada, orçamento e necessidade de pesos novos.
4. Inspecionar CSVs, distribuições, denominadores e figuras de cada etapa; registrar falhas e reruns em novos IDs. Gerar matriz de evidências C06, confrontar valores do manuscrito, atualizar tabelas, figuras, legendas, métodos e limitações.
5. Concluir a reconciliação M08: cada afirmação quantitativa do artigo tem item T/F, run, manifesto, CSV e denominador. Remover ou reformular afirmações sem evidência selecionada.

**Critério de saída:** pacote C06 estrito completo, relatório de diferenças históricas, figuras legíveis, lista de limitações e execução reprodutível a partir do commit/configuração fixados. O teste de integração local autoriza gastar computação; só a rodada empírica permite afirmar que os números do artigo foram reproduzidos.

## Prompt inicial para uma IDE agêntica

> Implemente somente a spec C01 deste plano no repositório `cnn_metagenome`. Leia `AGENTS.md`, as specs M00–M07 e o código atual antes de editar. Use os cabeçalhos exportados pelos notebooks como fonte de verdade e produza uma matriz versionada de T1–T12/F1–F5. Acrescente testes com os esquemas reais, sem executar dados do cluster, sem alterar resultados históricos e sem preencher medições ausentes. Ao terminar, apresente diff, testes executados, pontos não resolvidos e proposta da spec C02. Não implemente C02 nesta tarefa.

Para as specs seguintes, substituir `C01` pela etapa desejada e incluir as decisões aprovadas nas etapas anteriores. Manter o limite de escopo por spec; a validação C06 integrada tem sua própria etapa.
