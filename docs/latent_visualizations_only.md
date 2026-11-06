# Notebook 06b — visualizações de embeddings arquivados

Abra [`06b_latent_visualizations.ipynb`](../notebooks/reproduction/short/06b_latent_visualizations.ipynb) e execute todas as células. A origem padrão é `results_reproduction/runs/c01-20261104T161929937502/06_ictv_latent_space_and_hierarchy/`. Atualize também `src/viral_cnn/latent_plotting.py`, `src/viral_cnn/latent_explorer.html` e `src/viral_cnn/visualization_cache.py` no servidor, junto dos módulos já utilizados pelo notebook 06: `paired_umap.py`, `latent_resume.py` e `eval/latent.py`.

A primeira célula permite editar `SOURCE_DIR`, `TAXONOMY_PATH`, `MAX_GENERA`, `MAX_SAMPLES`, `SAMPLES_PER_GENUS`, `READ_BATCH_SIZE`, `EXECUTION_MODE` e `REUSE_RUN_ID`. Os overrides de ambiente usam o prefixo `LATENT_VIS_`, exceto `TAXONOMY_PATH`. O padrão remoto permanece ativo; smoke local com `REMOTE_SERVER=False` gera somente fixtures sintéticas, claramente identificadas no manifesto.

Para todos os gêneros elegíveis, escolha:

```python
MAX_GENERA = "all"
MAX_SAMPLES = "auto"
SAMPLES_PER_GENUS = 10
```

O orçamento é dez fragmentos por gênero, distribuído em rodadas sem reposição, com redistribuição das vagas de grupos pequenos. Isso não seleciona todos os fragmentos. Os padrões são até dez gêneros e até 100 fragmentos. Milhares de gêneros aumentam o custo de projeção; a amostra efetiva e as exclusões ficam registradas.

O painel usa uma grade 2×3: cinco modelos e um espaço para a legenda/instruções. Acima de 20 gêneros, a legenda extensa sai do PNG e fica disponível em `umap_genus_colors.csv` (gênero, cor e suporte). Pontos menores, transparência e uma ordem de desenho embaralhada com seed 42 reduzem a sobreposição sem remover fragmentos. As cores originais são preservadas, com a mesma relação de aspecto para os dois eixos de cada gráfico.

Abra `umap_comparison.html` pelo link exibido no notebook ou baixe e abra no navegador. Ele funciona offline, sem bibliotecas externas: digite/escolha um gênero para destacá-lo simultaneamente nos cinco modelos, consulte IDs ao passar o mouse, arraste para mover e use a roda para zoom. O botão “Mostrar todos” restaura as cores da população; “Restaurar zoom” restaura os limites. Milhares de cores não permitem distinguir todos os gêneros pela cor; a busca oferece a identificação explícita.

Para corrigir uma figura já gerada sem executar novamente UMAP ou conferir os Parquets de embeddings, defina na primeira célula:

```python
PROJECTION_SOURCE_DIR = "results_reproduction/runs/SEU_RUN/06b_latent_visualizations"
```

Use o caminho relativo à pasta de execução ou o caminho absoluto real. A origem deve conter `umap_comparison.json` completo e os cinco CSVs correspondentes; também são aceitas comparações C10 geradas pelo notebook 06. O override de ambiente é `LATENT_VIS_PROJECTION_SOURCE_DIR`. Nesse modo, rótulos, IDs, gêneros, método, contagens e hashes dos CSVs são conferidos; os CSVs são copiados byte a byte para uma pasta nova e somente a apresentação muda. Não precisa de embeddings, de-para ou pesos. `MAX_GENERA` e os limites de amostragem não alteram a amostra salva. A origem é preservada.

Entradas necessárias para gerar novas coordenadas:

- Exatamente um par `features_<modelo>_test_<fingerprint>.parquet`/`.json` para cada uma das cinco variantes. Havendo pares concorrentes para uma variante, indique uma pasta com os cinco pares escolhidos; o notebook rejeita ambiguidades.
- O de-para taxonômico original em Parquet, com `id` e `Genus`. O contrato é `model_label = id - 1`; as classes precisam corresponder ao conjunto declarado nos cinco manifests. O notebook procura o caminho registrado no manifesto original e, depois, a topologia do notebook 06. Use `TAXONOMY_PATH` se o arquivo foi movido. `taxonomy_rank_coverage.json` não contém esse de-para.

Treino, validação, sequências brutas, pesos, KNN, sonda linear e as tabelas 5/6 não são carregados. O CSV legado `umap_projection_convnext_f.csv` é preservado na origem; ele sozinho não fornece uma comparação pareada dos cinco modelos. O 06b gera novas projeções pelo protocolo [C10](paired_umap.md), com os mesmos fragmentos e cores entre modelos.

A cadeia utilizada é: **embeddings de teste arquivados + de-para → validação → amostra pareada → cinco projeções → painel e manifesto**. Cada Parquet passa por uma leitura completa de bytes para conferir SHA-256; essa etapa ainda pode demorar e é indicada no console. Em seguida são decodificadas apenas as colunas `label`/`row_id`; IDs ficam em buffers UTF-8 compactos. As features são lidas em batches limitados aos grupos de linhas necessários e somente os vetores selecionados são retidos. Uma amostra espalhada pode atingir todos os grupos, mas nenhuma matriz completa ou spill é alocada. O hash já conferido é reutilizado na escrita do manifesto, evitando outra varredura dos Parquets de origem.

Os arquivos anteriores nunca são alterados. Cada execução grava em `results_reproduction/runs/n06b-<timestamp>/06b_latent_visualizations/`:

- Cinco CSVs de coordenadas e `umap_sample_selection.csv`.
- `umap_comparison.png` com cinco painéis e espaço para legenda/instruções.
- `umap_comparison.html` com busca, destaque por gênero, consulta de IDs, zoom e movimento.
- `umap_genus_colors.csv` com cores e contagens para todos os gêneros selecionados.
- `umap_comparison.json`, publicado após concluir o painel, com parâmetros e identidades das entradas/saídas.
- `manifest.json` com versões, sementes, hashes, limites, estatísticas de leitura e limitações.
- Registros por variante `stage_umap_*` e `resume_report.json` para reutilização validada ao gerar coordenadas; o modo de redesenho registra sua origem sem ajustar projeções.

`EXECUTION_MODE = "reuse"` é o padrão. Reutiliza projeções compatíveis de execuções **06b** anteriores; `REUSE_RUN_ID` permite escolher uma delas. O SHA-256 dos embeddings continua sendo conferido. Não procura estágios quantitativos do 06. Cache de projeção incompatível é recalculado preservando o original; uma entrada corrompida interrompe o notebook porque ele não extrai novos embeddings. `EXECUTION_MODE = "rerun"` recalcula as projeções.

Sem `umap-learn` importável, o backend existente usa PCA e identifica o método nos títulos, CSVs e manifestos. Os eixos das cinco projeções são ajustados independentemente. Esta análise é qualitativa, sobre o split por fragmentos de classes conhecidas; não mede generalização a genomas novos, distância filogenética ou qualidade absoluta dos modelos. As identidades históricas dos checkpoints são registradas pelos sidecars; seus pesos e dados originais não são revalidados. O protocolo da interface é `c11_visualization_only_v1`; a apresentação registra `c12_large_taxonomy_render_v1`, preservando a seleção/projeção C10; não se declara reprodução exata da Figura 2.

Execução remota pela raiz do repositório, usando a origem padrão:

```bash
REMOTE_SERVER=True SMOKE_TEST=0 jupyter execute notebooks/reproduction/short/06b_latent_visualizations.ipynb
```

Verificação offline em CPU:

```bash
REMOTE_SERVER=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp PYTHONPATH=src \
  .venv/bin/python3 -m unittest tests.test_n06b_visualizations tests.test_latent_plotting tests.test_c10_paired_umap -q
```

Os testes verificam correspondência de linhas/rótulos, seleção de grupos, corrupção, identidade dos manifests, contrato taxonômico, cinco projeções e reutilização sem reler features ou ajustar a projeção. Fixtures sintéticas e PCA local não validam numericamente UMAP real nem os dados/checkpoints do servidor.

Verificação inicial do 06b: **22 testes aprovados** (`test_n06b_visualizations` e `test_c10_paired_umap`), incluindo duas execuções completas do 06b sobre fixtures e a regressão do notebook 06. O notebook novo também teve todas as células compiladas e foi salvo sem saídas de execução. Os arquivos do run remoto informado, seu de-para e os pesos empíricos não estavam disponíveis localmente.

Verificação da apresentação para muitos gêneros: **25 testes aprovados** nos três módulos do comando acima, incluindo redesenho sem embeddings/ajuste, cópia idêntica dos CSVs, rejeição de corrupção e de divergências entre rótulos. Uma prévia sintética com 23.200 pontos e 2.320 gêneros foi inspecionada visualmente. O JavaScript passou na verificação de sintaxe; busca, zoom e restauração foram exercitados em Node com DOM/Canvas simulado nessa mesma população. A integração em navegador/Jupyter remoto e as projeções empíricas não foram verificadas localmente. A mudança afeta somente a apresentação, com identidade própria nos metadados; não altera o método científico nem resolve divergências numéricas com o artigo.
