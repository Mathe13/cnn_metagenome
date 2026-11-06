# Projeções pareadas dos cinco modelos — C10

O notebook `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb`, célula 7, gera projeções para ResNet50, ResNet50_F, ConvNeXt_Z, ConvNeXt_T e ConvNeXt_F. Cada ajuste usa os mesmos fragmentos identificados do teste. O protocolo é `c10_paired_umap_v1`.

Na primeira célula, os parâmetros editáveis e seus overrides de ambiente são:

```python
LATENT_UMAP_MAX_SAMPLES = int(os.environ.get("LATENT_UMAP_MAX_SAMPLES", "100"))
LATENT_UMAP_MAX_GENERA = int(os.environ.get("LATENT_UMAP_MAX_GENERA", "10"))
```

O mínimo é três fragmentos e um gênero; os valores precisam ser inteiros. A seed é a do runtime (42). A seleção escolhe até dez gêneros entre os elegíveis, distribui até 100 fragmentos em rodadas equilibradas e redistribui vagas quando um gênero tem suporte insuficiente. Não há reposição. Os IDs são ordenados antes da amostragem, garantindo invariância à ordem das linhas dos modelos. Genus ausente exclui uma linha somente da visualização, com contagem registrada; as avaliações quantitativas mantêm seus conjuntos atuais.

Antes de ajustar qualquer projeção, os IDs selecionados são encontrados em todas as variantes e seus rótulos Species são conferidos. Um ID selecionado ausente, duplicado ou com label diferente interrompe a comparação. Não há interseção silenciosa de populações. A escolha de no máximo dez gêneros permite uma legenda legível e vários fragmentos por grupo; não representa todos os táxons do dataset.

Cada ajuste usa cosseno, duas dimensões, `min_dist=0.1`, seed 42 e `n_neighbors=min(15, N-1)`; o último limite evita parâmetros inválidos para amostras pequenas. Somente as features selecionadas são copiadas para a projeção. KNN e Silhouette continuam calculados no espaço original, com os protocolos e amostras vigentes.

O comportamento anterior do notebook de reprodução selecionava as primeiras 100 linhas de ConvNeXt_F. O notebook histórico já continha projeções por modelo e gráficos por Genus/Order; a Figura 2 do manuscrito é colorida por gênero. A nova seleção estratificada pode alterar todas as coordenadas de ConvNeXt_F e tem protocolo próprio. O painel C10 é exploratório; não se declara reprodução exata da figura publicada.

Saídas no diretório novo de execução:

- Cinco CSVs: `umap_projection_resnet_clean_species.csv`, `umap_projection_resnet_noise_species.csv`, `umap_projection_convnext_zero_species.csv`, `umap_projection_convnext_transfer_species.csv` e o nome preservado `umap_projection_convnext_f.csv`.
- `umap_sample_selection.csv`: IDs operacionais, labels 0-based e nomes de Genus compartilhados.
- `umap_comparison.png`: cinco painéis com cores/legenda comuns por Genus e identificação do método.
- `umap_comparison.json`: protocolo, status COMPLETE, seleção/contagens, parâmetros, backend, entradas, configurações, mapa de cores e hashes das saídas. É publicado após todas as projeções e o painel; sua ausência indica comparação incompleta.

Os CSVs de projeção incluem `row_id`, `label`, `Genus`, `umap_1`, `umap_2`, `projection_method` e `is_qualitative_only`. Por compatibilidade, os nomes das duas colunas de coordenadas continuam os mesmos quando o fallback PCA é usado; `projection_method`, títulos e eixos deixam explícito PCA. Os cabeçalhos comentados preservam os avisos qualitativos; para ler, use `pd.read_csv(path, comment="#")`.

Os ajustes são independentes e seus eixos não estão alinhados. Cores iguais indicam o mesmo gênero, não coordenadas comparáveis entre modelos. Separação visual não mede acurácia, qualidade absoluta do embedding nem distância filogenética. O split original é por fragmentos de classes conhecidas; esta visualização não avalia genomas novos.

Sem `umap-learn` importável, a função histórica usa PCA. O resultado é identificado como PCA em todos os CSVs, metadados e painel, e o manifesto final não declara `umap_exported=true`. O caminho UMAP usa diretamente o mesmo estimador e parâmetros da função histórica, propagando erros de ajuste; isso evita que um ImportError durante o fit seja convertido silenciosamente em PCA e rotulado como UMAP. Nenhuma dependência é baixada automaticamente. Sem IPython, o PNG é salvo e seu caminho é exibido; no Jupyter há exibição inline.

Reutilização segue `LATENT_EXECUTION_MODE=reuse` e `LATENT_REUSE_RUN_ID`, descritos em [latent_resume.md](latent_resume.md). Cada variante possui checkpoint de progresso próprio, vinculado à seleção, hash do embedding, parâmetros, código e método. Cache inválido provoca recálculo dessa variante; os arquivos anteriores são preservados. Seleção, painel e metadados são reconstruídos no novo run.

O cache atual usa hash global do código do notebook e dos módulos: atualizar de C09 para C10, ou editar parâmetros no notebook, pode invalidar também sonda, predições, KNN e Silhouette sem alteração de seus algoritmos. Isso é uma invalidação conservadora de proveniência. Os embeddings antigos compatíveis continuam reutilizáveis porque `eval/latent.py` permanece intacto. Não se deve ignorar hashes para evitar esse custo. Uma projeção antiga baseada nas primeiras 100 linhas não é promovida ao protocolo C10.

Verificação local: testes com fixtures, smoke do notebook, alinhamento de IDs/rótulos, falhas de publicação e corrupção por variante. A validação é sintética em CPU, com `REMOTE_SERVER=False`; a execução empírica remota e os checkpoints/dados canônicos não estão disponíveis nesta sessão. O caminho UMAP é exercitado por um estimador de teste com parâmetros observáveis quando a biblioteca não está instalada; o fallback PCA usa a função real. Isso não valida numericamente UMAP real nem reproduz integralmente o artigo.
