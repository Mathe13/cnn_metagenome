# Instruções para agentes — código do artigo de classificação taxonômica viral

## Objetivo e escopo

Este repositório reúne o código, os notebooks, as configurações e as instruções necessárias para reproduzir os experimentos do artigo **“Spatial one-hot nucleotide representations of DNA enable robust and interpretable taxonomic classification with 1D convolutional networks”**. O estudo compara adaptações 1D de ResNet50 e ConvNeXt para classificação taxonômica viral, análise de robustez, representações latentes, padrões de sequência e inferência por janelas.

Trate o artigo, os notebooks originais, os artefatos de dados e as configurações efetivamente usadas como evidências que precisam ser conciliadas. O artigo descreve o resultado publicado; o código e seus artefatos determinam o que foi executado. Se houver divergência, registre-a e apresente a evidência antes de corrigir o código ou afirmar que um resultado foi reproduzido.

## Antes de modificar qualquer coisa

1. Leia o `README`, este arquivo, o manifesto de dados, as configurações e os notebooks relacionados à tarefa. Descubra os caminhos reais no repositório; não presuma que a estrutura sugerida abaixo já exista.
2. Identifique a cadeia de dependências do resultado: dados de entrada → curadoria/split → codificação → checkpoint → inferência → métricas → tabela/figura do artigo. Anote os nomes e versões dos artefatos encontrados.
3. Separe código de produção ou reprodução dos notebooks históricos. Preserve os notebooks que originaram os resultados; extraia funções reutilizáveis para módulos sem alterar silenciosamente o experimento original.
4. Proponha ou faça a menor mudança verificável. Diga quais números, arquivos gerados e etapas da cadeia ela pode afetar.
5. Nunca apresente um script novo, um teste sintético ou um notebook que apenas abre como reprodução integral do resultado do artigo.

## Invariantes científicos

- **Tarefa:** classificação supervisionada *closed-set* de táxons virais conhecidos. As classes avaliadas estão presentes no treinamento. O split principal é por **fragmentos**, não por genoma: não descreva seus números como generalização para genomas ou espécies não vistos.
- **Classes:** a configuração descrita no artigo usa 13.683 Species e 2.702 Genera; análises da hierarquia incluem Order. Leia os mapeamentos versionados antes de confiar em números ou índices. A saída do modelo, a tabela de rótulos e o checkpoint devem usar exatamente a mesma ordem de classes.
- **Dados:** a curadoria parte de referências taxonômicas e inclui Kraken2, NCBI e ICTV. Guarde a proveniência e as versões das referências, regras de exclusão, balanceamento, deduplicação, mapeamentos Species→Genus→Order, totais por classe e identificadores dos splits. A semente 42 e a divisão estratificada 80/10/10 constam no método; confira a implementação antes de reutilizá-las.
- **Entrada:** janelas de 512 bp são codificadas em quatro canais na ordem A/C/G/T. Padding e bases desconhecidas/ambíguas geram vetor nulo. Uma alteração em ordem dos canais, truncamento, comprimento, alfabeto ou tratamento de padding exige nova avaliação e documentação.
- **Modelos:** mantenha identificáveis ResNet50 1D padrão e com fine-tuning para ruído; ConvNeXt 1D treinado do zero (`ConvNeXt_Z`), transferido de Genus para Species (`ConvNeXt_T`) e com fine-tuning para ruído (`ConvNeXt_F`). Não troque checkpoints entre níveis ou variantes apenas porque têm arquitetura ou dimensão de cabeça compatível.
- **Ruído:** o teste principal seleciona cada posição com probabilidade $\epsilon \in \{0, 0.02, 0.05, 0.10, 0.15\}$ e ressorteia uma base uniformemente entre A/C/G/T, inclusive a original. A taxa esperada de mudanças efetivas é $3\epsilon/4$. Registre a semente, o conjunto avaliado, a amostragem e a ordem das operações. Esse procedimento **não** simula integralmente erros de ONT: não inclui, por si só, inserções, deleções ou dependência de contexto.
- **Treinamento com ruído:** diferencie checkpoint limpo, transferência Genus→Species e fine-tuning com corrupção; o método descreve fine-tuning com $\epsilon=0.05$ fixo, aplicado também à validação nessa etapa. Preserve origem do checkpoint, épocas, otimizador, scheduler, hiperparâmetros, precisão mista e regra real de seleção do checkpoint. Não rebatize o último checkpoint como “melhor” sem critério documentado.
- **Sequências longas:** use o termo *long-sequence* para contigs/genomas variáveis; reserve *long-read* para leituras efetivas de tecnologias de sequenciamento. Documente comprimento da janela, stride, regra para a última janela, padding, mapeamento de rótulos e agregação (voto majoritário ou média de **logits antes de softmax**). Evite vazamento entre referências longas e os fragmentos do teste ao fazer afirmações de generalização.

## Métricas, análise e figuras

- Calcule Accuracy, Top-5, F1 e perda usando o mesmo conjunto de exemplos, nível taxonômico, ordem de classes e denominador declarados. Diferencie Macro-F1 de Weighted-F1 e diferenças em pontos percentuais de variação relativa.
- Para faixas de F1 por espécie, registre limiares, suporte, regra para classes sem amostras/predições e definição de confiança. Confiança Top-1 e entropia **não** equivalem a calibração; só use esse termo quando ECE, Brier, NLL ou outra medida formal tiver sido calculada.
- Extraia embeddings antes da camada classificadora. Para KNN e Silhouette, use o espaço original de alta dimensão; UMAP serve apenas à visualização. O método do artigo especifica KNN com $k=5$, split referência/consulta estratificado 80/20 sem auto-vizinhança, normalização $L_2$, distância cosseno e semente 42. Silhouette usa amostra limitada; registre tamanho, classes excluídas, método de amostragem e distância.
- Ao correlacionar níveis da hierarquia, agregue métricas dos filhos conforme o método, mantenha separados os conjuntos válidos de KNN e Silhouette e reporte coeficiente, $N$ e níveis comparados. Correlação entre níveis taxonômicos não demonstra, sozinha, distância filogenética nem superioridade absoluta do embedding.
- Para padrões extraídos da primeira camada, mantenha a construção exata dos 7-mers, o controle aleatório pareado da **mesma sequência**, a regra de não sobreposição, o mesmo padrão de substituições, as repetições, a semente e a correção para múltiplos testes. Diferencie associação de composição de tripletos de mecanismo causal, sítio funcional ou uso de códons em quadro de leitura comprovado.
- Toda tabela e figura regenerada deve registrar: comando/notebook, entradas e seus hashes ou versões, configuração, checkpoint, split, semente e saída. Não ajuste manualmente valores para coincidir com o PDF. Se o valor divergir, investigue e relate a diferença.
- Se houver comparação com Kraken2, BLASTN ou outro método, use o mesmo conjunto de consultas e a mesma versão da taxonomia; declare diferenças de cobertura, ausência de previsão, denominador e política de correspondência de nomes/IDs. Não compare percentuais de protocolos distintos como se fossem um benchmark controlado.

## Organização do repositório

Se o repositório ainda estiver em formação, prefira uma estrutura equivalente a esta, adaptada aos caminhos já existentes:

```text
README.md                     # escopo, ambiente, dados e ordem de reprodução
AGENTS.md                     # estas instruções
configs/                      # parâmetros por experimento, sem caminhos pessoais
src/                          # curadoria, datasets, modelos, treino, avaliação e análises
notebooks/original/           # notebooks históricos preservados
notebooks/reproduction/       # execução documentada das etapas reproduzíveis
scripts/                      # entradas de linha de comando para pipelines e figuras
tests/                        # verificações pequenas dos contratos científicos
docs/                         # proveniência e mapa resultado → experimento
```

Não mova nem renomeie notebooks, checkpoints, tabelas de rótulos ou diretórios de dados antes de atualizar as referências dependentes. Caminhos locais de uma máquina de pesquisa devem virar parâmetros de configuração ou variáveis de ambiente; não inclua credenciais, dados restritos nem caminhos absolutos pessoais em commits. Dados brutos, Parquet volumoso, checkpoints e saídas geradas exigem instruções de obtenção e verificação, e normalmente não pertencem ao Git.

## Convenções de implementação

- Use Python e as bibliotecas já empregadas pelo repositório. Registre versões de Python, PyTorch, CUDA, Dask/cuDF, pandas/pyarrow e demais dependências relevantes; não invente versões a partir de mensagens de notebook.
- Transforme células reutilizáveis em funções com entradas explícitas, sem depender de estado invisível, ordem de execução ou `import *`. Os notebooks podem continuar como interface de análise.
- Use configurações legíveis e salve-as com cada execução. Defina sementes para Python, NumPy, PyTorch, particionamento e perturbações; registre limites de determinismo em GPU.
- Preserve o tratamento correto de `IterableDataset`, a distribuição disjunta de arquivos entre workers e o comportamento de CPU/GPU. Testes rápidos devem usar dados sintéticos e não iniciar treinamento completo ou baixar bases automaticamente.
- Acrescente testes apenas para contratos com impacto científico: ordem A/C/G/T e padding; integridade e disjunção dos splits; mapeamento taxonômico e ordem de logits; taxa efetiva esperada de ruído; agregação de janelas; ausência de auto-vizinho no KNN; controle pareado de motivos. Testes não substituem validação com artefatos reais.
- Não mude arquitetura, pesos, pré-processamento, split, definição de métrica ou composição do conjunto e mantenha o mesmo nome do experimento. Uma mudança metodológica precisa de novo identificador, comparação explícita e atualização da documentação.

## Critério para encerrar uma tarefa

Ao concluir, informe: arquivos alterados; experimento ou resultado afetado; como verificar; verificações executadas e seus resultados; quais dados/checkpoints não estavam disponíveis; e qualquer divergência entre notebook, código e artigo. Para uma reprodução, forneça o comando exato e o manifesto dos artefatos. Não declare reprodução bem-sucedida quando apenas testes pequenos ou inspeção estática foram executados.
