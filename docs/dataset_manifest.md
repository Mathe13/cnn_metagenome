# Manifesto de Proveniência do Dataset e Auditoria de Splits — Classificação Taxonômica Viral

## 1. Escopo e Objetivo

Este manifesto documenta a proveniência dos dados, o histórico de curadoria e a auditoria das divisões de treino, validação e teste utilizados no artigo:
**“Spatial one-hot nucleotide representations of DNA enable robust and interpretable taxonomic classification with 1D convolutional networks”** (BioData Mining / submetido).

O objetivo desta auditoria é explicitar a origem de cada campo dos dados, documentar a metodologia do split estratificado por fragmentos e registrar a taxa de sobreposição exata e vazamento entre partições, fundamentando os limites científicos de generalização do modelo e orientando a spec S03 (avaliação por genoma de origem).

---

## 2. Bases de Referência Taxonômica

A taxonomia viral e as sequências de referência foram integradas a partir de três fontes principais:

1. **Kraken2 Viral Database:**
   - Base principal para a classificação supervisionada de fragmentos curtos (512 bp).
   - Mapeamentos estáticos:
     - `dataset_kraken/species_name_ids.csv` (13.683 classes de espécie com IDs 1 a 13.683).
     - `dataset_kraken/genus_name_ids.csv` (2.702 classes de gênero com IDs 1 a 2.702).
     - `dataset_kraken/family_name_ids.csv` (246 classes de família com IDs 1 a 246).
2. **NCBI Nucleotide:**
   - Fonte para obtenção de sequências longas e contigs completos via E-utilities (Entrez) no notebook `long/download_from_ncbi.ipynb`.
   - Consulta automatizada pelo nome binomial da espécie (`term=species_name`), gerando a tabela intermediária `raw_database/final.csv` com 33.023 sequências cobrindo 13.592 espécies.
3. **ICTV (International Committee on Taxonomy of Viruses):**
   - Planilha mestre `ICTV_Master_Species_List_2025_MSL41.v1.xlsx` utilizada em `short/ICTV_improve_kraken_database.ipynb` para estruturar a hierarquia superior (Species $\to$ Genus $\to$ Family $\to$ Order $\to$ Realm) e limpar discrepâncias de nomenclatura entre o NCBI e o ICTV.

---

## 3. Distinção entre Campos Observados e Campos Inferidos

Para garantir a rastreabilidade científica preconizada em `AGENTS.md`, distinguem-se rigorosamente os dados diretamente observados dos derivados ou inferidos:

### 3.1 Campos Diretamente Observados
- `species`, `genus`, `family`: Nomes taxonômicos textuais presentes nos CSVs de mapeamento.
- `id`: Identificador taxonômico numérico de 1 a $N$ (1-based), que converte para o índice de logit 0-based do modelo através do offset `model_label = id - 1` (AD-002).
- `sequence`: Fragmentos de DNA nucleotídico contendo caracteres `A, C, G, T` e bases ambíguas/padding.
- `label`: Rótulo de classe associado a cada fragmento em `species_seqs.csv`, `genus_seqs.csv` e `family_seqs.csv`.
- `total_rows`: Contagens exatas de fragmentos:
  - `dataset_kraken/species_seqs.csv`: 17.062.701 fragmentos.
  - `dataset_kraken/genus_seqs.csv`: 8.168.145 fragmentos.
  - `dataset_kraken/family_seqs.csv`: 1.273.788 fragmentos.

### 3.2 Campos Inferidos e Reconstruídos por Heurística
- **Acessos do NCBI (`accession`):** As tabelas `species_seqs.csv` não contêm os números de acesso (`NC_*` ou genbank ID). Os acessos foram inferidos posteriormente no notebook `download_from_ncbi.ipynb` buscando pelo nome da espécie no Entrez.
- **Identificador de Genoma de Origem:** Não está explícito na tabela de fragmentos curtos. Cada linha de `species_seqs.csv` é um fragmento isolado.
- **Gênero Derivado:** Em `download_from_ncbi.ipynb`, os gêneros foram inferidos por divisão léxica do primeiro termo binomial (`derived_genus_name = species_name.split()[0]`), criando 33 novos gêneros atribuídos dinamicamente.
- **Coordenadas de Fragmentação:** Janela deslizante de tamanho fixo 512 bp com passo/stride inferido de 256 bp nos contigs longos.
- **Inclusão de Fita Reversa Complementar:** Como comprovado na inspeção das primeiras linhas de `species_seqs.csv`, para cada fragmento forward existe o respectivo fragmento reverso complementar gerado sinteticamente durante a preparação da base bruta.

---

## 4. Metodologia do Split por Fragmentos e Parâmetros

A metodologia de treinamento e validação descrita no artigo e implementada nos notebooks segue os seguintes parâmetros:

- **Estratégia de Split:** Divisão estratificada por **fragmentos individuais** utilizando `sklearn.model_selection.train_test_split`.
- **Proporções:** 80% treino, 10% teste e 10% validação.
- **Semente Aleatória:** `random_state = 42`.
- **Estratificação:** Feita exclusivamente pelo rótulo da classe (`species_id` / `label`).
- **Filtragem de Classes Raras:** No notebook `prepare database.ipynb`, classes com menos de 3 amostras (`MIN_SAMPLES = 3`) foram excluídas antes da partição estratificada.
- **Sharding de Armazenamento:** Os dados foram salvos particionados em 256 arquivos Parquet por split (`part.0.parquet` .. `part.255.parquet`) dentro de `cnn_kraken/processed_kraken_species_splited_parquet/{train, test, validation}/`.

---

## 5. Auditoria de Sobreposição Exata e Vazamento de Dados

A auditoria direta executada sobre o split no notebook histórico `short/benchmark_cnns_kraken2_blast_overlap.ipynb` evidenciou o efeito da divisão por fragmentos quando genomas são fragmentados com sobreposição:

- **Amostragem de Teste:** 20 queries por espécie através de seleção determinística baseada em prioridade hash (`seed=42`), totalizando **273.660 queries** avaliando as 13.683 espécies virais.
- **Método de Hashing:** Digestão canônica invariante à fita: `hashlib.sha256(min(seq, reverse_complement(seq)).encode()).hexdigest()`.
- **Resultado da Auditoria Cruzada:**
  - **Queries com sobreposição exata com o treino:** **262.470 queries (95,91%)**.
  - **Queries sem sobreposição exata:** **11.190 queries (4,09%)** distribuídas em 7.691 classes.
  - **Classes com correspondência idêntica no treino:** Todas as 13.683 classes (100%) possuem ao menos uma query com correspondência idêntica ou reversa complementar no treino.

### Explicação Causal do Fenômeno:
Como o genoma viral de cada referência é fragmentado em múltiplas janelas (com passos sobrepostos e inclusão da fita reversa complementar) e o particionamento estratificado 80/10/10 sorteia **linhas individuais** (fragmentos) em vez de agrupar por accession/genoma, fragmentos contíguos ou idênticos da mesma molécula viral caem simultaneamente no conjunto de treinamento e no conjunto de teste.

---

## 6. Ressalva Científica Fundamental sobre Sobreposição Não Idêntica

> **Alerta Metodológico Conforme AGENTS.md:**
> **A ausência de correspondência de sequência idêntica NÃO equivale a zero sobreposição.**

Mesmo para as 11.190 queries (4,09%) onde não foi encontrado um fragmento perfeitamente idêntico no treino:
1. Fragmentos vizinhos deslocados por alguns pares de bases (decorrentes do stride da janela) compartilham até $512 - \text{stride}$ nucleotídeos idênticos com fragmentos de treino do mesmo genoma.
2. Alta similaridade de sequência e blocos conservados de $k$-mers existem entre variantes do mesmo genoma.
3. Portanto, os resultados de acurácia do modelo no split de fragmentos refletem a capacidade de reconhecer fragmentos de genomas **já vistos no treinamento** (tarefa closed-set de fragmentos), e **não devem ser reportados como generalização para genomas ou variantes virais não vistas**.

---

## 7. Diretrizes para a Spec S03 (Split Agrupado por Genoma)

Para mensurar a verdadeira capacidade de generalização da ResNet50-1D e ConvNeXt-1D dentro de espécies conhecidas sem vazamento genômico:
1. É necessário identificar as espécies virais que possuem ao menos 2 genomas/accessions independentes registrados.
2. O particionamento deve ser agrupado (`GroupKFold` ou `GroupShuffleSplit`) por identificador de genoma/accession, garantindo que nenhum fragmento proveniente do genoma de teste tenha sido exposto ao modelo durante o treino.
3. Para espécies com genoma único nas bases públicas, deve-se declarar formalmente a inviabilidade de teste desincorporado de genoma sem reduzir a população ou mascarar a taxa de cobertura taxonômica.
