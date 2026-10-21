# Estudo de Viabilidade: Avaliação por Genoma de Origem e Split Agrupado

## 1. Objetivo e Escopo Científico

Este documento formaliza a análise de viabilidade matemática e biológica para a criação de um protocolo de avaliação **desincorporado de genoma** (*genome-disjoint evaluation*) para os modelos 1D CNN (ResNet50-1D e ConvNeXt-1D) no artigo:
**“Spatial one-hot nucleotide representations of DNA enable robust and interpretable taxonomic classification with 1D convolutional networks”** (BioData Mining / submetido).

No manifesto [docs/dataset_manifest.md](dataset_manifest.md) (S02), comprovou-se que a partição estratificada por fragmentos (80/10/10 com `random_state=42`) gera **95,91% de queries de teste idênticas ou em fita reversa complementar** ao treino.
O objetivo da presente spec (S03) é determinar:
1. As condições necessárias para avaliar a generalização a genomas não vistos dentro de espécies conhecidas.
2. A fração de espécies virais que suporta esse particionamento.
3. O protocolo formal de agrupamento para experimentos futuros (S04 e seguintes).

---

## 2. Critérios Matemáticos e Biológicos de Viabilidade

### 2.1 O Teorema da Bipartição Closed-Set de Genomas
Em aprendizado supervisionado de classificação taxonômica *closed-set*:
- Todas as classes avaliadas no conjunto de teste $Y_{\text{test}}$ devem estar presentes no conjunto de treinamento $Y_{\text{train}}$:
  $$Y_{\text{test}} \subseteq Y_{\text{train}}$$
- Para que o teste seja desincorporado de genoma (*zero-leakage* por genoma de origem), o conjunto de genomas de teste $G_{\text{test}}$ e de treino $G_{\text{train}}$ devem ser disjuntos:
  $$G_{\text{train}} \cap G_{\text{test}} = \emptyset$$
- Seja $G(c)$ o conjunto de genomas de referência independentes disponíveis para a espécie $c$:
  - Se $|G(c)| = 1$:
    - Se alocamos o único genoma em $G_{\text{train}}$, então não há fragmentos para testar a espécie $c$ ($c \notin Y_{\text{test}}$).
    - Se alocamos o único genoma em $G_{\text{test}}$, então $c \notin Y_{\text{train}}$, o que transforma o problema em classificação *open-set* / *zero-shot*, violando a premissa closed-set.
    - Se dividimos os fragmentos do genoma entre treino e teste, reintroduz-se o vazamento de 95,91% comprovado em S02.

### 2.2 Conclusão de Viabilidade
> **Requisito Fundamental:**
> A avaliação supervisionada *closed-set* desincorporada de genoma **exige estritamente no mínimo 2 genomas de referência independentes por espécie** ($|G(c)| \ge 2$).
> Para toda espécie com genoma único ($|G(c)| = 1$), a avaliação desincorporada de genoma no teste é **matematicamente inviável**.

---

## 3. Inviabilidade de Classes com Genoma Único e Mudança de População

Na curadoria de vírus do NCBI e do Kraken2, a grande maioria dos vírus descritos possui apenas uma sequência genômica de referência catalogada (holótipo ou genoma de referência RefSeq).

### 3.1 Tratamento de Classes com Genoma Único
Para construir um split agrupado sem vazamento, espécies com $|G(c)| = 1$ devem ser tratadas de uma de duas formas:
1. **Política `train_only` (Recomendada):** Os fragmentos do genoma único são mantidos exclusivamente no conjunto de treinamento. Isso permite ao modelo aprender a representar o espaço taxonômico amplo (13.683 classes), mas o conjunto de teste só avalia espécies com suporte independente ($|G(c)| \ge 2$).
2. **Política de Exclusão:** As espécies com $|G(c)| = 1$ são completamente removidas do dataset. Isso reduz a complexidade da cabeça do classificador para apenas o subconjunto com múltiplos genomas.

### 3.2 Mudança de População e Cobertura Taxonômica
- A transição do split de fragmentos para o split agrupado implica uma **mudança drástica de população avaliada** (*population shift*).
- Se uma espécie não possui ao menos dois genomas, ela não pode ser testada no split agrupado. A taxa de acurácia obtida refletirá exclusivamente o desempenho sobre vírus frequentes ou intensamente sequenciados (ex.: *SARS-CoV-2*, *Influenza*, *HIV*, bacteriófagos de interesse industrial).

---

## 4. Regras de Comparabilidade de Protocolo (Invariante Metodológico)

Conforme determinado em `AGENTS.md`:

> **Regra Permanente de Comparabilidade:**
> **Nunca confronte a acurácia do split agrupado com a acurácia de 99%+ do split histórico de fragmentos como se fossem o mesmo benchmark.**
> 
> 1. O split histórico avalia **13.683 espécies** e mede a memorização e identificação de fragmentos de genomas conhecidos (onde 95,91% do teste é idêntico ao treino).
> 2. O split agrupado avalia um **subconjunto restrito de espécies** ($N \ll 13.683$) e mede a generalização filogenética a variantes ou genomas não vistos da mesma espécie.
> 3. Qualquer publicação ou relatório que compare ambos deve explicitar:
>    - O número de classes avaliadas no teste em cada protocolo.
>    - O denominador de fragmentos.
>    - O fato de que a queda esperada de desempenho decorre da eliminação do vazamento genômico e da mudança de distribuição.

---

## 5. Arquitetura do Motor de Particionamento (`viral_cnn.data.grouped_split`)

O particionamento agrupado é implementado em `src/viral_cnn/data/grouped_split.py` e opera sob os seguintes invariantes:
1. **Agrupamento Estrito por Genoma:**
   Dada uma tabela com `sequence`, `species_id` e `genome_id` (accession):
   - Todos os fragmentos que possuem o mesmo `genome_id` são alocados no mesmo split.
   - Garantia matemática:
     $$\text{set}(\text{train\_genomes}) \cap \text{test\_genomes} = \emptyset$$
     $$\text{set}(\text{train\_genomes}) \cap \text{val\_genomes} = \emptyset$$
     $$\text{set}(\text{val\_genomes}) \cap \text{test\_genomes} = \emptyset$$
2. **Divisão Proporcional por Classe:**
   Para cada espécie com $k \ge 2$ genomas:
   - Os genomas são alocados de forma a aproximar a proporção alvo (ex.: 80% treino, 20% teste), garantindo ao menos 1 genoma no treino e ao menos 1 genoma no teste.
3. **Isolamento de Classes Inviáveis:**
   - Classes com $k = 1$ genoma são alocadas em `train_only` (ou marcadas como excluídas), impedindo vazamento para o teste.
4. **Relatório de Cobertura Estruturado:**
   - Emissão de métricas completas em JSON: total de espécies, espécies elegíveis, espécies inviáveis, total de fragmentos retidos por partição e verificação de interseção nula.
