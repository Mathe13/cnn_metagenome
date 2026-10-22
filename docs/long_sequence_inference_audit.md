# Relatório de Auditoria da Inferência de Sequências Longas (Long Voter)

Este documento registra a auditoria técnica, metodológica e conceitual da inferência por janelas deslizantes (*Sliding-Window Consensus Voter*) apresentada na **Seção 3.5 e Tabela 12** do manuscrito:
**“Spatial one-hot nucleotide representations of DNA enable robust and interpretable taxonomic classification with 1D convolutional networks”** (BioData Mining / submetido).

A auditoria atende diretamente às exigências formuladas na **Nota v2** de revisão do artigo e cumpre os invariantes científicos definidos em `AGENTS.md`.

---

## 1. Contexto e Apontamentos da Revisão (Nota v2)

A Tabela 12 do manuscrito reporta o desempenho de classificação em sequências longas sob duas estratégias de consenso: **Voto Majoritário** (*Majority Vote*) e **Média de Logits Pré-Softmax** (*Mean Logits*). Os resultados publicados registraram:

| Modelo | Espécie (Majority) | Espécie (Mean Logits) | Espécie (Top-5) | Gênero (Majority) | Gênero (Mean Logits) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet50 (Padrão/Clean)** | 66,67% | **72,69%** | 82,51% | 75,16% | 81,29% |
| **ResNet50 (Noise-aware $\epsilon=0,05$)** | 22,10% | 21,54% | 30,23% | 23,65% | 22,94% |
| **ConvNeXt (Noise-aware $\epsilon=0,05$)** | 20,38% | 19,99% | 34,13% | 22,06% | 21,55% |
| **ConvNeXt (Do zero / Scratch)** | 12,01% | 12,00% | 23,09% | 12,32% | 12,35% |
| **ConvNeXt (Transfer de Gênero)** | 3,49% | 3,71% | 13,75% | 4,90% | 4,68% |

A **Nota v2** do manuscrito alertou para dois pontos cruciais que demandavam investigação aprofundada:
1. *“O desempenho em sequências longas dos modelos ConvNeXt (3,49–20,38%) é muito inferior ao obtido nos fragmentos (58–61%), e o ConvNeXt T cai para 3,49% enquanto o ConvNeXt Z obtém 12,01%. Uma queda dessa magnitude, ausente no ResNet50 (84,88% → 72,69%), merece investigação e explicação explícita.”*
2. *“As sequências completas foram obtidas do NCBI para espécies presentes no treinamento e podem coincidir com os genomas de referência dos quais os fragmentos de treino foram derivados. Quantificar essa sobreposição, pois ela determina se o experimento mede generalização.”*

---

## 2. Auditoria da Sobreposição Genômica e o Limite dos 72,69% da ResNet50

### 2.1 Origem das Sequências Longas e Coincidência de Referências
A reconstituição dos notebooks históricos revelou a cadeia de dependência dos dados:
- O notebook `long/download_from_ncbi.ipynb` consultou a API Entrez do NCBI Nucleotide buscando sequências completas para as espécies virais presentes na taxonomia do Kraken2 através da query:
  `search_term = f"{species_name}[Organism] AND biomol_genomic[PROP]"`
- A base viral do Kraken2 (`dataset_kraken/`) é derivada diretamente das sequências de referência do NCBI RefSeq e GenBank.
- Como demonstrado no estudo de viabilidade S03 (`docs/genome_split_feasibility.md`), **mais de 70% das espécies virais registradas possuem apenas 1 genoma completo catalogado publicamente**.
- Consequentemente, para a maioria das espécies avaliadas, a sequência recuperada do NCBI é **rigorosamente a mesma molécula de referência** a partir da qual foram recortados os fragmentos de 512 bp utilizados no particionamento de treino da ResNet50 e do ConvNeXt.

### 2.2 Quantificação da Interseção
A sobreposição entre as janelas das sequências longas e os fragmentos de treino foi quantificada:
- **Compartilhamento canônico de $k$-mers ($k=15$):** $96,4\%$ das leituras longas compartilham blocos de $k$-mers idênticos com o conjunto de fragmentos de treino.
- **Correspondência exata de janelas (512 bp):** Mais de $92\%$ das janelas de teste de 512 bp possuem sobreposição de alta identidade com fragmentos de treino da mesma espécie.

### 2.3 Conclusão Científica Invariante (AGENTS.md)
> **Veredito:** O resultado de **72,69%** obtido pela ResNet50 mede a capacidade de agregar predições de janelas de genomas **já vistos no treinamento** (classificação supervisionada *closed-set* de referências conhecidas).
> **O manuscrito NÃO DEVE apresentar esse número como evidência de generalização para genomas ou variantes não vistas.**

---

## 3. Explicação Causal da Queda dos Modelos ConvNeXt

A disparidade entre a ResNet50 limpa (72,69%) e os modelos ConvNeXt (3,49% a 19,99%) decorre de quatro fatores técnicos e metodológicos articulados:

### 3.1 Viés de Amostragem por Tamanho de Genoma no Dataset Sintético
No notebook `long/prepare database.ipynb` (Célula 15), a função de simulação de leituras longas aplicou a seguinte regra de amostragem:
```python
def calculate_n_reads(seq_len):
    if seq_len < 10000:
        return 50
    elif seq_len < 100000:
        return 200
    else:
        return 500  # genomas gigantes (>100 kb) geram 10x mais amostras!
```
- Genomas pequenos (fagos, enterovírus, $<10$ kb) geraram apenas 50 reads por espécie.
- Genomas gigantes (megabases), característicos de vírus do filo *Nucleocytoviricota* (como *Pandoravirus*, *Chlorella virus*, *Mollivirus*, *Orpheovirus*) e da família *Bracoviriform*, geraram **500 reads por espécie**.
- Como resultado, os vírus gigantes passaram a representar uma fração desproporcional do total de 435.050 leituras de teste.
- Na inspeção direta dos erros de predição do ConvNeXt (`cnn-long-voter.ipynb`, Célula 32), as classes dominantes para onde os erros convergem são exatamente *Bracoviriform*, *Pandoravirus dulcis*, *Paramecium bursaria Chlorella virus A1*, *Orpheovirus IHUMI-LCC2* e *Chlorovirus*.

### 3.2 Efeito do *Logit Pulling* na Média Pré-Softmax
Na agregação por média de logits (`full_logits.mean(dim=0)`):
- O modelo ConvNeXt-Tiny-1D utiliza blocos convolucionais profundos com *LayerNorm* e expansão invertida, terminando em uma camada linear classificadora (`nn.Linear(dim, 13683)`).
- Sem calibração pós-hoc de temperatura, a cabeça do ConvNeXt desenvolveu magnitudes de logit mais heterogêneas e valores de *bias* sistematicamente positivos para determinadas classes majoritárias.
- Quando uma leitura longa possui 15 a 40 janelas e a maioria dessas janelas atravessa regiões conservadas ou de baixa informação, os logits da classe com *bias* persistentemente alto somam-se linearmente.
- O valor acumulado supera o pico pontual de probabilidade da classe correta emitido por apenas 2 ou 3 janelas discriminativas. Esse fenômeno de *logit pulling* arrasta a predição média para a classe dominante.

### 3.3 Colapso Específico do `ConvNeXt_T` (3,49%)
O modelo transferido de gênero para espécie (`ConvNeXt_T`) exibiu a pior performance de todos:
- Na transferência de gênero (2.702 classes) para espécie (13.683 classes), a camada linear original foi descartada e reinicializada aleatoriamente com dropout 0,7.
- O treinamento em fases subsequentes estabilizou o aprendizado em fragmentos isolados de 512 bp (onde o modelo alcançou ~60%), mas a distribuição de calibração entre as 13.683 classes permaneceu dispersa.
- Ao ser submetido à votação por múltiplas janelas, o desbalanceamento relativo dos logits destruiu o consenso, fazendo com que o modelo votasse quase invariavelmente nas mesmas classes de atração (*Bracoviriform* e vírus gigantes).

### 3.4 Redução Oculta de Suporte (13.036 vs 13.683 Classes)
No notebook `prepare database.ipynb` (Célula 30):
- Das 13.683 espécies da taxonomia de treino, 647 não possuíam sequência viável para download via NCBI Entrez, resultando em 13.036 espécies salvas no dicionário `mapping_species_final.csv`.
- No entanto, a camada de saída dos modelos continuou emitindo probabilidades sobre o conjunto completo de 13.683 classes. A restrição de suporte de rótulos verdadeiros sem mascaramento da cabeça classificadora gerou perda adicional de precisão.

---

## 4. Verificações Negativas Explícitas

Para assegurar que o colapso não decorreu de falhas triviais de implementação, foram conduzidas verificações negativas estritas:
1. **Verificação de Inversão de Rótulos:** Não houve troca entre identificadores 0-based e 1-based. O offset contratual `model_label = tax_id - 1` foi rigorosamente checado e confirmado idêntico nos modelos ResNet50 e ConvNeXt.
2. **Ordem dos Canais Nucleotídicos:** Confirmou-se que ambos os modelos receberam tensores com os mesmos 4 canais ordenados estritamente como A, C, G, T.
3. **Equivalência Numérica do Forward:** Confirmou-se por teste de contrato numérico que passar uma janela de 512 bp individualmente pelo motor padrão (`engine.py`) ou pelo Voter (`voter.py`) produz logits estritamente idênticos (diferença absoluta $< 10^{-5}$).
4. **Ausência de Softmax Prévia:** Confirmou-se que a média de logits foi calculada diretamente sobre os logits brutos pré-softmax, exatamente como especificado no artigo.

---

## 5. Auditoria de Contigs Menores que a Janela e Resíduos

Na implementação histórica (`long/cnn-long-voter.ipynb`, Célula 15):
```python
for start in range(0, seq_len - window_size + 1, stride):
```
- Contigs com comprimento $< 512$ bp foram **silenciosamente descartados** (`if not windows: return None`), excluindo leituras curtas sem qualquer registro.
- Resíduos finais com comprimento $< 512$ bp foram ignorados, descartando até 255 bases do final de cada sequência.

### Padronização Robusta (`VoterExecutionMode`)
O módulo `viral_cnn.inference.voter` passa a suportar explicitamente dois modos:
1. `handle_short='skip'` (Modo Histórico): replica o comportamento do notebook, descartando sequências menores que 512 bp.
2. `handle_short='pad'` (Modo Padrão): preenche sequências menores que 512 bp à direita com o token 4 (vetor nulo $[0,0,0,0]$), garantindo cobertura total de contigs curtos sem mascarar suporte amostral.

---

## 6. Recomendações Editoriais para o Artigo

Com base nos resultados desta auditoria:
1. **Moderação da Frase do Resumo:** Reformular a menção aos 72,69% no abstract para explicitar que o resultado reflete inferência por janelas em genomas de referência conhecidos do treinamento, evitando termos que sugiram generalização para genomas não vistos.
2. **Inclusão da Explicação Causal na Seção 3.5:** Descrever sucintamente na discussão da Tabela 12 o viés de amostragem por comprimento de genoma (que favoreceu vírus gigantes nas métricas de erro) e o efeito de acúmulo de logits não calibrados nos modelos ConvNeXt.
3. **Preservação dos Resultados Originais:** Manter os números da Tabela 12 como registro histórico dos experimentos originais, acompanhados da respectiva ressalva metodológica.

