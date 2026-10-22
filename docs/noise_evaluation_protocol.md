# Protocolo de Avaliação Limpa e Corrupção Controlada por Ruído

Este documento estabelece o protocolo científico formal de avaliação e perturbação artificial de sequências de DNA para os modelos neurais 1D do artigo **“Spatial one-hot nucleotide representations of DNA enable robust and interpretable taxonomic classification with 1D convolutional networks”**.

---

## 1. Protocolo Histórico de Perturbação de Nucleotídeos

### 1.1 Mecanismo de Sorteio Bernoulli e Ressorteio Uniforme
A corrupção artificial de sequências simula taxas de erro de sequenciamento através do seguinte processo estocástico:
1. Para cada posição $i$ da sequência $S = (s_1, \dots, s_L)$, uma variável aleatória Bernoulli $B_i \sim \text{Bernoulli}(\epsilon)$ determina se a posição é selecionada para mutação, onde $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$ é a **taxa nominal de ruído**.
2. Quando $B_i = 1$, um novo nucleotídeo $X_i$ é sorteado de forma **uniforme e equiprovável** entre os quatro nucleotídeos canônicos $\{A, C, G, T\}$ (índices $\{0, 1, 2, 3\}$):
   $$P(X_i = b) = \frac{1}{4}, \quad \forall b \in \{A, C, G, T\}$$
3. A nova base $X_i$ substitui $s_i$, mesmo que $X_i$ coincida com a base original $s_i$.

### 1.2 Prova Matemática da Taxa Efetiva de Substituição ($3\epsilon/4$)
Como o sorteio uniforme inclui a própria base original com probabilidade $1/4$, uma posição sorteada só sofre uma mutação efetiva se a nova base for diferente da base preexistente:
$$P(X_i \ne s_i \mid B_i = 1) = \frac{3}{4}$$

Pela lei da probabilidade total, a probabilidade incondicional de que uma posição sofra alteração efetiva de nucleotídeo é:
$$P(s_i' \ne s_i) = P(B_i = 1) \cdot P(X_i \ne s_i \mid B_i = 1) = \epsilon \cdot \frac{3}{4} = \frac{3\epsilon}{4}$$

A tabela a seguir relaciona as taxas nominais $\epsilon$ e suas respectivas taxas efetivas esperadas:

| Taxa Nominal ($\epsilon$) | Taxa Efetiva Esperada ($3\epsilon/4$) | Percentual Efetivo de Substituição |
| :---: | :---: | :---: |
| `0.00` | $0.0000$ | $0.00\%$ (Avaliação Limpa) |
| `0.02` | $0.0150$ | $1.50\%$ |
| `0.05` | $0.0375$ | $3.75\%$ (Fine-tuning de Ruído) |
| `0.10` | $0.0750$ | $7.50\%$ |
| `0.15` | $0.1125$ | $11.25\%$ |

### 1.3 Preservação Estrita de Padding e Bases Ambíguas
- Posições de preenchimento (*padding*) ou bases ambíguas não resolvidas recebem índice `-1` (resultando no vetor espacial nulo $[0, 0, 0, 0]$).
- O operador de perturbação aplica máscara booleana estrita `(sequences >= 0)`, garantindo que posições de padding **nunca** sejam convertidas em bases válidas durante a injeção de ruído.

---

## 2. Métricas de Classificação e Faixas de Desempenho

### 2.1 Métricas Padrão Avaliadas
- **Top-1 Accuracy:** Proporção de consultas onde a classe de maior probabilidade prevista coincide com o rótulo verdadeiro.
- **Top-5 Accuracy:** Proporção de consultas onde o rótulo verdadeiro está presente entre as 5 classes mais prováveis.
- **Cross-Entropy Loss Multiclasse:** Média da perda logarítmica sobre todas as predições.
- **Macro-F1 Score:** Média aritmética não ponderada do F1-score calculada sobre todas as classes taxonômicas avaliadas ($N_{\text{classes}} = 13.683$ para Espécie, $2.702$ para Gênero):
  $$\text{Macro-F1} = \frac{1}{N} \sum_{c=1}^N \text{F1}_c$$
  Classes sem amostras no conjunto de teste recebem $\text{F1}_c = 0.0$ conforme convenção padrão do `scikit-learn` (`zero_division=0`).
- **Weighted-F1 Score:** Média do F1-score ponderada pelo suporte (número de instâncias verdadeiras) de cada classe.

### 2.2 Particionamento em Faixas Contínuas de F1 por Espécie
Para analisar a distribuição de acurácia por classe sem lacunas ou sobreposições artificiais, os F1-scores de cada espécie são agrupados em quatro intervalos contínuos e exaustivos:
1. `[0.00, 0.50)`: Baixo desempenho / não resolvidas
2. `[0.50, 0.80)`: Desempenho intermediário
3. `[0.80, 0.95)`: Alto desempenho
4. `[0.95, 1.00]`: Desempenho excelente (fechado à direita em 1.0)

A soma das espécies distribuídas nas quatro faixas conserva rigorosamente o total de classes do modelo:
$$\sum_{b \in \text{Bands}} N_{\text{species}}(b) = N_{\text{classes}}$$

---

## 3. Comparações Pareadas entre Modelos

Para avaliar diferenças estatisticamente consistentes entre dois modelos $A$ e $B$ avaliados sob o mesmo nível de ruído nas mesmas consultas:
1. **Diferença em Pontos Percentuais ($\Delta_{pp}$):**
   $$\Delta_{pp} = 100 \times (\text{Top1}_A - \text{Top1}_B)$$
2. **Tabela de Contingência Pareada:**
   - `both_correct`: $A$ correto e $B$ correto
   - `only_a_correct`: $A$ correto e $B$ incorreto
   - `only_b_correct`: $A$ incorreto e $B$ correto
   - `both_wrong`: $A$ incorreto e $B$ incorreto

---

## 4. Invariante Epistemológico: Bootstrap de Fragmentos vs Incerteza entre Genomas

Conforme o princípio científico estipulado em `AGENTS.md` e na especificação S05:
- O cálculo de intervalos de confiança via **bootstrap de fragmentos de teste** mensura unicamente a variabilidade amostral do particionamento de fragmentos de consulta.
- O bootstrap de fragmentos **não** representa, sob hipótese alguma, a incerteza filogenética entre genomas independentes ou a variância induzida por diferentes sementes de treinamento estocástico dos modelos neurais.
- Toda figura ou tabela resultante que utilize bootstrap deve explicitar que o intervalo é condicional ao conjunto específico de fragmentos avaliados.
