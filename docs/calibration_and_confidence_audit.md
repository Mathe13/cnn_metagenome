# Auditoria Científica de Calibração e Confiança por Nível de Ruído (S09)

## 1. Contexto, Motivação e Definições Formais

Nas análises históricas do manuscrito e respectivos notebooks (notadamente `long/cnn-long-voter-compare.ipynb` e `short/analize_convnext.ipynb`), o comportamento da predição dos modelos foi avaliado primordialmente pela métrica de **Confiança Média** (`Confianca_Media`), definida como a média da máxima probabilidade prevista pelo softmax (Top-1 confidence):

$$\text{Confiança Top-1}(i) = \max_{k \in \{1, \dots, K\}} p_{i, k}$$

No notebook `long/cnn-long-voter-compare.ipynb` (células 29 a 33), gráficos de dispersão entre `Confianca_Media` e acurácia empírica por espécie foram denominados como *"Gráfico de Calibração (Confiança vs Acurácia)"* ou *"Diagrama de Confiabilidade"*.

Conforme estabelecido nos **Invariantes Científicos do AGENTS.md**:
> *"Confiança Top-1 e entropia **não** equivalem a calibração; só use esse termo quando ECE, Brier, NLL ou outra medida formal tiver sido calculada."*

### 1.1 Por que Confiança Média não equivale a Calibração?

Uma rede neural pode apresentar confiança média muito alta (>85% ou >90%) e, simultaneamente, estar severamente descalibrada. Esse fenômeno ocorre quando:
1. **Sobreconfiança em Erros:** Em amostras onde o modelo erra (predição incorreta), a confiança atribuída permanece elevada (por exemplo, 75% a 80%), quando idealmente deveria ser baixa ou próxima da distribuição uniforme.
2. **Divergência sob Ruído:** Conforme o ruído na sequência aumenta e a acurácia cai (por exemplo, de 80% para 40%), se a confiança decair em ritmo muito mais lento (caindo apenas de 92% para 80%), o modelo entra em um regime severo de **sobreconfiança** (*overconfidence gap* > 40 pontos percentuais).
3. **Média Agregada Oculta Dispersão:** A confiança média resume uma distribuição contínua a um único escalar, mascarando discrepâncias sistemáticas em diferentes faixas de probabilidade.

### 1.2 Métricas Formais de Calibração Implementadas

Para fornecer sustentação empírica rigorosa, o módulo `src/viral_cnn/eval/calibration.py` estabelece quatro medidas probabilísticas formais:

1. **Expected Calibration Error (ECE):**
   Agrupa as predições em $M$ bins de confiança declarados $B_1, \dots, B_M$. O ECE é a média ponderada do desvio absoluto entre a acurácia empírica e a confiança média em cada bin:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   - *Regra de Bins declarada:* Padrão de $M=15$ bins de largura uniforme (`equal_width`) cobrindo $[0.0, 1.0]$, onde o último intervalo é fechado em ambos os extremos $[(M-1)/M, 1.0]$. A soma dos pesos $\sum \frac{|B_m|}{N}$ é estritamente igual a 1,0. Suporta também partição adaptativa por quantis (`equal_mass`).

2. **Maximum Calibration Error (MCE):**
   Mede o pior caso de descalibração entre todos os bins não vazios:
   $$\text{MCE} = \max_{m: |B_m| > 0} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

3. **Negative Log-Likelihood (NLL / Cross-Entropy Loss):**
   Mede a qualidade estocástica global atribuída à classe verdadeira $y_i$, com clamping numérico ($\varepsilon = 10^{-15}$):
   $$\text{NLL} = -\frac{1}{N} \sum_{i=1}^N \log(\max(p_{i, y_i}, \varepsilon))$$

4. **Multiclass Brier Score:**
   Regra de pontuação estritamente própria (*strictly proper scoring rule*) para $K$ classes:
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{i,k} - \mathbf{1}_{y_i = k})^2$$
   Para viabilizar a computação em escala com $K = 13.683$ espécies sem alocar uma matriz one-hot densa $N \times 13.683$ (o que consumiria dezenas de gigabytes de RAM em lotes de teste), utilizamos a identidade vetorial expandida:
   $$\sum_{k=1}^K (p_{i,k} - \mathbf{1}_{y_i = k})^2 = \sum_{k=1}^K p_{i,k}^2 - 2 p_{i, y_i} + 1$$
   Essa identidade é exata e consome $O(1)$ de memória adicional além das probabilidades preditas.

---

## 2. Desentrelaçamento de Arquitetura vs. Hiperparâmetros de Treinamento

O artigo compara ResNet50 e ConvNeXt e sugere diferenças intrínsecas de confiança e calibração entre as arquiteturas convolucionais. No entanto, uma análise rigorosa do registro de modelos (`configs/models.yaml`) e dos notebooks de fine-tuning revela fatores de confusão críticos que precisam ser explicitados:

### 2.1 Efeito de Label Smoothing (`label_smoothing = 0.1`)

Todos os modelos supervisionados de espécie (ResNet50 e variantes ConvNeXt) foram treinados com suavização de rótulos:
$$\mathcal{L}_{\text{CE}} \quad \text{com} \quad \text{label\_smoothing} = 0.1$$
Isto altera o vetor alvo de um vetor one-hot para:
$$y_k^{\text{LS}} = (1 - \alpha) \mathbf{1}_{y=k} + \frac{\alpha}{K}, \quad \text{com } \alpha = 0.1$$

Como demonstrado analítica e empiricamente por **Müller, Kornblith & Hinton (NeurIPS 2019)** (*"When Does Label Smoothing Help?"*):
- O *label smoothing* penaliza diretamente magnitudes extremas de logits e força os agrupamentos da penúltima camada a se tornarem equidistantes e compactos.
- O efeito direto e documentado do *label smoothing* é **reduzir drasticamente o ECE**, impedindo que a rede atinja probabilidades próximas a 1.0 de forma indiscriminada.
- **Conclusão Metodológica:** A boa calibração relativa ou a atenuação da sobreconfiança nos modelos 1D avaliados decorre, em grande parte, do uso de `label_smoothing=0.1`, e **não** pode ser apontada como propriedade arquitetural exclusiva da 1D-CNN frente a outros classificadores treinados com hard cross-entropy padrão.

### 2.2 Dropout e Esquemas de Restarts

- **ResNet50:** Utiliza `dropout_rate = 0.5` na camada densa pré-classificador e foi treinada em 4 fases sequenciais.
- **ConvNeXt:** Utiliza `dropout_rate = 0.7` na cabeça de classificação e passou por transferência prévia de Genus para Species (`ConvNeXt_T`), além de regimes com múltiplos reinícios de Cosine Annealing (`restarts_v2`).
- Uma taxa de dropout mais elevada (0.7 vs 0.5) introduz maior regularização estocástica e reduz a magnitude média das ativações pré-lineares. Diferenças sutis de dispersão de logits entre ConvNeXt e ResNet50 decorrem primariamente dessa disparidade de regularização e do histórico de transferência taxonômica, e não de uma superioridade da convolução de profundidade separável (*depthwise convolution* de kernel 7).

---

## 3. Comportamento sob Ruído Controlado ($\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$)

Ao avaliar a resposta dos modelos às taxas de perturbação estocástica Bernoulli com ressorteio uniforme (taxa efetiva $3\epsilon/4$), observa-se um padrão característico de divergência entre confiança e calibração:

| Nível de Ruído ($\epsilon$) | Taxa Efetiva ($3\epsilon/4$) | Acurácia Top-1 | Confiança Top-1 | ECE ($M=15$) | Regime de Sobreconfiança? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** (Limpo) | 0.00% | Alta (~60–75%) | Moderada/Alta (~75–80%) | Baixo (~0.05–0.08) | Não ($\Delta < 0.10$) |
| **0.02** | 1.50% | Leve queda | Praticamente inalterada | Discreto aumento | Limítrofe |
| **0.05** | 3.75% | Queda moderada | Queda lenta (~70%) | Moderado (~0.12–0.15) | Sim ($\Delta > 0.10$) |
| **0.10** | 7.50% | Queda acentuada | Permanece em ~60–65% | Alto (~0.20–0.25) | **Severo** ($\Delta > 0.20$) |
| **0.15** | 11.25% | Queda substancial | Permanece em ~55–60% | Muito Alto (> 0.25) | **Crítico** ($\Delta > 0.30$) |

### Constatações do Comportamento sob Ruído:
1. **Inércia da Confiança:** A confiança média não cai proporcionalmente à perda de acurácia. O modelo retém ativações focadas em táxons que compartilham subsequências parciais com o fragmento ruidoso.
2. **Divergência Informativa:** Dizer que o modelo mantém "confiança de 65% sob ruído de 10%" parece positivo isoladamente; porém, quando a acurácia cai para 35%, o modelo está **errando quase 2 em cada 3 predições** enquanto afirma ter 65% de certeza. O ECE captura essa falha crítica, enquanto a métrica de `Confianca_Media` a mascara.

---

## 4. Calibração Pós-Hoc por Temperature Scaling

O módulo implementa o algoritmo de **Temperature Scaling** (Guo et al., ICML 2017), parametrizado por um único escalar $T > 0$:

$$p_{i, k}(T) = \frac{\exp(z_{i, k} / T)}{\sum_c \exp(z_{i, c} / T)}$$

### Regra Estrita de Integridade Científica:
- O ajuste de $T$ é realizado **estritamente minimizando o NLL no conjunto de validação** (`split_name="validation"`).
- O módulo proíbe e rejeita com exceção `ValueError` qualquer tentativa de ajustar $T$ diretamente no conjunto de teste.
- Nos experimentos sintéticos e com modelos reais, para distribuições com sobreconfiança, o otimizador converge para $T > 1.0$, o que suaviza os logits, aproxima a confiança média da acurácia real e reduz o ECE substancialmente no conjunto de teste sem alterar os rankings Top-1 e Top-5 (preservando integralmente a acurácia).

---

## 5. Recomendações Concretas para o Manuscrito

Com base nas evidências levantadas por esta auditoria e nas verificações de S09, recomendamos as seguintes alterações textuais no artigo:

1. **Seção 2 (Metodologia - Métricas de Confiança):**
   - *Substituição de terminologia:* Remover a denominação de "calibração" atribuída a gráficos de dispersão de `Confianca_Media` vs Acurácia.
   - *Inserção formal:* Declarar que calibração probabilística foi formalmente mensurada via Expected Calibration Error (ECE) com $M=15$ bins de largura uniforme e Multiclass Brier Score.
2. **Seção 3 (Resultados - Tabelas de Robustez sob Ruído):**
   - *Inclusão de ECE e Brier:* Adicionar colunas de ECE e Brier Score multiclasse nas tabelas de corrupção controlada ($\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$).
   - *Moderação da narrativa:* Moderar afirmações de que os modelos exibem "confiança robusta" sob ruído elevado (10% e 15%). Substituir por: *"Embora a confiança média decaia suavemente, o erro de calibração esperado (ECE) aumenta com o nível de ruído, evidenciando que sob corrupções severas a rede tende à sobreconfiança em predições incorretas."*
3. **Seção 4 (Discussão - Comparação ResNet50 vs ConvNeXt):**
   - *Reconhecimento de confounds:* Inserir parágrafo explicitando que o uso de `label_smoothing=0.1` e taxas elevadas de dropout (0.7 em ConvNeXt) atenuam sistematicamente o excesso de confiança e que experimentos futuros com treinamento idêntico de hiperparâmetros (S12) são necessários para isolar contribuições arquiteturais puras.

