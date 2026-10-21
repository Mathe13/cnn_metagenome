# Sequência de specs para o repositório do artigo

Este roteiro foi preparado para usar com Codex e `tlc-spec-driven`. Ele parte do artigo sobre ResNet50-1D e ConvNeXt-1D na classificação viral e das notas da revisão. **A árvore do repositório de código não foi disponibilizada aqui**: os nomes abaixo são identificadores de specs sugeridos, não caminhos ou comandos já existentes. A primeira spec deve identificar a estrutura real antes de propor mudanças.

## Regra comum a todas as specs

Cada spec precisa identificar entradas e saídas, subconjunto de dados, checkpoint, versão do mapeamento de classes, semente, ambiente, recurso computacional necessário, teste pequeno e critério de aceitação. Preserve o experimento histórico e dê novo identificador a qualquer protocolo alterado. Um teste sintético valida um contrato de código; reproduzir uma tabela exige os artefatos reais e uma comparação registrada com os números do manuscrito. Não faça treinamento completo, download em massa ou publicação como efeito colateral de um teste.

Use o `AGENTS.md` como regra permanente do projeto. Mantenha em `.specs/` as decisões de cada mudança conforme o fluxo da skill, sem duplicar no `AGENTS.md` instruções transitórias ou resultados de um experimento.

## Ordem proposta

### S01 — Inventário, organização e extração gradual dos notebooks

**Objetivo:** tornar o repositório navegável e iniciar a execução por scripts antes de mexer nos resultados.

**Escopo:** levantar todos os notebooks, módulos, dados, checkpoints e saídas; construir o mapa `notebook/célula → entrada → saída → tabela/figura`; classificar notebooks como históricos, exploratórios ou executáveis; propor a estrutura real para `src/`, `scripts/`, `configs/`, `notebooks/`, `tests/` e `docs/`. Extrair primeiro um fluxo vertical pequeno (por exemplo, leitura/codificação de um lote e inferência limpa), com parâmetros explícitos e um ponto de entrada de linha de comando. Para os demais notebooks, registrar a ordem de extração nas specs seguintes. Manter notebooks de visualização quando agregarem valor e preservar os originais identificáveis.

**Aceitação:** inventário de todos os notebooks e artefatos encontrados; dependências e caminhos externos documentados; ao menos um fluxo extraído executa em lote pequeno; saídas numéricas desse fluxo coincidem com o notebook para as mesmas entradas/checkpoint, dentro da tolerância explicitada; não há alteração silenciosa de dados, rótulos, arquitetura ou métricas. Se dados/checkpoints faltarem, registrar precisamente o bloqueio e validar apenas contratos sintéticos.

**Prompt inicial para o Codex:**

> Use `tlc-spec-driven` e leia `AGENTS.md`. Especifique e implemente a primeira etapa de organização deste repositório científico. Antes de mover arquivos, inventarie notebooks, módulos, configurações, datasets, checkpoints e saídas e relacione-os às tabelas e figuras do artigo. Preserve os notebooks originais. Proponha a menor estrutura de pacote Python, scripts e configurações que sirva ao código já existente e extraia um único fluxo vertical de leitura/codificação/inferência limpa. Demonstre equivalência com a execução histórica em um lote verificável; registre lacunas de artefatos e não rode treinamento completo. Não invente comandos, caminhos ou valores ausentes.

### S02 — Proveniência do dataset e auditoria do split por fragmento

**Depende de:** S01.

**Objetivo:** saber exatamente o que uma amostra representa e quais genomas aparecem em cada partição.

**Escopo:** versionar ou reconstruir o manifesto de referência NCBI/ICTV/Kraken2, accession, taxid, Species/Genus/Order, identificador de genoma, coordenadas de fragmentação, janela/stride, regras de inclusão, balanceamento e split. Contar fragmentos e genomas por classe e partição. Medir duplicação exata, sobreposição por coordenadas e, quando viável, alta similaridade entre partições. Distinguir campos observados dos reconstruídos por inferência.

**Aceitação:** totais auditáveis para treino/validação/teste; taxa de fragmentos idênticos e sobrepostos; número de genomas compartilhados e classes com um único genoma; relatório do que pode e do que não pode ser concluído sobre vazamento. Nenhuma estimativa de sobreposição é chamada de zero apenas porque não se acharam sequências idênticas.

### S03 — Viabilidade de avaliação por genoma de origem

**Depende de:** S02.

**Objetivo:** medir a generalização dentro de espécies conhecidas sem misturar fragmentos do mesmo genoma entre treino e teste.

**Escopo:** identificar espécies com pelo menos dois genomas independentes; construir split agrupado por accession/genoma, com espécies presentes no treino, ou declarar inviabilidade para as classes sem suporte. Registrar cobertura taxonômica e distribuição por clado. Não substituir o split histórico 80/10/10 nem confrontar suas acurácias com as do subconjunto novo sem explicar a mudança de população.

**Aceitação:** decisão documentada sobre viabilidade, IDs disjuntos por grupo, taxa de cobertura por espécie, manifesto do novo split e plano de treinamento/avaliação sob esse protocolo. Executar o novo treinamento só como tarefa separada com orçamento e checkpoints definidos. Se a proveniência estiver ausente, registrar as evidências faltantes e a limitação correspondente no artigo.

### S04 — Registro de modelos, mapeamentos e treinamento

**Depende de:** S01; usa o manifesto de S02.

**Objetivo:** poder carregar e identificar cada um dos cinco modelos de espécie e os modelos de gênero sem confusão de cabeças ou classes.

**Escopo:** extrair definições de ResNet50-1D, ConvNeXt-1D e carregamento dos checkpoints; documentar canal A/C/G/T, padding e bases ambíguas. Para cada variante, registrar checkpoint de origem, mapeamento índice→taxid, fases, épocas, otimizador, scheduler, batch size, `label_smoothing`, AMP, GPU e critério **real** de seleção. Distinguir último checkpoint de melhor checkpoint. Centralizar parâmetros em configuração, preservando valores históricos.

**Aceitação:** cargas de checkpoint verificadas com classes e formas corretas; predição de um mesmo lote coincide com o notebook histórico; manifesto por execução e script de inferência sem caminhos pessoais. Lacunas de hiperparâmetros ficam marcadas como não verificadas, nunca preenchidas por suposição.

### S05 — Avaliação limpa e corrupção controlada reproduzíveis

**Depende de:** S02 e S04.

**Objetivo:** obter a fonte única de predições e métricas das tabelas principais.

**Escopo:** extrair avaliação limpa, Top-1, Top-5, perda, Macro-F1, Weighted-F1, F1 por espécie e faixas de desempenho. Implementar a perturbação histórica: seleção Bernoulli por posição com $\epsilon\in\{0,0.02,0.05,0.10,0.15\}$ e ressorteio entre A/C/G/T, com taxa efetiva esperada $3\epsilon/4$. Salvar IDs das consultas, sementes e predições por condição; medir variação entre realizações de ruído, se forem executadas várias. Preparar comparações pareadas e intervalos condicionais ao conjunto de teste, levando em conta agrupamento por genoma quando possível.

**Aceitação:** arquivos de predições rastreáveis; métricas refeitas a partir desses arquivos; diferenças em pontos percentuais corretas; limites de F1 contínuos e sem lacunas; divergências em relação ao paper listadas com causa conhecida ou como pendência. Bootstrap de fragmentos não é apresentado como incerteza entre genomas ou entre sementes de treinamento.

### S06 — Auditoria da inferência de sequências longas

**Depende de:** S02, S04 e S05.

**Objetivo:** explicar a queda grande dos ConvNeXt e conferir os 72,69% do ResNet50 antes de usar esse resultado como evidência central.

**Escopo:** extrair o `Voter` com janela 512, stride 256, média de logits pré-softmax, voto, empate e tratamento de contigs menores que a janela. Comparar logits da mesma janela pelo caminho curto e pelo caminho de sequências longas; checar mapeamento de rótulos, IUPAC, padding, cabeças, distribuição de comprimentos, composição e identificadores genômicos. Quantificar interseção das sequências longas com genomas de treino. Estratificar erro por espécie, comprimento e proporção de caracteres ambíguos.

**Aceitação:** relatório de causa da queda ou diagnóstico inconclusivo com verificações negativas explícitas; resultados anteriores preservados; resultados corrigidos recebem novo ID se houver erro; afirmação sobre genomas não vistos só aparece se a ausência de sobreposição foi demonstrada.

### S07 — Baselines Kraken2 e BLASTN no mesmo protocolo

**Depende de:** S02 e S05; considerar S06 se incluir sequências longas.

**Objetivo:** aproveitar os notebooks de benchmark já produzidos e transformá-los em comparação rastreável.

**Escopo:** integrar consultas e níveis de ruído usados nas CNNs; registrar base, versão, táxons cobertos, referência presente/ausente, conversão de IDs, regra de abstinência e denominadores. Gerar comparação pareada nas mesmas consultas. Se um método acessa o próprio genoma de origem da consulta, reportar esse cenário separadamente. Um baseline de frequências de $k$-mers é extensão opcional após Kraken2/BLASTN.

**Aceitação:** tabela por método, $\epsilon$, cobertura, Top-1 e diferenças pareadas com IC apropriado; nenhum método avaliado em conjunto de consultas diferente é rotulado como comparação direta; fonte das predições existente e comando de reprodução documentados.

### S08 — Reavaliação do espaço latente

**Depende de:** S02, S04 e S05.

**Objetivo:** verificar KNN, Silhouette e correlações taxonômicas sem confundir visualização, teste supervisionado e generalização.

**Escopo:** extrair embeddings pré-classificador e reproduzir KNN ($k=5$, cosseno, normalização, referência/consulta 80/20 disjuntas) e Silhouette no espaço original. Registrar taxa de espécies válidas no subconjunto de até 50 mil; considerar amostragem estratificada. Repetir KNN com grupos de genomas disjuntos se os IDs existirem. Verificar os números de gêneros/ordens e as correlações Pearson/Spearman. Se investigar o aparente “gargalo” do ConvNeXt, treinar uma sonda linear exclusivamente em dados de treino/validação; o KNN que usa rótulos do teste não prova gargalo na cabeça.

**Aceitação:** parâmetros, $N$ por nível, classes excluídas, tabela corrigida e figura UMAP identificada como qualitativa; inferências sobre filogenia ou a cabeça do classificador não ultrapassam as evidências.

### S09 — Calibração e confiança por nível de ruído

**Depende de:** S05.

**Objetivo:** sustentar ou moderar a parte central do artigo sobre comportamento da confiança.

**Escopo:** computar ECE com regra de bins declarada, NLL, Brier score multiclasse e diagramas de confiabilidade no mesmo conjunto e para cada $\epsilon$. Comparar modelos sem confundir diferença arquitetural com `label_smoothing` e cronogramas distintos. Calibração posterior, se usada, deve ajustar parâmetros na validação, nunca no teste.

**Aceitação:** resultados reproduzíveis, denominação correta de “calibração” e alteração do texto quando confiança média e erro de calibração contarem histórias diferentes.

### S10 — Motivos, controle composicional e tripletos

**Depende de:** S02, S04 e S05.

**Objetivo:** determinar o que resiste a controles para homopolímeros e composição.

**Escopo:** extrair consenso dos filtros de primeira camada; reproduzir perturbação de 7-mers e controle pareado original com sementes/repetições/FDR. Adicionar análise de sensibilidade com controle pareado por GC e/ou complexidade local, mantendo o resultado histórico separado. Inspecionar os 200 pares de coocorrência: testar dependência de orientação dos pares e construir análise simétrica/nula adequada antes de alegar significância. Reavaliar RSCU em fragmentos sem quadro biológico conhecido como normalização exploratória de composição de tripletos.

**Aceitação:** figura com cada painel ligado ao checkpoint correto, controles explicitados, correção de testes múltiplos, conclusão limitada a efeito preditivo local; se o controle composicional não sustentar o sinal, mover a análise ampliada para suplemento.

### S11 — Regeneração de tabelas/figuras e revisão final do manuscrito

**Depende de:** S05–S10 concluídas ou documentadas como indisponíveis.

**Objetivo:** fazer cada número e figura do artigo apontar para um experimento reproduzível e ajustar as afirmações à evidência resultante.

**Escopo:** gerar tabelas e figuras de arquivos de predição e manifestos; checar denominadores, arredondamentos, cinco empates da comparação por espécie, nomes de arquivos com maiúsculas/minúsculas e painel ConvNeXt correto. Atualizar título, resumo estruturado, contribuição frente a GeNet, Methods, limitações, referências, versões ICTV/NCBI e declarações de disponibilidade. Remover macros e notas de revisão na versão de submissão, mantendo uma cópia revisável em separado.

**Aceitação:** tabela `resultado no paper → script → configuração → dados/checkpoint`; compilação limpa em sistema sensível a maiúsculas; valores concordam com saídas verificadas ou trazem discrepância explicada; se a auditoria do split ou das sequências longas não foi resolvida, o título/resumo/conclusões deixam clara a restrição.

### S12 — Experimento controlado entre arquiteturas (extensão)

**Depende de:** S04, S05 e, idealmente, S03.

**Objetivo:** testar se o padrão ResNet50/ConvNeXt persiste sob treinamento comparável.

**Escopo:** definir previamente orçamento de épocas/steps, dados, ruído, `label_smoothing`, otimizador, seleção de checkpoints e sementes comparáveis; executar pelo menos um par de configurações com diferenças isoladas e relatar custo de GPU. Uma ablação de normalização ou kernel é um experimento distinto.

**Aceitação:** relatório com múltiplas sementes quando viável e comparações controladas. Na ausência desse experimento, manter todas as conclusões causais no nível das configurações avaliadas.

## Portões de decisão

1. **Após S01:** o mapa de execução existe? Se não, continuar a organização antes de novas análises.
2. **Após S02/S03:** há genomas compartilhados ou sobreposição? Se sim, delimitar imediatamente o alcance dos resultados existentes e decidir se o novo split exige retreinamento.
3. **Após S06:** a queda em sequências longas decorre de erro de implementação, shift de dados ou ainda é inexplicada? Só então decidir se esse experimento fica no resumo.
4. **Após S07/S09/S10:** resultados centrais sustentam os termos “robustez”, “confiança” e “interpretabilidade”? Ajustar o manuscrito a essa resposta, sem forçar concordância com valores históricos.
