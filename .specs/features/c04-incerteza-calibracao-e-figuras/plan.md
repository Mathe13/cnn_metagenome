# C04 — Incerteza, calibração e figuras rastreáveis

## Problem

Os resultados principais do notebook 01 não exportam intervalos de confiança para Accuracy, Top-5, Macro-F1 e Weighted-F1. O notebook 02 calcula métricas formais de calibração, mas não entrega os diagramas de confiabilidade associados aos mesmos exemplos. Quem revisa o artigo não consegue reconstruir a incerteza ou conferir os painéis sem repetir inferência e adivinhar decisões de amostragem e ajuste.

Inspeção em 2026-11-03, HEAD `e06801546070875137634ff697a0696b4ebc780c` e alterações locais C03 existentes:

- Notebook 01 avalia cinco variantes Species e duas Genus nas cinco taxas, mas `store_predictions` fica falso no modo empírico. Depois descarta arrays de rótulos/predições/confiança, exceto para dois especialistas em condições específicas. Top-5 fica como contagem global, insuficiente para reamostrar consultas ou estratos.
- `eval.metrics.compute_f1_scores` calcula Macro-F1 sobre todas as classes declaradas, com F1 zero para denominador zero; Weighted-F1 usa suporte da verdade. `ClassificationMetrics` usa Accuracy/Top-5 em porcentagem e F1 em fração. C03 tem outra convenção de Macro-F1, restrita a classes presentes; essas estimativas não podem ser misturadas silenciosamente.
- `evaluate_dataloader` e `audit_dataloader_calibration_by_noise` usam RNG por taxa e ordem dos lotes. A semente isolada não identifica qual corrupção cada consulta recebeu, sobretudo entre loaders/execuções diferentes.
- `audit_dataloader_calibration_by_noise` já calcula ECE/MCE/NLL/Brier em lotes, ajusta uma temperatura na validação limpa e a aplica a todas as taxas. Limita validação a 16 lotes por padrão, concatena logits de validação em RAM e não registra os IDs/seleção que definiram esse ajuste.
- `TemperatureScaler.fit` valida apenas `split_name="validation"`; arrays oriundos do teste podem ser passados com outro nome, como ocorreu com a sonda antes de C01. O fluxo precisa de proveniência, além dessa string.
- Notebook 02 avalia cinco modelos × cinco taxas no modo empírico, mas usa dois modelos × duas taxas no smoke. Não chama `save_reliability_diagram` e não persiste bins antes/depois. Essa função escreve texto no destino `.png` se a plotagem falha, apresentando um caminho de figura sem imagem válida.
- As funções atuais arredondam métricas/bins antes de retornar. Estatísticas compactas e ICs devem preservar precisão até a apresentação. Os arquivos em `reporting/figures.py` incluem geradores de imagens procedurais; a saída C04 não pode tratá-los como gráficos derivados de evidência.
- Os notebooks 01/02 ainda usam diretórios fixos e não integram toda a identidade de execução C00. O notebook 02 tem análises adicionais de confusão, GradCAM e entropia: não são provas de calibração e não devem produzir rótulo de completude para C04.
- Os manifestos locais de benchmark/calibração declaram `is_smoke_test: true`. Os Parquets locais Species/Genus têm 16 linhas por split, compatíveis com fixtures. O volume `/home/mlima/workspace` e os checkpoints Species inspecionados estão ausentes. C03 está em alterações locais, sem `verification.md` disponível nesta inspeção; sua validação empírica não pode ser presumida.

Com C04, cada estimativa terá população, denominador, unidade de reamostragem, configuração e insumos verificáveis. Serão exportados diagramas reais antes/depois para as 25 condições Species, com temperatura selecionada independentemente do teste. Resultados locais sintéticos continuarão separados da evidência do artigo.

## Flow

Reutiliza `evaluate_dataloader`, as definições de F1, o auditor de calibração, `TemperatureScaler` e o exportador de confiabilidade; integra as identidades C00/C01 e os contratos pareados C03 quando aplicáveis, sem novo registro de modelos ou cálculo alternativo de métrica.

1. Configuração e artefatos → notebooks 01/02 e `viral_cnn.notebook_runtime` (exists): preflight, seleção dos modelos/níveis, reserva de saídas e identidade de splits/checkpoints/ordem de classes.
2. Exemplos identificados e realização de ruído → `viral_cnn.eval.runner` (exists): inferência por lotes e exportação de estatísticas compactas por consulta/condição (door 1), sem matriz global de probabilidades.
3. Artefatos compactos → avaliação de incerteza (door 2): reconstrói estimativas e reamostra consultas dentro dos estratos taxonômicos declarados para exportar ICs condicionais do benchmark 01.
4. Validação identificada → `TemperatureScaler` e `viral_cnn.eval.calibration` (exists): ajusta T positivo em logits de validação limpa com processamento limitado por blocos, sela sua identidade e configuração antes do teste (door 3).
5. Teste e T selado → `audit_dataloader_calibration_by_noise` (exists): calcula ECE/MCE/NLL/Brier antes/depois sobre os mesmos exemplos/ruído, exporta estatísticas e bins.
6. Bins → `save_reliability_diagram` (exists): gera CSV/JSON e imagem válida para cada modelo/taxa, registrando sucesso/falha; não usa figuras substitutas de reporting.
7. Produtos e comparações elegíveis → notebooks 01/02 e `write_notebook_manifest` (exists): registra cobertura da matriz, ICs, temperaturas, figuras e limites. Comparações com baselines dependem dos artefatos C03 e C06 fará a incorporação no artigo.

## Impact

| Front | What changes |
| --- | --- |
| domain | Novo `uncertainty_protocol: c04_stratified_fragment_v1`: IC condicional aos fragmentos e composição taxonômica observados; não mede variabilidade de treino, independência genômica ou generalização. |
| domain | Novo `calibration_protocol: c04_clean_validation_temperature_v1`: uma T por checkpoint ajustada em validação limpa e congelada para todas as taxas. |
| domain | Unidades permanecem explícitas: proporções/F1 e porcentagens são distinguíveis; diferenças de Accuracy em pp, sem comparar diretamente valores de escalas diferentes. |
| stored data | Novas estatísticas por consulta em shards, bins e parâmetros T com manifesto. Não migrar ou completar retroativamente os CSVs históricos. |
| scientific | Adiciona incerteza às métricas e calibração rastreável; não altera logits originais, pesos, classes, split ou corrupção. Pós-calibração é resultado separado. |
| outputs | Resumos do 01 e calibração/figuras do 02 ganham ICs, suporte e identidade. Tabelas 1/3 e Figura 1 podem receber ICs em C06; Tabela 7/GradCAM permanecem análises distintas. |
| runtime | Integra C00 aos caminhos relevantes de 01/02, preserva análises existentes e permite executar a etapa C04 sem depender de sucesso em exemplos de confusão/GradCAM. |
| shared code | Extensões opcionais do runner/calibration preservam consumidores anteriores e C02/C03. Não importar runtime/reporting em módulos de nível inferior. |

## Relations

- Uma condição pertence a um modelo/checkpoint, nível taxonômico, split e realização identificada de ruído.
- Cada linha compacta corresponde a uma consulta com verdade, predição e estatísticas suficientes para as métricas declaradas (door 1).
- Cada estimativa/IC pertence a uma seleção identificada dessas linhas e a um protocolo de reamostragem (door 2).
- Uma temperatura pertence a um checkpoint e a uma seleção da validação, sem relação de ajuste com o teste (door 3).
- Cada condição Species possui dois conjuntos de bins, antes/depois, e um painel comparativo com os dois estados e histograma de suporte.
- Uma execução completa de calibração possui 25 condições; uma execução parcial conserva seus resultados, mas não recebe status completo.

## Surface

None - sem serviço HTTP novo. Interfaces locais, estado de erro e produtos estão em Observable; contratos persistidos estão em Landing.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1. Evidência compacta versionada | `schema_version: 1`, identidade de `query_id`, `split`, checkpoint, classes e corrupção; shards com verdade, predição, `top5_hit`, confiança, NLL por exemplo e estatísticas `p_true`/`sum_p_squared` antes/depois quando disponíveis | Somente média global, ou matriz `(N,C)` de probabilidades: a primeira não permite reamostragem de F1/Top-5, a segunda inviabiliza armazenamento/memória no universo principal. |
| 2. IC condicional com estimando definido | `uncertainty_protocol: c04_stratified_fragment_v1`, bootstrap com reposição de fragmentos dentro do táxon verdadeiro, suporte por táxon fixo, 2.000 repetições, seed 42, quantis 0.025/0.975; Macro-F1 mantém todas as classes declaradas e zero_division=0 | Reamostrar F1s prontos ou omitir classes da Macro-F1 em cada réplica: altera o estimando e ignora falsos positivos entre classes. Bootstrap de espécies C03 permanece outro protocolo para diferenças pareadas. |
| 3. Temperatura verificável | `calibration_protocol: c04_clean_validation_temperature_v1`; estados `UNFITTED -> FITTED -> SEALED`; registro de T não arredondada, checkpoint, IDs/hash da validação, regra de seleção e otimizador; teste só transforma após SEALED | Provar origem com `split_name` ou escolher T/época pela melhora no teste: não garante separação entre ajuste e avaliação. |

Nothing else in this change is hard to reverse. Registrar portas aprovadas no próximo ID livre de `.specs/STATE.md` ao implementar; não sobrescrever decisões ou protocolos de C03.

## Criteria

### S1: Estatísticas suficientes e ICs reproduzíveis (P1)

**Acceptance Criteria**

1. WHEN o notebook 01 executa C04 THEN o sistema SHALL exportar evidência compacta para cinco variantes Species e duas Genus nas taxas 0, 0.02, 0.05, 0.10 e 0.15, vinculando cada linha à consulta, checkpoint, split e realização de ruído.
2. WHEN as estatísticas são reprocessadas THEN o sistema SHALL reconstruir Accuracy, Top-5, Macro-F1 e Weighted-F1 com os mesmos exemplos e definições do cálculo original, com diferença absoluta ≤1e-6 antes do arredondamento de apresentação.
3. WHEN ICs do benchmark principal são calculados THEN o sistema SHALL reamostrar fragmentos dentro do táxon verdadeiro com suporte fixo, 2.000 réplicas, semente 42 e limites nos quantis 2,5%/97,5%, recomputando TP/FP/FN a partir das linhas ou contagens equivalentes em cada réplica.
4. WHEN uma estimativa é exportada THEN o sistema SHALL registrar valor, unidade, limites, nível 95%, N, classes declaradas/presentes, unidade/estratos de reamostragem, repetições e população avaliada.
5. IF faltam Top-5 por consulta, identidades ou estatísticas para a métrica solicitada THEN o sistema SHALL registrar a métrica/IC como indisponível com motivo, sem copiar Top-1 ou inferir IC a partir da média publicada.
6. IF uma réplica ou estrato é vazio ou uma métrica é indefinida THEN o sistema SHALL registrar contagens e motivo de indisponibilidade, sem descartar réplicas silenciosamente ou converter indefinido em zero.
7. WHEN métricas/ICs são computados THEN o sistema SHALL limitar memória a blocos e estatísticas compactas/esparsas, sem matriz global `(N,C)` de probabilidades nem confusão densa por réplica.

**Independent test:** fixture multiclasse com falsos positivos cruzados, classes ausentes e casos Top-5 conhecidos; confrontar cálculo compacto com reamostragem direta das linhas nas mesmas seleções. Classificador perfeito deve preservar seu valor nas réplicas, incluindo a convenção das classes declaradas sem suporte.

### S2: Temperatura ajustada somente na validação (P1)

**Acceptance Criteria**

8. WHEN o ajuste de T é solicitado THEN o fluxo auditado SHALL aceitar somente artefato/loader de validação com proveniência verificável e identidade operacional disjunta da do teste.
9. IF o artefato de teste é fornecido ao ajuste, mesmo nomeado `validation`, THEN o sistema SHALL rejeitá-lo antes da atualização de T.
10. WHEN a temperatura é ajustada THEN o sistema SHALL minimizar NLL sem label smoothing na validação limpa, registrar T finita e positiva, configuração/critério de convergência e seleção efetivamente usada, e selar seu estado antes da avaliação de teste.
11. WHILE T está selada o fluxo SHALL recusar novo ajuste nesse estado e aplicar a mesma T às cinco taxas de teste do respectivo checkpoint, sem escolher outra temperatura pelos resultados do teste.
12. WHEN a validação usa limites ou amostragem THEN o sistema SHALL registrar população, seleção, IDs, semente e contagem realizada, sem truncamento implícito em 16 lotes.
13. IF faltam validação válida, checkpoint compatível ou convergência finita THEN o sistema SHALL marcar pós-calibração indisponível/falha e impedir status completo, preservando resultados antes da calibração que sejam válidos.

**Independent test:** trocar apenas os logits/rótulos do teste não pode alterar T selecionada; fixture de teste renomeada deve falhar. Ajuste por blocos deve concordar com cálculo denso pequeno da mesma função objetivo dentro da tolerância numérica declarada nos checks.

### S3: Métricas e diagramas reais antes/depois (P1)

**Acceptance Criteria**

14. WHEN o notebook 02 avalia calibração THEN o sistema SHALL produzir ECE, MCE, NLL, Brier, Accuracy e confiança antes/depois para cinco variantes Species × cinco taxas sobre os mesmos IDs e sequências corrompidas.
15. WHEN bins são calculados THEN o sistema SHALL usar 15 intervalos de largura uniforme em [0,1], último fechado em 1, e exportar limites, contagem, soma de confiança, acertos e peso, com soma das contagens igual a N para cada estado.
16. WHEN Brier e NLL são computados THEN o sistema SHALL usar todas as classes por lote, Brier via `sum(p²) - 2*p_true + 1` e NLL com tratamento numérico declarado, sem aproximá-los pela confiança Top-1.
17. WHEN temperatura positiva é aplicada THEN a auditoria SHALL verificar a preservação de Top-1 no mesmo conjunto, expondo qualquer discrepância numérica em vez de atribuí-la a melhora de classificação.
18. WHEN os diagramas são exportados THEN o sistema SHALL produzir 25 painéis comparativos antes/depois em PNG e PDF, com CSV/JSON dos 50 conjuntos de bins e identificação de modelo, ruído, T, N e histograma de suporte.
19. IF a renderização falha ou o destino não contém imagem válida THEN o sistema SHALL registrar falha e não anunciar figura concluída nem escrever texto com extensão de imagem.
20. WHEN um bin tem contagem zero THEN sua saída SHALL indicar falta de suporte e o gráfico SHALL omitir o ponto de acurácia/confiança correspondente, sem apresentá-lo como observação de calibração zero.

**Independent test:** métricas e bins de fixture pequena calculados diretamente, incluindo confiança 0/1 e fronteiras; verificar assinatura/abertura dos PNGs e PDFs e forçar falha de plotagem. Não exigir que T reduza ECE/NLL no teste: a saída deve refletir o observado.

### S4: Integração, comparações e limite da evidência (P1)

**Acceptance Criteria**

21. IF uma comparação pareada usa artefatos C03 THEN o sistema SHALL exigir pacote empírico elegível com consultas/verdades/ruído/taxonomia alinhados, preservando protocolo, estimando e unidade dos ICs pareados C03 separadamente dos ICs individuais C04.
22. WHEN uma execução parcial encontra modelo/condição indisponível THEN o sistema SHALL exportar matriz de completude com motivo por célula, sem preencher valores históricos e sem declarar completa a matriz de 35 condições do benchmark ou 25 da calibração.
23. WHEN notebooks 01/02 escrevem produtos C04 THEN o sistema SHALL usar namespaces exclusivos C00 e manifestos com versões, configuração, splits/checkpoints e hashes, modos, temperaturas, estatísticas, ICs e figuras realmente gerados.
24. WHILE a execução é sintética o sistema SHALL usar fixtures temporárias CPU com `REMOTE_SERVER=False` e identificar todos os produtos como sintéticos, recusando sua admissão empírica.
25. WHEN C04 é verificada THEN o relatório SHALL distinguir testes locais, inspeção de entradas e execução real no JupyterHub, incluindo comando/configuração, manifestos, arquivos ausentes e divergências entre métricas novas e resultados históricos.

**Independent test:** executar passos C04 de 01/02 sobre fixtures isoladas; casos negativos com checkpoint ausente em `SMOKE_TEST=0`, pacote sintético C03 e figura faltante. Verificação empírica requer splits/checkpoints reais e manifestos `is_smoke_test: false`.

## Out of scope

| Excluded | Why |
| --- | --- |
| IC como variabilidade entre sementes de treinamento ou genomas independentes | Requer repetições de treino/proveniência não fornecidas pelo bootstrap de fragmentos. |
| Redesenhar split, deduplicar a população avaliada ou retreinar modelos | Muda o experimento; C04 preserva população e pesos identificados. |
| Ajustar T por taxa usando teste ou otimizar ECE no teste | Introduz seleção sobre a avaliação final. |
| Reformular análises de confusão, GradCAM, entropia hierárquica ou motivos | Não são pré-requisitos científicos de IC/calibração; preservar sua separação no notebook. |
| Gerar novos baselines ou converter o teste completo em amostra C03 | C03 tem consultas selecionadas próprias; não comparar percentuais de populações distintas como pares. |
| Exigir melhoria pós-calibração ou inferir efeito causal de arquitetura/label smoothing | A evidência pode indicar piora; diferenças históricas não controlam esses fatores. |
| Atualizar o manuscrito inteiro e substituir figuras finais automaticamente | C06 fará reconciliação e incorporação das saídas verificadas. |
| ICs de todas as análises exploratórias do repositório | Escopo: métricas principais do 01 e estratos explicitamente declarados para o artigo; adicionar novo estimando exige configuração identificada. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Perfil da skill | `light`; negativos científicos continuam obrigatórios | Sem perfil específico declarado no projeto | n |
| Dependências | C00 para saídas/proveniência; C03 somente para diferenças contra baselines | ICs individuais e calibração podem progredir sem certificação empírica C03 | n |
| Reamostragem principal | Estratificada por táxon verdadeiro, fragmentos com reposição dentro de cada classe | Mantém composição e denominador da Macro-F1 atuais; não reutilizar cluster-bootstrap C03 como se estimasse a mesma incerteza | n |
| Interpretação do IC | Condicional aos fragmentos, composição e checkpoints observados; dependência de sequência/genoma limita a interpretação | O split tem overlap conhecido e não há certificação de independência genômica | n |
| Estratos publicáveis | Universo completo por nível/taxa; estratos adicionais somente quando listados na configuração de resultados a publicar, com IDs e suporte | Evita produzir ICs arbitrários sem vínculo às afirmações do artigo | n |
| Seleção de T | Uma T por checkpoint sobre validação limpa; usar validação completa por padrão e logits em shards/blocos com objetivo acumulado | Mantém o desenho atual e evita concatenação global de logits; pilotos explicitam seu orçamento | n |
| Configuração inicial de T | Inicial 1.0, parametrização log(T), LBFGS até 100 iterações, lr 0.05; registrar tolerâncias e estado final | Reutiliza o ajuste atual com objetivo por blocos, sem inventar configurações históricas | n |
| Condições do benchmark | Cinco Species e duas Genus nas cinco taxas; calibração publicável exigida para cinco Species × cinco taxas | Respeita os resultados principais atuais e a matriz definida no roadmap | n |
| Estatísticas e unidades | Persistir precisão anterior ao arredondamento; Accuracy/Top-5 em %, F1/ECE/Brier em escala declarada; mudanças de escala só na apresentação identificada | Preserva os contratos existentes e evita ICs com centro/limites em unidades incompatíveis | n |
| Identidade de corrupção | IDs estáveis e mesma realização entre caminhos 01/02 quando comparados; mudanças de backend/lotes exigem identidade efetiva, não só seed igual | Não presumir que ordem multiworker ou RNG por lote garante pareamento | n |
| Dependência C03 em andamento | Preservar alterações locais e validar o contrato final antes de integrar pares | A inspeção encontrou trabalho C03 ainda sem relatório final, não uma conclusão empírica | n |

**Open questions:** none - defaults acima definem o plano; disponibilidade remota condiciona a evidência empírica, sem impedir especificação ou provas locais.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Benchmark 01 | Estatísticas, identidade e reconstrução | AC 1–2 |
| ICs | Estimando, precisão, suporte e casos indefinidos | AC 3–7 |
| Temperatura | Origem de validação e rejeição de teste renomeado | AC 8–9 |
| Temperatura | Ajuste, selagem, limites e falhas | AC 10–13 |
| Calibração 02 | Matriz, métricas formais e conservação de exemplos | AC 14–17 |
| Diagramas | Dados exportados e imagens válidas | AC 18–20 |
| Pares | Condições C03 e protocolos distintos | AC 21 |
| Matriz de execução | Modelo/condição faltante e resultado parcial | AC 22 |
| Saídas | Repetição e concorrência | AC 23; existing - reserva exclusiva C00 e publicação somente de artefatos completos |
| Console e log | Caminhos de splits, estatísticas e pesos | existing - `log_artifact_path`, contrato LOG_PATH de AGENTS.md |
| Verificação | Sintético, remoto e limites | AC 24–25 |
| HTTP | Autorização, rate limiting e paginação | n/a - pipeline local sem serviço HTTP novo |
| Interface visual | Navegação e loading | n/a - sem tela nova; diagramas são artefatos científicos exportáveis |

As nove dimensões foram consideradas: validação (AC 1–2 e 8–9), falha parcial (AC 5–6, 13 e 19), repetição (AC 11 e 23), autorização (n/a — uso local), concorrência (reserva C00 e arquivos completos), ciclo dos dados (AC 1 e 23), dependência externa (AC 13, 22 e 25), transições (AC 10–11) e observabilidade (AC 4, 15 e 22–25). A associação a provas será escrita em `checks.md` após revisão do plano.

## Sources

- `ROADMAP_CORRECOES_REPRODUCAO.md`, C04 e portões; `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` — métricas, memória, ruído e limites de evidência.
- Specs S09/N02/R03, plano/contratos C03 e verificação C00 — métodos existentes, temperatura em validação e dependências ainda condicionadas à execução real.
- Notebooks 01/02, `eval/runner.py`, `metrics.py`, `calibration.py`, exportadores/testes e manifestos locais — comportamento inspecionado, perdas de rastreabilidade e saída sintética histórica.
