# C02 — Auditoria empírica de sequências longas

## Problem

A cadeia `long/01 → long/02 → long/03` dos notebooks de reprodução ainda pode produzir a Tabela 12 incompleta e atribuir causas à queda de ConvNeXt sem medir os mecanismos alegados. Quem prepara a resposta aos revisores não consegue distinguir números calculados, regras históricas de simulação e interpretações sem teste.

Inspeção em 2026-11-03, HEAD `b85fbcfe907333e26f607ee7e19a80e3d5d9e480`:

- `long/01` lê os primeiros 50 exemplos longos e 200 fragmentos de treino. Depois grava 500 versus 50 fragmentos por genoma e razão 10 como constantes. O notebook histórico `long/prepare database.ipynb`, célula 15, define três faixas: 50 para origem <10.000 bp, 200 para [10.000, 100.000) e 500 para ≥100.000 bp. A regra no código não comprova as contagens sobreviventes à curadoria/split.
- A simulação histórica exporta `species_id`, `genus_id`, `origin_len`, `read_len` e `sequence`, sem accession por fragmento nessa célula. `origin_len` não identifica um genoma. A célula 30 remapeia IDs para índices locais e salva `mapping_species_final.csv`; comparar diretamente esses índices aos logits Species do modelo pode usar ordens de classes diferentes.
- `long/02` avalia apenas `resnet_clean_species`, lê no máximo 50 exemplos e gera `predictions_long_voter_resnet50_species.parquet` com `read_id` e `true_label`. `long/03` procura `resultados_inferencia_long_reads_*.parquet` com `species_true`. As duas variantes ResNet50 derivam o mesmo nome `ResNet50` e podem consumir o mesmo arquivo.
- O caminho empírico de `long/03` aceita pelo menos um arquivo de modelo. No smoke, a Tabela 12 é transcrita de constantes. `ambiguity_ratio` ausente é preenchida com zero. O diagnóstico grava `VERIFIED_AND_DOCUMENTED` sem teste de mecanismo.
- `LongSequenceVoter` já usa média dos logits antes de softmax e voto majoritário. Seu resultado guarda Top-5 por janela, mas não os logits brutos necessários para reconstruir a média ou medir influência de janelas. `slice_sequence` descarta a cauda incompleta para sequências ≥512 bp; `pad` só completa sequências menores que a janela.
- `stratify_long_errors` não cobre todos os comprimentos e deixa pequenos intervalos de ambiguidade sem estrato. `evaluate_long_baselines` converte `seen_genome` em booleano, sem preservar desconhecidos. O integrador pode usar a primeira coluna como ID e colapsar duplicatas em um dicionário.
- O teste histórico N07/C11 exige os próprios números 0,7269 e 0,0349 no smoke e a presença dos nomes dos mecanismos. Isso não comprova avaliação empírica nem diagnóstico. O artigo e AD-010 também apresentam causas e presença no treino como estabelecidas; essa divergência precisa de registro explícito.
- Localmente há um Parquet longo de 16 linhas e predições ResNet50 de quatro linhas, compatíveis com fixtures. Os cinco checkpoints Species, os mapeamentos longos nos caminhos inspecionados e `/home/mlima/workspace` estão ausentes. Os manifestos longos atuais não demonstram proveniência empírica.

Ao entregar C02, a Tabela 12 dependerá de cinco variantes identificadas, do mesmo conjunto de consultas e do mapeamento taxonômico comprovado. Cada diagnóstico terá evidência e limite declarado; falta de suporte produzirá hipótese ou inconclusão, sem preencher metadados desconhecidos.

## Flow

Reutiliza a topologia e manifestos C00, o registro de modelos, `LongSequenceVoter`, `quantify_training_overlap`, `stratify_long_errors` e os avaliadores de métricas/calibração existentes, sem alterar a arquitetura ou substituir a agregação publicada.

1. Entradas e configuração → `long/01` e `viral_cnn.notebook_runtime` (exists): preflight da cadeia, inventário identificado de consultas, treino e mapeamentos; saída exclusiva por execução.
2. Inventário → `viral_cnn.inference.audit_long` (exists): conta elegibilidade, comprimentos, ambiguidades e grupos realmente observados; executa overlap no universo ou em amostra previamente configurada e identificada.
3. Consultas e mapeamentos → `long/02` e `viral_cnn.models.registry` (exists): resolve os cinco checkpoints e traduz rótulos locais para a ordem canônica Species, preservando IDs de origem para auditoria.
4. Sequências e modelos → `viral_cnn.inference.voter` (exists): produz ambas as agregações sobre as mesmas janelas e exporta predições/estatísticas diagnósticas vinculadas às consultas (door 1).
5. Cinco manifestos de inferência → `long/03` e `viral_cnn.baselines.long_eval` (exists): valida identidade, completude e alinhamento exato antes de calcular a Tabela 12 e estratos.
6. Estatísticas por janela, consultas e verdades → `viral_cnn.inference.audit_long` e `viral_cnn.eval.calibration` (exists): avalia hipóteses de composição amostral, sensibilidade aos logits e calibração, produzindo achados com estados de evidência (door 2).
7. Resultados → `long/01–03` e `write_notebook_manifest` (exists): exporta manifestos, auditorias e matriz de divergências históricas; C06 consumirá essas saídas para reconciliar o artigo.

## Impact

| Front | What changes |
| --- | --- |
| domain | Novo `long_sequence_protocol: c02_audit_v1`: identifica a cadeia corrigida e distingue o universo completo de pilotos ou amostras. Não renomeia o experimento histórico como reproduzido. |
| domain | `seen_genome` torna-se evidência trivalente: seen, unseen ou unknown. Compartilhamento de k-mers ou táxon não resolve a origem genômica. |
| domain | Diagnósticos passam a `SUPPORTED`, `HYPOTHESIS` ou `INCONCLUSIVE`; suporte é de uma observação operacional definida, sem atribuição causal automática à arquitetura ou à transferência. |
| stored data | Novo contrato comum de consultas/predições entre long/02 e long/03. Artefatos históricos permanecem preservados; adaptação requer mapeamento explícito, não inferência por nome do arquivo. |
| scientific | Pode alterar N, acurácias da Tabela 12, estratos, overlap e status dos diagnósticos. Janela, canais, pesos, stride e regras históricas de agregação permanecem identificáveis. |
| consumers | Atualiza long/01–03, `audit_long.py`, `long_eval.py` e extensão diagnóstica do voter; consumidores Python existentes mantêm o resultado usual por padrão. Atualiza testes que exigem constantes históricas no smoke, mediante checks aprovados. |
| documentation | Atualiza auditoria de sequências longas e registra a divergência de AD-010; reescrita global de Results/Abstract/Conclusion fica em C06. |
| runtime | Integra C00 aos três notebooks desta cadeia e completa seus pré-requisitos locais; não declara C00 integralmente concluída. |

## Relations

- Um universo elegível contém consultas identificadas; uma amostra é subconjunto identificado desse universo e possui sua regra de seleção.
- Cada consulta tem uma sequência e uma verdade canônica; accession e vínculo ao genoma de treino são opcionais, com estado desconhecido preservado.
- Cada artefato de inferência pertence a uma variante/checkpoint e a uma combinação declarada de ruído e tratamento de sequência curta.
- Uma comparação completa possui exatamente as cinco variantes esperadas para o mesmo universo de consultas e protocolo (door 1).
- Cada achado diagnóstico referencia entradas, transformação, suporte e efeito observados; não herda a conclusão de um documento histórico (door 2).

## Surface

None - sem API HTTP ou serviço externo novo. As interfaces locais e os arquivos consumidos entre notebooks são definidos em Landing e Observable.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1. Contrato persistido de inferência pareada | `long_sequence_protocol: c02_audit_v1`; `predictions_<variant>_<condition>.parquet`; chave única `(query_id, variant, noise_rate, handle_short)`; `truth_label` em ordem canônica Species 0-based; manifesto referencia hashes de consultas, mapeamentos, checkpoint e parâmetros de janela | Derivar variante de `model_name.split()[0]`, alinhar por posição ou aceitar `species_id` sem mapa: causa colisão de variantes e comparação entre índices incompatíveis. |
| 2. Contrato de evidência diagnóstica | `status ∈ {SUPPORTED, HYPOTHESIS, INCONCLUSIVE}`, com `claim`, `test`, `effect`, `n`, `inputs` e `limitations`; metadados desconhecidos ficam nulos/unknown; resultado vazio não é efeito zero | `VERIFIED_AND_DOCUMENTED` baseado em texto fixo ou fixture: apresenta hipótese e resultado copiado como evidência empírica. |
| 3. Revisão explícita da interpretação AD-010 | Na implementação aprovada, nova decisão no próximo ID livre supersede AD-010: preserva 512/256, média pré-softmax e políticas pad/skip; causas e seen/unseen dependem de evidência por execução | Manter AD-010 como prova de causa/presença no treino ou editar sua história silenciosamente: contradiz a exigência de auditoria do roadmap. A nova decisão também corrige a justificativa de que a média necessariamente impede influência de janelas extremas. |

Nothing else in this change is hard to reverse. A decisão 3 é imposta pelo contrato de evidência do roadmap e AGENTS.md; será registrada antes da mudança de comportamento, sem modificar decisões nesta fase de planejamento.

## Criteria

### S1: Universo e suporte observados (P1)

**Acceptance Criteria**

1. WHEN long/01 audita as entradas THEN o sistema SHALL registrar arquivos e hashes, contagem disponível/elegível/excluída, regra de exclusão e natureza dos dados: contigs, genomas ou fragmentos simulados.
2. IF a execução usa uma amostra THEN o sistema SHALL registrar seu tamanho planejado/realizado, critério, semente e IDs selecionados, sem apresentar estatísticas como censo do universo.
3. WHEN overlap é calculado THEN o sistema SHALL declarar separadamente o universo/amostra de treino e consultas, k, canonicalização, numeradores e denominadores, incluindo consultas sem k-mers válidos.
4. WHEN comprimentos e composição amostral são auditados THEN o sistema SHALL medir contagens e pesos por estrato/táxon a partir dos dados e separar essas observações da regra histórica 50/200/500.
5. IF não existem IDs confiáveis de genoma por fragmento THEN o sistema SHALL marcar contagens por genoma e seen/unseen como desconhecidas, sem usar `origin_len`, táxon ou overlap de k-mers como substitutos de accession.

**Independent test:** fixture com faixas de origem em 10.000 e 100.000 bp, pesos diferentes da regra histórica e consultas sem k-mers; demonstrar que a saída depende das linhas observadas e declara a amostragem dos dois lados.

### S2: Inferência identificada e Tabela 12 completa (P1)

**Acceptance Criteria**

6. WHEN long/02 executa o protocolo principal THEN o sistema SHALL produzir predições para `resnet_clean_species`, `resnet_noise_species`, `convnext_zero_species`, `convnext_transfer_species` e `convnext_noise_species`, identificando cada checkpoint carregado por caminho e SHA-256.
7. WHEN rótulos longos são carregados THEN o sistema SHALL traduzir índices locais por mapeamentos identificados até a ordem canônica do checkpoint, preservando rótulo original e trilha da conversão.
8. IF falta mapeamento, há correspondência ambígua ou rótulo fora do domínio declarado THEN o sistema SHALL bloquear a comparação com os rótulos responsáveis, sem inferir offset pelo mínimo observado da coluna.
9. WHEN a inferência gera janelas THEN o sistema SHALL registrar janela 512, stride 256, tratamento da cauda incompleta, modo pad/skip, desempate por menor índice e número observado de janelas por consulta.
10. WHEN as duas agregações são calculadas THEN o sistema SHALL usar os mesmos logits de janelas, mantendo média antes de softmax e voto majoritário, com prova numérica de equivalência do caminho de janela única em FP32 com diferença máxima ≤1e-5.
11. WHEN long/03 recebe artefatos THEN o sistema SHALL validar cinco variantes distintas, condição idêntica e conjuntos de `query_id` únicos e exatamente iguais, com sequência/verdade alinhadas pelo ID.
12. IF uma variante está ausente, um arquivo representa outra variante, há consulta duplicada/faltante/extra ou verdade incompatível THEN o sistema SHALL interromper a geração da Tabela 12 empírica, enumerando as inconsistências.
13. WHEN a Tabela 12 é gerada THEN o sistema SHALL calcular cada acurácia a partir de acertos/N observados e exportar N, cobertura, exclusões e ambas as agregações, preservando valores históricos somente em referência separada.
14. WHEN o modo skip deixa uma consulta elegível sem janelas THEN o sistema SHALL preservar seu ID e registrar predição ausente como erro no denominador global, separando cobertura e acurácia condicional.

**Independent test:** cadeia com cinco modelos pequenos e verdade conhecida, embaralhando a ordem dos arquivos; recusar quatro modelos, duplicação ResNet50, ID ausente e mapa permutado. Mudar as predições deve mudar a Tabela 12 sem tocar constantes do artigo.

### S3: Diagnósticos condicionados a medições (P1)

**Acceptance Criteria**

15. WHEN erros são estratificados THEN o sistema SHALL exportar N, acertos e acurácias por modelo/agregação em faixas completas e disjuntas de comprimento, ambiguidade e táxon, incluindo unknown para metadados ausentes.
16. WHEN um estrato não tem observações THEN o sistema SHALL registrar N=0 e métrica indisponível, sem convertê-la em acurácia zero.
17. WHEN sensibilidade à composição amostral é avaliada THEN o sistema SHALL comparar acurácia por consulta e média de acurácias por táxon no mesmo conjunto identificado, registrando pesos e diferença em pontos percentuais sem atribuir efeito por genoma quando IDs faltam.
18. WHEN influência dos logits é auditada THEN o sistema SHALL medir discordância mean/vote, distribuição de magnitudes/margens e alteração do resultado ao retirar uma janela por vez na amostra diagnóstica identificada, sem mudar a agregação principal.
19. WHEN calibração é avaliada THEN o sistema SHALL calcular ECE com 15 bins e NLL das probabilidades de consenso obtidas por softmax da média dos logits, com N e confiança observados, sem tratar confiança ou entropia isoladas como calibração.
20. IF faltam logits/estatísticas suficientes para uma análise diagnóstica THEN o sistema SHALL marcá-la `INCONCLUSIVE`, sem reconstruir logits brutos a partir de Top-5 de probabilidades nem atribuir esse status à Tabela 12 válida separadamente.
21. WHEN um achado recebe `SUPPORTED` THEN o sistema SHALL vincular uma afirmação operacional delimitada ao teste executado, efeito e suporte observados conforme a regra declarada antes da análise; testes sintéticos não podem sustentar esse status empírico.
22. IF uma explicação não foi testada ou o teste não discrimina sua hipótese THEN o sistema SHALL registrar `HYPOTHESIS` ou `INCONCLUSIVE`, sem concluir que transferência, arquitetura ou calibração causaram a queda.

**Independent test:** logits conhecidos com uma janela dominante e controle sem influência; retirar a janela deve alterar somente o primeiro caso. Dados sem magnitudes devem gerar inconclusão. Verificar fronteiras 0/5% de ambiguidade, sequências <512 e >10.000 bp e estratos vazios.

### S4: Evidência preservada e verificação separada (P1)

**Acceptance Criteria**

23. WHEN long/01–03 escrevem produtos THEN o sistema SHALL usar namespaces exclusivos C00 e dependências explícitas entre execuções, registrando entradas, versões, configurações, checkpoints e hashes das saídas nos manifestos comuns.
24. WHILE os notebooks executam em smoke o sistema SHALL derivar métricas de fixtures determinísticas em CPU, marcar `is_smoke_test: true` e recusar sua promoção a evidência empírica.
25. WHEN C02 é verificada THEN o relatório SHALL separar contratos sintéticos, inspeção de arquivos e execução empírica remota, registrando diferenças entre o artigo, notebooks históricos e resultados realmente obtidos.

**Independent test:** executar a cadeia sintética em diretórios temporários, conferir manifestos e recusa empírica; caso negativo com entradas ausentes em `SMOKE_TEST=0`. Validar no JupyterHub somente quando dados, mapeamentos e os cinco checkpoints estiverem disponíveis.

## Out of scope

| Excluded | Why |
| --- | --- |
| Download automático NCBI, nova curadoria, geração de outro split ou retreinamento | Auditar dados existentes; não mudar o protocolo silenciosamente nem iniciar processamento pesado nesta fase. |
| Benchmark Kraken2/BLASTN e ICs gerais do artigo | C03/C04; os avaliadores existentes só serão alterados onde necessário à integridade da cadeia longa. |
| Ajuste de temperatura usando teste ou demonstração causal da transferência | C02 mede calibração; causalidade exige outro desenho experimental. Não ajustar parâmetros no teste. |
| Inferir accessions ou presença no treino a partir de taxonomia/k-mers | Não comprova identidade do genoma de origem. |
| Substituir média de logits por média de probabilidades | Mudaria o protocolo; diagnósticos não substituem a agregação publicada. |
| Atualizar globalmente tabelas, resumo e conclusão do manuscrito | C06 consumirá o mapa de evidências e divergências C02. |
| Migrar long/04 e todos os notebooks para C00 | Integração restrita a long/01–03; preservar consumidores existentes. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Perfil da skill | `light`; casos negativos do roadmap obrigatórios | Sem perfil específico declarado pelo projeto | n |
| Dependência C00 | Reutilizar APIs presentes e fechar integração/pré-requisitos destes três notebooks antes da prova empírica | Verificação C00 documenta integração parcial | n |
| Escopo principal | Tabela 12 clean (`epsilon=0`), Species e universo elegível completo por padrão | O notebook promete ruído sem executá-lo; não misturar taxas na mesma comparação | n |
| Execução com ruído opcional | Somente por configuração separada; corrupção uma vez por sequência antes do janelamento, mesma realização por consulta entre variantes, semente 42 e protocolo A/C/G/T | Garante coerência nas regiões sobrepostas; se divergir do histórico, registrar protocolo distinto e não declarar equivalência | n |
| Tratamento de curtos | Reportar pad e skip separadamente; cauda incompleta descartada como no código atual | Mantém comparabilidade do método e explicita mudança de cobertura | n |
| Escala da auditoria de overlap | Censo quando configurado com recursos suficientes; alternativa explícita por amostra determinística hash/seed 42 com orçamento salvo | Conjunto de todos os k-mers de treino pode não caber em RAM; não impor censo inviável nem esconder truncamento | n |
| Logits diagnósticos | Calcular estatísticas completas durante inferência; persistir logits em shards somente na amostra diagnóstica declarada, sem matriz global densa | Permite recomputar influência e calibração sem armazenar todas as classes de todas as janelas do universo | n |
| Critério de suporte dos achados | Composição: diferença observada entre ponderações; influência: casos identificados de reversão leave-one-window-out; calibração: discrepância medida, sem alegar causa; registrar valores/contagens, sem limiar pós-hoc de significância | Cada achado pode sustentar a observação delimitada, não um mecanismo causal universal | n |
| Estado diagnóstico | Não testado: HYPOTHESIS; testado sem dados suficientes ou sem discriminar hipótese: INCONCLUSIVE; observação operacional corroborada: SUPPORTED | Evita transformar texto fixo ou ausência de efeito em confirmação | n |
| Revisão de AD-010 | Superseder explicitamente sua interpretação causal/de genoma na implementação, preservando seus parâmetros técnicos e história | Regra de evidência do roadmap é mais recente e requer conciliação explícita | n |

**Open questions:** none - defaults definidos acima; disponibilidade remota condiciona validação empírica, sem bloquear o plano ou obrigar um resultado diagnóstico positivo.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| long/01 | Universo, amostragem, exclusões e regra histórica | AC 1–4 |
| Origem genômica | Known/unknown sem imputação | AC 5 |
| long/02 | Cinco variantes, mapas e entradas inválidas | AC 6–8 |
| Voter | Janelamento, cauda, curtos e agregações | AC 9–10 e 14 |
| long/03 | Artefatos faltantes, duplicados, desalinhados e incompletos | AC 11–12 |
| Tabela 12 | Valores calculados, denominadores e cobertura | AC 13–14 |
| Estratos | Fronteiras, vazio e metadados ausentes | AC 15–16 |
| Diagnósticos | Composição, logits e calibração | AC 17–19 |
| Diagnósticos | Dados insuficientes e afirmações não demonstradas | AC 20–22 |
| Execuções | Repetição, concorrência e isolamento | AC 23; existing - reserva exclusiva C00 |
| Smoke/verificação | Fixture, bloqueio empírico e limitações remotas | AC 24–25 |
| Console/log | Caminhos de datasets, mapas e pesos carregados | existing - `log_artifact_path`, contrato LOG_PATH em AGENTS.md |
| HTTP | Autorização, rate limiting e paginação | n/a - execução local sem serviço HTTP novo |
| Interface visual | Loading, navegação e layout | n/a - sem tela nova; CSVs e relatórios são os produtos |

As nove dimensões foram consideradas: validação (AC 6–12), falha parcial (AC 12 e 20), repetição (AC 23), autorização (n/a — processamento local), concorrência (reserva C00 e publicação somente de artefatos completos), ciclo dos dados (AC 1 e 23), dependência externa (AC 25 e preflight), transições de evidência (AC 20–22) e observabilidade (AC 1–4, 13 e 23). A associação a provas será escrita em `checks.md` após revisão deste plano.

## Sources

- `ROADMAP_CORRECOES_REPRODUCAO.md`, C02 e portões; `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` — escopo, preservação, classes, proveniência e limites das afirmações.
- Specs S06/N07/R07 e verificação C00; `long/prepare database.ipynb`, `long/cnn-long-voter.ipynb` e `long/cnn-long-voter-compare.ipynb` — regras e divergências históricas inspecionadas.
- Notebooks de reprodução long/01–03, `voter.py`, `audit_long.py`, `long_eval.py`, testes relacionados, manifestos locais e `paper/manuscript_submission.tex` — cadeia atual, alegações e lacunas observadas.
