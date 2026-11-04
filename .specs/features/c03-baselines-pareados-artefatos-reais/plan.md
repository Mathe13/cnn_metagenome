# C03 — Baselines pareados com artefatos reais

## Problem

O notebook `short/09_benchmark_cnns_kraken2_blast_overlap.ipynb` já exige predições preparadas em modo empírico, mas ainda não comprova que os métodos receberam as mesmas sequências corrompidas, usaram a mesma ordem de classes ou produziram saídas completas. Falta uma entrada reproduzível para preparar ou recuperar esses artefatos sem depender da execução manual de células históricas.

Inspeção em 2026-11-03, HEAD `e06801546070875137634ff697a0696b4ebc780c`:

- `load_prepared_baseline_benchmark` exige verdade/overlap e arquivos para sete métodos × cinco taxas, verifica unicidade/conjuntos de IDs e alinha linhas. Não exige manifesto por método, identidade dos FASTAs, protocolo de ruído, taxonomia, banco ou checkpoint. Escolhe o primeiro alias/arquivo encontrado.
- `audit_exact_overlap` usa SHA-256 apenas da sequência forward, apesar do nome canonical. O histórico usa `sha256(min(seq, reverse_complement(seq)))`. O módulo também trata qualquer string de comprimento 64 como hash já calculado, podendo confundir DNA de 64 bp com um digest.
- `has_overlap.astype(bool)` pode interpretar texto `"false"` como verdadeiro; `train_count` ausente pode virar zero. Essas conversões podem fabricar o estrato sem correspondência exata.
- O notebook histórico `short/benchmark_cnns_kraken2_blast_overlap.ipynb` já contém seleção determinística por prioridade hash, exportação de referências/consultas, auditoria SQLite, comandos Kraken2/BLASTN, inferência das cinco CNNs e parsers. Deve ser preservado; funções reutilizáveis serão extraídas para o caminho de reprodução.
- Esse histórico define `NOISE_MODE="substitution"`: toda posição selecionada muda. O modo alternativo `legacy_replacement` permite repetir a base original. Só o segundo corresponde à taxa efetiva esperada `3ε/4` descrita no artigo/AD-009. Igualdade de `query_id` e nome `noise_05` não demonstra igualdade dos protocolos.
- O parser Kraken atual assume taxonomia flat customizada, `custom_taxid = label + 2`; não serve como conversão genérica de TaxIDs NCBI. BLASTN espera sufixos `_label_<n>`, agrega por espécie e ordena por bitscore/e-value/label. O histórico solicita até 100 referências por consulta; a documentação mostra 5, que não equivalem a cinco espécies distintas.
- `evaluate_predictions` já mantém abstenções `-1` no denominador e separa cobertura/acurácia condicional. O notebook exporta `diff_mean` em fração, enquanto o bootstrap retorna limites em pontos percentuais, sem deixar a diferença de unidade explícita.
- A figura contém percentuais fixos de overlap nos títulos. O manifesto próprio do 09 não identifica modo sintético/empírico nem versões/bancos/pesos. Os artefatos locais declaram 40 consultas e overlap 20/20, compatíveis com fixtures, não com evidência do benchmark completo.
- Não existem diretórios preparados nos três caminhos locais inspecionados (`cnn_kraken/`, `short/` e raiz). `/home/mlima/workspace`, os cinco checkpoints Species e os executáveis `kraken2`, `kraken2-build`, `blastn` e `makeblastdb` estão ausentes nesta sessão. A disponibilidade remota permanece desconhecida.

C03 entregará ingestão estrita e preparação reproduzível: artefatos remotos existentes poderão ser reutilizados após validação; se faltarem, um fluxo explícito produzirá consultas, referências e comandos/inferência no cluster. A ausência local não autoriza fabricar predições ou declarar indisponibilidade no servidor.

## Flow

Reutiliza a seleção e os comandos evidenciados no notebook histórico, os parsers/métricas de `viral_cnn.baselines`, o registro de modelos e os namespaces/manifestos C00, evitando outra implementação de ruído ou outro registro de checkpoints.

1. Configuração e artefatos existentes → `viral_cnn.notebook_runtime` e `viral_cnn.baselines.overlap` (exists): inventaria pré-requisitos, identidades e lacunas por método/ruído; aceita reutilização somente após validação.
2. Partições explícitas train/test → entrada de preparação local (door 3): seleciona consultas uma única vez, identifica partições/linhas, exporta FASTAs e rótulos, e calcula overlap canônico contra todo o treino declarado (door 1).
3. Consultas selecionadas → `viral_cnn.eval.noise` e preparação (exists / door 3): materializa uma realização por consulta/taxa e a compartilha entre os sete métodos, identificando protocolo, representação e sementes.
4. Referências, consultas e configuração → preparação (door 3), parsers de `viral_cnn.baselines.taxonomy` e `viral_cnn.models.registry` (exists): gera plano de comandos; quando explicitamente executado, constrói/usa bancos, roda ferramentas e infere CNNs, produzindo arquivos e manifestos completos por método/condição (door 2).
5. Artefatos preparados → `load_prepared_baseline_benchmark` (exists): verifica proveniência, completude, domínios de rótulos e correspondência exata das consultas antes de admitir comparação empírica.
6. Arrays alinhados → `viral_cnn.baselines.paired` e notebook 09 (exists): calcula métricas nos três estratos, discordâncias e ICs de diferenças pareadas com unidades explícitas.
7. Produtos → notebook 09 e `write_notebook_manifest` (exists): exporta auditoria, métricas, pares, figura e manifestos em namespace exclusivo; C04/C06 consomem somente saídas com modo/protocolo identificados.

## Impact

| Front | What changes |
| --- | --- |
| domain | Novo `baseline_protocol: c03_paired_v1`: certifica identidade das consultas, taxonomia e condições, distinguindo o protocolo principal do cenário histórico de substituição obrigatória. |
| domain | Overlap passa a ser comparação exata canônica forward/RC da consulta limpa contra todo o treino declarado; não certifica independência genômica ou ausência de homologia. |
| domain | `missing prediction` distingue abstenção observada/no-hit de uma execução ou arquivo incompleto; somente abstenções comprovadas são normalizadas para `-1`. |
| stored data | Novos manifestos por método/condição e inventário de preparação. Legados preservados, sem hashes/versões retrospectivos inventados. |
| scientific | Pode alterar estratos, N, cobertura, acurácias, Macro-F1, diferenças e ICs. Não deve manter o nome do experimento histórico se mudar ruído, consultas, referências ou política de candidatos. |
| consumers | Atualiza notebook 09, `overlap.py`, `taxonomy.py`, `paired.py`, documentação e caminho short do CLI existente para compartilhar validação. Preservar contratos da cadeia longa C02. |
| runtime | Integra C00 ao notebook 09 e à preparação; falha empírica sem entradas não aciona smoke. Pendências gerais de C00 permanecem separadas. |
| outputs | Títulos da figura usam N e percentuais observados; ICs e métricas sintéticas continuam artefatos de teste, sem status publicável. |

## Relations

- Um benchmark pertence a um protocolo, configuração e seleção identificada de consultas.
- Cada consulta possui verdade canônica, identidade operacional e sequência limpa; cada condição de ruído possui uma realização identificada dessa consulta.
- A auditoria de overlap pertence às consultas limpas e a um inventário de treino, não é recalculada silenciosamente a cada taxa.
- Cada resultado por método/condição referencia consultas materializadas e uma referência/taxonomia ou checkpoint identificado.
- Um pacote completo contém as cinco variantes CNN, Kraken2-flat e BLASTN para todas as cinco taxas declaradas.
- Uma comparação pareada referencia exatamente o mesmo conjunto de consultas, verdades, condição e estrato.

## Surface

None - sem API HTTP ou serviço externo novo. A entrada CLI local e seus estados estão em Observable; o contrato persistido está em Landing.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1. Identidade e overlap canônico | `baseline_protocol: c03_paired_v1`; `canonicalization: sha256_min_forward_rc_v1`; hashes de conteúdo das consultas por taxa e inventário das partições de origem; digest pré-calculado exige tipo/campo explícito | Hash forward com nome canonical, ou decidir que DNA de 64 bp é hash pelo comprimento: muda incorretamente a auditoria e a associação das consultas. |
| 2. Manifesto por método/condição | Campos obrigatórios: `method`, `noise_rate`, `noise_protocol`, `query_manifest`, `class_order_sha256`, `reference`, `tool_version`, `command`, `abstention_policy`, `output_sha256`, `completion`, `is_smoke_test`; CNN vincula também variante e SHA-256 do checkpoint. Estados `PLANNED`, `RUNNING`, `COMPLETE`, `FAILED`; só COMPLETE admite ingestão empírica | Aceitar arquivos pelo nome ou repetir metadados globais sem vínculo ao conteúdo: não comprova uso da mesma consulta nem conclusão da execução. Campos não aplicáveis são explícitos, nunca versão inventada. |
| 3. Preparação reproduzível opt-in | Nova entrada `scripts/prepare_baseline_benchmark.py` com configuração salva e modo padrão de inventário/plano; execução de exportação, construção de bancos e classificação/inferência são etapas explicitamente selecionadas; saídas sob `runs/<run_id>/` | Executar células históricas que têm `RUN_BUILD=True`/`RUN_CLASSIFY=True`, ou iniciar banco/inferência ao importar ou validar: acopla inspeção a trabalho pesado e saídas não isoladas. |

Nothing else in this change is hard to reverse. As portas aprovadas serão registradas no próximo ID livre de `.specs/STATE.md` durante a implementação, preservando AD-009/AD-011 e a história dos protocolos anteriores.

## Criteria

### S1: Encontrar ou preparar consultas e referências identificadas (P1)

**Acceptance Criteria**

1. WHEN a entrada de preparação é executada em modo padrão THEN o sistema SHALL produzir inventário dos requisitos existentes/ausentes por método e taxa, sem exportar dados, construir bancos ou iniciar classificação/inferência.
2. WHEN a exportação é explicitamente solicitada THEN o sistema SHALL selecionar consultas do test por prioridade hash com semente 42 e até 20 exemplos elegíveis por espécie, registrando população disponível, selecionados, exclusões e identidade de cada linha.
3. WHEN o protocolo de ruído principal é materializado THEN o sistema SHALL ressorteiar A/C/G/T incluindo a base original nas taxas 0, 0.02, 0.05, 0.10 e 0.15, registrando sementes, posições elegíveis e mudanças efetivas observadas.
4. WHEN consultas são entregues aos métodos THEN o sistema SHALL usar a mesma realização por consulta/taxa, verificável por hashes do conteúdo e ordem de rótulos; CNN recebe os tokens correspondentes ao conteúdo exportado, sem nova corrupção independente.
5. WHEN overlap é preparado THEN o sistema SHALL comparar consultas limpas com todo o treino declarado usando canonicalização forward/RC e registrar `all = overlap + no_exact_overlap` com estratos disjuntos e contagens por classe.
6. IF uma preparação parcial de treino ou um protocolo de substituição obrigatória é usado THEN o sistema SHALL atribuir cenário separado e impedir sua mistura com o benchmark principal de ressorteio `3ε/4`.

**Independent test:** fixture com forward/RC, DNA de 64 bp e digest explicitamente tipado; seleção determinística por classe e corrupção compartilhada entre consumidores. Trocar somente o conteúdo de um FASTA, mantendo os IDs, deve invalidar a condição.

### S2: Produzir ou importar resultados completos com proveniência (P1)

**Acceptance Criteria**

7. WHEN bancos/execuções são explicitamente solicitados THEN o sistema SHALL registrar comandos, versões observadas, inventário/hash das referências e taxonomia, parâmetros efetivos, saídas e estado final por método/condição.
8. WHEN CNNs são executadas THEN o sistema SHALL usar as cinco variantes Species registradas e vincular cada resultado ao checkpoint efetivamente carregado por caminho/SHA-256 e ordem canônica das 13.683 classes.
9. WHEN Kraken2-flat e BLASTN são preparados THEN o sistema SHALL vincular `custom_taxid = label + 2` e sufixos `_label_<label>` ao mesmo mapeamento canônico, preservando identidade das referências realmente indexadas.
10. IF uma execução externa falha ou seu artefato é truncado/incompleto THEN o sistema SHALL registrar FAILED e impedir ingestão como COMPLETE, sem converter a falha em abstenções.
11. WHEN resultados legados são importados THEN o sistema SHALL verificar consultas, configuração, referências e mapeamentos contra evidências disponíveis, mantendo campos ausentes como unknown e bloqueando certificação empírica quando faltar identidade obrigatória.
12. IF dois aliases/arquivos concorrem pela mesma variante e condição ou uma previsão está fora de `{-1, 0, ..., 13682}` THEN o sistema SHALL rejeitar a ingestão com os arquivos/rótulos responsáveis.

**Independent test:** executáveis simulados com sucesso/falha e saídas parciais; importar pacote com checkpoint, versão, hash ou mapa divergente. Testes não executam ferramentas reais, bancos volumosos ou downloads.

### S3: Alinhamento, abstenções e denominador estrito (P1)

**Acceptance Criteria**

13. WHEN o notebook 09 aceita um pacote principal THEN o sistema SHALL exigir sete métodos × cinco taxas com IDs únicos/não nulos e conjuntos exatamente iguais aos da verdade, alinhando por `query_id`.
14. IF há duplicatas, IDs inesperados, consulta/verdade divergente ou linha ausente em predições preparadas THEN o sistema SHALL interromper a comparação antes de gerar métricas empíricas.
15. WHEN saídas brutas válidas são normalizadas THEN o sistema SHALL atribuir `-1` a Kraken unclassified/root e BLASTN sem hit, mantendo uma linha por consulta no pacote preparado com motivo de abstenção explícito.
16. WHEN a política BLASTN é aplicada THEN o sistema SHALL agregar candidatos por espécie, ordenar por bitscore decrescente/e-value crescente/label crescente e registrar o limite de referências recuperadas separadamente dos até cinco rótulos candidatos.
17. WHEN overlap preparado é carregado THEN o sistema SHALL validar booleanos/contagens sem converter texto `"false"` em verdadeiro ou contagem desconhecida em ausência de overlap.
18. WHEN métricas são exportadas THEN o sistema SHALL usar N total do estrato para Accuracy/Top-5, separar N coberto e acurácia condicional e declarar a regra de Macro-F1 sobre classes presentes na verdade.
19. IF um método não fornece ranking Top-5 ou um estrato não tem observações THEN o sistema SHALL registrar a métrica como indisponível, sem copiar Top-1 ou produzir zero como resultado calculado.

**Independent test:** permutar linhas sem mudar métricas, remover uma previsão preparada e exigir erro; BLASTN sem hits em execução concluída deve permanecer como erro global. Todas as abstenções dão cobertura zero e acurácia condicional indisponível. Conferir fronteiras e conservação dos estratos.

### S4: Comparações pareadas e evidência publicável (P1)

**Acceptance Criteria**

20. WHEN discordâncias são calculadas THEN o sistema SHALL exigir mesma condição/estrato e comprovar que as quatro células da tabela 2×2 somam N dos mesmos IDs avaliados.
21. WHEN ICs pareados são gerados THEN o sistema SHALL usar bootstrap de espécies com reposição, 2.000 repetições, semente 42 e quantis 2,5/97,5%, exportando diferença central e limites todos em pontos percentuais, com N e número de espécies.
22. IF o estrato tem menos de duas espécies THEN o sistema SHALL registrar IC indisponível e motivo, sem omitir a estimativa pontual calculável.
23. WHEN a identidade de genoma/referência não pode ser demonstrada THEN o relatório SHALL preservar unknown para presença do genoma da consulta no banco e declarar que no_exact_overlap não demonstra generalização para genomas inéditos.
24. WHEN o notebook escreve resultados THEN o sistema SHALL usar namespace exclusivo C00, manifesto comum e referências aos manifestos por método, mantendo modo/protocolo nos CSVs, ICs e metadados da figura.
25. WHILE a execução usa fixtures sintéticas o sistema SHALL marcar os produtos como sintéticos e rejeitar sua admissão empírica, inclusive os ICs das 40 consultas locais.
26. WHEN C03 é verificada THEN o relatório SHALL separar prova local, inventário remoto e execução real, listar o comando/configuração/manifestos e registrar divergências de ruído, overlap e BLASTN em relação aos documentos históricos.

**Independent test:** pares pequenos com discordâncias e diferença conhecidas; confirmar sinal e unidade pp, IC indisponível com uma espécie e rejeição do pacote sintético como publicável. Execução remota exige artefatos reais completos e `SMOKE_TEST=0`.

## Out of scope

| Excluded | Why |
| --- | --- |
| Instalar ferramentas, baixar bancos ou executar o benchmark completo nesta fase de plano | Aqui somente especificação; a preparação implementada poderá ser executada explicitamente no ambiente disponível. |
| Treinar ou alterar as CNNs | Comparação usa checkpoints congelados; treinamento controlado pertence a C05. |
| Misturar taxonomia flat com base NCBI genérica | Outro cenário exige conversão/versionamento explícitos e não recebe o nome Kraken2-flat. |
| Benchmark novo long-sequence | C02 possui sua cadeia; manter compatibilidade dos módulos compartilhados sem ampliar C03. |
| ICs gerais de todos os resultados e ajuste de calibração | C04; aqui somente estimativas/ICs do benchmark pareado. |
| Recuperar accessions ou concluir causalidade/generalização | Dados taxonômicos e no_exact_overlap não bastam para essas afirmações. |
| Atualizar números do manuscrito ou publicar figuras finais | C06 fará reconciliação após evidência empírica. |
| Benchmark de velocidade entre ferramentas | Tempos podem ser registrados com escopo; comparação de desempenho computacional exige outro controle de hardware/bancos e não bloqueia C03. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Perfil da skill | `light`; negativos do roadmap obrigatórios | Sem perfil específico no projeto | n |
| Dependência C00 | Usar infraestrutura existente e completar os requisitos e integração do 09 | A lista inicial de preflight não comprova os 35 resultados nem a proveniência exigida aqui | n |
| Disponibilidade de artefatos | Inspecionar caminho remoto quando conectado; enquanto isso implementar preparação/validação sobre o procedimento histórico | Diretórios e ferramentas estão ausentes apenas nos caminhos locais observados; não inferir estado remoto | n |
| Caminho de geração | Extrair funções reutilizáveis do histórico; separar inventário, exportação, construção, inferência e ingestão | Reutiliza procedimento evidenciado sem alterar notebooks originais | n |
| Método canônico ResNet50 clean | Usar ID registrado `resnet_clean_species`; `resnet50_species` é alias histórico validado no limite de importação | Alinha C03 a C01/C02 sem troca silenciosa de variante | n |
| Parâmetros das ferramentas | Preservar como ponto inicial o histórico: Kraken confidence 0.0; BLAST task blastn, e-value 1e-5, dust yes, max_target_seqs 100 e max_hsps 1; registrar versão e comando efetivos | Não adotar o exemplo documental de 5 referências como se fosse o procedimento histórico de 100 | n |
| Bases de referência | Mesmo conjunto de fragmentos de treino identificado para ambos os bancos e CNNs, com multiplicidades e exclusões declaradas; nenhuma adição do teste | Permite comparação no cenário conhecido e auditar cobertura/composição sem remover overlap artificialmente | n |
| Representação de sequência | Preservar coordenadas, comprimento e alfabeto observado; configuração explicita padding/IUPAC → representação das ferramentas e tokens nulos A/C/G/T da CNN, com hashes de cada transformação | Não ressuscitar limpeza histórica de `_` sem declarar seu efeito; mudanças de elegibilidade recebem protocolo identificado | n |
| Estratos de overlap | Fixados sobre consultas limpas antes do ruído, mantendo duplicatas entre consultas com IDs distintos | Mede a relação do split de origem com treino sem alterar a população por taxa | n |
| Dados brutos sem hits | BLASTN vazio é resultado válido somente com conclusão comprovada; Kraken deve ter registro C/U para cada consulta | Distingue ausência biológica de predição de arquivo interrompido | n |
| ICs condicionais | Reamostragem de espécies estima incerteza da comparação na composição observada, não variabilidade entre treinos | Evita interpretação de múltiplas sementes não executadas | n |

**Open questions:** none - defaults acima deixam a implementação definida; suporte empírico depende do armazenamento/execução remotos e não será inferido de fixtures.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Preparação CLI | Estado inicial, lacunas e ausência de execução automática | AC 1 |
| Consultas | Seleção, ruído e identidade de conteúdo | AC 2–4 e 6 |
| Overlap | RC, universo de treino e partição dos estratos | AC 5 e 17 |
| Ferramentas/CNNs | Versões, configuração, pesos e mapas | AC 7–9 |
| Execução externa | Falha, interrupção e importação legada | AC 10–11 |
| Ingestão | Aliases conflitantes e domínios inválidos | AC 12–14 |
| Normalização | Abstenção/no-hit e ranking | AC 15–16 |
| Métricas | Denominador, vazio, cobertura e Top-5 ausente | AC 18–19 |
| Pares | Contingência, IC, pp e suporte insuficiente | AC 20–22 |
| Interpretação | Genoma/banco desconhecido | AC 23 |
| Saídas | Repetição e concorrência | AC 24; existing - reserva exclusiva C00 e publicação somente após conclusão |
| Smoke/verificação | Bloqueio empírico, pendências e divergências | AC 25–26 |
| Console e log | Caminhos de consultas, predições, referências e pesos | existing - `log_artifact_path`, contrato LOG_PATH de AGENTS.md |
| HTTP | Autorização, rate limiting e paginação | n/a - CLI e notebook locais, sem serviço HTTP |
| Interface visual | Loading, navegação e layout | n/a - sem tela nova; figura usa contagens observadas do benchmark |

As nove dimensões foram consideradas: validação (AC 11–17), falha parcial (AC 10), repetição (AC 24), autorização (n/a — execução local), concorrência (reserva C00 e estados COMPLETE/FAILED), ciclo dos dados (AC 2–9 e 24), dependência externa (AC 1, 7 e 26), transições (AC 7 e 10) e observabilidade (AC 18–26). A associação às provas será feita em `checks.md` depois da revisão do plano.

## Sources

- `ROADMAP_CORRECOES_REPRODUCAO.md`, C03 e portões; `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` e AD-009/AD-011 — escopo, ruído, classes, denominador e proveniência.
- Specs S07/N06/R06 e verificação C00; `short/benchmark_cnns_kraken2_blast_overlap.ipynb` — comportamento e geração histórica preservados, com divergências de protocolo identificadas.
- Notebook de reprodução 09, `baselines/overlap.py`, `taxonomy.py`, `paired.py`, CLI/testes relacionados e `docs/baseline_benchmark_protocol.md` — contratos atuais e lacunas; artefatos locais de 40 consultas não constituem reprodução empírica.
