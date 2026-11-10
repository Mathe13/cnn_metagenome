# M00 — Referência preservada e plano executável da reprodução

## Problem

A rodada de reprodução combina namespaces exclusivos com notebooks que ainda escrevem em pastas fixas. Não existe um plano persistido da rodada que enumere as entradas selecionadas de cada etapa e diferencie produtos futuros de evidência já disponível. O pesquisador precisa resolver essas dependências manualmente, com risco de misturar saídas históricas e novas.

O roadmap de 09/11/2026 determina M00 antes das demais fases. Nesta entrega, a população científica permanece closed-set com split por fragmentos; não há cálculo de métricas nem execução integral. O resultado será um inventário de referência e um plano inspecionável pelo preflight, com impedimentos explícitos antes do processamento pesado.

## Flow

Reutilizar `reserve_run_directory`, `JupyterHubDataTopology`, `file_identity` e o preflight existentes em `viral_cnn.notebook_runtime`; preservar os validadores C03/C09 em vez de criar outra interpretação de seus protocolos.

1. Configuração explícita e ambiente entram no runtime (exists), que resolve entradas e modo sem criar fixtures em inspeção empírica.
2. Caminhos de referência entram no inventário M00 (door 1), que persiste metadados e hashes limitados por orçamento no namespace recém-reservado pelo runtime (exists).
3. Seleção de etapas, recursos e manifestos entra no plano M00 (door 2), que fixa IDs, destinos e dependências, incluindo os produtos ainda planejados.
4. O plano entra no preflight (exists), que inspeciona metadados e dependências disponíveis; entradas futuras ficam pendentes até existirem e serem verificadas. O relatório não executa notebooks.
5. A CLI de planejamento (door 2) publica inventário, plano e relatório no novo namespace. A CLI `scripts/preflight_notebook.py` (exists) permite reinspeção de uma etapa pelo plano antes da sua execução.

## Impact

| Front | What changes |
| --- | --- |
| arquivos previstos | Runtime, CLI de preflight, uma CLI leve de planejamento, configuração M00, README e testes dos contratos. Novos auxiliares em `src/viral_cnn/` podem ser extraídos conforme as convenções. |
| referência | Inventariar `paper/`, `results_reproduction/` e raízes externas explicitamente configuradas, sem mover/copiar fontes. Excluir o namespace da rodada atual do inventário. |
| identidade | Acrescentar vínculo de rodada `parent_run_id` e IDs de etapas ao plano sem reinterpretar manifestos C00/C03/C09 legados. |
| prontidão | `READY` passa a descrever requisitos técnicos da etapa inspecionada; dependência apenas planejada não equivale a manifesto completo. |
| compatibilidade | Preservar uso atual de preflight sem plano e namespaces C00. A adoção do plano pelos notebooks com caminhos fixos fica nas fases seguintes. |
| números e resultados | Nenhum número do artigo é recalculado; apenas arquivos administrativos novos são gerados. Não alterar pesos, split, ordem A/C/G/T, ruído, métricas ou população. |
| configurações científicas | `default_species.yaml` declara dropout 0,3; `models.yaml` registra 0,5/0,7 nas variantes. M00 registra hashes de ambos, sem escolher silenciosamente qual reconstrói um modelo. |

## Relations

Uma rodada possui um inventário, um plano e relatórios de inspeção. Cada etapa possui um destino exclusivo e pertence a uma rodada. Cada dependência aponta para uma etapa planejada ou para um manifesto externo explicitamente selecionado; não pode apontar para si própria nem formar ciclos. Manifestos externos e seus produtos são somente leitura. Uma retomada é entrada de uma nova execução; a validação especializada C09 continua obrigatória antes de reutilizar seus caches.

## Surface

None - nenhuma rota HTTP ou serviço externo. As assinaturas das CLIs locais são descritas abaixo; seus status são códigos de saída de processo.

CLI de planejamento M00: `--config` obrigatório; `--data-root` com precedência sobre DATA_ROOT/config; `--run-id` opcional com geração exclusiva se omitido; `--smoke-test` desligado por padrão. Produz `reference_inventory.csv`, `run_plan.json`, relatório JSON e resumo de impedimentos; código 0 quando pronto e 2 quando bloqueado, com configuração inválida ou colisão.

`scripts/preflight_notebook.py`: notebook_id e `--run-plan` opcional; sem plano mantém opções atuais. Produz relatório JSON por etapa e requisitos resolvidos; código 0 para READY e 2 para BLOCKED.

A CLI somente inspeciona e reserva destinos: não dispara Jupyter, downloads, inferência ou treinamento. Recursos esperados são declarações do operador (dispositivo, memória e armazenamento), com `unknown` quando não fornecidos; não são estimativas de desempenho medidas. Logs de caminhos usam `LOG_PATH` existente; o relatório persistido contém o diagnóstico completo mesmo se o log estiver desativado.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — referência auditável | `reference_inventory.csv`: `path`, `size_bytes`, `sha256`, `hash_status`, `origin_status`. SHA-256 por arquivo regular até `hash_max_bytes=16777216` por padrão; maiores ficam `sha256=unknown`, `hash_status=skipped_size`; ausentes/ilegíveis são registrados explicitamente. Links simbólicos são registrados sem percorrer diretórios externos implicitamente. | Copiar checkpoints/Parquets ou calcular hash irrestrito de tudo: custo desnecessário de armazenamento/leitura antes de conhecer o orçamento. Usar nome/data como prova de origem: não comprova identidade. |
| 2 — plano versionado da rodada | `protocol_id=m00_run_plan_v1`, `schema_version=1`; `run_plan.json` e inventário em `results_reproduction/runs/<parent_run_id>/m00/`. Etapas com `run_id`, `parent_run_id`, `notebook_id`, `output_dir`, ação `run/skip`, motivo, recursos e dependências com manifesto explícito. Dependência `planned` ou `external`; inspeção da etapa `READY/BLOCKED/SKIPPED`. Publicação por temporário e substituição atômica apenas no namespace novo. | Descoberta por arquivo mais recente ou reuso do diretório de origem: mistura evidências e invalida a imutabilidade. Marcar como READY um manifesto que ainda será produzido: esconde uma dependência ausente. |

O plano persiste commit Git, árvore modificada, configuração/hash, modo, semente 42, `population_id` e `split_id` (inclusive `unknown`) e caminhos exatos dos manifestos. Seleção de dependências empíricas registra SHA-256 do manifesto disponível; a reinspeção exige correspondência. Hash de manifesto não certifica sozinho seus produtos nem substitui validadores científicos especializados.

## Criteria

### S1: referência preservada e destinos exclusivos (P1)

**Acceptance Criteria**

1. WHEN inventaria as raízes configuradas THEN o sistema SHALL registrar cada arquivo com caminho, tamanho, estado de origem e estado do hash em `reference_inventory.csv`.
2. WHEN um arquivo regular tem até `hash_max_bytes` THEN o inventário SHALL registrar seu SHA-256 calculado sobre todo o conteúdo.
3. IF um arquivo excede o limite, está ausente ou é ilegível THEN o inventário SHALL registrar a causa e `sha256=unknown` sem copiar o arquivo nem interromper a enumeração dos demais.
4. WHEN cria uma rodada THEN o sistema SHALL reservar destinos exclusivos sob `results_reproduction/runs/` sem modificar referências históricas.
5. IF um destino já existe ou um ID permite sair de `runs/` THEN o sistema SHALL recusar a reserva preservando os arquivos existentes.

**Independent test:** árvore temporária com arquivos pequenos, arquivo acima do limite, origem ausente e sentinelas; duas rodadas e duas tentativas concorrentes de reserva do mesmo destino; comparação de conteúdo das fontes antes/depois.

### S2: dependências e intenção da execução explícitas (P1)

**Acceptance Criteria**

6. WHEN gera o plano THEN o sistema SHALL registrar identidade da rodada, commit/dirty, configuração/hash, modo, semente, população/split e recursos declarados, usando `unknown` para identidade não comprovada.
7. WHEN usa a configuração padrão THEN o plano SHALL incluir short/01–11, short/06b e long/01–04, com short/03 e short/04 em `skip` justificado pelo uso de checkpoints existentes.
8. WHEN registra dependências THEN o plano SHALL fixar manifestos para 07→08, 06→06b, C03→09 e long/01→02→03→04, identificando explicitamente cada origem futura ou externa.
9. IF falta seleção obrigatória, há ID duplicado, ciclo ou dependência desconhecida THEN o planejador SHALL retornar código 2 com a causa antes de publicar um plano válido.
10. WHEN seleciona um manifesto externo existente THEN o plano SHALL registrar caminho e SHA-256, tratando a origem como somente leitura.
11. WHEN seleciona uma origem de retomada 06/06b THEN o plano SHALL distingui-la de uma reserva nova e declarar a validação C09 pendente até que seu validador específico a confirme.

**Independent test:** configuração pequena com DAG válido e casos negativos de ciclo/duplicata/origem omitida; conferir caminhos exatos dos manifestos e skips sem usar relógio para escolher entradas.

### S3: prontidão por etapa sem simular execução empírica (P1)

**Acceptance Criteria**

12. WHEN inspeciona uma etapa do plano THEN o preflight SHALL enumerar entradas resolvidas, ausências e incompatibilidades de esquema/metadados, incluindo short/06b.
13. IF uma dependência é apenas planejada, está incompleta, é sintética no modo empírico ou tem hash divergente THEN o preflight SHALL retornar `BLOCKED` com a causa.
14. WHILE o modo é empírico o preflight SHALL inspecionar caminhos, cabeçalhos, metadados e manifestos sem criar fixtures, carregar pesos, iterar sequências, executar notebooks ou gerar métricas.
15. WHEN publica arquivos administrativos THEN o planejador SHALL usar escrita temporária no namespace novo, sem apresentar arquivo parcial como plano pronto após falha de escrita.
16. WHEN a etapa é `skip` justificado THEN o relatório SHALL identificá-la como `SKIPPED`, sem criar resultado ou manifesto científico para essa etapa.
17. WHEN o usuário reinspeciona um notebook sem fornecer plano THEN a CLI SHALL preservar as opções atuais de preflight e códigos 0/2.
18. WHEN documenta a entrega THEN o README SHALL fornecer comandos exatos de planejamento e reinspeção, distinguindo testes sintéticos locais, preflight remoto pendente e reprodução empírica não executada.

**Independent test:** Parquet/CSV e manifestos mínimos em diretório temporário; bloquear criação de modelos/leitura de linhas; adulterar manifesto após seleção; simular falha de escrita. Executar provas locais com `REMOTE_SERVER=False`. Preflight no servidor será uma evidência separada, obrigatória antes da rodada integral.

## Out of scope

| Excluded | Why |
| --- | --- |
| Migrar saídas dos notebooks 05/07/08/10/11/long04 | A adoção por cada produtor pertence às fases seguintes; M00 prepara o contrato comum. |
| Corrigir fallback de 05 e variáveis forçadas de 09 | Correções M01/M02, a executar após M00; a existência do plano não torna esses notebooks seguros por si só. |
| Validar as 35 execuções C03 ou todos os caches C09 integralmente | Contratos especializados existentes/fases M02/M03; M00 fixa suas fontes e bloqueia dependências pendentes. |
| Decidir T11/F5, redesenhar F3 ou montar C06 | Dependem de M04/M05/M07 e de revisão científica própria. |
| Executar a rodada remota, retreinar modelos ou atualizar manuscrito | Dados/checkpoints canônicos não estão montados; execução pesada e M08 ficam posteriores. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Recorte inicial | Apenas M00 nesta feature | É a fase sem dependências que habilita as demais; mantém obrigações verificáveis. | n |
| Perfil | light | Padrão da skill, sem override de projeto. Verificador independente obrigatório após implementação. | n |
| Recursos do cluster | unknown até configuração explícita | Não há medição de hardware ou disponibilidade nesta sessão. | n |

**Open questions:** none — defaults acima para revisão humana.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| CLI de planejamento | flags, defaults, relatórios e falhas | Surface; Criteria 4–9, 15–16 |
| CLI de preflight | modo com/sem plano, status e erro por entrada | Criteria 12–14, 17 |
| Inventário | ordenação por caminho, duplicatas por caminho resolvido, hashes e ausências | Criteria 1–3; Landing 1 |
| Plano | namespaces, DAG, manifestos, identidade e skips | Criteria 4–11, 16 |
| Documentação | instruções de uso e limites da evidência | Criterion 18 |
| Tela/API/autorização/rate limit | estados visuais e limites de requisição | n/a — CLIs locais, sem serviço ou interface web |
| Confirmação destrutiva | remoção/migração de referências | n/a — fontes somente leitura, sem exclusão/migração |

Nove dimensões consideradas para derivação posterior: validação (5, 9, 12); falha parcial (3, 15); idempotência/retry (4–5); autorização (n/a — ferramentas locais sob permissões do sistema); concorrência/ordem (4–5, 8–9); ciclo de dados (1–4, 10); dependência externa (12–14); transições (13, 15–16); observabilidade (1, 6, 12, 18).

## Sources

- `ROADMAP_MELHORIAS_REPRODUCTION_2026-11-09.md`, contratos comuns e M00; pedido do usuário de iniciar sua implementação com `tlc-spec-lean`.
- `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` AD-001–027 e `docs/dataset_manifest.md`: invariantes e limites científicos. `default_species.yaml` e `models.yaml` apresentam dropout divergente, preservado como evidência.
- `src/viral_cnn/notebook_runtime.py`, `scripts/preflight_notebook.py`, testes C00 e notebooks de reprodução 05/07/08: base existente. O 08 local já declara diagnóstico descritivo entre variantes; isso difere da descrição de T11 no roadmap e será reconciliado em M05, sem assumir qual versão foi executada remotamente.
