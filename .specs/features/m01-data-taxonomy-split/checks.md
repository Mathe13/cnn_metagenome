# M01 checks

Profile: light
Plan: `.specs/features/m01-data-taxonomy-split/plan.md` — aprovado pelo usuário (“prossiga”).

23 checks em cinco slices; três portas; nenhuma questão bloqueante.

## Checks

### S1

**C1** — WHEN executa 10, 11 ou 05 THEN o fluxo SHALL resolver e registrar os caminhos selecionados segundo a precedência definida em Surface. (AC 1).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_paths -k test_paths`

**C2** — WHEN executa duas análises sem plano THEN o runtime SHALL produzir destinos distintos sem alterar fontes ou saídas históricas. (AC 2).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_isolation -k test_isolation`

**C3** — WHEN executa uma etapa selecionada do plano M00 THEN o fluxo SHALL usar seus run_id, parent_run_id e destino reservado vazio. (AC 3).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_plan_reservation -k test_plan_reservation`

**C4** — IF destino contém produtos, há override conflitante ou falta entrada empírica obrigatória THEN o fluxo SHALL falhar antes de publicar manifesto COMPLETE. (AC 4).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_rejection -k test_rejection`

**C5** — WHEN conclui cada análise THEN o fluxo SHALL publicar manifesto individual com protocolo, modo, configuração, versões e identidades das entradas e produtos reais. (AC 5).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_manifests -k test_manifests`

### S2

**C6** — WHEN audita splits completos THEN a análise 10 SHALL registrar N por split/classe, distribuição de comprimentos e linhas vazias após remover padding para train/validation/test. (AC 6).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_split_counts -k test_split_counts`

**C7** — WHEN calcula overlap de test contra train THEN a análise 10 SHALL registrar correspondências diretas, RC, interseção e união sem dupla contagem. (AC 7).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_orientations -k test_orientations`

**C8** — WHEN calcula a taxa principal THEN a análise 10 SHALL usar todas as linhas de test no denominador, inclusive vazias e sem A/C/G/T, reportando o subconjunto compatível separadamente. (AC 8).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_denominator -k test_denominator`

**C9** — IF faltam IDs genômicos confiáveis por fragmento em auditoria integral THEN a decisão 10 SHALL manter `PROVENANCE_REQUIRED`. (AC 9).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_provenance -k test_provenance`

**C10** — IF há limite de linhas de test THEN a decisão 10 SHALL registrar `INCONCLUSIVE_PARTIAL_TEST` e o limite aplicado. (AC 10).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_pilot -k test_pilot`

**C11** — WHEN emite o relatório 10 THEN o fluxo SHALL identificar overlap de fragmentos train→test como população distinta das consultas C03. (AC 11).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_population -k test_population`

### S3

**C12** — WHEN audita o híbrido THEN a análise 11 SHALL detectar IDs duplicados/ausentes/extras, model_label divergente de id-1 e nomes originais alterados contra o CSV selecionado. (AC 12).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_mapping -k test_mapping`

**C13** — IF decisão/CSV de 10 têm totais incompatíveis, identidade divergente ou escopo piloto THEN a análise 11 SHALL rejeitar a estratificação como auditoria integral. (AC 13).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_dependency -k test_dependency`

**C14** — WHEN faltam Genus/Order ou existem placeholders THEN 11/05 SHALL registrar labels afetados e `hierarchy_ready=false`, sem preencher ranks ou remover classes silenciosamente. (AC 14).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_ranks -k test_ranks`

**C15** — WHEN estratifica overlap por rank THEN a análise 11 SHALL conservar o denominador completo e reportar separadamente os rótulos com rank não resolvido. (AC 15).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_strata -k test_strata`

**C16** — WHEN audita o Parquet existente em 05 THEN o fluxo SHALL emitir `m01_existing_taxonomy_audit_v1`, preservando a fonte e sem afirmar reconstrução pela planilha MSL. (AC 16).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_existing -k test_existing`

### S4

**C17** — IF falta a planilha ICTV no modo empírico `curate` THEN o notebook 05 SHALL levantar FileNotFoundError sem produzir Parquet sintético ou manifesto COMPLETE. (AC 17).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_missing_workbook -k test_missing_workbook`

**C18** — IF faltam abas/colunas obrigatórias na planilha ou o CSV de classes é inválido THEN o fluxo 05 SHALL rejeitar a curadoria antes de publicar Parquet validado. (AC 18).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_invalid_workbook -k test_invalid_workbook`

**C19** — WHEN conclui `curate` com identidade de classes e Genus/Order válidos THEN 05 SHALL publicar novo Parquet sob `m01_ictv_curate_v1` com a planilha/CSV e eventual fallback NCBI local identificados. (AC 19).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_curate -k test_curate`

**C20** — WHILE executa smoke explicitamente THEN o fluxo 05 SHALL identificar seus produtos/manifestações como sintéticos, mantendo essa origem em dependências de 11. (AC 20).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_synthetic -k test_synthetic`

**C21** — WHEN falta evidência de accession por fragmento THEN o relatório 05 SHALL manter proveniência genômica desconhecida, inclusive quando a taxonomia estiver válida. (AC 21).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_unknown_genome -k test_unknown_genome`

### S5

**C22** — WHEN carrega artefatos persistentes nos fluxos M01 THEN o código SHALL registrar seus caminhos por LOG_PATH, respeitando LOG_PATH=0/false. (AC 22).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_logging -k test_logging`

**C23** — WHEN documenta M01 THEN a documentação SHALL fornecer comandos/parâmetros exatos para 10→11→05, caminhos dos manifestos e distinção entre smoke local, preflight remoto e validação empírica. (AC 23).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split.M01Tests.test_documentation -k test_documentation`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Assemblies (3) | short/10 C6 · short/11 C15 · short/05 C17 | - |
| Manifest states (2) | COMPLETE C5 · FAILED C4 | - |
| Modes (3) | curate C19 · audit_existing C16 · smoke C20 | - |
| Splits (3) | train C6 · validation C6 · test C6 | - |
| Orientation (4) | direct-only C7 · RC-only C7 · both C7 · neither C7 | - |
| Mapping failures (5) | duplicate C12 · missing C12 · extra C12 · label-offset C12 · renamed-original C12 | - |
| Ranks (2) | Genus C14 · Order C14 | - |
| Provenance (3) | PROVENANCE_REQUIRED C9 · INCONCLUSIVE_PARTIAL_TEST C10 · unknown C21 | - |
| Doors (3) | manifest C5 · population C11 · taxonomy-modes C16 | - |
| Entities (4) | round C3 · analysis C2 · predecessor C13 · class C12 | - |

## Swept

- validation: C4, C12, C14, C18
- failure modes: C4, C17, C18
- idempotency: C2, C3, C4
- authorization: n/a — execução local sob permissões do sistema
- concurrency: C3, reserva exclusiva com claim de execução atômico
- data lifecycle: C2, C5, C16
- dependency failure: C4, C13, C17
- state transitions: C4, C5, C20
- observability: C5, C22, C23

## Handoff

Um builder. Arquivos existentes (runtime ~39 KB, planejamento ~22 KB, três notebooks ~30 KB, auditorias/curadoria ~35 KB, testes N04 ~23 KB, README/configs ~35 KB) somam ~184.000 bytes / 4 = 46.000 tokens; auxiliares/testes/documentação novos estimados em 64.000 bytes / 4 = 16.000 tokens; skill/plano ~20.000 tokens. Total estimado 82.000 < orçamento padrão 150.000. Sem handoff; Verifier independente após último commit de implementação. Perfil light. Testes sintéticos não substituem preflight remoto nem validação empírica real.

- **Boundary:** C1–C23 implementados; 27 testes M01, 51 regressões M00/C00/runtime/taxonomia/split/dataset, cinco provas N04 afetadas e uma prova N09 de SQLite passaram (84 testes). Validações de plan/checks: zero erros/avisos. Revisão independente pendente após commit.
- **Settled mid-build:** notebooks usam `m01_workflow` como camada comum; 05 não atribui proveniência genômica a táxons sintéticos, e testes N04 observam o namespace da execução. openpyxl 3.1.5 foi instalado e registrado para testar leitura XLSX real mínima.
- **Abandoned:** nenhum retreinamento, consulta NCBI, mudança de heurística fuzzy, migração de fontes históricas ou execução empírica integral.

- **Review round 1:** Verifier independente em `21dfc84`: 16/23 comprovados, 84 testes verdes; C1/C2/C5/C6/C13/C19/C22 com lacunas de alcance. Relatório preservado em `verification-round1.md`. Claims/seletores não foram alterados.
- **Proof repair:** os sete testes nomeados foram ampliados sem remover exigências: entradas reais via três notebooks/plano/defaults, sentinela histórica, protocolos/modos/hashes reais, matriz split/classe/comprimentos, totais incompatíveis com hashes válidos, fallback NCBI local aplicado e logs 10/11/05. 27/27 testes M01 verdes após ampliação; nenhuma mudança de implementação nesta rodada. Reverificação independente pendente.

- **Boundary final:** Verifier independente PASS 23/23 em `c6821ad`, Round 2 scoped, perfil light; gate zero erros/avisos. Provas M01 reexecutadas (27) e regressões carregadas explicitamente da rodada 1 (57). Sete gaps encerrados sem alterar implementação/claims.
- **Validation incident:** execução inicial do 05 legado em smoke na raiz sobrescreveu sete Parquets ignorados; não há hashes anteriores para restaurá-los. O PASS cobre invariância do fluxo novo, não preservação global da árvore ignorada durante o BUILD. CSVs Species/Genus têm hashes confirmados. Registro completo no relatório, docs e L-011. Preflight remoto e validação empírica continuam pendentes.
