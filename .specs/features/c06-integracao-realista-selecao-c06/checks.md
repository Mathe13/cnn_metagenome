# C06 — Integracao Realista e Selecao C06 checks

Profile: light
Plan: `.specs/features/c06-integracao-realista-selecao-c06/plan.md`

10 checks in 3 slices · 3 one-way doors · 0 open, of which 0 block

## Checks

Grouped by the plan's slices; numbering runs across the whole feature.

### S1 - Construtor de configuracao candidata a partir de run_plan · 2 files · 25 KB · ~8k

**C1** - A funcao `build_candidate_publication_config` mapeia os estagios de `run_plan.json` e calcula SHA-256 dos artefatos encontrados (AC 1)
Proof: `.venv/bin/python -m unittest -k test_candidate_config_maps_stages tests/test_c06_realistic_integration.py`

**C2** - A funcao `build_candidate_publication_config` emite relatorio estruturado de diferencas entre configuracao base e candidatos (AC 2)
Proof: `.venv/bin/python -m unittest -k test_candidate_config_emits_diff_report tests/test_c06_realistic_integration.py`

**C3** - A invocacao CLI de `scripts/build_candidate_publication_config.py` gera configuracao candidata e codigo de saida apropriado (AC 3)
Proof: `.venv/bin/python -m unittest -k test_candidate_config_cli tests/test_c06_realistic_integration.py`

### S2 - Teste de integracao realista ponta a ponta para os 17 itens · 1 file · 30 KB · ~10k

**C4** - A transformacao de dados brutos realistas via `execute_publication_transformation` emite artefatos e manifestos derivados validos (AC 4)
Proof: `.venv/bin/python -m unittest -k test_realistic_raw_transformations_emit_derived tests/test_c06_realistic_integration.py`

**C5** - A montagem estrita via `assemble_publication(strict_empirical=True)` aceita todos os 17 itens como `DERIVED_VERIFIED` ou `EMPIRICAL_VERIFIED` (AC 5)
Proof: `.venv/bin/python -m unittest -k test_realistic_assembly_all_17_items tests/test_c06_realistic_integration.py`

**C6** - O teste integrado exerce a cadeia de dependencias entre cadernos `07->08`, `C03->09`, `06->F2` e `long/01->03->T12` (AC 6)
Proof: `.venv/bin/python -m unittest -k test_inter_notebook_dependencies_chain tests/test_c06_realistic_integration.py`

### S3 - Bateria de testes negativos de violacao e seguranca cientifica · 1 file · 25 KB · ~8k

**C7** - A montagem em modo estrito rejeita artefatos com hash SHA-256 adulterado com `PublicationValidationError` (AC 7)
Proof: `.venv/bin/python -m unittest -k test_negative_tampered_hash tests/test_c06_realistic_integration.py`

**C8** - A validacao de selecao rejeita divergencia de split ID, seed ou ordem de classes com status `INVALID` (AC 8)
Proof: `.venv/bin/python -m unittest -k test_negative_scientific_identity_mismatch tests/test_c06_realistic_integration.py`

**C9** - A validacao estrita rejeita itens com dependencias transitivas sinteticas (`is_smoke_test=True`) (AC 9)
Proof: `.venv/bin/python -m unittest -k test_negative_synthetic_dependency tests/test_c06_realistic_integration.py`

**C10** - O sistema rejeita metricas com zeros fabricados (`0.0`), CSV com comentarios sem parsing e agregacoes T12 incompletas (AC 10)
Proof: `.venv/bin/python -m unittest -k test_negative_data_integrity_violations tests/test_c06_realistic_integration.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Configuracao candidata (3) | cand_mapeia_estagios C1 · cand_relatorio_diff C2 · cand_cli_exec C3 | - |
| Integracao ponta a ponta (3) | integracao_derivados C4 · integracao_17_itens C5 · integracao_cadeia_dependencias C6 | - |
| Seguranca e negativos (4) | neg_hash_adulterado C7 · neg_identidade_invalida C8 · neg_dependencia_sintetica C9 · neg_integridade_dados C10 | - |

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1, C4, C7, C8, C9, C10
- failure modes: C7, C8, C9, C10
- idempotency: existing - transformacao e montagem deterministas sem mutacao de dados brutos
- authorization: n/a - operacao em sistema de arquivos local
- concurrency: n/a - execucao sequencial monotarefa
- data lifecycle: existing - isolamento de saidas em diretorio exclusivo de execucao
- dependency failure: C6, C7, C8, C9
- state transitions: existing - transicao para `DERIVED_VERIFIED` auditada
- observability: existing - registro em log dos artefatos inspecionados e carregados

## Handoff

S1-S3 = ~35k tokens acumulados (publication, cli script, realistic integration test suite), bem abaixo do orcamento de 150k - one builder.

- Boundary: C1-C10 implementados e testados em conjunto por um unico builder
