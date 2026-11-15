Created At: 2026-11-15T13:35:00-03:00
Completed At: 2026-11-15T13:35:00-03:00
Verdict: PASS
Status: PASS
Profile: light
Round: 1 - full
Verifier: independent sub-agent (author != verifier)
Plan: `.specs/features/c06-integracao-realista-selecao-c06/plan.md`
Checks: `.specs/features/c06-integracao-realista-selecao-c06/checks.md`

# C06 — Integracao Realista e Selecao C06 verification

## Scope Resolution

A verificação independente avaliou a suíte de integração e os módulos de suporte implementados para a especificação C06 no repositório `cnn_metagenome`. O escopo cobriu a validação de formatos brutos exportados pelos cadernos de reprodução (`short/01`, `short/02`, `short/06b`, `short/07`, `short/08`, `short/09`, `short/10`, `short/11`, `long/03`), a execução da cadeia de dependências entre cadernos (`07 -> 08`, `C03 -> 09`, `06 -> F2`, `long/01 -> 03 -> T12`), a bateria fail-closed de testes negativos de integridade científica e criptográfica, e a proposição assistida de configurações candidatas a partir de `run_plan.json`.

- **Arquivos modificados no escopo:**
  - `src/viral_cnn/reporting/publication.py`: implementação de `build_candidate_publication_config`, geração de relatório estruturado de diferenças entre artefatos configurados e candidatos observados, e fortalecimento das regras de diagnóstico e rejeição estrita (T11 com exigência de teste nulo por permutação de nós $N_{\text{perm}} \ge 100$).
  - `src/viral_cnn/reporting/__init__.py`: exportação da função `build_candidate_publication_config`.
  - `scripts/build_candidate_publication_config.py`: script CLI autônomo com suporte a `--run-plan`, `--base-config`, `--output`, `--project-root` e formatação JSON/texto.
  - `tests/test_c06_realistic_integration.py`: suíte com 10 testes de integração e testes negativos fail-closed.
- **Conformidade de fronteiras:** inspeção de `git diff` e `git status` confirmou que nenhuma alteração ultrapassou os limites acordados no plano C06.

## Verification Results

Todos os 10 checks definidos no plano e em `checks.md` foram executados e validados de maneira independente utilizando `.venv/bin/python` e `MPLCONFIGDIR=/tmp/mpl`:

| Check | Description | Command Executed | Evidence | Proof | Result |
| --- | --- | --- | --- | --- | --- |
| C1 | `build_candidate_publication_config` mapeia os estágios de `run_plan.json` e calcula SHA-256 dos artefatos encontrados (AC 1) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_candidate_config_maps_stages tests/test_c06_realistic_integration.py` | `tests/test_c06_realistic_integration.py:448` | Ran 1 test in 0.049s (OK) | PASS |
| C2 | `build_candidate_publication_config` emite relatório estruturado de diferenças entre configuração base e candidatos (AC 2) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_candidate_config_emits_diff_report tests/test_c06_realistic_integration.py` | `tests/test_c06_realistic_integration.py:481` | Ran 1 test in 0.051s (OK) | PASS |
| C3 | A invocação CLI de `scripts/build_candidate_publication_config.py` gera configuração candidata e código de saída apropriado (AC 3) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_candidate_config_cli tests/test_c06_realistic_integration.py` | `tests/test_c06_realistic_integration.py:509` | Ran 1 test in 0.676s (OK) | PASS |
| C4 | A transformação de dados brutos realistas via `execute_publication_transformation` emite artefatos e manifestos derivados válidos (AC 4) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_realistic_raw_transformations_emit_derived tests/test_c06_realistic_integration.py` | `tests/test_c06_realistic_integration.py:540` | Ran 1 test in 0.720s (OK) | PASS |
| C5 | A montagem estrita via `assemble_publication(strict_empirical=True)` aceita todos os 17 itens como `DERIVED_VERIFIED` ou `EMPIRICAL_VERIFIED` (AC 5) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_realistic_assembly_all_17_items tests/test_c06_realistic_integration.py` | `tests/test_c06_realistic_integration.py:553` | Ran 1 test in 4.131s (OK) | PASS |
| C6 | O teste integrado exerce a cadeia de dependências entre cadernos `07->08`, `C03->09`, `06->F2` e `long/01->03->T12` (AC 6) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_inter_notebook_dependencies_chain tests/test_c06_realistic_integration.py` | `tests/test_c06_realistic_integration.py:613` | Ran 1 test in 0.538s (OK) | PASS |
| C7 | A montagem em modo estrito rejeita artefatos com hash SHA-256 adulterado com `PublicationValidationError` (AC 7) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_negative_tampered_hash tests/test_c06_realistic_integration.py` | `tests/test_c06_realistic_integration.py:635` | Ran 1 test in 0.563s (OK) | PASS |
| C8 | A validação de seleção rejeita divergência de split ID, seed ou ordem de classes com status `INVALID` (AC 8) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_negative_scientific_identity_mismatch tests/test_c06_realistic_integration.py` | `tests/test_c06_realistic_integration.py:649` | Ran 1 test in 0.570s (OK) | PASS |
| C9 | A validação estrita rejeita itens com dependências transitivas sintéticas (`is_smoke_test=True`) (AC 9) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_negative_synthetic_dependency tests/test_c06_realistic_integration.py` | `tests/test_c06_realistic_integration.py:664` | Ran 1 test in 0.721s (OK) | PASS |
| C10 | O sistema rejeita métricas com zeros fabricados (`0.0`), CSV com comentários sem parsing e agregações T12 incompletas (AC 10) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_negative_data_integrity_violations tests/test_c06_realistic_integration.py` | `tests/test_c06_realistic_integration.py:680` | Ran 1 test in 0.038s (OK) | PASS |

- **Execução consolidada da suíte C06:**
  - Comando: `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest tests/test_c06_realistic_integration.py`
  - Resultado: 10 testes executados em 7.715s, 0 falhas, 0 erros.
- **Execução da suíte de regressão de reporting de publicação (C01–C06 + M07):**
  - Comando: `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest tests/test_c06_realistic_integration.py tests/test_c05_figure_readiness.py tests/test_c04_t11_scientific_decision.py tests/test_c03_publication_adapters.py tests/test_c02_derived_manifests.py tests/test_c01_data_contracts.py tests/test_reporting_figures.py tests/test_m07_publication_contracts.py`
  - Resultado: 89 testes executados em 12.821s, 0 falhas, 0 erros.

## Boundary Conditions Checked

1. **One-Way Door 1 — Teste de integração ponta a ponta realista:**
   - O teste integrado alimenta os 17 itens de publicação (T1–T12 e F1–F5) utilizando formatos brutos fieis aos exportados pelos cadernos de reprodução (`short/01`, `short/02`, `short/06b`, `short/07`, `short/08`, `short/09`, `short/10`, `short/11`, `long/03`).
   - `execute_publication_transformation` processa os esquemas brutos e emite manifestos derivados vinculados à linhagem original, com verificação transitiva completa em `assemble_publication(strict_empirical=True)`, aceitando todos os 17 itens como `DERIVED_VERIFIED` ou `EMPIRICAL_VERIFIED` (comprovado em C4 e C5).

2. **One-Way Door 2 — Proteção fail-closed incondicional:**
   - Sob `strict_empirical=True`, o validador bloqueia a montagem com `PublicationValidationError` ou atribui status `INVALID` diante de qualquer quebra de integridade:
     - Adulteração de hash SHA-256 de arquivo ou manifesto pai (comprovado em C7).
     - Divergência de identidade científica (split ID, semente ou ordem de classes) (comprovado em C8).
     - Linhagem transitiva contaminada por dados sintéticos (`is_smoke_test=True`) (comprovado em C9).
     - Preenchimento artificial de medições com zero (`0.0`), tabelas com linhas comentadas sem parser e pares de agregação de T12 incompletos (comprovado em C10).

3. **One-Way Door 3 — Proposição assistida com revisão humana:**
   - A rotina `build_candidate_publication_config` e o script CLI `scripts/build_candidate_publication_config.py` mapeiam saídas de estágios a partir de `run_plan.json`, calculando hashes criptográficos e emitindo relatório comparativo estruturado (`UNCHANGED`, `PROPOSED`, `MISSING`).
   - O construtor impõe a exigência explícita de revisão humana (`review_required=True` e diretiva formal), recusando a promoção cega de arquivos pelo último timestamp (comprovado em C1, C2 e C3).

## Non-Functional Properties Swept

- **Validation:** Coberta em C1, C4, C7, C8, C9 e C10 garantindo validação prévia de esquemas brutos, conformidade de contratos e integridade de manifestos.
- **Failure modes:** Comprovado em C7, C8, C9 e C10 mediante lançamento de `PublicationValidationError` com diagnósticos específicos.
- **Idempotency:** As transformações e o processo de montagem são determinísticos, preservando integralmente os arquivos brutos e produzindo saídas idênticas sob os mesmos artefatos e sementes.
- **Authorization:** Não aplicável; execução restrita ao sistema de arquivos local.
- **Concurrency:** Não aplicável; pipeline projetado para execução sequencial determinista.
- **Data lifecycle:** Isolamento completo de saídas no diretório de destino da publicação (`derived/` e diretórios de execução).
- **Dependency failure:** Coberta em C6, C7, C8 e C9, bloqueando o avanço na presença de dependências corrompidas ou elos faltantes na cadeia inter-notebooks.
- **State transitions:** Transições para `DERIVED_VERIFIED` ou `EMPIRICAL_VERIFIED` são auditadas e irreversíveis sem atendimento integral dos critérios.
- **Observability:** Todas as funções registram caminhos absolutos ou relativos inspecionados (`[LOG_PATH]`) e retornam relatórios estruturados para auditoria.

## Residual Risks & Recommendations

- **Execução em cluster GPU (C07):** Os testes de integração utilizam fixtures representativas dos formatos brutos; a transição para a execução em larga escala no cluster de GPU (M00/C07) exigirá monitoramento contínuo de tempos de processamento e alocação de memória RAM/VRAM.
- **Revisão humana mandatória:** O relatório estruturado de diferenças gerado por `build_candidate_publication_config.py` deve ser inspecionado manualmente antes de promover qualquer seleção de artefatos à configuração definitiva de publicação.
