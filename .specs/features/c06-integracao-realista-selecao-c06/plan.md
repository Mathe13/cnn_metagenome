# C06 — Integracao Realista e Selecao C06

## Problem

A integração final da esteira de publicação pré-cluster exige validação de ponta a ponta sem atalhos nem aprovações silenciosas:

1. **Validação realista de formatos brutos de notebooks:**
   - Em testes anteriores, diversas etapas forneciam tabelas sintéticas já prontas no formato final do artigo.
   - O teste de integração realista deve alimentar os 17 itens de publicação (T1–T12 e F1–F5) com dados que imitam com exatidão os formatos reais exportados pelos cadernos de reprodução (`short/01`, `short/02`, `short/06b`, `short/07`, `short/08`, `short/09`, `short/10`, `short/11`, `long/03`), executando `execute_publication_transformation` para produzir as tabelas derivadas e seus manifestos.

2. **Cadeia de dependências inter-notebooks:**
   - O teste integrado deve exercer explicitamente a cadeia de derivação entre cadernos: `07 -> 08` (motivos aprendidos alimentam coocorrência e sinergia de tripletos), `C03 -> 09` (predições pareadas alimentam degradação por ruído), `06 -> F2` (embeddings alimentam projeção dimensional com metadados) e `long/01 -> 03 -> T12` (predições por janela alimentam o consenso longo).
   - Para T11, a decisão metodológica C04 (associação exploratória de 7-mers controlada por modelo nulo de permutação de nós com $N_{\text{perm}} \ge 100$) deve ser rigorosamente respeitada; se a evidência estiver pendente ou faltante, o teste deve reportar o bloqueio formal sem mascarar falhas.

3. **Bateria estrita de testes negativos (Fail-Closed):**
   - Sob `strict_empirical=True`, o sistema deve falhar fechado com diagnóstico específico diante de qualquer violação científica ou criptográfica: adulteração de hash de arquivo fonte ou manifesto pai, divergência de split/seed, ordem trocada de classes, ausência de checkpoint, ancestralidade sintética (`is_smoke_test=True`), CSV comentado sem parser apropriado, coluna faltante, métricas preenchidas com zeros artificiais (`0.0`), substituição indevida do denominador experimental pelo número de linhas do resumo (`len(df)`), seleção de figuras incompleta ou par de agregações de T12 incompleto.

4. **Construção assistida de configuração candidata a partir de `run_plan.json`:**
   - Para permitir a transição segura dos resultados de um run de reprodução para a configuração de publicação sem seleção manual propensa a erros, é necessário fornecer uma rotina (`build_candidate_publication_config`) e script CLI (`scripts/build_candidate_publication_config.py`).
   - O comando lê um `run_plan.json` explícito e uma configuração base, inspeciona os estágios concluídos, propõe caminhos de artefatos e manifestos com seus respectivos hashes SHA-256 e imprime um relatório de diferenças. O processo exige revisão humana antes de promover a seleção a definitiva, recusando a escolha cega do último timestamp.

## Flow

1. Execução de `build_candidate_publication_config` (exists/new) - consome `run_plan.json` e configuração base, inspecionando saídas dos estágios para propor caminhos, hashes SHA-256 e relatório de diferenças entre itens configurados e observados (door 1, door 3).
2. Exportações brutas dos cadernos entren em `execute_publication_transformation` (exists/updated) - aplicar adaptadores registrados para T1–T12 e F1–F5, validando esquemas brutos e emitindo manifestos derivados com linhagem e suporte experimental (door 1, door 2).
3. Artefatos e manifestos derivados entram em `validate_selection` sob `strict_empirical=True` (exists) - validar integridade criptográfica transitiva, identidade científica e rejeitar dados sintéticos ou fabricados (door 1, door 2).
4. Dados validados entram em `assemble_publication` (exists) - renderizar tabelas, gerar figuras e compilar variantes do manuscrito em diretório isolado de publicação (door 1, door 3).

## Impact

| Front | What changes |
| --- | --- |
| domain | new function: `build_candidate_publication_config` - inspeção e construção de configuração candidata de publicação a partir de um plano M00 |
| domain | new script: `scripts/build_candidate_publication_config.py` - interface CLI para proposição assistida de seleção com impressão de diferenças |
| testing | new suite: `tests/test_c06_realistic_integration.py` - teste integrado ponta a ponta dos 17 itens com formatos brutos e bateria de testes negativos |
| stored data | nada a migrar |

## Relations

`None - no stored-data shape change`

## Surface

| Route | In | Out | Status |
| --- | --- | --- | --- |
| `scripts/build_candidate_publication_config.py` | `--run-plan`, `--base-config`, `--output` | `int` | `200, 400` |
| `viral_cnn.reporting.publication:build_candidate_publication_config` | `Path`, `Path`, `Optional[Path]` | `dict[str, Any]` | `200, 400` |
| `viral_cnn.reporting.publication:assemble_publication` | `Path`, `Path`, `bool`, `bool` | `dict[str, Any]` | `200, 400` |

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — Teste de integração ponta a ponta realista | O teste de integração deve produzir saídas nos formatos brutos reais dos cadernos, executar os adaptadores oficiais e validar os 17 itens em modo estrito. | Usar tabelas sintéticas previamente formatadas no padrão final do artigo. |
| 2 — Proteção fail-closed incondicional | Qualquer violação de integridade criptográfica, proveniência ou contrato científico bloqueia a montagem com `PublicationValidationError`. | Tratar violações ou dados ausentes com substituição tolerante por zeros (`0.0`) ou ignorar erros. |
| 3 — Proposição assistida com revisão humana | `build_candidate_publication_config` mapeia e propõe artefatos a partir de `run_plan.json`, mas exige conferência humana e não valida automaticamente sem revisão. | Promover cegamente os artefatos mais recentes baseados em timestamp. |

## Criteria

### S1: Construtor de configuracao candidata a partir de run_plan (P1)

Garantir mapeamento seguro e assistido de saídas de reprodução para a configuração de publicação.

**Acceptance Criteria**

1. WHEN `build_candidate_publication_config` receives a valid `run_plan.json` and base config THEN the system SHALL identify stage output artifacts, compute their SHA-256 hashes and propose matching publication items.
2. WHEN `build_candidate_publication_config` executes THEN the system SHALL compute a structured diff report contrasting current configuration against candidate artifacts.
3. WHEN `scripts/build_candidate_publication_config.py` is invoked via CLI THEN the system SHALL output candidate YAML/JSON and return exit code 0 on success or exit code 2 when required inputs are missing or invalid.

**Independent test:** executar `build_candidate_publication_config` e o script CLI sobre planos e saídas simuladas de `run_plan.json`, verificando hashes e o relatório de diferenças.

### S2: Teste de integracao realista ponta a ponta para os 17 itens (P1)

Garantir que os dados brutos de todas as etapas alimentam a publicação com proveniência e suporte auditados.

**Acceptance Criteria**

4. WHEN realistic raw notebook outputs for all 17 items (T1–T12, F1–F5) are transformed via `execute_publication_transformation` THEN the system SHALL emit verified derived artifacts and manifests.
5. WHEN `assemble_publication(strict_empirical=True)` is executed on the derived publication suite THEN the system SHALL accept all 17 items with status `DERIVED_VERIFIED` or `EMPIRICAL_VERIFIED` and assemble tables, figures, and manuscripts.
6. The system SHALL verify the explicit stage dependencies `07 -> 08`, `C03 -> 09`, `06 -> F2`, and `long/01 -> 03 -> T12`.

**Independent test:** executar teste integrado com fixtures nos formatos brutos dos cadernos, gerando derivados e montando pacote completo.

### S3: Bateria de testes negativos de violacao e seguranca cientifica (P1)

Garantir proteção contra adulterações, dados sintéticos e manipulações estatísticas.

**Acceptance Criteria**

7. IF any artifact or manifest has an altered SHA-256 hash THEN `assemble_publication(strict_empirical=True)` SHALL fail with `PublicationValidationError`.
8. IF split ID, seed, or class order diverges from expected scientific identity THEN the system SHALL reject the item with status `INVALID`.
9. IF any item depends transitively on a synthetic output (`is_smoke_test=True`) THEN the strict gate SHALL reject the item with status `SYNTHETIC` or `INVALID`.
10. IF an empirical metric contains fabricated zero-fills (`0.0`), unparsed commented headers, or missing T12 aggregation pairs THEN the system SHALL fail with `PublicationValidationError`.

**Independent test:** executar testes negativos para cada modo de falha exigido, confirmando rejeição estrita sem exceções silenciosas.

## Out of scope

| Excluded | Why |
| --- | --- |
| Execução completa em GPU dos cadernos de treinamento | Objeto do pacote C07 após validação das correções |
| Download automático de gigabytes de dados remotos | Inviável no ambiente de teste local e CI |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Formatos de saída dos notebooks | Conforme documentado em `docs/publication_data_contracts.md` | Reflete as exportações reais e esquemas validados em C01–C05 | y |
| Modo de validação candidata | Não modifica arquivos originais sem autorização | Preserva isolamento e rastreabilidade | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Script `scripts/build_candidate_publication_config.py` | geração de configuração candidata com CLI | AC 1, AC 2, AC 3 |
| Module `viral_cnn.reporting.publication:build_candidate_publication_config` | função de mapeamento e cálculo de diferenças | AC 1, AC 2 |
| Module `viral_cnn.reporting.publication:assemble_publication` | montagem estrita ponta a ponta e testes negativos | AC 4, AC 5, AC 6, AC 7, AC 8, AC 9, AC 10 |

## Sources

- `PLANO_CORRECOES_CNN_METAGENOME_AGENTS_2026-11-15.md` - Planejamento de correções e especificação do pacote C06
- `docs/publication_data_contracts.md` - Matriz de contratos de dados e esquemas reais dos cadernos
- `.specs/STATE.md` - Decisões AD-035, AD-038, AD-043, AD-044, AD-045, AD-046, AD-047
