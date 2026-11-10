# M02 benchmark intervals calibration verification

**Verdict**: PASS
**Proofs**: revisão independente sobre 1d9cd0f; reparo e provas verdes em d86ece1 (38 testes em 261,774s).
**Verifier**: independent sub-agent (author != verifier)
**Reviewed**: diff `161137f..1d9cd0f`, spec, design, Checks/matriz de tasks e provas; Handoff não usado para julgar.

O Verifier não reexecutou as provas. O autor ampliou os testes e corrigiu F1–F8 em uma passada; não houve segunda revisão independente. Os julgamentos originais estão preservados em verification-round1.md.

As provas da rodada 1 não foram reexecutadas pelo Verifier. `Holds=no` também identifica prova mais fraca que a claim, sem afirmar que o comportamento correspondente está necessariamente errado. Os testes abaixo pertencem a `tests/test_m02_benchmark_intervals_calibration.py`, classe `M02Tests`.

## Checks

| Check | Test | Round 1 | Repair by author | inalterado; prova original mantida |
| --- | --- | --- | --- |
| C1 | `test_notebook09_preserves_environment` | yes — executa 09 e verifica valores do ambiente, modo e produtos sintéticos. | inalterado; prova original mantida |
| C2 | `test_source_paths_and_hashes` | no — só fonte absoluta por ambiente e identidade fornecida manualmente ao helper; plano não é percorrido e preflight relativo diverge do runtime (F2/F3). | preflight relativo/absoluto e fonte via plano; paths/hashes e log |
| C3 | `test_conflicting_plan_source` | yes — conflito com plano falha antes de ocupar a reserva. | inalterado; prova original mantida |
| C4 | `test_c03_completeness_matrix` | no — mutações de ausência/incompletude atingem inspector em uma condição; não exercita os preflights nem hashes corrompidos de preparação/build nas 35 condições (F3/F8). | 35 condições ausentes/FAILED/hash inválido; dois preflights e hashes prep/build |
| C5 | `test_c03_query_content_identity` | no — IDs/rótulos têm hash atualizado, mas sequência alterada mantém hash antigo: pode passar só pela rejeição criptográfica sem verificar conteúdo compartilhado (F3). | segunda realização com hashes coerentes no seletor |
| C6 | `test_c03_classes_reference_abstention` | yes — rejeita digest de classes, referência inexistente e política BLASTN incompatíveis; loader confere classe/referência comum e abstenções. | inalterado; prova original mantida |
| C7 | `test_c03_synthetic_dependency` | no — modifica só o manifesto de método; referência sintética com hashes coerentes passa READY (F1). | método/prep/query/build/referência/nested, hashes coerentes |
| C8 | `test_notebook09_consumes_without_inference` | yes — executa notebook empírico sobre pacote fixture, bloqueia construção/carregamento de CNN e verifica consumo; inspeção confirma ausência de chamadas Kraken2/BLASTN. | inalterado; prova original mantida |
| C9 | `test_three_notebook_reservations` | no — testa begin_m02 standalone nos três IDs, sem executar notebooks nem adoção de plano/parent (F3). | células dos três notebooks, duas reservas e adoção M00/parent |
| C10 | `test_collision_and_missing_inputs` | no — cobre colisão e entrada ausente no complete do helper de 01, não entradas obrigatórias dos três notebooks (F3). | colisão e entradas obrigatórias nos três notebooks, bytes preservados |
| C11 | `test_manifest_identity_and_lineage` | yes — lê manifesto real, exige campos, protocolo/seed/população/split e hashes de entrada/checkpoint/produto; parent de plano depende da lacuna C9. | inalterado; prova original mantida |
| C12 | `test_failed_run_manifest` | no — exceção artificial dentro de guard do helper nos três IDs; não detecta célula sem guard nos notebooks (F3). | falha dentro de células reais nos três fluxos |
| C13 | `test_c04_interval_contract` | no — unidade confere metadados/estimativa .75, mas não verifica os ICs publicados por 01 no nível integration da matriz (F3). | 35 ICs publicados pelo notebook 01; estimativas e parâmetros |
| C14 | `test_validation_temperature_seal` | no — unidade de TemperatureScaler, sem verificar fluxo real fit/clean-validation/seal/test nem sobreposição parcial de partições no notebook (F3). | células 02 com fit→seal→test observado, ruído somente em test, overlap parcial rejeitado |
| C15 | `test_reliability_support_and_n` | no — N e contagens conferidos em renderer com bins idênticos; não vincula ECE/Brier/NLL aos dados nem verifica métricas antes/depois (F4). | logits binários literais; ECE .25, Brier .625 e NLL .836988; valores calibrados e identidades persistidas |
| C16 | `test_gradcam_smoke_only` | no — executa célula empírica e comprova omissão; ramo smoke não é exercitado pelo seletor (F3). | ambos os ramos da célula, inclusive entrada sintética 1×512 |
| C17 | `test_uncertainty_scope` | no — completa helpers vazios; os três notebooks poderiam publicar escopo diferente sem falhar (F3). | manifestos finais reais dos três notebooks |
| C18 | `test_full_metric_grid` | no — tamanho 105, nomes dos métodos e uma linha literal não fixam o produto cartesiano de taxas/estratos nem as três métricas em todas as células (F5). | produto cartesiano 105 único e métricas de todas as células |
| C19 | `test_abstention_denominators` | yes — todas as abstenções preservam denominador agregado, acurácia/cobertura zero e condicional indisponível. | inalterado; prova original mantida |
| C20 | `test_accuracy_coverage_panels` | no — títulos, sete linhas/labels e três painéis conferidos; valores das curvas, bandas Wilson e N exato não são confrontados com CSV (F5). | N literal, coordenadas e bandas do CSV, Wilson [.045586,.699364] |
| C21 | `test_paired_difference_grid` | no — fixa pares, tamanho, +75 pp e metadados; limites apenas precisam conter 75, portanto IC fictício [-100,100] passaria e demais células não são fixadas (F5). | produto cartesiano 150 único, IC [50,100] e caso negativo/suporte desigual [-100,0] |
| C22 | `test_paired_unavailable_intervals` | yes — distingue vazio/espécie única com motivos, preserva estimativa quando possível e exige ambos os limites ausentes. | inalterado; prova original mantida |
| C23 | `test_csv_figure_regeneration_and_hashes` | no — CSV e figuras são hasheados, mas vínculo fonte→figura só existe em retorno Python; teste não exige sua persistência no manifesto/artefato vinculado (F6). | 09 completo; figure_sources.json persistido e hashes transitivos |
| C24 | `test_complete_with_corrupted_product` | yes — produto alterado mantendo COMPLETE/hash antigo fica BLOCKED por hash. | inalterado; prova original mantida |
| C25 | `test_reservation_preserves_bytes` | yes — snapshot de todos os bytes da reserva é idêntico após reuso recusado. | inalterado; prova original mantida |
| C26 | `test_empirical_failure_never_creates_fixtures` | no — chama complete com arquivo inexistente e compara apenas lista de Parquets; não executa os três fluxos nem compara bytes de fontes existentes (F3). | três fluxos empíricos com entradas ausentes e snapshot de fontes |

## Findings

| Finding | Evidence | Fix | Closed |
| --- | --- | --- | --- |
| F1 — Major: referência sintética aceita como empírica | Exemplo próprio 1: referência real da fixture declarada `is_smoke_test=true`, hashes da referência/build/35 condições atualizados; `inspect_c03_package` retorna READY. `baselines/overlap.py:350–368` verifica conteúdo/hash, mas não modo da referência nem da referência aninhada. C7 só altera modo do método. | Rejeitar referência direta/aninhada sintética ou de modo desconhecido; ampliar C7 para cada dependência com hashes coerentes. |`d86ece1` — testes ampliados e guards corrigidos |
| F2 — Major: preflight e notebook resolvem fontes relativas diferentes | Exemplo próprio 2: `DATA_ROOT=/tmp/...`, `C03_PREPARED_DIR=prepared`. Runtime seleciona `/tmp/.../prepared`; `notebook_runtime.py:345` passa string relativa ao inspector, que resolve sob cwd, e bloqueia pacote válido. Pode também inspecionar outro pacote se existir em cwd. | Compartilhar resolução sob DATA_ROOT entre preflight e workflow; C2 deve comparar caminho/hash nas duas entradas e via plano. |`d86ece1` — testes ampliados e guards corrigidos |
| F3 — Major: seletores não sustentam o nível/alcance de integração declarado | C2/C4/C5/C9/C10/C12/C13/C14/C16/C17/C26 nas linhas Checks. Existem testes suplementares úteis (`test_c03_preflight_entrypoints`, `test_c03_rejects_second_noise_realization`, `test_notebook01_complete`, `test_notebook02_complete`, `test_m00_reserved_notebook01`), mas os seletores aprovados podem continuar verdes se regras dos notebooks/planos forem removidas. C5 em particular só prova hash antigo de sequência. | Ampliar os seletores preservando asserções existentes: executar células/fluxos relevantes; testar plano e parent nos três IDs, falhas reais e bytes das fontes; conferir metadados C04 publicados, ordem fit/seal/test e sobreposição parcial; mutações coerentes de conteúdo e modo devem alcançar preflight e avaliação. |`d86ece1` — testes ampliados e guards corrigidos |
| F4 — Major: prova de calibração não verifica vínculo estatístico | C15 usa mesmos bins antes/depois e não lê ECE/Brier/NLL, identidades ou hashes. `test_notebook02_complete` adicional verifica existência das chaves e N, não valores nem vínculo. | Exercitar calibração identificada em notebook/adapter com logits literais; comparar suporte antes/depois, valores esperados ECE/Brier/NLL e entradas/proveniência persistidas, além de N no painel. |`d86ece1` — testes ampliados e guards corrigidos |
| F5 — Major: cardinalidade/legendas/metadados substituem valores científicos nas provas | C18 não exige chaves cartesianas únicas; C20 não confronta dados desenhados/bandas com CSV; C21 aceita quaisquer limites que contenham 75. Para a fixture literal de C21, IC por espécie é [50,100] pp, não um intervalo arbitrário que contenha o ponto. | Fixar conjunto completo de 105/150 chaves e parâmetros, valores/direção das células relevantes; confrontar curvas/bandas/N com CSV e assertar limites literais para casos positivos, negativos e suporte desigual de espécies. |`d86ece1` — testes ampliados e guards corrigidos |
| F6 — Minor: prova não exige persistência do vínculo CSV→figura | C23 guarda `records` só em memória; `run.complete` recebe CSV, e nenhum `figure_sources.json` é escrito pelo teste. Código real de 09 o escreve, mas a prova selecionada não falharia se essa gravação fosse removida. | Executar a publicação de 09 ou persistir e reler o artefato de vínculos, exigir seu hash no manifesto e conferir fonte/PNG/PDF pelas identidades publicadas. |`d86ece1` — testes ampliados e guards corrigidos |
| F7 — Major: seleção explícita de dados/plano ainda não é inteiramente fixa | Inspeção `m02_workflow.py:140–159`: `plan.seed` não entra em pinned, sendo silenciosamente substituído por 42; `M02Run.__init__` usa runtime cujo auto-climb em `notebook_runtime.py:490` pode substituir DATA_ROOT explícito pelo pai com cnn_kraken. C3/C9 não cobrem esses casos. | Validar seed de plano e preservar DATA_ROOT explicitamente selecionado em ctx/configuração; provar rejeição de seed divergente e raiz filha de pai contendo cnn_kraken. |`d86ece1` — testes ampliados e guards corrigidos |
| F8 — Major: preflight 09 remove contrato C00 existente | Inspeção `notebook_runtime.py:346` retorna só campos especializados e remove `requirements`. Autor comunicou regressão externa: 62/63 passam, `TestC00Contracts.test_missing_inputs_block_preflight_without_products` gera `KeyError: requirements`. Não reexecutei essa regressão. | Preservar schema C00 e anexar inspeção especializada; comprovar C4 através do preflight e manter teste C00 existente. |`d86ece1` — testes ampliados e guards corrigidos |

## Independent examples

Executei somente dois exemplos novos em `TemporaryDirectory(dir='/tmp')`, com `REMOTE_SERVER=False`, `LOG_PATH=0`, `PYTHONPATH=src`, `.venv/bin/python3`; importei o construtor `package` da suíte sem executar testes. A fixture contém 4 queries, 13.683 nomes de classes, 35 condições e bytes dummy de checkpoint/bases. Nenhum modelo/ferramenta foi executado.

1. Regra C7, dependência sintética: construir pacote; carregar `reference_manifest` indicado pela preparação, trocar `is_smoke_test` para `True`; regravar hash em `database_build_manifest.reference_manifest_sha256` e em `reference.sha256` dos 35 manifestos, mais hash de build nos dois métodos ferramenta. Resultado observado: `READY`; esperado: `BLOCKED`. Diretório temporário removido ao término.
2. Regra C2/C4, fonte explícita: construir pacote em `<tmp>/prepared`; fornecer ambiente `DATA_ROOT=<tmp>`, `SMOKE_TEST=0`, `C03_PREPARED_DIR=prepared`; chamar `preflight_notebook('short/09', root)` e `begin_m02('short/09', project_root=root)`. Resultado: preflight BLOCKED em `${PROJECT_ROOT}/prepared`, runtime configurado corretamente em `<tmp>/prepared`. Diretório temporário removido ao término.

Comando reprodutível para o segundo caso (o primeiro segue as mutações acima):

```bash
REMOTE_SERVER=False LOG_PATH=0 MPLCONFIGDIR=/tmp/m02-verifier-mpl PYTHONPATH=src .venv/bin/python3 - <<'PY'
import os, tempfile
from pathlib import Path
from unittest.mock import patch
from tests.test_m02_benchmark_intervals_calibration import package
from viral_cnn.notebook_runtime import preflight_notebook
from viral_cnn.m02_workflow import begin_m02
with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
    root = Path(tmp)
    package(root)
    with patch.dict(os.environ, {'DATA_ROOT': str(root), 'SMOKE_TEST': '0', 'C03_PREPARED_DIR': 'prepared'}):
        report = preflight_notebook('short/09', root)
        run = begin_m02('short/09', project_root=root)
        print(report['status'], report['path'], run.configuration['c03_prepared_dir'])
PY
```

## Evidence limits

Cadeia inspecionada: CSV/Parquet e mapeamentos → split por fragmentos → A/C/G/T 512 bp → registro de variantes/checkpoints → inferência C04 em 01/02 ou 35 resultados C03 pré-calculados em 09 → métricas/IC/calibração → CSV/PNG/PDF e manifestos M02. Configurações identificadas: `baseline_c03.yaml`, `reproduction_m00.yaml`, `reproduction_m02.yaml`, `models.yaml`; protocolos C03 `c03_paired_v1`, C04 `c04_stratified_fragment_v1` e `c04_clean_validation_temperature_v1`, M02 `m02_benchmark_audit_v1`.

Artefatos canônicos do cluster, checkpoints reais, bases efetivas e pacote C03 empírico completo não estavam disponíveis para verificação. Provas verdes relatadas pelo autor e exemplos locais sustentam contratos sintéticos, sem reproduzir números do artigo. Não houve comparação quantitativa com PDF/notebooks históricos nem confirmação de restauração dos Parquets locais afetados na M01. Não foi alterado experimento científico; a revisão afeta admissibilidade das futuras execuções 01/02/09. Uma realização de ruído por taxa e bootstrap condicional não estimam variabilidade entre treinamentos nem generalização para genomas não vistos.

Única escrita persistente do Verifier: este `verification.md`; implementação/testes intactos, sem commits. Sem UI/capturas. As regressões C03/C04/calibration/M00/C00/runtime foram executadas separadamente pelo autor. O gate final deve ser executado pelo autor após fechar achados e regenerar provas no novo HEAD; o autor fecha os achados por commit e registra a rodada mecânica final abaixo.

## Repair validation (author)

F1–F8 foram corrigidos em d86ece1. Nenhuma asserção anterior foi removida ou afrouxada; os seletores aprovados continuam os mesmos. C13 revelou arredondamento no resumo F1 (seis casas), enquanto o IC guardava precisão completa: 01 agora publica o ponto com a mesma precisão do IC, preservando definição e denominador. A descrição da CLI passou a declarar que 09 lê tabelas e hashes. Documentação usa python -m nbconvert no ambiente remoto preparado; nbconvert/Jupyter não estão instalados na .venv local e não foram executados aqui.

Gates direcionados: 8 testes de modo/entrada/reserva/C00 passaram; 13/14 provas ampliadas passaram inicialmente. A única falha restante foi a precisão F1; após alinhamento, C13 passou (30,841 s), sem mudar a asserção. A rodada final completa deve confirmar os 38 métodos M02 e as 63 regressões no HEAD corrigido.

Ambiente observado: Python 3.10.12, PyTorch 2.14.1+cu130 (build CUDA 13.0), NumPy 2.2.6, pandas 2.3.3, pyarrow 25.0.1 e Matplotlib 3.10.9. Testes M02 forçam CPU; o wheel CUDA emite aviso de driver antigo na inicialização, sem executar inferência em GPU. Versões Dask/cuDF e demais dependências são registradas pelo runtime quando presentes, sem inferência de versões históricas.

## Changed files

- Notebooks de reprodução: short/01_benchmark_clean_and_noise.ipynb, short/02_analyze_confidence_and_calibration.ipynb e short/09_benchmark_cnns_kraken2_blast_overlap.ipynb.
- Coordenação: src/viral_cnn/m02_workflow.py, notebook_runtime.py, run_planning.py e scripts/preflight_notebook.py.
- Benchmark: src/viral_cnn/baselines/overlap.py, preparation.py e novo benchmark_plotting.py.
- Calibração: src/viral_cnn/eval/calibration.py e confusion.py (batches com IDs).
- Configuração/documentação: configs/reproduction_m00.yaml, novo reproduction_m02.yaml, README.md, docs/m02_benchmark_intervals_calibration.md e .specs/STATE.md.
- Provas: tests/test_m02_benchmark_intervals_calibration.py e arquivos spec/design/tasks/proofs/verification da feature, com relatórios anteriores preservados.

Nenhum diff em short/ ou long/ históricos, checkpoints, mapeamentos canônicos ou resultados empíricos. Para verificar os contratos locais, executar o comando Run m02 de tasks.md; manifesto de artefatos reais e comando da futura rodada estão no guia M02. Não há manifesto empírico novo nesta tarefa.
