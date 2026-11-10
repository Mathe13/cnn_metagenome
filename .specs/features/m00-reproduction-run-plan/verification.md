# M00 — verification

**Verdict**: FAIL
**Profile**: light
**Diff range**: 9fdaefe9860f75293ce90e38daf6932335026b0f..e5606556d11e01c32b2f889e16619139ff81e589
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier), `/root/verify_m00`

Revisão de toda a feature, dos dez arquivos do diff e dos 18 checks. O HEAD foi confirmado como `e5606556d11e01c32b2f889e16619139ff81e589`. Nenhum código, teste ou check foi alterado. Os artefatos locais não relacionados foram preservados.

## Gate

Provas executadas pelo verificador, em uma única invocação:

```bash
REMOTE_SERVER=False LOG_PATH=0 PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan -v
```

Resultado: **22 testes passaram, 0 falharam**, 30,333 segundos. Os 18 seletores declarados existem no diff, foram encontrados com `rg -n -A 23 '^    def test_' tests/test_m00_run_plan.py` e apareceram individualmente como `ok` no resultado verbose. Os quatro testes adicionais `test_all_partition_schemas_and_paths_are_logged`, `test_csv_and_planned_manifest_inspection`, `test_reference_exclusion_and_directory_symlinks` e `test_reservation_rejects_symlink_alias` também apareceram como `ok`. Suíte verde não encerra as lacunas abaixo.

Nas linhas seguintes, `test_*` identifica o método de `tests.test_m00_run_plan.RunPlanTests` executado nessa invocação; todas as localizações de asserções referem-se a `tests/test_m00_run_plan.py` no HEAD verificado.

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | Inventário ordenado/deduplicado com tamanho e estados | `test_inventory_records` — ok | `tests/test_m00_run_plan.py:48` — `self.assertEqual([r['path'] for r in records], sorted(set(r['path'] for r in records)))`; linhas 50–52 verificam `size_bytes == 10`, `origin_status == 'present'` e `hash_status == 'hashed'`. | PASS |
| C2 | Hash integral até o limite | `test_inventory_hash_boundary` — ok | `tests/test_m00_run_plan.py:57` — `self.assertEqual(rows[0]['sha256'], hashlib.sha256(b'abcd').hexdigest())`; linha 58 exige `skipped_size` ao reduzir o limite de 4 para 3. | PASS |
| C3 | Grande/ausente/ilegível/symlink explícitos, sem travessia/cópia | `test_inventory_exceptions` — ok | `tests/test_m00_run_plan.py:70` — `self.assertEqual({r['hash_status'] for r in rows}, {'skipped_size','unreadable','symlink','missing'})`; linhas 71–73 exigem SHA desconhecido, ausência de `keep.txt` no inventário do link e conteúdo original `b'12345'`. | PASS |
| C4 | Rodadas distintas preservam fontes | `test_exclusive_rounds` — ok | `tests/test_m00_run_plan.py:77` — `self.assertNotEqual(first['stages'][0]['output_dir'],second['stages'][0]['output_dir'])`; linha 78 — `self.assertEqual((self.root/'reference/keep.txt').read_text(),'historical')`. | PASS |
| C5 | Colisão/concorrência/traversal recusados | `test_reservation_rejection` — ok | `tests/test_m00_run_plan.py:89` — `self.assertEqual(sum(p is not None for p in pool.map(lambda _:reserve(),range(2))),1)`; linhas 91–95 exigem ValueError nos quatro casos enumerados, FileExistsError na colisão e sentinela `'keep'`. | PASS |
| C6 | Identidade/configuração/modo/recursos persistidos | `test_plan_identity` — ok | `tests/test_m00_run_plan.py:101` — `self.assertEqual(plan['protocol_id'],'m00_run_plan_v1')`; linhas 102–109 exigem versão 1, seed 42, modo não smoke, população/split unknown, SHA-256 da configuração, campos Git, recursos unknown e parent `round-a`. | PASS |
| C7 | 16 notebooks padrão e skips 03/04 | `test_default_stages` — ok | `tests/test_m00_run_plan.py:114` — `self.assertEqual(set(stages),{f'short/{i:02}' for i in range(1,12)}\|{'short/06b'}\|{f'long/{i:02}' for i in range(1,5)})`; linha 116 exige action skip e motivo contendo checkpoint para 03/04. | PASS |
| C8 | Manifestos das quatro cadeias fixados | `test_dependency_paths` — ok | `tests/test_m00_run_plan.py:124` — `self.assertEqual(dep['manifest'],stages[source]['manifest'])`, sobre cinco arestas explicitamente enumeradas na linha 122; linhas 125–128 exigem planned nessas arestas e external/C03 com nome `preparation_manifest.json`. | PASS |
| C9 | Quatro configurações inválidas retornam 2 sem plano | `test_invalid_configuration_cli` — ok | `tests/test_m00_run_plan.py:139` — `self.assertEqual(result.returncode,2,result.stderr)`, para seleção ausente/duplicata/ciclo/desconhecida das linhas 132–135; linha 141 — `self.assertFalse(list(self.root.rglob('run_plan.json')))`. | PASS |
| C10 | Manifesto externo por caminho/hash, somente leitura | `test_external_identity` — ok | `tests/test_m00_run_plan.py:152` — `self.assertEqual(dep['sha256'],hashlib.sha256(before).hexdigest())`; linhas 153–155 exigem caminho absoluto, READY e bytes originais preservados. | PASS |
| C11 | Retomada 06/06b com nova reserva e C09 pendente | `test_resume_pending` — ok | `tests/test_m00_run_plan.py:161` — `self.assertEqual(stage['resume']['validation'],'PENDING_C09')`; linhas 162–164 exigem destino novo, BLOCKED e fonte intacta. Porém a única etapa é `short/06` na linha 159: **short/06b não é provado**. | FAIL |
| C12 | Requisitos/esquemas enumerados, inclusive 06b | `test_metadata_requirements` — ok | `tests/test_m00_run_plan.py:173` — `self.assertEqual(report['status'],'BLOCKED')`; linhas 174–176 exigem dois requisitos e diagnóstico de coluna/arquivo ausente em 06b. Não cobre XLSX inválido: o defeito confirmado abaixo aborta a enumeração com BadZipFile. | FAIL |
| C13 | Dependência futura/incompleta/sintética/adulterada bloqueada | `test_dependency_gate` — ok | `tests/test_m00_run_plan.py:179` — `self.assertEqual(inspect_run_stage(planned,'short/08')['status'],'BLOCKED')`; linha 183 exige BLOCKED nos três payloads enumerados e linha 187 exige BLOCKED mais detalhe contendo hash após adulteração. | PASS |
| C14 | Inspeção empírica sem carga científica | `test_metadata_only` — ok | `tests/test_m00_run_plan.py:192` — patches de `pandas.read_parquet`, `torch.load` e `ensure_smoke_test_artifacts` levantam AssertionError se chamados; linha 196 exige READY sob esses bloqueios, linhas 197–198 exigem nenhum `.pt` e apenas o Parquet original. Inspeção complementar de `run_planning.py:204` e `run_planning.py:315` não encontrou executor de notebooks ou treino; subprocessos são apenas Git nas linhas 375–378. | PASS |
| C15 | Falha administrativa não publica plano pronto parcial | `test_atomic_failure` — ok | `tests/test_m00_run_plan.py:201` — patch de `os.replace` lança OSError; linha 202 exige a exceção, linha 203 exige ausência de `run_plan.json`, linhas 205–206 exigem um marcador FAILED. | PASS |
| C16 | Skip sem produto científico | `test_skip_no_products` — ok | `tests/test_m00_run_plan.py:211` — `self.assertEqual(inspect_run_stage(plan,'short/03')['status'],'SKIPPED')`; linhas 212–213 exigem inexistência do output_dir e manifesto científico. | PASS |
| C17 | CLI com/sem plano conserva opções e códigos 0/2 | `test_preflight_cli` — ok | `tests/test_m00_run_plan.py:219` — READY/0 **com plano**; linha 220 — BLOCKED/2 com plano; linha 222 — `self.assertEqual(legacy.returncode,2)` e smoke verdadeiro **sem plano**. Falta caminho legado READY/0, portanto a conservação dos dois códigos sem plano não está demonstrada. | FAIL |
| C18 | Comandos e limites local/remoto/empírico documentados | `test_documented_limits` — ok | `tests/test_m00_run_plan.py:227` — `self.assertIn(literal,text)` para sete literais da linha 226. Leitura direta de `README.md:217` e `README.md:220` confirma comandos completos; `README.md:249` distingue testes, preflight remoto e reprodução empírica. | PASS |

## Lacunas e defeitos, em ordem de impacto

1. **C12 — XLSX inválido causa exceção fora do relatório.** `src/viral_cnn/run_planning.py:245` abre `ZipFile(path)`, mas o except da linha 257 não captura `zipfile.BadZipFile`; o except da CLI (`scripts/preflight_notebook.py:26`) tampouco. Uma entrada existente de formato incompatível pode abortar a inspeção dos demais requisitos e escapar sem JSON BLOCKED/código 2. Confirmação adicional somente leitura: chamar `_inspect_requirement({'path': str(Path('README.md').resolve()), 'kind': 'xlsx'})` nesta árvore produziu `zipfile.BadZipFile: File is not a zip file`. Não foi alterado código nem arquivo de entrada. Acrescentar prova de arquivo XLSX inválido que retorna BLOCKED e permite enumerar os requisitos restantes.
2. **C11 — limite de amostragem incompatível com o claim.** 1 de 2 IDs de retomada exercitado. Embora a implementação compartilhe o ramo (`src/viral_cnn/run_planning.py:160` e `src/viral_cnn/run_planning.py:362`), uma regressão específica de 06b passaria pela prova atual. Exercitar separadamente ambos os IDs com os requisitos de dependência de 06b satisfeitos.
3. **C17 — nível correto, mas matriz de CLI incompleta.** As provas cruzam a fronteira de processo; falta o par sem plano/READY/0. O teste existente de C00 não oferece essa evidência: a busca `rg -n 'test_.*preflight' tests/test_c00_preflight_manifest.py` localizou apenas a prova de entrada ausente. Acrescentar cenário legado pronto preservando `--data-root` e `--smoke-test`.

## Fontes consultadas e alcance light

Foram lidos o plano, checks, instruções AGENTS da raiz e de src, README, manifesto de dados, STATE/AD-001–028, contratos comuns/M00 do roadmap, configuração M00, configurações default_species/models, runtime/preflight e fontes pertinentes dos notebooks 05/07/08. O roadmap exige planejamento e preflight sem recalcular resultados; o recorte M00 adia migração dos produtores e validadores científicos especializados. Não se encontrou contradição adicional entre esse recorte e os checks.

A cadeia preparada é entradas declaradas → inventário → plano por etapa → inspeção de metadados/dependências. Split/curadoria/codificação/checkpoints/inferência/métricas/tabelas continuam sob os produtores e contratos científicos existentes. `m00_run_plan_v1`, schema 1, inventário SHA-256 limitado a 16 MiB e estados planned/external foram inspecionados. O plano conserva a distinção entre manifesto administrativo e evidência científica.

Não houve recomputação integral do join Coverage, julgamento formal de Test policy nem mutação da implementação: esses passos são de standard/ui. `checks.md` não contém Test policy. A checagem de fontes aqui serve a julgar os claims locais; não constitui auditoria ui.

## Swept existing

Não há linhas Swept explicitamente classificadas como `existing` em checks.md: as nove dimensões apontam C1–C18 ou n/a. Foram relidos os pontos existentes realmente reutilizados: `src/viral_cnn/notebook_runtime.py:78` (topologia), `src/viral_cnn/notebook_runtime.py:260` (registro legado), `src/viral_cnn/notebook_runtime.py:338` (preflight legado), `src/viral_cnn/notebook_runtime.py:374` (reserva) e a montagem CLI em `scripts/preflight_notebook.py:16`. A topologia e registro legado não foram substituídos pela M00; a reserva acrescenta validação de IDs/traversal e conserva criação exclusiva. O preflight legado permanece acessível, com a lacuna C17 registrada acima. Não foram assumidos como passados os 60 testes de regressão ou os 31 de integração relatados pelo autor; esta rodada executou independentemente os 22 M00.

## Limites científicos e de amostragem

Validação **sintética local**, CPU/offline, REMOTE_SERVER=False. Dados e checkpoints canônicos do cluster não estão disponíveis; **preflight remoto não feito** e **reprodução empírica não feita**. Nenhum valor do artigo foi recalculado. Os testes amostram arquivos pequenos, quatro formas de erro DAG, concorrência em duas threads e uma falha na primeira substituição atômica; não comprovam todos os pontos de falha de I/O, paralelismo entre processos ou integridade de produtos empíricos. A prova C14 intercepta três primitivas concretas e usa inspeção estática complementar para ausência de execução, sem certificação dinâmica de toda possível biblioteca de leitura.

Divergências preservadas: `configs/default_species.yaml:16` e linha 19 declaram dropout 0,3; `configs/models.yaml:11` e outras variantes declaram 0,5/0,7. O notebook 05 ainda usa caminho remoto fixo (`notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb:116`), enquanto a topologia existente aponta `short/ICTV_Master_Species_List_2025_MSL41.v1.xlsx`; a parametrização dos produtores fica em M01. Manifestos 07/08 históricos seguem sem promoção automática a evidência C06. Essas diferenças não foram corrigidas pelo verificador nem confundidas com reprodução do artigo.

## Completion gate

Comando: `python3 .agent/skills/tlc-spec-lean/scripts/validate_verification.py m00-reproduction-run-plan`.

Executado: exit **1**, um erro (`verdict is FAIL - route the ranked gaps back as fixes, then re-verify`), zero avisos. A rejeição corresponde ao verdict e aos três checks não aprovados, sem problema de resolução da feature.

O verdict FAIL requer correção e reverificação; 15/18 claims possuem a evidência exigida nesta rodada. Lições candidatas para registro pelo orquestrador após o reparo: testar cada ID nominalmente prometido; cobrir ambos os códigos na compatibilidade da CLI; converter erros de contêiner XLSX em diagnóstico por requisito. Somente este relatório foi escrito pelo verificador.
