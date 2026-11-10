# M01 verification

**Verdict**: FAIL
**Profile**: light
**Diff range**: fd39616a1b80e70199150d66c50444856ffe4e76..21dfc8488e02efae6d120624184a2ead71d406e0
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

Todos os 23 testes nomeados existem, foram executados no HEAD acima e passaram. Dezesseis claims estão demonstradas; sete têm prova parcial. FAIL refere-se ao alcance das provas, não a uma falha de execução observada. Nenhuma implementação, teste, obrigação aprovada ou artefato histórico foi alterado pelo Verifier.

## Scope and sources

Lidos: plano e checks completos; skill `tlc-spec-lean` e `references/verify.md`; AGENTS da raiz e de `src/viral_cnn`; README; manifesto de dados; roadmap (contratos comuns e M01); STATE (ADs ativos, inclusive AD-028/029); configurações M00/M01. Revisado o diff completo dos 18 arquivos, incluindo notebooks, configuração, dependências e mudanças N04. No histórico `short/ICTV_improve_kraken_database.ipynb`, inspecionadas as células relevantes de CSV, XLSX, consolidação e saneamento, sem execução.

As identidades CSV foram recalculadas: Species, 13.683 linhas, SHA-256 `51002670cf805a448c01b6d2c65c3dfb4038e020277f238c0a4719155c216517`; Genus, 2.702 linhas, SHA-256 `14e0859f43dc29401f2aca957d615c910801037d61df7ec9b88060fbbe56fb8d`. Correspondem às fontes do plano. O histórico preenche ranks ausentes; o novo fluxo estrito deliberadamente os reporta. A auditoria de referência não demonstra reconstrução MSL nem identifica accession por fragmento.

Perfil light: não houve recomputação da tabela Coverage, comparação UI de fontes vinculantes nem fault injection. Não há Test policy no artefato aprovado. Não há interface humana a validar. Isso não certifica resistência a uma implementação errada que os testes não distingam.

## Proof runs

R1, no HEAD verificado:

```bash
REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_m01_data_taxonomy_split
```

Exit 0; **27 testes em 2,873 s**, todos `ok`. O runner imprimiu individualmente todos os seletores nomeados abaixo, sem skips. `rg -n -A 35 '^    def test_' tests/test_m01_data_taxonomy_split.py` localizou seus corpos e asserções; todos pertencem a arquivo novo no diff. Os quatro adicionais também passaram: `test_real_m00_chain`, `test_invalid_fragment_label_and_shared_file`, `test_iupac_orientation_and_large_identity`, `test_synthetic_reference_is_not_empirical`.

R2, regressões independentes:

```bash
REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_m00_run_plan tests.test_c00_preflight_manifest tests.test_notebook_runtime tests.test_taxonomy tests.test_grouped_split tests.test_dataset_audit tests.test_n04_taxonomy_and_latent_space.TestN04TaxonomyAndLatentSpace.test_c6_notebook_05_execution_smoke_test tests.test_n04_taxonomy_and_latent_space.TestN04TaxonomyAndLatentSpace.test_c7_notebook_05_imports_and_decoupling tests.test_n04_taxonomy_and_latent_space.TestN04TaxonomyAndLatentSpace.test_c8_notebook_05_artifacts_generated tests.test_n04_taxonomy_and_latent_space.TestN04TaxonomyAndLatentSpace.test_c9_historical_curation_notebooks_preserved tests.test_n04_taxonomy_and_latent_space.TestN04TaxonomyAndLatentSpace.test_r04_notebook_05_provenance_and_smoke_execution tests.test_n09_pipeline_optimization.TestN09PipelineOptimization.test_c8_batched_sqlite_retraining_audit
```

Exit 0; **57 testes em 41,992 s**, todos `ok`: 51 regressões dos seis módulos, cinco N04 afetados e uma N09 SQLite. Total independente: **84 testes**. Houve aviso de driver CUDA durante a detecção do runtime e cache Matplotlib temporário; nenhum teste falhou, nenhum treinamento foi iniciado. Todos os comandos forçaram `REMOTE_SERVER=False`. `git diff --check` do intervalo também passou.

## Checks

Nesta tabela, `test_*` abrevia `tests.test_m01_data_taxonomy_split.M01Tests.test_*`; todos foram executados em R1 com resultado individual `ok`. Evidência parcial continua FAIL quando não alcança a claim inteira.

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | Precedência e registro dos caminhos em 10/11/05 | R1 `test_paths` ok | `tests/test_m01_data_taxonomy_split.py:85` — `self.assertEqual(resolved['original_mapping'],str(alternate))`; linha 86 verifica validation. Só short/10, sem registro/assert de caminhos selecionados nos três fluxos. F1. | FAIL |
| C2 | Duas execuções isoladas preservam fontes e históricos | R1 `test_isolation` ok | `tests/test_m01_data_taxonomy_split.py:91` — `self.assertNotEqual(a['output_dir'],b['output_dir'])`; linha 92 preserva bytes CSV/Parquet. Nenhuma sentinela de saída histórica, só 10 repetido. F2. | FAIL |
| C3 | Adoção da reserva e IDs M00 | R1 `test_plan_reservation` e `test_real_m00_chain` ok | `tests/test_m01_data_taxonomy_split.py:107` — `self.assertEqual(m['run_id'],'child');self.assertEqual(m['parent_run_id'],'parent')`; linha 108 verifica target; linha 109 rejeita repetição. Linha 279 verifica parent/COMPLETE para as três etapas do plano real. | PASS |
| C4 | Colisão, override e ausência de entrada não publicam COMPLETE | R1 `test_rejection`, `test_plan_reservation` ok | `tests/test_m01_data_taxonomy_split.py:115` — `self.assertTrue(all(json.loads(p.read_text())['status']=='FAILED' for p in failed))`; linha 119 exige ValueError para conflito; linha 109 exige FileExistsError na repetição. | PASS |
| C5 | Manifestos individuais completos e identidades reais | R1 `test_manifests` ok | `tests/test_m01_data_taxonomy_split.py:126` — `self.assertEqual(m['status'],'COMPLETE');self.assertEqual(m['notebook_id'],stage)`; linhas 127–132 só verificam presença de chaves/hash não unknown. Modo, valores e reconciliação com arquivos não são afirmados. F3. | FAIL |
| C6 | N por split/classe, qualidade e comprimento | R1 `test_split_counts` ok, executa notebook 10 | `tests/test_m01_data_taxonomy_split.py:139` — `self.assertEqual({r['split']:int(r['rows']) for r in rows},{'train':3,'validation':3,'test':6})`; linha 142 verifica somente `cls['rows'].sum()==12`, sem matriz split/classe; linhas 141/144 verificam vazio e 513 bp em validation. F4. | FAIL |
| C7 | Direto, RC, interseção e união | R1 `test_orientations` ok | `tests/test_m01_data_taxonomy_split.py:148` — `self.assertEqual((d['direct_matches'],d['reverse_complement_matches'],d['both_matches'],d['exact_test_matches_train']),(2,2,1,3))`. | PASS |
| C8 | Denominador integral e subconjunto separado | R1 `test_denominator` ok | `tests/test_m01_data_taxonomy_split.py:152` — `self.assertEqual(d['test_rows'],6);self.assertEqual(d['exact_overlap_pct'],50.)`; linhas 153–154 fixam 4 e 75% no subconjunto compatível. | PASS |
| C9 | PROVENANCE_REQUIRED sem IDs genômicos | R1 `test_provenance` ok | `tests/test_m01_data_taxonomy_split.py:157` — `self.assertEqual(self.overlap()['metrics_summary']['status'],'PROVENANCE_REQUIRED')`. | PASS |
| C10 | Piloto inconclusivo com limite explícito | R1 `test_pilot` ok | `tests/test_m01_data_taxonomy_split.py:161` — `self.assertEqual(d['status'],'INCONCLUSIVE_PARTIAL_TEST');self.assertEqual(d['test_rows'],2)`; linha 162 fixa `first_2_test_rows_pilot`. | PASS |
| C11 | População distinta de queries C03 | R1 `test_population` ok | `tests/test_m01_data_taxonomy_split.py:166` — `self.assertEqual(m['metrics_summary']['population'],'fragment_test_vs_train_not_c03_queries')`; linha 165 fixa o protocolo. | PASS |
| C12 | IDs, offset e nomes originais inválidos detectados | R1 `test_mapping` ok | `tests/test_m01_data_taxonomy_split.py:173` — `self.assertFalse(report['mapping_integrity_ok'])` sobre os casos explícitos da linha 169; linha 174 testa ID ausente; linha 176 testa offset pelo workflow 11. | PASS |
| C13 | Dependência rejeita identidade, piloto e totais incompatíveis | R1 `test_dependency` ok | `tests/test_m01_data_taxonomy_split.py:182` — `with self.assertRaises(ValueError):self.run_stage('short/11',overlap_manifest=str(path))`; linha 184 cobre piloto. Linhas 185–187 adulteram CSV sem atualizar hash: não provam validação de totais após identidade válida. F5. | FAIL |
| C14 | Ranks não resolvidos são reportados em 11/05 | R1 `test_ranks` ok | `tests/test_m01_data_taxonomy_split.py:194` — `self.assertFalse(m['metrics_summary']['hierarchy_ready'])`; linha 195 exige label 0 em unresolved_ranks; linhas 197–198 negam readiness/Parquet no 05 para Genus/Order e quatro valores inválidos. | PASS |
| C15 | Estratos conservam N e rank não resolvido | R1 `test_strata` ok, executa notebook 11 | `tests/test_m01_data_taxonomy_split.py:206` — `self.assertEqual(grouped['test_rows'].sum(),6)` e linha 207 fixa 3 matches para genus/family/order; linha 209 exige `[unresolved]`. | PASS |
| C16 | audit_existing é distinto e somente leitura | R1 `test_existing` ok | `tests/test_m01_data_taxonomy_split.py:214` — `self.assertEqual(m['protocol_id'],'m01_existing_taxonomy_audit_v1')`; linha 215 preserva bytes/nega Parquet novo; linha 216 nega reconstructed_from_msl. | PASS |
| C17 | Workbook ausente falha no notebook, sem sintético | R1 `test_missing_workbook`, `test_rejection` ok | `tests/test_m01_data_taxonomy_split.py:220` — `with self.assertRaises(FileNotFoundError):self.execute_notebook('05_',dict(self.settings,ictv_master='missing.xlsx'))`; linha 219 proíbe geração sintética, linha 221 nega Parquet. C4 acrescenta a asserção FAILED no mesmo ramo compartilhado. | PASS |
| C18 | Workbook/CSV inválido não publica Parquet | R1 `test_invalid_workbook` ok | `tests/test_m01_data_taxonomy_split.py:232` — `with self.subTest(case=case),self.assertRaises(ValueError):self.run_stage('short/05',ictv_master=str(p))`; casos ausente/coluna/IDs nas linhas 224–231; linha 233 nega Parquet. | PASS |
| C19 | Curate publica identidade válida e fontes ICTV/NCBI | R1 `test_curate` ok | `tests/test_m01_data_taxonomy_split.py:237` — `self.assertEqual(m['protocol_id'],'m01_ictv_curate_v1')`; linhas 240–242 verificam labels/nomes/Genus/readiness e caminho XLSX. Nenhum NCBI local fornecido/exercitado nem hash dessa origem. F6. | FAIL |
| C20 | Smoke sintético e dependência 11 preservada | R1 `test_synthetic`, `test_synthetic_reference_is_not_empirical` ok | `tests/test_m01_data_taxonomy_split.py:246` — `self.assertTrue(m['is_smoke_test']);self.assertEqual(m['evidence']['status'],'synthetic')`; linha 251 exige flag sintética no 11 e sua dependência; linhas 249/299–301 rejeitam promoção empírica. | PASS |
| C21 | Proveniência genômica desconhecida nos dois modos | R1 `test_unknown_genome` ok | `tests/test_m01_data_taxonomy_split.py:256` — `self.assertEqual(m['metrics_summary']['genome_provenance'],'unknown')` em audit_existing/curate. | PASS |
| C22 | LOG_PATH em carregamentos M01 e desativação | R1 `test_logging` ok | `tests/test_m01_data_taxonomy_split.py:264` — `for path in [str(self.mapping),*self.paths.values()]:self.assertIn(path,text)`; linha 265 nega LOG_PATH em 0/false. Somente 10; 11/05 não são capturados. F7. | FAIL |
| C23 | Comandos, manifestos e limites documentados | R1 `test_documentation` ok; leitura direta | `tests/test_m01_data_taxonomy_split.py:306` — `self.assertIn(value,text)` com lista explícita da linha 305. Leitura adicional de `docs/m01_data_taxonomy_split.md:13` a :20 confirma comandos 10→11→05; :23 descreve manifestos; :59–67 separa local, preflight e empírico pendentes. | PASS |

## Level and sampling gaps

1. **F1 — C1:** afirmar precedência e caminhos efetivamente registrados por 10/11/05 na entrada de notebook; incluir configuração, ambiente, plano e fallback de topologia. O teste atual chama somente o resolver para 10, portanto há lacuna de nível e de amostragem. A cadeia M00 adicional passa, mas só afirma parent/status e integridade, não a seleção de caminhos por precedência.
2. **F2 — C2:** instalar sentinela em saída histórica e reconciliar seus bytes após execuções repetidas dos fluxos, além dos arquivos de referência. Os dois runs atuais são 10; a prova não observa saída histórica alguma.
3. **F3 — C5:** afirmar protocolo/modo/configuração esperados por etapa e comparar registros de entrada/produto com caminhos e hashes recalculados dos arquivos consumidos/publicados. `sha256!='unknown'` sozinho não demonstra identidade real ou associação correta entre arquivo e registro.
4. **F4 — C6:** afirmar a matriz exata `(split, model_label)→rows` e qualidade/comprimentos esperados nos três splits. Somar 12 linhas não detecta contagens transferidas entre classes/splits; a claim explicitamente exige contagens por classe.
5. **F5 — C13:** fornecer decisão/CSV com totais incompatíveis e hashes válidos no manifesto para alcançar o guard de totais. A prova atual de CSV alterado para antes, em `_predecessor`, na identidade; ela demonstra rejeição por identidade, não por inconsistência interna de totais. O guard existe em `src/viral_cnn/data/taxonomy_split_audit.py:98`, mas sua presença não é execução comprovada pelo teste nomeado.
6. **F6 — C19:** exercitar curadoria que realmente necessite do fallback NCBI local fornecido, afirmar ranks/IDs resultantes e caminho/hash desse artefato no manifesto. `test_curate` cobre só correspondência direta ICTV. A condicional NCBI faz parte da claim aprovada.
7. **F7 — C22:** capturar 11 e 05 nos modos pertinentes, incluindo taxonomia, predecessor/decisão/CSV, XLSX e fallback local, com padrão/0/false. Logs de 10 não demonstram observabilidade de carregamentos nas outras análises.

Não foram identificadas falhas de implementação nesses ramos pelas execuções realizadas. Não houve injeção de falha para estimar sensibilidade das provas. Os achados acima exigem ampliar evidência sem enfraquecer claims; não justificam alterar o protocolo científico.

## Swept existing

Não há linhas Swept marcadas literalmente `existing`; as nove dimensões apontam para checks ou n/a. Relidas as restrições reutilizadas relevantes: reserva M00 valida namespace/manifesto/parent e diretório vazio em `src/viral_cnn/m01_workflow.py:175` e :179; claim de arquivo exclusivo em :182; reserva standalone em :185. Identidade dos produtos predecessores é verificada antes do consumo; escopo piloto é rejeitado; COMPLETE é publicado após identidades por temporário/rename. Os guards existem. As lacunas de prova correspondentes estão discriminadas na tabela, sem transformar inspeção de código em teste executado. Autorização é n/a aprovado: execução local sem serviço; não foi inventada uma política adicional.

## Scientific limits and preserved artifacts

As fixtures temporárias e a planilha mínima XLSX verificam contratos locais. Não houve preflight remoto, varredura dos fragmentos canônicos, validação da MSL41.v1 real, reconstrução do híbrido canônico ou reprodução de qualquer número do artigo. Planilha e híbrido canônicos continuam indisponíveis localmente segundo o inventário do plano; identidade dos Parquets locais não foi demonstrada. Checkpoints não eram necessários e não foram carregados. Accuracy, Top-5, F1, embeddings e tabelas/figuras do artigo não foram recalculados. A população continua closed-set por fragmentos; taxonomia válida não resolve independência genômica.

O diff não modifica notebooks históricos, CSVs de classes, checkpoints nem saídas antigas. Os testes N04 observaram namespaces temporários novos. Alteração desta revisão: somente este relatório. As divergências históricas de preenchimento de ranks, offset silencioso e fallback sintético estão documentadas no plano e na documentação M01; diferenças de títulos/dropout permanecem fora do escopo.

## Gate

`python3 .agent/skills/tlc-spec-lean/scripts/validate_verification.py m01-data-taxonomy-split` — exit **1**, **1 erro, 0 avisos**: `verdict is FAIL - route the ranked gaps back as fixes, then re-verify`. Gate bloqueado corretamente. Nenhuma conclusão de done é autorizada nesta rodada. HEAD permaneceu `21dfc8488e02efae6d120624184a2ead71d406e0`; status final mostrou apenas este relatório novo.
