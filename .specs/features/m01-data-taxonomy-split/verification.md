# M01 verification

**Verdict**: PASS
**Profile**: light
**Diff range**: fd39616a1b80e70199150d66c50444856ffe4e76..c6821add12862e520247638b0e942ed0f4009e7c
**Round**: 2 - scoped
**Verifier**: independent sub-agent (author != verifier)

Todos os 23 checks estão comprovados. As sete lacunas de alcance da rodada 1 foram resolvidas por ampliação das provas, sem mudança de implementação ou de claims. Relatório anterior preservado em `verification-round1.md`.

**Fix diff**: `21dfc8488e02efae6d120624184a2ead71d406e0..c6821add12862e520247638b0e942ed0f4009e7c`. Verificado neste HEAD: testes M01 e registros da revisão. A comparação Git confirmou diff vazio em `src/`, notebooks, configs e requirements; nenhuma asserção foi enfraquecida. Checks só recebeu notas de revisão no Handoff; claims, seletores, perfil e Coverage não mudaram. Somente este relatório foi editado pelo Verifier.

## Scope and sources

**Carried from 21dfc84** — fontes e comparação do diff original da rodada 1; fontes de implementação não mudaram no reparo.

Lidos: plano e checks completos; skill `tlc-spec-lean` e `references/verify.md`; AGENTS da raiz e de `src/viral_cnn`; README; manifesto de dados; roadmap (contratos comuns e M01); STATE (ADs ativos, inclusive AD-028/029); configurações M00/M01. Revisado o diff completo dos 18 arquivos, incluindo notebooks, configuração, dependências e mudanças N04. No histórico `short/ICTV_improve_kraken_database.ipynb`, inspecionadas as células relevantes de CSV, XLSX, consolidação e saneamento, sem execução.

As identidades CSV foram recalculadas: Species, 13.683 linhas, SHA-256 `51002670cf805a448c01b6d2c65c3dfb4038e020277f238c0a4719155c216517`; Genus, 2.702 linhas, SHA-256 `14e0859f43dc29401f2aca957d615c910801037d61df7ec9b88060fbbe56fb8d`. Correspondem às fontes do plano. O histórico preenche ranks ausentes; o novo fluxo estrito deliberadamente os reporta. A auditoria de referência não demonstra reconstrução MSL nem identifica accession por fragmento.

Perfil light: não houve recomputação da tabela Coverage, comparação UI de fontes vinculantes nem fault injection. Não há Test policy no artefato aprovado. Não há interface humana a validar. Isso não certifica resistência a uma implementação errada que os testes não distingam.

## Proof runs

**Verified at c6821ad** — todas as provas nomeadas reexecutadas, em uma invocação:

```bash
REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_m01_data_taxonomy_split
```

Exit 0; **27 testes em 5,763 s**, todos `ok`, sem skips. Os 23 seletores da tabela apareceram individualmente como executados. `rg -n -A 55 '^    def test_' tests/test_m01_data_taxonomy_split.py` confirmou existência e localizou as asserções atuais. Os quatro adicionais também passaram: `test_real_m00_chain`, `test_invalid_fragment_label_and_shared_file`, `test_iupac_orientation_and_large_identity`, `test_synthetic_reference_is_not_empirical`. O aviso de driver CUDA veio da detecção do runtime; nenhum treinamento foi iniciado.

**Carried from 21dfc84** — as **57 regressões** da rodada 1 passaram independentemente em 41,992 s: 51 dos módulos M00/C00/runtime/taxonomy/grouped_split/dataset_audit, cinco N04 afetadas e uma N09 SQLite. Comando exato e resultados estão em `verification-round1.md`, seção Proof runs, R2. Não foram reexecutadas nesta rodada: o reparo altera somente testes M01 e registros. Essa decisão não carrega os resultados M01; todos os 27 foram reexecutados no novo HEAD. Todas as execuções forçaram `REMOTE_SERVER=False`. O total de evidência continua 84 testes, sendo 27 atuais e 57 carried. `git diff --check 21dfc84..c6821ad` passou.

## Checks

**Verified at c6821ad** — todas as provas e citações atualizadas neste HEAD. Para os 16 checks já aprovados, o julgamento de alcance é **carried from 21dfc84**, com reexecução atual e atualização das linhas movidas. C1/C2/C5/C6/C13/C19/C22 receberam revisão de alcance nova. `test_*` abrevia `tests.test_m01_data_taxonomy_split.M01Tests.test_*`; cada seletor executou individualmente com `ok` na invocação acima.

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | Precedência e registro dos caminhos em 10/11/05 | `test_paths` ok | `tests/test_m01_data_taxonomy_split.py:98` — `self.assertEqual(manifest['configuration']['original_mapping'],str(alternate))`, com taxonomy_path/ictv_master nas linhas 99–100 para os três notebooks executados; linha 104 — `self.assertIn(path,log.getvalue())`; linhas 111–113 fixam defaults; linhas 127–130 fixam fontes do plano e rejeitam override. F1 resolvido. | PASS |
| C2 | Duas execuções isoladas preservam fontes e históricos | `test_isolation` ok | `tests/test_m01_data_taxonomy_split.py:141` — `self.assertNotEqual(a['output_dir'],b['output_dir'])`; linha 142 — `self.assertEqual(before,{str(p):p.read_bytes() for p in sources})`, repetido para 10/11/05, com sentinela histórica explícita em :133–135 e snapshot em :136. F2 resolvido. | PASS |
| C3 | Adoção da reserva e IDs M00 | `test_plan_reservation`, `test_real_m00_chain` ok | `tests/test_m01_data_taxonomy_split.py:157` — `self.assertEqual(m['run_id'],'child');self.assertEqual(m['parent_run_id'],'parent')`; :158 verifica target, :159 rejeita repetição; :396 verifica parent/COMPLETE nas três etapas. | PASS |
| C4 | Colisão, override e ausência de entrada não publicam COMPLETE | `test_rejection`, `test_plan_reservation` ok | `tests/test_m01_data_taxonomy_split.py:165` — `self.assertTrue(all(json.loads(p.read_text())['status']=='FAILED' for p in failed))`; :168 exige ValueError para conflito; :159 exige FileExistsError na repetição. | PASS |
| C5 | Manifestos individuais completos e identidades reais | `test_manifests` ok | `tests/test_m01_data_taxonomy_split.py:186` — `self.assertEqual(m['protocol_id'],protocols[stage])`, valores literais em :172–173; :189 fixa modo; :190 versão pandas; :191 igualdade manifesto persistido; :195 compara conjunto explícito de produtos (:174–176); :199 — `self.assertEqual(record['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())` para entradas/produtos, além de tamanho e fontes em :200–203. F3 resolvido. | PASS |
| C6 | N por split/classe, qualidade e comprimento | `test_split_counts` ok, notebook 10 | `tests/test_m01_data_taxonomy_split.py:210` — `self.assertEqual({r['split']:int(r['rows']) for r in rows},{'train':3,'validation':3,'test':6})`; :214–215 fixam matriz literal classe/split (2/1, 1/2, 3/3); :216–217 fixam qualidade por split; :220–221 fixam distribuição integral de comprimentos, incluindo vazio e 513 bp. F4 resolvido. | PASS |
| C7 | Direto, RC, interseção e união | `test_orientations` ok | `tests/test_m01_data_taxonomy_split.py:225` — `self.assertEqual((d['direct_matches'],d['reverse_complement_matches'],d['both_matches'],d['exact_test_matches_train']),(2,2,1,3))`. | PASS |
| C8 | Denominador integral e subconjunto separado | `test_denominator` ok | `tests/test_m01_data_taxonomy_split.py:229` — `self.assertEqual(d['test_rows'],6);self.assertEqual(d['exact_overlap_pct'],50.)`; :230–231 fixam 4 e 75% no subconjunto compatível. | PASS |
| C9 | PROVENANCE_REQUIRED sem IDs genômicos | `test_provenance` ok | `tests/test_m01_data_taxonomy_split.py:234` — `self.assertEqual(self.overlap()['metrics_summary']['status'],'PROVENANCE_REQUIRED')`. | PASS |
| C10 | Piloto inconclusivo com limite explícito | `test_pilot` ok | `tests/test_m01_data_taxonomy_split.py:238` — `self.assertEqual(d['status'],'INCONCLUSIVE_PARTIAL_TEST');self.assertEqual(d['test_rows'],2)`; :239 fixa `first_2_test_rows_pilot`. | PASS |
| C11 | População distinta de queries C03 | `test_population` ok | `tests/test_m01_data_taxonomy_split.py:243` — `self.assertEqual(m['metrics_summary']['population'],'fragment_test_vs_train_not_c03_queries')`; :242 fixa protocolo. | PASS |
| C12 | IDs, offset e nomes originais inválidos detectados | `test_mapping` ok | `tests/test_m01_data_taxonomy_split.py:250` — `self.assertFalse(report['mapping_integrity_ok'])` sobre casos explícitos de :246; :251 testa ID ausente; :253 testa offset pelo workflow 11. | PASS |
| C13 | Dependência rejeita identidade, piloto e totais incompatíveis | `test_dependency` ok | `tests/test_m01_data_taxonomy_split.py:259` — `with self.assertRaises(ValueError):self.run_stage('short/11',overlap_manifest=str(path))`; :261 cobre piloto; :275 — `with self.subTest(field=field),self.assertRaisesRegex(ValueError,message)` exige `test total`/`match total` literais de :266 após atualizar SHA-256/tamanho em :272–273, alcançando o guard interno de totais. F5 resolvido. | PASS |
| C14 | Ranks não resolvidos são reportados em 11/05 | `test_ranks` ok | `tests/test_m01_data_taxonomy_split.py:283` — `self.assertFalse(m['metrics_summary']['hierarchy_ready'])`; :284 exige label 0 em unresolved_ranks; :286–287 negam readiness/Parquet no 05 para Genus/Order e quatro valores inválidos. | PASS |
| C15 | Estratos conservam N e rank não resolvido | `test_strata` ok, notebook 11 | `tests/test_m01_data_taxonomy_split.py:295` — `self.assertEqual(grouped['test_rows'].sum(),6)`; :296 fixa 3 matches em genus/family/order; :298 exige `[unresolved]`. | PASS |
| C16 | audit_existing é distinto e somente leitura | `test_existing` ok | `tests/test_m01_data_taxonomy_split.py:303` — `self.assertEqual(m['protocol_id'],'m01_existing_taxonomy_audit_v1')`; :304 preserva bytes/nega Parquet novo; :305 nega reconstructed_from_msl. | PASS |
| C17 | Workbook ausente falha no notebook, sem sintético | `test_missing_workbook`, `test_rejection` ok | `tests/test_m01_data_taxonomy_split.py:309` — `with self.assertRaises(FileNotFoundError):self.execute_notebook('05_',dict(self.settings,ictv_master='missing.xlsx'))`; :308 proíbe geração sintética; :310 nega Parquet; :311 afirma status FAILED. | PASS |
| C18 | Workbook/CSV inválido não publica Parquet | `test_invalid_workbook` ok | `tests/test_m01_data_taxonomy_split.py:322` — `with self.subTest(case=case),self.assertRaises(ValueError):self.run_stage('short/05',ictv_master=str(p))`; casos explícitos em :314–321; :323 nega Parquet; :324 afirma FAILED. | PASS |
| C19 | Curate publica identidade válida e fontes ICTV/NCBI | `test_curate` ok | `tests/test_m01_data_taxonomy_split.py:328` — `self.assertEqual(m['protocol_id'],'m01_ictv_curate_v1')`; :331 fixa labels/nomes; :343–345 fixam Genus `['G1','LocalGenus']`, Order `['O1','LocalOrder']` e fontes `['ICTV Master','NCBI Fallback']` após fallback local; :348 exige CSV/XLSX/NCBI entre entradas; :349 compara SHA-256 real. F6 resolvido. | PASS |
| C20 | Smoke sintético e dependência 11 preservada | `test_synthetic`, `test_synthetic_reference_is_not_empirical` ok | `tests/test_m01_data_taxonomy_split.py:353` — `self.assertTrue(m['is_smoke_test']);self.assertEqual(m['evidence']['status'],'synthetic')`; :358 exige flag sintética no 11/dependência; :356/:416–418 rejeitam promoção empírica. | PASS |
| C21 | Proveniência genômica desconhecida nos dois modos | `test_unknown_genome` ok | `tests/test_m01_data_taxonomy_split.py:363` — `self.assertEqual(m['metrics_summary']['genome_provenance'],'unknown')` em audit_existing/curate. | PASS |
| C22 | LOG_PATH em carregamentos M01 e desativação | `test_logging` ok | `tests/test_m01_data_taxonomy_split.py:381` — `self.assertIn(path,text)` para CSV, três splits, taxonomia, manifesto, decisão, per_species, XLSX e NCBI listados em :379–380; :382 — `self.assertNotIn('[LOG_PATH]',text)` em 0/false; :368/:371 exercitam default sem variável; :373–376 executam 10/11/05 curate/audit_existing. F7 resolvido. | PASS |
| C23 | Comandos, manifestos e limites documentados | `test_documentation` ok; leitura carried | `tests/test_m01_data_taxonomy_split.py:423` — `self.assertIn(value,text)` com lista explícita de :422. `docs/m01_data_taxonomy_split.md:13` a :20 fornece comandos 10→11→05; :23 descreve manifestos; :59–67 separa local/preflight/empírico pendentes. | PASS |

## Level and sampling gaps

**Verified at c6821ad** — F1–F7 encerrados pelas asserções localizadas acima. O reparo alcança os três notebooks para configuração/ambiente, os três workflows no plano e os defaults do resolver; observa sentinela histórica, valores/identidades reais, matriz split/classe/comprimento, inconsistência de totais com hashes válidos, aplicação do NCBI local e logs dos três fluxos. Nenhuma lacuna de nível/amostragem permanece nas claims aprovadas sob light.

O ensaio usa fixtures pequenas e não amostra a população científica integral. A camada Jupyter de execução via nbconvert e a topologia real do cluster não foram revalidadas; a prova de notebook executa suas células Python no processo local, e a adoção M00 adicional usa os workflows compartilhados. Esses limites não equivalem a preflight remoto ou reprodução empírica e continuam explícitos. Não houve fault injection/recomputação de Coverage, conforme perfil light; não se afirma sensibilidade a todas as implementações erradas possíveis.

## Swept existing

**Carried from 21dfc84** — implementação e Swept sem mudanças, confirmado pelo diff vazio nos módulos.

Não há linhas Swept marcadas literalmente `existing`; as nove dimensões apontam para checks ou n/a. Relidas as restrições reutilizadas relevantes: reserva M00 valida namespace/manifesto/parent e diretório vazio em `src/viral_cnn/m01_workflow.py:175` e :179; claim de arquivo exclusivo em :182; reserva standalone em :185. Identidade dos produtos predecessores é verificada antes do consumo; escopo piloto é rejeitado; COMPLETE é publicado após identidades por temporário/rename. Os guards existem. As lacunas de prova da primeira rodada foram encerradas por testes atuais; a inspeção carregada não substitui essas execuções. Autorização é n/a aprovado: execução local sem serviço; não foi inventada uma política adicional.

## Scientific limits and preserved artifacts

**Carried from 21dfc84** — fontes históricas e limites científicos sem mudanças; reafirmados no HEAD atual.

As fixtures temporárias e a planilha mínima XLSX verificam contratos locais. Não houve preflight remoto, varredura dos fragmentos canônicos, validação da MSL41.v1 real, reconstrução do híbrido canônico ou reprodução de qualquer número do artigo. Planilha e híbrido canônicos continuam indisponíveis localmente segundo o inventário do plano; identidade dos Parquets locais não foi demonstrada. Checkpoints não eram necessários e não foram carregados. Accuracy, Top-5, F1, embeddings e tabelas/figuras do artigo não foram recalculados. A população continua closed-set por fragmentos; taxonomia válida não resolve independência genômica.

O diff versionado não modifica notebooks históricos, CSVs de classes ou checkpoints. Os testes N04 da rodada 1 observaram namespaces temporários novos. Alteração feita pelo Verifier nesta revisão: somente este relatório. As divergências históricas de preenchimento de ranks, offset silencioso e fallback sintético estão documentadas no plano e na documentação M01; diferenças de títulos/dropout permanecem fora do escopo.

**Incidente anterior, informado após R1:** o autor relatou que, no início do BUILD, executou o notebook 05 ainda legado em smoke na raiz. Essa execução chamou `ensure_smoke_test_artifacts` e sobrescreveu sete Parquets ignorados: `cnn_kraken/processed_kraken_species_splited_parquet/{train,test,val}.parquet`, `cnn_kraken/processed_kraken_genus_splited_parquet/{train,test,val}.parquet` e `processed_ncbi_splited_parquet/test.parquet`. Não houve hash anterior; o conteúdo anterior e sua recuperação não são demonstráveis nesta verificação. O Verifier não observou a sobrescrita ocorrer; leu o relato e seu registro acrescentado ao final de `docs/m01_data_taxonomy_split.md` (alteração documental ainda não commitada, posterior a c6821ad). Os CSVs de classes têm hashes confirmados na rodada 1. Esses Parquets devem ser tratados como fixtures sintéticas, sem valor empírico, até nova obtenção/seleção de fontes verificadas.

Consequentemente, não se afirma preservação global da árvore ignorada durante todo o desenvolvimento. O PASS de C2 demonstra invariância do fluxo M01 atual nas execuções isoladas com sentinelas; não certifica a operação do notebook legado que antecedeu o reparo. Essa ressalva corrige o alcance da expressão “saídas antigas preservadas” da rodada 1 sem alterar o relatório histórico arquivado. O autor também acrescentou registros de lição durante esta rodada; não houve alteração de implementação ou testes após o HEAD testado.

## Gate

**Verified at c6821ad** — `python3 .agent/skills/tlc-spec-lean/scripts/validate_verification.py m01-data-taxonomy-split`: exit **0**, **0 erros, 0 avisos**. O gate só avalia a conclusão local desta feature, não a disponibilidade/validação empírica no cluster. HEAD c6821ad; mudanças concorrentes posteriores foram apenas o registro documental do incidente e lições, discriminados acima.
