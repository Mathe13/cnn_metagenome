# C09 — Retomada validada: verification

**Verdict**: PASS
**Profile**: light
**Diff range**: 4d4f616ef06069c2d8d8e1cccc838fb3c7ceee60..c26f17934184e56d1ca4766e0dc6244a9f03f7cd
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier), `/root/verify_c09`

18/18 checks com evidência localizada. Todas as provas foram executadas independentemente no HEAD acima; nenhuma conclusão depende apenas do relato do implementador. Não houve edição da implementação, dos checks ou das suas asserções pelo Verifier.

## Proof runs

P executou todos os seletores de `checks.md` em um único processo, com `-v` para identificar cada método:

```bash
REMOTE_SERVER=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_c01_latent_audit tests.test_c07_latent_memory tests.test_c08_knn_workspace tests.test_latent_space tests.test_c09_latent_resume
```

Exit 0: **79 testes passaram**, em 35,863 s; C09 contém 17 testes. Saídas preservadas durante a sessão em `/tmp/c09-independent-proof.stdout` e `/tmp/c09-independent-proof.stderr`. Uma conferência adicional extraiu os seletores do próprio `checks.md` e encontrou os **15 métodos distintos** correspondentes aos 18 checks, cada um presente e concluído com `ok` no log. As definições e corpos foram localizados com `rg -n -A35 'def test_' tests/test_c09_latent_resume.py`, depois conferidos novamente no HEAD final.

Regressão de fronteira do notebook:

```bash
REMOTE_SERVER=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n04_taxonomy_and_latent_space -k notebook_06
```

Exit 0: **9 testes passaram**, em 33,961 s; logs `/tmp/c09-independent-n04.stdout` e `/tmp/c09-independent-n04.stderr`. A bateria N04 completa não foi repetida pelo Verifier: as colisões de manifesto do notebook 05 relatadas pelo implementador ficam fora desta verificação de C09 e não foram confirmadas independentemente aqui.

## Checks

Todos os nomes abaixo pertencem a `tests.test_c09_latent_resume`; `P` é o comando executado acima, não uma substituição por inspeção estática.

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | Notebook inicia em reuse e rejeita modo inválido antes do runtime/extração | P · `ResumeTests.test_modes_and_sources`, ok | `tests/test_c09_latent_resume.py:57` — `self.assertEqual(scope['LATENT_EXECUTION_MODE'], 'reuse')`; `:61` — `assertRaisesRegex(ValueError, 'LATENT_EXECUTION_MODE')`; `:63` — `extract.assert_not_called()`; `:64` — `setup.assert_not_called()` | PASS |
| C2 | Rerun ignora cache e executa cálculo isolado | P · `ResumeTests.test_rerun_and_isolation`, ok | `tests/test_c09_latent_resume.py:85` — `self.assertIsNone(rerun.load_embedding(self.a.manifest, 4))`; `:86` — etapa devolve o novo `{'value': 9}`; nova pasta também verificada no smoke em `:320` | PASS |
| C3 | Recência, origem explícita e erros de origem | P · `ResumeTests.test_modes_and_sources`, ok | `tests/test_c09_latent_resume.py:69` — `self.assertEqual(self.cache().sources, [source, self.old])`; `:70` — origem explícita resulta em `[self.old]`; `:75` — `assertRaises((ValueError, FileNotFoundError))` para os IDs inválidos/ausentes enumerados em `:74` | PASS |
| C4 | Identidade científica, SHA-256 e fingerprint invalidam embeddings | P · `ResumeTests.test_embedding_identity_and_integrity`, ok | `tests/test_c09_latent_resume.py:105` — `self.assertIsNone(self.cache().load_embedding(expected, 4))`, sobre as alterações explícitas em `:98`; `:108` e `:114` recusam bytes e fingerprint adulterados | PASS |
| C5 | Par incompleto recua para origem válida; dimensão/contagem inválidas são recusadas | P · `ResumeTests.test_embedding_fallback_and_shape`, ok | `tests/test_c09_latent_resume.py:124` — `self.assertEqual(result[1], good)`; `:126` — dimensão 5 recusada; `:128` e `:135` — contagem declarada incorreta recusada | PASS |
| C6 | Carregamento válido usa spill sem extração | P · `ResumeTests.test_embedding_budget_and_no_extraction`, ok | `tests/test_c09_latent_resume.py:141` — extrator bloqueado por `AssertionError('extracted')`; `:143` — `self.assertIsInstance(loaded.features, np.memmap)`; `:144` — igualdade dos valores carregados | PASS |
| C7 | Origens permanecem byte a byte intactas | P · `ResumeTests.test_rerun_and_isolation`, ok | `tests/test_c09_latent_resume.py:92` — `self.assertEqual(before, {p: p.read_bytes() for p in self.old.iterdir()})` após rerun e reuso | PASS |
| C8 | Payload e dependências precisam de hashes corretos; falha de publicação não conclui etapa | P · `ResumeTests.test_stage_atomic_and_integrity`, ok | `tests/test_c09_latent_resume.py:157` — dependência corrompida exige novo valor `42`; `:166` — payload corrompido exige `11`; `:162` — `self.assertFalse(list(fresh.glob('stage_*.json')))` após falha de rename | PASS |
| C9 | Dependência alterada recalcula apenas a unidade afetada | P · `ResumeTests.test_stage_dependency_invalidation`, ok | `tests/test_c09_latent_resume.py:173` — KNN alterado retorna `3`; `:174` — UMAP válido retorna `2` com `lambda: self.fail('recomputed')` | PASS |
| C10 | Restauração da sonda preserva seleção/selagem e recusa sobreposição e pesos adulterados | P · `ResumeTests.test_sealed_probe_roundtrip`, ok | `tests/test_c09_latent_resume.py:185` — igualdade do SHA selado; `:186` — igualdade da seleção; `:193` — `assertRaises(ValueError)` para teste sobreposto; `:197` e `:199` recusam pesos alterados e splits trocados | PASS |
| C11 | Inferência nativa coincide com backbone e confere identidades do teste | P · `ResumeTests.test_native_predictions_identity`, ok | `tests/test_c09_latent_resume.py:208` — `np.testing.assert_array_equal(pred, model(torch.from_numpy(test.features)).argmax(1).detach().numpy())`; `:210` — `assertRaises(ValueError)` com ID divergente | PASS |
| C12 | Saídas legadas sem registro são recusadas e justificadas | P · `ResumeTests.test_legacy_stage_not_admitted`, ok | `tests/test_c09_latent_resume.py:216` — cálculo retorna `8`; `:217` — `self.assertTrue(any('legacy' in e.get('reason', '').lower() for e in cache.events))` | PASS |
| C13 | Dois consumidores escrevem em diretórios separados e mantêm origem intacta | P · `ResumeTests.test_rerun_and_isolation`, ok | `tests/test_c09_latent_resume.py:93` — `self.assertNotEqual(other, self.new)`; `:92` — comparação integral dos bytes da origem | PASS |
| C14 | Smoke completo retoma componentes e registra modo/hashes | P · `NotebookResumeTests.test_smoke_resume_and_manifest`, ok | `tests/test_c09_latent_resume.py:312` — cálculos bloqueados por spies; `:313`–`:315` bloqueiam ajuste/avaliação/inferência; `:328` — modo `reuse`; `:331` — exatamente 15 embeddings reutilizados; `:333` — hashes presentes; `:325`–`:326` preservam avisos no CSV UMAP | PASS |
| C15 | Documentação cobre parâmetros, legados e limites sintéticos/remotos | P · `NotebookResumeTests.test_documented_limits`, ok | `tests/test_c09_latent_resume.py:337` — enumeração literal dos parâmetros, modos e limites; `:338` — `self.assertIn(token, doc)`; conteúdo conferido em `docs/latent_resume.md:10` e `docs/latent_resume.md:30` | PASS |
| C16 | Interrupção entre payload/marcador obriga recálculo; erro de permissão propaga | P · `ResumeTests.test_stage_interrupted_manifest_and_operational_failure`, ok | `tests/test_c09_latent_resume.py:230` — nenhum `stage_*.json`; `:232` — recálculo retorna `11`; `:235` — `assertRaises(PermissionError)` | PASS |
| C17 | Origem explícita com referências transitivas restaura resultados; RNG da sonda é preservado | P · `ResumeTests.test_explicit_transitive_reuse_and_probe_initialization`, ok | `tests/test_c09_latent_resume.py:245` — embedding transitivo admitido; `:246` — etapa retorna `21` sem cálculo; `:254` — `torch.testing.assert_close(before, torch.get_rng_state())` | PASS |
| C18 | Inicialização da sonda usa seed antes da construção e identificador próprio | P · `ResumeTests.test_initialization_protocol_is_explicit`, ok | `tests/test_c09_latent_resume.py:262` — `self.assertEqual(ast.unparse(fit.body[0]), 'torch.manual_seed(ctx.seed)')`; `:263` e `:264` — identificador `c09_seeded_probe_init_v1` e prefixo `c09-` | PASS |

## Scientific provenance and existing constraints

Foi relido o plano, os checks, o diff completo, as instruções de agentes da raiz e de `src/viral_cnn`, os contratos AD-017/019/025 em `.specs/STATE.md`, as configurações de checkpoints e a documentação de retomada. A cadeia permanece partições/hashes → checkpoint identificado → embeddings pré-classificador → sonda/estatísticas/UMAP → tabelas e manifesto. Os cinco checkpoints mantêm os caminhos de `configs/models.yaml`; não houve acesso aos arquivos remotos.

`src/viral_cnn/eval/latent.py` não mudou no diff: isso preserva a identidade histórica do extrator e os algoritmos C01/C07/C08. A validação adicional de embeddings está em `src/viral_cnn/latent_resume.py:157`; stages vinculam identidade, payload e arquivos externos antes da restauração em `src/viral_cnn/latent_resume.py:207`, carregando tensores com `weights_only=True` em `:250`. A restauração da sonda reconstrói as identidades train/validation em `:291`; o notebook verifica disjunção das partições antes de executar as etapas. O protocolo de inicialização novo aparece na identidade da sonda e no manifesto, em `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb:389` e `:710`.

O ajuste identificado `c09_seeded_probe_init_v1` pode divergir dos números históricos da sonda: ele fixa o seed antes de construir a cabeça, corrigindo dependência anterior do RNG global. Isso está explicitamente registrado no plano e em `docs/latent_resume.md:20`; não há afirmação de equivalência dos pesos históricos. Backbones, classes e embeddings empíricos permanecem sujeitos às identidades antigas. Os limites mantidos são Silhouette 5.000 e projeção 100; a menção histórica a 50.000 continua documentada como divergência. KNN continua no espaço original com k=5/cosseno/80–20; o split por fragmentos não prova independência genômica.

O `Swept` não contém linhas marcadas `existing` sem prova própria: resolve os contratos pelos checks ou por n/a. Ainda assim, a reserva exclusiva existente foi relida em `src/viral_cnn/notebook_runtime.py:374`, a validação histórica em `src/viral_cnn/eval/latent.py:262` e a proteção do teste selado em `src/viral_cnn/eval/latent.py:1298`; permanecem no fluxo.

## Resolved finding

A primeira revisão identificou uma lacuna de nível em C1: a prova testava apenas o default da classe de cache, embora a obrigação nomeasse o notebook. O implementador acrescentou a avaliação da atribuição real da primeira célula e sua execução com modo inválido, bloqueando runtime/extração. A correção está em `tests/test_c09_latent_resume.py:49` e foi executada no HEAD final. Não houve enfraquecimento ou exclusão de asserções.

O sinal `ac_gap` foi registrado pelo script de lições como **L-006**, candidate, recorrência 1; o script atualizou `.specs/lessons.json` e `.specs/LESSONS.md`.

## Level and sampling limits

O nível exercitado inclui funções, persistência real em arquivos temporários e execução integral das células do notebook em CPU com fixtures sintéticas; não é apenas inspeção de imports. Os testes de interrupção simulam falhas de `os.replace`, e os de concorrência verificam namespaces/imutabilidade com consumidores separados; não simulam dois kernels empíricos concorrentes nem queda de energia real. O caso negativo de inferência nativa altera IDs; a comparação de rótulos também existe no código, mas não recebeu um negativo isolado nesta prova.

Não foram disponibilizados os Parquets empíricos do cluster, a taxonomia empírica nem os cinco checkpoints reais. Não foi inspecionado o run remoto citado pelo usuário, nem validada sua compatibilidade efetiva com este checkout. Nenhum número do artigo foi reproduzido. Arquivos locais preexistentes, inclusive resultados não rastreados e o relatório C06, foram preservados; testes podem acrescentar novos runs sintéticos.

Perfil light: a tabela Coverage foi lida, sem recomputação independente de todos os conjuntos; nenhuma fault injection foi executada. Não há seção Test policy em `checks.md`, nem fluxo visual novo que exija julgamento humano. Esses limites não são equivalentes à validação empírica remota.

## Gate

Provas independentes: **79 + 9 testes passaram, zero falhas**. Gate de conclusão:

`python3 .agent/skills/tlc-spec-lean/scripts/validate_verification.py c09-latent-resume-validated`

Resultado: exit 0, zero erros e zero avisos.
