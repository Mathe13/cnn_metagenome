# C08 — verificação independente

**Verdict**: PASS
**Profile**: light
**Diff range**: `d3d52ea78f667341f43038a1c15a4530f8740d63..f7ed9dca48ad1adbfb4c0e70688c881a42d3b098`
**Round**: 3 - full
**Verifier**: independent sub-agent (author != verifier)

O diff completo foi relido em HEAD, junto de `plan.md`, `checks.md`, `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` (AD-012/017/018/019/025), `docs/latent_space_audit.md`, `README.md`, notebook 06, módulo KNN e testes C08. Os checks são compatíveis com essas fontes. A auditoria histórica descreve Silhouette até 50.000 linhas; o protocolo do notebook 06 usa 5.000 e o README explicita a distinção.

## Gate

`.venv/bin/python3 -m unittest -v tests.test_c08_knn_workspace` executou em uma invocação todos os 12 testes nomeados C1–C12 em HEAD: **12 passed, 0 failed**. Cada nome existe em `tests/test_c08_knn_workspace.py`, apareceu individualmente como `ok` e é novo no diff. O smoke executa o notebook completo em CPU sintética. Manifesto observado: workspace 1024 MiB, lote 256, dispositivo `cpu`, 15 blocos efetivos, 15 amostras Silhouette, 15 contagens de classes retidas/excluídas e UMAP de 16 linhas. C13 é critério do conteúdo deste relatório; `validate_verification.py` é gate de formato/fechamento, sem prova empírica própria.

## Checks

| Check | Proof run | Located assertion | Result |
| --- | --- | --- | --- |
| C1 | `test_notebook_knn_defaults_overrides_and_device` ok | `tests/test_c08_knn_workspace.py:100` `self.assertEqual(namespace[name], expected)`; `:108` `self.assertEqual(ast.unparse(device_argument), "ctx.device")`; `:119` `self.assertEqual(configuration["latent_knn_workspace_mb"], 1024)`; `:121` `self.assertEqual(configuration["latent_knn_device"], "cpu")` | PASS — defaults, override e manifesto smoke. |
| C2 | `test_invalid_workspace_and_too_small_budget_fail_before_matmul` ok | `tests/test_c08_knn_workspace.py:130` `self.assertEqual(mm.call_count, 0)`; `:133` `self.assertRaisesRegex(RuntimeError, "requested.*minimum")`; `:136` `self.assertEqual(mm.call_count, 0)` | PASS — ambos caminhos falham antes de `torch.mm`. |
| C3 | `test_tiled_knn_keeps_scientific_protocol` ok | `tests/test_c08_knn_workspace.py:143` `self.assertEqual((result["n_reference"], result["n_query"]), (80, 20))`; `:158–159` `assert_array_equal` para partições independentes; `:162` `assert_array_equal(result["y_pred"], expected_vote)` | PASS — k=5, 42, cosseno/L2 por oráculo denso e zero auto-vizinho. |
| C4 | `test_workspace_bounds_matmul_and_never_gathers_full_features` ok | `tests/test_c08_knn_workspace.py:178` `self.assertLessEqual(max(guarded.gathers), 31)`; `:184` `self.assertLessEqual(estimated, 1024 * 1024)`; `:189` `self.assertIn(result["n_query"] % 17, [left[0] for left, _ in products])` | PASS — 1.031 linhas, lote final e estimativa dos tensores explícitos; não mede RSS/VRAM. |
| C5 | `test_tiled_topk_matches_dense_and_breaks_ties_by_original_index` ok | `tests/test_c08_knn_workspace.py:200–201` `assert_array_equal(tiled["neighbor_indices"], expected)` e `assert_array_equal(one["neighbor_indices"], expected)` | PASS — empate exato em blocos distintos. |
| C6 | `test_tiled_knn_global_metrics_match_dense_reference` ok | `tests/test_c08_knn_workspace.py:211–213` N, Accuracy e Top-5 globais versus referência densa; `:217` suporte por classe; `:220–222` `self.assertAlmostEqual` para `knn_accuracy` e `knn_top5_accuracy` por classe; `:236` `assert_allclose(scores[0], expected_scores, atol=1e-5, rtol=0)` | PASS — lacuna da rodada 2 fechada. |
| C7 | `test_rank_selection_and_silhouette_sample_before_feature_gather` ok | `tests/test_c08_knn_workspace.py:245` consultas contidas na seleção; `:248–249` amostra Silhouette 20 e gather `<=30`; `:256–258` compara `y_true` e `taxonomic_rank` para Species, Genus, Order | PASS — três níveis sintéticos; Silhouette amostra antes de reunir features. |
| C8 | `test_compact_knn_audit_roundtrip_and_artifact_binding` ok | `tests/test_c08_knn_workspace.py:267` hash do artefato test; `:275–278` versão, IDs ausentes e vínculo; `:280–284` N, cinco vizinhos, `assert_array_equal(table.column("query_index").to_numpy(), result["query_indices"])` e vizinhos densos; `:286–288` hashes reconstruídos | PASS — lacuna da rodada 2 fechada. |
| C9 | `test_knn_audit_metadata_hashes_and_counts` ok | `tests/test_c08_knn_workspace.py:298–301` compara os três hashes ordenados; `:302–308` compara N, auto-vizinho, dispositivo, protocolo e blocos | PASS — metadados em CPU. |
| C10 | `test_audit_write_failure_is_atomic` ok | `tests/test_c08_knn_workspace.py:315–319` falha Parquet e ausência de arquivo/temporário; `:322–324` falha JSON; `:330–335` ausência de auditoria completa/manifesto após execução falha | PASS — falha injetada e notebook smoke sob falha. |
| C11 | `test_legacy_knn_details_and_notebook_compact_mode` ok | `tests/test_c08_knn_workspace.py:343–346` tamanhos, shape e IDs legados; `:348–350` chamada compacta e ausência de serialização integral no notebook | PASS — retorno legado executado, modo compacto estático. |
| C12 | `test_notebook_silhouette_umap_limits_and_manifest` ok | `tests/test_c08_knn_workspace.py:354–355` limites 5.000/100 presentes; `:362–364` 15 registros de amostra/classes; `:366–372` tamanho real versus Tabela 5, classes e UMAP ≤100 | PASS — manifesto smoke exercitado, ramo remoto de 5.000 inspecionado estaticamente. |
| C13 | `validate_verification.py c08-knn-bounded-workspace` | Este relatório, `## Level and sampling`, distingue CPU sintética, CUDA/dados reais pendentes e reprodução não realizada | PASS — critério procedural do texto do relatório; gate não oferece prova científica autônoma. |

## Level and sampling

C1/C10/C12 executam notebook smoke; C8 salva e relê artefato test sintético com hash real. O smoke usa 16 linhas por split, cinco variantes e três níveis. C3–C7 e C9–C11 testam contratos Python em amostras sintéticas de até 1.031 linhas. C4 estima o workspace explícito conforme o plano, sem medir RSS/VRAM, overhead de PyTorch ou cache CUDA. Validação CUDA, memória e equivalência numérica com dados/checkpoints canônicos no servidor remoto seguem pendentes. **Não houve reprodução científica das Tabelas 5/6 ou da Figura 2.**

## Swept existing

Validação: limites e seleção em `src/viral_cnn/eval/latent.py:571–592`, cobertos por C2. Falha parcial/ciclo de dados: temporários removidos em `latent.py:545` e `:750–755`, C10 prova ausência de manifesto completo. Idempotência/concorrência: diretório por `run_id` no notebook e recusa de alvo existente em `latent.py:538` e `:705`; escrita simultânea no mesmo alvo não é garantia declarada. Autorização: n/a, pipeline local. Dependências: falha de disco local coberta, sem rede. Transição: JSON vem após Parquets e antes do manifesto, C10. Observabilidade: parâmetros, blocos e tamanhos no manifesto, C1/C9/C12. Linhas n/a do Swept aprovado não exigem prova em código.

## Ranked gaps

Nenhum gap de check no escopo C08. A execução CUDA com dados reais permanece trabalho empírico posterior e não é apresentada como reprodução.
