# M01 — auditar dados, taxonomia e split

M01 parametriza os notebooks `short/10`, `short/11` e `short/05`, preservando os históricos em `short/`. O manifesto COMPLETE indica execução concluída, não uma taxonomia aprovada nem reprodução do artigo. Use `mapping_integrity_ok` e `hierarchy_ready` para decidir se a hierarquia pode alimentar 02/06. Nenhum modelo é carregado e nenhum treinamento é iniciado.

## Execução coordenada com M00

No servidor, ajuste DATA_ROOT aos artefatos reais e selecione um ID novo. O comando M00 pode retornar código 2 enquanto dependências ainda são futuras; inspecione os impedimentos por etapa.

```bash
export DATA_ROOT=/volume/dados
export REMOTE_SERVER=True
export SMOKE_TEST=0
PYTHONPATH=src python3 scripts/plan_reproduction_run.py --config configs/reproduction_m00.yaml --run-id rodada-m01
export M01_RUN_PLAN="$PWD/results_reproduction/runs/rodada-m01/m00/run_plan.json"
python3 scripts/preflight_notebook.py short/10 --run-plan "$M01_RUN_PLAN"
jupyter nbconvert --execute --to notebook notebooks/reproduction/short/10_decisao_retreinamento_por_genoma.ipynb --output /tmp/m01-10-executado.ipynb
python3 scripts/preflight_notebook.py short/11 --run-plan "$M01_RUN_PLAN"
jupyter nbconvert --execute --to notebook notebooks/reproduction/short/11_auditoria_taxonomia_ictv_split.ipynb --output /tmp/m01-11-executado.ipynb
python3 scripts/preflight_notebook.py short/05 --run-plan "$M01_RUN_PLAN"
jupyter nbconvert --execute --to notebook notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb --output /tmp/m01-05-executado.ipynb
```

Confirme READY da etapa antes de executar. As três reservas são `results_reproduction/runs/rodada-m01-short-10/short/10/`, `.../rodada-m01-short-11/short/11/` e `.../rodada-m01-short-05/short/05/`. Cada uma contém `manifest.json`. O 11 e o 05 selecionam exatamente o manifesto 10 previsto no plano, sem descoberta por data. Adotar reserva vazia usa claim exclusivo; repetir uma etapa exige nova rodada. Não remova o claim para reutilizar um diretório parcial.

A configuração M00 fixa os caminhos por `role` (original_mapping, ictv_master, taxonomy_path e train/validation/test). Configuração `m01` por etapa acrescenta modo/opções; dependência `source: short/10` fixa o manifesto. Overrides conflitantes são recusados. Para auditar um híbrido existente sem a planilha, crie cópia da configuração M00 com `m01: {mode: audit_existing}` em 05 e substitua seu requisito ictv_master pelo Parquet com `role: taxonomy_path`. Isso é outro protocolo; não recria o híbrido histórico.

## Execução individual

Copie `configs/reproduction_m01.yaml` e informe entradas reais. Entradas relativas usam DATA_ROOT; configuração pode usar caminhos absolutos no arquivo local, sem versioná-los. `M01_CONFIG` seleciona esse YAML; sem plano, ambiente prevalece sobre configuração. Sem ambos, usa a topologia AD-017. `M01_PROJECT_ROOT` permite direcionar a raiz das reservas em testes; saídas sempre usam `results_reproduction/runs/` nessa raiz, nunca pastas históricas.

```bash
export M01_CONFIG="$PWD/configs/reproduction_m01.yaml"
export DATA_ROOT=/volume/dados
export SMOKE_TEST=0
jupyter nbconvert --execute --to notebook notebooks/reproduction/short/10_decisao_retreinamento_por_genoma.ipynb --output /tmp/m01-10-executado.ipynb
# Copie o caminho EXATO mostrado por 10; não selecione por data.
export OVERLAP_MANIFEST=/volume/repositorio/results_reproduction/runs/ID-SELECIONADO/short/10/manifest.json
jupyter nbconvert --execute --to notebook notebooks/reproduction/short/11_auditoria_taxonomia_ictv_split.ipynb --output /tmp/m01-11-executado.ipynb
export M01_MODE=audit_existing
jupyter nbconvert --execute --to notebook notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb --output /tmp/m01-05-executado.ipynb
```

Overrides disponíveis sem plano: `ORIGINAL_MAPPING_PATH`, `TAXONOMY_PATH` (arquivo ou diretório), `ICTV_MASTER_PATH`, `TRAIN_SPLIT_PATH`, `VALIDATION_SPLIT_PATH`, `TEST_SPLIT_PATH`, `OVERLAP_MANIFEST`, `OVERLAP_DIR` (diretório contendo manifesto M01), `AUDIT_WORK_DIR`, `NCBI_TAXONOMY_PATH`, `M01_MODE`, `MAX_TEST_ROWS`. Não são aceitas saídas fixas antigas: cada execução reserva novo run. M01 não promove `decision.json` legado sem manifesto a evidência identificada. Regere 10 ou mantenha o legado apenas como referência histórica.

## Produtos e interpretação

- 10: `decision.json`, `per_species.csv`, `split_counts.csv`, `split_class_counts.csv`, `split_lengths.csv` e manifesto. Contagens dos três splits são integrais mesmo em piloto; somente o overlap do test é limitado por MAX_TEST_ROWS. O índice SQLite temporário cresce em disco com o treino. A contagem completa faz uma passagem adicional pelos fragmentos; este custo não é um benchmark de desempenho.
- 11: `taxonomy_audit.json`, `per_taxon.csv`, `per_genus.csv`, `per_family.csv`, `per_order.csv`, `rank_validation.csv` e manifesto. Classes com rank nulo/placeholder entram no grupo `[unresolved]`; nenhuma é removida do denominador. O relatório identifica os labels afetados por Genus/Order. Taxonomia sem model_label não é certificada.
- 05: `curation_audit.json`, `rank_validation.csv`, `class_validation.csv` e manifesto. `curate` só publica `main_taxonomy_hybrid_CLEANSED.parquet` se identidade/ranks forem válidos. `audit_existing` nunca publica outro Parquet nem altera a fonte. Se splits.test for configurado, preserva o diagnóstico de duplicação intra-teste, limitado a 50.000 linhas em modo empírico (100 em smoke); esse diagnóstico não substitui overlap train→test.

Taxas: `exact_overlap_pct = 100 * exact_test_matches_train / test_rows`, incluindo vazias e sem A/C/G/T no denominador. `direct_matches + reverse_complement_matches - both_matches` dá a união; ambas as contagens de orientação incluem palíndromos. Linhas sem A/C/G/T não entram no índice. IUPAC tem complemento completo; símbolos desconhecidos são comparados somente no sentido direto, como no auditor anterior. A normalização remove padding `_` à direita, converte `_` interno em N e usa maiúsculas. O subconjunto compatível com até 512 bp é separado. Essa população é distinta das queries selecionadas C03.

Sem ID genômico confiável por fragmento, 10 conserva PROVENANCE_REQUIRED. Limite de teste produz INCONCLUSIVE_PARTIAL_TEST; 11 rejeita esse piloto. A validação de taxonomia não identifica os genomas de origem. Genus/Order nulos ou Unassigned/Unclassified resultam em hierarchy_ready=false. Esses rótulos não viram clados biológicos por preenchimento automático; para métricas hierárquicas, resolva a fonte ou defina outro protocolo explícito antes de 02/06.

Curadoria usa MSL41.v1, abas MSL e Taxa Renamed or Abolished, CSV original e fallback NCBI local opcional. Não consulta NCBI. O algoritmo existente de matching é preservado; fuzzy matching não demonstra linhagem resolvida. Casos não resolvidos bloqueiam publicação. O saneamento legado continua disponível para consumidores antigos, mas M01 usa validação estrita sem correção silenciosa.

Manifestos guardam modo, protocolo, configuração, semente 42, versões medidas, Git/dirty, hashes integrais de entradas pequenas e produtos. Fragmentos grandes têm inventário com tamanho e hash_status=skipped_size acima de hash_max_bytes (16 MiB padrão), sha256=unknown; isso não comprova identidade criptográfica dos fragmentos. O manifesto predecessor e todos os seus produtos têm hash verificado antes de 11. Dependência sintética não entra em execução empírica. COMPLETE é publicado por temporário/rename; falhas após reserva geram FAILED e propagam a exceção. LOG_PATH registra entradas por padrão; LOG_PATH=0/false desativa.

## Verificação local e limites

```bash
REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m01_data_taxonomy_split -v
```

As provas usam apenas fixtures temporárias CSV/Parquet e planilha mínima real XLSX, sem rede/GPU ou pesos. Para demonstrar 05 sintético isoladamente, use `REMOTE_SERVER=False SMOKE_TEST=1` com M01_CONFIG local e M01_PROJECT_ROOT em diretório temporário. 10/11 não inventam fragmentos: mesmo em smoke precisam de fixtures explicitamente fornecidas. O modo padrão não habilita smoke quando falta entrada.

Preflight remoto e validação empírica integral permanecem pendentes. Planilha MSL41.v1 e híbrido canônico não estão disponíveis localmente; Parquets locais não têm origem empírica demonstrada. Não houve reprodução de números do artigo. Divergências históricas preservadas: 05 preenchia ranks/corrigia labels e simulava quando faltava a planilha; 11 não validava todos os model_label; percentuais legados sem normalização/identidade não são promovidos a novos resultados. Dropout e títulos divergentes em configs/manuscrito não foram alterados.

## Incidente na validação inicial do código legado

Antes da substituição do notebook 05, sua execução smoke antiga chamou `ensure_smoke_test_artifacts` com a raiz local. Essa função sobrescreveu sete Parquets ignorados pelo Git: `cnn_kraken/processed_kraken_species_splited_parquet/{train,test,val}.parquet`, `cnn_kraken/processed_kraken_genus_splited_parquet/{train,test,val}.parquet` e `processed_ncbi_splited_parquet/test.parquet`. Não foram guardados hashes anteriores, portanto não é possível demonstrar ou restaurar seu conteúdo anterior nesta sessão. Os CSVs de Species/Genus não foram sobrescritos; seus hashes foram confirmados pelo Verifier.

Esses Parquets locais devem ser tratados como fixtures sintéticas, sem valor de evidência empírica. Não se afirma que toda a árvore ignorada foi preservada durante o desenvolvimento. O novo fluxo M01 não chama esse gerador; suas provas e as provas N04 afetadas usam namespaces temporários. Fontes empíricas devem ser selecionadas novamente e verificadas no cluster antes da execução integral. O incidente não autoriza regenerar dados históricos por suposição.
