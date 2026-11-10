# Guia e Documentação da Fase M03: Espaço Latente e Visualizações Úteis

## Escopo e Invariantes Científicos

A fase **M03** padroniza a execução dos notebooks `short/06_ictv_latent_space_and_hierarchy.ipynb` e `short/06b_latent_visualizations.ipynb` sob o protocolo `m03_latent_audit_v1`.

### Invariantes
1. **Espaço original de alta dimensão**: Avaliação de KNN Cosseno ($k=5$), Silhouette Score e Sonda Linear selada é realizada estritamente nas ativações pré-classificadoras de alta dimensão (2048 para ResNet50 e 1024 para ConvNeXt).
2. **Natureza qualitativa de UMAP/PCA**: As projeções 2D são ajustadas de forma independente para cada uma das cinco variantes de modelo. Os eixos 2D são independentes e não representam distância filogenética nem substituem métricas de clusterização quantitativa.
3. **Amostra pareada C10**: Todas as variantes são projetadas exatamente sobre os mesmos fragmentos (`row_id`), classes de espécies e gêneros, sem interseção silenciosa ou amostragem visualmente tendenciosa.
4. **Painel Seletivo e Fundo Neutro**: Além do painel completo, é gerado `umap_selective_panel.png` destacando os gêneros com maior suporte amostral, enquanto os demais fragmentos são representados em cinza neutro ao fundo (`#d0d0d0`). São exportados `selected_genera.csv` e `sampled_query_ids.csv`.
5. **Adoção M00 e Eliminação de Caminhos Fixos**: Todos os caminhos de saída usam namespaces reservados sob `results_reproduction/runs/<run_id>/short/{06,06b}` e integram-se ao plano de execução `m00_run_plan_v1`.

## Comandos de Execução

### 1. Preflight e Verificação de Requisitos
```bash
# Inspecionar pré-requisitos de 06
PYTHONPATH=src python3 scripts/preflight_notebook.py --notebook short/06 --data-root /path/to/data

# Inspecionar pré-requisitos de 06b
LATENT_SOURCE_DIR=/path/to/run/short/06 PYTHONPATH=src python3 scripts/preflight_notebook.py --notebook short/06b --data-root /path/to/data
```

### 2. Execução Local (Smoke Test em CPU)
```bash
# Execução sintética de 06
SMOKE_TEST=1 PYTHONPATH=src python3 -c "from viral_cnn.m03_workflow import begin_m03; run = begin_m03('short/06'); run.prepare_smoke(); print(run.manifest)"

# Execução sintética de 06b
SMOKE_TEST=1 PYTHONPATH=src python3 -c "from viral_cnn.m03_workflow import begin_m03; run = begin_m03('short/06b'); print(run.manifest)"
```

### 3. Execução Remota no Cluster (Empírica, com GPUs)
```bash
# Com plano M00 aprovado
RUN_PLAN=results_reproduction/runs/<run_id>/run_plan.json jupyter nbconvert --to notebook --execute notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb
```
