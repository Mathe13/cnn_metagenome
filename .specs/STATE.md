# Project state

## Decisions

| ID | Decision | Rationale | Status | Date |
| --- | --- | --- | --- | --- |
| AD-001 | Closed-set viral taxonomy with 13,683 Species and 2,702 Genera | Established by the published manuscript and confirmed by species_name_ids.csv (13683 rows) and genus_name_ids.csv (2702 rows). | active | 2026-10-21 |
| AD-002 | 1-based CSV ID to 0-based model label offset (`model_label = id - 1`) | Historical models output logits indexed [0, NUM_CLASSES - 1]; taxonomic mapping CSVs store 1-based IDs [1, NUM_CLASSES]. | active | 2026-10-21 |
| AD-003 | Strict 4-channel spatial encoding A/C/G/T (0, 1, 2, 3) with null vector for padding and ambiguous bases | Preserves exact invariant from manuscript and `short/utils/models.py:OneHotEmbedding` where invalid/padding indices produce zero vectors. | active | 2026-10-21 |
| AD-004 | Preserve historical notebooks in original locations without in-place breaking moves | Guarantees auditability and prevents silent corruption of notebook experiments; new code extracted cleanly into `src/` and `scripts/`. | active | 2026-10-21 |
| AD-005 | Synthetic contract verification when model weights (`.pt`) or parquet splits are unavailable on disk | Manuscript artifacts (.pt checkpoints) are missing from local filesystem; synthetic deterministic contracts verify pipeline mechanics without faking weights. | active | 2026-10-21 |
| AD-006 | Dados volumosos e checkpoints residem no armazenamento do JupyterHub remoto | Preserva o repositório Git leve, viabiliza execução em GPU com Dask/cuDF no JupyterHub remoto, e parametriza caminhos via `configs/` ou variável `DATA_ROOT`. | active | 2026-10-21 |
| AD-007 | Avaliação supervisionada closed-set desincorporada de genoma exige $\ge 2$ genomas por espécie | Espécies com genoma único são inviáveis para teste desincorporado de genoma sem vazamento ou violação closed-set; acurácias sob split agrupado refletem população reduzida e não podem ser confrontadas diretamente com o split de fragmentos (13.683 classes) sem explicitar a mudança de suporte. | active | 2026-10-21 |
| AD-008 | Registro canônico de modelos com validação de cabeça e hiperparâmetros não evidenciados explicitamente rotulados como unverified | Normalização unificada para bare state_dict e wrapper model_state_dict, remoção do prefixo _orig_mod. e checagem rígida de dimensão de classes (fc.weight) impedem misturas de ranks; lacunas históricas permanecem unverified conforme AGENTS.md. | active | 2026-10-21 |
| AD-009 | Protocolo de ruído com taxa efetiva $3\epsilon/4$, faixas contínuas de F1 e métricas vetorizadas | O ressorteio uniforme entre A/C/G/T produz taxa efetiva $3\epsilon/4$; faixas de F1 cobrem o intervalo [0, 1] continuamente conservando o número de classes; cálculo exato de F1 vetorizado via bincount dispensa dependências externas pesadas e preserva correspondência com o paper. | active | 2026-10-21 |
| AD-010 | Auditoria do Long Voter: desincorporação de generalização para genomas não vistos e causas da queda de ConvNeXt | As sequências longas do NCBI coincidem com as referências da base Kraken2 (mesmo genoma de origem), de modo que os 72,69% da ResNet50 medem acurácia em sequências de genomas já vistos e não generalização; a queda de ConvNeXt decorre de viés na simulação de leituras (super-representação de vírus gigantes com 500 reads vs 50), logit pulling na média pré-softmax e perda de calibração em ConvNeXt_T; contigs <512 bp passam a ser tratados explicitamente via modo padronizado robusto (handle_short='pad') ou histórico (handle_short='skip'). | active | 2026-10-21 |
| AD-011 | Protocolo unificado de baselines Kraken2 e BLASTN com denominador estrito e estratificação de overlap/genoma | O cálculo rigoroso de acurácia Top-1/Top-5 mantém o total de consultas $N$ no denominador e trata abstenções (unclassified ou sem hit) como rótulo -1 (erro), computando cobertura e acurácia condicional separadamente; estratifica consultas curtas por overlap exato de treino (all, overlap, no_exact_overlap com ~95,89% de sobreposição no split de fragmentos) e consultas longas por presença do genoma de origem (seen_genome vs unseen_genome) e faixas de comprimento; comparações diretas exigem os mesmos identificadores de consulta e cluster bootstrap reamostrando espécies com reposição (2.000 repetições, semente 42) para IC de 95% das diferenças pareadas. | active | 2026-10-21 |
| AD-012 | Reavaliação rigorosa do espaço latente pré-classificador, desvinculação de UMAP e auditoria de sonda linear independente do teste | Embeddings extraídos imediatamente antes de self.fc sem redução de dimensionalidade; UMAP é estritamente qualitativo 2D; KNN (k=5, cosseno, L2-norm) usa split 80/20 disjunto para evitar auto-vizinhança; Silhouette Score amostral registra taxa de classes válidas (<2 amostras excluídas) e contrasta amostragem aleatória vs estratificada; hierarquia taxonômica mantém conjuntos válidos independentes entre KNN e Silhouette; e a investigação de 'gargalo' no ConvNeXt exige treino de sonda linear estritamente em treino/validação, pois KNN em teste não constitui prova de gargalo na cabeça. | active | 2026-10-21 |

## Handoff

**Feature**: s08-latent-space-evaluation
**Where**: plan.md validated cleanly (0 errors, 0 warnings); awaiting human confirmation before checks.md
**In progress**: none
**Next step**: User review of plan.md, then write checks and build
**Blockers**: none
**Uncommitted**: none
**Branch**: main
