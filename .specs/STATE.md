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

## Handoff

**Feature**: s04-model-registry-and-training-manifest
**Where**: C1-C11 closed at 50db814 and verified (PASS)
**In progress**: none
**Next step**: S05 — Pipeline de inferência por janelas e agregação
**Blockers**: none
**Uncommitted**: none
**Branch**: main
