# Project state

## Decisions

| ID | Decision | Rationale | Status | Date |
| --- | --- | --- | --- | --- |
| AD-001 | Closed-set viral taxonomy with 13,683 Species and 2,702 Genera | Established by the published manuscript and confirmed by species_name_ids.csv (13683 rows) and genus_name_ids.csv (2702 rows). | active | 2026-10-21 |
| AD-002 | 1-based CSV ID to 0-based model label offset (`model_label = id - 1`) | Historical models output logits indexed [0, NUM_CLASSES - 1]; taxonomic mapping CSVs store 1-based IDs [1, NUM_CLASSES]. | active | 2026-10-21 |
| AD-003 | Strict 4-channel spatial encoding A/C/G/T (0, 1, 2, 3) with null vector for padding and ambiguous bases | Preserves exact invariant from manuscript and `short/utils/models.py:OneHotEmbedding` where invalid/padding indices produce zero vectors. | active | 2026-10-21 |
| AD-004 | Preserve historical notebooks in original locations without in-place breaking moves | Guarantees auditability and prevents silent corruption of notebook experiments; new code extracted cleanly into `src/` and `scripts/`. | active | 2026-10-21 |
| AD-005 | Synthetic contract verification when model weights (`.pt`) or parquet splits are unavailable on disk | Manuscript artifacts (.pt checkpoints) are missing from local filesystem; synthetic deterministic contracts verify pipeline mechanics without faking weights. | active | 2026-10-21 |

## Handoff

**Feature**: s01-inventory-and-vertical-slice
**Where**: C1-C8 closed at 3ae1a73 and verified (PASS)
**In progress**: none
**Next step**: S02 — Proveniência do dataset e auditoria do split por fragmento
**Blockers**: none
**Uncommitted**: none
**Branch**: main

