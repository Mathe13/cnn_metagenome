# Improvement Plan — `cnn_metagenome`

**Repository:** `Mathe13/cnn_metagenome`  
**Execution approach:** `tlc-spec-driven` + Antigravity  
**Primary objective:** eliminate the remaining discrepancies between scientific contracts, implementation, empirical artifacts, and manuscript claims before freezing the final reproduction package.

---

## 1. Context

The repository has evolved from a notebook-oriented research codebase into a modular reproducibility framework with:

- explicit scientific invariants;
- Architecture Decision Records (ADRs);
- deterministic smoke tests;
- empirical reproduction notebooks;
- artifact manifests and provenance;
- strict publication contracts;
- calibration analysis;
- tiled KNN evaluation;
- long-sequence consensus inference;
- Kraken2/BLASTN benchmarking;
- controlled robustness experiments.

The remaining work is not a redesign of the research pipeline.

The current priority is to close a small number of methodological and provenance gaps identified during code review, particularly:

1. paired motif perturbation does not currently reuse the exact same nucleotide substitutions between motif and control;
2. GC/complexity-matched controls can silently degrade into relaxed or unmatched controls;
3. historical hardcoded reporting remains accessible through the default non-strict path;
4. the manuscript must be reconciled with the measured train/test sequence overlap;
5. a few APIs can be hardened to prevent scientifically ambiguous execution.

The implementation strategy should prefer **fail-closed scientific contracts** over silent fallbacks.

---

# 2. Global principles

The following rules apply to every improvement spec.

## 2.1 Scientific evidence must be traceable

Every artifact used by the manuscript must be traceable to:

```text
input dataset
→ configuration
→ checkpoint
→ code revision
→ execution manifest
→ output artifact
→ publication table/figure
```

No publication output should depend implicitly on historical constants.

---

## 2.2 Smoke tests are not empirical evidence

Synthetic and smoke-test artifacts may validate:

- control flow;
- schema;
- deterministic behavior;
- API compatibility;
- mathematical invariants.

They must never be promoted automatically to manuscript evidence.

---

## 2.3 Historical results must be explicit

Historical values may remain in the repository for:

- regression checking;
- documentation;
- reconciliation;
- provenance.

They must be clearly identified as:

```text
historical_snapshot
```

and never emitted by default publication commands.

---

## 2.4 Scientific fallbacks must be observable

Any fallback that weakens the intended experimental control must be:

- explicitly recorded;
- quantifiable;
- available in generated artifacts;
- filterable from the primary analysis.

Silent relaxation is not allowed.

---

## 2.5 Avoid unnecessary retraining

None of the improvements below should trigger full model retraining unless a specification explicitly requires it.

The priority is:

```text
fix analysis
→ validate
→ rerun affected downstream analysis
→ reconcile manuscript
```

not:

```text
retrain all neural networks
```

---

# 3. Priority overview

| ID | Improvement | Priority | Requires retraining? | Requires analysis rerun? |
|---|---|---:|---:|---:|
| S10 | Exact paired motif perturbation | P0 | No | Yes — motif pipeline |
| S11 | Strict GC/complexity control matching | P0 | No | Yes — motif pipeline |
| S12 | Publication reporting fail-closed by default | P0 | No | No, but regenerate publication package |
| S13 | Dataset overlap/manuscript reconciliation contract | P0 | No | No |
| S14 | Motif regression and scientific-contract tests | P0 | No | No |
| S15 | Empirical artifact provenance hardening | P1 | No | Publication regeneration |
| S16 | Calibration label validation | P1 | No | Optional |
| S17 | Training scheduler API hardening | P1 | No | No historical rerun |
| S18 | ConvNeXt naming and architecture metadata | P1 | No | No |
| S19 | Historical reporting isolation | P1 | No | No |
| S20 | Final empirical reproduction gate | P0 final | No | Yes — affected stages |
| S21 | Manuscript reconciliation report | P0 final | No | No |

---

# 4. S10 — Exact paired motif perturbation

## Problem

The manuscript describes a paired perturbation experiment in which the motif region and matched control region receive the **same nucleotide substitution pattern**.

Current execution instead performs two independent calls:

```python
m_pert = perturb_sequence_mask(..., generator=generator)
c_pert = perturb_sequence_mask(..., generator=generator)
```

Although both use the same generator, the second call consumes a different portion of the RNG stream.

Therefore:

```text
same RNG != same mutation offsets
```

The current comparison does not isolate genomic location as strictly as intended.

---

## Scientific contract

For each motif/control pair and repeat:

```text
substitution offsets applied to motif
==
substitution offsets applied to control
```

Only the genomic position may differ.

For a motif of length `K`:

```text
offsets = [δ1, δ2, ..., δK]
δi ∈ {1, 2, 3}
```

Both regions must use the same vector.

---

## Proposed implementation

Introduce explicit separation between:

### A. Sampling substitution pattern

```python
sample_substitution_offsets(
    n_positions: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor
```

### B. Applying substitution pattern

```python
apply_substitution_offsets(
    sequence,
    mask,
    offsets,
)
```

or preferably:

```python
perturb_region_with_offsets(
    sequence,
    start,
    offsets,
)
```

Then:

```python
offsets = sample_substitution_offsets(K, generator, device)

motif_perturbed = perturb_region_with_offsets(
    sequence,
    motif_start,
    offsets,
)

control_perturbed = perturb_region_with_offsets(
    sequence,
    control_start,
    offsets,
)
```

---

## Required metadata

Persist at least:

```text
sequence_id
motif
motif_start
control_start
repeat_id
mutation_pattern_hash
control_policy
```

The full offsets do not necessarily need to be written if a deterministic hash is sufficient for auditing.

---

## Acceptance criteria

- [ ] motif and control use identical substitution offsets within each paired repeat;
- [ ] substitutions always change the nucleotide;
- [ ] all substitutions remain in `{A,C,G,T}`;
- [ ] unmasked positions remain unchanged;
- [ ] deterministic execution under fixed seed;
- [ ] output records contain enough provenance to prove pairing;
- [ ] existing API users remain compatible or migration is documented.

---

## Required test

Add:

```text
test_paired_motif_control_reuses_identical_substitution_offsets
```

A strong test should construct two identical 7-bp regions and verify:

```text
mutated_motif_region == mutated_control_region
```

when the same offsets are applied.

---

## Rerun scope

Rerun:

```text
N05 motif perturbation
→ motif statistical validation
→ BH-FDR
→ motif tables
→ motif figures
→ downstream analyses depending on validated motif selection
```

Do **not** rerun model training.

---

# 5. S11 — Strict matched-control policies

## Problem

`gc_matched` currently allows:

```text
exact GC match
or
±1 GC base fallback
```

For a 7-mer:

```text
1 / 7 = 14.29%
```

which is not negligible.

`complexity_matched` is even more permissive:

```text
exact complexity
→ ±1 run length
→ arbitrary valid control
```

The final fallback is no longer meaningfully complexity matched.

---

## Scientific contract

A control must never silently claim stronger matching than actually achieved.

Each control selection must produce:

```text
control_policy_requested
control_match_quality
control_match_distance
```

---

## Proposed match quality vocabulary

### GC

```text
exact
relaxed_1
unavailable
```

### Complexity

```text
exact
relaxed_1
unmatched_fallback
unavailable
```

---

## Recommended API

Replace:

```python
Optional[int]
```

with structured metadata:

```python
@dataclass
class MatchedControl:
    start: int
    policy: str
    quality: str
    distance: int
```

Example:

```python
MatchedControl(
    start=127,
    policy="gc_matched",
    quality="exact",
    distance=0,
)
```

---

## Primary scientific analysis

The primary motif analysis should use:

```text
GC quality == exact
```

Relaxed cases may be reported as:

```text
sensitivity analysis
```

If too many observations are lost, report:

```text
exact matching coverage
```

rather than silently relaxing the criterion.

---

## Suggested summary metrics

For each model/control policy:

```text
n_requested
n_exact
n_relaxed
n_unmatched
n_unavailable
exact_match_rate
```

---

## Acceptance criteria

- [ ] no silent matching relaxation;
- [ ] every selected control records match quality;
- [ ] primary GC-controlled analysis uses exact matching;
- [ ] exact matching coverage is reported;
- [ ] unmatched complexity controls cannot be labelled `complexity_matched`;
- [ ] smoke tests cover exact, relaxed and unavailable cases.

---

# 6. S12 — Publication reporting must fail closed by default

## Problem

Current API:

```python
generate_all_tables(
    ...,
    strict_empirical=False,
)
```

allows historical constants to generate apparently publication-ready tables.

This is useful for backwards compatibility but unsafe as a publication default.

---

## Scientific contract

The default publication path must use **empirical selected artifacts only**.

Historical snapshots must require explicit opt-in.

---

## Proposed change

Change:

```python
strict_empirical: bool = False
```

to:

```python
strict_empirical: bool = True
```

or, preferably, remove the ambiguity entirely.

### Recommended APIs

```python
generate_publication_tables(
    publication_config=...,
)
```

and:

```python
generate_historical_snapshot_tables(...)
```

Do not combine both behaviors behind a boolean if avoidable.

---

## Fail-closed behavior

This must fail:

```python
generate_publication_tables()
```

when no explicit empirical selection exists.

Expected error:

```text
Publication generation requires an explicit admissible empirical artifact selection.
```

---

## Acceptance criteria

- [ ] default publication path cannot emit historical constants;
- [ ] smoke artifacts are rejected;
- [ ] historical artifacts require explicit legacy API;
- [ ] all publication tables identify source manifests;
- [ ] manuscript table generation is reproducible from C06;
- [ ] no directory-wide artifact auto-discovery is accepted as scientific evidence.

---

# 7. S13 — Dataset overlap scientific contract

## Problem

The historical manuscript states or implied sequence-level disjointness.

Current dataset audit demonstrates approximately:

```text
95.91% exact or reverse-complement overlap
```

for the audited test-query sample.

The experiment remains valid as a closed-set reference-fragment recognition benchmark, but the scientific scope must be explicit.

---

## Scientific contract

Three concepts must never be conflated:

### Closed-set class membership

```text
Species_test ⊆ Species_train
```

Expected by design.

### Row-disjoint split

```text
row_train ∩ row_test = ∅
```

### Sequence/genome disjointness

Not guaranteed.

---

## Required terminology

Use:

```text
fragment-level closed-set classification
```

and:

```text
known-reference distribution
```

Avoid:

```text
unseen-genome generalization
```

unless a genome-disjoint experiment is performed.

---

## `no_exact_overlap` terminology

Allowed:

```text
no-exact-overlap subset
```

Not allowed:

```text
unseen genomes
novel genomes
genome-disjoint test
```

unless provenance proves it.

---

## Required machine-readable audit artifact

Produce something like:

```json
{
  "split_unit": "fragment",
  "row_disjoint": true,
  "genome_disjoint": false,
  "sequence_identity_audit": {
    "canonicalizes_reverse_complement": true,
    "n_queries": 273660,
    "n_exact_or_rc_overlap": 262470,
    "overlap_fraction": 0.9591
  }
}
```

---

## Acceptance criteria

- [ ] split semantics documented in code and README;
- [ ] manuscript wording reconciled;
- [ ] no claim of exact sequence disjointness remains unless proven;
- [ ] `no_exact_overlap` retains its precise meaning;
- [ ] overlap audit artifact is linked to publication provenance.

---

# 8. S14 — Scientific regression tests for motif controls

Create tests beyond ordinary implementation correctness.

Required tests:

```text
test_paired_offsets_are_identical
test_paired_offsets_are_deterministic
test_gc_exact_control_has_identical_gc_count
test_gc_relaxed_control_is_labeled_relaxed
test_gc_unavailable_returns_unavailable
test_complexity_exact_control_matches_run_length
test_complexity_fallback_is_not_labeled_exact
test_control_never_overlaps_target_motif
test_ambiguous_bases_are_not_selected_as_control
test_motif_control_artifact_records_match_quality
```

Add one end-to-end synthetic case demonstrating:

```text
same sequence
same mutation offsets
different target locations
```

---

# 9. S15 — Empirical artifact provenance hardening

## Goal

Every publication artifact should contain or reference:

```text
git commit SHA
dataset manifest SHA256
checkpoint SHA256
configuration SHA256
seed
model variant
split
noise level
execution timestamp
software/environment metadata
parent artifacts
smoke_test=false
```

---

## Required invariant

Publication artifact:

```text
artifact
```

must be invalid if:

```text
artifact.is_smoke_test == true
```

or if one of its required upstream empirical artifacts is missing.

---

## Suggested lineage

```text
C03 predictions
   ↓
C04 calibration
   ↓
C05 motifs / latent / robustness
   ↓
C06 publication selection
   ↓
Tables/Figures
```

The exact naming can follow the current architecture.

---

# 10. S16 — Calibration label fail-fast validation

## Problem

Calibration functions currently tolerate labels outside:

```text
[0, n_classes)
```

by assigning `p_true=0`.

This can produce very large NLL values rather than identifying a malformed evaluation set.

---

## Proposed contract

For neural-network closed-set evaluation:

```python
assert 0 <= y_true < n_classes
```

for all evaluated samples.

If abstention labels such as `-1` are required elsewhere, they must use an explicit metric mode rather than being silently accepted.

---

## Proposed API

```python
compute_calibration_metrics(
    ...,
    invalid_label_policy="raise",
)
```

Default:

```text
raise
```

Optional explicit alternative:

```text
exclude
```

only if scientifically justified.

---

## Acceptance criteria

- [ ] invalid neural classifier labels fail fast;
- [ ] error reports invalid count and examples;
- [ ] calibration denominator cannot change silently.

---

# 11. S17 — Training scheduler API hardening

## Problem

Two optimization interfaces currently exist:

### `train_step`

Applies internal warmup/cosine schedule.

### `train_epoch`

Can use an external scheduler.

This is valid but easy to misuse.

---

## Goal

Make scheduling semantics impossible to misunderstand.

---

## Recommended option

Declare scheduler ownership explicitly:

```python
scheduler_mode:
    internal_step
    external_batch
    external_epoch
    constant
```

A configuration may select exactly one.

---

## Invalid examples

Fail if:

```text
internal scheduler + external scheduler
```

or:

```text
warmup_steps > 0
but no scheduler owner exists
```

where the configuration requires a schedule.

---

## Important constraint

Do not alter historical checkpoint interpretation.

Historical reproduction must retain its recorded training behavior.

This improvement is primarily for:

```text
future controlled runs
```

and code safety.

---

# 12. S18 — ConvNeXt architecture identity

## Problem

The current class name:

```text
ConvNeXt_Tiny_1D
```

can imply a direct 1D conversion of canonical ConvNeXt-Tiny.

The implementation is intentionally hybrid:

- ResNet-style stem;
- `[3,4,6,3]` stages;
- ConvNeXt-inspired blocks;
- custom channel progression;
- global max pooling.

---

## Scientific contract

Architecture metadata should distinguish:

```text
canonical architecture family
```

from:

```text
repository implementation identity
```

---

## Recommended internal name

```text
HybridConvNeXt1D
```

or:

```text
ConvNeXtInspired1D
```

Maintain compatibility alias:

```python
ConvNeXt_Tiny_1D = ConvNeXtInspired1D
```

until historical checkpoints no longer depend on the old symbol.

---

## Required metadata

```json
{
  "architecture_family": "convnext-inspired",
  "implementation": "HybridConvNeXt1D",
  "stage_depths": [3,4,6,3],
  "stem": "resnet-style",
  "pooling": "global-max"
}
```

---

## Manuscript terminology

Preferred:

```text
1D ConvNeXt-inspired architecture
```

Avoid presenting it as canonical ConvNeXt-Tiny.

---

# 13. S19 — Isolate historical reporting code

Move static historical values away from production reporting.

Suggested structure:

```text
src/viral_cnn/reporting/
├── publication.py
├── empirical_tables.py
├── empirical_figures.py
└── legacy/
    ├── historical_tables.py
    └── historical_figures.py
```

Historical functions should clearly expose metadata:

```json
{
  "evidence_class": "historical_snapshot",
  "publication_admissible": false
}
```

---

## Remove dead strict branch

Current `generate_all_tables()` contains a second:

```python
if strict_empirical:
```

after an earlier strict branch has already returned.

Remove unreachable code.

---

# 14. S20 — Final empirical reproduction gate

This is the final gate before freezing manuscript values.

## Prerequisites

Must be complete:

```text
S10 exact paired perturbation
S11 strict matched controls
S12 strict reporting
S13 overlap reconciliation
S14 scientific regression tests
```

P1 tasks should preferably also be complete.

---

## Execution sequence

Use current reproduction dependency graph.

Conceptually:

```text
M00 preflight
 ↓
canonical prediction/evaluation stages
 ↓
calibration
 ↓
latent analyses
 ↓
corrected motif analysis
 ↓
baselines
 ↓
long-sequence analyses
 ↓
C06 publication selection
 ↓
publication tables/figures
```

Do not reuse pre-fix motif artifacts.

---

## Final artifact freeze

Create a final immutable publication manifest containing:

```text
paper version
git SHA
all selected evidence manifests
all table hashes
all figure hashes
dataset audit hash
checkpoint hashes
generation timestamp
```

Suggested tag:

```text
paper-reproduction-v1
```

or equivalent.

---

# 15. S21 — Manuscript reconciliation report

Generate a machine-readable and human-readable reconciliation report.

For every manuscript result:

```text
claim
table/figure
historical value
reproduced value
difference
evidence artifact
status
```

Status vocabulary:

```text
MATCH
MATCH_WITH_ROUNDING
UPDATED
REMOVED
EXPLORATORY
NOT_REPRODUCED
BLOCKED
```

Example:

| Claim | Historical | Reproduced | Status |
|---|---:|---:|---|
| ResNet clean Species accuracy | 84.88 | ... | MATCH |
| ConvNeXt F eps=.15 | 17.38 | ... | MATCH |
| Species KNN | 63.01 | ... | MATCH |
| Motif CCCCCCC effect | historical | corrected | UPDATED |
| GC matched effect | historical | strict exact | UPDATED |

The manuscript should only consume values marked publication-admissible.

---

# 16. Antigravity execution strategy

The tasks should be implemented sequentially rather than as a large unconstrained refactor.

Recommended order:

```text
S10
 ↓
S14
 ↓
S11
 ↓
rerun motif smoke tests
 ↓
S12
 ↓
S19
 ↓
S13
 ↓
S15
 ↓
S16
 ↓
S17
 ↓
S18
 ↓
S20
 ↓
S21
```

---

## Per-spec Antigravity workflow

For each spec:

### Phase 1 — Inspect

Antigravity must identify:

```text
affected files
current behavior
scientific invariant
downstream dependencies
existing tests
```

No code change yet.

### Phase 2 — Plan

Produce:

```text
implementation design
API changes
migration risks
test cases
artifact impact
```

### Phase 3 — Implement

Prefer minimal changes.

Do not refactor unrelated systems during P0 scientific fixes.

### Phase 4 — Verify

Run:

```text
unit tests
contract tests
affected notebook smoke test
static validation
```

### Phase 5 — Evidence

Generate:

```text
verification report
git diff summary
tests executed
artifacts invalidated
artifacts requiring rerun
```

### Phase 6 — Close

Only close a spec when every acceptance criterion is mechanically verifiable.

---

# 17. Dependency / invalidation map

## S10 invalidates

```text
motif perturbation empirical artifacts
motif Wilcoxon results
motif BH-FDR
validated motif list
downstream analyses using validated motifs
```

Does not invalidate:

```text
CNN checkpoints
clean accuracy
noise robustness
calibration
latent embeddings
KNN
Silhouette
long-voter inference
```

---

## S11 invalidates

```text
GC-controlled motif outputs
complexity-controlled motif outputs
interpretations based on those controls
```

---

## S12/S19 invalidate

No empirical experiment.

They invalidate only improperly generated publication outputs.

---

## S13 invalidates

No numerical result.

It changes:

```text
interpretation
scope
manuscript wording
```

---

# 18. Explicit non-goals

The improvement cycle should **not** automatically include:

- retraining all five neural-network configurations;
- redesigning ResNet50;
- redesigning ConvNeXt;
- adding new foundation models;
- creating an open-set classifier;
- rebuilding the original dataset;
- generating genome-disjoint data when provenance is unavailable;
- adding realistic ONT simulation unless separately approved;
- replacing Kraken2/BLASTN experiments;
- conducting large new architectural ablations.

These may become future research, but they should not block the current manuscript.

---

# 19. Definition of Done

The repository is ready for final manuscript freeze when:

- [ ] motif/control perturbations use identical nucleotide substitutions;
- [ ] this invariant is enforced by tests;
- [ ] GC/complexity match quality is explicit;
- [ ] primary GC analysis uses exact matches;
- [ ] historical tables cannot be emitted by the default publication path;
- [ ] publication generation fails closed without admissible empirical evidence;
- [ ] dataset overlap is represented in machine-readable provenance;
- [ ] manuscript no longer claims sequence-identity disjointness;
- [ ] `no_exact_overlap` is not confused with genome-disjoint evaluation;
- [ ] all publication artifacts identify dataset/checkpoint/code provenance;
- [ ] corrected motif analysis has been rerun;
- [ ] downstream motif analyses use the corrected validated motif set;
- [ ] final C06 evidence selection is explicit;
- [ ] manuscript tables/figures are regenerated from empirical artifacts;
- [ ] manuscript reconciliation report contains no unexplained discrepancies;
- [ ] final publication package is associated with an immutable Git revision/tag.

---

# 20. Recommended implementation milestones

## Milestone A — Scientific blockers

```text
S10 Exact paired perturbation
S11 Strict matched controls
S14 Regression tests
```

**Output:** corrected motif framework.

---

## Milestone B — Publication safety

```text
S12 Strict publication by default
S19 Historical reporting isolation
S15 Artifact provenance
```

**Output:** impossible to accidentally publish synthetic/historical constants.

---

## Milestone C — Scientific scope

```text
S13 Dataset overlap contract
S18 Architecture identity
```

**Output:** code, documentation and manuscript use the same scientific terminology.

---

## Milestone D — Defensive hardening

```text
S16 Calibration validation
S17 Scheduler API
```

**Output:** fewer silent failure modes in future experiments.

---

## Milestone E — Final reproduction

```text
S20 Final empirical reproduction
S21 Manuscript reconciliation
```

**Output:** frozen evidence package for article and dissertation.

---

# 21. Final priority recommendation

If time is constrained, the **minimum required implementation** is:

```text
1. S10 — exact same-offset motif perturbation
2. S14 — regression test proving it
3. S11 — exact GC control + match-quality provenance
4. S12 — empirical reporting strict by default
5. S13 — dataset overlap/manuscript contract
6. rerun affected motif analyses
7. regenerate publication package
8. S21 — final reconciliation
```

Everything else can be implemented after the manuscript freeze if necessary.

The central rule for this improvement cycle should be:

> **Do not change the scientific question or retrain working models unless necessary. Make the existing experiments reproducible, auditable, correctly controlled, and impossible to overinterpret.**