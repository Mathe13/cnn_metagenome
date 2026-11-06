# LESSONS - auto-maintained by scripts/lessons.py

> Machine-owned. Do NOT hand-edit. Changes are overwritten on the next `lessons.py` write.
> Canonical state lives in `.specs/lessons.json`. Edit lessons only via the script.
> promote_threshold=2 distinct features · window_days=45 · quarantine_threshold=2

## Confirmed (load these at Plan/Checks)

Corroborated across multiple features. Safe to apply as guidance.

_none_

## Candidates (under observation - do NOT load as guidance yet)

Seen once or not yet corroborated. Tracked, not trusted.

### L-001 - Assert every declared lineage field, including input hashes and checkpoint selection evidence.
- signal: `ac_gap` · recurrence: 1 feature(s) · scope: `training` · harmful: 0
- features: c05-finetune-e-comparacao-controlada
- evidence: C2; tests/test_c05_finetune_audit.py:92-101 (training)
- last seen: 2026-11-04T03:12:22Z

### L-002 - Use executable assertions for support distinctions and metric values when a check claims runtime evaluation behavior.
- signal: `spec_precision_gap` · recurrence: 1 feature(s) · scope: `training` · harmful: 0
- features: c05-finetune-e-comparacao-controlada
- evidence: C5; tests/test_c05_finetune_audit.py:187-190 (training)
- last seen: 2026-11-04T03:12:26Z

### L-003 - Treat a completion gate failure as a failed feature verification and route its cited gaps for repair.
- signal: `gate_fail` · recurrence: 1 feature(s) · scope: `verification` · harmful: 0
- features: c05-finetune-e-comparacao-controlada
- evidence: .specs/features/c05-finetune-e-comparacao-controlada/verification.md:3 (verification)
- last seen: 2026-11-04T03:12:29Z

### L-004 - Assert every promised input identity hash individually, not only a representative input.
- signal: `ac_gap` · recurrence: 1 feature(s) · scope: `training-audit` · harmful: 0
- features: c05-finetune-e-comparacao-controlada
- evidence: C2; tests/test_c05_finetune_audit.py:116 (training-audit)
- last seen: 2026-11-04T03:27:15Z

### L-005 - Treat an unproven check as a release-blocking verification failure until its evidence is complete.
- signal: `gate_fail` · recurrence: 1 feature(s) · scope: `training-audit` · harmful: 0
- features: c05-finetune-e-comparacao-controlada
- evidence: C2; validate_verification.py output (training-audit)
- last seen: 2026-11-04T03:27:20Z

### L-006 - Teste parâmetros e falhas de configuração na primeira célula real do notebook, além do default da classe auxiliar, quando o contrato nomear o notebook.
- signal: `ac_gap` · recurrence: 1 feature(s) · scope: `notebooks` · harmful: 0
- features: c09-latent-resume-validated
- evidence: C1; tests/test_c09_latent_resume.py:49; verification.md Resolved finding (notebooks)
- last seen: 2026-11-06T01:01:16Z

### L-007 - Assert preservation of labels and row IDs for every model when validating paired selection input immutability.
- signal: `ac_gap` · recurrence: 1 feature(s) · scope: `latent` · harmful: 0
- features: c10-paired-umap-all-models
- evidence: verification.md C2; tests/test_c10_paired_umap.py:128 (latent)
- last seen: 2026-11-06T01:23:09Z

## Quarantined (failed when applied - ignore)

A confirmed lesson that recurred alongside failure. Kept for the maintainer to review.

_none_
