# C09 checks

Profile: light
Plan: `.specs/features/c09-latent-resume-validated/plan.md` (aprovado pelo usuário: “prossiga”).

18 checks em duas slices; três portas (terceira descoberta no build e registrada antes da implementação); zero questões bloqueantes.

## Checks

### S1 — seleção e embeddings

**C1** — O notebook usa reuse por padrão e rejeita outro modo antes de extrair (AC 1–2).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_modes_and_sources -k test_modes_and_sources`

**C2** — Rerun ignora caches, reserva saída nova e executa o cálculo (AC 3).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_rerun_and_isolation -k test_rerun_and_isolation`

**C3** — Busca automática ordena runs por recência e origem explícita restringe a busca; origem inválida/ausente falha (AC 4–6).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_modes_and_sources -k test_modes_and_sources`

**C4** — Reutilização rejeita alterações nos campos de identidade científica e no checksum/fingerprint (AC 7).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_embedding_identity_and_integrity -k test_embedding_identity_and_integrity`

**C5** — Par incompleto, contagem ou dimensão inválidas são ignorados, permitindo fallback para origem anterior (AC 7–8).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_embedding_fallback_and_shape -k test_embedding_fallback_and_shape`

**C6** — Embedding válido é carregado sem extração, por orçamento RAM/spill (AC 9).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_embedding_budget_and_no_extraction -k test_embedding_budget_and_no_extraction`

**C7** — Reuso e rerun deixam as fontes anteriores byte a byte inalteradas (AC 10, 18; door 1).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_rerun_and_isolation -k test_rerun_and_isolation`

### S2 — progresso e proveniência

**C8** — Checkpoints só são admitidos quando payload e arquivos vinculados têm SHA-256 correto; publicação interrompida não cria cache concluído (AC 11, 17; door 2).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_stage_atomic_and_integrity -k test_stage_atomic_and_integrity`

**C9** — Etapas completas são restauradas sem cálculo; corrupção recalcula a unidade dependente preservando outra válida (AC 12–13).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_stage_dependency_invalidation -k test_stage_dependency_invalidation`

**C10** — Sonda restaurada verifica hash dos pesos e seleção train/validation; teste sobreposto é recusado (AC 14).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_sealed_probe_roundtrip -k test_sealed_probe_roundtrip`

**C11** — Predições nativas ausentes são inferidas no backbone com conferência de IDs/rótulos do teste (AC 15).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_native_predictions_identity -k test_native_predictions_identity`

**C12** — Arquivos legados sem registro de etapa são recusados como conclusão, com motivo registrado (AC 16).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_legacy_stage_not_admitted -k test_legacy_stage_not_admitted`

**C13** — Dois runs gravam em namespaces separados, reutilizando a mesma origem somente para leitura (AC 18).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_rerun_and_isolation -k test_rerun_and_isolation`

**C14** — Smoke executa notebook, reusa embeddings/sonda/predições/KNN/Silhouette/auditoria/UMAP sem seus cálculos e registra modo, origens, descartes e hashes no manifesto (AC 12, 19).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.NotebookResumeTests.test_smoke_resume_and_manifest -k test_smoke_resume_and_manifest`

**C15** — A documentação contém reuse/rerun/origem, limites legados e separa validação sintética de reprodução empírica (AC 20).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.NotebookResumeTests.test_documented_limits -k test_documented_limits`

**C16** — Falha após publicar o payload e antes do marcador obriga recálculo; erro de permissão ao carregar embedding é propagado (AC 17).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_stage_interrupted_manifest_and_operational_failure -k test_stage_interrupted_manifest_and_operational_failure`

**C17** — Origem explícita contendo referências reutilizadas restaura os arquivos transitivos e a restauração da sonda preserva o RNG global (AC 5, 14).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_explicit_transitive_reuse_and_probe_initialization -k test_explicit_transitive_reuse_and_probe_initialization`

**C18** — A inicialização identificada `c09_seeded_probe_init_v1` fixa `ctx.seed` antes da construção da sonda e usa namespace `c09-` (door 3).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_c09_latent_resume.ResumeTests.test_initialization_protocol_is_explicit -k test_initialization_protocol_is_explicit`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Modos (3) | reuse C1 · rerun C2 · inválido C1 | - |
| Origens (4) | automática C3 · explícita C3 · ausente C3 · inválida C3 | - |
| Estados de retomada (4) | reutilizado C9 · descartado C5 · recalculado C9 · erro operacional C8 | - |
| Identidades (13) | variante C4 · split C4 · partições C4 · hashes-partições C4 · checkpoint C4 · hash-checkpoint C4 · classes C4 · codificação C4 · amostragem C4 · modo C4 · extrator C4 · contagem C5 · dimensão C5 | - |
| Componentes (7) | embedding C6 · sonda C10 · predições C11 · KNN C14 · Silhouette C14 · avaliação-sonda C14 · UMAP C14 | - |
| Portas (3) | namespace-leitura C7 · progresso-versionado C8 · inicialização-sonda C18 | - |
| Entidades (4) | run C13 · embedding C4 · sonda C10 · etapa C9 | - |
| Inicialização (1) | primeira-célula-notebook C14 | - |

## Swept

- validation: C1, C3–C5
- failure modes: C5, C8
- idempotency: C6, C9
- authorization: n/a — Python local sem serviço ou credenciais novas
- concurrency: C7, C13
- data lifecycle: C7, C12
- dependency failure: C8, C11; erros operacionais propagados
- state transitions: C8–C10
- observability: C12, C14, C15

## Handoff

`wc -c` dos arquivos existentes relevantes = 145279 bytes / 4 = 36320 tokens; novo módulo/testes/docs estimados em 45000 bytes / 4 = 11250 tokens; total ~47570 < orçamento padrão 150000. Mecanismo: one builder, seguido de Verifier independente conforme a skill. Nenhuma delegação de implementação.

- **Boundary:** C1–C18 implementados; 17 testes C09 verdes, incluindo smoke em duas execuções, 78 testes da bateria latente verdes (antes do teste negativo adicional de manifesto) e 9 testes notebook 06 verdes. Verificação independente pendente do commit.
- **Settled mid-build:** identificação da inicialização da sonda via door 3; mudanças nos dois seletores de reutilização não invalidam o hash do código científico. Um teste negativo adicional precisou restringir a origem explicitamente para não selecionar o cache válido criado pela iteração anterior; nenhuma asserção ou prova C1–C18 foi enfraquecida.
- **Abandoned:** não houve exclusão de artefatos históricos nem alteração de `eval/latent.py`. A bateria ampliada N04 revelou duas falhas e um erro do notebook 05 por manifesto já existente, fora do escopo; os nove testes de notebook 06 passaram após manter compatibilidade e antecipar a checagem de checkpoints ausentes.
