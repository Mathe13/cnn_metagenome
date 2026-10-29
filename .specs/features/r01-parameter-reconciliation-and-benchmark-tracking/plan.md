# r01-parameter-reconciliation-and-benchmark-tracking — Plan

## Problem

1. **Discrepância na contagem de parâmetros entre arquiteturas 1D reais e tabelas do manuscrito:** `src/viral_cnn/training/profiler.py` (`profile_inference_model`) retorna apenas `total_params` e `trainable_params`, sem decompor `backbone_params` e `head_params`. Enquanto o manuscrito (`paper/manuscript_submission.tex`) e `src/viral_cnn/reporting/tables.py` reportam `23.5M` (ResNet50) e `28.1M` (ConvNeXt) — números copiados das versões 2D do ImageNet-1K —, as arquiteturas 1D reais (`ResNet50_1D` e `ConvNeXt_Tiny_1D` em `src/viral_cnn/models/`) possuem dimensão pré-classificador $D = 2.048$, resultando em:
   - `ResNet50_1D`: `backbone_params = 15,955,584`, `head_params = 5,536,398` (Genus, $2.702$ classes) / `28,036,467` (Species, $13.683$ classes), `total_params = 21,491,982` (Genus) / `43,992,051` (Species).
   - `ConvNeXt_Tiny_1D` (`HybridConvNeXt`): `backbone_params = 135,977,344`, `head_params = 5,536,398` (Genus) / `28,036,467` (Species), `total_params = 141,513,742` (Genus) / `164,013,811` (Species).
2. **Saídas ausentes e falta de guarda `fail-fast` no Notebook `01`:** Em `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb`, as células de profiling (Célula 3), avaliação limpa e com ruído (Célula 4) e faixas de F1 (Célula 5) salvam os CSVs sem imprimir os `DataFrame`s completos (`df_prof`, `df_summary`, `df_f1_bands`) no stdout da célula; a Célula 4 descarta os vetores de predição de $\epsilon = 0.10$ necessários para a comparação especialista da Nota 45; a Célula 7 grava `"checkpoints": {}` no `manifest.json`; e não há validação explícita de que os 7 checkpoints existem no disco e de que `n_samples == 1,706,270` (Species) e `816,756` (Genus) quando `ctx.is_smoke_test == False`.
3. **Bloqueio de artefatos leves pelo `.gitignore`:** A linha `results_reproduction/` em `.gitignore` impede que o Git rastreie novos arquivos `.csv`, `.json`, `.yaml` e `.png` gerados no cluster, mantendo `results_reproduction/n02_benchmark/*` congelado nos resultados antigos de smoke test (`16` amostras, `is_smoke_test: true`).

Quando esta especificação for concluída, `profile_inference_model` decomporá `backbone_params` e `head_params` exatos, o notebook `01` validará estritamente os 7 checkpoints e contagens de amostras fora do modo smoke, imprimirá todas as tabelas (incluindo a comparação especialista em $\epsilon = 0.05$ e $\epsilon = 0.10$) e metadados dos checkpoints no manifesto, e o `.gitignore` bloqueará apenas binários volumosos (`*.pt`, `*.parquet`, `*.sqlite*`).

## Flow

Reutiliza `build_registered_model`, `evaluate_dataloader`, `compute_species_f1_bands`, `compute_species_specialist_comparison` e `write_notebook_manifest` já existentes em `src/viral_cnn/`.

1. `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb` (existing) -> valida existência dos 7 checkpoints no disco quando `ctx.is_smoke_test == False` (door 2) e instancia cada variante via `src/viral_cnn/models/registry.py` (`build_registered_model`, existing).
2. `src/viral_cnn/training/profiler.py` (`profile_inference_model`, existing) -> calcula `backbone_params`, `head_params`, `total_params` e métricas de latência/FLOPs (door 1) e retorna dicionário consolidado em `df_prof`.
3. `src/viral_cnn/eval/runner.py` (`evaluate_dataloader`, existing) -> executa avaliação em $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$, preserva vetores de `resnet_noise_species` e `convnext_noise_species` em $\epsilon \in \{0.05, 0.10\}$, e valida `n_samples` reais quando `ctx.is_smoke_test == False` (door 2).
4. out: tabelas `df_prof`, `df_summary`, `df_f1_bands` e comparações especialistas ($\epsilon = 0.05$ e $\epsilon = 0.10$) impressas no output das células e salvas em `results_reproduction/n02_benchmark/` com rastreamento Git seletivo em `.gitignore` (door 3).

## Impact

| Front | What changes |
| --- | --- |
| domain | `profile_inference_model` em `src/viral_cnn/training/profiler.py` passa a incluir as chaves inteiras `backbone_params` e `head_params` além das 7 chaves já existentes |
| stored data | `.gitignore` substitui o bloqueio global `results_reproduction/` por filtros de extensão binária (`*.pt`, `*.pth`, `*.parquet`, `*.sqlite*`, `*.db`), permitindo versionar CSVs, JSONs, YAMLs e PNGs |
| notebook outputs | `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb` imprime `df_prof`, `df_summary`, `df_f1_bands`, a comparação especialista em $\epsilon = 0.05$ e $\epsilon = 0.10$, e popula `checkpoints` e `input_files` em `manifest.json` |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Door 1: Decomposição de parâmetros de backbone e cabeça classificadora em `profile_inference_model` | `head_params = sum(p.numel() for p in model.fc.parameters()) if hasattr(model, "fc") and isinstance(model.fc, nn.Module) else 0` e `backbone_params = int(total_params - head_params)` retornados no dicionário de `profile_inference_model` | Manter apenas `total_params` (impede distinguir o tamanho do extrator de características 1D do peso da camada `fc` entre Genus e Species nas Tabelas 1 e 2) |
| Door 2: Guarda `fail-fast` de artefatos e cardinalidade no notebook `01` quando `not ctx.is_smoke_test` | Verificação explícita `if not ckpt or not Path(ckpt).exists(): raise FileNotFoundError(...)` para os 7 modelos e `if res.metrics.n_samples != expected_samples: raise RuntimeError(...)` ($1.706.270$ em Species e $816.756$ em Genus) | Permitir fallback silencioso para `checkpoint_path=None` ou partições incompletas quando `is_smoke_test == False` (viola o princípio inegociável 1 de `AGENTS.md`) |
| Door 3: Regra seletiva para `results_reproduction/` no `.gitignore` | Substituir `results_reproduction/` por `results_reproduction/**/*.pt`, `results_reproduction/**/*.pth`, `results_reproduction/**/*.parquet`, `results_reproduction/**/*.sqlite*`, `results_reproduction/**/*.db` | Ignorar todo o diretório `results_reproduction/` (bloqueia o commit automático dos CSVs e manifestos JSON regenerados no cluster) |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Decomposição de parâmetros em `profile_inference_model` e rastreamento seletivo no `.gitignore` (P1)

Decompõe parâmetros de backbone e camada classificadora `fc` no profiler e libera o rastreamento Git de artefatos tabulares/JSON em `results_reproduction/`.

**Acceptance Criteria**

1. WHEN `profile_inference_model` is called on `ResNet50_1D(num_classes=13683)` and `ResNet50_1D(num_classes=2702)` THEN the function SHALL return `backbone_params == 15955584`, `head_params == 28036467` and `total_params == 43992051` for Species, and `backbone_params == 15955584`, `head_params == 5536398` and `total_params == 21491982` for Genus.
2. WHEN `profile_inference_model` is called on `ConvNeXt_Tiny_1D(num_classes=13683)` and `ConvNeXt_Tiny_1D(num_classes=2702)` THEN the function SHALL return `backbone_params == 135977344`, `head_params == 28036467` and `total_params == 164013811` for Species, and `backbone_params == 135977344`, `head_params == 5536398` and `total_params == 141513742` for Genus.
3. WHEN `git check-ignore` evaluates paths under `results_reproduction/` THEN `.gitignore` SHALL not ignore `.csv`, `.json`, `.yaml`, or `.png` files (including `results_reproduction/n02_benchmark/benchmark_clean_noise_summary.csv`) and SHALL ignore `.pt`, `.parquet`, and `.sqlite` files (including `results_reproduction/checkpoints/resnet50_noise/finetune_noise_last.pt`).

**Independent test:** Executar testes unitários sobre `profile_inference_model` para as 4 combinações de arquitetura/rank e `git check-ignore` sobre caminhos tabulares vs. binários.

### S2: Guardas `fail-fast`, visibilidade de tabelas, comparação especialista em $\epsilon \in \{0.05, 0.10\}$ e manifesto no Notebook `01` (P1)

Garante falha imediata na ausência de checkpoints/amostras reais fora do modo smoke e imprime todas as tabelas e comparações especialistas na saída das células do notebook `01`.

**Acceptance Criteria**

4. IF `ctx.is_smoke_test` is `False` and any of the 7 registered model checkpoint files does not exist on disk THEN `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb` SHALL raise `FileNotFoundError` before running model profiling or evaluation.
5. IF `ctx.is_smoke_test` is `False` and `evaluate_dataloader` returns `n_samples != 1706270` for any species model or `n_samples != 816756` for any genus model THEN `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb` SHALL raise `RuntimeError`.
6. WHEN `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb` executes cells 3, 4, 5, 6, and 7 THEN the notebook SHALL print `df_prof.to_string()`, `df_summary.to_string()`, `df_f1_bands.to_string()`, and the specialist comparison summaries for both $\epsilon = 0.05$ (`specialist_comparison.csv`) and $\epsilon = 0.10$ (`specialist_comparison_eps010.csv`), and SHALL populate `checkpoints` in `manifest.json` with all 7 evaluated model variant keys.

**Independent test:** Executar a suíte de testes validando a execução do notebook `01` em modo smoke (capturando stdout e artefatos gerados) e o disparo de `FileNotFoundError` / `RuntimeError` quando `is_smoke_test=False` e um checkpoint ou contagem de amostras é inválido.

## Out of scope

| Excluded | Why |
| --- | --- |
| Atualização dos valores de parâmetros e tabelas diretamente em `paper/manuscript_submission.tex` e `src/viral_cnn/reporting/tables.py` | Escopo exclusivo da spec `R08` (`r08-dynamic-paper-tables-and-manuscript-reconciliation`) após consolidação de `R01–R07` |
| Modificação de notebooks históricos em `short/` ou `long/` | Proibido por `AGENTS.md` e `AD-004` |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Identificação da camada classificadora em `profile_inference_model` | Inspecionar `model.fc` (`isinstance(model.fc, nn.Module)`) para somar `head_params`, definindo `backbone_params = total_params - head_params` | Tanto `ResNet50_1D` quanto `ConvNeXt_Tiny_1D` definem a cabeça classificadora no atributo `self.fc` | y |
| Persistência da comparação especialista em $\epsilon = 0.10$ | Manter `specialist_comparison.csv` para $\epsilon = 0.05$ (compatibilidade com testes existentes) e salvar `specialist_comparison_eps010.csv` para $\epsilon = 0.10$, imprimindo ambos no stdout da Célula 6 | Preserva o contrato de `specialist_comparison.csv` (Tabela 4) enquanto fornece os dados exatos da Nota 45 ($\epsilon = 0.10$) | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- `ROADMAP_CORRECAO_AUDITORIA.md` (Seção 3, `R01 — r01-parameter-reconciliation-and-benchmark-tracking`) - define as contagens exatas de parâmetros 1D, regras de `.gitignore` e requisitos de visibilidade e `fail-fast` do notebook `01`.
- `src/viral_cnn/models/resnet.py`, `src/viral_cnn/models/convnext.py`, `src/viral_cnn/training/profiler.py` - implementação canônica das arquiteturas 1D e do profiler de inferência.

