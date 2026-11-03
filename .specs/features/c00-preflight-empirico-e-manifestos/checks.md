# C00 — Checks

Profile: light
Plan: `.specs/features/c00-preflight-empirico-e-manifestos/plan.md`

## Checks

- C1: CLI lista/valida todos os 15 IDs e retorna BLOCKED com entradas ausentes; metadata Parquet é lida sem iteração de batches.
- C2: falha de reserva em diretório já existente não altera arquivos existentes; diretórios distintos por notebook permanecem independentes.
- C3: gate rejeita legado, incompleto e `is_smoke_test=true`; aceita apenas manifesto schema 1 completo com evidência observada.
- C4: manifesto registra commit, estado dirty, configuração/comando, modo, semente, versões, identidades observadas/unknown e saídas; nunca sobrescreve destino.
- C5: a compilação do manuscrito e o teste existente direcionam PDFs/auxiliares a diretório temporário.
- C6: ambiente reporta versões instaladas ou `unavailable` para PyTorch/CUDA, NumPy, pandas, pyarrow, Dask/cuDF.
- C7: verificação final preserva hash do PDF versionado e distingue inspeção local/sintética de execução empírica remota.

## Casos negativos

- Requisitos faltantes ou colunas Parquet incompatíveis → `BLOCKED` enumerando caminho e motivo.
- Reutilização concorrente/sequencial do mesmo namespace → `FileExistsError`, arquivos anteriores intactos.
- Manifesto sintético/legacy sem identidade → gate rejeita com nomes dos campos.
- Compilação não escreve em `paper/`.

## Limites

Preflight não verifica contagens científicas, conteúdo de pesos nem lê linhas. Entrada identificada por hash é observada apenas quando medida. Inspeção de entradas reais não equivale à execução empírica no JupyterHub.
