# C00 — Verification

Profile: light

## Execução local

- `.venv/bin/python -m unittest tests.test_c00_preflight_manifest tests.test_notebook_runtime tests.test_manuscript_compilation` — 17 testes passaram.
- `.venv/bin/python scripts/preflight_notebook.py short/01 --data-root /tmp/c00-no-inputs` — reportou `BLOCKED` para Parquet e checkpoint ausentes e saiu com código 2; não criou produtos científicos.
- Ambiente medido em 2026-11-03: Python 3.10.12, PyTorch 2.14.1+cu130, NumPy 2.2.6, pandas 2.3.3, pyarrow 25.0.1; Dask/cuDF indisponíveis. Driver local antigo não inicializa CUDA. Nenhum treinamento ou download foi iniciado.
- O teste de compilação passou direcionando a saída para temporário. Hash pré/pós do PDF não foi calculado nesta sessão; arquivo versionado preexistente permaneceu fora dos caminhos de escrita da compilação.

## Execução empírica remota

Pendente: JupyterHub e volume remoto não estão conectados nesta sessão. Um status `READY` do preflight atesta somente presença, leitura e metadados dos pré-requisitos declarados.

## Limitações e divergências

- Os notebooks existentes ainda têm caminhos fixos e manifestos próprios; a reserva e o formato comum foram adicionados como APIs do runtime, mas a integração de saída de todos os 15 notebooks ainda requer migração cuidadosa dos consumidores de artefatos entre notebooks. Portanto, AC 6–14 estão parcialmente implementados na infraestrutura e não certificados ponta a ponta.
- A inspeção do long/03 revelou que o notebook procura arquivos por cinco nomes derivados de `model_name.split()[0]`, o que causa colisão de nomes entre variantes; o preflight requer nomes distintos e falta preencher o contrato no notebook antes de executar a Tabela 12. O manifesto histórico do long/02 registra somente um conjunto ResNet50, não prova os resultados por variante.
- A lista declarativa é uma primeira cobertura operacional, não substitui auditoria de cada ramo condicional, checkpoints por variante ou todos os requisitos preparados do short/09 e long/03.
- Não houve hash retroativo nem promoção de manifestos históricos.
- `paper/manuscript_submission.pdf` já estava modificado antes desta implementação e foi preservado.
