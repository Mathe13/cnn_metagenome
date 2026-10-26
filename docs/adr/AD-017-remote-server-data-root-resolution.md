# AD-017: Resolução Automática de Topologia do Servidor Remoto via REMOTE_SERVER e DATA_ROOT

## Status
Aceito (Accepted) — 2026-10-26

## Contexto
O estudo viral CNN apoia-se em dados de sequências metagenômicas de larga escala:
- Partições Parquet disjuntas com 256 arquivos por split ($N = 1.706.270$ fragmentos de 512 bp para espécies e gêneros).
- Checkpoints de modelos treinados com pesos completos: ResNet50 (82,3 MB) e ConvNeXt 1D (625,7 MB).

No ambiente de pesquisa do JupyterHub remoto (servidor da universidade/laboratório), esses dados e pesos residem no diretório do workspace:
```text
/home/mlima/workspace/
├── cnn_kraken/                               # Parquets e checkpoints ConvNeXt
│   ├── processed_kraken_species_splited_parquet/
│   ├── processed_kraken_genus_splited_parquet/
│   ├── checkpoint_species_conv_zero/
│   ├── checkpoint_transfer_species_conv/
│   └── checkpoint_finetune_resnet/
├── cnn_kraken_species/                       # Checkpoints ResNet50 espécie
├── cnn_kraken_genus/                         # Checkpoints ResNet50 gênero
├── dataset_kraken/                           # Mapeamentos taxonômicos
└── cnn_metagenome/                           # Repositório Git clonado
    └── notebooks/reproduction/short/         # Notebooks de reprodução
```

Em contrapartida, em ambientes de desenvolvimento local, CI e sandbox offline, o repositório opera com fixtures sintéticos pequenos sem necessidade de arquivos gigabyte ou GPUs (AD-005, AD-006).

A parametrização manual de caminhos relativos (como `"../"`) a partir de notebooks em subpastas profundas (`notebooks/reproduction/short/`) gerava divergência de caminhos, carregamento involuntário de fixtures sintéticas no servidor e omissão dos checkpoints treinados.

## Decisão

1. **Variável de Ambiente `REMOTE_SERVER` (ou `remote_server`):**
   - **Padrão no Servidor (`True`):** A menos que explicitamente definida como `'False'`, `'0'`, `'no'` ou `'local'`, o runtime assume que está no servidor de pesquisa remoto.
   - **Resolução Automática do `DATA_ROOT`:**
     - Quando no servidor e `DATA_ROOT` não estiver definido, o runtime detecta o repositório (`project_root`) e verifica se o diretório pai (`project_root.parent`) contém `cnn_kraken/`. Se positivo, adota `/home/mlima/workspace` (ou o diretório pai correspondente) como `DATA_ROOT`.
     - Permite sobreposição explícita com caminhos customizados de cluster (ex: `/data/metagenome`).
   - **Desativação Padrão do Smoke Test:** No servidor remoto, `SMOKE_TEST` assume `0` por padrão, executando a inferência real nas 256 partições.

2. **Isolamento Completo dos Testes Locais:**
   - Em testes unitários locais (`tests/__init__.py` e `tests/conftest.py`), `REMOTE_SERVER` é sempre configurado como `False` e `SMOKE_TEST` como `1`.
   - Garante que a suíte de testes nunca tente acessar caminhos remotos inexistentes, executando de forma ultrarrápida (segundos) e determinística.

3. **Carregamento Mandatório de Checkpoints:**
   - Nos notebooks de reprodução (`notebooks/reproduction/short/` e `long/`), as chamadas de instanciação de modelos devem passar obrigatoriamente:
     ```python
     ckpt_path = ctx.topology.paths["checkpoints"].get(name) if not ctx.is_smoke_test else None
     model = build_registered_model(name, checkpoint_path=ckpt_path, device=ctx.device)
     ```
   - Em modo real (`is_smoke_test=False`), os pesos de 625 MB / 82 MB são carregados e validados contra as dimensões da camada classificadora (13.683 classes para espécies, 2.702 para gênero).

4. **Documentação e Arquivos de Configuração:**
   - Criação dos arquivos `.env.example` e `env.example` exemplificando as opções de configuração (`REMOTE_SERVER`, `DATA_ROOT`, `SMOKE_TEST`, `LOG_PATH`).

## Consequências
- Os notebooks executam de forma transparente e zero-configuração tanto no servidor remoto conectado via JupyterHub quanto em máquinas de desenvolvimento local.
- Erros de caminho e execução com pesos aleatórios não treinados são completamente prevenidos.

