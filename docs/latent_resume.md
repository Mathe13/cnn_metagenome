# Retomada do notebook 06

Na primeira célula de `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb`:

```python
LATENT_EXECUTION_MODE = os.environ.get("LATENT_EXECUTION_MODE", "reuse")
LATENT_REUSE_RUN_ID = os.environ.get("LATENT_REUSE_RUN_ID") or None
```

`reuse` é o padrão: executar todas as células valida os candidatos anteriores e reutiliza componentes completos. Para restringir a origem, edite a segunda linha para um identificador como `"c01-20261104T161929937502"`, ou defina essa variável de ambiente antes de abrir o kernel. `rerun` recalcula tudo em uma pasta nova. As variáveis de ambiente têm precedência no exemplo padrão; o modo é validado antes da extração. Uma origem explícita inexistente provoca erro; sem origem automática disponível, o notebook calcula normalmente.

Cada execução cria um diretório exclusivo em `results_reproduction/runs`. A busca automática considera somente o notebook 06, da pasta mais recente à mais antiga. Uma origem pode referenciar artefatos reutilizados de outra execução: os vínculos são mantidos no relatório e todos os arquivos referenciados são novamente validados. Preserve as pastas de origem enquanto forem usadas por outras execuções.

Os pares de embeddings Parquet/JSON são conferidos por identidade, fingerprint e SHA-256: variante/split, partições e hashes, checkpoint e hash, ordem de classes, codificação, semente/amostragem, modo, código extrator, contagem e dimensão. Smoke e execução empírica não se misturam. A leitura respeita o orçamento RAM/spill C07. Checar hashes e ler arquivos grandes ainda pode levar tempo; reutilizar não significa execução instantânea.

Os checkpoints `stage_*.json`/`.pt` do protocolo `c09_latent_resume_v1` persistem sonda selada, predições nativas, KNN e Silhouette por variante/nível, avaliação da sonda e UMAP. Os pesos são carregados com `weights_only=True`; o JSON de conclusão só é publicado após os arquivos e seus hashes. Dependências/configuração/código/ambiente divergentes invalidam a unidade; unidades independentes permanecem reutilizáveis. Tabelas e manifesto são reconstruídos na nova pasta.

A identidade das etapas inclui versões instaladas de UMAP, scikit-learn e SciPy, além das versões do runtime. A função histórica de projeção usa PCA quando UMAP não está disponível; instalar/remover UMAP invalida esse cache. A retomada preserva os dois avisos qualitativos no cabeçalho do CSV. Editar apenas os seletores `LATENT_EXECUTION_MODE` e `LATENT_REUSE_RUN_ID` não invalida o hash do código científico.

As execuções novas usam prefixo `c09-`. A sonda usa a inicialização `c09_seeded_probe_init_v1`, com seed 42 antes de construir a cabeça. No notebook anterior o seed era fixado apenas dentro de `fit_select_and_seal`, depois da construção: a inicialização dependia do estado global consumido pelas etapas anteriores. Portanto, uma sonda recalculada pode produzir números diferentes da auditoria histórica; não é uma reprodução exata desses pesos. A alteração é identificada no manifesto e no cache; os backbones, embeddings empíricos, protocolo de seleção por validação e teste selado não mudam.

Embeddings legados válidos podem ser reaproveitados. CSV de UMAP, tabelas, Parquets KNN ou JSON avulsos legados não substituem registros completos de etapa. A sonda histórica não tinha pesos persistidos: seu ajuste pode precisar ser refeito. Predições nativas históricas ausentes exigem nova inferência do backbone com conferência de IDs/rótulos, sem extrair novamente embeddings ou presumir equivalência da cabeça aplicada ao cache.

“Descartar” significa ignorar o candidato e registrar o motivo, preservando o arquivo original. O notebook nunca apaga runs anteriores ou arquivos de outro kernel. Arquivos temporários ou parciais não comprovam conclusão. Erros operacionais de disco/permissão são propagados. Uma etapa interrompida é recalculada desde o início dessa unidade: não há retomada interna de lote, época ou bloco KNN. Execute desde a primeira célula para reservar outra pasta, depois de encerrar a execução anterior caso ainda esteja ativa.

`resume_report.json` é atualizado durante a execução e contém modos, candidatos, descartes, componentes recalculados e caminhos/hashes dos reutilizados. Consulte esse arquivo no JupyterHub quando o stdout antigo não estiver disponível. O manifesto final inclui essas origens como entradas verificáveis; o relatório parcial não equivale ao manifesto final.

As métricas permanecem no espaço original: KNN cosseno k=5 estratificado 80/20, seed 42 e auditoria C08; Silhouette até 5.000 exemplos (200 no smoke). A visualização [C10](paired_umap.md) usa uma amostra pareada para cinco variantes, por padrão até 100 fragmentos/10 gêneros, com limites configuráveis. A documentação histórica de auditoria menciona 50.000 para Silhouette; esta feature mantém os 5.000 implementados. O split é por fragmentos closed-set e não comprova generalização para genomas novos.

Validação desta feature: testes sintéticos CPU e smoke offline com `REMOTE_SERVER=False`, incluindo reexecução com funções de cálculo bloqueadas. Os checkpoints/dados canônicos do cluster não estão disponíveis nesta sessão; a validação empírica remota permanece pendente. Estes testes não constituem reprodução integral do artigo.
