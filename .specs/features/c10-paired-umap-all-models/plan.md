# C10 — Projeções pareadas para os cinco modelos no notebook 06

## Problem

A célula 7 do notebook de reprodução 06 usa apenas `convnext_noise_species` e as primeiras 100 linhas de teste. Não verifica se a mesma seleção de fragmentos pode ser encontrada nas demais variantes, e não gera um painel comparativo. O usuário solicita implementar a comparação dos cinco modelos com os mesmos fragmentos identificados e parâmetros, usando `tlc-spec-lean`.

A inspeção do notebook histórico `short/ICTV_latent_space-bkp.ipynb` mostrou um loop que projeta todos os modelos, com cosseno, 15 vizinhos, min_dist 0,1 e seed 42, seguido de gráficos por Genus e Order. A legenda da Figura 2 em `paper/manuscript_submission.tex` descreve comparação entre arquiteturas colorida por gênero. O notebook de reprodução atual é uma aproximação exploratória reduzida, não uma regeneração integral comprovada dessa figura. Dados/checkpoints canônicos do servidor não estão acessíveis nesta sessão.

## Flow

Reutiliza os embeddings test com identidade operacional do C01, o orçamento RAM/spill C07, a função de projeção existente e os checkpoints de progresso C09; a seleção compartilhada e a apresentação são acrescentadas fora de `eval/latent.py` para preservar os hashes dos embeddings.

1. Notebook 06 (exists) resolve os parâmetros de amostragem e valida seus limites antes de gerar projeções. A extração, sonda, KNN e Silhouette mantêm seus algoritmos/configuração atuais.
2. Notebook 06 (exists) fornece labels, IDs, mapeamento Species→Genus e hashes dos cinco embeddings test à seleção compartilhada (door 1). Somente fragmentos com Genus conhecido são elegíveis; a exclusão fica registrada.
3. A seleção compartilhada (door 1) escolhe até dez gêneros e até 100 fragmentos, uma vez, com seed 42 e quotas equilibradas sem reposição. A ordenação canônica dos IDs elimina dependência da ordem de armazenamento. Os mesmos IDs são localizados nos outros quatro modelos, validando rótulos; inconsistências interrompem a comparação antes da primeira projeção.
4. Notebook 06 (exists) ajusta UMAP uma vez por variante quando importável e reutiliza o fallback PCA de `export_umap_projection` em `viral_cnn.eval.latent` (exists) quando não importável, somente sobre os vetores selecionados do espaço original. Mantém os parâmetros existentes, ajustando de forma comum o número de vizinhos se a amostra for menor que 16; registra o backend efetivamente disponível. Evidência no build: a função histórica também captura ImportError durante o ajuste, sem informar o backend no retorno; o caminho UMAP da nova camada propaga falhas de ajuste em vez de poder rotular silenciosamente um fallback como UMAP.
5. `LatentResumeCache.run_stage` em `viral_cnn.latent_resume` (exists) valida ou publica o progresso de cada variante, vinculado à seleção, parâmetros, backend e embedding correspondente. Um resultado C09 com primeiras 100 linhas não substitui o novo protocolo pareado.
6. Notebook 06 (exists) salva cinco CSVs identificados, a seleção compartilhada, metadados da comparação e um painel PNG com cores e legenda comuns por gênero (door 2). O manifesto e a documentação registram entradas, configurações, hashes e limites interpretativos.

## Impact

| Front | What changes |
| --- | --- |
| projeções | Passam de uma variante/primeiras 100 linhas para cinco variantes sobre uma seleção pareada e estratificada por gênero. Os valores 2D de ConvNeXt_F podem mudar devido à nova amostra; o novo protocolo é `c10_paired_umap_v1`. |
| gráfico | Acrescenta um painel qualitativo com cinco subplots, título por variante, cores comuns de Genus e aviso de que os eixos são ajustados independentemente. Não mede superioridade nem distância filogenética. |
| população | A amostra padrão cobre no máximo dez gêneros, com até 100 fragmentos no total; não representa a taxonomia completa. Contagens disponíveis, elegíveis, selecionadas e excluídas são registradas. |
| recursos | Somente as features selecionadas são copiadas para a projeção; não há conversão integral dos embeddings em arrays novos. A seleção pode percorrer labels/IDs e precisa validar sua integridade. |
| retomada | Mantém reuse/rerun, origem explícita, fontes somente para leitura e cache por variante. O mecanismo C09 usa hash global do código do notebook e dos módulos: a primeira execução após esta alteração pode recalcular sonda/predições/KNN/Silhouette por incompatibilidade de identidade, mesmo sem mudança de algoritmo. Embeddings históricos válidos continuam reutilizáveis porque `eval/latent.py` não será alterado. Não será enfraquecida a validação para evitar esse custo. |
| histórico | Notebooks históricos, manuscrito e saídas de runs anteriores ficam preservados. O novo painel é exploratório C10 e não será apresentado como reprodução exata da Figura 2 publicada. |

## Relations

Uma comparação C10 referencia cinco artefatos test identificados, uma seleção compartilhada de IDs e uma configuração de projeção. Cada ID selecionado ocorre exatamente uma vez em cada modelo, com o mesmo rótulo Species e Genus. Cada variante possui uma projeção independente cujas linhas seguem a mesma ordem de IDs da seleção. O painel compartilha a legenda por gênero, mas não um espaço geométrico alinhado. Checkpoints de progresso e manifesto vinculam os arquivos ao protocolo, à seleção e aos embeddings.

## Surface

None - sem nova rota HTTP ou serviço; a interface é a primeira célula, os arquivos e a saída gráfica do notebook.

Primeira célula: `LATENT_UMAP_MAX_SAMPLES=100` e `LATENT_UMAP_MAX_GENERA=10`, editáveis e com override de ambiente, seed de `ctx.seed` (42). Resultados possíveis: configuração válida ou `ValueError` para formato/limites inválidos. Os parâmetros comuns de projeção ficam explícitos: `n_neighbors=15`, `min_dist=0.1`, `n_components=2`, `metric="cosine"`.

Comparação: recebe embeddings/labels/IDs e taxonomia já validados. Resultados possíveis: cinco projeções completas; erro de amostra insuficiente; erro de IDs/rótulos; erro operacional de leitura/escrita. A retomada conserva os estados existentes reutilizado, descartado e recalculado.

Falhas de dependência durante o ajuste UMAP são propagadas sem publicar comparação completa. Quando IPython não está instalado, o notebook salva o painel e registra seu caminho; a exibição inline é usada no Jupyter.

Saídas: cinco CSVs de projeção, `umap_sample_selection.csv`, `umap_comparison.json` e `umap_comparison.png`, todos no run novo. O nome `umap_projection_convnext_f.csv` permanece para ConvNeXt_F; as quatro variantes adicionais usam `umap_projection_<variant>.csv`. O CSV contém coordenadas, ID operacional, rótulo Species e gênero, além de identificação de método e caráter qualitativo.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — população pareada identificável | `c10_paired_umap_v1`: selecionar `min(10, gêneros elegíveis, MAX_SAMPLES)` gêneros com seed 42 a partir de nomes ordenados; selecionar até `MAX_SAMPLES=100` fragmentos em rodadas de quotas por gênero, sem reposição e com IDs canônicos; registrar IDs/labels/Genus em `umap_sample_selection.csv` e hashes no manifesto. Validar os mesmos IDs/labels nos cinco modelos antes do ajuste. | Primeiras linhas por variante: depende da ordem dos arquivos e pode comparar fragmentos diferentes. Incluir um fragmento por cada uma de 100 classes: reduz a utilidade do painel para observar agrupamentos locais. |
| 2 — saídas qualitativas e cache por variante | Cinco CSVs mais `umap_comparison.json` e painel `umap_comparison.png`; cada projeção tem progresso separado com identidade de seleção/embedding/parâmetros/backend. `projection_method` distingue UMAP de PCA fallback; títulos/eixos acompanham o método real. | Misturar PCA e UMAP sob o mesmo rótulo ou usar a projeção antiga de ConvNeXt_F como se tivesse a nova seleção: produziria comparação sem identidade comum verificável. |

Nenhuma dependência nova. Matplotlib já está no ambiente do projeto. A função histórica de projeção e o seu fallback são preservados; a nova camada registra o método e não altera o arquivo que participa dos hashes históricos de extração. O painel não requer browser, rede ou serviço externo.

## Criteria

### S1: seleção comum e projeções das cinco variantes (P1)

**Acceptance Criteria**

1. WHEN o notebook executa sem override THEN o sistema SHALL usar `LATENT_UMAP_MAX_SAMPLES=100` e `LATENT_UMAP_MAX_GENERA=10`, registrando os valores efetivos e seed 42.
2. IF os limites não são inteiros ou MAX_SAMPLES é menor que 3 ou MAX_GENERA é menor que 1 THEN o sistema SHALL lançar `ValueError` antes de ajustar qualquer projeção.
3. WHEN seleciona a população C10 THEN o sistema SHALL excluir somente da visualização as linhas sem Genus conhecido e registrar as contagens disponíveis/elegíveis/excluídas.
4. WHEN seleciona fragmentos THEN o sistema SHALL produzir uma seleção determinística sem reposição, de no máximo MAX_SAMPLES IDs e MAX_GENERA gêneros, equilibrando quotas e redistribuindo vagas de gêneros com suporte insuficiente.
5. WHEN a ordem de armazenamento de um modelo muda mantendo IDs/rótulos THEN o sistema SHALL localizar e projetar os mesmos fragmentos na ordem canônica da seleção compartilhada.
6. IF um ID selecionado está ausente/duplicado ou seu rótulo diverge entre modelos THEN o sistema SHALL interromper a comparação antes da primeira projeção, sem reduzir silenciosamente a população por interseção.
7. WHEN projeta os embeddings THEN o sistema SHALL produzir uma projeção para cada variante `resnet_clean_species`, `resnet_noise_species`, `convnext_zero_species`, `convnext_transfer_species` e `convnext_noise_species` sobre exatamente os mesmos IDs/labels.
8. WHEN ajusta cada projeção THEN o sistema SHALL usar min_dist 0,1, duas dimensões, cosseno, seed 42 e `n_neighbors=min(15, N_selecionado-1)` idênticos para as cinco variantes.
9. IF há menos de três fragmentos elegíveis selecionados THEN o sistema SHALL informar amostra insuficiente sem produzir uma comparação marcada como concluída.
10. WHEN lê features para a projeção THEN o sistema SHALL copiar somente as linhas selecionadas, mantendo KNN/Silhouette no espaço original e seus protocolos/amostras atuais.

**Independent test:** fixtures com gêneros desequilibrados, linhagem ausente, matrizes com ordens diferentes, IDs ausentes/duplicados e rótulos trocados; projeção espiada confere parâmetros e valores das linhas para todas as variantes, sem depender de GPU.

### S2: figura, proveniência e retomada (P1)

**Acceptance Criteria**

11. WHEN salva uma comparação THEN o sistema SHALL persistir cinco CSVs com IDs/labels/Genus e método, a seleção compartilhada e metadados com hashes, configuração e contagens por gênero.
12. WHEN gera o painel THEN o sistema SHALL apresentar as cinco variantes com o mesmo mapa Genus→cor e legenda comum, incluindo aviso qualitativo e de ajustes/eixos independentes.
13. IF UMAP não está disponível e o fallback histórico PCA é usado THEN o sistema SHALL identificar `projection_method="PCA"` nos metadados, CSVs e gráfico, sem declarar execução UMAP bem-sucedida.
14. WHEN uma projeção completa compatível já existe em reuse THEN o sistema SHALL restaurar a variante sem reajustá-la, validando IDs, seleção, embedding, configuração, backend e arquivos vinculados.
15. IF o cache de uma variante está incompleto/corrompido ou incompatível THEN o sistema SHALL recalcular essa projeção sem reaproveitar suas coordenadas antigas e sem sobrescrever a origem.
16. WHEN uma escrita falha THEN o sistema SHALL impedir a publicação de um registro completo para a comparação incompleta, permitindo retomada posterior em run novo.
17. WHEN a tarefa termina THEN a documentação SHALL distinguir a seleção histórica de primeiras 100 linhas da seleção C10, explicar o possível recálculo de etapas C09 pelo hash global e informar os limites da validação sintética CPU.

**Independent test:** smoke offline gera cinco CSVs e painel, repete em reuse com o ajuste de projeção bloqueado e adultera um arquivo de variante para verificar recálculo isolado; valida metadados, backend e preservação das fontes. Verificador independente no perfil light após o último commit da implementação.

## Out of scope

| Excluded | Why |
| --- | --- |
| Alinhamento geométrico entre modelos e comparação quantitativa de coordenadas 2D | Projeções são ajustadas independentemente; exigiria outro método/experimento. |
| Projetar todos os fragmentos de teste por padrão | O pedido é ampliar as variantes; preservar orçamento pequeno evita novo processamento massivo sem necessidade. |
| Migrar caches C09 antigos ignorando hashes ou refazer o sistema geral de invalidação | O hash global é conservador; sua substituição/migração exige evidência própria para preservar proveniência de todas as etapas. |
| Retreinar backbones, mudar métricas, split, taxonomia ou a amostra de KNN/Silhouette | Alterariam os resultados quantitativos e excedem o ajuste de visualização. |
| Alterar figuras/manuscrito publicados ou notebooks históricos | C10 é uma comparação exploratória nova, com entradas ainda não avaliadas no servidor. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Perfil | light | Padrão da skill, sem declaração diferente no AGENTS.md. | n |
| Nível de cor/estratificação | Genus | A legenda da Figura 2 e o fluxo histórico usam gênero para comparação entre arquiteturas; o limite de dez gêneros mantém a legenda legível. | n |
| Formato gráfico | PNG estático exibido pelo notebook | Usa a biblioteca existente, permite comparação dos cinco painéis e exportação independente de navegador. | n |

**Open questions:** none — defaults e custo de transição do cache estão explícitos para revisão.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Primeira célula | parâmetros, defaults e limites | Criteria 1–2 |
| Seleção compartilhada | exclusões, quotas, determinismo e duplicatas | Criteria 3–6, 9 |
| Cinco projeções | variantes, parâmetros e acesso por linhas | Criteria 7–10 |
| Painel do notebook | cores, legenda, títulos e avisos | Criteria 12–13 |
| Estado vazio/erro | amostra insuficiente, identidade divergente e I/O | Criteria 6, 9, 16 |
| Arquivos/exportação | nomes, método, identidade e contagens | Criteria 11, 13 |
| Retomada | acerto, invalidação e escrita parcial | Criteria 14–16 |
| Documentação | exemplos e limites históricos/empíricos | Criterion 17 |
| Notebook local | autorização HTTP e rate limits | n/a — sem API ou serviço externo |
| Runs anteriores | confirmação destrutiva | n/a — fontes somente para leitura, sem exclusão física |

As nove dimensões foram consideradas: validação (2, 6, 9); falhas parciais (15–16); idempotência/retry/duplicatas (4–6, 14–16); autorização (n/a — Python local); concorrência/ordem (5–6, 15–16 e namespaces exclusivos C09); ciclo de dados (11, 14–16); dependências externas (13 e erros operacionais propagados); transições (14–16); observabilidade (1, 3, 11, 13, 17). As provas serão derivadas em checks após revisão.

## Sources

- Pedido do usuário: implementar com `tlc-spec-lean` o ajuste discutido, comparando cinco modelos nos mesmos fragmentos identificados e parâmetros.
- `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` AD-012/019/025/026, notebook de reprodução 06 e `latent_resume.py`: invariantes científicos e identidade de cache vigente.
- `short/ICTV_latent_space-bkp.ipynb` (células 6, 7, 10–11), `paper/manuscript_submission.tex` (Figura 2), `configs/default_species.yaml` e `docs/dataset_manifest.md`: evidência histórica, nível Genus, entradas e limites de reprodução.
