# C09 — Retomada validada do notebook 06

## Problem

O notebook 06 extrai novamente embeddings de cinco variantes e três splits a cada execução. A pasta de saída é exclusiva e não há leitura de cache no fluxo, embora `load_embedding_artifact` já valide identidade e SHA-256. Os logs fornecidos confirmam três splits concluídos para ResNet clean e início de gravação de treino para ResNet noise; o usuário relata também JSON, UMAP e Parquets KNN na pasta remota. Não houve inspeção desses arquivos remotos nesta sessão. Desconectar do JupyterHub não prova interrupção do kernel, e a presença de arquivos isolados não prova conclusão científica.

## Flow

Reutiliza a validação de `load_embedding_artifact`, o armazenamento por orçamento C07, a sonda selada C01 e a auditoria compacta KNN C08; não duplica os algoritmos científicos.

1. Notebook 06 (exists) lê `LATENT_EXECUTION_MODE` e `LATENT_REUSE_RUN_ID` antes da inicialização do runtime e reserva uma nova pasta exclusiva com `reserve_run_directory` (exists). O modo e as origens selecionadas ficam registrados nessa nova execução.
2. Notebook 06 (exists) resolve taxonomia, partições, hashes, checkpoints e configuração atuais antes de procurar resultados compatíveis. Busca em execuções anteriores do mesmo notebook, da mais recente para a mais antiga; uma origem explícita restringe essa busca. Arquivos de origens ficam somente para leitura.
3. Notebook 06 (exists) tenta carregar cada par de embedding/manifesto com `load_embedding_artifact` em `viral_cnn.eval.latent` (exists). Um candidato inválido é registrado e ignorado; a busca continua até encontrar candidato válido ou esgotar as origens. Apenas splits sem candidato válido são extraídos e publicados na pasta nova.
4. `LinearProbeClassifier` em `viral_cnn.eval.latent` (exists) mantém treino em train, seleção em validation e selagem anterior ao teste. Estados de sonda e predições nativas passam a ser persistidos com identidade verificável (door 2). Se ausentes, o notebook recalcula esses componentes; os embeddings históricos continuam reutilizáveis independentemente disso.
5. Notebook 06 (exists) valida checkpoints de progresso por etapa (door 2) antes de KNN/Silhouette por variante/rank, auditoria da sonda e UMAP. Reutiliza etapas completas compatíveis e recalcula as ausentes ou inválidas. Um processo interrompido reinicia somente a unidade incompleta; não há retomada interna de lote/época/bloco KNN.
6. Notebook 06 (exists) reconstrói tabelas 5/6, cobertura taxonômica e manifesto a partir dos componentes admitidos. Registra caminhos e hashes das origens reutilizadas, motivo dos descartes e componentes recalculados, sem promover saídas parciais a execução concluída.

## Impact

| Front | What changes |
| --- | --- |
| execução | Reutilização é o padrão; rerun força todos os cálculos em uma nova pasta. O histórico nunca é sobrescrito. |
| resultado científico | Afeta a origem operacional dos embeddings, Tabelas 5/6, auditoria de sonda e projeção UMAP do notebook 06. Mantém classes, pesos dos backbones, split, codificação, amostras, KNN e definição de métricas. |
| arquivos existentes | Embeddings históricos com proveniência suficiente podem ser carregados. Arquivos incompletos, incompatíveis ou corrompidos são descartados da seleção, preservando seus bytes para auditoria e para não interferir em kernels ativos. |
| persistência nova | Acrescenta checkpoints atômicos de progresso e pesos da sonda selada. Tabelas/CSV/JSON avulsos antigos não recebem proveniência retrospectiva. |
| inicialização da sonda | Evidência encontrada no build: `LinearProbeClassifier(...)` sorteia pesos antes de `fit_select_and_seal` chamar `manual_seed`. Para que o cache não altere um ajuste posterior, a inicialização passa a usar seed 42 explicitamente antes da construção, identificada por `c09_seeded_probe_init_v1`. A sonda recalculada pode divergir da auditoria histórica; não se afirma equivalência dos seus números. |
| recursos | Validar hashes e ler embeddings ainda exige I/O. Reutilização usa o orçamento de RAM/spill existente, sem carregar todo o Parquet de uma só vez. |
| histórico metodológico | A documentação de auditoria descreve Silhouette até 50.000; o notebook atual limita a 5.000 e UMAP a 100. Esta tarefa preserva os limites efetivamente implementados e registra a divergência. |

## Relations

Uma execução nova referencia zero ou mais execuções anteriores do notebook 06. Cada embedding admitido pertence a exatamente uma variante e um split, com identidade de checkpoint, partições e ordem de classes verificadas. Uma sonda selada depende dos embeddings train/validation e de sua configuração de treino. Uma avaliação KNN/Silhouette depende do embedding test, taxonomia e configuração desse nível. Cada checkpoint de progresso referencia todos os arquivos necessários à sua restauração, com hashes; resultados sem essas dependências verificáveis não são considerados completos. O manifesto da nova execução retém a proveniência das referências reutilizadas, inclusive quando pertencem a outra pasta.

## Surface

None - não há rota HTTP ou códigos de resposta de serviço; as superfícies são parâmetros e arquivos do notebook.

A primeira célula recebe `LATENT_EXECUTION_MODE="reuse"` por padrão, ou `"rerun"`, via parâmetro editável e override de ambiente; devolve modo efetivo no console e manifesto. Resultados: configurado ou `ValueError` para valor inválido.

A seleção de origem recebe `LATENT_REUSE_RUN_ID=None` por padrão (busca automática), ou run ID anterior explícito; devolve candidatos e plano de reaproveitamento por etapa. Resultados: origem encontrada, nenhuma origem automática ou erro de origem explícita ausente/inválida.

A retomada recebe arquivos e identidades esperadas; devolve componentes e relatório persistente. Resultados: reutilizado, descartado, recalculado ou erro operacional de I/O. Cada conjunto será enumerado nas provas de Coverage.

Não há serviço HTTP, autenticação nova ou interface gráfica. Os parâmetros aparecem no início da primeira célula de código e são documentados com exemplos de uso em Jupyter.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — comando e semântica de descarte | `LATENT_EXECUTION_MODE = "reuse"`, valores `reuse/rerun`; `LATENT_REUSE_RUN_ID = None`; novo namespace por execução; fontes anteriores somente para leitura; inválidos ficam fora da seleção, com motivo no relatório. | Reabrir e limpar a pasta anterior: pode sobrescrever resultados e apagar arquivos que um kernel remoto ainda esteja produzindo. |
| 2 — contrato de progresso | `c09_latent_resume_v1`: registro por componente, identidade dos insumos/configuração/código relevante, arquivos de saída com SHA-256 e marcador de conclusão publicado por substituição atômica após as saídas; sonda persistida como state_dict e seleção/hash selados, carregada com `weights_only=True`. | Confiar só na existência de CSV/Parquet ou serializar objetos arbitrários: não comprova integridade, compatibilidade ou isolamento de test. |
| 3 — inicialização identificável da sonda | `c09_seeded_probe_init_v1`: seed de `ctx.seed` antes de construir a cabeça da sonda; namespace de execução `c09-*` e identificador no manifesto e identidade da etapa. | Manter inicialização dependente do estado global após extração: um miss isolado de cache pode mudar o ajuste ao pular os loaders, impedindo equivalência entre reexecuções C09. |

Nenhuma dependência nova. O esquema existente dos embeddings permanece intacto. A implementação evitará alterar `latent.py` apenas para orquestração, pois seu hash integral faz parte da identidade histórica dos embeddings; se uma mudança nesse arquivo se mostrar necessária, registrará a invalidação resultante antes de fazê-la, sem ignorar o hash antigo.

## Criteria

### S1: reutilização segura de embeddings (P1)

**Acceptance Criteria**

1. WHEN o notebook inicia sem override THEN o sistema SHALL usar `LATENT_EXECUTION_MODE="reuse"` e exibir o modo efetivo.
2. IF o modo não é `reuse` ou `rerun` THEN o sistema SHALL lançar `ValueError` antes de extrair embeddings.
3. WHEN o modo é `rerun` THEN o sistema SHALL executar todos os componentes sem carregar resultados anteriores em uma nova pasta exclusiva.
4. WHEN a origem não é especificada THEN o sistema SHALL buscar candidatos em pastas anteriores do notebook 06, ordenadas da mais recente para a mais antiga, sem misturar smoke e execução empírica.
5. WHEN uma origem explícita válida é configurada THEN o sistema SHALL restringir a busca de reutilização a essa execução.
6. IF a origem explícita é ausente ou seu identificador é inválido THEN o sistema SHALL informar o identificador e interromper antes da extração.
7. WHEN um embedding é reutilizado THEN o sistema SHALL validar o par Parquet/JSON, fingerprint, checksum, variante/split, partições e hashes, checkpoint e hash, ordem de classes, codificação, semente/amostragem, modo, código extrator, contagem e dimensão esperadas.
8. IF um candidato está incompleto, corrompido ou incompatível THEN o sistema SHALL registrar a causa, excluir esse candidato da seleção e continuar procurando outro candidato compatível antes de recalcular o split.
9. WHEN um split válido já existe THEN o sistema SHALL carregar seus embeddings pelo orçamento RAM/spill sem executar novamente a extração desse split.
10. WHILE retoma uma execução THEN o sistema SHALL manter todos os arquivos das origens anteriores inalterados.

**Independent test:** fixtures sintéticas com duas origens, hashes adulterados, JSON ausente, dimensões/contagens incorretas, checkpoint/taxonomia diferentes e spies na extração; comparar arquivos de origem antes/depois e testar rerun com cache válido presente.

### S2: progresso posterior e conclusão auditável (P1)

**Acceptance Criteria**

11. WHEN uma etapa termina THEN o sistema SHALL publicar um registro `c09_latent_resume_v1` completo somente após publicar e calcular os hashes de todos os seus arquivos necessários.
12. WHEN uma etapa compatível está completa THEN o sistema SHALL restaurar seu resultado sem repetir o cálculo dessa etapa, incluindo pesos/seleção da sonda, predições nativas, KNN e Silhouette por variante/rank, avaliação da sonda e UMAP.
13. IF falta um arquivo ou sua identidade/hash diverge THEN o sistema SHALL recalcular a unidade incompleta e seus dependentes sem invalidar componentes independentes válidos.
14. WHEN a sonda é restaurada THEN o sistema SHALL verificar pesos selados, seleção train/validation e disjunção do teste antes de admitir sua avaliação, mantendo test fora do ajuste e da seleção.
15. IF predições nativas históricas não foram persistidas com identidade verificável THEN o sistema SHALL refazer sua inferência sobre o mesmo teste com conferência de IDs, sem inferir equivalência numérica apenas aplicando a cabeça aos embeddings salvos.
16. WHEN somente saídas legadas de KNN/UMAP ou tabelas avulsas existem sem identidade completa de etapa THEN o sistema SHALL recusá-las como cache concluído e informar a ausência de proveniência, permitindo reutilizar embeddings legados válidos.
17. IF a execução falha durante uma escrita THEN o sistema SHALL deixar essa unidade sem marcador de conclusão válido, permitindo recalculá-la em outra execução.
18. WHEN duas execuções leem a mesma origem THEN o sistema SHALL gravar cada execução em seu próprio namespace, sem apagar temporários ou modificar arquivos da outra execução.
19. WHEN o notebook conclui THEN o sistema SHALL registrar modo, origens, caminhos/hashes reutilizados, descartes e recálculos no manifesto/relatório e manter os parâmetros científicos C01/C07/C08 existentes.
20. WHEN a feature é validada localmente THEN o relatório SHALL separar testes sintéticos CPU da validação empírica remota não executada e listar os dados/checkpoints indisponíveis.

**Independent test:** executar smoke offline, executar novamente em reuse com funções de cálculo monitoradas, interromper uma publicação e adulterar um resultado KNN; confirmar retomada por unidade, sonda selada, proveniência e isolamento entre execuções. Verificador independente após a implementação, perfil light.

## Out of scope

| Excluded | Why |
| --- | --- |
| Recuperar stdout perdido ou conectar a um kernel remoto existente | A tarefa é retomada de artefatos persistidos, sem acesso ao estado remoto. |
| Retomada dentro de lote de extração, época de sonda ou bloco KNN | Os algoritmos atuais não persistem esse estado; a unidade incompleta é recalculada. |
| Excluir fisicamente arquivos de runs históricos | Descarte significa não reutilizar; preservar os arquivos evita interferência em kernels ativos e perda de evidência científica. |
| Mudanças em arquitetura, amostragem, métricas, dados ou checkpoints dos backbones | Alterariam o experimento em vez de acrescentar retomada operacional. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Perfil de verificação | light | Padrão da skill na ausência de declaração no AGENTS.md; provas negativas de integridade continuam necessárias. | n |
| Fonte de artefatos | Diretório `results_reproduction/runs` do mesmo projeto | É a topologia existente e a origem indicada nos logs. | n |
| Versões remotas | Validar os arquivos disponíveis contra o código/configuração efetivos no momento da retomada | A sessão local não acessou o servidor; não se presume que seu checkout coincide com o local. | n |

**Open questions:** none — os defaults e o significado de descarte estão explícitos para revisão.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Primeira célula | flags, defaults, exemplos e erro de valor | Criteria 1–6 |
| Seleção de cache | ordenação, duplicatas, ausência, incompatibilidade e integridade | Criteria 4–9 |
| Coleção de runs | agrupamento por notebook, descarte e histórico | Criteria 4–10, 18 |
| Execução longa | progresso visível, origem absoluta via LOG_PATH, motivo de descarte | Criteria 8, 19 |
| Progresso persistido | estados parcial/completo, publicação e falha | Criteria 11–18 |
| Documentação | exemplos reuse/rerun/origem e limites de validação | Criteria 1–6, 16, 20 |
| Notebook | autorização HTTP, rate limits e estados de tela | n/a — execução Python sem serviço ou tela nova |
| Histórico | confirmação de exclusão destrutiva | n/a — nenhuma exclusão física de origens |

As nove dimensões foram examinadas: validação e limites (2, 6–8); falha parcial (11, 17); idempotência/retry/duplicatas (4, 8–13); autorização/rate limits (n/a — sem serviço); concorrência/ordem (4, 10, 18); ciclo de dados (10–11, 16); dependência externa (15, 17 e erro operacional de I/O propagado em vez de tratado como corrupção); transições (11–14, 17); observabilidade (1, 8, 19–20). As provas serão derivadas em `checks.md` após revisão.

## Sources

- Pedido do usuário nesta conversa: parâmetro inicial reuse/rerun, padrão reutilizar, validação e retomada onde possível, com `tlc-spec-lean`.
- `AGENTS.md`, `src/viral_cnn/AGENTS.md` e `.specs/STATE.md` (AD-017, AD-019, AD-025): topologia, contratos científicos e proveniência.
- Notebook 06 e `src/viral_cnn/eval/latent.py`: extração sem leitura de cache, artefatos versionados, sonda selada e auditoria KNN atualmente implementados.
