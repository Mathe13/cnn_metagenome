# C05 — Fine-tuning auditável e comparação controlada condicional

## Problem

Os notebooks 03/04 oferecem fine-tuning com ruído, mas seus manifestos não comprovam toda a linhagem dos pesos, a seleção efetiva ou a integridade da retomada. O script controlado ainda não executa dados reais. Isso impede distinguir uma adaptação histórica de execução única, um teste da mecânica e uma comparação pareada capaz de sustentar afirmações mais restritas sobre as configurações avaliadas.

Inspeção em 2026-11-03, HEAD `5a7487e565e2fda9bcd826a666266e0a330be6d2` e alterações locais C04 existentes:

- O 03 parte de `resnet_clean_species`; o 04 parte de `convnext_transfer_species`. Ambos configuram 15 épocas e epsilon=0.05 em modo empírico. O 04 tem três de cinco células de código com execution_count: isso não prova falha nem conclusão do treinamento.
- Os notebooks usam saídas fixas, carregam automaticamente `finetune_noise_last.pt` se existe e não verificam se pertence à mesma execução/configuração/dados. A configuração declara seleção last para ResNet e best-val-Macro-F1 para ConvNeXt, mas o loop sempre salva last e uma cópia best; o manifesto só identifica last.
- O manifesto usa o nome lógico `base_model` como se fosse caminho de checkpoint. Os manifestos locais de ambos os modelos declaram `is_smoke_test: true`, uma época e métricas finais zeradas: não demonstram fine-tuning empírico nem atualização de pesos.
- `save_epoch_checkpoint` escreve diretamente com `torch.save`, sem publicação atômica. Salva modelo/otimizador/scheduler/scaler e parte da configuração, mas não inclui toda a identidade do experimento, RNGs, melhor métrica ou contador de passos. A restauração ignora exceções de otimizador/scheduler/scaler.
- O histórico cria `CosineAnnealingLR(T_max=len(train_loader)*15)` e chama `step()` por época. Os notebooks de reprodução seguem essa unidade. Alterá-la para lote com o mesmo nome mudaria a trajetória de LR; o valor pretendido não substitui a implementação observada.
- O histórico ConvNeXt, célula 9, instancia dropout=0.5; o registro/configuração de reprodução usa 0.7. Dropout não é recuperável por state_dict sozinho. Há também divergência entre a afirmação documental de best-val-loss na época 7 e a regra Macro-F1 do notebook atual; nome/sufixo de arquivo não comprova seleção.
- `run_controlled_training.py` cai em dados sintéticos quando não encontra diretório local e lança `NotImplementedError` quando o encontra. Só avalia no fim, sem usar `eval_interval_steps`. `train_step` não injeta o ruído configurado; o caminho sintético o aplica apenas ao treino na geração dos dados. O dry-run reduz a cabeça a até 100 classes quando CUDA não está disponível, mas anuncia validação de paridade.
- `baseline_pair.yaml` define 200 passos, batch 32, warmup 20 e sementes 42/43/44. Esse orçamento não demonstra treinamento completo. O loader de config usa modo histórico por padrão se o YAML não declara mode, enquanto a dataclass usa controlled_parity.
- O profiler sincroniza CUDA em cada passo e fornece FLOPs por heurística de parâmetros. `docs/controlled_architecture_experiment.md` contém tabelas de custo/resultados sem manifesto de execução correspondente inspecionado e números de parâmetros que divergem da reconciliação R08. Não usar esses valores como medições.
- Os checkpoints base canônicos e `/home/mlima/workspace` não existem nos caminhos locais inspecionados. C04 tem alterações locais e verificação sintética registrada; isso não fornece dados/pesos reais para C05.

A entrega mínima será uma trilha segura e rastreável para validar 03/04 no JupyterHub, preservar checkpoints históricos e limitar conclusões a configurações de execução única. A comparação controlada é uma extensão opcional, executável somente com orçamento e GPU disponíveis; sua ausência não bloqueia C06, desde que o limite causal fique explícito.

## Flow

Reutiliza `ControlledTrainer`, configuração, registro de modelos, datasets Parquet e infraestrutura C00; preserva os notebooks históricos e mantém a trilha opcional distinta do fine-tuning histórico.

1. Escolha de trilha, configuração e entradas → notebooks 03/04 e `viral_cnn.notebook_runtime` (exists): inventário/preflight, valida splits/ordem de classes/checkpoints, reserva namespace exclusivo.
2. Checkpoint base e evidências históricas → `viral_cnn.models.registry` e notebooks 03/04 (exists): identifica pesos realmente carregados, dropout efetivo e divergências; registra lacunas sem inferi-las do nome do arquivo.
3. Treino/validação e protocolo salvo → `viral_cnn.training.trainer` (exists): executa fine-tuning epsilon=0.05 com logs, LR/AMP observados e contagens; publica checkpoints com identidade e regra de seleção explícitas (doors 1–2).
4. Retomada explicitamente escolhida → `ControlledTrainer.load_resume_checkpoint` (exists): valida identidade/configuração e restaura estados necessários; rejeita retomada incompatível antes de atualizar pesos.
5. Checkpoint selecionado congelado → avaliadores existentes em `viral_cnn.eval.runner` (exists): avaliam teste identificado sem seleção pelo teste e exportam relatório entre configurações, separado de resultados históricos.
6. Somente na trilha opcional → `scripts/run_controlled_training.py` e `configs/controlled/baseline_pair.yaml` (exists): loaders reais, paridade efetiva de dados/hiperparâmetros/orçamento, seis execuções planejadas e checkpoints por validação (door 3).
7. Produtos das trilhas executadas → `write_notebook_manifest`, relatório de comparação e `docs/controlled_architecture_experiment.md` (exists): registra evidência, custos medidos/estimados, execução parcial e afirmações permitidas para C06.

## Impact

| Front | What changes |
| --- | --- |
| domain | Novo `c05_finetune_audit_v1`: execução mínima identificada, com linhagem e divergências explícitas; não equivale automaticamente à reprodução dos pesos publicados. |
| domain | `selected_checkpoint` diferencia last, best e checkpoint histórico escolhido; best exige regra/época/métrica comprovadas. |
| domain | Retomada deixa de ocorrer por coincidência de nome/diretório; requer seleção explícita e validação da identidade. |
| stored data | Novos checkpoints/manifestações dentro da execução; preserva pesos e históricos existentes. Checkpoints legados sem estados completos podem inicializar nova adaptação identificada, mas não provar retomada exata. |
| scientific | Pode alterar artefatos e métricas de novas adaptações; qualquer mudança de dropout, scheduler, inicialização ou seleção recebe novo protocolo/identificador e comparação explícita. |
| implementation | Atualiza 03/04, checkpointing/configuração do trainer e testes científicos. A extensão opcional atualiza o CLI controlado e seus loaders, sem tornar treinamento pesado parte de testes locais. |
| reporting | Relatório separa execução única histórica, piloto, comparação pareada e experimento não executado; custos estimados não viram horas-GPU medidas. |

## Relations

- Uma execução mínima pertence a uma variante base, seu checkpoint e configuração efetiva identificados.
- Cada época/passagem possui contagens e métricas de treino/validação; cada checkpoint referencia a execução e o estado salvo.
- Uma seleção aponta a um checkpoint e à regra/observação que a determinou; last e best podem coincidir, mas permanecem papéis diferentes.
- Uma retomada pertence a um checkpoint compatível e registra continuidade ou lacunas de estado (door 2).
- Um experimento opcional possui dois modelos por semente e três sementes; nenhum par incompleto constitui comparação completa (door 3).
- O teste pertence à avaliação dos checkpoints selecionados, não à seleção ou ao ajuste de hiperparâmetros.

## Surface

None - sem API HTTP ou serviço externo novo. Notebooks/CLI são interfaces locais descritas em Observable; protocolos persistidos em Landing.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1. Identidade e escolha do checkpoint | `protocol: c05_finetune_audit_v1`; manifesto vincula `base_variant`, `base_checkpoint_sha256`, dados/classes/configuração e papéis `last`, `best`, `selected`; seleção registra regra, época/passo, valor e hash | Usar nome de variante como caminho ou chamar last de best: não identifica pesos nem critério de seleção. |
| 2. Retomada verificável e publicação atômica | Checkpoint versionado com modelo, otimizador, scheduler, scaler, época, passos tentados/efetivos, melhor métrica/histórico, RNG Python/NumPy/Torch/ruído e identidade de configuração/dados; gravar temporário e publicar por rename; resume exige caminho explícito | Reutilizar automaticamente last de outro run ou ignorar falhas de restauração: pode alterar LR, corrupção e seleção sem registrar mudança. |
| 3. Experimento opcional distinto | `protocol: c05_controlled_scratch_v1`, inicialização do zero para ambas as arquiteturas, sementes 42/43/44 e condições efetivas pareadas; estados `NOT_RUN`, `PILOT`, `PARTIAL`, `COMPLETE` | Parear fine-tunings de origens ResNet limpa e ConvNeXt transferida como efeito isolado de arquitetura, ou aceitar 200 passos/fixtures como treinamento completo. |

Nothing else in this change is hard to reverse. Registrar portas aprovadas no próximo ID livre de `.specs/STATE.md` ao implementar; preservar AD-016/AD-018 e documentar divergências antes de qualquer correção metodológica.

## Criteria

### S1: Trilha mínima com origem e protocolo auditáveis (P1)

**Acceptance Criteria**

1. WHEN a trilha mínima é preparada THEN o sistema SHALL exigir train/validation, mapeamento canônico e checkpoint base da variante correta antes de instanciar treinamento empírico.
2. IF falta uma entrada obrigatória em modo empírico THEN o sistema SHALL bloquear a execução sem inicialização aleatória ou fallback sintético.
3. WHEN um modelo base é carregado THEN o manifesto SHALL registrar caminho/SHA-256, variante/rank, ordem de classes e configuração observada, distinguindo dropout efetivo e parâmetros históricos desconhecidos.
4. WHEN a configuração é consolidada THEN o relatório SHALL confrontar scheduler, unidade de passo, dropout e seleção com notebooks históricos, registro e código atual, identificando diferenças antes do treino.
5. WHEN o fine-tuning mínimo é executado THEN o sistema SHALL aplicar epsilon=0.05 com ressorteio incluindo a base original tanto no treino quanto na validação, registrando RNG, amostragem e contagens por época.
6. WHEN a avaliação pré-treino é executada THEN o sistema SHALL registrar métricas clean/ruído e suporte efetivamente avaliado, distinguindo prévia limitada de avaliação completa.
7. WHEN épocas são concluídas THEN o sistema SHALL registrar passos de otimizador tentados/efetivos, LR/unidade do scheduler, estado AMP/scaler, perda, Accuracy, Top-5, Macro/Weighted-F1 e contagens observadas de treino/validação.
8. WHEN um checkpoint é selecionado THEN o sistema SHALL registrar o critério explícito e sua evidência, preservando last/best separadamente e sem inferir qualidade ou número de época pelo sufixo do arquivo.

**Independent test:** modelo pequeno com checkpoint base conhecido, corrupção nas duas partições e trajetória de validação controlada; regra last deve escolher último, best deve escolher melhoria real, inclusive primeira métrica igual a zero. Ausência de checkpoint empírico deve falhar antes de atualizar pesos.

### S2: Isolamento e retomada confiável (P1)

**Acceptance Criteria**

9. WHEN 03/04 escrevem resultados THEN o sistema SHALL usar namespaces C00 exclusivos para logs e checkpoints, preservando todos os pesos/resultados históricos.
10. IF uma retomada é solicitada THEN o sistema SHALL validar variante, classes, dados, configuração e estados antes da primeira atualização, recusando incompatibilidade com diagnóstico explícito.
11. WHEN uma retomada completa é aceita THEN o sistema SHALL restaurar os estados da door 2 e registrar checkpoint de origem, próximo passo/época e limites de determinismo de workers/GPU.
12. IF a restauração de otimizador, scheduler, scaler ou RNG obrigatório falha THEN o sistema SHALL interromper a retomada exata, sem engolir a exceção nem substituir estado silenciosamente.
13. WHEN um checkpoint é publicado THEN o sistema SHALL usar escrita atômica, mantendo o último arquivo íntegro se houver interrupção antes da publicação.
14. WHEN a trilha mínima termina ou falha THEN o manifesto SHALL registrar etapas/épocas realmente concluídas, saídas/hashes e estado final, sem tratar abertura de notebook, execution_count ou logs zerados como treino empírico completo.

**Independent test:** em CPU e retomada na fronteira de época, comparar pesos, LR e próxima corrupção com execução contínua equivalente. Injete estado incompatível e falha de gravação; o checkpoint anterior deve permanecer íntegro. Para workers/GPU, registrar o que não foi demonstrado localmente.

### S3: Avaliação e limites da trilha mínima (P1)

**Acceptance Criteria**

15. WHEN checkpoints selecionados são avaliados THEN o sistema SHALL mantê-los congelados e usar teste identificado nos mesmos níveis de ruído, sem carregar o teste durante seleção ou alterar a regra a partir dele.
16. WHEN um relatório compara os dois fine-tunings THEN o sistema SHALL declarar origem distinta dos pesos, configuração/seleção observadas e uma execução por configuração, sem atribuir diferenças exclusivamente à arquitetura.
17. IF a trilha controlada opcional não é executada THEN o relatório SHALL registrar `NOT_RUN` e conclusão causal indisponível, permitindo o consumo da evidência mínima por C06 com essa limitação.
18. WHEN C05 é verificada THEN o relatório SHALL separar contratos sintéticos locais, inspeção de entradas, fine-tuning real e comparação controlada real, incluindo comandos, configurações, manifestos e bloqueios de dados/GPU.

**Independent test:** tentativa de usar teste como validação deve ser rejeitada no fluxo auditado; manifesto sintético ou orçamento não cumprido não pode receber status empírico completo. Relatório sem trilha opcional continua válido como evidência mínima limitada.

### S4: Extensão controlada, somente se explicitamente ativada (P2, opcional)

**Acceptance Criteria**

19. WHERE a trilha controlada está ativada o CLI SHALL usar loaders Parquet reais para train/validation/test, sem fallback sintético por ausência de diretório; smoke exige opção explícita e fica em execução separada.
20. WHERE a trilha controlada está ativada o sistema SHALL validar paridade efetiva de dados/ordem de exemplos, corrupção, otimizador, scheduler, loss, dropout, batch efetivo, precisão, seleção, inicialização do zero e orçamento nas duas arquiteturas para sementes 42/43/44.
21. WHEN o treinamento controlado alcança o intervalo configurado THEN o sistema SHALL avaliar validação e selecionar `best_val_macro_f1`, registrando desempate e passos reais, sem chamar o último estado de melhor por ter sido a única avaliação final.
22. WHEN o dry-run é executado THEN o sistema SHALL informar a dimensão de cabeça realmente verificada e qualquer redução sintética, sem certificar paridade das 13.683 classes com uma cabeça reduzida.
23. WHEN recursos são medidos THEN o sistema SHALL registrar hardware, VRAM, sequências processadas, tempo e horas-GPU com escopo observado; FLOPs heurísticos e previsões de custo ficam explicitamente estimados e separados.
24. IF faltam pares/sementes, o orçamento completo não foi definido/cumprido ou foram usadas fixtures THEN o sistema SHALL classificar a execução como PARTIAL/PILOT/sintética, sem apresentá-la como experimento controlado completo.
25. WHEN uma comparação controlada completa é reportada THEN o sistema SHALL fornecer resultados por semente no mesmo teste identificado e restringir interpretação às arquiteturas/configurações, dados e orçamento efetivamente avaliados, sem superioridade universal.

**Independent test:** fixtures verificam paridade, intervalo de validação e caso negativo de dropout/origem/orçamento divergente; nunca substituem seis execuções reais. Microbenchmark de 200 passos permanece piloto mesmo se as seis combinações sintéticas terminarem.

## Out of scope

| Excluded | Why |
| --- | --- |
| Iniciar treinamento pesado, reservar GPU ou baixar dados durante planejamento | Esta tarefa produz o plano; execução real dependerá do ambiente e da trilha selecionada. |
| Tornar o experimento pareado condição para submissão | O roadmap permite C06 com afirmações limitadas e trilha opcional NOT_RUN. |
| Corrigir LR por lote, dropout ou inicialização sob o mesmo nome histórico | Mudança metodológica exige identificador/configuração distintos e comparação explícita. |
| Reconstruir configurações históricas a partir apenas de pesos ou nomes | Dropout, seleção e linhagem não são comprovados por esses artefatos isolados. |
| Busca de hiperparâmetros, ablação de arquitetura e mudança de split | Outro desenho experimental, fora da auditoria mínima e do par controlado definido. |
| Usar IC C04 como variabilidade de treinamento | C04 mede incerteza condicional de fragmentos; múltiplas sementes reais são evidência diferente. |
| Alterar checkpoints históricos, promovê-los automaticamente ao registro ou reescrever o artigo | Preservar os originais; C06 fará incorporação e reconciliação explícitas. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Perfil da skill | `light`; negativos científicos obrigatórios | Não há perfil específico no projeto | n |
| Escopo inicial | Implementar a trilha mínima; S4 permanece opcional e só entra no build após seleção explícita | Não há prazo/orçamento/GPU informados para retreinamento; a ausência não bloqueia C06 | n |
| Estado da opcional sem decisão | `NOT_RUN`, sem estimativa causal | Não transformar falta de recursos ou fixtures em conclusão | n |
| Protocolo mínimo | Preservar 15 épocas, AdamW lr 5e-5, weight_decay 1e-4, label_smoothing 0.1 e epsilon=0.05, identificando a configuração efetiva | Defaults atuais não comprovam equivalência aos pesos publicados | n |
| Seleção mínima | ResNet last e ConvNeXt best-val-Macro-F1 como no código atual, com empate pela primeira época; registrar divergência histórica de loss/época | Torna a regra verificável sem afirmar que corresponde à origem de `finetune_noise_7.pt` | n |
| Scheduler mínimo | Preservar unidade por época e T_max observado até comparação/documentação; correção opcional recebe novo identificador | A forma peculiar do histórico é evidência do que o código faz, não autorização para corrigi-lo silenciosamente | n |
| Retomada suportada | Na fronteira de época; retomada no meio de época fora do escopo inicial | Permite conferir continuidade de estados/ordem sem prometer posição exata de workers em curso | n |
| Trilha opcional | Ambas do zero, dropout 0.5, AdamW, warmup/cosseno, best-val-Macro-F1, seeds 42/43/44 conforme AD-016 | Não comparar ResNet limpa com ConvNeXt transferida como intervenção só de arquitetura | n |
| Orçamento da opcional | YAML atual de 200 passos é piloto; orçamento empírico completo precisa ser explicitado na configuração antes da execução | Não inventar tempo/GPU nem declarar convergência pelo número de passos | n |
| Tempo e profiling | Medições delimitadas sem sincronização global a cada batch no caminho de produção; FLOPs heurísticos não são medição | Preserva AD-018 e evita viés de instrumentação não declarado | n |
| Dropout histórico divergente | Registrar 0.5 do notebook versus 0.7 do registro e configuração usada; equivalência aos pesos publicados permanece desconhecida sem logs/config real | Pesos sozinhos não resolvem a divergência | n |

**Open questions:** none - trilha mínima e defaults definidos; ativação de S4 e orçamento de GPU são decisões futuras explícitas, mantendo NOT_RUN até sua definição.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Preflight 03/04 | Entrada ausente e bloqueio empírico | AC 1–2 |
| Configuração/linhagem | Peso base, hiperparâmetros e divergências | AC 3–4 |
| Treinamento mínimo | Ruído, prévia, épocas e seleção | AC 5–8 |
| Saídas | Repetição, concorrência e isolamento | AC 9; existing - reserva exclusiva C00 |
| Resume | Pedido explícito, incompatibilidade e falha parcial | AC 10–12 |
| Checkpoints | Interrupção de gravação | AC 13 |
| Manifesto | Conclusão, falha e evidência insuficiente | AC 14 |
| Avaliação/relatório | Teste independente e limite causal | AC 15–18 |
| CLI opcional | Parquet real, paridade e seleção | AC 19–21 |
| Dry-run/recursos | Cabeça reduzida, medições e estimativas | AC 22–23 |
| Comparação opcional | Piloto, parcial e completa | AC 24–25 |
| Console/log | Caminhos de dados e pesos carregados | existing - `log_artifact_path`, contrato LOG_PATH de AGENTS.md |
| HTTP | Autorização, rate limiting e paginação | n/a - notebooks/CLI locais sem serviço HTTP novo |
| Interface visual | Navegação, loading e layout | n/a - não há tela nova; produtos são logs, checkpoints e relatórios |

As nove dimensões foram consideradas: validação (AC 1–4 e 10), falha parcial (AC 12–14), repetição (AC 9–11), autorização (n/a — processamento local), concorrência (reserva C00/publicação atômica), ciclo dos dados (AC 3 e 9–14), dependência externa (AC 2 e 18), transições (AC 14, 17 e 24) e observabilidade (AC 6–8, 18 e 23). Associação às provas em `checks.md` somente após revisão do plano; identificar separadamente obrigações mínimas e opcionais ativadas.

## Sources

- `ROADMAP_CORRECOES_REPRODUCAO.md`, C05 e regra de C06; `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` AD-016/AD-018 — escopo condicional, preservação e limite causal.
- Specs N03/R02/S12; notebooks históricos `short/finetune_resnet50_species.ipynb` e `short/finetune_convnext_species.ipynb` — implementação original de ruído, scheduler e salvamento.
- Notebooks de reprodução 03/04, `training/trainer.py`, `config.py`, `profiler.py`, script/config controlados, testes e manifestos locais — comportamento inspecionado e lacunas de evidência; `docs/controlled_architecture_experiment.md` não substitui logs de execução real.
