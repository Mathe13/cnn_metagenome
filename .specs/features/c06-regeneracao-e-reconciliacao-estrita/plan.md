# C06 — Regeneração estrita, matriz de evidências e reconciliação do artigo

## Problem

A geração atual de tabelas, figuras e manuscrito pode apresentar valores históricos ou imagens sem vínculo com os dados como resultados regenerados. As etapas não compartilham uma seleção explícita de execuções, e compilar o PDF não comprova que seus números correspondem aos artefatos avaliados.

Inspeção em 2026-11-04, HEAD `4275311d7e32b299c974138054bfd1f0915acf20`:

- `generate_all_tables(..., strict_empirical=True)` procura manifestos recursivamente e rejeita alguns smoke, mas ignora JSON inválido, aceita manifesto sem modo e continua criando as 12 tabelas de funções com valores fixos. CSVs disponíveis substituem algumas tabelas; falhas de leitura são ignoradas. Tabelas 1–3 e o resumo pareado da Tabela 4 continuam vindos de constantes.
- Os CLIs de tabelas/figuras não expõem seleção de manifestos ou modo empírico estrito. O gerador de tabelas também escreve reconciliação no diretório de entradas, e o CLI de figuras usa `paper/figures` como destino padrão.
- `figures.py` copia imagens de `/tmp/extracted_img-*` ou cria padrões PNG procedurais. O painel (e) recebe bytes distintos por construção; isso não comprova sua origem no checkpoint ConvNeXt_F. O teste atual verifica diferença de bytes entre (d)/(e), não identidade científica.
- `manuscript.py` mantém corpo, tabelas e afirmações numéricas num template fixo. `paper/manuscript_submission.tex` contém tabelas embutidas, sem consumir as tabelas geradas. Reexecutar esse gerador pode reintroduzir texto incompatível com C01–C05.
- `paper/tables/reconciliation_report.csv` marca fontes dinâmicas e MATCH, mas isso não comprova proveniência. A auditoria documental ainda chama números de verificados, cita parâmetros antigos, ConvNeXt 768-d, distância euclidiana, causas de queda e genomas vistos sem vincular evidência atual.
- A Tabela 11 no gerador descreve correlações de tripletos entre arquiteturas, mas o caminho de ingestão procura `table_11_cooccurrence_synergy.csv`. Igualdade do número da tabela não demonstra identidade da análise. Essa associação precisa ser resolvida pela definição semântica da tabela, sem renomear dados incompatíveis.
- Há CSVs e figuras locais, mas os manifestos antigos de benchmark, calibração e espaço latente declaram smoke. Manifestos antigos de motivos/longas/baselines não declaram modo nem identidade suficiente. Existência do arquivo não os torna empíricos.
- C01–C05 têm implementações/contratos locais, com limites de execução empírica. C05 registra verificação independente da trilha mínima e comparação controlada NOT_RUN. Não foi encontrado `verification.md` de C03 no checkout inspecionado. Os rótulos do estado do projeto não substituem os artefatos e a execução real.
- O notebook mestre `notebooks/reproduction/00_regenerate_all_paper_artifacts.ipynb` não existe. `/home/mlima/workspace` está ausente nesta sessão. Nenhum pacote empírico completo das 12 tabelas e cinco figuras foi identificado.

Ao entregar C06, o pesquisador terá uma única seleção de evidências por execução que alimenta tabelas, figuras e textos. Uma matriz mostrará cada lacuna e discrepância. O modo estrito não completará números por referência histórica; um rascunho com resultados históricos poderá existir apenas com marcação explícita de ausência de reprodução e sem status de pacote empírico pronto.

## Flow

Reutiliza os geradores/serializadores de reporting, os manifestos C00–C05 e a compilação existente; substitui a escolha implícita de arquivos pela seleção explícita de evidências e faz o manuscrito consumir os mesmos resultados.

1. Seleção de execuções/configuração → entrada mestre (door 3) e `viral_cnn.notebook_runtime` (exists): reserva namespace C00 e inventaria dependências sem executar treino, download ou notebooks de inferência automaticamente.
2. Entradas selecionadas → reporting (door 1): valida manifestos, hashes, modo, protocolo, classes, população e suporte específico de cada tabela/figura; registra matriz de evidências e bloqueios (door 2).
3. Evidências elegíveis → `viral_cnn.reporting.tables` (exists): deriva as Tabelas 1–12 e suas referências históricas separadas, exportando CSV/Markdown/LaTeX com identidade de transformação.
4. Evidências elegíveis → `viral_cnn.reporting.figures` (exists): renderiza Figuras 1–5 a partir dos dados e parâmetros declarados; valida painéis e caminhos case-sensitive sem imagens substitutas.
5. Tabelas/figuras e matriz → `viral_cnn.reporting.manuscript` (exists): gera versões review/submission no diretório da execução, consumindo os arquivos e atualizando texto/legendas conforme status e resultados observados.
6. Versões e fontes → `scripts/compile_paper.py` (exists): compila o pacote isolado, preserva logs/PDFs revisáveis e verifica referências, sem sobrescrever PDFs/fontes históricos.
7. Produtos → manifesto C06 e matriz (door 2): registra completude, hashes, diferenças, limitações e revisão editorial pendente; publicação/submissão externa permanece fora do fluxo.

## Impact

| Front | What changes |
| --- | --- |
| domain | `strict_empirical` passa a validar as entradas selecionadas de cada resultado, não a procurar a palavra smoke em todo o diretório. Resultados sintéticos não relacionados em outras execuções não bloqueiam um pacote válido. |
| domain | Status de evidência distingue empírico verificado, derivado estrutural, histórico não reproduzido, sintético, ausente e inválido. MATCH exige comparação calculada, não concordância declarada. |
| stored data | Novo índice de entradas e matriz versionados; não migrar nem completar manifestos históricos retrospectivamente. Saídas ficam no namespace C06. |
| scientific | Afeta admissibilidade e apresentação de todos os números, tabelas, figuras e afirmações; não altera modelos, splits, métricas ou dados para aproximar resultados do PDF. |
| manuscript | Review/submission passam a compartilhar resultados e limitações, preservando as 56 notas da versão revisável e removendo-as da limpa; fonte do gerador também é atualizada para impedir regressão do texto. |
| tests | Provas passam de existência de imagem/valores fixos para conteúdo derivado e linhagem. Alterações de critérios legados precisam ser explícitas nos checks aprovados; não enfraquecer testes para torná-los verdes. |
| interfaces | CLIs ganham seleção de entradas/mode/run_id e destinos isolados; modo principal estrito falha com diagnóstico. Geração histórica fica explicitamente separada. |

## Relations

- Um pacote C06 possui seleção explícita de execuções e entradas, configuração e uma matriz de evidências.
- Cada Tabela 1–12 e Figura 1–5 possui pelo menos uma linha na matriz; resultados com várias métricas/painéis possuem relações adicionais por célula ou painel.
- Cada valor ou elemento gráfico consumido referencia entradas/manifestações e transformação identificadas; N pode ser não aplicável somente com motivo, como contagem estrutural de parâmetros.
- Cada referência histórica pertence a uma versão/fonte do manuscrito e permanece separada de valor calculado.
- Um pacote possui duas variantes de manuscrito que consomem os mesmos resultados, com notas editoriais distintas.
- Um pacote estrito só é completo quando todas as suas dependências obrigatórias e saídas esperadas são válidas; C05 controlada NOT_RUN não é uma dependência obrigatória.

## Surface

None - sem API HTTP ou serviço externo novo. Notebooks/CLIs e estados de erro estão em Observable; formatos persistidos em Landing.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1. Seleção explícita de evidências | `publication_protocol: c06_strict_v1`, `schema_version: 1`, configuração indexa T1–T12/F1–F5 por `manifest`, `artifact`, `sha256`, `protocol`, transformação e condições esperadas; referências históricas têm fonte/versionamento separados | `rglob` de manifestos ou primeiro CSV encontrado: mistura execuções e aceita entrada semanticamente errada apenas pelo nome. |
| 2. Matriz de evidências consumível | Relação `claim_id/item_id -> input/manifest -> transformation -> output -> N -> status`; status `EMPIRICAL_VERIFIED`, `DERIVED_VERIFIED`, `HISTORICAL_UNREPRODUCED`, `SYNTHETIC`, `MISSING`, `INVALID`; discrepância tem valores/unidades/compatibilidade e explicação observada ou pendente | DYNAMIC_ARTIFACT_LOADED/MATCH sem validação e histórico usado para preencher célula ausente: presença de arquivo não demonstra reprodução. |
| 3. Notebook mestre de montagem | `notebooks/reproduction/00_regenerate_all_paper_artifacts.ipynb` usa as mesmas funções dos CLIs, resolução autônoma de src antes de importar viral_cnn, e reserva `runs/<run_id>/publication/`; padrão empírico estrito, sem lançamento automático das etapas científicas | Notebook que executa todos os treinos/inferências implicitamente ou duplica geradores em células: muda escopo, custo e lógica sem entrada explícita. |

Nothing else in this change is hard to reverse. Registrar as portas aprovadas no próximo ID livre de `.specs/STATE.md` durante implementação; preservar AD-015 e atualizar explicitamente sua interpretação de rastreabilidade à evidência atual.

## Criteria

### S1: Inventário e portão estrito por resultado (P1)

**Acceptance Criteria**

1. WHEN C06 inventaria o pacote THEN o sistema SHALL produzir linhas explícitas para as 12 tabelas e cinco figuras, declarando entradas, transformação, saídas, N/suporte e status de cada resultado esperado.
2. WHEN uma seleção é validada THEN o sistema SHALL verificar existência, leitura, hash, modo empírico explícito, protocolo, ordem de classes, checkpoint e composição requerida dos dados/modelos/ruídos para cada entrada científica selecionada.
3. IF uma entrada obrigatória está ausente, corrompida, sintética, sem proveniência suficiente ou incompleta THEN `strict_empirical=True` SHALL falhar com item/campo/caminho responsável, sem substituição por constante ou artefato de outra execução.
4. WHEN uma entrada é escolhida THEN o sistema SHALL validar sua definição semântica contra o resultado esperado, rejeitando, por exemplo, coocorrência como substituta de correlação entre arquiteturas na Tabela 11.
5. IF há manifestos sintéticos em diretórios não selecionados THEN o portão SHALL ignorá-los para admissibilidade do pacote selecionado, sem ignorar dependências sintéticas transitivas de uma entrada escolhida.
6. WHEN apenas parte das entradas é elegível THEN o sistema SHALL preservar um relatório diagnóstico das lacunas, sem marcar pacote completo nem publicar um conjunto parcial como empírico integral.

**Independent test:** pacote pequeno representando os 17 itens; remover ou adulterar uma entrada de cada família, trocar mapa/modelo e apontar Tabela 11 ao CSV semanticamente errado. Um manifesto sintético não selecionado não deve invalidar um pacote que passa os demais contratos.

### S2: Tabelas e figuras derivadas das entradas (P1)

**Acceptance Criteria**

7. WHEN Tabelas 1–12 são geradas no modo estrito THEN o sistema SHALL calcular os valores das entradas validadas, exportando CSV/Markdown/LaTeX e mantendo referências históricas em colunas/artefatos separados.
8. WHEN parâmetros de arquitetura ou custo são incluídos THEN o sistema SHALL distinguir contagem estrutural calculada e identificada por código/configuração de medições empíricas de latência/throughput e estimativas de FLOPs, sem completar medições ausentes por heurística.
9. WHEN métricas, ICs e suportes são incorporados THEN o sistema SHALL preservar unidades, estimandos, denominadores, regras de classes sem suporte e estratos dos protocolos C01–C04, sem equiparar populações de benchmark distintas.
10. WHEN uma discrepância é reportada THEN o sistema SHALL calcular observado versus histórico sem arredondamento intermediário, usando pp para percentuais compatíveis e unidade apropriada às demais grandezas; protocolos/populações incompatíveis recebem motivo e comparação numérica indisponível.
11. WHEN Figuras 1–5 são geradas no modo estrito THEN o sistema SHALL renderizar dados validados com parâmetros salvos e produzir PNG/PDF válidos, sem copiar imagens extraídas do PDF ou gerar padrões procedurais como evidência.
12. WHEN Figura 3 é montada THEN o sistema SHALL ligar cada painel ao manifesto/checkpoint da variante correspondente, exigindo ConvNeXt_F em (e) e ResNet50_F em (d), sem usar diferença de bytes como prova de identidade científica.
13. WHEN uma figura usa seleção qualitativa ou estatística THEN o sistema SHALL registrar IDs, níveis, amostragem, semente e filtros realmente usados, mantendo UMAP qualitativo e composição de tripletos exploratória.
14. IF falha transformação, renderização ou validação de caminho case-sensitive THEN o sistema SHALL registrar o erro e impedir status completo, sem placeholder ou fallback para arquivo histórico.

**Independent test:** alterar um valor de fixture deve alterar sua tabela e o dado gráfico correspondente. Injetar arquivo extraído em /tmp não deve afetar a saída. Panel (e) com bytes distintos, mas manifesto ResNet50, deve ser recusado; conferir também arquivos PNG/PDF e referências Linux.

### S3: Manuscrito e interpretação ligados à evidência (P1)

**Acceptance Criteria**

15. WHEN review/submission são gerados THEN o sistema SHALL consumir as mesmas tabelas/figuras e parâmetros de evidência do pacote, sem tabelas numéricas independentes embutidas no template.
16. WHEN Methods/Results são atualizados THEN o texto SHALL descrever split por fragmentos, corrupção `3ε/4`, variantes/seleção efetivas, KNN/cosseno de alta dimensão e ICs/calibração com suas populações e limites observados.
17. IF origem genômica ou mecanismo de queda longa permanece desconhecido/inconclusivo THEN Abstract, Discussion e Conclusions SHALL refletir essa limitação, sem afirmar seen/unseen ou mecanismo causal como demonstrado.
18. IF comparação controlada C05 está NOT_RUN/PILOT/PARTIAL THEN o texto SHALL restringir conclusões às configurações avaliadas; NOT_RUN isoladamente não bloqueia um pacote C06 válido nas demais dependências.
19. WHEN um rascunho não estrito inclui número histórico não reproduzido THEN o sistema SHALL identificá-lo como `HISTORICAL_UNREPRODUCED` no texto/matriz e impedir rotulá-lo como pacote empírico pronto; esse número não completa a saída estrita.
20. WHEN as variantes editoriais são produzidas THEN o sistema SHALL preservar as 56 notas na review e excluir as notas/macros na submission, mantendo números, referências de resultados e limitações científicas equivalentes.
21. WHEN a revisão editorial é registrada THEN o relatório SHALL enumerar status de referências, afiliações e redação, distinguindo itens efetivamente revisados de pendentes; compilação não aprova automaticamente esses itens.

**Independent test:** introduzir fixture com diagnóstico INCONCLUSIVE, C05 NOT_RUN e valor histórico divergente; verificar os qualificadores no template e em ambas as variantes e impedir retorno de frases causais predefinidas. Nenhum dado é ajustado para coincidir com o artigo.

### S4: Montagem isolada e definição de pronto (P1)

**Acceptance Criteria**

22. WHEN CLIs ou notebook mestre executam montagem THEN o sistema SHALL usar a mesma configuração/portão de admissibilidade e namespace C00 exclusivo, sem escrever nos diretórios de entrada ou sobrescrever paper/fontes/PDFs históricos.
23. WHEN o notebook mestre é executado da raiz ou de notebooks/reproduction THEN sua célula inicial SHALL resolver src antes de importar viral_cnn, sem exigir instalação editável e sem iniciar etapas de treino/download/inferência por efeito colateral.
24. WHEN um pacote é compilado THEN o sistema SHALL reter PDF e logs dentro da saída da execução, validando referências de tabelas/figuras e preservando hashes dos artefatos históricos anteriores.
25. WHEN a montagem termina THEN o manifesto SHALL registrar comando/notebook, commit/estado sujo, configuração, fontes, hashes, versões, outputs e status dos portões local, empírico, científico e editorial.
26. WHILE a verificação usa fixtures o sistema SHALL identificar os resultados como sintéticos e recusar alegação de reprodução integral; os contratos locais podem passar com o portão empírico ainda bloqueado.
27. WHEN C06 é verificada THEN o relatório SHALL fornecer comandos exatos/configuração/manifestos, saídas alteradas, discrepâncias e dependências não disponíveis, separando implementação validada de regeneração empírica e prontidão editorial.

**Independent test:** montagem sintética controlada em temporário por CLI e notebook, comparação dos dados gerados e hashes pré/pós do paper. Caso negativo com entrada científica ausente deve produzir diagnóstico e saída não completa. Compilação só valida o pacote textual/visual, sem certificar números.

## Out of scope

| Excluded | Why |
| --- | --- |
| Treinar, baixar bases ou executar automaticamente C01–C05 | C06 monta/valida evidência já produzida; instruir as etapas faltantes sem iniciá-las implicitamente. |
| Corrigir método experimental para aproximar números publicados | Divergência deve ser investigada e registrada; mudança exige outro protocolo. |
| Inferir accessions, causalidade ou função de motivos sem dados | Limites científicos permanecem explícitos. |
| Recriar análises de motivos/entropia ausentes só para completar tabelas | Auditar/adaptar proveniência existente; sem evidência suficiente, bloquear o item e indicar seu produtor. |
| Transformar scaffold, dry-run ou figura extraída em reprodução | São referências históricas ou testes, não novas medições. |
| Publicar, enviar artigo, sobrescrever versões históricas ou contactar coautores | Montagem e revisão local não autorizam distribuição externa. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Perfil da skill | `light`; negativos do roadmap obrigatórios | Sem perfil específico no projeto | n |
| Dependências | C01–C04 e análises produtoras das 17 saídas; C05 mínima somente se seus resultados forem citados; trilha controlada opcional | Preserva roadmap sem confundir contratos implementados com execução empírica | n |
| Notebook mestre | Criar como interface de montagem/diagnóstico, reutilizando reporting e CLIs | Previsto no roadmap; não duplicar pipeline experimental | n |
| Modo padrão | Estrito para novos CLIs/notebook; modo histórico explícito para revisão de referências | Evita sucesso silencioso com tabelas fixas; manter compatibilidade legada identificada quando necessária | n |
| Contagens estruturais | Aceitar `DERIVED_VERIFIED` quando calculadas do código/configuração identificados, N não aplicável com motivo | Contagem de parâmetros não exige inferência, mas não valida latência ou acurácia | n |
| Fonte histórica | Snapshot/hash dos arquivos existentes do manuscrito e tabelas antes da montagem | Não atribuir ao PDF publicado uma versão que não foi identificada | n |
| Lacuna de proveniência | Não fabricar manifesto retrospectivo; exigir evidência externa identificada ou regeneração pelo produtor | `is_smoke_test: false` isolado não comprova origem | n |
| Numeração 1–12/1–5 | Manter IDs estáveis e revisar a semântica de cada item antes de adaptar fontes | O conflito da Tabela 11 e a seleção parcial UMAP podem revelar saída inadequada; não resolvê-los por nome | n |
| Granularidade da matriz | Uma linha por item e linhas adicionais por métrica/condição/painel quando suas origens divergem | Uma tabela pode combinar parâmetros derivados, métricas empíricas e custos ausentes | n |
| Escrita com dados ausentes | Produzir diagnóstico/rascunho histórico marcado; pacote estrito permanece bloqueado | Permite revisão sem declarar submissão pronta ou resultado reproduzido | n |
| Fontes editoriais | Revisar consistência local e registrar pendências; verificação externa de bibliografia/afiliações requer fontes identificadas quando executada | Não inventar aprovação editorial por testes de existência ou compilação | n |

**Open questions:** none - defaults acima definem a implementação; acesso aos artefatos reais e revisão editorial condicionam prontidão, sem bloquear a especificação ou testes locais.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Inventário | 17 itens, origem e suporte | AC 1 |
| Portão estrito | Ausência, corrupção, incompletude e dependências sintéticas | AC 2–6 |
| Tabelas | Valores derivados, unidades, ICs e discrepâncias | AC 7–10 |
| Figuras | Dados reais, identidade de painéis, amostras e falhas | AC 11–14 |
| Manuscrito | Fonte comum e protocolos observados | AC 15–16 |
| Afirmações | Unknown, hipóteses, C05 opcional e histórico | AC 17–19 |
| Revisão editorial | Duas variantes, notas e pendências | AC 20–21 |
| CLI/notebook | Paridade, imports e ausência de efeitos pesados | AC 22–23 |
| Compilação | Referências, retenção de PDF/logs e preservação | AC 24 |
| Manifesto/verificação | Portões e limite da evidência | AC 25–27 |
| Saídas | Repetição e concorrência | existing - reserva exclusiva C00; publicação de pacote completo após validação |
| Console/log | Caminhos de fontes e dados carregados | existing - `log_artifact_path`, contrato LOG_PATH de AGENTS.md |
| HTTP | Autorização, rate limiting e paginação | n/a - pipeline local, sem serviço HTTP |
| Interface visual | Navegação/loading | n/a - sem aplicação nova; tabelas/figuras/PDF são artefatos exportáveis |

As nove dimensões foram consideradas: validação (AC 2–5), falha parcial (AC 6 e 14), repetição (AC 22), autorização (n/a — uso local), concorrência (reserva C00 e pacote completo), ciclo dos dados (AC 22 e 24–25), dependência externa (AC 21 e 27), transições de prontidão (AC 6 e 25–27) e observabilidade (AC 1, 10 e 25). Associação às provas em `checks.md` somente depois da revisão deste plano.

## Sources

- `ROADMAP_CORRECOES_REPRODUCAO.md`, C06/portões; `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` — definição de pronto, invariantes e caráter opcional da comparação controlada.
- Specs S11/R08 e planos/verificações C00–C05 disponíveis — intenção anterior e limites de evidência registrados; ausência de relatório não é prova de conclusão nem de falha científica.
- `reporting/tables.py`, `figures.py`, `manuscript.py`, CLIs/testes, `paper/`, auditoria de reconciliação e manifestos locais — comportamento atual e divergências inspecionadas.
