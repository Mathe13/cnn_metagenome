# Auditoria de inferência de sequências longas

## Valores históricos e evidência atual

A Tabela 12 histórica registra 72,69% de acurácia por média de logits para ResNet50 e valores de 3,49% a 20,38% para variantes ConvNeXt. Esses valores são referências publicadas/históricas; não são resultados regenerados pelo protocolo C02. Os antigos manifestos locais não identificam de forma suficiente o conjunto de consultas, a ordem de classes e os cinco checkpoints para sustentar uma comparação empírica atual.

A regra histórica de simulação de 50/200/500 sequências conforme faixas do tamanho de origem está registrada no notebook original. Ela não comprova que essas proporções estejam presentes nos dados sobreviventes. C02 mede os comprimentos e pesos observados diretamente.

## Limites de inferência

O conjunto principal é de contigs longos; sua avaliação não demonstra generalização para genomas não vistos. `seen_genome` permanece `unknown` sem accession ou vínculo confiável entre query e referência de treino. `origin_len`, identidade taxonômica e compartilhamento de k-mers não substituem essa evidência.

Diferenças de média de logits e voto, magnitudes, margens, exclusões por comprimento e calibração são diagnósticos operacionais. Nenhuma delas, isoladamente, prova que arquitetura, transferência, calibração ou composição amostral causou a diferença observada. Diagnósticos sem logits ou suporte suficiente são `INCONCLUSIVE`; resultados sintéticos de smoke não são evidência empírica.

## Contrato técnico preservado

C02 usa janela 512 bp, stride 256 bp, descarta a cauda incompleta, desempata voto pelo menor índice e mantém média de logits antes de softmax e voto majoritário. `pad` e `skip` são avaliados separadamente. A equivalência em caminho de janela única é verificada em FP32 com tolerância máxima de `1e-5`.

Consulte [AD-020](adr/AD-020-long-sequence-evidence-boundary.md) e [o estado de evidência C02](long_sequence_audit.md). A reprodução empírica permanece pendente até que dados remotos, mapeamento de ordem Species e os cinco checkpoints estejam disponíveis.
