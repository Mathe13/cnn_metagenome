| model | taxonomic_level | backbone_m | fc_params_m | parameters_m | accuracy | top_5 | macro_f1 | latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet50 (Species, Clean) | Species | 15.96 | 28.04 | 43.99 | 84.88 | 96.65 | 84.12 | 1.45 |
| ResNet50 (Species, Noise Fine-tuned) | Species | 15.96 | 28.04 | 43.99 | 80.12 | 94.3 | 79.55 | 1.45 |
| ConvNeXt_Z (Species, Zero) | Species | 135.98 | 28.04 | 164.01 | 59.18 | 82.45 | 58.2 | 2.1 |
| ConvNeXt_T (Species, Transfer) | Species | 135.98 | 28.04 | 164.01 | 60.83 | 83.9 | 59.75 | 2.1 |
| ConvNeXt_F (Species, Noise Fine-tuned) | Species | 135.98 | 28.04 | 164.01 | 61.14 | 84.22 | 60.05 | 2.1 |
| ResNet50 (Genus, Clean) | Genus | 15.96 | 5.54 | 21.49 | 87.42 | 97.8 | 86.95 | 1.42 |
| ConvNeXt (Genus, Clean) | Genus | 135.98 | 5.54 | 141.51 | 85.0 | 92.5 | 84.0 | 2.05 |