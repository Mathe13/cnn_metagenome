| model | taxonomic_level | parameters_m | accuracy | top_5 | macro_f1 | latency_ms |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet50 (Species, Clean) | Species | 23.5 | 84.88 | 96.65 | 84.12 | 1.45 |
| ResNet50 (Species, Noise Fine-tuned) | Species | 23.5 | 80.12 | 94.3 | 79.55 | 1.45 |
| ConvNeXt_Z (Species, Zero) | Species | 28.1 | 59.18 | 82.45 | 58.2 | 2.1 |
| ConvNeXt_T (Species, Transfer) | Species | 28.1 | 60.83 | 83.9 | 59.75 | 2.1 |
| ConvNeXt_F (Species, Noise Fine-tuned) | Species | 28.1 | 61.14 | 84.22 | 60.05 | 2.1 |
| ResNet50 (Genus, Clean) | Genus | 23.5 | 87.42 | 97.8 | 86.95 | 1.42 |
| ConvNeXt (Genus, Clean) | Genus | 28.1 | 85.0 | 92.5 | 84.0 | 2.05 |