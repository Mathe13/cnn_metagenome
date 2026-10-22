# S09: Calibration and Confidence by Noise Level

## Problem

The manuscript and historical notebooks (notably `long/cnn-long-voter-compare.ipynb` and `short/analize_convnext.ipynb`) evaluate model confidence by computing mean Top-1 predicted probability (`Confianca_Media`) and plotting it directly against accuracy in scatter charts labeled "Gráfico de Calibração" or "Diagrama de Confiabilidade".
However, several major scientific and methodological deficiencies affect these analyses:

1. **Top-1 Confidence vs. Calibration Conflation**: Top-1 confidence (mean maximum predicted probability) and prediction entropy do not measure calibration. A model can exhibit high average confidence (>85%) while suffering from severe overconfidence and large Expected Calibration Error (ECE), particularly on erroneous predictions or under high input noise.
2. **Absence of Formal Calibration Metrics**: The codebase lacks formal probabilistic calibration metrics: Expected Calibration Error (ECE) under declared binning rules, Maximum Calibration Error (MCE), Negative Log-Likelihood (NLL), and multiclass Brier score.
3. **Confounding Architecture with Training Hyperparameters**: The manuscript attributes differences in confidence and robustness between ResNet50 and ConvNeXt to neural architecture. However, both models were trained with `label_smoothing=0.1`, which mathematically shrinks logit margins and inherently lowers ECE (Müller et al., 2019). Furthermore, ConvNeXt used dropout 0.7 and multi-phase transfer from Genus, whereas ResNet50 used dropout 0.5. Evaluating calibration without isolating the effects of label smoothing, dropout, and training schedules creates confounded causal claims.
4. **Post-Hoc Calibration Data Leakage Risk**: Post-hoc probability calibration (e.g., Temperature Scaling) must learn parameters strictly on a dedicated validation split. If temperature parameters are fitted on test queries, test set integrity is compromised.
5. **Divergence of Confidence and Accuracy Under Controlled Noise**: As nominal sequence noise $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$ increases, true accuracy drops substantially. If model confidence decays at a slower rate than accuracy, the model becomes increasingly overconfident and miscalibrated. The manuscript must moderate any claim of "robust confidence" if calibration error escalates under corruption.

When this ships, `src/viral_cnn/eval/calibration.py` provides auditable functions for declared-bin ECE, MCE, multiclass NLL, memory-efficient multiclass Brier score, reliability diagram generation, and validation-only temperature scaling; `scripts/run_calibration_evaluation.py` provides an auditable CLI; `docs/calibration_and_confidence_audit.md` provides an exhaustive scientific audit evaluating calibration across models and noise levels, isolating label smoothing and hyperparameter confounds, and establishing text modifications where confidence and calibration diverge; and contract tests verify all calculations deterministically.

## Flow

Reuses `src/viral_cnn/eval/metrics.py`, `src/viral_cnn/eval/noise.py`, and `src/viral_cnn/models/registry.py` while keeping existing evaluation pipelines intact.

1. model logits/probabilities and ground-truth labels -> `viral_cnn.eval.calibration:compute_calibration_metrics` (new, door 1) - computes Top-1 ECE, MCE, multiclass NLL, and multiclass Brier score under declared binning rules
2. probabilities, ground-truth labels, and bin configuration -> `viral_cnn.eval.calibration:compute_reliability_diagram` (new, door 2) - computes bin-level accuracy, confidence, sample counts, and calibration gaps for equal-width or equal-mass binning
3. validation logits and validation labels -> `viral_cnn.eval.calibration:TemperatureScaler.fit` (new, door 3) - optimizes scalar temperature $T > 0$ strictly on validation data via NLL minimization
4. calibrated temperature $T$ and test logits -> `viral_cnn.eval.calibration:TemperatureScaler.transform` (new, door 4) - applies $z / T$ scaling to test logits and computes post-calibration ECE, NLL, and Brier metrics
5. evaluation predictions across models and noise $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$ -> `scripts/run_calibration_evaluation.py` (new, no door - placement per conventions) - CLI orchestrating calibration evaluation, reliability diagram export, and multi-model comparison
6. out: calibration summary JSON, reliability diagram tables/plots, and scientific audit report in `docs/calibration_and_confidence_audit.md`

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `ExpectedCalibrationError` - difference between empirical accuracy and confidence weighted across $M$ declared bins ($\sum_{m=1}^M \frac{\|B_m\|}{N} \|\text{acc}(B_m) - \text{conf}(B_m)\|$), lives in `src/viral_cnn/eval/calibration.py` |
| domain | new term: `MulticlassBrierScore` - strictly proper scoring rule measuring probability calibration across all $K$ classes ($\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - \mathbf{1}_{y_i = k})^2$), computed via memory-efficient expansion without dense one-hot allocation |
| domain | new term: `ValidationTemperatureScaling` - post-hoc logit scaling ($z / T$) where temperature $T > 0$ is fitted exclusively on validation logits and evaluated on test queries |
| domain | new term: `OverconfidenceGap` - mean difference between confidence and accuracy ($\text{conf}_i - \mathbf{1}_{y_i = \hat{y}_i}$) partitioned by correct and incorrect predictions |
| stored data | nothing to migrate: historical notebooks and parquet splits remain unmodified; calibration artifacts write to structured output directories |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Declared binning rule specification | `binning_strategy: Literal["equal_width", "equal_mass"] = "equal_width"` with explicit `n_bins: int = 15` spanning $[0.0, 1.0]$ | implicit or undocumented bin edges - rejected because ECE value depends strongly on bin count and binning scheme |
| Memory-efficient multiclass Brier score calculation | `np.mean(np.sum(probs**2, axis=1) - 2.0 * probs[np.arange(N), y_true] + 1.0)` | allocating dense $N \times 13683$ one-hot array in memory - rejected because it causes out-of-memory errors on large test sets (e.g. 1.7M rows) |
| Strict validation-only temperature fitting | `TemperatureScaler.fit(val_logits, val_labels)` taking explicit validation inputs and raising an error if test logits are passed or unvalidated | fitting temperature on test predictions - rejected because tuning calibration parameters on the test set is data leakage |
| Explicit separation of confidence and calibration | documentation and metrics reporting `top1_confidence` separately from `ece`, `nll`, and `brier_score`, with automated divergence flag | labeling mean confidence as calibration - rejected because high confidence under miscalibration is the exact error cautioned against in AGENTS.md |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Formal Probabilistic Calibration Metrics (P1)

Computation of Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Negative Log-Likelihood (NLL), and multiclass Brier score under explicit binning rules.

**Acceptance Criteria**

1. The system SHALL compute Top-1 Expected Calibration Error (ECE) using an explicitly declared binning rule (`equal_width` with $M=15$ bins by default, or `equal_mass`), where the sum of bin weights strictly equals 1.0.
2. The system SHALL compute Maximum Calibration Error (MCE) as the maximum absolute difference between accuracy and confidence across all non-empty bins.
3. The system SHALL compute multiclass Negative Log-Likelihood (NLL) with numerical clamping ($\varepsilon = 10^{-15}$) to prevent negative infinity.
4. The system SHALL compute the exact multiclass Brier score $\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - \mathbf{1}_{y_i=k})^2$ using the expanded identity $\sum_k p_{ik}^2 - 2 p_{i, y_i} + 1$ without materializing a dense $N \times K$ one-hot matrix.
5. WHEN all probabilities equal 1.0 on correct predictions THEN the system SHALL return ECE equal to 0.0, MCE equal to 0.0, and Brier score equal to 0.0.
6. The system SHALL calculate mean confidence and empirical accuracy separately for correct and incorrect predictions to quantify overconfidence.

**Independent test:** `pytest tests/test_calibration.py -k test_calibration_metrics`

### S2: Reliability Diagrams and Bin Analysis (P2)

Generation of structured reliability tables and publication-ready reliability diagram plots across declared bins.

**Acceptance Criteria**

7. WHEN generating a reliability diagram THEN the system SHALL partition predictions into declared confidence bins and return a structured table containing bin edges, sample count, mean confidence, empirical accuracy, and calibration gap per bin.
8. The system SHALL ensure that every sample is allocated to exactly one bin and the sum of sample counts across bins strictly equals the total sample count $N$.
9. The system SHALL export reliability diagrams as both structured JSON/CSV data and a plot illustrating empirical accuracy versus mean confidence against the diagonal identity line, accompanied by a sample count histogram.

**Independent test:** `pytest tests/test_calibration.py -k test_reliability_diagram`

### S3: Validation-Only Temperature Scaling (P3)

Post-hoc calibration via Temperature Scaling with strict enforcement of train/validation parameter estimation.

**Acceptance Criteria**

10. WHEN fitting temperature scaling THEN the system SHALL optimize a single scalar temperature $T > 0$ strictly minimizing NLL on validation logits.
11. The system SHALL enforce that temperature fitting requires validation data and SHALL reject fitting directly on test data without an explicit validation split.
12. WHEN applying a fitted temperature scaler to test logits THEN the system SHALL scale logits by $z / T$ before softmax and compute post-calibration ECE, NLL, and Brier score.
13. IF an already well-calibrated input has optimal $T=1.0$ THEN the system SHALL leave probabilities and predictions unchanged.

**Independent test:** `pytest tests/test_calibration.py -k test_temperature_scaling`

### S4: Controlled Noise Calibration Audit and Scientific Moderation (P4)

Multi-model evaluation across noise levels $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$, analysis of `label_smoothing` and training confounds, and manuscript text moderation.

**Acceptance Criteria**

14. The system SHALL evaluate calibration metrics (ECE, MCE, NLL, Brier, Top-1 confidence, and accuracy) across all published species model variants (ResNet50 clean/noise, ConvNeXt zero/transfer/noise) across nominal noise levels $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$.
15. The system SHALL generate an exhaustive scientific audit report in `docs/calibration_and_confidence_audit.md` documenting:
    - Calibration metrics table across models and noise levels.
    - Explicit distinction between Top-1 confidence and calibration error.
    - Disentanglement of architectural effects from `label_smoothing=0.1`, dropout rates, and restart schedules.
    - Concrete recommendations for amending manuscript text where confidence remains high while accuracy degrades under noise.
16. The system SHALL flag any condition where mean confidence exceeds accuracy by more than 10 percentage points as an overconfidence regime.

**Independent test:** `pytest tests/test_calibration.py -k test_noise_calibration_audit`

### S5: CLI Entrypoint and Reproducible Manifests (P5)

Standalone command-line interface for running calibration evaluations and emitting reproducible artifacts.

**Acceptance Criteria**

17. The system SHALL provide a CLI entrypoint `scripts/run_calibration_evaluation.py` supporting arguments `--input`, `--val-input`, `--model`, `--noise-rates`, `--n-bins`, `--binning`, `--apply-temperature-scaling`, and `--output-dir`.
18. WHEN executed via CLI THEN the system SHALL output `calibration_metrics.json`, `reliability_diagram.csv`, `reliability_diagram.png`, and `run_manifest.json` containing deterministic seeds, input hashes, and model configurations.

**Independent test:** `pytest tests/test_calibration.py -k test_cli_calibration`

## Out of scope

| Excluded | Why |
| --- | --- |
| Non-parametric isotonic regression across 13,683 classes | Isotonic regression on high-cardinality multi-class output overfits severely without massive validation data; temperature scaling is the standard monotonic post-hoc method |
| Retraining full 1D-CNN models from scratch without label smoothing | Re-executing GPU pretraining across multiple seeds is deferred to S12 controlled architecture experiments; S09 audits and isolates the impact of existing label smoothing analytically and experimentally |
| Real-time streaming calibration adjustments | Metagenomic taxonomic classification operates on batched reads and offline sequences; streaming calibration is unnecessary |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Default number of calibration bins | $M=15$ equal-width bins in $[0, 1]$ | 15 bins is standard in calibration literature (Guo et al. 2017) and provides sufficient resolution for high-accuracy predictions | y |
| Bin boundary convention | Left-closed, right-open $[b_i, b_{i+1})$, except the final bin which is closed on both ends $[b_{M-1}, 1.0]$ | Prevents probability 1.0 from falling out of bounds while avoiding double-counting | y |
| Handling of empty bins | Excluded from ECE weighted sum (weight $|B_m|/N = 0$) and excluded from MCE max computation | Empty bins have no empirical accuracy or samples; standard formulation sets contribution to 0 | y |
| Temperature scaling optimization method | L-BFGS or Adam on scalar log temperature $\ln(T)$ | Constrains $T > 0$ naturally without boundary violations and converges quickly | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- `ROADMAP_SPECS_ARTIGO.md` - S09 specification: Calibração e confiança por nível de ruído
- `AGENTS.md` - Invariantes científicos: "Confiança Top-1 e entropia não equivalem a calibração; só use esse termo quando ECE, Brier, NLL ou outra medida formal tiver sido calculada"
- Guo et al., 2017 ("On Calibration of Modern Neural Networks", ICML 2017) - ECE, Reliability Diagrams, and Temperature Scaling definitions
- Müller et al., 2019 ("When Does Label Smoothing Help?", NeurIPS 2019) - Label smoothing impact on calibration and logit representations

