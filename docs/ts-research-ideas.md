# Time-Series / Time-Series-Classification Research Ideas

*Compiled 2026-06-27. Grounded in a scan of recent NeurIPS/ICML/ICLR/KDD/AAAI/VLDB/AISTATS and TKDE/DAMI/JMLR work (citations are rough — verify before use).*

Every idea below is filtered against three bars: **RARE** (names the closest work and the gap it leaves), **RIGOROUS** (falsifiable hypothesis + named datasets + honest baselines), **USEFUL** (a named beneficiary who can't already do this). Ideas I judged probably-already-done were cut, not padded in — see the *DO NOT pursue* section for the tempting ones I dropped.

---

## State of the field (the crowding map)

**Saturated / diminishing returns — deliberately avoided below:**
- **Random convolutional kernels.** ROCKET (Dempster et al., DAMI 2020) → MiniRocket → MultiRocket → Hydra (DAMI 2023) → Quant (Dempster 2023) → KG-MTP (2025) → Hydra+Quant meta-learning (arXiv:2512.06666, 2025). Bake-offs now show parity; the design space is mined out.
- **Contrastive SSL with new augmentations.** TS2Vec (AAAI 2022) → SoftCLT (ICLR 2024) → parametric augmentation, bad-pair mining, uniformity-tolerance balancing (2024–25). Marginal gains, reviewer fatigue.
- **Forecasting transformer architectures.** PatchTST/iTransformer/Crossformer/BasisFormer lineage — extremely crowded, and DLinear-style linear baselines remain competitive.
- **LLM reprogramming for forecasting.** Time-LLM (ICLR 2024) is highly cited but under heavy skepticism (Tan et al., NeurIPS 2024, "Are Language Models Actually Useful for Time Series Forecasting?"; modality-alignment critique arXiv:2410.12326).

**Genuinely open — where the ideas below live:**
- In-context / zero-shot **classification** (Rethinking Zero-Shot TSC, arXiv:2602.00620; MantisV2, arXiv:2602.17868) — months old, sparse.
- Test-time adaptation **for classification** (explicitly "rarely studied" outside vision/forecasting).
- Benchmark **contamination/leakage** auditing for TS foundation models — done for LLMs, not for TS.
- **Common-corruption** robustness — there is no ImageNet-C analogue for time series.
- **Manifold-aware** explanation faithfulness; **data-centric** archive auditing; **shortcut-learning** audits — all standard in vision/NLP, largely absent in TS.

### Coverage table (the spread)

| # | Title (short) | Primary axis | Effort |
|---|---|---|---|
| 1 | Per-dataset temporal-dependence score + leaderboard correction | Evaluation/benchmarking | one-paper |
| 2 | Contamination audit of "zero-shot" TS foundation models | Evaluation/benchmarking | one-paper |
| 3 | Data cartography for UCR/UEA/MONSTER (label noise + pruning) | Evaluation/benchmarking | one-paper |
| 4 | Test-time adaptation for TSC: covariate vs. concept shift | Distribution shift | one-paper |
| 5 | Spurious-channel identification under shift in MTSC | Distribution shift | one-paper |
| 6 | Shortcut-learning audit for TSC (device/subject signatures) | Distribution shift / robustness | one-paper |
| 7 | In-context TSC scaling laws (when does it beat ROCKET?) | Foundation models | one-paper |
| 8 | Failure modes of forecasting-pretrained features for TSC | Foundation models | one-paper |
| 9 | Programmatic weak supervision via shapelet labeling functions | Label-efficiency | one-paper |
| 10 | Self-supervised warp/morphology disentanglement | Representation learning | one-paper → multi-year |
| 11 | A principled kernel-sampling distribution for ROCKET (theory) | Representation learning | multi-year |
| 12 | Manifold-aware faithfulness metrics for TS explanations | Interpretability & trust | one-paper |
| 13 | Concept-vector (TCAV-style) explanations for TSC | Interpretability & trust | one-paper |
| 14 | TSC-C: a physically-grounded common-corruption benchmark | Deployment/robustness | one-paper |
| 15 | Conformal classification under temporal drift | Deployment/robustness | one-paper |
| 16 | Do long-series classifiers use long context? | Long-horizon / irregular | one-paper |
| 17 | Is complex temporal modeling needed for irregular MTSC? | Irregular / multivariate | one-paper |

---

## The ideas

### 1. A continuous temporal-dependence score for TSC benchmarks, with a leaderboard correction
**Primary axis:** Evaluation & benchmarking critiques

1. **Thesis:** Estimate, per dataset, the fraction of achievable accuracy that depends on temporal *ordering* (vs. order-invariant tabular features), then re-rank classifiers on the genuinely temporal subset and show "SOTA" gains partly evaporate.
2. **Gap:** "Revisit TSC Benchmark: The Impact of Temporal Information" (arXiv:2503.20264, 2025) flags that many UCR datasets are effectively tabular and proposes *UCR-Augmented*, but it gives a binary-ish framing, not a continuous, significance-tested *temporal-information statistic* per dataset, and doesn't quantify how much each method family's reported edge is attributable to non-temporal signal.
3. **Hypothesis (falsifiable):** A permutation/shuffle-based estimator of order-dependence will show ≥30% of UCR datasets have negligible temporal dependence; on the temporal-only subset, the accuracy gap between the best deep model and a bag-of-features baseline shrinks by >50%.
4. **Method sketch:** For each dataset, fit a strong order-invariant baseline (TSFresh/FreshPRINCE / catch22 + ridge) and a strong temporal model (InceptionTime/Hydra). Define temporal-dependence as the accuracy drop of the temporal model under within-series permutation of timesteps, normalized; bootstrap CIs. Build a registry; reweight published leaderboards by temporal-dependence.
5. **Datasets + baselines:** UCR-128, UEA-30, MONSTER. Baselines: catch22+ridge, FreshPRINCE, 1NN-DTW, Hydra, InceptionTime, HIVE-COTE 2.0.
6. **Evaluation:** Spearman rank-correlation between original and corrected leaderboards; count of rank inversions; report which methods lose most. **Refutes** if temporal-dependence is uniformly high (archive is fine) or rankings are stable.
7. **RARE/RIGOROUS/USEFUL:** Rare = a *quantitative, per-dataset* statistic, not a binary flag. Rigorous = permutation test with CIs and a pre-registered subset. Useful = lets reviewers and authors stop overclaiming temporal modeling on tabular data.
8. **Feasibility:** Data fully public; trivial compute. Main risk: 2503.20264 already claims part of this — you must differentiate sharply on the *statistic* and the leaderboard reweighting, or it reads as incremental.
9. **Effort tier:** one-paper.

### 2. Contamination audit: are "zero-shot" TS foundation models actually zero-shot?
**Primary axis:** Evaluation & benchmarking critiques

1. **Thesis:** TSFMs (Chronos, Moirai, TimesFM, Mantis) are reported "zero-shot" on UCR/UEA, but their pretraining corpora may overlap with those archives; measure the leakage and the inflation it causes.
2. **Gap:** "TS Foundation Models: Benchmarking Challenges and Requirements" (arXiv:2510.13654, 2025) raises evaluation concerns abstractly, but no one has run an LLM-style *train–test contamination* audit (membership inference + near-duplicate shape detection between pretraining corpora and evaluation archives) for time series.
3. **Hypothesis:** For models whose pretraining corpora are public, ≥X% of evaluation series have a near-duplicate in pretraining, and zero-shot accuracy on contaminated datasets exceeds that on provably-novel datasets by a statistically significant margin.
4. **Method sketch:** (a) Near-duplicate detection across corpora using z-normalized DTW/SAX-hash with a tuned threshold validated on synthetic dup/non-dup pairs. (b) Membership-inference via loss/embedding-distance gap. (c) Curate a "provably novel" hold-out (datasets released after model cutoff, e.g., new MONSTER tasks or freshly collected sensor data). Compare zero-shot accuracy contaminated vs. novel.
5. **Datasets + baselines:** Mantis/MantisV2, Chronos, Moirai, TimesFM embeddings → linear probe / 1-NN, on UCR/UEA + a post-cutoff novel set. Baseline reference: ROCKET trained in-domain (a true no-leakage point).
6. **Evaluation:** Contamination rate per dataset; paired accuracy gap (contaminated vs. novel) with bootstrap CI; correlation between per-dataset contamination and per-dataset zero-shot gain. **Refutes** if no correlation.
7. **RARE/RIGOROUS/USEFUL:** Rare = nobody has done TS contamination auditing. Rigorous = validated detector + held-out novel set. Useful = recalibrates the entire "zero-shot TSC" narrative; vendors and reviewers need this.
8. **Feasibility:** Chronos/Moirai/TimesFM corpora partly documented; Mantis is synthetic-pretrained (a useful contrast). Risk: some corpora are not fully public, so the audit is partial — frame as a lower bound.
9. **Effort tier:** one-paper.

### 3. Data cartography for TS archives: label noise, redundancy, and principled pruning
**Primary axis:** Evaluation & benchmarking critiques (data-centric)

1. **Thesis:** Port dataset-cartography / confident-learning to time series to map every UCR/UEA/MONSTER example as easy / hard / mislabeled, surface label errors, and show archives can be pruned heavily with no accuracy loss.
2. **Gap:** Dataset Cartography (Swayamdipta et al., EMNLP 2020) and confident learning / cleanlab (Northcutt et al., JAIR 2021) are standard in NLP/vision; nobody has produced training-dynamics data maps or a systematic label-error estimate for TS archives, despite well-known anecdotal UCR label issues.
3. **Hypothesis:** ≥X% of archive examples are flagged high-confidence-mislabeled and verifiable by hand; pruning the redundant/easy 40–60% leaves test accuracy unchanged; correcting flagged labels changes published rankings on ≥k datasets.
4. **Method sketch:** Train an ensemble (InceptionTime + Hydra + ridge-on-catch22), log per-epoch confidence/variability → data map. Apply confident-learning to estimate label-error indices. Manually adjudicate a sample. Re-train on pruned/cleaned sets.
5. **Datasets + baselines:** UCR-128, UEA-30, MONSTER. Baselines = full-data training; random pruning; coreset baselines.
6. **Evaluation:** Precision of label-error flags vs. human adjudication; accuracy retention vs. pruning fraction; ranking shifts after relabeling. **Refutes** if flags are no better than chance or pruning hurts.
7. **RARE/RIGOROUS/USEFUL:** Rare = data-centric lens basically untouched in TS. Rigorous = human-verified precision + retention curves. Useful = cheaper, cleaner benchmarks; guidance for data collection.
8. **Feasibility:** All public; moderate compute. Risk: human adjudication of TS is harder than text/images (you can't always "see" the right label) — recruit domain experts for ECG/HAR subsets.
9. **Effort tier:** one-paper.

### 4. Test-time adaptation for TSC that distinguishes covariate from concept shift
**Primary axis:** Distribution shift / non-stationarity

1. **Thesis:** Build the first TSC-specific TTA benchmark that *separates* sensor covariate shift from label-conditional (concept) shift, plus a method that detects which is occurring before adapting — because blind entropy-minimization fails under label shift.
2. **Gap:** TTA is mature in vision and emerging for TS *forecasting* (TAFAS, DynaTTA, 2025) and *anomaly detection* (CANDI, arXiv:2604.01845), but for **classification** it is, per recent surveys, "rarely studied," and no benchmark disentangles shift types.
3. **Hypothesis:** A shift-type detector + conditional adaptation rule retains >Y% more accuracy than TENT-style entropy minimization under label-conditional shift, while matching it under pure covariate shift.
4. **Method sketch:** Construct controlled shifts: covariate (sensor gain/placement, cross-device) vs. concept (cross-subject/cross-hospital where p(y|x) changes). Detector = compare feature-distribution shift (e.g., BN-stat divergence) against pseudo-label-distribution shift. Route to BN-adaptation, prototype refinement, or abstention accordingly.
5. **Datasets + baselines:** Cross-subject HAR (PAMAP2, Opportunity, DSADS), cross-hospital/lead ECG (PTB-XL, Chapman, MIT-BIH), SleepEDF cross-cohort. Baselines: source-only, TENT, T3A, pseudo-label self-training, DynaTTA-adapted.
6. **Evaluation:** Accuracy retention and calibration (ECE) per shift type; ablate the detector (oracle vs. learned); failure analysis on label-shift cases. **Refutes** if the detector adds nothing over uniform TTA.
7. **RARE/RIGOROUS/USEFUL:** Rare = TTA-for-TSC gap is explicit. Rigorous = factorized shift design + oracle ablation. Useful = wearables/clinical models that drift across users and sites.
8. **Feasibility:** Datasets public, modest compute. Risk: cleanly separating covariate vs. concept shift in real data is hard — lean on synthetic-shift control plus a few real cross-site splits.
9. **Effort tier:** one-paper.

### 5. Identifying *spurious channels* under distribution shift in multivariate TSC
**Primary axis:** Distribution shift / non-stationarity

1. **Thesis:** Channel independence helps OOD generalization wholesale; go finer — *identify and drop the specific channels whose information is spurious/shifted*, beating both channel-dependence and blanket channel-independence under shift.
2. **Gap:** "Channel Independence Improves OOD Generalisation in MTSC" (OpenReview, 2025) shows CI > CD for robustness but treats it as all-or-nothing; nobody learns a *per-deployment channel subset* tied to which channels are spurious, nor diagnoses spuriousness causally.
3. **Hypothesis:** A learned channel-relevance score (invariance across environments) will single out spurious channels; masking them yields higher OOD accuracy than CI, and the dropped channels are the shifted ones (verified on synthetic injections).
4. **Method sketch:** Multi-environment training (IRM/REx-style penalty per channel); rank channels by cross-environment stability of their marginal contribution; mask low-stability channels at test time. Validate the diagnosis by injecting synthetic spurious channels with known shift.
5. **Datasets + baselines:** UEA multivariate with subject/device splits; HAR multi-sensor; EEG montages. Baselines: CD backbone, full CI, random channel drop, mutual-info channel selection.
6. **Evaluation:** OOD accuracy vs. channels kept; precision of spurious-channel identification on synthetic injections. **Refutes** if selective masking ≤ blanket CI, or if "spurious" channels aren't recovered.
7. **RARE/RIGOROUS/USEFUL:** Rare = selective, diagnostic extension of an all-or-nothing result. Rigorous = synthetic ground truth for the diagnosis. Useful = sensor-array deployments (deciding which sensors to trust/cut).
8. **Feasibility:** Public data; modest compute. Risk: borderline-incremental on the CI paper — the *spurious-channel identification with synthetic ground truth* must carry the novelty.
9. **Effort tier:** one-paper.

### 6. A shortcut-learning audit for time-series classifiers
**Primary axis:** Distribution shift / robustness

1. **Thesis:** Deep TSC models exploit shortcuts — recording-device signatures, baseline wander, subject identity, acquisition artifacts — rather than the physiological/physical signal; build a diagnostic suite and show the resulting OOD collapse.
2. **Gap:** Shortcut learning is a named, well-studied failure in vision (Geirhos et al., Nat. Mach. Intell. 2020) but is barely articulated for TSC; the channel-independence OOD paper touches robustness without naming/measuring shortcuts.
3. **Hypothesis:** State-of-the-art ECG/HAR classifiers achieve above-chance accuracy on *the wrong target* (predicting device/subject from the signal) and lose >Z% accuracy when the shortcut is decorrelated from the label.
4. **Method sketch:** (a) Probe: can a classifier predict device/site/subject from learned features? (b) Decorrelation test: re-split so shortcut ⟂ label, measure drop. (c) Mitigations: group-DRO, domain-adversarial removal, channel-independence; report robustness recovered.
5. **Datasets + baselines:** PTB-XL (multi-device ECG), Chapman/Shaoxing, MIT-BIH; HAR (subject as shortcut); EEG (montage/site). Baselines: ERM, IRM, group-DRO, DANN.
6. **Evaluation:** Shortcut-predictability AUC; ERM accuracy drop under decorrelated splits; mitigation recovery. **Refutes** if features carry no shortcut signal or accuracy is unaffected by decorrelation.
7. **RARE/RIGOROUS/USEFUL:** Rare = shortcut framing is essentially new for TSC. Rigorous = probe + counterfactual re-split + mitigation. Useful = directly relevant to clinical-AI reliability and regulatory scrutiny.
8. **Feasibility:** Metadata (device/site/subject) exists in clinical archives. Risk: obtaining clean device/site labels for some datasets; start with PTB-XL where metadata is rich.
9. **Effort tier:** one-paper.

### 7. In-context TSC scaling laws: when does in-context beat fitting a ROCKET?
**Primary axis:** Foundation models for TS

1. **Thesis:** Map the regime — context size, number of classes, intra-class diversity, series length — where in-context classification with a TSFM beats simply training ROCKET/Hydra from scratch (which takes ~1 second).
2. **Gap:** "Rethinking Zero-Shot TSC: From Task-specific Classifiers to In-Context Inference" (arXiv:2602.00620, 2026) and MantisV2 (arXiv:2602.17868) propose in-context *methods*, but nobody charts the *decision boundary* that tells a practitioner which approach to use — the actionable scaling law.
3. **Hypothesis:** In-context wins only in a bounded regime (few shots per class, low intra-class diversity); beyond a crossover context size, a from-scratch ROCKET fit matches or beats it at far lower cost, and the crossover is predictable from a cheap diversity statistic.
4. **Method sketch:** Sweep shots/class, #classes, length across many datasets for each TSFM in-context vs. ROCKET/Hydra/1NN-DTW trained on the same shots. Fit accuracy-vs-resource curves; derive a crossover predictor from dataset statistics.
5. **Datasets + baselines:** UCR/UEA/MONSTER subsampled to controlled shot counts. Baselines: ROCKET, Hydra, 1NN-DTW, TS2Vec linear probe.
6. **Evaluation:** Accuracy vs. (shots, compute) frontiers; predictive accuracy of the crossover rule on held-out datasets; wall-clock and $ cost. **Refutes** if in-context dominates everywhere (then it's just better) or never (then it's a non-starter).
7. **RARE/RIGOROUS/USEFUL:** Rare = scaling-law/decision-rule framing, not another method. Rigorous = controlled sweeps + held-out crossover prediction. Useful = a concrete "use X when Y" rule practitioners can apply today.
8. **Feasibility:** Public data; inference-only (no pretraining). Risk: the in-context area is heating fast — ship quickly; the durable contribution is the *decision rule*, not the leaderboard.
9. **Effort tier:** one-paper.

### 8. Failure modes of forecasting-pretrained features for classification
**Primary axis:** Foundation models for TS

1. **Thesis:** Forecasting-pretrained TSFM features are good zero-shot classifiers *on average* but predictably fail on class structures the forecasting objective can't encode (amplitude/offset-invariant or purely distributional classes); characterize and predict these failures.
2. **Gap:** "Pre-trained Forecasting Models: Strong Zero-Shot Feature Extractors for TSC" (arXiv:2510.26777, 2025) shows that forecasting features *work* but stops at "it works"; the *when/why it fails* — a design guideline — is missing.
3. **Hypothesis:** Because next-step forecasting rewards reconstructing levels/trends, forecasting features will underperform on classes distinguished by amplitude-invariant shape, by higher-order/distributional properties, or by long-range phase — and a synthetic taxonomy will predict per-dataset transfer success.
4. **Method sketch:** Build synthetic class families isolating each property (shape-only vs. level-only vs. variance-only vs. phase-only). Probe forecasting-pretrained features; map failure curves. Then predict real-dataset transfer from where each dataset sits in the taxonomy.
5. **Datasets + baselines:** Synthetic suite + UCR/UEA. Models: TimesFM/Chronos/Moirai features vs. classification-pretrained Mantis vs. ROCKET.
6. **Evaluation:** Per-property accuracy; correlation between taxonomy placement and real transfer gap. **Refutes** if forecasting features are uniformly strong across all synthetic properties.
7. **RARE/RIGOROUS/USEFUL:** Rare = a characterization/negative-result study, not "we beat SOTA." Rigorous = synthetic ground-truth properties. Useful = tells practitioners when to reach for forecasting vs. classification foundation models.
8. **Feasibility:** Inference-only; cheap. Risk: close enough to 2510.26777 that you must lead with the *failure taxonomy*, not re-confirm the positive result.
9. **Effort tier:** one-paper.

### 9. Programmatic weak supervision for TS via shapelet/motif labeling functions
**Primary axis:** Label-efficiency (weak supervision)

1. **Thesis:** Bring Snorkel-style programmatic weak supervision to time series: experts write *labeling functions* as shapelet/motif/threshold rules, a label model denoises them, and the result beats SSL pretraining in low-label regimes.
2. **Gap:** Weak supervision is mature in text/tabular (Snorkel; interactive weak supervision, 2021) but a *formal labeling-function framework for time series* (motif-, shapelet-, derivative-, threshold-based rules + a TS-aware label model) is essentially absent; SSL4TS reviews don't cover it.
3. **Hypothesis:** With 0 hand labels but k expert rules, a denoised weak-supervision model matches a supervised model trained on N hand labels (N ≫ k), and beats TS2Vec linear-probe at equal expert effort.
4. **Method sketch:** Define a labeling-function API over shapelets/motifs/spectral thresholds; learn LF accuracies/correlations with a generative label model adapted for temporally-correlated abstentions; train an end classifier on probabilistic labels.
5. **Datasets + baselines:** ECG arrhythmia (PTB-XL), HAR (PAMAP2), industrial fault detection, sleep staging — domains with real expert rules. Baselines: supervised at matched labels, TS2Vec/SoftCLT linear probe, self-training.
6. **Evaluation:** Accuracy vs. expert-effort (rules written / labels given); robustness to LF noise; ablate the label model. **Refutes** if denoised weak labels never beat naive majority-vote or supervised-at-equal-effort.
7. **RARE/RIGOROUS/USEFUL:** Rare = LF framework for TS is missing. Rigorous = effort-matched comparison + label-model ablation. Useful = medical/industrial settings where experts have rules but not labels.
8. **Feasibility:** Datasets public; rules elicitable from literature. Risk: writing good LFs needs domain expertise — partner with a clinician/engineer or mine rules from guidelines.
9. **Effort tier:** one-paper.

### 10. Self-supervised disentanglement of warping ("when") from morphology ("what")
**Primary axis:** Representation learning

1. **Thesis:** An SSL objective that factorizes a series into a *warping field* and a *canonical shape*, yielding representations invariant to nuisance time-warping but sensitive to morphology.
2. **Gap:** Soft-DTW and DTW-based augmentation handle alignment as a loss or a transform; TS2Vec and friends learn entangled representations with no explicit warp/shape factorization. No SSL method produces a disentangled (warp, shape) latent.
3. **Hypothesis:** Explicit warp/shape disentanglement improves (a) robustness to temporal jitter/speed change and (b) alignment-free retrieval, vs. entangled SSL, *without* losing classification accuracy.
4. **Method sketch:** Encoder → (shape code, monotonic warp field via a differentiable time-warp module); reconstruct via warp∘canonical-shape; SSL loss = reconstruction + invariance of shape code to applied synthetic warps + smoothness/monotonicity on the warp field. Evaluate the shape code as the classification representation.
5. **Datasets + baselines:** UCR (strongly warped classes — gestures, ECG), UEA. Baselines: TS2Vec, SoftCLT, soft-DTW-augmented supervised, 1NN-DTW.
6. **Evaluation:** Accuracy under injected speed/jitter shifts; retrieval mAP without DTW; a disentanglement probe (can shape code predict applied warp? — it shouldn't). **Refutes** if disentanglement doesn't improve warp-robustness or costs accuracy.
7. **RARE/RIGOROUS/USEFUL:** Rare = explicit warp/shape factorization is new in SSL4TS. Rigorous = controlled warp injections + disentanglement probe. Useful = retrieval and robustness where speed/phase is nuisance (gestures, biosignals).
8. **Feasibility:** Public data; differentiable warping is finicky. Risk: disentanglement is notoriously hard to evaluate and to optimize stably — start univariate, lean on synthetic warps for ground truth.
9. **Effort tier:** one-paper → multi-year if pushed to multivariate/theory.

### 11. Why does ROCKET work? Theory + a principled kernel-sampling distribution
**Primary axis:** Representation learning (theory)

1. **Thesis:** Random convolutional kernels work because they implement random frequency-selective filters; characterize what they capture and derive a *data-adaptive* sampling distribution that matches accuracy with an order of magnitude fewer kernels.
2. **Gap:** The ROCKET family is empirical engineering; despite random-features theory (Rahimi & Recht, NeurIPS 2007), there is no principled account of *why random kernels* suffice for TS or how to sample them better — every follow-up tweaks heuristics, not theory.
3. **Hypothesis:** ROCKET features approximate a kernel whose spectrum is dominated by a few frequency bands per dataset; sampling kernels from a data-estimated spectral density attains MiniRocket accuracy with ≤10% of the kernels.
4. **Method sketch:** Analyze ROCKET features in the frequency domain; connect PPV pooling to band-power statistics; estimate per-dataset spectral importance cheaply (periodogram), then sample kernels (length/dilation/weights) from it. Compare accuracy vs. #kernels.
5. **Datasets + baselines:** UCR/UEA/MONSTER. Baselines: MiniRocket, MultiRocket, Hydra at matched feature budgets.
6. **Evaluation:** Accuracy vs. kernel count (efficiency frontier); does the spectral theory predict which datasets need many kernels? **Refutes** if data-adaptive sampling gives no efficiency gain over random.
7. **RARE/RIGOROUS/USEFUL:** Rare = a *theory + principled sampling*, not another kernel variant. Rigorous = falsifiable efficiency prediction. Useful = far cheaper ROCKET for edge/large-scale (MONSTER) deployment.
8. **Feasibility:** Public data; cheap. Risk: the kernel space is crowded — a purely empirical "fewer kernels" result will look incremental; the theory must do real work.
9. **Effort tier:** multi-year (theory is the hard part).

### 12. Manifold-aware faithfulness metrics for time-series explanations
**Primary axis:** Interpretability & trust

1. **Thesis:** Standard deletion/insertion faithfulness metrics for TS use perturbations (zeroing, mean-fill) that create off-manifold artifacts because of temporal autocorrelation; replace them with autocorrelation-preserving perturbations and show current method rankings flip.
2. **Gap:** UTS-XAI (Sci. Reports 2026) and "Consistency and Robustness of Saliency for TSC" (arXiv:2309.01457) evaluate explanations but inherit vision-style perturbation baselines; no one has shown these are *confounded by temporal structure* or fixed it.
3. **Hypothesis:** Under autocorrelation-preserving perturbation (conditional sampling / phase-randomized surrogates), the relative ranking of SHAP vs. LIME vs. saliency on TSC changes materially vs. zero/mean perturbation.
4. **Method sketch:** Define manifold-aware perturbation operators (Gaussian-process conditional fill, phase-randomized surrogates, segment swap from same class). Re-run deletion/insertion faithfulness for major explainers; measure ranking stability across operators.
5. **Datasets + baselines:** UCR/UEA + ECG; explainers: SHAP, LIME, Integrated Gradients, saliency, attention. Reference: ground-truth-feature synthetic datasets where the "true" important region is known.
6. **Evaluation:** Faithfulness under each perturbation operator; rank-correlation across operators; recovery of known-important regions on synthetics. **Refutes** if rankings are invariant to the perturbation operator.
7. **RARE/RIGOROUS/USEFUL:** Rare = nobody has questioned the perturbation baseline for TS faithfulness. Rigorous = synthetic ground truth + multiple operators. Useful = corrects how the whole XAI4TS subfield measures itself.
8. **Feasibility:** Public data; cheap. Risk: defining "the" manifold-aware perturbation is itself contestable — present several and report robustness, don't claim one canonical operator.
9. **Effort tier:** one-paper.

### 13. Concept-vector (TCAV-style) explanations for time-series classification
**Primary axis:** Interpretability & trust

1. **Thesis:** Replace per-timestep saliency heatmaps (which are inconsistent and hard to read) with human-auditable *concept* explanations — "this prediction is driven by the presence of motif M / a rising trend / high band-power" — via concept-activation vectors adapted to TS.
2. **Gap:** TCAV (Kim et al., ICML 2018) is standard in vision; for TS, the closest is symbolic temporal-logic concept learning (arXiv:2508.03269, 2025), which is rule-extraction, not concept-vector attribution. CAV-style global explanations for TSC are basically untried.
3. **Hypothesis:** Motif/shapelet/spectral concepts admit stable CAVs whose sensitivities are more consistent (across seeds/perturbations) and more human-aligned than saliency, at comparable faithfulness.
4. **Method sketch:** Define a concept bank (mined shapelets, spectral-band detectors, trend/seasonality probes); learn CAVs in a trained classifier's latent space; compute directional sensitivity (TCAV scores) per class. Validate concept meaningfulness with a human study.
5. **Datasets + baselines:** ECG (clinically named morphologies), HAR (named activities), sleep staging. Baselines: saliency, SHAP, attention, the temporal-logic method.
6. **Evaluation:** CAV stability across seeds; faithfulness (does ablating the concept change the prediction?); human-rated interpretability vs. heatmaps. **Refutes** if CAVs are unstable or no more interpretable than saliency.
7. **RARE/RIGOROUS/USEFUL:** Rare = concept-vector attribution is new for TSC. Rigorous = stability + faithfulness + human eval. Useful = clinicians/engineers reason in concepts, not per-sample heatmaps.
8. **Feasibility:** Datasets public; needs a small human study. Risk: building a good concept bank per domain is real work — start with ECG where morphologies are named.
9. **Effort tier:** one-paper.

### 14. TSC-C: a physically-grounded common-corruption benchmark for time-series classification
**Primary axis:** Deployment / robustness

1. **Thesis:** L∞ adversarial attacks don't reflect how TS models fail in the field; build the ImageNet-C analogue for time series — clock drift, sensor dropout, gain/offset change, quantization, baseline wander, packet loss — and benchmark robustness with a single corruption-error metric.
2. **Gap:** ImageNet-C (Hendrycks & Dietterich, ICLR 2019) reshaped vision robustness; for TS, work is dominated by *artificial adversarial* perturbations (e.g., "Are TS Foundation Models Deployment-Ready?", arXiv:2505.19397; DTW-based adversarial framework). No standardized *realistic corruption* suite for classification exists.
3. **Hypothesis:** Standard accuracy rankings on clean UCR/UEA do *not* predict robustness rankings under realistic corruptions; some "SOTA" models are disproportionately fragile to mundane sensor faults.
4. **Method sketch:** Define ~12 physically-motivated corruption types × 5 severities with documented generators; release as a reproducible package. Evaluate ROCKET, InceptionTime, Hydra, TSFMs; report mean Corruption Error (mCE) relative to a baseline.
5. **Datasets + baselines:** UCR/UEA/MONSTER + HAR/ECG with corruptions; baselines across the classifier zoo + foundation models.
6. **Evaluation:** mCE; rank-correlation between clean accuracy and mCE; per-corruption breakdown; do augmentation/TTA help? **Refutes** if clean accuracy already predicts corruption robustness (then the benchmark adds little).
7. **RARE/RIGOROUS/USEFUL:** Rare = the ImageNet-C-shaped gap is genuinely open for TS. Rigorous = documented generators + standardized metric. Useful = anyone shipping wearable/IoT/clinical models needs a realistic robustness number.
8. **Feasibility:** Pure software on public data; low compute. Risk: corruption realism is a design judgment — ground each generator in a cited sensor-failure mode so it's defensible, and version the suite.
9. **Effort tier:** one-paper (and a lasting community artifact).

### 15. Conformal classification for time series under temporal drift
**Primary axis:** Deployment / robustness

1. **Thesis:** Give TSC distribution-free, class-conditional coverage guarantees that survive non-exchangeable, drifting streams — the classification analogue of recent conformal-forecasting work.
2. **Gap:** Conformal prediction for TS is almost entirely *regression/forecasting* (e.g., CPTC, NeurIPS 2025, arXiv:2509.02844). Conformal *classification* for TS — class-conditional prediction sets under temporal drift — is thin, despite high stakes (a "set of plausible diagnoses" is clinically natural).
3. **Hypothesis:** Adaptive/Mondrian conformal classification with a drift-aware recalibration window maintains nominal class-conditional coverage under temporal drift where split-conformal under-covers, at a modest set-size cost.
4. **Method sketch:** Online conformal classification with class-conditional (Mondrian) calibration and a change-point-triggered or weighted recalibration window; nonconformity from classifier softmax/feature distances. Compare coverage/size under stationary vs. drifting streams.
5. **Datasets + baselines:** Streaming/sequential TSC: cross-time ECG cohorts, HAR over sessions, sensor-fault streams, MONSTER tasks with temporal order preserved. Baselines: split conformal, adaptive conformal (ACI), temperature-scaling calibration.
6. **Evaluation:** Marginal and class-conditional coverage vs. nominal; average set size; coverage during drift episodes. **Refutes** if drift-aware recalibration doesn't restore coverage or blows up set size.
7. **RARE/RIGOROUS/USEFUL:** Rare = classification-side conformal for TS is under-served. Rigorous = coverage is a hard, falsifiable target. Useful = safety-critical TSC (clinical triage) needs calibrated *sets*, not point predictions.
8. **Feasibility:** Public data; light compute; strong theory base. Risk: guaranteeing *class-conditional* coverage under drift is genuinely hard — be honest about which guarantees are exact vs. heuristic.
9. **Effort tier:** one-paper.

### 16. Do long-series classifiers actually use long context, or collapse to a local window?
**Primary axis:** Long-horizon / irregular series

1. **Thesis:** For very long series (MONSTER-scale, satellite, audio-length), probe whether deep TSC models exploit global context or effectively classify from a short local window, and build a "context-utilization" diagnostic.
2. **Gap:** Long-horizon *forecasting* has a vigorous "do transformers use long context?" debate; long-series *classification* has no analogous scrutiny — MONSTER (arXiv:2502.15122) provides the scale but not this analysis.
3. **Hypothesis:** On a large fraction of long-series datasets, masking all but a short contiguous window leaves accuracy nearly unchanged, i.e., long-context models are over-specified; a context-utilization score predicts which datasets truly need global modeling.
4. **Method sketch:** Train long-context models (Transformer, S4/SSM, InceptionTime); measure accuracy as a function of available context (windowing, random masking, receptive-field ablation). Define context-utilization = accuracy sensitivity to context length.
5. **Datasets + baselines:** MONSTER, satellite land-cover (TiSeLaC / Sentinel), audio-length UCR, long ECG. Baselines: short-window ROCKET, full-context Transformer/SSM.
6. **Evaluation:** Accuracy-vs-context curves; fraction of datasets where short window suffices; does the score generalize? **Refutes** if long context is broadly necessary (then long-context models are justified).
7. **RARE/RIGOROUS/USEFUL:** Rare = the long-context question is unasked for classification. Rigorous = context-ablation with a defined score. Useful = saves practitioners from expensive long-context models when a window suffices.
8. **Feasibility:** MONSTER + satellite data public; compute moderate (long series are heavier). Risk: windowing must respect label semantics (some labels are inherently global) — categorize datasets first.
9. **Effort tier:** one-paper.

### 17. Is complex temporal modeling necessary for irregular medical MTSC? A fair, compute-matched bake-off
**Primary axis:** Irregular / multivariate series

1. **Thesis:** For irregular, missing-heavy medical MTSC, a strong summary-statistics + gradient-boosting baseline (with *informative-missingness* features) rivals GRU-D / Latent-ODE / Hi-patch under matched tuning — and a dataset taxonomy says when ODEs actually earn their keep.
2. **Gap:** A recent statistical two-step pipeline (rs-6198987, 2025) and MissTSM (arXiv:2502.15785) hint that simple methods are competitive, but there's no *compute- and tuning-matched* head-to-head, nor a characterization of *when* irregularity itself (the sampling pattern) is the signal — "informative sampling" as a feature.
3. **Hypothesis:** On PhysioNet-2012/2019 and MIMIC, GBM on summary stats + missingness-pattern features matches deep continuous-time models within noise on most tasks; deep models win only where inter-observation dynamics (not sampling-pattern) carry the label.
4. **Method sketch:** Fix a tuning/compute budget. Features: per-channel stats, last-value, observation counts, inter-arrival times (informative sampling). Compare to GRU-D, Latent-ODE, mTAND, Hi-patch. Probe: does masking the sampling-pattern features hurt the GBM? (tests informative sampling).
5. **Datasets + baselines:** PhysioNet-2012/2019, MIMIC-III/IV mortality/phenotyping, human-activity irregular benchmarks. Baselines as above.
6. **Evaluation:** AUROC/AUPRC at matched compute; ablate informative-sampling features; identify dataset properties predicting deep-model advantage. **Refutes** if deep models clearly dominate at matched budget across the board.
7. **RARE/RIGOROUS/USEFUL:** Rare = the *compute-matched* protocol + informative-sampling taxonomy, not just "simple is competitive." Rigorous = budget-matched, ablated. Useful = clinical ML teams choosing whether to maintain heavy ODE pipelines.
8. **Feasibility:** PhysioNet public; MIMIC needs credentialed access (standard). Risk: "fair tuning" is contestable — pre-register budgets and search spaces.
9. **Effort tier:** one-paper.

---

## DO NOT pursue (tempting but saturated)

1. **Another random-convolutional-kernel variant** (ROCKET/MiniRocket/MultiRocket/Hydra/Quant/KG-MTP/Hydra+Quant). The accuracy–speed frontier is mined out; bake-offs show parity. *Exception:* idea #11's theory angle, not a new heuristic.
2. **Yet another contrastive SSL method with novel augmentations** (post-TS2Vec/SoftCLT). Gains are marginal and the augmentation-design space is exhausted; reviewers are fatigued.
3. **A new transformer architecture for long-horizon forecasting** (post-PatchTST/iTransformer/Crossformer). Extremely crowded, and linear/DLinear baselines remain stubbornly competitive — high bar, low marginal payoff.
4. **LLM-reprogramming for TS forecasting** (Time-LLM successors). Under direct empirical attack (Tan et al., NeurIPS 2024, showed simpler ablations match; modality-alignment critique arXiv:2410.12326). High risk your own ablation kills the result.
5. **Training a bigger general-purpose forecasting foundation model** (Chronos/Moirai/TimesFM successor). A compute and data game owned by large labs; near-impossible to win on resources, and the marginal scientific delta is shrinking.

---

## Top-5 shortlist (ranked by novelty-per-unit-effort)

1. **#14 — TSC-C common-corruption benchmark.** A proven, high-impact recipe (ImageNet-C) with a genuinely empty slot in TS; pure software on public data, lasting community artifact. Best effort-to-impact ratio on the list.
2. **#2 — Contamination audit of "zero-shot" TSFMs.** Timely, mostly engineering, and could recalibrate a whole hot narrative — high citations-per-GPU-hour.
3. **#3 — Data cartography for TS archives.** Port a settled NLP/vision method to a place it's absent; cheap, broadly useful, and benchmark-relevant to everyone.
4. **#6 — Shortcut-learning audit for TSC.** Names and measures a failure mode the clinical-TS field is quietly suffering from; low compute, high reliability payoff, durable (won't be scooped by next month's foundation model).
5. **#12 — Manifold-aware faithfulness metrics.** Crisp, falsifiable, cheap, and self-correcting for the XAI4TS subfield — a small experiment with outsized influence on how the area evaluates itself.

*Honorable mention:* **#7 (in-context TSC scaling laws)** if you want to ride the hottest current wave — but move fast; the in-context-classification area is accelerating and the durable contribution there is the decision rule, not the leaderboard.

---

## Sources (representative; verify before citing)

- [TS Foundation Models: Benchmarking Challenges and Requirements (arXiv:2510.13654)](https://arxiv.org/html/2510.13654v1)
- [Mantis: Lightweight Calibrated Foundation Model for TSC (arXiv:2502.15637)](https://arxiv.org/pdf/2502.15637)
- [MantisV2: Closing the Zero-Shot Gap in TSC (arXiv:2602.17868)](https://arxiv.org/pdf/2602.17868)
- [Rethinking Zero-Shot TSC: From Task-specific Classifiers to In-Context Inference (arXiv:2602.00620)](https://arxiv.org/pdf/2602.00620)
- [Pre-trained Forecasting Models: Strong Zero-Shot Feature Extractors for TSC (arXiv:2510.26777)](https://arxiv.org/pdf/2510.26777)
- [Revisit TSC Benchmark: The Impact of Temporal Information (arXiv:2503.20264)](https://arxiv.org/html/2503.20264v1)
- [MONSTER: Monash Scalable Time Series Evaluation Repository (arXiv:2502.15122)](https://arxiv.org/abs/2502.15122)
- [HYDRA: Competing convolutional kernels for TSC (arXiv:2203.13652)](https://arxiv.org/pdf/2203.13652)
- [QUANT: A Minimalist Interval Method for TSC (arXiv:2308.00928)](https://arxiv.org/pdf/2308.00928)
- [The Meta-Learning Gap: Combining Hydra and Quant (arXiv:2512.06666)](https://arxiv.org/pdf/2512.06666)
- [Soft Contrastive Learning for Time Series — SoftCLT (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/ccc48eade8845cbc0b44384e8c49889a-Paper-Conference.pdf)
- [Time-LLM: Reprogramming LLMs for Forecasting (ICLR 2024, arXiv:2310.01728)](https://arxiv.org/pdf/2310.01728)
- [Understanding Why LLMs Can Be Ineffective in Time Series (arXiv:2410.12326)](https://arxiv.org/html/2410.12326)
- [Navigating Concept Drift and Temporal Shift (OpenReview)](https://openreview.net/forum?id=Klx0Rq9vbC)
- [Channel Independence Improves OOD Generalisation in MTSC (OpenReview)](https://openreview.net/forum?id=CLImhawlGn)
- [Investigating a Model-Agnostic, Imputation-Free Approach for Irregular MTS — MissTSM (arXiv:2502.15785)](https://arxiv.org/html/2502.15785v3)
- [On the Consistency and Robustness of Saliency Explanations for TSC (arXiv:2309.01457)](https://arxiv.org/pdf/2309.01457)
- [Unified TSC Framework for Explainable AI (Scientific Reports 2026)](https://www.nature.com/articles/s41598-026-49467-2)
- [Interpretable Concept Learning over TS via Temporal Logic (arXiv:2508.03269)](https://arxiv.org/pdf/2508.03269)
- [Counterfactual Explanation Bake-Off for TSC (Springer MLJ 2026)](https://link.springer.com/article/10.1007/s10994-026-07056-4)
- [Are TS Foundation Models Deployment-Ready? Adversarial Robustness (arXiv:2505.19397)](https://arxiv.org/abs/2505.19397)
- [Conformal Prediction for TS Forecasting with Change Points — CPTC (NeurIPS 2025, arXiv:2509.02844)](https://arxiv.org/abs/2509.02844)
- [CANDI: Curated Test-Time Adaptation for MTS Anomaly Detection (arXiv:2604.01845)](https://arxiv.org/pdf/2604.01845)
- [Awesome-SSL4TS (curated list)](https://github.com/qingsongedu/Awesome-SSL4TS)
