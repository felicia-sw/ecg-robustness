# Are Time-Series Classifiers Robust to Realistic Sensor Corruptions?

### TSC-C: A Physically-Grounded Common-Corruption Benchmark and Robustness Audit for Time-Series Classification

**Status:** research proposal v1 (gap-checked against 2024–2026 literature). **Type:** datasets & benchmarks / empirical robustness audit. **Target:** NeurIPS/ICLR Datasets & Benchmarks track (primary); Q1/Q2 journal route — *Data Mining and Knowledge Discovery* (DAMI), *TKDE*, TMLR, or *IEEE Access*. **Date:** June 2026.

**Honesty note.** This is an achievable, high-utility *benchmark + audit* paper, not a new architecture or a claimed breakthrough. The underlying recipe — measuring robustness to *common corruptions* rather than worst-case adversarial perturbations — is proven in vision (ImageNet-C [1]) and has been ported to other modalities (driving multi-sensor, video action detection). The contribution is (i) transferring it rigorously to time-series classification (TSC) with *physically-grounded, severity-calibrated* corruptions instead of "just add Gaussian noise," (ii) a standardized metric (a TS analogue of mean Corruption Error), and (iii) a *falsifiable* test that clean-accuracy rankings do **not** predict corruption robustness. Domain-specific TS robustness work exists (adversarial attacks on foundation models [12]; additive-noise robustness in anomaly detection; OOD-domain benchmarks like WOODS [13]); none provides a general, reusable common-corruption suite for TSC. The space is moving, so speed + rigor + a reusable artifact are the edge.

---

## 1. One-paragraph summary

Time-series classifiers are increasingly deployed on wearables, clinical monitors, and industrial/IoT sensors, where the real failure modes are not L∞ adversarial perturbations but *mundane sensor faults*: baseline wander, electrode/motion artifact, gain miscalibration, clock drift, packet dropout, quantization, and powerline interference. The TSC literature evaluates almost exclusively on *clean* archives (UCR [10], UEA [11], MONSTER [9]) or on artificial adversarial attacks; there is no ImageNet-C-style benchmark of realistic corruptions. This project builds **TSC-C**, a versioned, reproducible suite of physically-grounded corruption generators (a domain-agnostic core plus biosignal/IMU-specific add-ons, each calibrated across 5 severities and, where possible, driven by *real recorded* noise from the MIT-BIH Noise Stress Test Database [15]), and runs the first systematic robustness audit of the TSC model zoo (ROCKET-family, InceptionTime, HIVE-COTE 2.0, 1NN-DTW, and foundation models). It tests one sharp, falsifiable hypothesis: **clean accuracy is a weak predictor of corruption robustness** — i.e., the rankings re-order, and some "SOTA" models are disproportionately fragile to ordinary faults. It also tests whether cheap training-time augmentation closes the gap *and generalizes* to held-out corruption types. The deliverable is a reusable benchmark + harness, a standardized metric, and an honest robustness leaderboard.

## 2. Relationship to ImageNet-C and the robustness-benchmark tradition (and why standing on it is a strength)

ImageNet-C [1] (and ImageNet-C-bar [2]) established the protocol this paper transfers: evaluate on **common corruptions** at graded severities, report a normalized **mean Corruption Error (mCE)**, and *forbid training on the benchmark corruptions* so that augmentation results measure genuine generalization. That protocol is validated, widely adopted, and reproducible — so we begin with a settled metric, a known-good experimental design, and a falsifiable hypothesis carried over from vision ("clean accuracy ≠ robustness"). The novelty is not "inventing corruption robustness"; it is doing the *time-series-specific* engineering honestly — choosing physically realistic corruptions, calibrating severities against physical parameters (SNR, drift amplitude, dropout rate), and using real recorded sensor noise where available. This is a head start that is reproducible, not a secrecy moat.

## 3. The honest gap (what already exists vs. what does not)

Robustness work adjacent to this exists, each leaving the wedge open:

- **Vision / other modalities.** ImageNet-C/-P [1] and ImageNet-C-bar [2] for images; multi-sensor corruption benchmarks for autonomous driving (e.g., MSC-Bench, 2025) and temporal corruption robustness for video action detection (CVPR 2024). None covers generic 1-D time-series classification on standard archives.
- **Adversarial robustness for TS.** "Are Time-Series Foundation Models Deployment-Ready?" [12] and DTW-based adversarial frameworks study *worst-case* perturbations — the opposite end from realistic common corruptions.
- **Anomaly detection.** Robustness is often probed by injecting *additive noise* of increasing strength — a single corruption family, no severity calibration, no classification leaderboard, no standardized metric.
- **OOD / domain shift.** WOODS [13] benchmarks *distribution/domain* shift in time series (different subjects/environments), not synthetic, physically-grounded *signal corruptions* at controlled severities.
- **Augmentation libraries.** TS augmentation surveys [25] and methods [24] define jitter/scaling/warping *as training-time augmentations*, never as a held-out *evaluation* corruption suite with a metric.

**What is NOT done:** a *general, physically-grounded, multi-severity common-corruption benchmark for TSC* on UCR/UEA/MONSTER (plus biosignal/IMU domains), with (i) a standardized normalized robustness metric, (ii) a clean-vs-robust ranking analysis across the full model zoo, and (iii) a held-out-corruption augmentation protocol. That triad is the wedge.

## 4. Research questions

1. **Ranking validity.** Across the TSC model zoo, does the *clean-accuracy* ranking predict the *corruption-robustness* (mCE) ranking, or do they re-order under realistic corruptions?
2. **Differential fragility.** Are specific model families (deep nets, foundation models) disproportionately fragile to specific mundane corruptions (e.g., gain/offset, quantization, clock drift) relative to simple feature/kernel methods (ROCKET, catch22)?
3. **Augmentation generalization.** Does cheap training-time augmentation improve mCE, and does it *generalize to held-out corruption types*, or only to the corruptions it trained on (i.e., is the benchmark trivially gameable)?
4. **Calibration under corruption (secondary).** Does predictive calibration (ECE) degrade faster than accuracy under corruption — i.e., do models become confidently wrong?

## 5. Method

**The TSC-C suite (the core artifact).** A versioned (`v1.0`) set of corruption generators, each parameterized over 5 severities with severities calibrated against a *physical* quantity (target SNR in dB, drift amplitude as % of signal range, dropout fraction, ENOB for quantization, ppm for clock drift), so severity is monotone and comparable across datasets.

- *Domain-agnostic core (applies to any series):* additive Gaussian noise; additive pink (1/f) noise; low-frequency baseline drift/wander; impulse/spike noise (electrode pop, packet glitch); sensor dropout (contiguous missing segments, last-value/NaN-filled); amplitude gain change; DC offset shift; quantization / bit-depth reduction; clock drift / sampling-rate jitter (resampling + warping); temporal lag / phase shift; trend injection; saturation/clipping.
- *Biosignal add-ons:* powerline interference (50/60 Hz); **real recorded** motion artifact, electrode-motion, and muscle-artifact noise convolved/added from the MIT-BIH Noise Stress Test Database (NSTDB) [15, 16] at calibrated SNRs — the realism anchor that distinguishes TSC-C from synthetic-only suites.
- *IMU/HAR add-ons:* motion artifact and sensor-drift profiles drawn from the wearable-sensor literature.

**Protocol rule (inherited from ImageNet-C [1]).** Models are **never trained on the benchmark corruptions.** A disjoint *held-out* corruption family is reserved to test augmentation generalization (RQ3). Clean test sets, frozen splits, fixed seeds.

**Model zoo (honest baselines).** Feature/kernel: ROCKET [3], MiniRocket [4], MultiRocket [5], Hydra [6], Quant [7], catch22+ridge [26]. Deep: InceptionTime [8], FCN/ResNet [—, Wang et al. 2017]. Ensemble: HIVE-COTE 2.0 [—, Middlehurst et al. 2021]. Distance: 1NN-DTW. Foundation: Mantis [—] and forecasting-pretrained features (Chronos / Moirai / TimesFM) + linear probe.

**Mitigations tested.** Training-time TS augmentation (jitter/scaling/magnitude-warp/time-warp [24, 25], mixup) and a test-time-adaptation baseline (entropy-minimization style [—, TENT]). Reported as robustness *recovered*, with augmentation evaluated only on *held-out* corruptions.

**Metrics.** Primary: a TS analogue of **mean Corruption Error**, `mCE_f = mean_c [ (Σ_s E^f_{c,s}) / (Σ_s E^{ref}_{c,s}) ]`, normalized to a fixed reference classifier, plus **relative mCE** (degradation above clean error). Secondary: balanced accuracy under corruption, per-corruption/per-severity breakdown, and calibration (ECE) under corruption. Clean accuracy always reported alongside.

**Analysis.** Spearman rank-correlation between clean-accuracy and mCE leaderboards (RQ1); per-family fragility heatmaps (RQ2); augmentation gain on seen vs. held-out corruptions (RQ3); all with bootstrap confidence intervals over datasets/seeds.

## 6. Rigor and pre-registration

The corruption set, severity-calibration formulas, model list, primary metric, and the **decision rule are pre-registered before any evaluation.** Concretely, the confirmatory test for RQ1 is registered as: *clean-vs-mCE Spearman ρ < 0.7 across the model zoo* (with bootstrap CI excluding 0.9) would confirm "clean accuracy is a weak proxy for robustness." Strict no-leakage rules: models never see benchmark corruptions; a disjoint corruption family is held out for augmentation tests; frozen inference settings per model; severities fixed before evaluation; bootstrap CIs over datasets and seeds; honest reporting of nulls. The suite is released versioned with seeds, generator code, environment, and result tables as a reusable artifact.

## 7. Contributions (stated honestly)

1. **TSC-C**: the first general, physically-grounded, severity-calibrated **common-corruption benchmark for time-series classification**, spanning UCR/UEA/MONSTER plus biosignal/IMU domains, with real recorded noise where available.
2. A **standardized robustness metric** (TS mean Corruption Error + relative mCE) and a reusable evaluation harness.
3. The first **clean-vs-robust ranking audit** of the TSC model zoo — a falsifiable test of whether clean accuracy predicts robustness, plus per-family fragility maps.
4. A **held-out-corruption augmentation study** showing whether cheap augmentation genuinely generalizes or merely overfits the benchmark.

The claim is never "we discovered corruption robustness"; it is the TS-specific suite + standardized metric + the falsifiable ranking test.

## 8. Related-work positioning (foundations, not inventions)

Cited as foundations: the common-corruption protocol and metric [1, 2]; the TSC model zoo being audited [3–9, and InceptionTime/HIVE-COTE/catch22]; the archives [9, 10, 11]; biosignal noise sources and the NSTDB [14, 15, 16]; adjacent (but distinct) robustness work [12, 13]; augmentation methods used as mitigations [24, 25]. The contribution is the benchmark, the metric, and the falsifiable audit — never the invention of robustness evaluation.

## 9. Risks and mitigations

- **Corruption realism is contestable (primary).** A reviewer can argue "these aren't the real failures." Mitigation: ground every generator in a cited sensor-failure mode, calibrate severities against physical parameters, and use **real recorded** noise (NSTDB [15]) for biosignals; version the suite so the design is auditable and improvable.
- **"It's just adding noise."** Mitigation: the distinctions are physical grounding, severity calibration, real recorded artifacts, the held-out-augmentation protocol, and the falsifiable ranking analysis — none of which a naive noise-injection study provides.
- **Domain specificity.** Corruptions differ across modalities. Mitigation: separate a domain-agnostic *core* from domain-specific *add-ons*; report per-domain and never average across incomparable modalities.
- **Competition / fast-moving field.** Robustness benchmarks for adjacent modalities appear regularly. Mitigation: move fast, lean on the validated ImageNet-C protocol, and make the *standardized metric + reusable harness* the moat.
- **Compute (MONSTER + foundation models).** Mitigation: subsample consistently, document budgets, report wall-clock honestly; the core result needs only a representative subset.
- **Incrementality.** Framed honestly as a benchmark + falsifiable audit (Datasets & Benchmarks / Q2), not a breakthrough.

## 10. Venue calibration (no exaggeration)

The natural home is a **NeurIPS/ICLR Datasets & Benchmarks** track (benchmark + audit fits exactly). Journal routes: *Data Mining and Knowledge Discovery* (DAMI — the canonical TSC venue), *IEEE TKDE*, **TMLR** (rigorous, open, benchmark-friendly), or *IEEE Access* / *PeerJ CS* as accessible options; an ICLR/NeurIPS workshop as a fast fallback. Confirm quartiles on SJR/JCR and match each venue's current scope before submission.

## 11. Immediate next step (1-week pilot)

Implement **4 corruption generators × 5 severities** (additive noise, baseline wander, gain change, quantization) and run **ROCKET, InceptionTime, and Mantis** on ~10 UCR datasets + a PTB-XL subset. Compute clean accuracy and mCE; check whether the clean-vs-mCE Spearman ρ already drops below the pre-registered threshold on this slice. If the ranking breaks even here, the thesis is confirmed and the full suite is greenlit. If SHAP/calibration extras prove heavy, keep them as secondary and lead with the mCE ranking audit. Then write the full pre-registration.

## 12. Dataset, model & resource links

**Archives / datasets**
- UCR Time Series Archive — https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
- UEA Multivariate TSC Archive — http://www.timeseriesclassification.com/
- MONSTER (Monash Scalable Time Series Evaluation Repository) — https://github.com/Navidfoumani/monster · https://huggingface.co/monster-monash
- PTB-XL 12-lead ECG — https://physionet.org/content/ptb-xl/
- MIT-BIH Noise Stress Test Database (NSTDB, real recorded noise) — https://physionet.org/content/nstdb/
- PhysioNet (host) — https://physionet.org/
- PAMAP2 Physical Activity Monitoring — https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
- UCI HAR Using Smartphones — https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Speech Commands (audio) — https://www.tensorflow.org/datasets/catalog/speech_commands

**Models / libraries / harness**
- ImageNet-C reference implementation (protocol/metric) — https://github.com/hendrycks/robustness
- aeon (TSC toolkit: ROCKET-family, InceptionTime, HIVE-COTE, catch22) — https://github.com/aeon-toolkit/aeon
- sktime — https://github.com/sktime/sktime
- tsai (deep TSC) — https://github.com/timeseriesAI/tsai
- Mantis (TSC foundation model) — https://arxiv.org/pdf/2502.15637
- Chronos — https://github.com/amazon-science/chronos-forecasting
- Moirai (uni2ts) — https://github.com/SalesforceAIResearch/uni2ts
- TimesFM — https://github.com/google-research/timesfm
- catch22 — https://github.com/DynamicsAndNeuralSystems/catch22

## 13. References

1. Hendrycks, D., & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (ImageNet-C / ImageNet-P). ICLR. arXiv:1903.12261. https://arxiv.org/abs/1903.12261
2. Mintun, E., Kirillov, A., & Xie, S. (2021). *On Interaction Between Augmentations and Corruptions in Natural Corruption Robustness* (ImageNet-C-bar). NeurIPS. arXiv:2102.11273. https://arxiv.org/abs/2102.11273
3. Dempster, A., Petitjean, F., & Webb, G. I. (2020). *ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels.* Data Mining and Knowledge Discovery. arXiv:1910.13051. https://arxiv.org/abs/1910.13051
4. Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). *MiniRocket: A very fast (almost) deterministic transform for time series classification.* KDD. arXiv:2012.08791. https://arxiv.org/abs/2012.08791
5. Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). *MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification.* DAMI. arXiv:2102.00457. https://arxiv.org/abs/2102.00457
6. Dempster, A., Schmidt, D. F., & Webb, G. I. (2023). *Hydra: Competing convolutional kernels for fast and accurate time series classification.* DAMI. arXiv:2203.13652. https://arxiv.org/abs/2203.13652
7. Dempster, A., Schmidt, D. F., & Webb, G. I. (2024). *QUANT: A minimalist interval method for time series classification.* DAMI. arXiv:2308.00928. https://arxiv.org/abs/2308.00928
8. Ismail Fawaz, H., et al. (2020). *InceptionTime: Finding AlexNet for time series classification.* DAMI. arXiv:1909.04939. https://arxiv.org/abs/1909.04939
9. Dempster, A., Foumani, N., et al. (2025). *MONSTER: Monash Scalable Time Series Evaluation Repository.* DMLR. arXiv:2502.15122. https://arxiv.org/abs/2502.15122
10. Dau, H. A., et al. (2019). *The UCR Time Series Archive.* IEEE/CAA Journal of Automatica Sinica. arXiv:1810.07758. https://arxiv.org/abs/1810.07758
11. Bagnall, A., et al. (2018). *The UEA multivariate time series classification archive, 2018.* arXiv:1811.00075. https://arxiv.org/abs/1811.00075
12. Authors (2025). *Are Time-Series Foundation Models Deployment-Ready? A Systematic Study of Adversarial Robustness Across Domains.* arXiv:2505.19397. https://arxiv.org/abs/2505.19397
13. Gagnon-Audet, J.-C., et al. (2022). *WOODS: Benchmarks for Out-of-Distribution Generalization in Time Series.* arXiv:2203.09978. https://arxiv.org/abs/2203.09978
14. Wagner, P., et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data 7, 154. https://physionet.org/content/ptb-xl/
15. Moody, G. B., Muldrow, W. E., & Mark, R. G. (1984). *A noise stress test for arrhythmia detectors* (MIT-BIH Noise Stress Test Database, NSTDB). Computers in Cardiology. https://physionet.org/content/nstdb/
16. Goldberger, A. L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 101(23), e215–e220. https://physionet.org/
17. Reiss, A., & Stricker, D. (2012). *Introducing a new benchmarked dataset for activity monitoring (PAMAP2).* ISWC. https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
18. Anguita, D., et al. (2013). *A public domain dataset for human activity recognition using smartphones (UCI HAR).* ESANN. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
19. Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* arXiv:1804.03209. https://arxiv.org/abs/1804.03209
20. Authors (2025). *Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification.* arXiv:2502.15637. https://arxiv.org/abs/2502.15637
21. Ansari, A. F., et al. (2024). *Chronos: Learning the Language of Time Series.* TMLR. arXiv:2403.07815. https://arxiv.org/abs/2403.07815
22. Woo, G., et al. (2024). *Unified Training of Universal Time Series Forecasting Transformers (Moirai).* ICML. arXiv:2402.02592. https://arxiv.org/abs/2402.02592
23. Das, A., et al. (2024). *A decoder-only foundation model for time-series forecasting (TimesFM).* ICML. arXiv:2310.10688. https://arxiv.org/abs/2310.10688
24. Um, T. T., et al. (2017). *Data augmentation of wearable sensor data for Parkinson's disease monitoring using convolutional neural networks* (jitter/scaling/warping augmentations). ICMI. arXiv:1706.00527. https://arxiv.org/abs/1706.00527
25. Iwana, B. K., & Uchida, S. (2021). *An empirical survey of data augmentation for time series classification with neural networks.* PLOS ONE. arXiv:2007.15951. https://arxiv.org/abs/2007.15951
26. Lubba, C. H., et al. (2019). *catch22: CAnonical Time-series CHaracteristics.* DAMI. arXiv:1901.10200. https://arxiv.org/abs/1901.10200
27. Wang, D., et al. (2021). *Tent: Fully Test-Time Adaptation by Entropy Minimization.* ICLR. arXiv:2006.10726. https://arxiv.org/abs/2006.10726
28. Wang, Z., Yan, W., & Oates, T. (2017). *Time series classification from scratch with deep neural networks: A strong baseline* (FCN/ResNet). IJCNN. arXiv:1611.06455. https://arxiv.org/abs/1611.06455
29. Middlehurst, M., et al. (2021). *HIVE-COTE 2.0: a new meta ensemble for time series classification.* Machine Learning. arXiv:2104.07551. https://arxiv.org/abs/2104.07551

**Verification status:** Citations [1–11, 13, 16, 24–29] are well-established and were sanity-checked during preparation (June 2026). Entries marked "Authors (2025)" ([12], [20]) are recent preprints whose exact author lists/venues should be confirmed at write-up, following the same convention as the source template. Confirm all arXiv IDs, venues, pages, and live repository links before submission; reference numbering will be tidied in the manuscript.
