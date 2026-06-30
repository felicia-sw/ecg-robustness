# Are ECG Classifiers Robust to Realistic Sensor Corruptions?

### ECG-C: A Physically-Grounded Common-Corruption Benchmark and Robustness Audit for Electrocardiogram Classification

**Status:** research proposal v3 (domain-scoped ECG; model zoo trimmed and a statistical-comparison analysis added, per advisor feedback; gap-checked against 2024–2026 literature). **Type:** datasets & benchmarks / empirical robustness audit — single domain: 12-lead and single-lead ECG. **Target:** NeurIPS/ICLR Datasets & Benchmarks track; biomedical venues — *Physiological Measurement*, *Computers in Biology and Medicine*, *IEEE J. Biomedical and Health Informatics*; ML route — *Data Mining and Knowledge Discovery* (DAMI) / TMLR. **Date:** June 2026.

**Honesty note.** This is an achievable, high-utility *benchmark + audit* paper, not a new architecture. The recipe — measuring robustness to *common corruptions* rather than worst-case adversarial perturbations — is proven in vision (ImageNet-C [1]). Following advisor feedback, the scope is deliberately narrowed to a **single domain, ECG**, which (i) makes corruption realism airtight because ECG has a well-characterized noise taxonomy and a database of **real recorded** noise (MIT-BIH NSTDB [16]), and (ii) keeps the experiments tractable (one modality, public datasets, no credentialed-access hurdle). The contribution is a physically-grounded, severity-calibrated ECG corruption suite, a standardized robustness metric, and a *falsifiable* test that clean-accuracy rankings do **not** predict corruption robustness. The methodology is portable to other sensor domains (IMU, PPG) as explicit future work.

---

## 1. One-paragraph summary

ECG classifiers are moving from clean hospital archives onto Holter monitors, patches, and consumer wearables, where the real failure modes are not L∞ adversarial perturbations but *ordinary signal corruptions*: baseline wander, muscle (EMG) and electrode-motion artifact, powerline interference, gain miscalibration, lead dropout, low-bit quantization, and sampling-rate mismatch. The ECG-classification literature evaluates almost exclusively on *clean* signals (PTB-XL [11], Chapman-Shaoxing [12], CPSC2018 [13]) or on artificial adversarial attacks; there is no ImageNet-C-style benchmark of realistic ECG corruptions with a standardized metric. This project builds **ECG-C**, a versioned, reproducible suite of physically-grounded corruption generators — anchored by **real recorded** baseline-wander, muscle-artifact, and electrode-motion noise from the MIT-BIH Noise Stress Test Database [16] added at calibrated signal-to-noise ratios, plus synthetic-but-grounded faults — and runs the first systematic robustness audit of the ECG-classification model zoo (ROCKET-family, InceptionTime, 1D-ResNet/SE-ResNet, and time-series foundation features). It tests one sharp, falsifiable hypothesis: **clean diagnostic accuracy is a weak predictor of corruption robustness** — the model rankings re-order under realistic noise, and some high-scoring models are disproportionately fragile to mundane faults. It also tests whether cheap noise augmentation closes the gap *and generalizes* to a held-out corruption type. The deliverable is a reusable ECG robustness benchmark + harness, a standardized metric, and an honest robustness leaderboard.

## 2. Relationship to ImageNet-C and the robustness-benchmark tradition (and why standing on it is a strength)

ImageNet-C [1] (and ImageNet-C-bar [2]) established the protocol this paper transfers: evaluate on **common corruptions** at graded severities, report a normalized **mean Corruption Error (mCE)**, and *forbid training on the benchmark corruptions* so that augmentation results measure genuine generalization. That protocol is validated, widely adopted, and reproducible — so we begin with a settled metric, a known-good experimental design, and a falsifiable hypothesis carried over from vision ("clean accuracy ≠ robustness"). The novelty is not "inventing corruption robustness"; it is doing the *ECG-specific* engineering honestly — choosing physiologically realistic corruptions, calibrating severities against physical parameters (SNR in dB, drift amplitude, dropout duration, ADC bit-depth), and using *real recorded* clinical noise where available. This is a reproducible head start, not a secrecy moat.

## 3. The honest gap (what already exists vs. what does not)

Robustness work adjacent to ECG-C exists, each leaving the wedge open:

- **Vision / other modalities.** ImageNet-C/-P [1] and ImageNet-C-bar [2]. The protocol and metric are proven, but no analogue exists for ECG classification.
- **ECG denoising and noise detection.** A large literature removes baseline wander/powerline/motion artifact, and recent work detects ECG noise across sources [20]; the NSTDB [16] is the standard *noise source*. But these target signal *cleaning/quality*, not a standardized *classification-robustness* benchmark with a metric and a model-ranking audit.
- **Adversarial robustness for time series.** "Are Time-Series Foundation Models Deployment-Ready?" [21] and DTW-based adversarial frameworks study *worst-case* perturbations — the opposite end from realistic common corruptions.
- **OOD / cross-source generalization.** Cross-dataset ECG evaluation and time-series OOD benchmarks like WOODS [22] probe *distribution/domain* shift (different hospitals/devices), not synthetic, physically-grounded *signal corruptions* at controlled severities.
- **Augmentation methods.** TS/ECG augmentation [23] defines jitter/scaling/noise *as training-time augmentation*, never as a held-out *evaluation* corruption suite with a metric.

**What is NOT done:** a *general, physically-grounded, multi-severity common-corruption benchmark for ECG classification* (12-lead and single-lead) with (i) a standardized normalized robustness metric, (ii) a clean-vs-robust ranking audit across the model zoo, and (iii) a held-out-corruption augmentation protocol — with corruptions anchored in real recorded clinical noise. That triad is the wedge.

## 4. Research questions

1. **Ranking validity.** Across the ECG-classification model zoo, does the *clean* diagnostic-accuracy ranking predict the *corruption-robustness* (mCE) ranking, or do they re-order under realistic noise?
2. **Differential fragility.** Are specific model families (deep CNNs, foundation-feature probes) disproportionately fragile to specific mundane corruptions (e.g., baseline wander, powerline interference, quantization, sampling-rate mismatch) relative to simple feature/kernel methods (ROCKET, catch22)?
3. **Augmentation generalization.** Does cheap noise augmentation improve mCE, and does it *generalize to a held-out corruption type*, or only to the corruptions it trained on (i.e., is the benchmark trivially gameable)?
4. **Calibration under corruption (secondary).** Does multi-label calibration (ECE/Brier) degrade faster than accuracy under corruption — i.e., do models become confidently wrong on noisy ECG?

## 5. Method

**The ECG-C suite (the core artifact).** A versioned (`v1.0`) set of corruption generators, each parameterized over 5 severities calibrated against a *physical* quantity, so severity is monotone and comparable across datasets.

- *Real recorded noise (the realism anchor), from MIT-BIH NSTDB [16]:* **baseline wander (bw)**, **muscle/EMG artifact (ma)**, and **electrode-motion artifact (em)**, added to clean records at calibrated SNRs (e.g., {18, 12, 6, 0, −6} dB) — the standard NSTDB regime. This is what distinguishes ECG-C from synthetic-only suites.
- *Synthetic but physiologically grounded:* powerline interference (50/60 Hz + harmonics); Gaussian/instrumentation noise; low-frequency baseline drift (respiration-like); amplitude gain miscalibration; lead/electrode dropout (contiguous flatline) and saturation/clipping; quantization (reduced ADC bit-depth, low-cost wearables); sampling-rate mismatch / clock drift (resampling across 500/250/125 Hz).
- *Severity calibration:* 5 levels per corruption, calibrated by the relevant physical parameter (SNR dB; drift amplitude as % of QRS amplitude; dropout duration; effective number of bits; ppm clock drift).

**Protocol rule (inherited from ImageNet-C [1]).** Models are **never trained on the benchmark corruptions.** One corruption family (e.g., electrode-motion artifact) is held out to test augmentation generalization (RQ3). Clean test sets, frozen recommended folds, fixed seeds.

**Model zoo (lean by design).** Per advisor feedback on keeping the experiment tractable, the *primary* zoo is deliberately small — **five models spanning five paradigms**, enough for a meaningful ranking without bloat: (1) **MiniRocket** [4] (random convolutional kernels — the fast representative of the ROCKET family); (2) **catch22 + ridge** [9] (interpretable summary features); (3) **InceptionTime** [8] (general deep TSC); (4) a **1D SE-ResNet** ECG classifier (PhysioNet/CinC-2020-style [15]) as the clinically validated deep reference; (5) **time-series foundation features (Mantis** [25]**) + linear probe**. An *optional extended* set, run only if compute allows, adds the remaining ROCKET-family transforms [3, 5, 6, 7], FCN/ResNet [10], and the Ribeiro et al. DNN [18] — useful for completeness but not required to test the hypothesis. (1NN-DTW is omitted on 12-lead records as computationally impractical.)

**Mitigations tested.** Training-time noise augmentation (Gaussian, synthetic baseline wander, lead masking) and a test-time-adaptation baseline [24], reported as robustness *recovered*, with augmentation evaluated only on the *held-out* corruption family.

**Metrics.** Primary: a TS analogue of **mean Corruption Error** over the ECG error rate (error = 1 − macro-AUROC on PTB-XL diagnostic superclasses), `mCE_f = mean_c [ (Σ_s E^f_{c,s}) / (Σ_s E^{ref}_{c,s}) ]`, normalized to a fixed reference classifier, plus **relative mCE** (degradation above clean error). Secondary: macro-F1, the CinC-2020 challenge metric where applicable, per-corruption/per-severity breakdown, and multi-label calibration (ECE/Brier) under corruption. Clean macro-AUROC always reported alongside.

**Analysis.** Spearman rank-correlation between clean and mCE leaderboards (RQ1); per-family fragility heatmaps (RQ2); augmentation gain on seen vs. held-out corruptions (RQ3); cross-dataset external validity via PhysioNet/CinC 2020 [15]; all with bootstrap confidence intervals over records/seeds.

**Statistical comparison of models (complementary analysis, per advisor feedback).** Beyond point estimates, differences between models are tested for significance with the standard multiple-classifier-comparison protocol: a **Friedman test** across the corruption×severity conditions, followed by **post-hoc pairwise Wilcoxon signed-rank tests with Holm–Bonferroni correction**, summarized as **critical-difference (CD) diagrams** over mCE ranks [26, 27]. Each model's clean→corrupted degradation is assessed with **paired tests** (paired bootstrap / Wilcoxon signed-rank) on per-record scores, reporting **effect sizes alongside p-values**, and — where informative — a **Bayesian signed-rank** alternative to null-hypothesis testing [28]. This is positioned as a supplementary rigor layer that strengthens the ranking claims, not the paper's central contribution.

## 6. Rigor and pre-registration

The corruption set, severity-calibration formulas, model list, primary metric, and the **decision rule are pre-registered before any evaluation.** Concretely, the confirmatory test for RQ1: *clean-vs-mCE Spearman ρ < 0.7 across the model zoo* (bootstrap CI excluding 0.9) confirms "clean accuracy is a weak proxy for robustness." Strict no-leakage rules: models never see benchmark corruptions; a disjoint corruption family is held out for augmentation tests; recommended train/test folds (PTB-XL) used as published; frozen inference settings per model; bootstrap CIs over records and seeds; honest reporting of nulls. The suite is released versioned with seeds, generator code, environment, and result tables as a reusable artifact.

## 7. Contributions (stated honestly)

1. **ECG-C**: the first general, physically-grounded, severity-calibrated **common-corruption benchmark for ECG classification** (12-lead and single-lead), with corruptions anchored in *real recorded* clinical noise (NSTDB).
2. A **standardized robustness metric** (ECG mean Corruption Error + relative mCE) and a reusable evaluation harness.
3. The first **clean-vs-robust ranking audit** of the ECG-classification model zoo — a falsifiable test of whether clean accuracy predicts robustness, plus per-family fragility maps.
4. A **held-out-corruption augmentation study** showing whether cheap augmentation genuinely generalizes or merely overfits the benchmark.

The claim is never "we discovered corruption robustness"; it is the ECG-specific suite + standardized metric + the falsifiable ranking test.

## 8. Related-work positioning (foundations, not inventions)

Cited as foundations: the common-corruption protocol and metric [1, 2]; the ECG-classification model zoo being audited [3–10, 15, 18]; the ECG datasets [11–15]; the ECG noise taxonomy, the NSTDB, and PhysioNet [16, 17, 20]; adjacent (but distinct) robustness/OOD work [21, 22]; augmentation used as mitigation [23]. The contribution is the benchmark, the metric, and the falsifiable audit — never the invention of robustness evaluation.

## 9. Risks and mitigations

- **Corruption realism is contestable (primary).** Mitigation: anchor the core corruptions in *real recorded* NSTDB noise [16], ground every synthetic generator in a cited ECG-noise mechanism [20], and calibrate severities by SNR/physical parameters; version the suite so the design is auditable.
- **"It's just adding noise."** Mitigation: the distinctions are real recorded clinical noise, physical severity calibration, the held-out-augmentation protocol, and the falsifiable ranking analysis — none of which naive noise injection provides.
- **Corruption vs. domain shift confound.** Mitigation: keep signal-level corruptions strictly separate from cross-hospital/device *domain* shift; report cross-dataset (PhysioNet/CinC 2020) as a separate external-validity analysis, not mixed into mCE.
- **Multi-label evaluation subtleties.** Mitigation: use established PTB-XL macro-AUROC and the CinC-2020 metric; report per-class results; avoid averaging incomparable label sets across datasets.
- **Compute.** 12-lead, full-length records are heavier than UCR-scale series. Mitigation: use published folds, subsample where needed, document budgets and wall-clock honestly.
- **Scope/generality.** ECG-only by design (advisor feedback); positioned as a focused, defensible benchmark with portability to IMU/PPG as future work — a strength, not a limitation.
- **Data access.** All datasets (PTB-XL, Chapman-Shaoxing, CPSC2018, MIT-BIH, NSTDB) are openly available on PhysioNet/public archives — no credentialed access required (unlike MIMIC), de-risking reproducibility.

## 10. Venue calibration (no exaggeration)

Two natural homes: a **NeurIPS/ICLR Datasets & Benchmarks** track (benchmark + audit fits exactly), or a **biomedical-signal venue** — *Physiological Measurement*, *Computers in Biology and Medicine*, or *IEEE J. Biomedical and Health Informatics* — where ECG robustness is directly in scope. ML alternatives: *Data Mining and Knowledge Discovery* (DAMI) or **TMLR** (rigorous, open, benchmark-friendly). Confirm quartiles on SJR/JCR and match each venue's current scope before submission.

## 11. Immediate next step (1-week pilot)

On **PTB-XL** (published folds) corrupt the test set with **4 generators × 5 severities** — NSTDB baseline wander, powerline interference, gain miscalibration, and quantization — and evaluate **MiniRocket, InceptionTime, and a 1D SE-ResNet** (three of the five primary models). Compute clean macro-AUROC and mCE; check whether the clean-vs-mCE Spearman ρ already drops below the pre-registered threshold on this slice. If the ranking breaks even here, the thesis is confirmed and the full suite (all corruptions + datasets + foundation features) is greenlit. Then write the full pre-registration.

## 12. Dataset, model & resource links

**ECG datasets / noise source**
- PTB-XL (12-lead, multi-label diagnostic; primary) — https://physionet.org/content/ptb-xl/
- Chapman-Shaoxing / Ningbo 12-lead ECG — https://physionet.org/content/ecg-arrhythmia/
- CPSC 2018 (China Physiological Signal Challenge) — http://2018.icbeb.org/Challenge.html
- MIT-BIH Arrhythmia Database (single/2-lead, beat-level) — https://physionet.org/content/mitdb/
- MIT-BIH Noise Stress Test Database (NSTDB; real recorded bw/ma/em noise) — https://physionet.org/content/nstdb/
- PhysioNet/CinC Challenge 2020 (multi-source 12-lead; cross-dataset validity) — https://physionet.org/content/challenge-2020/
- PhysioNet (host) — https://physionet.org/

**Models / libraries / harness**
- ImageNet-C reference implementation (protocol/metric) — https://github.com/hendrycks/robustness
- aeon (TSC toolkit: ROCKET-family, InceptionTime, catch22) — https://github.com/aeon-toolkit/aeon
- sktime — https://github.com/sktime/sktime
- tsai (deep TSC) — https://github.com/timeseriesAI/tsai
- NeuroKit2 (ECG processing / synthetic noise utilities) — https://github.com/neuropsychology/NeuroKit
- Mantis (time-series classification foundation model) — https://arxiv.org/pdf/2502.15637

## 13. References

1. Hendrycks, D., & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (ImageNet-C / ImageNet-P). ICLR. arXiv:1903.12261. https://arxiv.org/abs/1903.12261
2. Mintun, E., Kirillov, A., & Xie, S. (2021). *On Interaction Between Augmentations and Corruptions in Natural Corruption Robustness* (ImageNet-C-bar). NeurIPS. arXiv:2102.11273. https://arxiv.org/abs/2102.11273
3. Dempster, A., Petitjean, F., & Webb, G. I. (2020). *ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels.* Data Mining and Knowledge Discovery. arXiv:1910.13051. https://arxiv.org/abs/1910.13051
4. Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). *MiniRocket: A very fast (almost) deterministic transform for time series classification.* KDD. arXiv:2012.08791. https://arxiv.org/abs/2012.08791
5. Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). *MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification.* DAMI. arXiv:2102.00457. https://arxiv.org/abs/2102.00457
6. Dempster, A., Schmidt, D. F., & Webb, G. I. (2023). *Hydra: Competing convolutional kernels for fast and accurate time series classification.* DAMI. arXiv:2203.13652. https://arxiv.org/abs/2203.13652
7. Dempster, A., Schmidt, D. F., & Webb, G. I. (2024). *QUANT: A minimalist interval method for time series classification.* DAMI. arXiv:2308.00928. https://arxiv.org/abs/2308.00928
8. Ismail Fawaz, H., et al. (2020). *InceptionTime: Finding AlexNet for time series classification.* DAMI. arXiv:1909.04939. https://arxiv.org/abs/1909.04939
9. Lubba, C. H., et al. (2019). *catch22: CAnonical Time-series CHaracteristics.* DAMI. arXiv:1901.10200. https://arxiv.org/abs/1901.10200
10. Wang, Z., Yan, W., & Oates, T. (2017). *Time series classification from scratch with deep neural networks: A strong baseline* (FCN/ResNet). IJCNN. arXiv:1611.06455. https://arxiv.org/abs/1611.06455
11. Wagner, P., et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data 7, 154. https://physionet.org/content/ptb-xl/
12. Zheng, J., et al. (2020). *A 12-lead electrocardiogram database for arrhythmia research (Chapman-Shaoxing).* Scientific Data 7, 48. https://physionet.org/content/ecg-arrhythmia/
13. Liu, F., et al. (2018). *An open-access database for evaluating the algorithms of ECG rhythm and morphology abnormality detection (CPSC 2018).* Journal of Medical Imaging and Health Informatics. http://2018.icbeb.org/Challenge.html
14. Moody, G. B., & Mark, R. G. (2001). *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng. in Medicine and Biology. https://physionet.org/content/mitdb/
15. Alday, E. A. P., et al. (2020). *Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020.* Physiological Measurement. https://physionet.org/content/challenge-2020/
16. Moody, G. B., Muldrow, W. E., & Mark, R. G. (1984). *A noise stress test for arrhythmia detectors* (MIT-BIH Noise Stress Test Database, NSTDB). Computers in Cardiology. https://physionet.org/content/nstdb/
17. Goldberger, A. L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 101(23), e215–e220. https://physionet.org/
18. Ribeiro, A. H., et al. (2020). *Automatic diagnosis of the 12-lead ECG using a deep neural network.* Nature Communications 11, 1760. https://doi.org/10.1038/s41467-020-15432-4
19. Hannun, A. Y., et al. (2019). *Cardiologist-level arrhythmia detection and classification in ambulatory ECGs using a deep neural network.* Nature Medicine 25, 65–69. https://doi.org/10.1038/s41591-018-0268-3
20. Authors (2025). *Investigating the Generalizability of ECG Noise Detection Across Diverse Data Sources and Noise Types.* arXiv:2502.14522. https://arxiv.org/abs/2502.14522
21. Authors (2025). *Are Time-Series Foundation Models Deployment-Ready? A Systematic Study of Adversarial Robustness Across Domains.* arXiv:2505.19397. https://arxiv.org/abs/2505.19397
22. Gagnon-Audet, J.-C., et al. (2022). *WOODS: Benchmarks for Out-of-Distribution Generalization in Time Series.* arXiv:2203.09978. https://arxiv.org/abs/2203.09978
23. Iwana, B. K., & Uchida, S. (2021). *An empirical survey of data augmentation for time series classification with neural networks.* PLOS ONE. arXiv:2007.15951. https://arxiv.org/abs/2007.15951
24. Wang, D., et al. (2021). *Tent: Fully Test-Time Adaptation by Entropy Minimization.* ICLR. arXiv:2006.10726. https://arxiv.org/abs/2006.10726
25. Authors (2025). *Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification.* arXiv:2502.15637. https://arxiv.org/abs/2502.15637
26. Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* Journal of Machine Learning Research 7, 1–30. https://www.jmlr.org/papers/v7/demsar06a.html
27. García, S., & Herrera, F. (2008). *An Extension on "Statistical Comparisons of Classifiers over Multiple Data Sets" for all Pairwise Comparisons.* Journal of Machine Learning Research 9, 2677–2694. https://www.jmlr.org/papers/v9/garcia08a.html
28. Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). *Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis.* Journal of Machine Learning Research 18, 1–36. arXiv:1606.04316. https://arxiv.org/abs/1606.04316

**Verification status:** Citations [1–19, 22–24, 26–28] are well-established and were sanity-checked during preparation (June 2026). Entries marked "Authors (2025)" ([20], [21], [25]) are recent preprints whose exact author lists/venues should be confirmed at write-up. Confirm all arXiv IDs, DOIs, venues, and live dataset/repository links before submission; reference numbering will be tidied in the manuscript.
