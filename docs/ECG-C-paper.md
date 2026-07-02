# Are ECG Classifiers Robust to Realistic Sensor Corruptions? A Physically-Grounded Common-Corruption Audit on PTB-XL

**Author:** [Your name], [Course / Institution]
**Date:** July 2026

---

## Abstract

Electrocardiogram (ECG) classification models are almost always validated on clean recordings, yet they increasingly run on wearable and ambulatory hardware where sensor noise is the norm. This paper asks whether clean diagnostic accuracy predicts robustness to realistic corruption. We build ECG-C, a common-corruption suite for the PTB-XL dataset comprising five corruption families at five calibrated severities, with baseline wander and muscle artifact drawn from real MIT-BIH Noise Stress Test Database recordings. Using ECG-C we audit four CPU-trained classifiers (MiniRocket, Rocket, Hydra, and catch22 with a ridge probe) against one pre-registered hypothesis: that clean accuracy is a weak predictor of robustness, operationalised as a Spearman correlation below 0.7 between the clean-accuracy and mean-Corruption-Error (mCE) rankings. The hypothesis is not supported, but neither is its opposite. With only four models the Spearman correlation is confined to a few discrete values (here 0.80); the exact permutation test is far from significant (one-sided p = 0.17); and because the strongest and weakest models are rank-stable, only the two middle models can move, so the pre-registered threshold of 0.7 was in fact unreachable by construction. We therefore read the headline test as inconclusive rather than as evidence in either direction, and the informative findings lie beneath the aggregate. No large re-ordering occurs: the strongest model (MiniRocket) and the weakest (catch22) keep their positions, and the single middle-of-table swap is not statistically significant. Corruption-specific weaknesses are nonetheless clear: Hydra scores well on clean data but degrades sharply under Gaussian noise and muscle artifact, while Rocket and Hydra resist gain miscalibration — not by architecture but because the transform z-normalises each series by default in the implementation used (MiniRocket, unnormalised, is not gain-invariant). Clean accuracy tracks the robustness ordering here, but the study is too small to establish how tightly; robustness should be measured directly, which is what ECG-C enables.

**Keywords:** ECG classification; robustness benchmarking; common corruptions; PTB-XL; time-series classification; pre-registration

---

## 1. Introduction

ECG classifiers trained on clean hospital archives are now deployed on Holter monitors, chest patches, and consumer wearables, where the recorded signal is rarely clean. In those settings the dominant failure modes are not adversarial attacks but ordinary corruptions: baseline wander, muscle (EMG) artifact, powerline interference, gain miscalibration, and low-bit quantization. Standard ECG-classification benchmarks, however, report accuracy almost exclusively on clean signals such as PTB-XL, Chapman-Shaoxing, and CPSC2018. This leaves a practical question open: if a model scores well on clean recordings, does it hold up once the signal degrades?

Computer vision confronted the analogous question with ImageNet-C (Hendrycks & Dietterich, 2019), which evaluates models on common corruptions at graded severities, reports a normalized mean Corruption Error (mCE), and forbids training on the corruptions themselves. We transfer that protocol to ECG and build ECG-C on PTB-XL. The study tests a single pre-registered hypothesis: that clean accuracy is a weak predictor of robustness, in the sense that the model ranking reorders under noise, operationalised as a clean-versus-mCE Spearman correlation below 0.7. The scope is deliberately narrow (one dataset, four CPU-trained models, five corruptions), so this is a focused audit rather than a full benchmark.

The contributions are: (i) ECG-C, a small physically-grounded corruption suite for PTB-XL with calibrated severities and real recorded noise; (ii) a pre-registered robustness audit of four time-series classifiers reporting a normalized mCE and per-corruption fragility; and (iii) an honest null result on the headline hypothesis, together with the corruption-specific findings that the aggregate correlation obscures.

## 2. Related Work

**Common-corruption robustness.** ImageNet-C (Hendrycks & Dietterich, 2019) and its follow-up ImageNet-C-bar (Mintun et al., 2021) established the protocol and metric adopted here. The design is well validated in vision, but no equivalent exists for ECG classification.

**ECG signal quality and denoising.** A large body of work removes baseline wander, powerline interference, and motion artifact, and the MIT-BIH Noise Stress Test Database (Moody et al., 1984) is the standard source of real recorded noise. That literature targets signal cleaning and quality estimation rather than a standardized classification-robustness benchmark with a metric and a model-ranking audit.

**Adversarial and out-of-distribution evaluation.** Adversarial-robustness and time-series out-of-distribution benchmarks are adjacent but distinct: they study worst-case perturbations or cross-hospital and cross-device shift, rather than physically grounded corruptions applied at controlled severities.

**Time-series classifiers.** The audited models come from two families: random convolutional kernels (Dempster et al., 2020, 2021, 2023) and interpretable summary features (Lubba et al., 2019).

## 3. Methods

### 3.1 Dataset
We use PTB-XL (Wagner et al., 2020) at 100 Hz, 12-lead, comprising 21,799 records mapped to the five diagnostic superclasses (NORM, MI, STTC, CD, HYP) through the accompanying `scp_statements.csv`. Labels are multi-label; class positives are NORM 9,514, MI 5,469, STTC 5,235, CD 4,898, and HYP 2,649, and 411 records carry no diagnostic superclass. We follow the published `strat_fold` split: folds 1–8 for training (17,418 records), fold 9 for validation (2,183), and fold 10 as the held-out test set (2,198). Signals are stored channel-major with shape (n, 12, 1000).

### 3.2 Model zoo
The audit uses four CPU-trained models spanning two paradigms. Each is a pipeline of a time-series transform, feature standardisation, and a ridge probe: MiniRocket with 10,000 kernels, Rocket with 2,000 kernels, and Hydra, all from the ROCKET family, together with catch22 (22 interpretable features per lead) as the feature-based approach. The probe is a multi-output ridge regressor (`RidgeCV`) fit on the 0/1 label matrix. Ridge is preferred over a logistic head for two reasons: macro-AUROC is rank-based, so ridge's continuous outputs are a valid per-class score, and the closed-form solution is much faster than an iterative fit. Transforms are applied in row-batches to keep peak memory bounded, which matters for the torch-based Hydra. Transform parameters are fit on the full training set; in practice only MiniRocket learns data-dependent parameters (its bias quantiles), because Rocket and Hydra use random kernels and catch22 is a fixed feature extractor.

Deep models (InceptionTime, 1D SE-ResNet) and a foundation-feature probe require a GPU and are left for future work. The four-model zoo is the principal constraint on the correlation estimate (Section 6).

### 3.3 The ECG-C corruption suite
Five corruption families are applied to the test set only; models never see corrupted data during training. Severity is tied to a physical parameter so that it increases monotonically and is comparable across records.

| Corruption | Type | Severity ladder |
|---|---|---|
| Baseline wander (bw) | Real NSTDB noise | SNR = {18, 12, 6, 0, −6} dB |
| Muscle artifact (ma) | Real NSTDB noise | SNR = {18, 12, 6, 0, −6} dB |
| Gaussian | Synthetic white noise | SNR = {18, 12, 6, 0, −6} dB |
| Gain miscalibration | Multiplicative | Gain = {1.1, 1.25, 1.5, 2.0, 3.0} |
| Quantization | Reduced ADC bit-depth | Bits = {10, 8, 6, 5, 4} |

The NSTDB records are sampled at 360 Hz. We resample them to 100 Hz, draw independent random windows per lead, and scale each window per (record, lead) to reach the target signal-to-noise ratio, using scale = √(P_signal / (P_noise · 10^(SNR/10))). The calibration was verified: measured SNR matched every target to within 0.00 dB. Powerline interference belongs to the ECG-C design but was excluded at 100 Hz because 50 Hz mains sits at the Nyquist frequency (fs/2 = 50 Hz) and cannot be represented without aliasing; it is deferred to the 500 Hz records (Section 6).

### 3.4 Metrics
The primary metric is macro-AUROC (one-versus-rest over the five superclasses). Robustness is summarised by a mean Corruption Error defined with error = 1 − macro-AUROC and normalized to a fixed reference model (MiniRocket): CE(f, c) = Σ_s error(f, c, s) / Σ_s error(ref, c, s) and mCE(f) = mean_c CE(f, c). We also report a relative mCE that measures degradation above each model's own clean error, and clean macro-AUROC is always reported alongside. Using MiniRocket as the reference is a choice; because it is the top model here, every other mCE exceeds 1. This affects the scale but not the ranking, which was verified to be identical under all four possible reference models.

### 3.5 Statistical analysis
The pre-registered test for the research question is the Spearman correlation between the clean-accuracy ranking and the robustness ranking (−mCE). Because the zoo is small, we assess it with the *exact permutation test* (enumerating all 4! relabellings) rather than the asymptotic t-approximation, which is invalid at n = 4, and we report the full set of correlation values attainable at this n. We also report a record-level bootstrap (1,000 resamples of the test records); this quantifies the stability of ρ to the sampling of *test records* with the four models held fixed, and is therefore a statement about record sampling, not about uncertainty over the population of models — which four fixed models cannot capture. Across the 25 (corruption × severity) conditions we run a Friedman test, followed by pairwise Wilcoxon signed-rank tests under Holm–Bonferroni correction and a critical-difference diagram (Demšar, 2006; García & Herrera, 2008). Because the five severities within a family are a dependent ladder rather than independent blocks, we additionally report a *family-blocked* Friedman test (severities averaged to five near-independent blocks) as a conservative check on the significance.

### 3.6 Pre-registration
The hypothesis, the decision rule (ρ < 0.7 with a bootstrap CI excluding 0.9), the model list, the corruption set, the severity ladders, and the metric were fixed before the full evaluation was run. Models are trained on clean data only, using the published folds, and never on the benchmark corruptions. Four deviations are recorded. (i) Powerline was dropped after the fact once it was found to be degenerate at 100 Hz (Sections 3.3 and 6); this is immaterial to the conclusions, as the correlation and the ranking are identical with and without it. (ii) The pre-registered bootstrap was specified "over records and seeds", but a single training seed was used, so the bootstrap covers only record sampling and the ranking's stability to training randomness is not yet quantified (Section 6); the released runner supports multiple seeds for this purpose. (iii) The pre-registered per-model effect sizes and Bayesian signed-rank analysis were not run and are left for future work. (iv) Most consequentially, the decision rule itself proved ill-posed at this zoo size: with four models ρ is confined to the discrete set {..., 0.6, 0.8, 1.0}, and once the best and worst models are rank-stable only the two middle models can move, so ρ can only be 0.8 or 1.0 and the threshold ρ < 0.7 was unreachable regardless of the data. We therefore report RQ1 as inconclusive and treat the rule as a lesson about powering a rank-correlation test, not as a hypothesis the data cleanly confirmed or rejected.

## 4. Results

Table 1 reports the leaderboard. mCE and relative mCE are normalized to MiniRocket, and lower values indicate greater robustness. All values are computed on the 2,198-record test fold across five corruptions.

**Table 1.** Clean accuracy and robustness (mCE, relative mCE) for the four models.

| Model | Clean AUROC | mCE | Relative mCE |
|---|---|---|---|
| MiniRocket | 0.9011 | 1.000 | 1.000 |
| Rocket | 0.8875 | 1.041 | 0.759 |
| Hydra | 0.8912 | 1.198 | 1.231 |
| catch22 + ridge | 0.8406 | 1.567 | 2.424 |

The clean ranking (MiniRocket, Hydra, Rocket, catch22) and the mCE ranking (MiniRocket, Rocket, Hydra, catch22) differ only in that Rocket and Hydra trade places, and that swap is not statistically significant (Wilcoxon p_holm = 0.10, reported below). The best and worst models are unchanged. The relative-mCE column tells a slightly different story: Rocket has the smallest value (0.759), meaning it degrades less than MiniRocket relative to its own clean baseline, a consequence of its gain-invariance and gentle decay (Section 5).

For the pre-registered question, the Spearman correlation between the clean and robustness (−mCE) rankings is 0.800. The exact permutation test gives one-sided p = 0.167 and two-sided p = 0.333; the asymptotic t-approximation (p = 0.200) is invalid at n = 4 and is not used. At four models ρ can take only the values {−1.0, −0.8, −0.6, …, 0.6, 0.8, 1.0}, and with MiniRocket and catch22 rank-stable at the extremes only 0.8 or 1.0 is attainable — so the pre-registered rule ρ < 0.7 could not have been met regardless of the data (Section 3.6). The record-level bootstrap concentrates on exactly those two values, {0.8, 1.0}, giving the degenerate interval [0.800, 1.000]; and because it resamples records with the models fixed, it reflects record-sampling stability, not uncertainty over the space of models. We therefore read the result as inconclusive: no strong re-ordering appeared, but the study can neither confirm a tight coupling between clean accuracy and robustness nor rule out a weak effect. For reference, even perfect rank agreement (ρ = 1.0) would give a two-sided permutation p of only 0.083 at n = 4, so no four-model result could reach significance.

The models are clearly not interchangeable. The Friedman test over the 25 conditions gives χ² = 41.9, p ≈ 4.3 × 10⁻⁹. Every pairwise Wilcoxon comparison is significant after Holm correction except Rocket versus Hydra (p_holm = 0.10), which is exactly the pair that produces the ranking difference; the two middle models are thus statistically indistinguishable, and the swap between them lies within noise. Because the five severities within a family are a dependent ladder rather than independent blocks, the 25-condition test is anti-conservative; a family-blocked Friedman over five near-independent blocks is far weaker but still significant (χ² = 9.0, p = 0.029), so the conclusion that the models differ survives the more conservative test.

Figure 1 shows per-corruption fragility, Figure 2 the macro-AUROC-versus-severity curves, and Figure 3 the critical-difference diagram over ranks.

![Figure 1. Per-corruption Corruption Error (reference = MiniRocket). Values above 1 indicate greater fragility than the reference.](/Users/feliciasword/ecg-robustness/results/fig_fragility_heatmap.png)

![Figure 2. Macro-AUROC as a function of severity, by corruption family.](/Users/feliciasword/ecg-robustness/results/fig_severity_curves.png)

![Figure 3. Critical-difference diagram over mean ranks across the 25 conditions.](/Users/feliciasword/ecg-robustness/results/fig_cd_diagram.png)

Table 2 gives the per-corruption Corruption Error underlying Figure 1.

**Table 2.** Per-corruption CE (reference = MiniRocket; values above 1 indicate greater fragility than the reference).

| Model | bw | ma | Gaussian | Gain | Quant |
|---|---|---|---|---|---|
| MiniRocket | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Rocket | 1.09 | 1.15 | 1.10 | 0.76 | 1.11 |
| Hydra | 1.01 | 1.42 | 1.65 | 0.79 | 1.11 |
| catch22 + ridge | 1.71 | 1.69 | 1.27 | 1.25 | 1.91 |

## 5. Discussion

The headline is that clean accuracy tracks the robustness ordering reasonably well here (ρ = 0.80): the two extremes are fixed, only the middle pair swaps, and that swap is not significant. This runs against the ImageNet-C-style expectation the study pre-registered. It is not a positive result in the opposite direction either, because four models cannot establish a tight coupling; the fair reading is that no strong re-ordering was detected.

The per-corruption view (Table 2) is more informative and reveals weaknesses that clean accuracy hides. Hydra is the second-best model on clean data but is unusually fragile to high-variance additive noise, with a Corruption Error of 1.65 on Gaussian and 1.42 on muscle artifact, both well above MiniRocket. This weakness is what pulls its aggregate robustness down toward Rocket's level despite the stronger clean score. catch22 is the most fragile model across the board (CE from 1.25 to 1.91) and worst on quantization and baseline wander, consistent with hand-crafted summary features losing discriminative power as the waveform degrades.

The ROCKET models are effectively immune to gain errors, with Corruption Errors of 0.76 and 0.79. The reason is mechanical: the Rocket transform z-normalises each series by default in the implementation used here, so a change in amplitude is removed before classification. This is a genuine robustness property, but it follows from a configuration default rather than anything intrinsic to the architecture, and MiniRocket, which does not normalise, is affected by gain. A side effect is that the gain corruption barely tests the normalising models. Across the suite, Gaussian noise and muscle artifact do the most damage to the strong models, while quantization down to four bits is the mildest corruption.

Two senses of "robust" should be separated. By absolute mCE, MiniRocket is the most robust model. By relative mCE, which measures how far each model falls from its own clean baseline, Rocket is ahead: it starts lower but decays gently and ignores gain. A deployment that prioritises worst-case absolute performance and one that prioritises graceful degradation would not select the same model.

For deployment the practical implication is direct. Clean accuracy is a useful first filter and it correctly identifies the best and worst models here, but it gives no warning of Hydra's fragility to Gaussian noise and muscle artifact, which is precisely the kind of failure a noisy wearable would surface. Robustness has to be measured directly, which is the purpose of ECG-C.

## 6. Limitations and Future Work

The four-model zoo is the most consequential limitation, and it bites twice. First, four points leave the rank-correlation test not merely underpowered but ill-posed: as shown in Sections 3.6 and 4, the pre-registered threshold was unreachable at this n. Second, three of the four models (MiniRocket, Rocket, Hydra) belong to the ROCKET family and share the same random-convolution-plus-linear mechanism, so the four correlation points represent only about two independent modelling paradigms — the effective diversity is even smaller than n = 4 suggests. Enlarging and diversifying the zoo with the deep models (InceptionTime, SE-ResNet) and the foundation-feature probe (Mantis) that require a GPU is therefore the single highest-leverage next step, and the released runner already supports it (`make eval-seeds`, and additional model builders). A single training seed was used per model, so the stability of the ranking to training randomness is not yet quantified; the bootstrap covers only the sampling of test records, and multi-seed runs are the immediate remedy. The powerline corruption is excluded at 100 Hz because 50 Hz mains lies at the Nyquist frequency and cannot be represented without aliasing; it is deferred to the 500 Hz PTB-XL records. Finally, the study uses one dataset. Cross-dataset validity (Chapman, CPSC, CinC-2020), the remaining corruptions (electrode motion, lead dropout, sampling-rate mismatch), and mitigation experiments (augmentation, test-time adaptation with a held-out corruption family) are left for future work.

## 7. Conclusion

The study pre-registered and tested whether clean diagnostic accuracy is a weak proxy for corruption robustness, using four PTB-XL classifiers and five physically grounded corruptions. The test was inconclusive: the Spearman correlation of 0.80 does not meet the ρ < 0.7 rule, and with four models it can neither confirm nor rule out a coupling, while the single middle-of-table swap is not significant. The more useful findings sit beneath that single number. Robustness depends on how it is defined, since Rocket wins on relative degradation, and clean accuracy conceals real corruption-specific weaknesses, the clearest being Hydra's fragility to Gaussian noise and muscle artifact. The practical point holds regardless of how the hypothesis lands: for deployment on noisy hardware, a clean leaderboard is a starting point, not a substitute for measuring robustness directly. The main obstacles to a firmer answer are the small zoo and the single seed, and enlarging the zoo with deep and foundation models, alongside cross-dataset evaluation, is the natural next step.

## Declarations

**Data and code availability.** All data are publicly available: PTB-XL and the MIT-BIH Noise Stress Test Database are distributed on PhysioNet (Goldberger et al., 2000). The accompanying repository contains the corruption suite, the model zoo, the end-to-end evaluation runner (`src/run_eval.py`), the analysis code, and a test suite. After the data are downloaded, the full grid and per-condition predictions are regenerated with `make eval` (equivalently `python -m src.run_eval --seeds 0`), all tables and figures with `make analysis`, and the multi-seed stability check with `make eval-seeds`. Exact dependency versions are pinned in `requirements.lock.txt` (produced by `make freeze`); the aeon version in particular is load-bearing, because the gain-miscalibration result depends on that library's default per-series normalisation.

**Ethics.** This work uses only de-identified, publicly released datasets and involves no new human-subjects data collection, so no additional ethics approval was required.

**Author contributions (CRediT).** [Your name]: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review and editing, Visualization.

**Funding.** This research received no external funding.

**Conflict of interest.** The author declares no competing interests.

**Use of AI tools.** Generative AI tools (Anthropic Claude, via the Claude Code environment) were used to assist with code implementation, running the evaluation pipeline, statistical analysis, figure generation, and drafting and editing of this manuscript. All experimental results are produced by the described code on the cited public data and were verified by the author, who takes full responsibility for the content.

## References

Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. *Data Mining and Knowledge Discovery, 34*(5), 1454–1495. https://arxiv.org/abs/1910.13051

Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). MiniRocket: A very fast (almost) deterministic transform for time series classification. In *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining* (pp. 248–257). https://arxiv.org/abs/2012.08791

Dempster, A., Schmidt, D. F., & Webb, G. I. (2023). Hydra: Competing convolutional kernels for fast and accurate time series classification. *Data Mining and Knowledge Discovery, 37*, 1779–1805. https://arxiv.org/abs/2203.13652

Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7*, 1–30.

García, S., & Herrera, F. (2008). An extension on "Statistical comparisons of classifiers over multiple data sets" for all pairwise comparisons. *Journal of Machine Learning Research, 9*, 2677–2694.

Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation, 101*(23), e215–e220.

Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. In *International Conference on Learning Representations*. https://arxiv.org/abs/1903.12261

Lubba, C. H., Sethi, S. S., Knaute, P., Schultz, S. R., Fulcher, B. D., & Jones, N. S. (2019). catch22: CAnonical Time-series CHaracteristics. *Data Mining and Knowledge Discovery, 33*(6), 1821–1852. https://arxiv.org/abs/1901.10200

Mintun, E., Kirillov, A., & Xie, S. (2021). On interaction between augmentations and corruptions in natural corruption robustness. In *Advances in Neural Information Processing Systems* (Vol. 34). https://arxiv.org/abs/2102.11273

Moody, G. B., Muldrow, W. E., & Mark, R. G. (1984). A noise stress test for arrhythmia detectors. *Computers in Cardiology, 11*, 381–384.

Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data, 7*, 154. https://doi.org/10.1038/s41597-020-0495-6
