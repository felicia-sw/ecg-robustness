# Are ECG Classifiers Robust to Realistic Sensor Corruptions?
### ECG-C: a physically-grounded common-corruption robustness audit on PTB-XL

*Focused report, July 2026. Companion to [ECG-C-proposal.md](ECG-C-proposal.md) and
[pre-registration.md](pre-registration.md). Code in `src/`, results in `results/`.*

---

## Abstract

ECG classification models are almost always validated on clean recordings, even though
they increasingly run on wearables where sensor noise is the rule rather than the
exception. We built ECG-C, a common-corruption suite for PTB-XL: five corruption families
at five calibrated severities, with baseline wander and muscle artifact taken from real
MIT-BIH NSTDB recordings. Using it we audited four CPU-trained classifiers (MiniRocket,
Rocket, Hydra, and catch22 with a ridge probe) against one pre-registered question, namely
whether clean diagnostic accuracy predicts robustness to corruption (clean-vs-mCE Spearman
ρ below 0.7 would indicate that it does not). The data do not support that hypothesis. We
measured ρ = 0.80, but with only four models the test is badly underpowered (p = 0.20), so
it cannot confirm the alternative either. What we can say is that no large re-ordering
occurred: the strongest model (MiniRocket) and the weakest (catch22) keep their positions
from clean to corrupted data, and the single swap in the middle of the table is not
statistically significant. Corruption-specific weaknesses are nonetheless easy to find.
Hydra scores well on clean data but degrades sharply under Gaussian noise and muscle
artifact, and the ROCKET models turn out to be insensitive to gain errors. Clean accuracy
is a reasonable first screen, then, but it hides failure modes that appear only once the
signal degrades.

## 1. Introduction

ECG classifiers trained on clean hospital archives are now being deployed on Holter
monitors, chest patches, and consumer wearables, and in those settings the signal is rarely
clean. The dominant failure modes there are not adversarial attacks but ordinary
corruptions: baseline wander, muscle (EMG) artifact, powerline interference, gain
miscalibration, low-bit quantization. Almost every ECG-classification benchmark, however,
reports accuracy on clean signals (PTB-XL, Chapman-Shaoxing, CPSC2018), which leaves an
obvious deployment question open: if a model scores well on clean recordings, does it hold
up once the signal degrades?

Computer vision answered the analogous question with ImageNet-C, which evaluates models on
common corruptions at graded severities, reports a normalized mean Corruption Error (mCE),
and forbids training on the corruptions themselves. We port that protocol to ECG and build
ECG-C on PTB-XL. The study tests a single pre-registered hypothesis: that clean accuracy is
a weak predictor of robustness, in the sense that the model ranking reorders under noise
(clean-vs-mCE Spearman ρ below 0.7). Scope is kept narrow on purpose — one dataset, four CPU
models, five corruptions — so this is a focused study rather than a full benchmark.

## 2. Related work

ImageNet-C [1] and its follow-up ImageNet-C-bar [2] established the protocol and the mCE
metric we borrow. The design (graded severities, no training on the corruptions) is by now
well validated and reproducible, but there is no equivalent for ECG classification.

There is a large ECG signal-processing literature on removing baseline wander, powerline
interference, and motion artifact, and the MIT-BIH Noise Stress Test Database [8] is the
standard source of real recorded noise. That work is about cleaning signals and estimating
quality, not about a standardized classification-robustness benchmark with a metric and a
model-ranking audit. Adversarial-robustness and time-series OOD benchmarks are also
adjacent but different: they study worst-case perturbations or cross-hospital and
cross-device shift, rather than physically grounded signal corruptions at controlled
severities.

The models we audit come from two families of time-series classifier: random convolutional
kernels (ROCKET [3], MiniRocket [4], Hydra [5]) and interpretable summary features
(catch22 [6]).

## 3. Method

### 3.1 Dataset
We use PTB-XL [7] at 100 Hz, 12-lead, 21,799 records, mapped to the five diagnostic
superclasses (NORM, MI, STTC, CD, HYP) through `scp_statements.csv`. The labels are
multi-label; class positives are NORM 9,514, MI 5,469, STTC 5,235, CD 4,898, and HYP 2,649,
and 411 records carry no diagnostic superclass. We follow the published `strat_fold` split:
folds 1–8 for training (17,418 records), fold 9 for validation (2,183), and fold 10 as the
held-out test set (2,198). Signals are stored channel-major, as arrays of shape
(n, 12, 1000).

### 3.2 Model zoo
The report uses four CPU-trained models spanning two paradigms. Each is a pipeline of a
time-series transform, a `StandardScaler`, and a ridge probe: MiniRocket with 10,000
kernels, Rocket with 2,000 kernels, and Hydra from the ROCKET family, plus catch22 (22
interpretable features per lead) as the feature-based approach. The probe is a multi-output
`RidgeCV` fit on the 0/1 label matrix. We use ridge rather than a logistic head for two
reasons: macro-AUROC is rank-based, so ridge's continuous outputs are a valid per-class
score, and the closed-form solve is far faster than an iterative fit. Transforms are applied
in row-batches to keep peak memory bounded, which matters for the torch-based Hydra. We fit
the transform parameters on the full training set; in practice only MiniRocket learns
anything data-dependent (its bias quantiles), since Rocket and Hydra use random kernels and
catch22 is a fixed feature extractor.

Deep models (InceptionTime, 1D SE-ResNet) and the Mantis foundation-feature probe need a
GPU and are left for future work. The four-model zoo is the main constraint on the ρ
estimate, and we return to it in Section 6.

### 3.3 The ECG-C corruption suite
Five corruption families are applied to the test set only; the models never see corrupted
data. Severity is tied to a physical parameter so that it increases monotonically and is
comparable across records.

| Corruption | Type | Severity ladder |
|---|---|---|
| Baseline wander (bw) | real NSTDB noise | SNR = {18, 12, 6, 0, −6} dB |
| Muscle artifact (ma) | real NSTDB noise | SNR = {18, 12, 6, 0, −6} dB |
| Gaussian | synthetic white noise | SNR = {18, 12, 6, 0, −6} dB |
| Gain miscalibration | multiplicative | gain = {1.1, 1.25, 1.5, 2.0, 3.0} |
| Quantization | reduced ADC bit-depth | bits = {10, 8, 6, 5, 4} |

The NSTDB records are sampled at 360 Hz. We resample them to 100 Hz, take independent random
windows per lead, and scale each window per (record, lead) to hit the target SNR
(`scale = √(P_signal / (P_noise · 10^(SNR/10)))`). We checked the calibration and the
measured SNR matched every target to within 0.00 dB. Powerline interference belongs to the
ECG-C design but we had to drop it at 100 Hz: 50 Hz mains sits at the Nyquist frequency
(fs/2 = 50 Hz), so the tone cannot be represented without aliasing. We defer it to the 500 Hz
records; Section 6 gives the detail.

### 3.4 Metrics
The primary metric is macro-AUROC, one-vs-rest over the five superclasses. For robustness we
use a mean Corruption Error, defined with `error = 1 − macro-AUROC` and normalized to a fixed
reference model (MiniRocket): `CE(f,c) = Σ_s error(f,c,s) / Σ_s error(ref,c,s)` and
`mCE(f) = mean_c CE(f,c)`. We also report a relative mCE that measures degradation above each
model's own clean error, and we always show clean macro-AUROC alongside. Using MiniRocket as
the reference is a choice; because it is the top model here (unlike ImageNet-C's weak AlexNet
baseline), every other mCE comes out above 1. That affects the scale but not the ranking,
which we checked is the same under all four possible reference models.

### 3.5 Statistical analysis
The pre-registered test for RQ1 is the Spearman correlation between the clean-accuracy
ranking and the robustness ranking (−mCE), with a 95% confidence interval from a
record-level bootstrap (1,000 resamples of the test records). Across the 25 (corruption ×
severity) conditions we run a Friedman test, follow it with pairwise Wilcoxon signed-rank
tests under Holm–Bonferroni correction, and summarize the model ranks with a
critical-difference diagram.

### 3.6 Pre-registration
We fixed the hypothesis, the decision rule (ρ < 0.7 with a bootstrap CI that excludes 0.9),
the model list, the corruption set, the severity ladders, and the metric before running the
full grid; see [pre-registration.md](pre-registration.md). Models are trained on clean data
only, using the published folds, and never on the benchmark corruptions. One deviation:
powerline was dropped after the fact once we found it degenerate at 100 Hz (Sections 3.3 and
6). It changes nothing that matters — ρ and the ranking are identical with and without it.

## 4. Results

Table 1 gives the leaderboard. mCE and relative mCE are normalized to MiniRocket, and lower
is more robust. Everything is evaluated on the 2,198-record test fold over five corruptions.

**Table 1. Leaderboard.**

| Model | Clean AUROC | mCE | Relative mCE |
|---|---|---|---|
| MiniRocket | 0.9011 | 1.000 | 1.000 |
| Rocket | 0.8875 | 1.041 | 0.759 |
| Hydra | 0.8912 | 1.198 | 1.231 |
| catch22 + ridge | 0.8406 | 1.567 | 2.424 |

The clean ranking (MiniRocket, Hydra, Rocket, catch22) and the mCE ranking (MiniRocket,
Rocket, Hydra, catch22) differ only in that Rocket and Hydra trade places, and that swap is
not statistically significant (Wilcoxon p_holm = 0.10, below). The best and worst models are
unchanged. The relative-mCE column tells a slightly different story: Rocket has the smallest
value (0.759), meaning it degrades less than MiniRocket relative to its own clean baseline,
a consequence of its gain-invariance and gentle decay (Section 5).

For RQ1, the Spearman correlation between the clean and robustness (−mCE) rankings is 0.800
(p = 0.200; record-level bootstrap 95% CI [0.800, 1.000]). This does not meet the
pre-registered rule of ρ < 0.7 with a CI excluding 0.9. With four models the test is
underpowered — ρ can take only a few discrete values, the p-value is not significant, and
the interval is degenerate — so we read RQ1 as inconclusive. No strong re-ordering appeared,
but we can neither confirm a tight coupling between clean accuracy and robustness nor rule
out a weak effect.

The models are clearly not interchangeable. The Friedman test over the 25 conditions gives
χ² = 41.9, p ≈ 4.3 × 10⁻⁹. Every pairwise Wilcoxon comparison is significant after Holm
correction except Rocket versus Hydra (p_holm = 0.10), which is exactly the pair that
produces the ranking difference. In other words, the two middle models are statistically
indistinguishable, and the swap between them is within noise.

The three figures (in `results/`) are the per-corruption CE heatmap (Table 2 below), the
macro-AUROC-versus-severity curves per corruption, and the critical-difference diagram over
ranks.

**Table 2. Per-corruption CE** (reference = MiniRocket; values above 1 mean more fragile
than MiniRocket).

| Model | bw | ma | gaussian | gain | quant |
|---|---|---|---|---|---|
| MiniRocket | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Rocket | 1.09 | 1.15 | 1.10 | 0.76 | 1.11 |
| Hydra | 1.01 | 1.42 | 1.65 | 0.79 | 1.11 |
| catch22 + ridge | 1.71 | 1.69 | 1.27 | 1.25 | 1.91 |

## 5. Analysis and discussion

The headline is that clean accuracy tracks the robustness ordering here reasonably well
(ρ = 0.80): the two extremes are fixed and only the middle pair swaps, and that swap is not
significant. This runs against the ImageNet-C-style expectation we pre-registered. It is not
a positive result in the other direction either, because four models cannot establish a
tight coupling; the fair reading is that we did not detect a strong re-ordering.

The per-corruption view (Table 2) is more informative, and it shows blind spots that clean
accuracy misses. Hydra is the second-best model on clean data but is unusually fragile to
high-variance additive noise: its CE is 1.65 on Gaussian and 1.42 on muscle artifact, well
above MiniRocket's. That weakness is what pulls its aggregate robustness down toward Rocket's
level despite the stronger clean score. catch22 is the most fragile model across the board
(CE from 1.25 to 1.91), and worst on quantization and baseline wander, which fits the idea
that hand-crafted summary features lose discriminative power as the waveform degrades.

The ROCKET models are effectively immune to gain errors, with CE of 0.76 and 0.79. The
reason is mundane: aeon's `Rocket` z-normalizes each series by default, so a change in
amplitude is erased before classification. This is a real robustness property but it comes
from a configuration default rather than anything intrinsic to the architecture, and
MiniRocket, which does not normalize, is affected by gain. A side effect is that the gain
corruption barely tests the normalizing models at all. Across the suite, Gaussian noise and
muscle artifact do the most damage to the strong models, while quantization down to four
bits is the mildest.

It is worth separating two senses of "robust." By absolute mCE, MiniRocket is the most
robust model. By relative mCE, which measures how far each model falls from its own clean
baseline, Rocket comes out ahead: it starts lower but decays gently and ignores gain. A
deployment that cares about worst-case absolute performance and one that cares about graceful
degradation would not pick the same model.

For deployment the practical lesson is straightforward. Clean accuracy is a fine first filter
and it correctly picks out the best and worst models here, but it says nothing about Hydra's
weakness to Gaussian noise and muscle artifact, which is exactly the kind of failure that
shows up on a noisy wearable. Robustness has to be measured directly, and that is what ECG-C
is for.

## 6. Limitations and future work

The most important limitation is the powerline corruption at 100 Hz. Since 50 Hz mains is at
the Nyquist frequency for a 100 Hz signal, the tone cannot be represented without aliasing,
so we excluded it and left it for the 500 Hz PTB-XL records. The four-model CPU zoo is the
next constraint: it leaves the ρ test underpowered, and the deep models (InceptionTime,
SE-ResNet) and the Mantis foundation-feature probe, which would round out the zoo, need a
GPU. We also use a single training seed per model, so we have not quantified how stable the
ranking is to training randomness; the bootstrap only covers sampling of test records.
Finally, the study uses one dataset. Cross-dataset validity (Chapman, CPSC, CinC-2020), the
remaining corruptions (electrode motion, lead dropout, sampling-rate mismatch), and the
mitigation experiments (augmentation, test-time adaptation with a held-out corruption family)
are all left for later.

## 7. Conclusion

We pre-registered and tested whether clean diagnostic accuracy is a weak proxy for corruption
robustness, using four PTB-XL classifiers and five physically grounded corruptions. The test
came out inconclusive: the Spearman correlation of 0.80 does not meet the ρ < 0.7 rule, and
with four models it can neither confirm nor rule out a coupling, while the one middle-of-table
swap is not significant. The more useful findings sit underneath that single number.
Robustness depends on how you define it, since Rocket wins on relative degradation, and clean
accuracy hides real corruption-specific weaknesses, the clearest being Hydra's fragility to
Gaussian noise and muscle artifact. The practical point survives regardless of how the
hypothesis lands: for deployment on noisy hardware, a clean leaderboard is a starting point,
not a substitute for measuring robustness directly. The main things holding back a firmer
answer are the small zoo and the single seed, and enlarging the zoo with deep and foundation
models, together with cross-dataset evaluation, is the obvious next step.

## References

1. Hendrycks, D., & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations (ImageNet-C). ICLR. arXiv:1903.12261.
2. Mintun, E., Kirillov, A., & Xie, S. (2021). On Interaction Between Augmentations and Corruptions in Natural Corruption Robustness (ImageNet-C-bar). NeurIPS. arXiv:2102.11273.
3. Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. Data Mining and Knowledge Discovery. arXiv:1910.13051.
4. Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). MiniRocket: A very fast (almost) deterministic transform for time series classification. KDD. arXiv:2012.08791.
5. Dempster, A., Schmidt, D. F., & Webb, G. I. (2023). Hydra: Competing convolutional kernels for fast and accurate time series classification. Data Mining and Knowledge Discovery. arXiv:2203.13652.
6. Lubba, C. H., et al. (2019). catch22: CAnonical Time-series CHaracteristics. Data Mining and Knowledge Discovery. arXiv:1901.10200.
7. Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data 7, 154.
8. Moody, G. B., Muldrow, W. E., & Mark, R. G. (1984). A noise stress test for arrhythmia detectors (MIT-BIH Noise Stress Test Database). Computers in Cardiology.
9. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23), e215–e220.
10. Middlehurst, M., et al. (2024). aeon: a Python toolkit for learning from time series. JMLR (software).
11. Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR 7, 1–30.
12. García, S., & Herrera, F. (2008). An Extension on "Statistical Comparisons of Classifiers over Multiple Data Sets" for all Pairwise Comparisons. JMLR 9, 2677–2694.

*The full 28-entry reference list is in [ECG-C-proposal.md](ECG-C-proposal.md) §13. arXiv IDs
and DOIs should be checked against live sources before any external submission; entries [20],
[21], and [25] there are still placeholders with unresolved author lists.*
