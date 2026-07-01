# Are ECG Classifiers Robust to Realistic Sensor Corruptions?
### ECG-C: A physically-grounded common-corruption robustness audit on PTB-XL

*Focused report, July 2026. Companion to [ECG-C-proposal.md](ECG-C-proposal.md) and
[pre-registration.md](pre-registration.md). Reproducibility: code in `src/`, results in `results/`.*

---

## Abstract

_ECG classification models are validated almost exclusively on clean recordings, yet are
deployed on wearables where ordinary sensor corruptions dominate. We build ECG-C, a
physically-grounded common-corruption suite for PTB-XL — five corruption families at five
calibrated severities, anchored in real recorded MIT-BIH NSTDB noise — and audit a CPU
model zoo (MiniRocket, Rocket, Hydra, catch22+ridge). We pre-register the test that clean
diagnostic accuracy is a weak predictor of corruption robustness (clean-vs-mCE Spearman
ρ < 0.7). The test is **inconclusive**: ρ = 0.80 (p = 0.20; record-bootstrap 95% CI
[0.80, 1.00]) does not meet the rule, but with only four models it cannot confidently
support or refute it. What we can say is that no strong re-ordering appeared — the best
(MiniRocket) and worst (catch22) models are stable from clean to corrupted evaluation, and
the single mid-rank swap (Rocket↔Hydra) is not statistically significant. Corruption-specific
fragility is nonetheless real: Hydra, second on clean data, degrades most under Gaussian
noise and muscle artifact (per-corruption CE 1.65× and 1.42× the reference), while the
ROCKET family is invariant to gain miscalibration. Clean accuracy is a useful first filter
but hides model-specific blind spots._

## 1. Introduction

ECG classifiers are moving from clean hospital archives onto Holter monitors, patches,
and consumer wearables, where the real failure modes are not adversarial perturbations
but *ordinary signal corruptions*: baseline wander, muscle (EMG) artifact, powerline
interference, gain miscalibration, and low-bit quantization. Yet the ECG-classification
literature evaluates almost exclusively on **clean** signals (PTB-XL, Chapman-Shaoxing,
CPSC2018), leaving a basic deployment question unanswered: **does a model's clean
diagnostic accuracy predict how well it holds up under realistic noise?**

Borrowing the common-corruption protocol that ImageNet-C established for vision —
evaluate at graded severities, report a normalized mean Corruption Error (mCE), and never
train on the benchmark corruptions — we build **ECG-C** for PTB-XL and run a focused
robustness audit. We test one sharp, pre-registered hypothesis: clean accuracy is a
**weak** proxy for robustness, i.e., the model ranking re-orders under noise
(clean-vs-mCE Spearman **ρ < 0.7**). This report is deliberately scoped (one dataset, a
four-model CPU zoo, five corruptions) as a focused study rather than the full benchmark.

## 2. Related work

**Common-corruption robustness.** ImageNet-C [1] and ImageNet-C-bar [2] established the
protocol and the mCE metric we transfer here; the design (graded severities, no training on
the corruptions) is validated and reproducible. No analogue exists for ECG classification.

**ECG denoising / noise detection.** A large literature removes baseline wander /
powerline / motion artifact, and the MIT-BIH Noise Stress Test Database (NSTDB) [8] is the
standard *source* of real recorded noise. These target signal *cleaning/quality*, not a
standardized *classification-robustness* benchmark with a metric and a model-ranking audit.

**Adversarial and OOD work.** Adversarial-robustness and time-series OOD/domain-shift
benchmarks probe worst-case perturbations or cross-hospital/device shift — distinct from
synthetic, physically-grounded *signal corruptions* at controlled severities.

**Time-series classifiers.** The zoo we audit spans random convolutional kernels
(ROCKET [3] / MiniRocket [4] / Hydra [5]) and interpretable summary features (catch22 [6]).

## 3. Method

### 3.1 Dataset
PTB-XL [7] @ 100 Hz, 12-lead, 21,799 records, mapped to the 5 diagnostic **superclasses**
(NORM, MI, STTC, CD, HYP) via `scp_statements.csv`. Labels are **multi-label**
(class positives: NORM 9514, MI 5469, STTC 5235, CD 4898, HYP 2649; 411 records carry no
diagnostic superclass). We use the published `strat_fold` split: folds 1–8 train
(17,418), fold 9 validation (2,183), fold 10 test (2,198). Signals are channel-major
`(n, 12, 1000)`.

### 3.2 Model zoo (CPU)
Four models over two paradigms, each a `transform → StandardScaler → ridge probe`:
**MiniRocket** (10,000 kernels), **Rocket** (2,000 kernels), and **Hydra** (competing
convolutional kernels) from the ROCKET family, and **catch22+ridge** (22 interpretable
features per lead) as the feature-based paradigm. The probe is a multi-output `RidgeCV`
(closed-form) on the 0/1 label matrix; because macro-AUROC is rank-based, ridge's
continuous outputs are a valid per-class ranking signal, and ridge is both the standard
ROCKET head and dramatically faster than an iterative logistic probe. Transforms are
applied in row-batches to bound peak memory (needed for the torch-based Hydra).
Transform parameters are fit on the **full training set**; empirically only MiniRocket
learns data-dependent parameters (bias quantiles), while Rocket and Hydra use
data-independent random kernels and catch22 is a stateless extractor. Deep models
(InceptionTime, 1D SE-ResNet) and the Mantis foundation-feature probe are GPU-bound and
left as future work; the small zoo is a stated limitation for the ρ estimate.

### 3.3 The ECG-C corruption suite
Five corruption families, each at five severities, applied to the **test set only**
(models never see corrupted data). Severity is calibrated to a physical parameter so it
is monotone and comparable across records.

| Corruption | Type | Severity ladder |
|---|---|---|
| Baseline wander (bw) | **real** NSTDB noise | SNR = {18, 12, 6, 0, −6} dB |
| Muscle artifact (ma) | **real** NSTDB noise | SNR = {18, 12, 6, 0, −6} dB |
| Gaussian | synthetic (white noise) | SNR = {18, 12, 6, 0, −6} dB |
| Gain miscalibration | multiplicative | gain = {1.1, 1.25, 1.5, 2.0, 3.0} |
| Quantization | reduced ADC bit-depth | bits = {10, 8, 6, 5, 4} |

Real NSTDB records (360 Hz) are resampled to 100 Hz and added as independent per-lead
random windows, scaled per (record, lead) to the target SNR
(`scale = √(P_signal / (P_noise · 10^(SNR/10)))`). The calibration was verified: measured
SNR matched every target to 0.00 dB. **Powerline interference is part of the ECG-C design
but is excluded at 100 Hz:** 50 Hz mains lies at the Nyquist frequency (fs/2 = 50 Hz), so
the tone cannot be represented without aliasing; it is deferred to the 500 Hz records (§6).

### 3.4 Metrics
Primary: macro-AUROC (one-vs-rest over the 5 superclasses). Robustness: mean Corruption
Error, with `error = 1 − macro-AUROC`, normalized to a fixed reference model (MiniRocket):
`CE(f,c) = Σ_s error(f,c,s) / Σ_s error(ref,c,s)`, `mCE(f) = mean_c CE(f,c)`. We also report
relative mCE (degradation above each model's own clean error). Clean macro-AUROC is
always reported alongside. (Reference = MiniRocket is a choice; unlike ImageNet-C's weak
AlexNet baseline it is the top model here, so all mCE > 1 by construction — a caveat, not a
bias in the *ranking*, which is reference-invariant.)

### 3.5 Statistical analysis
The pre-registered RQ1 test is the Spearman ρ between the clean-accuracy ranking and the
robustness ranking (−mCE), with a **record-level bootstrap** 95% CI (1,000 resamples of
the test records). Across the 25 corruption×severity conditions we run a **Friedman** test,
post-hoc **pairwise Wilcoxon signed-rank with Holm–Bonferroni** correction, and a
**critical-difference diagram** over ranks.

### 3.6 Pre-registration
The hypothesis, decision rule (ρ < 0.7 with bootstrap CI excluding 0.9), model list,
corruption set, severity ladders, and metric were fixed **before** running the full grid
(see [pre-registration.md](pre-registration.md)). Models are trained on clean data only,
using published folds; no training on benchmark corruptions. **Deviation:** powerline was
dropped post-hoc after we found it degenerate at 100 Hz (§3.3, §6); no result depends on it
(ρ and the ranking are identical with or without it).

## 4. Results

**Table 1 — Leaderboard** (clean macro-AUROC; mCE and relative mCE normalized to MiniRocket;
lower mCE = more robust). Evaluated on the 2,198-record PTB-XL test fold, 5 corruptions.

| Model | Clean AUROC | mCE ↓ | Relative mCE ↓ |
|---|---|---|---|
| MiniRocket | **0.9011** | **1.000** | 1.000 |
| Rocket | 0.8875 | 1.041 | **0.759** |
| Hydra | 0.8912 | 1.198 | 1.231 |
| catch22+ridge | 0.8406 | 1.567 | 2.424 |

The clean ranking (MiniRocket > Hydra > Rocket > catch22) and the mCE ranking
(MiniRocket > Rocket > Hydra > catch22) differ only by a **Rocket↔Hydra swap** — and that
swap is **not statistically significant** (Wilcoxon p_holm = 0.10; see below); the best and
worst models are stable. By *relative* mCE — degradation above each model's own clean error
— **Rocket is the most robust** (0.76, i.e. it degrades *less* than MiniRocket relative to
its own baseline), a consequence of its gain-invariance and gentle decay (§5).

**RQ1 — clean accuracy vs robustness.** Spearman ρ between the clean and robustness
(−mCE) rankings = **0.800** (p = 0.200; record-level bootstrap 95% CI **[0.800, 1.000]**).
The pre-registered rule (ρ < 0.7 with CI excluding 0.9) is **not met**. With only four
models the test is underpowered (ρ takes few discrete values, p is not significant, the CI
is degenerate), so we treat RQ1 as **inconclusive**: no *strong* re-ordering appeared, but
we can neither confirm a tight clean↔robust coupling nor rule out a weak effect.

**Significance across the 25 corruption×severity conditions.** Friedman χ² = 41.9,
p ≈ 4.3 × 10⁻⁹ — the models are not interchangeable. Post-hoc Wilcoxon signed-rank with
Holm–Bonferroni: all pairs differ significantly **except Rocket vs Hydra** (p_holm = 0.10).
The two mid-rank models are statistically indistinguishable, so the swap that produces the
clean-vs-robust ranking difference is within noise.

**Figures** (in `results/`): `fig_fragility_heatmap.png` (per-corruption CE, Table 2 below),
`fig_severity_curves.png` (AUROC vs severity per corruption), `fig_cd_diagram.png`
(critical-difference diagram over ranks).

**Table 2 — Per-corruption CE** (ref = MiniRocket; > 1 = more fragile than MiniRocket).

| Model | bw | ma | gaussian | gain | quant |
|---|---|---|---|---|---|
| MiniRocket | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Rocket | 1.09 | 1.15 | 1.10 | **0.76** | 1.11 |
| Hydra | 1.01 | 1.42 | **1.65** | 0.79 | 1.11 |
| catch22+ridge | 1.71 | 1.69 | 1.27 | 1.25 | 1.91 |

## 5. Analysis and discussion

**RQ1 — the ranking mostly holds, but the test is inconclusive.** Clean accuracy tracks the
robustness ordering here (ρ = 0.80): the extremes are fixed and only the two middle models
trade places — and that swap is not statistically significant (Wilcoxon p_holm = 0.10). This
is the opposite of the ImageNet-C-style expectation we pre-registered, but with four models
we cannot claim a *tight* coupling either; the honest reading is that no strong re-ordering
was detected, not that clean accuracy is a proven proxy.

**RQ2 — differential fragility is real.** The per-corruption CE (Table 2) exposes
corruption-specific blind spots that clean accuracy does not reveal:
- **Hydra** is the second-best clean model but is **disproportionately fragile to Gaussian
  noise (CE 1.65×) and muscle artifact (1.42×)** — high-variance additive corruptions. This
  is what erodes its aggregate robustness toward Rocket's level despite a stronger clean score.
- **catch22+ridge** is uniformly the most fragile (CE 1.25–1.91 across all five), worst on
  quantization (1.91×) and baseline wander (1.71×) — consistent with hand-crafted summary
  features losing discriminative content as the waveform degrades.
- **The ROCKET family is gain-invariant** (Rocket/Hydra CE 0.76–0.79 on gain): aeon's
  `Rocket(normalise=True)` z-normalizes each series, so amplitude miscalibration is erased
  before classification. This is an implementation-conferred robustness property (a config
  default, not an intrinsic model property), and MiniRocket (no normalization) *is* affected;
  the `gain` corruption is thus a near-no-op for normalizing models.
- **Gaussian noise and muscle artifact are the most damaging corruptions** for the strong
  models; quantization (down to 4 bits) is the mildest.

**Absolute vs relative robustness.** By mCE (absolute), MiniRocket is most robust; by
*relative* mCE (how far each model falls from its own clean baseline), **Rocket** is most
robust (0.76) — it starts lower but decays gently and shrugs off gain. "Robust" is thus not
a single ordering: a deployment optimizing worst-case absolute performance and one optimizing
graceful degradation would pick different models.

**Takeaway for deployment.** Clean accuracy is a reasonable *first* filter but not a
sufficient one: it correctly identifies the best and worst models here, yet hides Hydra's
Gaussian/muscle-artifact brittleness — the kind of failure that matters on a noisy wearable.
Robustness must be measured directly, which is what ECG-C provides.

## 6. Limitations and future work

**Powerline at 100 Hz.** 50 Hz mains sits at the Nyquist frequency for 100 Hz signals, so a
powerline tone cannot be represented without aliasing; we excluded it and defer it to the
500 Hz PTB-XL records. **Small CPU zoo** of four models across two paradigms leaves the ρ
test underpowered; the deep models (InceptionTime, SE-ResNet) and the Mantis foundation-feature
probe are GPU-bound and deferred. **Single dataset** (PTB-XL) — cross-dataset external validity
(Chapman/CPSC/CinC-2020) is future work. **Single training seed** per model — the ranking's
seed-stability is unquantified (the bootstrap covers test-record sampling only). The remaining
proposed corruptions (electrode motion, lead dropout, sampling-rate mismatch) and the mitigation
studies (augmentation, test-time adaptation with a held-out corruption family) are out of scope.

## 7. Conclusion

On a focused four-model CPU zoo of PTB-XL classifiers under five physically-grounded
corruptions, we pre-registered and tested whether clean diagnostic accuracy is a weak proxy
for corruption robustness. The test is **inconclusive**: the clean-vs-mCE Spearman ρ = 0.80
does not meet our ρ < 0.7 rule, but with four models it can neither confirm nor confidently
refute a coupling — and the one mid-rank swap (Rocket↔Hydra) is not significant. What is
clear is more nuanced than a single correlation: robustness is metric-dependent (Rocket wins
on *relative* degradation), and clean accuracy hides real, corruption-specific blind spots —
most notably Hydra's fragility to Gaussian noise and muscle artifact. The practical lesson
stands regardless: for deployment on noisy hardware, clean leaderboards are a useful first
cut but no substitute for direct robustness measurement. The main limitations are the
four-model zoo and single seed, which leave the ρ estimate underpowered; enlarging the zoo
(deep + foundation models) and adding cross-dataset evaluation are the natural next steps.

## 8. References

1. Hendrycks, D., & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (ImageNet-C). ICLR. arXiv:1903.12261.
2. Mintun, E., Kirillov, A., & Xie, S. (2021). *On Interaction Between Augmentations and Corruptions in Natural Corruption Robustness* (ImageNet-C-bar). NeurIPS. arXiv:2102.11273.
3. Dempster, A., Petitjean, F., & Webb, G. I. (2020). *ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels.* Data Mining and Knowledge Discovery. arXiv:1910.13051.
4. Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). *MiniRocket: A very fast (almost) deterministic transform for time series classification.* KDD. arXiv:2012.08791.
5. Dempster, A., Schmidt, D. F., & Webb, G. I. (2023). *Hydra: Competing convolutional kernels for fast and accurate time series classification.* Data Mining and Knowledge Discovery. arXiv:2203.13652.
6. Lubba, C. H., et al. (2019). *catch22: CAnonical Time-series CHaracteristics.* Data Mining and Knowledge Discovery. arXiv:1901.10200.
7. Wagner, P., et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data 7, 154.
8. Moody, G. B., Muldrow, W. E., & Mark, R. G. (1984). *A noise stress test for arrhythmia detectors* (MIT-BIH Noise Stress Test Database). Computers in Cardiology.
9. Goldberger, A. L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 101(23), e215–e220.
10. Middlehurst, M., et al. (2024). *aeon: a Python toolkit for learning from time series.* JMLR (software).
11. Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 7, 1–30.
12. García, S., & Herrera, F. (2008). *An Extension on "Statistical Comparisons of Classifiers over Multiple Data Sets" for all Pairwise Comparisons.* JMLR 9, 2677–2694.

*Full proposal reference list (28 entries) in [ECG-C-proposal.md](ECG-C-proposal.md) §13.
arXiv IDs / DOIs to be verified against live sources before any external submission; entries
[20], [21], [25] there are `Authors (2025)` placeholders whose author lists remain unresolved.*
