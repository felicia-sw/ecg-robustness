# ECG-C — Pre-registration (report version)

*Written 2026-07-01, **before** evaluating the full model zoo under corruptions.*
*Purpose: fix the hypothesis, design, metric, and decision rule in advance so the
confirmatory result cannot be a post-hoc story.*

## Confirmatory hypothesis (RQ1)
Clean diagnostic accuracy is a **weak** predictor of corruption robustness.

**Decision rule:** across the model zoo, the Spearman rank correlation between the
**clean** macro-AUROC leaderboard and the **mCE** leaderboard is

> **ρ < 0.7**, with a bootstrap 95% CI that **excludes 0.9**.

If met → "clean accuracy is a weak proxy for robustness" is supported (rankings
re-order under realistic noise). If ρ ≥ 0.7 we report the null honestly.

## Fixed design (no changes after seeing results)

**Dataset.** PTB-XL @ 100 Hz, 5 diagnostic superclasses (multi-label). Published
`strat_fold`: train = folds 1–8, val = 9, **test = 10**. Models see clean data only.

**Models (report zoo, CPU).** Four models across two paradigms, each a
`transform → StandardScaler → ridge probe`: **MiniRocket · Rocket · Hydra**
(ROCKET family) and **catch22+ridge** (feature-based). Deep models (InceptionTime,
1D SE-ResNet) and the Mantis foundation-feature probe are upgrades / future work
(GPU-bound); noted as a zoo-size limitation for the ρ estimate.

**Corruptions (6 × 5 severities).** Applied to the clean **test** set only:
baseline wander (NSTDB bw, real) · muscle artifact (NSTDB ma, real) · powerline ·
Gaussian · gain miscalibration · quantization.

**Severity calibration.** Additive-noise corruptions use a fixed SNR ladder
(severity 1→5 = **18, 12, 6, 0, −6 dB**). Gain = multiplicative factor ladder;
quantization = effective-bits ladder. Ladders fixed in `src/corruptions.py`.

## Metrics
- **Primary:** macro-AUROC (one-vs-rest over the 5 superclasses).
- **Robustness:** mean Corruption Error (ImageNet-C analogue), with
  `error = 1 − macro-AUROC`, normalized to a fixed **reference model = MiniRocket**:
  `CE(f,c) = Σ_s error(f,c,s) / Σ_s error(ref,c,s)`, `mCE(f) = mean_c CE(f,c)`.
- **Relative mCE:** degradation above each model's own clean error.
- Clean macro-AUROC always reported alongside mCE.

## Statistical analysis
- Spearman ρ (clean vs mCE) with bootstrap CI over records/seeds — the RQ1 test.
- Friedman test across (corruption × severity) conditions → post-hoc pairwise
  Wilcoxon signed-rank with Holm–Bonferroni → critical-difference diagram over mCE ranks.
- Paired bootstrap / Wilcoxon for each model's clean→corrupted degradation, with effect sizes.

## No-leakage rules
Models never trained on benchmark corruptions; published folds used as-is; fixed
inference settings; fixed seeds; bootstrap CIs over records/seeds; nulls reported honestly.

## Out of scope (→ Limitations / Future work)
Cross-dataset external validity (Chapman/CPSC/CinC-2020); the full ~10-corruption
suite (electrode motion, dropout, sampling-rate mismatch, ...); mitigations
(augmentation / TTA) and the held-out-corruption augmentation protocol; artifact release.

## Deviations from pre-registration (disclosed)
- **Powerline dropped (6 → 5 corruptions).** At fs = 100 Hz, 50 Hz mains is at the Nyquist
  frequency, so the powerline tone samples to ~0 and the SNR scaler amplifies numerical
  noise rather than injecting a real tone. Discovered post-hoc during review and removed;
  deferred to the 500 Hz records. Verified no result depends on it — ρ and the mCE ranking
  are identical with and without powerline. Confirmatory RQ1 conclusion unchanged.
