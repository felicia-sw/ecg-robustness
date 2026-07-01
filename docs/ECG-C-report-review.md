# Peer Review — *Are ECG Classifiers Robust to Realistic Sensor Corruptions? (ECG-C)*

*Simulated multi-perspective review (academic-paper-reviewer, full mode). Read-only: this
document does not modify the manuscript. Target framing: Datasets & Benchmarks / empirical
robustness-audit report. Reviewed file: `docs/ECG-C-report.md` + `docs/pre-registration.md`.*

---

## Phase 0 — Field analysis & reviewer configuration

- **Primary field:** ML robustness benchmarking for biomedical time series (ECG).
- **Paradigm:** empirical, pre-registered, confirmatory + exploratory.
- **Maturity:** focused report / short-paper stage; honest and reproducible, but small-scale.
- **Panel:** EIC (benchmarks/venue fit), R1 (methodology/statistics), R2 (ECG/signal-processing
  domain), R3 (cross-disciplinary/deployment), Devil's Advocate (core-claim challenge).

---

## Phase 1 — Reviews (independent)

### Editor-in-Chief
**Fit & significance.** The question — does clean ECG accuracy predict corruption robustness —
is well-motivated and deployment-relevant, and transferring the ImageNet-C protocol is
appropriate. The pre-registration (§3.6), the honest reporting of a **null** on the headline
hypothesis, and the reproducible artifact (code + results + tests) are genuine strengths that
many submissions lack. **However**, two problems keep this below acceptance: (1) a
signal-processing validity defect in the powerline corruption (see Devil's Advocate / R2) that
compromises a headline empirical claim, and (2) the central statistic (ρ over **n = 4** models,
p = 0.20) cannot actually support or refute the pre-registered rule. The paper is currently
honest *about* being underpowered but still leans on conclusions the data can't carry.
**Recommendation: Major Revision.**

### Reviewer 1 — Methodology & statistics
1. **(MAJOR) The ρ test is not interpretable at n = 4.** Spearman ρ over four models can take
   only a handful of discrete values; p = 0.20 is non-significant; and the pre-registered rule
   ("ρ < 0.7 *and* CI excludes 0.9", §3.6) is essentially untestable with four points. §4.1
   states this, but the abstract and §7 still conclude clean accuracy is "a decent proxy" —
   that is over-reading a null. State plainly: **the study is inconclusive on RQ1.**
2. **(MAJOR) The bootstrap CI mis-quantifies the relevant uncertainty.** The record-level
   bootstrap (§3.5) resamples *test records* but ρ is a statistic over the *model population*.
   Resampling records with the four models fixed yields the degenerate CI [0.80, 1.00]; it does
   **not** capture the dominant uncertainty, which is model sampling. Either add models or
   reframe the CI's meaning.
3. **(MAJOR) mCE is normalized to the best model (MiniRocket).** ImageNet-C normalizes to a
   fixed *weak* baseline; using the top performer forces every other model to mCE > 1 and
   couples the metric to the strongest model's idiosyncrasies. Justify or switch to a fixed
   reference and report sensitivity.
4. **(MAJOR) Single training seed.** Each model is trained once (`random_state=0`). A robustness
   *ranking* claim needs seed variability — the Rocket↔Hydra swap that drives the whole
   narrative may be within seed noise. Report ≥5 seeds or bound the ranking's stability.
5. **(MINOR, undisclosed) Severity confounds SNR with noise realization.** Real-noise severities
   draw *different* random windows per level, so degradation mixes SNR with window variance.
   Disclose, and ideally fix the window across severities.

*Rigor: 5/10 · Reproducibility: 9/10.*

### Reviewer 2 — ECG / signal-processing domain
1. **(CRITICAL) Powerline at 100 Hz is physically ill-posed.** Nyquist at fs = 100 Hz is 50 Hz;
   a 50 Hz mains tone lands exactly at Nyquist (samples to ≈0) and the 100/150 Hz harmonics are
   above Nyquist. The corruption cannot represent powerline interference at this sampling rate.
   Empenically the injected tone has max amplitude ≈ 9e-13 (see DA). Real 50/60 Hz powerline
   robustness requires the 500 Hz PTB-XL records (or a sub-Nyquist surrogate honestly labeled).
2. **(MAJOR) "Gain-invariance" is an implementation artifact, not a model property.** §5 attributes
   ROCKET's gain-robustness to `Rocket(normalise=True)`. That is an aeon default, not an intrinsic
   ROCKET property, and MiniRocket (same family) is *not* invariant. The claim "the ROCKET family
   is gain-invariant" (abstract) overgeneralizes from a config flag.
3. **(MINOR) No standalone reference list.** §8 points to the proposal; a self-contained report
   should carry its own references (NSTDB, PTB-XL, ImageNet-C, ROCKET, catch22, Demšar).
4. *Strength:* the real-NSTDB anchoring, the exact SNR calibration (verified 0.00 dB), and the
   multi-label macro-AUROC handling are done correctly and are a real contribution.

*Domain rigor: 5/10 · Contribution: 6/10.*

### Reviewer 3 — Cross-disciplinary / deployment
- The vision→ECG transfer of the common-corruption paradigm is apt and the "clean leaderboards
  ≠ deployment robustness" message is valuable to practitioners.
- **Absolute vs relative mCE** (§5) is a genuinely useful distinction and well handled.
- **(MAJOR)** The deployment takeaway rests on the powerline and Hydra-fragility results; if
  powerline is invalid (R2/DA), the most quotable practitioner claim evaporates. Re-derive the
  deployment message from the corruptions that *are* valid (bw, ma, gaussian, quantization).
- **(MINOR)** "catch22 loses discriminative content as the waveform degrades" is plausible but
  asserted, not shown; a short feature-drift analysis would support it.

*Impact: 6/10 · Clarity: 8/10.*

### Devil's Advocate
**Strongest counter-argument.** Two of the paper's three memorable empirical claims are unsafe.
(a) **Powerline is a mislabeled corruption.** I verified: at fs = 100 Hz the powerline tone is
≈0 (max |amp| = 8.6e-13); the SNR scaler then divides by ~0 power and amplifies floating-point
residuals ~10¹²×, and the FFT of the injected "powerline" noise peaks at **38 Hz, not 50 Hz**.
So "Hydra is disproportionately fragile to powerline (CE 2.33×)" (Table 2, §5, abstract) is a
statement about amplified numerical noise, not powerline interference. This is CRITICAL: a
core finding is built on an artifact. (b) **The RQ1 conclusion is a coin-flip dressed as a
result.** p = 0.20 on n = 4 means "no evidence either way," yet the abstract frames clean
accuracy as "a useful but incomplete proxy" — a directional read the statistics don't license.

**Issue list.** CRITICAL: powerline validity (§3.3, Table 2, §5, abstract). MAJOR: over-reading
the n = 4 null; single-seed ranking; mCE reference choice. MINOR: gain-invariance overgeneralization;
missing reference list; severity/window confound undisclosed.

**"So what?" test.** With powerline removed and RQ1 stated as inconclusive, the paper still has a
real, honest contribution (a verified real-noise ECG corruption harness + a candid null + the
absolute/relative-robustness distinction) — but the current headline framing oversells it.

---

## Phase 2 — Editorial decision

**Decision: MAJOR REVISION.** (A Devil's Advocate CRITICAL finding precludes Accept.)

**Consensus (all reviewers):** pre-registration, honest null reporting, exact SNR calibration,
real-NSTDB anchoring, and reproducibility (code + tests + results) are real strengths.

**Consensus problems:** (1) powerline corruption is invalid at 100 Hz [CRITICAL]; (2) the RQ1
conclusion over-reads an underpowered n = 4 null [MAJOR, EIC+R1+DA]; (3) robustness ranking not
shown stable to model seed [MAJOR, R1].

### Revision roadmap (prioritized)
1. **Fix or drop powerline.** Either evaluate powerline on 500 Hz PTB-XL, or remove it from the
   100 Hz suite and note the Nyquist limitation; re-run mCE/Table 2 and revise every
   powerline-dependent sentence (abstract, §4, §5). *(blocks acceptance)*
2. **Restate RQ1 honestly as inconclusive** in the abstract and §7 (p = 0.20, n = 4); do not
   imply a direction. Consider enlarging the zoo (the CPU-cheap MultiRocket/QUANT, or the
   deferred deep models) to make ρ testable.
3. **Add training-seed variability** (≥5 seeds) and report whether the Rocket↔Hydra swap survives.
4. **Justify the mCE reference** (or switch to a fixed weak baseline) and show sensitivity.
5. **Soften "gain-invariance"** to "aeon's normalise flag confers gain-robustness on Rocket/Hydra";
   disclose the severity/window confound; add a self-contained reference list.

**Provisional scores (0–10):** rigor 5 · novelty 6 · clarity 8 · reproducibility 9 · overall 5.5.
The honesty and engineering are strong; the empirical claims need the powerline fix and an
honest RQ1 restatement before they can be trusted.
