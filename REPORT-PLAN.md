# ECG-C — Report Plan & Session Handoff

*Last updated: 2026-06-28. This file is the continuity note for picking the project up in a new session.*

## What this is
A **focused robustness study / report** built from idea #14 in [ts-research-ideas.md](ts-research-ideas.md):
**ECG-C — a physically-grounded common-corruption robustness benchmark for ECG classification.**
Full proposal: [ECG-C-proposal.md](ECG-C-proposal.md) (v3) — also a Google Doc (Times New Roman, professional formatting):
**https://docs.google.com/document/d/1AISvpLOHtGpewHjJEC3eUCc9BgQWUIwqEbnkDpxRT0c/edit**

Folder context: this is the `trustworthy-shift` paper project. The original format template was a tabular-foundation-model trustworthiness proposal; ECG-C is the current paper being developed here.

## Deadline & deliverable
- **Deliverable:** a written report / paper.
- **Deadline:** ~2 weeks from 2026-06-28 → **target ≈ 2026-07-12** (confirm exact date).
- Verdict: 2 weeks = a **thin-but-complete focused report**, NOT the full benchmark. Frame it as a focused study.

## Key decisions already made (with the professor)
1. **Domain scoped to ECG** (was multi-modal) — leverages real recorded noise from MIT-BIH NSTDB; all data open-access.
2. **Model zoo trimmed** to **5 primary** (one per paradigm): MiniRocket, catch22+ridge, InceptionTime, 1D SE-ResNet (PhysioNet/CinC-2020-style), Mantis features + linear probe. Extended set (other ROCKET variants, FCN, Ribeiro DNN) is **optional**.
3. **Statistical significance analysis added**: Friedman → Wilcoxon signed-rank + Holm → critical-difference diagram; paired tests + effect sizes; Bayesian signed-rank as extra. (Professor called it supplementary; we keep the basic test in the main results because the thesis IS a ranking claim.)

## Scope for the 2-week report
**In:** PTB-XL@100Hz only (5 diagnostic superclasses) · all 5 primary models · **6 corruptions × 5 severities** (NSTDB baseline wander + muscle artifact [real], powerline, Gaussian, gain, quantization) · clean leaderboard, mCE + relative mCE, Spearman ρ (clean vs mCE), Friedman + Wilcoxon-Holm + CD diagram, bootstrap CIs.
**Out → "Limitations / Future work":** Chapman/CPSC/CinC cross-dataset, full ~10-corruption suite, mitigations (augmentation/TTA), artifact release.

## Falsifiable hypothesis (pre-register before full results)
Clean-vs-mCE Spearman **ρ < 0.7** across the model zoo (bootstrap CI excluding 0.9) → "clean accuracy is a weak proxy for robustness."

## 14-day plan
| Days | Block | Milestone |
|------|-------|-----------|
| 1–2 | Setup + data + 1 model | Env; download PTB-XL@100Hz + NSTDB (~2 GB); multi-label loader w/ published folds; MiniRocket → clean macro-AUROC working |
| 3–4 | Corruptions + pre-reg | 6 corruptions × 5 severities (real NSTDB noise at SNRs); confirm accuracy drops with severity; write the ρ<0.7 decision rule |
| 5–7 | Full zoo + eval | Train/extract all 5 models on clean → clean leaderboard; run 5×6×5 → mCE table (GPU stretch for SE-ResNet/InceptionTime) |
| 8 | Stats + figures | Spearman ρ, Friedman + Wilcoxon-Holm, CD diagram, per-corruption heatmap, bootstrap CIs |
| 9–12 | Write report | Reuse proposal for Intro/Method; add Results, Analysis, Discussion, Limitations/Future work |
| 13–14 | Buffer + polish | Proofread, re-run breakages, format refs. Protect this buffer. |

## Safety strategy
Lock **fast models first** (MiniRocket → catch22 → Mantis-probe; all CPU, minutes) across all corruptions → always have a submittable 3-model report. Add InceptionTime + SE-ResNet (GPU) as upgrades, not dependencies.

## Top risks
1. Multi-label PTB-XL plumbing (budget a full day).
2. GPU/training wall-clock (fast-first strategy is the insurance).
3. Underestimating writing time (proposal reuse mitigates).

## Data (staged — do NOT download everything)
- Now: **PTB-XL@100Hz + NSTDB ≈ ~2 GB**. Tools: `wfdb`, `neurokit2`. Libs: `aeon`, `sktime`, `tsai`.
- Other datasets only if cross-dataset gets back in scope (not for the 2-week report).

## Start here (this week)
1. Env + download PTB-XL@100Hz + NSTDB.
2. MiniRocket classifying clean PTB-XL end-to-end with macro-AUROC.
3. Add NSTDB baseline wander at 5 SNRs; confirm accuracy drops with severity.

## Working preference
User **learns by doing** — give directions to run, don't write the whole codebase. OK to unblock specific stuck points fast (PTB-XL fold loader, mCE formula, CD-diagram code).

## Doc-editing note (for the assistant)
The Google Doc is generated from the markdown via scripts in the session scratchpad (`make_doc.py` / `update_doc.py`): parse markdown → create/clear doc → insert text → apply Times New Roman + heading sizes/alignment via the Docs API (`gws docs documents batchUpdate`). Doc ID: `1AISvpLOHtGpewHjJEC3eUCc9BgQWUIwqEbnkDpxRT0c`. To re-sync after editing the markdown, re-run the clear-and-rebuild flow.
