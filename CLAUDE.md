# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is
**ECG-C** — a physically-grounded common-corruption robustness benchmark/audit for ECG classification. Research-stage project; the immediate goal is a focused **written report**, not the full benchmark.

**Read first:** [REPORT-PLAN.md](REPORT-PLAN.md) (plan, scope, decisions) then [docs/ECG-C-proposal.md](docs/ECG-C-proposal.md) (full method). This was idea #14 in `docs/ts-research-ideas.md`.

## Central hypothesis (falsifiable)
Clean diagnostic accuracy is a weak predictor of corruption robustness: clean-vs-mCE Spearman **ρ < 0.7** across the model zoo → rankings re-order under realistic noise.

## Scope (report version — keep it tight)
- Dataset: **PTB-XL @ 100 Hz**, 5 diagnostic superclasses (multi-label).
- Models (5 primary): **MiniRocket, catch22+ridge, InceptionTime, 1D SE-ResNet, Mantis+linear-probe**. Extended set is optional.
- Corruptions: **6 × 5 severities** — NSTDB baseline wander, NSTDB muscle artifact (real recorded), powerline, Gaussian, gain, quantization.
- Metric: macro-AUROC; **mCE** + relative mCE. Stats: Spearman ρ, Friedman + Wilcoxon-Holm, critical-difference diagram, bootstrap CIs.
- Out of scope (→ Limitations/Future work): cross-dataset (Chapman/CPSC/CinC), full ~10-corruption suite, mitigations, artifact release.

## Working style (important)
- **The user learns by doing.** Give clear, runnable *directions and small unblocking snippets* — do **not** write the whole codebase for them. Implementing corruption generators / models / the eval loop is their work.
- OK to help fast on specific stuck points: the PTB-XL fold loader, the mCE formula, the CD-diagram code.
- **Fast models first:** lock MiniRocket → catch22 → Mantis-probe (CPU, fast) across all corruptions so there's always a submittable result; treat InceptionTime + SE-ResNet (GPU) as upgrades.

## Environment & commands
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
- GPU needed only for InceptionTime / SE-ResNet; the rest run on CPU.
- No test runner configured yet.

## Conventions
- Code in `src/`, exploration in `notebooks/`, outputs in `results/` (git-ignored).
- **Never commit data** — `data/` is git-ignored; download PTB-XL@100Hz + NSTDB from PhysioNet (see README), ~2 GB total.
- **Pre-register before the full run:** write the decision rule (ρ < 0.7) before evaluating the full zoo; don't train on the benchmark corruptions.
- Keep a results manifest (model × corruption × severity × seed) early to avoid CSV chaos.

## Deadline
Written report due ~2 weeks from late June 2026 (target ≈ 2026-07-12 — confirm exact date).

## Related
This repo was split out of the `trustworthy-shift` folder (a separate paper). ECG-C is independent of that project.
