# ECG-C — Robustness of ECG Classifiers to Realistic Sensor Corruptions

A physically-grounded **common-corruption robustness benchmark and audit** for ECG classification.
Central question: *does clean diagnostic accuracy predict robustness to ordinary sensor faults (baseline wander, muscle/electrode artifact, powerline interference, gain miscalibration, dropout, quantization)?* Hypothesis: **it does not** — model rankings re-order under realistic noise.

> Status: research-stage. Current goal is a focused **written report** (PTB-XL only, 5 models, 6 corruptions). See [REPORT-PLAN.md](REPORT-PLAN.md).

## Documents
- **[REPORT-PLAN.md](REPORT-PLAN.md)** — the 14-day execution plan, scope, and decisions. *Read this first.*
- **[docs/ECG-C-proposal.md](docs/ECG-C-proposal.md)** — full proposal (v3).
- [docs/ts-research-ideas.md](docs/ts-research-ideas.md) — provenance (this is idea #14 from the idea bank).
- [docs/archive/](docs/archive/) — superseded drafts.

## Scope (report version)
- **Dataset:** PTB-XL @ 100 Hz, 5 diagnostic superclasses (multi-label).
- **Models (5 primary):** MiniRocket · catch22 + ridge · InceptionTime · 1D SE-ResNet · Mantis features + linear probe.
- **Corruptions (6 × 5 severities):** NSTDB baseline wander, NSTDB muscle artifact (real recorded noise), powerline interference, Gaussian noise, gain miscalibration, quantization.
- **Metric:** macro-AUROC; mean Corruption Error (mCE) + relative mCE.
- **Analysis:** clean-vs-mCE Spearman ρ (decision rule: ρ < 0.7), Friedman + Wilcoxon-Holm + critical-difference diagram, bootstrap CIs.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data (download separately — not committed)
Staged; for the report you only need ~2 GB:
- **PTB-XL @ 100 Hz** — https://physionet.org/content/ptb-xl/
- **MIT-BIH Noise Stress Test Database (NSTDB)** — https://physionet.org/content/nstdb/

Place under `data/` (git-ignored). Use `wfdb` to read records.

## Repository structure
```
.
├── REPORT-PLAN.md       # execution plan (start here)
├── docs/                # proposal, idea provenance, archive
├── src/                 # your code: corruptions, models, eval, stats
├── notebooks/           # exploration / figures
├── data/                # downloaded datasets (git-ignored)
└── results/             # result tables / figures (git-ignored)
```

## License / status
Personal research project. No license set yet.
