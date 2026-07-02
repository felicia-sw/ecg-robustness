# ECG-C — Robustness of ECG Classifiers to Realistic Sensor Corruptions

A physically-grounded **common-corruption robustness benchmark and audit** for ECG classification.
Central question: *does clean diagnostic accuracy predict robustness to ordinary sensor faults (baseline wander, muscle/electrode artifact, powerline interference, gain miscalibration, dropout, quantization)?* Hypothesis: **it does not** — model rankings re-order under realistic noise.

> Status: research-stage. Current goal is a focused **written report** (PTB-XL only, 4 CPU models, 5 corruptions). See [REPORT-PLAN.md](REPORT-PLAN.md).

## Documents
- **[REPORT-PLAN.md](REPORT-PLAN.md)** — the 14-day execution plan, scope, and decisions. *Read this first.*
- **[docs/ECG-C-proposal.md](docs/ECG-C-proposal.md)** — full proposal (v3).
- [docs/ts-research-ideas.md](docs/ts-research-ideas.md) — provenance (this is idea #14 from the idea bank).
- [docs/archive/](docs/archive/) — superseded drafts.

## Scope (report version, as run)
- **Dataset:** PTB-XL @ 100 Hz, 5 diagnostic superclasses (multi-label).
- **Models (4, CPU):** MiniRocket · Rocket · Hydra (ROCKET family) · catch22 + ridge. Deep/foundation models (InceptionTime, SE-ResNet, Mantis) are GPU-bound future work.
- **Corruptions (5 × 5 severities):** NSTDB baseline wander, NSTDB muscle artifact (real recorded noise), Gaussian noise, gain miscalibration, quantization. Powerline is excluded at 100 Hz (50 Hz = Nyquist) and deferred to the 500 Hz records.
- **Metric:** macro-AUROC; mean Corruption Error (mCE) + relative mCE.
- **Analysis:** clean-vs-mCE Spearman ρ (reported with the EXACT permutation test — the n=4 zoo is underpowered and the ρ<0.7 rule is unreachable when the extremes are rank-stable), Friedman (all-conditions **and** family-blocked) + Wilcoxon-Holm + critical-difference diagram, record-level bootstrap, and mCE reference-sensitivity.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
make setup           # editable install + requirements
make freeze          # LOCK exact versions -> requirements.lock.txt (commit this)
```

## Reproduce the results
With PTB-XL@100Hz + NSTDB under `data/` (see below):
```bash
make smoke           # fast wiring check (dry-run + 200-record run)
make eval            # single seed -> results/grid.csv + results/preds/ (reproduces the report)
make analysis        # -> results/leaderboard.csv, results/summary.md, results/fig_*.png
make eval-seeds      # 5 seeds -> results/grid_multiseed.csv (ranking stability)
make test            # test suite (18 plumbing + inference-layer stats tests)
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
├── src/                 # corruptions, models, run_eval (orchestration), evaluate, stats, analysis
├── notebooks/           # exploration / figures
├── data/                # downloaded datasets (git-ignored)
└── results/             # result tables / figures (git-ignored)
```

## License / status
Personal research project. Released under the MIT License (see [LICENSE](LICENSE)).
