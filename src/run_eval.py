"""ECG-C end-to-end evaluation runner (the orchestration script).

This is the missing link between the model zoo / corruption suite and the analysis
layer: it trains each model on CLEAN data, evaluates it on the clean test fold and on
every (corruption x severity) condition, and writes exactly the artifacts that
``src/analysis.py`` and the notebooks consume:

    results/grid.csv                      # model, corruption, severity, macro_auroc
    results/preds/<model>__<corr>__s<S>.npy   # per-condition probabilities (n_test, 5)
    results/preds/<model>__clean__s0.npy
    results/preds/y_true.npy              # (n_test, 5) multi-hot ground truth

Run (single seed, reproduces the committed results):

    python -m src.run_eval --seeds 0

Run several seeds to quantify ranking stability to training randomness
(writes results/seed<S>/ per seed and a combined results/grid_multiseed.csv):

    python -m src.run_eval --seeds 0 1 2 3 4

Then regenerate all tables/figures:

    python -m src.analysis

Determinism: corruption noise is fixed (corruption seed = 0 inside src/corruptions.py),
so across training seeds ONLY the model's random kernels change -- which is exactly the
variance a robustness-ranking claim needs to be shown stable against. Use --dry-run to
print the plan (and the exact output paths) without loading data or training.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.corruptions import CORRUPTIONS, SEVERITIES, apply_corruption
from src.evaluate import macro_auroc

# Mirrors src.models.PRIMARY_MODELS; kept here so --dry-run needs no heavy imports (aeon/torch).
_DEFAULT_MODELS = ["minirocket", "rocket", "catch22_ridge", "hydra"]


def _conditions() -> list[tuple[str, int]]:
    """The full evaluation grid: clean once, then every (corruption, severity)."""
    grid = [("clean", 0)]
    for c in CORRUPTIONS:
        for s in SEVERITIES:
            grid.append((c, s))
    return grid


def _load_split(data_dir: str, limit: int | None):
    """Load PTB-XL (cached data/*_100.npy if present, else from raw) + fold masks."""
    from src.data import load_ptbxl, split_by_fold

    xcache, ycache = os.path.join(data_dir, "X_100.npy"), os.path.join(data_dir, "y_100.npy")
    meta = pd.read_csv(os.path.join(data_dir, "ptb-xl", "ptbxl_database.csv"), index_col="ecg_id")
    if os.path.exists(xcache) and os.path.exists(ycache):
        X, y = np.load(xcache), np.load(ycache)
    else:  # fall back to the raw waveform loader (also builds meta consistently)
        X, y, meta = load_ptbxl(os.path.join(data_dir, "ptb-xl"))
    train, _val, test = split_by_fold(meta)
    Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
    if limit:  # smoke test: subsample train and test
        Xtr, ytr, Xte, yte = Xtr[:limit], ytr[:limit], Xte[:limit], yte[:limit]
    return Xtr.astype("float32"), ytr, Xte.astype("float32"), yte


def _run_one(name: str, seed: int, n_kernels: int, Xtr, ytr, Xte, yte, preds_dir: str):
    """Train one model on clean data; predict clean + every condition; save preds; return rows."""
    from src.models import build_model, fit, predict_proba

    model = fit(build_model(name, n_kernels=n_kernels, random_state=seed), Xtr, ytr)
    rows = []
    for corr, sev in _conditions():
        Xc = Xte if corr == "clean" else apply_corruption(Xte, corr, sev)
        proba = predict_proba(model, Xc).astype("float32")
        np.save(os.path.join(preds_dir, f"{name}__{corr}__s{sev}.npy"), proba)
        rows.append(dict(model=name, corruption=corr, severity=sev,
                         macro_auroc=macro_auroc(yte, proba)))
    return rows


def run(models, seeds, data_dir, results_dir, n_kernels, limit):
    conds = _conditions()
    print(f"[run_eval] {len(models)} models x {len(seeds)} seeds x {len(conds)} conditions "
          f"= {len(models) * len(seeds) * len(conds)} evaluations")
    Xtr, ytr, Xte, yte = _load_split(data_dir, limit)
    print(f"[run_eval] train={Xtr.shape} test={Xte.shape}")

    per_seed = []
    for seed in seeds:
        out_dir = results_dir if len(seeds) == 1 else os.path.join(results_dir, f"seed{seed}")
        preds_dir = os.path.join(out_dir, "preds")
        os.makedirs(preds_dir, exist_ok=True)
        np.save(os.path.join(preds_dir, "y_true.npy"), yte)
        rows = []
        for name in models:
            print(f"[run_eval] seed={seed} model={name} ...", flush=True)
            rows += _run_one(name, seed, n_kernels, Xtr, ytr, Xte, yte, preds_dir)
        grid = pd.DataFrame(rows)
        grid.to_csv(os.path.join(out_dir, "grid.csv"), index=False)
        grid["seed"] = seed
        per_seed.append(grid)
        print(f"[run_eval] wrote {out_dir}/grid.csv and {len(rows)} prediction arrays")

    if len(seeds) > 1:
        combined = pd.concat(per_seed, ignore_index=True)
        combined.to_csv(os.path.join(results_dir, "grid_multiseed.csv"), index=False)
        print(f"[run_eval] wrote {results_dir}/grid_multiseed.csv ({len(seeds)} seeds) "
              f"-> summarize with src.analysis.multiseed_leaderboard")


def main():
    ap = argparse.ArgumentParser(description="ECG-C end-to-end evaluation runner.")
    ap.add_argument("--models", nargs="+", default=None,
                    help="models to run; CPU zoo {minirocket,rocket,hydra,catch22_ridge} and/or "
                         "GPU extended {inceptiontime,resnet,mantis} (default: CPU zoo). "
                         "'mantis' needs an encoder wired via MantisProbe(embed_fn=...).")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="training seeds; >1 writes results/seed<S>/ + grid_multiseed.csv")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n-kernels", type=int, default=10_000, help="MiniRocket kernels")
    ap.add_argument("--limit", type=int, default=None, help="subsample train/test (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and output paths without loading data or training")
    args = ap.parse_args()
    models = args.models or _DEFAULT_MODELS

    if args.dry_run:
        conds = _conditions()
        n_preds = len(models) * len(conds) + 1  # + y_true.npy, per seed
        print(f"PLAN: models={models} seeds={args.seeds}")
        print(f"conditions/model = {len(conds)} (clean + {len(CORRUPTIONS)} corruptions x {len(SEVERITIES)} severities)")
        for seed in args.seeds:
            out = args.results_dir if len(args.seeds) == 1 else f"{args.results_dir}/seed{seed}"
            print(f"  seed {seed}: -> {out}/grid.csv, {out}/preds/*.npy ({n_preds} arrays incl. y_true)")
        return

    run(models, args.seeds, args.data_dir, args.results_dir, args.n_kernels, args.limit)


if __name__ == "__main__":
    main()
