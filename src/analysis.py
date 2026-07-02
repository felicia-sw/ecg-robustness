"""Analysis: leaderboard, mCE, the RQ1 rho test (record-level bootstrap),
significance tests, and figures.

Reads ``results/grid.csv`` + ``results/preds/``; writes ``results/leaderboard.csv``,
``results/summary.md`` and ``results/fig_*.png``. All logic lives here (not in a
notebook or scratchpad) so it is importable and testable; the notebooks only present.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from itertools import combinations

RESULTS_DIR = "results"
SEVERITIES = [1, 2, 3, 4, 5]
MODEL_ORDER = ["minirocket", "rocket", "catch22_ridge", "hydra"]


# ---------------------------------------------------------------- core metrics
def clean_leaderboard(grid: pd.DataFrame) -> dict:
    return grid[grid.corruption == "clean"].set_index("model").macro_auroc.to_dict()


def error_sums(grid: pd.DataFrame) -> pd.DataFrame:
    """Sum of (1 - AUROC) over severities, per (model, corruption)."""
    corr = grid[grid.corruption != "clean"].copy()
    corr["error"] = 1.0 - corr["macro_auroc"]
    return corr.groupby(["model", "corruption"])["error"].sum().unstack("corruption")


def mean_corruption_error(grid: pd.DataFrame, reference_model: str) -> pd.Series:
    """mCE per model, normalized to a reference (mCE[reference] == 1.0)."""
    es = error_sums(grid)
    ce = es.divide(es.loc[reference_model], axis=1)
    return ce.mean(axis=1).rename("mCE")


def relative_mce(grid: pd.DataFrame, reference_model: str, eps: float = 1e-6) -> pd.Series:
    """Relative mCE (degradation above each model's own clean error).

    Guards the ImageNet-C relative normalization: corruptions where the reference's
    own relative degradation is <= eps are DROPPED (dividing by ~0 is meaningless /
    explosive), and a note is left in the returned Series' attrs.
    """
    clean = clean_leaderboard(grid)
    es = error_sums(grid)
    clean_err = pd.Series({m: 1.0 - clean[m] for m in es.index})
    rel = es.subtract(clean_err * len(SEVERITIES), axis=0)          # (corrupted - clean) errors
    ref_rel = rel.loc[reference_model]
    keep = ref_rel[ref_rel > eps].index                            # drop near-zero reference cols
    dropped = [c for c in ref_rel.index if c not in keep]
    rce = rel[keep].divide(ref_rel[keep], axis=1).mean(axis=1).rename("relative_mCE")
    rce.attrs["dropped_corruptions"] = dropped
    return rce


# ------------------------------------------------------------- RQ1: rho + CI
def _safe_macro(y_true: np.ndarray, y_score: np.ndarray) -> float:
    aucs = [roc_auc_score(y_true[:, k], y_score[:, k])
            for k in range(y_true.shape[1]) if y_true[:, k].min() != y_true[:, k].max()]
    return float(np.mean(aucs)) if aucs else float("nan")


def rho_clean_vs_robustness(clean: dict, mce: dict, models: list) -> tuple[float, float]:
    """Spearman between clean AUROC and robustness (-mCE). rho<0.7 => weak proxy."""
    rho, p = spearmanr([clean[m] for m in models], [-mce[m] for m in models])
    return float(rho), float(p)


def bootstrap_rho(preds_dir: str, models: list, corrs: list, reference_model: str,
                  n_boot: int = 1000, seed: int = 0) -> tuple[float, float, np.ndarray]:
    """Record-level bootstrap CI for the clean-vs-robustness Spearman rho."""
    yt = np.load(f"{preds_dir}/y_true.npy")
    preds = {os.path.basename(f)[:-4]: np.load(f)
             for f in glob.glob(f"{preds_dir}/*__*.npy")}
    rng = np.random.default_rng(seed)
    n = yt.shape[0]
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        clean_b = {m: _safe_macro(yt[idx], preds[f"{m}__clean__s0"][idx]) for m in models}
        ref_sum = {c: sum(1 - _safe_macro(yt[idx], preds[f"{reference_model}__{c}__s{s}"][idx])
                          for s in SEVERITIES) for c in corrs}
        mce_b = {}
        for m in models:
            ces = [sum(1 - _safe_macro(yt[idx], preds[f"{m}__{c}__s{s}"][idx])
                       for s in SEVERITIES) / ref_sum[c] for c in corrs]
            mce_b[m] = float(np.mean(ces))
        r, _ = spearmanr([clean_b[m] for m in models], [-mce_b[m] for m in models])
        if not np.isnan(r):
            rhos.append(r)
    rhos = np.asarray(rhos)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(lo), float(hi), rhos


# ------------------------------------------------------------- significance
def significance(grid: pd.DataFrame, models: list):
    corr = grid[grid.corruption != "clean"]
    cond = {m: corr[corr.model == m].sort_values(["corruption", "severity"]).macro_auroc.values
            for m in models}
    fstat, fp = friedmanchisquare(*[cond[m] for m in models])
    pairs, pvals = [], []
    for a, b in combinations(models, 2):
        _, pv = wilcoxon(cond[a], cond[b]); pairs.append((a, b)); pvals.append(pv)
    rej, padj, _, _ = multipletests(pvals, method="holm")
    wh = [dict(pair=pr, p=float(pv), p_holm=float(ph), reject=bool(r))
          for pr, pv, ph, r in zip(pairs, pvals, padj, rej)]
    return (float(fstat), float(fp)), wh


def significance_family_blocked(grid: pd.DataFrame, models: list):
    """Friedman over CORRUPTION FAMILIES (severities averaged) -> near-independent blocks.

    The default ``significance`` test treats all 25 (corruption x severity) cells as
    independent blocks, but the five severities within a family are strongly dependent
    (a monotone ladder on the same records), which inflates the effective sample size
    and the significance. Averaging each family to a single block (5 near-independent
    blocks) is the conservative complement; report both. Returns ((stat, p), n_blocks).
    """
    corr = grid[grid.corruption != "clean"]
    fam = (corr.groupby(["model", "corruption"]).macro_auroc.mean().unstack("corruption")).loc[models]
    fstat, fp = friedmanchisquare(*[fam.loc[m].values for m in models])
    return (float(fstat), float(fp)), int(fam.shape[1])


def mce_reference_sensitivity(grid: pd.DataFrame, models: list | None = None):
    """mCE under EVERY possible reference model, and whether the induced ranking is invariant.

    mCE(f) = mean_c error_sum(f,c)/error_sum(ref,c) is a *reweighted* mean whose weights
    1/error_sum(ref,c) change with the reference, so ranking-invariance is NOT structural --
    it is an empirical property to be checked, not assumed. Returns
    (DataFrame[ref x model] of mCE, {ref: ranked-model tuple}, invariant: bool).
    """
    es = error_sums(grid)
    refs = list(es.index) if models is None else [m for m in models if m in es.index]
    table = {ref: es.divide(es.loc[ref], axis=1).mean(axis=1) for ref in refs}
    df = pd.DataFrame(table).T
    if models is not None:
        df = df[models]
    rankings = {ref: tuple(df.loc[ref].sort_values().index) for ref in refs}
    invariant = len(set(rankings.values())) == 1
    return df, rankings, invariant


# ------------------------------------------------------------- multi-seed stability
def multiseed_leaderboard(multi_grid: pd.DataFrame, reference_model: str = "minirocket"):
    """Aggregate a multi-seed grid (``grid_multiseed.csv`` with a ``seed`` column).

    Addresses the single-seed limitation: with random-kernel models (Rocket/Hydra) the
    ranking can move with the training seed, so a ranking claim must be shown stable.
    Returns (per-model DataFrame with clean/mCE mean+sd across seeds, and the mCE
    rank each model most often holds; and a dict {mCE-ranking-tuple: n_seeds} showing
    how often each full ordering occurs -- e.g. whether the Rocket<->Hydra swap is stable).
    """
    rows, ranking_counts = [], {}
    for seed, g in multi_grid.groupby("seed"):
        clean = clean_leaderboard(g)
        mce = mean_corruption_error(g, reference_model)
        order = tuple(mce.sort_values().index)                 # best (lowest mCE) first
        ranking_counts[order] = ranking_counts.get(order, 0) + 1
        rank_pos = {m: i + 1 for i, m in enumerate(order)}
        for m in mce.index:
            rows.append(dict(seed=seed, model=m, clean=clean[m],
                             mce=float(mce[m]), mce_rank=rank_pos[m]))
    df = pd.DataFrame(rows)
    agg = df.groupby("model").agg(
        clean_mean=("clean", "mean"), clean_sd=("clean", "std"),
        mCE_mean=("mce", "mean"), mCE_sd=("mce", "std"),
        mCE_rank_median=("mce_rank", "median"),
    ).sort_values("mCE_mean")
    return agg, ranking_counts


# ------------------------------------------------------------- figures + report
def make_figures(grid: pd.DataFrame, models: list, corrs: list, reference_model: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CE = error_sums(grid)
    CE = CE.divide(CE.loc[reference_model], axis=1).loc[models, corrs]
    plt.figure(figsize=(7, 3.2))
    plt.imshow(CE.values, aspect="auto", cmap="Reds")
    plt.colorbar(label="CE (higher = more fragile)")
    plt.xticks(range(len(corrs)), corrs, rotation=30, ha="right"); plt.yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(len(corrs)):
            plt.text(j, i, f"{CE.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title(f"Per-corruption fragility (CE, ref={reference_model})"); plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/fig_fragility_heatmap.png", dpi=150); plt.close()

    corr = grid[grid.corruption != "clean"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for ax, c in zip(axes.ravel(), corrs):
        for m in models:
            d = corr[(corr.model == m) & (corr.corruption == c)].sort_values("severity")
            ax.plot(d.severity, d.macro_auroc, marker="o", label=m)
        ax.set_title(c); ax.set_xlabel("severity"); ax.grid(alpha=.3)
    axes.ravel()[0].set_ylabel("macro-AUROC"); axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Accuracy vs severity, by corruption"); fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/fig_severity_curves.png", dpi=150); plt.close(fig)

    try:
        from src.stats import critical_difference_diagram
        cond = {m: corr[corr.model == m].sort_values(["corruption", "severity"]).macro_auroc.values
                for m in models}
        fig, _ = critical_difference_diagram(cond, savepath=f"{RESULTS_DIR}/fig_cd_diagram.png")
        plt.close(fig)
        return "ok"
    except Exception as e:  # CD needs >=2 models & enough conditions
        return f"CD diagram skipped: {e}"


def main(reference_model: str = "minirocket", n_boot: int = 1000):
    grid = pd.read_csv(f"{RESULTS_DIR}/grid.csv")
    models = [m for m in MODEL_ORDER if m in set(grid.model)]
    corrs = list(grid[grid.corruption != "clean"].corruption.unique())

    clean = clean_leaderboard(grid)
    mce = mean_corruption_error(grid, reference_model)
    rce = relative_mce(grid, reference_model)
    lb = pd.DataFrame({"clean_auroc": pd.Series(clean), "mCE": mce, "relative_mCE": rce}).loc[models]
    lb.sort_values("mCE").to_csv(f"{RESULTS_DIR}/leaderboard.csv")

    from src.stats import spearman_exact

    rho, p_approx = rho_clean_vs_robustness(clean, mce.to_dict(), models)
    exact = spearman_exact([clean[m] for m in models], [-mce[m] for m in models])
    lo, hi, rhos = bootstrap_rho(f"{RESULTS_DIR}/preds", models, corrs, reference_model, n_boot=n_boot)
    boot_vals = [float(v) for v in sorted(set(np.round(rhos, 3)))]
    (fstat, fp), wh = significance(grid, models)
    (fstat_fam, fp_fam), n_blocks = significance_family_blocked(grid, models)
    _, _, ref_invariant = mce_reference_sensitivity(grid, models)
    n_refs = grid.model.nunique()
    cd = make_figures(grid, models, corrs, reference_model)

    reachable_below_07 = [r for r in exact["achievable_rho"] if r < 0.7 and r >= rho - 1e-9]
    rule_met = (rho < 0.7) and (hi < 0.9)
    verdict = "SUPPORTED" if rule_met else "NOT met (pre-registered weak-proxy rule)"
    lines = [
        "# ECG-C — Results summary\n",
        f"Zoo: {', '.join(models)} | reference (mCE): {reference_model}\n",
        "## Leaderboard (sorted by mCE)\n", lb.sort_values("mCE").round(4).to_markdown(),
        f"\n## RQ1 — clean accuracy vs robustness (n={len(models)} models — UNDERPOWERED)",
        f"- Spearman rho (clean vs -mCE) = **{rho:.3f}**",
        f"- EXACT permutation p: one-sided={exact['p_one_sided']:.3f}, two-sided={exact['p_two_sided']:.3f}"
        f"  (scipy t-approx p={p_approx:.3f} is INVALID at this n and is not used)",
        f"- Attainable rho at n={len(models)}: {exact['achievable_rho']}",
        f"- Record-bootstrap: distinct rho values = {boot_vals}; 95% CI [{lo:.3f}, {hi:.3f}]"
        f"  (resamples RECORDS with the {len(models)} models fixed; does NOT capture model-sampling uncertainty)",
        f"- Pre-registered rule (rho<0.7 & CI excludes 0.9): **{verdict}**",
    ]
    if not reachable_below_07 and rho >= 0.8:
        lines.append("  - NOTE: with the best and worst models rank-stable on both leaderboards, only "
                     "rho in {0.8, 1.0} is attainable, so rho<0.7 was UNREACHABLE by construction — "
                     "RQ1 is inconclusive, not evidence in either direction.")
    lines += [
        f"\n## Significance across the {len(corrs)*len(SEVERITIES)} conditions (blocks NOT independent)",
        f"- Friedman (all {len(corrs)*len(SEVERITIES)} conditions): stat={fstat:.2f}, p={fp:.2e}",
        f"- Friedman (family-blocked, {n_blocks} near-independent blocks): stat={fstat_fam:.2f}, p={fp_fam:.3f}"
        f"  (conservative; severities within a family are dependent, so the 25-condition p is anti-conservative)",
    ]
    for w in wh:
        lines.append(f"  - {w['pair'][0]} vs {w['pair'][1]}: p_holm={w['p_holm']:.4f} "
                     f"({'significant' if w['reject'] else 'ns'})")
    lines.append("\n## mCE reference-sensitivity")
    lines.append(f"- mCE ranking identical across all {n_refs} reference choices: **{ref_invariant}** "
                 "(verified empirically; not a structural property of the metric)")
    if rce.attrs.get("dropped_corruptions"):
        lines.append(f"\n_relative_mCE dropped (reference ~0 degradation): {rce.attrs['dropped_corruptions']}_")
    lines.append(f"\n## Figures\n- fig_fragility_heatmap.png\n- fig_severity_curves.png\n- fig_cd_diagram.png ({cd})\n")
    with open(f"{RESULTS_DIR}/summary.md", "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("\n[analysis] wrote results/leaderboard.csv, results/summary.md, figures.")


if __name__ == "__main__":
    main()
