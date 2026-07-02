"""Statistical comparison of models (the significance layer).

Core (main results):
- Spearman rho between the CLEAN-accuracy ranking and the ROBUSTNESS ranking
  (pre-registered decision rule: rho < 0.7 => clean accuracy is a weak proxy).
- Friedman test across (corruption x severity) conditions, then post-hoc
  pairwise Wilcoxon signed-rank with Holm-Bonferroni correction.
- Critical-difference (CD) diagram over mCE ranks (Demsar 2006).
"""
from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon


def clean_vs_mce_spearman(clean_scores: dict, mce_scores: dict) -> tuple[float, float]:
    """Spearman rho between clean accuracy and ROBUSTNESS (= -mCE) across models.

    We correlate clean macro-AUROC (higher = better) with -mCE (higher = more
    robust), so rho ~ +1 means clean accuracy strongly predicts robustness. The
    pre-registered rule: rho < 0.7 => clean accuracy is a weak proxy for robustness.
    Returns (rho, p_value).

    NOTE: the returned p-value is scipy's asymptotic (t/permutation) approximation and
    is NOT valid for the tiny n of a model zoo (e.g. n=4). For confirmatory reporting
    use ``spearman_exact`` below, which enumerates the exact permutation null and also
    reports which rho values are even attainable at this n.
    """
    models = list(clean_scores)
    clean = [clean_scores[m] for m in models]
    robustness = [-mce_scores[m] for m in models]
    rho, p = spearmanr(clean, robustness)
    return float(rho), float(p)


def achievable_spearman_rho(n: int) -> list[float]:
    """Every DISTINCT Spearman rho value attainable with ``n`` items (exact enumeration).

    At small n, rho is highly discrete. For n=4 the only attainable values are
    {-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, so a threshold such as
    the pre-registered ``rho < 0.7`` can only be met by rho <= 0.6 -- there is no value in
    (0.6, 0.8). If the best and worst models hold their ranks on both leaderboards, only the
    two middle models can move, so rho can ONLY be 1.0 (no swap) or 0.8 (one swap): the
    ``rho < 0.7`` rule is then unreachable regardless of the data. Callers should report
    this alongside the observed rho.
    """
    base = list(range(n))
    denom = n * (n * n - 1)
    vals = {round(1.0 - 6.0 * sum((a - b) ** 2 for a, b in zip(base, p)) / denom, 10)
            for p in permutations(base)}
    return sorted(vals, reverse=True)


def spearman_exact(clean: list[float], robustness: list[float]) -> dict:
    """Exact permutation inference for Spearman rho -- the correct test at small n.

    Enumerates all n! relabellings of one ranking against the other to build the exact
    null distribution of rho, instead of scipy's t/normal approximation (invalid for the
    n=4 model zoo). Returns the observed rho, exact one- and two-sided permutation
    p-values, n, and the attainable-rho set.

    For the ECG-C zoo (clean=[MiniRocket,Rocket,Hydra,catch22] AUROC vs -mCE) this gives
    rho=0.80 with exact p_one_sided=0.167, p_two_sided=0.333 -- i.e. the observed value is
    the *second-highest attainable* rho and is far from significant; the smallest two-sided
    p attainable at n=4 (perfect rank agreement, rho=1.0) is 2/24=0.083, so significance is
    impossible by construction.
    """
    n = len(clean)
    if n != len(robustness):
        raise ValueError("clean and robustness must have equal length")
    rc = np.asarray(rankdata(clean), dtype=float)
    rr = np.asarray(rankdata(robustness), dtype=float)
    denom = n * (n * n - 1)

    def _rho(a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - 6.0 * float(np.sum((a - b) ** 2)) / denom

    obs = _rho(rc, rr)
    null = np.array([_rho(rc, np.asarray(p, dtype=float)) for p in permutations(rr)])
    tol = 1e-9
    return {
        "rho": float(obs),
        "p_one_sided": float(np.mean(null >= obs - tol)),
        "p_two_sided": float(np.mean(np.abs(null) >= abs(obs) - tol)),
        "n": n,
        "n_permutations": int(null.size),
        "achievable_rho": achievable_spearman_rho(n),
    }


def _stack(scores_by_model_condition) -> tuple[list, np.ndarray]:
    """dict {model: [per-condition scores]} -> (models, array shape (n_models, n_conditions))."""
    models = list(scores_by_model_condition)
    mat = np.asarray([np.asarray(scores_by_model_condition[m], dtype=float) for m in models])
    return models, mat


def friedman_test(scores_by_model_condition) -> tuple[float, float]:
    """Friedman test over conditions; return (statistic, p_value)."""
    _, mat = _stack(scores_by_model_condition)
    stat, p = friedmanchisquare(*mat)   # one sample per model, paired across conditions
    return float(stat), float(p)


def wilcoxon_holm(scores_by_model_condition, alpha: float = 0.05):
    """Pairwise Wilcoxon signed-rank with Holm-Bonferroni correction.

    Returns a list of dicts: {pair, p, p_holm, reject}.
    """
    from statsmodels.stats.multitest import multipletests

    models, mat = _stack(scores_by_model_condition)
    idx = {m: i for i, m in enumerate(models)}
    pairs, pvals = [], []
    for a, b in combinations(models, 2):
        _, p = wilcoxon(mat[idx[a]], mat[idx[b]])
        pairs.append((a, b))
        pvals.append(p)
    reject, p_holm, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    return [dict(pair=pr, p=float(pv), p_holm=float(ph), reject=bool(r))
            for pr, pv, ph, r in zip(pairs, pvals, p_holm, reject)]


def average_ranks(scores_by_model_condition, lower_is_better: bool = False) -> dict:
    """Average rank of each model across conditions (rank 1 = best)."""
    models, mat = _stack(scores_by_model_condition)
    signed = mat if lower_is_better else -mat        # smaller = better
    ranks = np.apply_along_axis(rankdata, 0, signed)  # per-condition ranks; ties get mean rank
    return {m: float(r) for m, r in zip(models, ranks.mean(axis=1))}


def critical_difference_diagram(scores_by_model_condition, lower_is_better: bool = False,
                                savepath: str | None = None):
    """Draw a CD diagram (Demsar) from per-condition scores using aeon's helper."""
    from aeon.visualisation import plot_critical_difference

    models, mat = _stack(scores_by_model_condition)
    # aeon expects (n_datasets/conditions, n_estimators); higher = better
    scores = mat.T if not lower_is_better else (-mat).T
    fig, ax = plot_critical_difference(scores, list(models))
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
    return fig, ax
