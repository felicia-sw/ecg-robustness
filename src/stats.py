"""Statistical comparison of models (the significance layer).

Core (main results):
- Spearman rho between the CLEAN-accuracy ranking and the ROBUSTNESS ranking
  (pre-registered decision rule: rho < 0.7 => clean accuracy is a weak proxy).
- Friedman test across (corruption x severity) conditions, then post-hoc
  pairwise Wilcoxon signed-rank with Holm-Bonferroni correction.
- Critical-difference (CD) diagram over mCE ranks (Demsar 2006).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon


def clean_vs_mce_spearman(clean_scores: dict, mce_scores: dict) -> tuple[float, float]:
    """Spearman rho between clean accuracy and ROBUSTNESS (= -mCE) across models.

    We correlate clean macro-AUROC (higher = better) with -mCE (higher = more
    robust), so rho ~ +1 means clean accuracy strongly predicts robustness. The
    pre-registered rule: rho < 0.7 => clean accuracy is a weak proxy for robustness.
    Returns (rho, p_value).
    """
    models = list(clean_scores)
    clean = [clean_scores[m] for m in models]
    robustness = [-mce_scores[m] for m in models]
    rho, p = spearmanr(clean, robustness)
    return float(rho), float(p)


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
