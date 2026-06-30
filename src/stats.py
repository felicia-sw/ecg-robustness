"""Statistical comparison of models (the significance layer).

Core (main results):
- Spearman rho between the CLEAN-accuracy ranking and the mCE ranking
  (pre-registered decision rule: rho < 0.7 => clean accuracy is a weak proxy).
- Friedman test across (corruption x severity) conditions, then post-hoc
  pairwise Wilcoxon signed-rank with Holm-Bonferroni correction.
- Critical-difference (CD) diagram over mCE ranks (Demsar 2006).

Supplementary:
- paired bootstrap / Wilcoxon for per-model clean->corrupted degradation,
  reported with effect sizes; optional Bayesian signed-rank.

Stubs only — implement the bodies. Refs [26-28] in docs/ECG-C-proposal.md.
"""
from __future__ import annotations

import numpy as np


def clean_vs_mce_spearman(clean_scores: dict, mce_scores: dict) -> float:
    """Spearman rho between clean-accuracy ranking and mCE ranking across models."""
    raise NotImplementedError("TODO: scipy.stats.spearmanr on the two rank vectors")


def friedman_test(scores_by_model_condition) -> tuple[float, float]:
    """Friedman test over conditions; return (statistic, p_value)."""
    raise NotImplementedError("TODO: scipy.stats.friedmanchisquare")


def wilcoxon_holm(scores_by_model_condition, alpha: float = 0.05):
    """Pairwise Wilcoxon signed-rank with Holm-Bonferroni correction."""
    raise NotImplementedError("TODO: per-pair wilcoxon + statsmodels multipletests(method='holm')")


def critical_difference_diagram(avg_ranks: dict, savepath: str | None = None):
    """Draw a CD diagram from average ranks (e.g., aeon/scikit-posthocs helper)."""
    raise NotImplementedError("TODO")
