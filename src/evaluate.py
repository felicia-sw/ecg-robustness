"""Metrics + the corruption-evaluation loop.

Primary metric: macro-AUROC. Robustness: mean Corruption Error (mCE), an ECG
analogue of ImageNet-C, normalized to a fixed reference model:

    error(f, c, s)   = 1 - macro_auroc(f on corruption c at severity s)
    CE(f, c)         = sum_s error(f,c,s) / sum_s error(ref,c,s)
    mCE(f)           = mean_c CE(f, c)
    relative mCE     = degradation above each model's own clean error

Stubs only — implement the bodies. Keep a results manifest
(model, corruption, severity, seed, macro_auroc) from the start.
"""
from __future__ import annotations

import numpy as np


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged one-vs-rest AUROC over the 5 superclasses."""
    raise NotImplementedError("TODO: sklearn roc_auc_score(average='macro')")


def evaluate_clean(model, X_test, y_test) -> float:
    """Macro-AUROC on the clean test set."""
    raise NotImplementedError("TODO")


def evaluate_under_corruptions(model, X_test, y_test, corruptions, severities) -> "pd.DataFrame":
    """Run model across all (corruption, severity); return per-condition macro-AUROC rows."""
    raise NotImplementedError("TODO: loop corruptions x severities, record rows")


def mean_corruption_error(results, reference_model: str) -> "pd.Series":
    """Compute mCE per model from the per-condition results, normalized to a reference."""
    raise NotImplementedError("TODO: implement CE(f,c) then mCE(f)")


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
    """Percentile bootstrap CI over records/seeds."""
    raise NotImplementedError("TODO")
