"""Metrics + the corruption-evaluation loop.

Primary metric: macro-AUROC. Robustness: mean Corruption Error (mCE), an ECG
analogue of ImageNet-C, normalized to a fixed reference model:

    error(f, c, s)   = 1 - macro_auroc(f on corruption c at severity s)
    CE(f, c)         = sum_s error(f,c,s) / sum_s error(ref,c,s)
    mCE(f)           = mean_c CE(f, c)
    relative mCE     = degradation above each model's own clean error
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged one-vs-rest AUROC over the label columns.

    Columns with only one class present are skipped (a degenerate column has no
    defined AUROC); the result is the mean over evaluable columns. On the full
    PTB-XL test fold all 5 classes are present, so this equals the plain macro AUROC.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    aucs = [roc_auc_score(y_true[:, k], y_score[:, k])
            for k in range(y_true.shape[1]) if y_true[:, k].min() != y_true[:, k].max()]
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate_clean(model, X_test, y_test) -> float:
    """Macro-AUROC on the clean test set."""
    from .models import predict_proba
    return macro_auroc(y_test, predict_proba(model, X_test))


def evaluate_under_corruptions(model, X_test, y_test, corruptions, severities,
                               model_name: str = "model") -> pd.DataFrame:
    """Run one model across all (corruption, severity); return per-condition rows.

    Returns a DataFrame with columns: model, corruption, severity, macro_auroc.
    """
    from .corruptions import apply_corruption
    from .models import predict_proba

    rows = []
    for c in corruptions:
        for s in severities:
            Xc = apply_corruption(X_test, c, s)
            auroc = macro_auroc(y_test, predict_proba(model, Xc))
            rows.append(dict(model=model_name, corruption=c, severity=s, macro_auroc=auroc))
    return pd.DataFrame(rows)


# mCE, relative mCE, and the record-level bootstrap live in src/analysis.py
# (single source of truth: they operate on the full results/grid.csv).
