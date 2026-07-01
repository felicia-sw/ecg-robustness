"""The model zoo.

Report version runs a CPU-only zoo of four models spanning two paradigms, each a
`transform -> StandardScaler -> ridge probe` pipeline that handles PTB-XL's
MULTI-LABEL targets and yields per-class probabilities for macro-AUROC:

    minirocket    - MiniRocket random convolutional kernels (ROCKET family)
    rocket        - original Rocket kernels (ROCKET family)
    hydra         - Hydra competing convolutional kernels (ROCKET family)
    catch22_ridge - catch22 interpretable summary features (feature-based)

Why a transform + ridge probe rather than aeon's built-in classifiers? PTB-XL is
multi-LABEL and macro-AUROC needs per-class scores; ridge (closed-form) is the
standard, fast ROCKET head, and AUROC is rank-based so its continuous outputs are
a valid ranking signal.

Deep models (InceptionTime, 1D SE-ResNet) and the Mantis foundation-feature probe
are upgrades / future work (see REPORT-PLAN.md).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from aeon.transformations.collection.convolution_based import (
    HydraTransformer,
    MiniRocket,
    Rocket,
)
from aeon.transformations.collection.feature_based import Catch22

PRIMARY_MODELS = ["minirocket", "rocket", "catch22_ridge", "hydra"]


class RocketProbe:
    """Container bundling a collection transform + scaler + multi-label head."""

    def __init__(self, transform, scaler, head):
        self.transform = transform
        self.scaler = scaler
        self.head = head


class RidgeProbe:
    """Fast multi-label linear head for time-series features.

    Ridge regression on the 0/1 label matrix — closed-form and multi-output, the
    standard (and very fast) ROCKET head. macro-AUROC is rank-based, so the raw
    ridge scores are a valid ranking signal; we squash them through a logistic to
    return proba-like values in (0, 1) from predict_proba.
    """

    def __init__(self, alphas=np.logspace(-3, 3, 10)):
        self.reg = RidgeCV(alphas=alphas)   # picks alpha by efficient leave-one-out

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict_proba(self, X):
        scores = self.reg.predict(X)
        return 1.0 / (1.0 + np.exp(-scores))


def _build_transform(name: str, n_kernels: int, random_state: int):
    if name == "minirocket":
        return MiniRocket(n_kernels=n_kernels, random_state=random_state)
    if name == "rocket":
        return Rocket(n_kernels=2000, random_state=random_state)
    if name == "hydra":
        return HydraTransformer(random_state=random_state)
    if name == "catch22_ridge":
        return Catch22(replace_nans=True, n_jobs=-1)
    raise NotImplementedError(
        f"{name!r} not in the CPU zoo {PRIMARY_MODELS} (deep/Mantis are future work)."
    )


def build_model(name: str, n_kernels: int = 10_000, random_state: int = 0, **kwargs):
    """Return an untrained model object for one of PRIMARY_MODELS."""
    transform = _build_transform(name, n_kernels, random_state)
    return RocketProbe(transform, StandardScaler(), RidgeProbe())


def _transform_batched(transform, X: np.ndarray, batch: int = 1000) -> np.ndarray:
    """Transform in row-batches to cap peak memory (matters for torch-based Hydra)."""
    parts = [np.asarray(transform.transform(X[i:i + batch])) for i in range(0, len(X), batch)]
    return np.concatenate(parts, axis=0)


def fit(model: RocketProbe, X: np.ndarray, y: np.ndarray):
    """Train on CLEAN data only (never on benchmark corruptions).

    X : (n, 12, 1000) float32      y : (n, 5) multi-hot
    """
    model.transform.fit(X)                               # fit transform params on the FULL train set
    features = _transform_batched(model.transform, X)    # batched transform (caps peak memory)
    features = model.scaler.fit_transform(features)
    model.head.fit(features, y)
    return model


def predict_proba(model: RocketProbe, X: np.ndarray) -> np.ndarray:
    """Return per-class probabilities, shape (n_records, 5)."""
    features = _transform_batched(model.transform, X)    # transform only, never fit here
    features = model.scaler.transform(features)
    return model.head.predict_proba(features)
