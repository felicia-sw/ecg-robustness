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

An EXTENDED zoo adds architecturally diverse GPU models so the clean-vs-mCE correlation
rests on more than the ~2 independent paradigms of the CPU zoo (three of the four CPU
models are ROCKET-family): InceptionTime and ResNet deep classifiers (multi-label via
one-vs-rest) and a Mantis foundation-feature + ridge probe. These require GPU deep-learning
extras and are UNTESTED in the review sandbox; validate on the target machine.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# aeon is imported lazily inside the builders so this module (and the dispatch/Mantis
# logic) imports without the heavy TSC/deep-learning extras installed.

PRIMARY_MODELS = ["minirocket", "rocket", "catch22_ridge", "hydra"]
# Extended GPU zoo (architecturally diverse; enlarges the effective n for the rho test).
EXTENDED_MODELS = ["inceptiontime", "resnet", "mantis"]
ALL_MODELS = PRIMARY_MODELS + EXTENDED_MODELS


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


class DeepOvRProbe:
    """Multi-LABEL wrapper around single-label aeon deep classifiers (one-vs-rest).

    PTB-XL is multi-label but aeon's deep classifiers are single-label, so we train one
    binary classifier per superclass and stack their positive-class probabilities into an
    (n, 5) score matrix for macro-AUROC -- preserving the per-class-score interface the
    rest of the pipeline expects. GPU strongly recommended (~5x the cost of one net).
    UNTESTED in the review sandbox (aeon deep-learning extras / torch absent here).
    """

    def __init__(self, make_clf):
        self._make_clf = make_clf            # zero-arg factory -> a fresh single-label clf
        self.clfs: list = []

    def fit(self, X, y):
        y = np.asarray(y)
        self.clfs = [self._make_clf() for _ in range(y.shape[1])]
        for k, clf in enumerate(self.clfs):
            clf.fit(X, y[:, k].astype(int))
        return self

    def predict_proba(self, X):
        cols = []
        for clf in self.clfs:
            p = np.asarray(clf.predict_proba(X))
            cols.append(p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel())
        return np.stack(cols, axis=1)


class MantisProbe:
    """Foundation-model FEATURES + the same multi-output ridge probe.

    Matches the pre-registered "Mantis features + linear probe": embed each record with a
    frozen encoder, standardise, then fit the multi-output ridge head. Pass ``embed_fn``
    mapping (n, 12, T) -> (n, d); if omitted, fitting raises with instructions (the Mantis
    weights install separately). GPU required for the encoder; UNTESTED here.
    """

    def __init__(self, embed_fn=None, random_state: int = 0):
        self.embed_fn = embed_fn
        self.scaler = StandardScaler()
        self.head = RidgeProbe()

    def _embed(self, X):
        if self.embed_fn is None:
            raise NotImplementedError(
                "MantisProbe needs embed_fn=(n,12,T)->(n,d) from the pretrained Mantis "
                "encoder; see the Mantis repo for loading weights.")
        return np.asarray(self.embed_fn(X))

    def fit(self, X, y):
        self.head.fit(self.scaler.fit_transform(self._embed(X)), y)
        return self

    def predict_proba(self, X):
        return self.head.predict_proba(self.scaler.transform(self._embed(X)))


def _build_transform(name: str, n_kernels: int, random_state: int):
    from aeon.transformations.collection.convolution_based import (
        HydraTransformer, MiniRocket, Rocket,
    )
    from aeon.transformations.collection.feature_based import Catch22
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


def _build_extended(name: str, random_state: int):
    """Builders for the GPU extended zoo (lazy imports so the CPU zoo needs no deep deps)."""
    if name == "inceptiontime":
        from aeon.classification.deep_learning import InceptionTimeClassifier
        return DeepOvRProbe(lambda: InceptionTimeClassifier(random_state=random_state))
    if name == "resnet":
        # Wang et al. 1D ResNet. For a squeeze-and-excite SE-ResNet, swap in tsai's XResNet1d.
        from aeon.classification.deep_learning import ResNetClassifier
        return DeepOvRProbe(lambda: ResNetClassifier(random_state=random_state))
    if name == "mantis":
        return MantisProbe(random_state=random_state)
    raise NotImplementedError(f"{name!r} not in EXTENDED_MODELS {EXTENDED_MODELS}.")


def build_model(name: str, n_kernels: int = 10_000, random_state: int = 0, **kwargs):
    """Return an untrained model object for one of ALL_MODELS (CPU zoo or extended GPU zoo)."""
    if name in EXTENDED_MODELS:
        return _build_extended(name, random_state)
    transform = _build_transform(name, n_kernels, random_state)
    return RocketProbe(transform, StandardScaler(), RidgeProbe())


def _transform_batched(transform, X: np.ndarray, batch: int = 1000) -> np.ndarray:
    """Transform in row-batches to cap peak memory (matters for torch-based Hydra)."""
    parts = [np.asarray(transform.transform(X[i:i + batch])) for i in range(0, len(X), batch)]
    return np.concatenate(parts, axis=0)


def fit(model, X: np.ndarray, y: np.ndarray):
    """Train on CLEAN data only (never on benchmark corruptions).

    X : (n, 12, 1000) float32      y : (n, 5) multi-hot
    RocketProbe uses the transform -> scaler -> ridge path; other models (deep / Mantis)
    expose their own ``.fit`` and are delegated to.
    """
    if not isinstance(model, RocketProbe):
        return model.fit(X, y)
    model.transform.fit(X)                               # fit transform params on the FULL train set
    features = _transform_batched(model.transform, X)    # batched transform (caps peak memory)
    features = model.scaler.fit_transform(features)
    model.head.fit(features, y)
    return model


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Return per-class probabilities, shape (n_records, 5)."""
    if not isinstance(model, RocketProbe):
        return model.predict_proba(X)
    features = _transform_batched(model.transform, X)    # transform only, never fit here
    features = model.scaler.transform(features)
    return model.head.predict_proba(features)
