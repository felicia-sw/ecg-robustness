"""The 5 primary models (one per paradigm).

Two interfaces, depending on the model:
- fit/predict classifiers (MiniRocket, catch22+ridge, InceptionTime, SE-ResNet)
- feature extractor + linear probe (Mantis)

Multi-label (5 superclasses): predict per-class probabilities; evaluate with
macro-AUROC. Stubs only — implement the bodies.

Fast-first: get MiniRocket, catch22+ridge, Mantis-probe working (CPU) before
the GPU models (InceptionTime, SE-ResNet). See ../REPORT-PLAN.md.
"""
from __future__ import annotations

import numpy as np

PRIMARY_MODELS = ["minirocket", "catch22_ridge", "inceptiontime", "se_resnet", "mantis_probe"]


def build_model(name: str, **kwargs):
    """Return an untrained model object for one of PRIMARY_MODELS."""
    raise NotImplementedError("TODO: dispatch by name (aeon / tsai / custom SE-ResNet / Mantis)")


def fit(model, X: np.ndarray, y: np.ndarray):
    """Train on CLEAN data only (never on benchmark corruptions)."""
    raise NotImplementedError("TODO")


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Return per-class probabilities, shape (n_records, 5)."""
    raise NotImplementedError("TODO")
