"""Mutation-resistant tests for the ECG-C pipeline.

Fast, mostly data-free (synthetic signals). Tests that need downloaded data or the
cached arrays are skipped when those are absent. Run: `pytest` from the repo root.
"""
import os

import numpy as np
import pandas as pd
import pytest

from src.corruptions import (
    SEVERITIES, SNR_DB_BY_SEVERITY, BITS_BY_SEVERITY, CORRUPTIONS,
    add_nstdb_noise, gaussian_noise, gain_miscalibration, quantization, apply_corruption,
)
from src.analysis import mean_corruption_error
from src.stats import average_ranks

RNG = np.random.default_rng(0)


def _sig(n=64):
    return (RNG.standard_normal((n, 12, 1000)) * 0.5).astype("float32")


def _measured_snr(x, xc):
    noise = xc - x
    ps = np.mean(x ** 2, axis=-1)
    pn = np.mean(noise ** 2, axis=-1)
    m = pn > 0
    return 10 * np.log10(ps[m] / pn[m])


# ---- corruption calibration (mutation-resistant: catches a broken scale formula) ----
@pytest.mark.parametrize("sev", SEVERITIES)
def test_gaussian_snr_calibration(sev):
    x = _sig()
    assert np.isclose(_measured_snr(x, gaussian_noise(x, sev)).mean(),
                      SNR_DB_BY_SEVERITY[sev], atol=0.2)


def test_powerline_excluded_and_degenerate_at_100hz():
    # 50 Hz mains at fs=100 Hz is exactly Nyquist -> the tone samples to ~0, so it is
    # excluded from the suite (see report Limitations). Guards against silent re-inclusion.
    assert "powerline" not in CORRUPTIONS
    t = np.arange(1000) / 100.0
    assert np.abs(np.sin(2 * np.pi * 50.0 * t)).max() < 1e-9


def test_corruption_shapes_and_dtype():
    x = _sig(8)
    for name in ["gaussian", "gain", "quantization"]:
        xc = apply_corruption(x, name, 3)
        assert xc.shape == x.shape and xc.dtype == np.float32
    assert gaussian_noise(x[0], 3).shape == x[0].shape  # 2D single-record path


def test_quantization_reduces_levels():
    xc = quantization(_sig(4), 5)  # 4 bits -> <= 16 levels per lead
    for lead in xc.reshape(-1, 1000):
        assert len(np.unique(lead)) <= 2 ** BITS_BY_SEVERITY[5]


def test_gain_is_multiplicative():
    x = _sig(4)
    assert np.allclose(gain_miscalibration(x, 3), x * 1.5)


def test_gaussian_determinism_and_severity():
    x = _sig()
    assert np.array_equal(gaussian_noise(x, 3), gaussian_noise(x, 3))       # reproducible
    assert not np.array_equal(gaussian_noise(x, 1), gaussian_noise(x, 5))   # severity matters


# ---- mCE invariants (catches summation/normalization bugs) ----
def test_mce_reference_is_one_and_scales():
    rows = []
    for m, err in [("ref", 0.1), ("bad", 0.2)]:  # 'bad' has exactly 2x the error everywhere
        for c in ["a", "b", "c"]:
            for s in SEVERITIES:
                rows.append(dict(model=m, corruption=c, severity=s, macro_auroc=1 - err))
        rows.append(dict(model=m, corruption="clean", severity=0, macro_auroc=0.9))
    mce = mean_corruption_error(pd.DataFrame(rows), "ref")
    assert np.isclose(mce["ref"], 1.0)
    assert np.isclose(mce["bad"], 2.0)


def test_average_ranks_handles_ties():
    r = average_ranks({"a": [0.9, 0.8, 0.7], "b": [0.9, 0.8, 0.7]})
    assert np.isclose(r["a"], r["b"]) and np.isclose(r["a"], 1.5)


# ---- data-dependent tests (skip if artifacts absent) ----
@pytest.mark.skipif(not os.path.exists("data/nstdb/bw.dat"), reason="NSTDB not downloaded")
@pytest.mark.parametrize("sev", SEVERITIES)
def test_nstdb_snr_calibration(sev):
    x = _sig()
    assert np.isclose(_measured_snr(x, add_nstdb_noise(x, "bw", sev)).mean(),
                      SNR_DB_BY_SEVERITY[sev], atol=0.2)


@pytest.mark.skipif(not os.path.exists("data/y_100.npy"), reason="cached labels absent")
def test_loader_class_counts():
    from src.data import DIAGNOSTIC_SUPERCLASSES
    y = np.load("data/y_100.npy")
    counts = dict(zip(DIAGNOSTIC_SUPERCLASSES, y.sum(0).astype(int)))
    assert y.shape == (21799, 5)
    assert counts == {"NORM": 9514, "MI": 5469, "STTC": 5235, "CD": 4898, "HYP": 2649}


# ---- inference-layer tests (the statistics the paper's conclusions rest on) ----
def test_macro_auroc_matches_sklearn():
    """macro_auroc must equal the mean of per-column sklearn AUROC on evaluable columns."""
    from sklearn.metrics import roc_auc_score
    from src.evaluate import macro_auroc
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=(200, 5))
    # guarantee every column has both classes
    y_true[0, :] = 0; y_true[1, :] = 1
    y_score = rng.random((200, 5))
    expected = np.mean([roc_auc_score(y_true[:, k], y_score[:, k]) for k in range(5)])
    assert np.isclose(macro_auroc(y_true, y_score), expected)


def test_macro_auroc_skips_degenerate_column():
    """A single-class column has no defined AUROC and must be skipped, not error."""
    from src.evaluate import macro_auroc
    y_true = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])  # col 0 is all-ones (degenerate)
    y_score = np.array([[0.9, 0.2], [0.1, 0.8], [0.4, 0.3], [0.6, 0.7]])
    from sklearn.metrics import roc_auc_score
    assert np.isclose(macro_auroc(y_true, y_score), roc_auc_score(y_true[:, 1], y_score[:, 1]))


def test_achievable_spearman_rho_is_discrete_at_n4():
    """At n=4 there is NO attainable rho in (0.6, 0.8): the pre-registered rho<0.7 rule
    can only be met by rho<=0.6, which the ECG-C ranking (rho=0.8) cannot reach."""
    from src.stats import achievable_spearman_rho
    vals = achievable_spearman_rho(4)
    assert 0.8 in vals and 1.0 in vals and 0.6 in vals
    assert not any(0.6 < v < 0.8 for v in vals)      # nothing between 0.6 and 0.8
    assert not any(0.7 <= v < 0.8 for v in vals)      # so rho<0.7 needs rho<=0.6


def test_spearman_exact_matches_ecgc_ranking():
    """Exact permutation p for the ECG-C ranking (one middle swap): rho=0.8, p_two=1/3."""
    from src.stats import spearman_exact
    clean = [0.9011, 0.8875, 0.8912, 0.8406]          # MiniRocket, Rocket, Hydra, catch22
    mce = [1.000, 1.041, 1.198, 1.567]
    r = spearman_exact(clean, [-m for m in mce])
    assert np.isclose(r["rho"], 0.8)
    assert np.isclose(r["p_one_sided"], 4 / 24) and np.isclose(r["p_two_sided"], 8 / 24)
    assert r["n_permutations"] == 24


def test_spearman_exact_perfect_agreement_still_insignificant_at_n4():
    """Even perfect rank agreement (rho=1.0) has exact two-sided p=2/24=0.083 at n=4,
    so significance at alpha=0.05 is impossible by construction."""
    from src.stats import spearman_exact
    r = spearman_exact([1, 2, 3, 4], [1, 2, 3, 4])
    assert np.isclose(r["rho"], 1.0)
    assert np.isclose(r["p_two_sided"], 2 / 24) and r["p_two_sided"] > 0.05


def test_mce_reference_sensitivity_ranking_invariant_on_toy():
    """On the 2x-error toy, mCE ranking must be identical under either reference."""
    from src.analysis import mce_reference_sensitivity
    rows = []
    for m, err in [("ref", 0.1), ("bad", 0.2)]:
        for c in ["a", "b", "c"]:
            for s in SEVERITIES:
                rows.append(dict(model=m, corruption=c, severity=s, macro_auroc=1 - err))
        rows.append(dict(model=m, corruption="clean", severity=0, macro_auroc=0.9))
    _, rankings, invariant = mce_reference_sensitivity(pd.DataFrame(rows), ["ref", "bad"])
    assert invariant and rankings["ref"] == ("ref", "bad")


def test_multiseed_leaderboard_aggregates_and_counts_rankings():
    """Two seeds with a stable ordering must aggregate to that ranking with sd defined."""
    from src.analysis import multiseed_leaderboard
    rows = []
    for seed in (0, 1):
        for m, err in [("ref", 0.10), ("bad", 0.20)]:
            for c in ["a", "b"]:
                for s in SEVERITIES:
                    rows.append(dict(seed=seed, model=m, corruption=c, severity=s,
                                     macro_auroc=1 - err))
            rows.append(dict(seed=seed, model=m, corruption="clean", severity=0, macro_auroc=0.9))
    agg, ranking_counts = multiseed_leaderboard(pd.DataFrame(rows), "ref")
    assert list(agg.index) == ["ref", "bad"]              # ref (mCE=1) ranks above bad (mCE=2)
    assert ranking_counts[("ref", "bad")] == 2            # stable across both seeds
    assert np.isclose(agg.loc["bad", "mCE_mean"], 2.0)


# ---- extended (GPU) zoo: dispatch, Mantis probe, polymorphic fit/predict (no aeon/torch) ----
def test_zoo_membership_and_mantis_dispatch():
    from src.models import PRIMARY_MODELS, EXTENDED_MODELS, ALL_MODELS, build_model, MantisProbe
    assert ALL_MODELS == PRIMARY_MODELS + EXTENDED_MODELS
    assert {"inceptiontime", "resnet", "mantis"} <= set(EXTENDED_MODELS)
    assert isinstance(build_model("mantis"), MantisProbe)   # no aeon import needed for Mantis


def test_mantis_probe_requires_embed_fn_then_works():
    from src.models import build_model, fit, predict_proba
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 12, 40)).astype("float32")
    y = rng.integers(0, 2, size=(20, 5)).astype("float32")
    m = build_model("mantis")
    with pytest.raises(NotImplementedError):        # no embed_fn -> clear failure, not silent
        fit(m, X, y)
    m.embed_fn = lambda a: np.asarray(a).reshape(len(a), -1)   # trivial "encoder"
    fit(m, X, y)
    p = predict_proba(m, X)
    assert p.shape == (20, 5) and (p >= 0).all() and (p <= 1).all()


def test_polymorphic_fit_predict_delegate_to_non_rocketprobe():
    """Module-level fit/predict_proba must delegate to any object exposing .fit/.predict_proba."""
    from src.models import fit, predict_proba

    class _Fake:
        fitted = False
        def fit(self, X, y): self.fitted = True; return self
        def predict_proba(self, X): return np.zeros((len(X), 5))

    m = _Fake()
    assert fit(m, np.zeros((3, 12, 10)), np.zeros((3, 5))) is m and m.fitted
    assert predict_proba(m, np.zeros((3, 12, 10))).shape == (3, 5)


def test_deep_ovr_probe_stacks_per_class_probabilities():
    """One-vs-rest wrapper turns 5 single-label classifiers into an (n, 5) score matrix."""
    from src.models import DeepOvRProbe

    class _FakeBinaryClf:
        def fit(self, X, y): return self
        def predict_proba(self, X):
            p = np.linspace(0.1, 0.9, len(X))
            return np.c_[1 - p, p]                    # (n, 2): P(0), P(1)

    X = np.zeros((8, 12, 10)); y = np.zeros((8, 5))
    probe = DeepOvRProbe(lambda: _FakeBinaryClf()).fit(X, y)
    out = probe.predict_proba(X)
    assert out.shape == (8, 5) and len(probe.clfs) == 5
