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
