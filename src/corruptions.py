"""ECG-C corruption generators.

Each corruption maps a clean signal -> corrupted signal at a given severity
(1..5). Severity is calibrated to a physical parameter (SNR in dB, gain factor,
ENOB) so it is monotone and comparable across records.

Signals are channel-major float arrays: (n, 12, T) for a batch or (12, T) for a
single record, T=1000 (10 s @ 100 Hz). Real recorded noise (baseline wander /
muscle / electrode-motion) comes from the MIT-BIH NSTDB (../data/nstdb); the
rest are synthetic-but-grounded.

Severity is PRE-REGISTERED (docs/pre-registration.md); models are never trained
on these corruptions.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import wfdb
from scipy.signal import resample_poly

SEVERITIES = [1, 2, 3, 4, 5]
# SNR (dB) ladder for additive-noise corruptions (high -> low SNR = worse):
SNR_DB_BY_SEVERITY = {1: 18, 2: 12, 3: 6, 4: 0, 5: -6}
# multiplicative gain ladder (gain miscalibration):
GAIN_BY_SEVERITY = {1: 1.1, 2: 1.25, 3: 1.5, 4: 2.0, 5: 3.0}
# effective bit-depth ladder (quantization; fewer bits = worse):
BITS_BY_SEVERITY = {1: 10, 2: 8, 3: 6, 4: 5, 5: 4}

_PTBXL_FS = 100
_NSTDB_FS = 360


def _add_noise_at_snr(x: np.ndarray, noise: np.ndarray, severity: int) -> np.ndarray:
    """Scale `noise` per (record, lead) so x+noise hits the severity's SNR; return x+noise.

    SNR_dB = 10*log10(P_signal / P_noise)  =>  scale = sqrt(P_sig / (P_noise * 10^(SNR/10)))
    """
    snr_lin = 10.0 ** (SNR_DB_BY_SEVERITY[severity] / 10.0)
    p_sig = np.mean(x ** 2, axis=-1, keepdims=True)
    p_noise = np.mean(noise ** 2, axis=-1, keepdims=True)
    scale = np.sqrt(np.divide(p_sig, p_noise * snr_lin,
                              out=np.zeros_like(p_sig), where=p_noise > 0))
    return (x + scale * noise).astype(np.float32)


# --- real recorded noise (NSTDB) ---
@lru_cache(maxsize=3)
def _nstdb_noise_pool(noise_type: str, nstdb_dir: str = "data/nstdb") -> np.ndarray:
    """Load an NSTDB record, resample 360 -> 100 Hz, return one long 1-D pool (float32).

    Both channels are resampled and concatenated for a large noise pool. 100/360
    reduces to 5/18, so resample_poly(., 5, 18).
    """
    rec = wfdb.rdrecord(f"{nstdb_dir}/{noise_type}")
    sig = rec.p_signal.astype(np.float64)  # (650000, 2) @ 360 Hz
    pool = np.concatenate([resample_poly(sig[:, ch], up=5, down=18)
                           for ch in range(sig.shape[1])])
    return pool.astype(np.float32)


def add_nstdb_noise(x: np.ndarray, noise_type: str, severity: int,
                    nstdb_dir: str = "data/nstdb", seed: int = 0) -> np.ndarray:
    """Add real recorded NSTDB noise ('bw' | 'ma' | 'em') at the severity's SNR.

    Each lead gets an independent random window of real noise (reproducible via seed).
    """
    x = np.asarray(x, dtype=np.float32)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None]                       # (12, T) -> (1, 12, T)
    n, c, T = x.shape

    pool = _nstdb_noise_pool(noise_type, nstdb_dir)
    rng = np.random.default_rng(seed + severity)          # reproducible per severity
    starts = rng.integers(0, len(pool) - T, size=(n, c))
    noise = pool[starts[..., None] + np.arange(T)]        # (n, c, T) random windows

    out = _add_noise_at_snr(x, noise, severity)
    return out[0] if squeeze else out


# --- synthetic but physiologically grounded ---
def powerline_interference(x: np.ndarray, severity: int, freq_hz: float = 50.0) -> np.ndarray:
    """Add a 50/60 Hz sinusoid (+ 2nd/3rd harmonics) at the severity's SNR.

    WARNING: invalid at fs = 100 Hz. Nyquist is 50 Hz, so a 50 Hz tone samples to ~0
    and the SNR scaler then amplifies floating-point residuals instead of injecting a
    real powerline tone. EXCLUDED from CORRUPTIONS at 100 Hz; kept for use at >= 500 Hz.
    """
    x = np.asarray(x, dtype=np.float32)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None]
    T = x.shape[-1]
    t = np.arange(T) / _PTBXL_FS
    tone = (np.sin(2 * np.pi * freq_hz * t)
            + 0.3 * np.sin(2 * np.pi * 2 * freq_hz * t)
            + 0.15 * np.sin(2 * np.pi * 3 * freq_hz * t)).astype(np.float32)
    noise = np.broadcast_to(tone, x.shape)
    out = _add_noise_at_snr(x, noise, severity)
    return out[0] if squeeze else out


def gaussian_noise(x: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    """Add white Gaussian (instrumentation) noise at the severity's SNR."""
    x = np.asarray(x, dtype=np.float32)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None]
    rng = np.random.default_rng(seed + severity)
    noise = rng.standard_normal(x.shape).astype(np.float32)
    out = _add_noise_at_snr(x, noise, severity)
    return out[0] if squeeze else out


def gain_miscalibration(x: np.ndarray, severity: int) -> np.ndarray:
    """Multiply amplitude by a calibration gain factor (GAIN_BY_SEVERITY)."""
    return (np.asarray(x, dtype=np.float32) * GAIN_BY_SEVERITY[severity]).astype(np.float32)


def quantization(x: np.ndarray, severity: int) -> np.ndarray:
    """Reduce ADC bit-depth per lead (BITS_BY_SEVERITY): quantize to 2**bits levels."""
    x = np.asarray(x, dtype=np.float32)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None]
    levels = 2 ** BITS_BY_SEVERITY[severity]
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    span = np.where(hi > lo, hi - lo, 1.0)
    q = np.round((x - lo) / span * (levels - 1)) / (levels - 1) * span + lo
    out = q.astype(np.float32)
    return out[0] if squeeze else out


# Registry: name -> callable(x, severity). Single source of truth.
# powerline is EXCLUDED at fs = 100 Hz (50 Hz = Nyquist -> degenerate tone; see the
# powerline_interference warning and docs/ECG-C-report.md Limitations). Re-add for >= 500 Hz.
CORRUPTIONS = {
    "baseline_wander": lambda x, s: add_nstdb_noise(x, "bw", s),
    "muscle_artifact": lambda x, s: add_nstdb_noise(x, "ma", s),
    "gaussian": gaussian_noise,
    "gain": gain_miscalibration,
    "quantization": quantization,
}


def apply_corruption(x: np.ndarray, name: str, severity: int) -> np.ndarray:
    """Dispatch to a corruption by name from CORRUPTIONS."""
    return CORRUPTIONS[name](x, severity)
