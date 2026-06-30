"""ECG-C corruption generators.

Each corruption maps a clean signal -> corrupted signal at a given severity
(1..5). Severity must be calibrated to a physical parameter (SNR in dB, drift
amplitude, ENOB, etc.) so it is monotone and comparable across records.

Real recorded noise (baseline wander / muscle / electrode-motion) comes from
the MIT-BIH NSTDB (../data/nstdb). Everything else is synthetic-but-grounded.

Stubs only — implement the bodies. PRE-REGISTER the final set + severities
before the full run; do NOT train models on these corruptions.
"""
from __future__ import annotations

import numpy as np

SEVERITIES = [1, 2, 3, 4, 5]
# Suggested SNR (dB) ladder for additive-noise corruptions (high -> low SNR):
SNR_DB_BY_SEVERITY = {1: 18, 2: 12, 3: 6, 4: 0, 5: -6}


# --- real recorded noise (NSTDB) ---
def add_nstdb_noise(x: np.ndarray, noise_type: str, severity: int,
                    nstdb_dir: str = "data/nstdb") -> np.ndarray:
    """Add real recorded NSTDB noise ('bw' | 'ma' | 'em') at the severity's SNR."""
    raise NotImplementedError("TODO: load noise record, scale to target SNR, add")


# --- synthetic but physiologically grounded ---
def powerline_interference(x: np.ndarray, severity: int, freq_hz: float = 50.0) -> np.ndarray:
    """Add 50/60 Hz sinusoid (+ harmonics) at the severity's amplitude/SNR."""
    raise NotImplementedError("TODO")


def gaussian_noise(x: np.ndarray, severity: int) -> np.ndarray:
    """Add white Gaussian noise at the severity's SNR."""
    raise NotImplementedError("TODO")


def gain_miscalibration(x: np.ndarray, severity: int) -> np.ndarray:
    """Multiply amplitude by a gain factor (calibration error)."""
    raise NotImplementedError("TODO")


def quantization(x: np.ndarray, severity: int) -> np.ndarray:
    """Reduce ADC bit-depth (ENOB) — coarser quantization at higher severity."""
    raise NotImplementedError("TODO")


# Registry: name -> callable(x, severity). Keep this as the single source of truth.
CORRUPTIONS = {
    "baseline_wander": lambda x, s: add_nstdb_noise(x, "bw", s),
    "muscle_artifact": lambda x, s: add_nstdb_noise(x, "ma", s),
    "powerline": powerline_interference,
    "gaussian": gaussian_noise,
    "gain": gain_miscalibration,
    "quantization": quantization,
}


def apply_corruption(x: np.ndarray, name: str, severity: int) -> np.ndarray:
    """Dispatch to a corruption by name from CORRUPTIONS."""
    raise NotImplementedError("TODO: look up CORRUPTIONS[name](x, severity)")
