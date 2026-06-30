"""PTB-XL loading: waveforms, recommended folds, and 5-superclass labels.

Stubs only — implement the bodies. Data lives in ../data/ptb-xl (git-ignored);
see ../data/README.md for the download.
"""
from __future__ import annotations

import numpy as np

DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SAMPLING_RATE_HZ = 100


def load_ptbxl(data_dir: str = "data/ptb-xl", sampling_rate: int = SAMPLING_RATE_HZ):
    """Load PTB-XL signals + metadata.

    Returns
    -------
    X : np.ndarray, shape (n_records, n_leads=12, n_samples)
    y : np.ndarray, shape (n_records, 5)  # multi-hot over DIAGNOSTIC_SUPERCLASSES
    meta : pandas.DataFrame  # incl. 'strat_fold'
    """
    raise NotImplementedError("TODO: read ptbxl_database.csv, load records100 via wfdb")


def scp_to_superclasses(scp_codes: dict, scp_statements_csv: str) -> list[str]:
    """Map a record's raw `scp_codes` dict to diagnostic superclasses."""
    raise NotImplementedError("TODO: join via scp_statements.csv 'diagnostic_class'")


def split_by_fold(meta, test_fold: int = 10, val_fold: int = 9):
    """Return boolean masks (train, val, test) using PTB-XL's `strat_fold`."""
    raise NotImplementedError("TODO: standard PTB-XL split (test=10, val=9, rest=train)")
