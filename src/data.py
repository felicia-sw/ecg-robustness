"""PTB-XL loading: waveforms, recommended folds, and 5-superclass labels.

Data lives in ../data/ptb-xl (git-ignored); see ../data/README.md for the
download. Returns channel-major signals (n, 12, n_samples) ready for aeon/sktime.
"""
from __future__ import annotations

import ast
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import wfdb

DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
_SUPER_IDX = {c: i for i, c in enumerate(DIAGNOSTIC_SUPERCLASSES)}
SAMPLING_RATE_HZ = 100


@lru_cache(maxsize=4)
def _agg_df(scp_statements_csv: str) -> pd.DataFrame:
    """Diagnostic SCP statements only (cached); index = SCP code."""
    df = pd.read_csv(scp_statements_csv, index_col=0)
    return df[df.diagnostic == 1]


def scp_to_superclasses(scp_codes: dict, scp_statements_csv: str) -> list[str]:
    """Map a record's raw `scp_codes` dict to diagnostic superclasses."""
    agg = _agg_df(scp_statements_csv)
    return sorted({agg.loc[code, "diagnostic_class"]
                   for code in scp_codes if code in agg.index})


def load_ptbxl(data_dir: str = "data/ptb-xl", sampling_rate: int = SAMPLING_RATE_HZ):
    """Load PTB-XL signals + metadata.

    Returns
    -------
    X : np.ndarray (n_records, 12, n_samples)  float32, channel-major
    y : np.ndarray (n_records, 5)              multi-hot over DIAGNOSTIC_SUPERCLASSES
    meta : pandas.DataFrame                     incl. 'strat_fold', 'diagnostic_superclass'
    """
    meta = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"), index_col="ecg_id")
    meta.scp_codes = meta.scp_codes.apply(ast.literal_eval)   # stringified dict -> dict

    scp_csv = os.path.join(data_dir, "scp_statements.csv")
    meta["diagnostic_superclass"] = meta.scp_codes.apply(
        lambda d: scp_to_superclasses(d, scp_csv))

    # multi-hot labels
    y = np.zeros((len(meta), len(DIAGNOSTIC_SUPERCLASSES)), dtype=np.float32)
    for row, classes in enumerate(meta.diagnostic_superclass):
        for c in classes:
            y[row, _SUPER_IDX[c]] = 1.0

    # waveforms: wfdb returns (n, n_samples, 12) time-major -> transpose to (n, 12, n_samples)
    fname_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    signals = [wfdb.rdsamp(os.path.join(data_dir, f))[0] for f in meta[fname_col]]
    X = np.asarray(signals, dtype=np.float32).transpose(0, 2, 1)

    return X, y, meta


def split_by_fold(meta, test_fold: int = 10, val_fold: int = 9):
    """Return boolean masks (train, val, test) using PTB-XL's `strat_fold`."""
    fold = meta.strat_fold.to_numpy()
    test = fold == test_fold
    val = fold == val_fold
    train = ~(test | val)
    return train, val, test
