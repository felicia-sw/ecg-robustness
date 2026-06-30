# Data

Datasets are **not committed** (this folder is git-ignored except for this README and `.gitkeep`).
For the report you only need **~2 GB**: PTB-XL @ 100 Hz + the NSTDB noise records.

> Version numbers below are the latest known as of mid-2026. **Check the PhysioNet page** for the current version and adjust the URL if needed.

## 1. PTB-XL (12-lead diagnostic ECG) — 100 Hz only (~2 GB)
Page: https://physionet.org/content/ptb-xl/

Download the 100 Hz waveforms + the metadata CSVs (skips the larger 500 Hz set):
```bash
# from the repo root
mkdir -p data/ptb-xl

# 100 Hz waveform records only
wget -r -N -c -np -nH --cut-dirs=3 \
  https://physionet.org/files/ptb-xl/1.0.3/records100/ \
  -P data/ptb-xl/

# metadata
wget -N -c https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv -P data/ptb-xl/
wget -N -c https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv -P data/ptb-xl/
```
Notes:
- `ptbxl_database.csv` has the labels (`scp_codes`) and the **recommended `strat_fold`** column — use fold 10 as test, 9 as validation (the standard PTB-XL split).
- Map `scp_codes` → 5 diagnostic **superclasses** via `scp_statements.csv` (`diagnostic_class`).
- The `wfdb` helper `wfdb.dl_database('ptb-xl', ...)` downloads **both** sampling rates (large) — prefer the `wget` command above to stay at ~2 GB.

## 2. MIT-BIH Noise Stress Test Database (NSTDB) — real recorded noise (tiny)
Page: https://physionet.org/content/nstdb/

```bash
mkdir -p data/nstdb
wget -r -N -c -np -nH --cut-dirs=3 \
  https://physionet.org/files/nstdb/1.0.0/ \
  -P data/nstdb/
```
This gives the three real noise records used as the realism anchor:
- `bw` — baseline wander
- `ma` — muscle (EMG) artifact
- `em` — electrode-motion artifact

Read with `wfdb.rdrecord('data/nstdb/bw')` and add to clean signals at calibrated SNRs.

## 3. (Later / optional — not needed for the report)
Cross-dataset external validity only:
- Chapman-Shaoxing: https://physionet.org/content/ecg-arrhythmia/
- CPSC 2018: http://2018.icbeb.org/Challenge.html
- PhysioNet/CinC Challenge 2020: https://physionet.org/content/challenge-2020/

## Expected layout
```
data/
├── README.md
├── ptb-xl/
│   ├── records100/...          # .dat / .hea waveform files
│   ├── ptbxl_database.csv
│   └── scp_statements.csv
└── nstdb/
    ├── bw.dat / bw.hea
    ├── ma.dat / ma.hea
    └── em.dat / em.hea
```
