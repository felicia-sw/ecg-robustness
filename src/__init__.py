"""ECG-C: robustness of ECG classifiers to realistic sensor corruptions.

Modules
-------
- data        : load PTB-XL (100 Hz), folds, and 5-superclass multi-labels
- corruptions : physically-grounded corruption generators (incl. real NSTDB noise)
- models      : the 5 primary classifiers / feature extractors
- evaluate    : macro-AUROC, mean Corruption Error (mCE), eval loop
- stats       : significance tests + critical-difference diagram

These are stubs — fill in the implementations. See ../REPORT-PLAN.md.
"""
