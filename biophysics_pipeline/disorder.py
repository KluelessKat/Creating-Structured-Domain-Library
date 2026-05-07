"""
disorder.py
===========

Compute fraction of disordered residues per sequence using metapredict V2-FF
(Emenecker, Griffith, Holehouse — same lab that built ALBATROSS, and the
disorder predictor used throughout Lotthammer 2024).

If metapredict is not installed, we fall back to a "no-disorder-info"
sentinel so the rest of the pipeline still runs. (Folded domains tend to
have very low disorder fractions anyway — this metric is mostly informative
on full-protein distributions.)
"""

from __future__ import annotations
import numpy as np

try:
    import metapredict as meta
    _HAS_METAPREDICT = True
except ImportError:
    _HAS_METAPREDICT = False


def disorder_fraction(seq: str, threshold: float = 0.5) -> float:
    """
    Fraction of residues with metapredict disorder score >= threshold.
    Returns NaN if metapredict is not installed.

    threshold = 0.5 is the standard cutoff used in metapredict V2 / ALBATROSS.
    """
    if not _HAS_METAPREDICT:
        return np.nan
    if not isinstance(seq, str) or len(seq) == 0:
        return np.nan
    try:
        scores = meta.predict_disorder(seq)
        scores = np.asarray(scores)
        return float((scores >= threshold).mean())
    except Exception:
        return np.nan


def mean_disorder_score(seq: str) -> float:
    """Mean metapredict disorder score across the sequence."""
    if not _HAS_METAPREDICT:
        return np.nan
    if not isinstance(seq, str) or len(seq) == 0:
        return np.nan
    try:
        scores = meta.predict_disorder(seq)
        return float(np.mean(scores))
    except Exception:
        return np.nan
