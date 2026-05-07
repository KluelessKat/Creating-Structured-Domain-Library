"""
properties.py
=============

Core biophysical property calculations for protein sequences.

All functions take an amino acid sequence (uppercase string) and return a float.
Designed to work on any sequence — full proteins, structured domains, or IDRs.

References:
- Kyte-Doolittle hydropathy: J. Mol. Biol. 157, 105 (1982)
- kappa (charge patterning): Das & Pappu, PNAS 110, 13392 (2013)
- omega (general patterning): Martin et al., NARDINI; Holehouse et al., CIDER
- SCD (sequence charge decoration): Sawle & Ghosh, J. Chem. Phys. 143 (2015)
- SHD (sequence hydropathy decoration): Zheng et al., J. Phys. Chem. Lett. 11 (2020)

For kappa/omega/SCD/SHD we rely on `localcider` (Holehouse lab) when available,
which is the canonical reference implementation used by both Kappel et al. and
Lotthammer et al. ALBATROSS. We provide a self-contained fallback for SCD/SHD
so the pipeline still runs without localcider.
"""

from __future__ import annotations
import numpy as np

# ---- amino acid sets -----------------------------------------------------

POSITIVE = set("RK")
NEGATIVE = set("DE")
CHARGED  = POSITIVE | NEGATIVE
AROMATIC = set("FWY")           # canonical aromatic set used in CondenSeq paper
HYDROPHOBIC_ALIPHATIC = set("AILMV")
POLAR = set("STNQH")            # H counted as polar here; treat charge separately
PROLINE = set("P")
GLYCINE = set("G")

# Kyte-Doolittle hydropathy index
KD_HYDROPATHY = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}
# normalized to [0, 1] for SHD (matches Zheng et al. 2020 convention)
_kd_min, _kd_max = min(KD_HYDROPATHY.values()), max(KD_HYDROPATHY.values())
KD_NORM = {aa: (h - _kd_min) / (_kd_max - _kd_min) for aa, h in KD_HYDROPATHY.items()}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _clean(seq: str) -> str:
    """Uppercase and strip non-canonical residues. Returns empty string if invalid."""
    if not isinstance(seq, str):
        return ""
    s = "".join(c for c in seq.upper() if c in VALID_AA)
    return s


# ---- composition metrics -------------------------------------------------

def fraction(seq: str, residues) -> float:
    """Fraction of residues in `seq` that are members of `residues` (a set or string)."""
    s = _clean(seq)
    if not s:
        return np.nan
    rset = set(residues)
    return sum(1 for c in s if c in rset) / len(s)


def fcr(seq: str) -> float:
    """Fraction of charged residues (R+K+D+E)."""
    return fraction(seq, CHARGED)


def ncpr(seq: str) -> float:
    """Net charge per residue: ((#R+#K) - (#D+#E)) / length."""
    s = _clean(seq)
    if not s:
        return np.nan
    pos = sum(1 for c in s if c in POSITIVE)
    neg = sum(1 for c in s if c in NEGATIVE)
    return (pos - neg) / len(s)


def fraction_aromatic(seq: str) -> float:
    return fraction(seq, AROMATIC)


def fraction_positive(seq: str) -> float:
    return fraction(seq, POSITIVE)


def fraction_negative(seq: str) -> float:
    return fraction(seq, NEGATIVE)


def fraction_R(seq: str) -> float: return fraction(seq, "R")
def fraction_K(seq: str) -> float: return fraction(seq, "K")
def fraction_D(seq: str) -> float: return fraction(seq, "D")
def fraction_E(seq: str) -> float: return fraction(seq, "E")
def fraction_proline(seq: str) -> float: return fraction(seq, PROLINE)
def fraction_glycine(seq: str) -> float: return fraction(seq, GLYCINE)
def fraction_hydrophobic(seq: str) -> float: return fraction(seq, HYDROPHOBIC_ALIPHATIC)
def fraction_polar(seq: str) -> float: return fraction(seq, POLAR)


def mean_hydropathy(seq: str) -> float:
    """Mean Kyte-Doolittle hydropathy (raw scale)."""
    s = _clean(seq)
    if not s:
        return np.nan
    return float(np.mean([KD_HYDROPATHY[c] for c in s]))


def mean_hydropathy_normalized(seq: str) -> float:
    """Mean Kyte-Doolittle hydropathy normalized to [0, 1]. Used by SHD."""
    s = _clean(seq)
    if not s:
        return np.nan
    return float(np.mean([KD_NORM[c] for c in s]))


# ---- patterning metrics --------------------------------------------------
# kappa and omega are computationally non-trivial. We use localcider when
# available (the Holehouse lab reference implementation). SCD and SHD are
# simple closed-form expressions, so we always implement them ourselves.

try:
    from localcider.sequenceParameters import SequenceParameters
    _HAS_LOCALCIDER = True
except ImportError:
    _HAS_LOCALCIDER = False


def kappa(seq: str) -> float:
    """
    Charge patterning parameter (Das & Pappu 2013).
    Range [0, 1]: 0 = perfectly mixed +/- charges, 1 = fully segregated.
    Requires at least one + and one - residue and length >= ~10.
    """
    s = _clean(seq)
    if len(s) < 10:
        return np.nan
    if not _HAS_LOCALCIDER:
        return np.nan  # see SCD as alternative
    try:
        return SequenceParameters(s).get_kappa()
    except Exception:
        return np.nan


def omega(seq: str) -> float:
    """
    Generalized patterning parameter for charged + proline residues vs others
    (Martin et al. 2020 / NARDINI). Range [0, 1].
    """
    s = _clean(seq)
    if len(s) < 10:
        return np.nan
    if not _HAS_LOCALCIDER:
        return np.nan
    try:
        return SequenceParameters(s).get_Omega()
    except Exception:
        return np.nan


def scd(seq: str) -> float:
    """
    Sequence Charge Decoration (Sawle & Ghosh 2015).
    SCD = (1/N) * sum_{i<j} q_i * q_j * sqrt(j - i)

    Negative SCD => alternating charges (compact / well-mixed)
    Less negative / positive SCD => block charges (expanded)
    No reliance on localcider; closed form.
    """
    s = _clean(seq)
    N = len(s)
    if N < 2:
        return np.nan
    q = np.array([1 if c in POSITIVE else (-1 if c in NEGATIVE else 0) for c in s],
                 dtype=float)
    # Vectorized double sum using outer products
    idx = np.arange(N)
    sep = np.sqrt(np.abs(idx[:, None] - idx[None, :]))  # |j - i|^(1/2)
    qq = np.outer(q, q)
    # take strictly upper triangle (i < j)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    return float((qq * sep)[mask].sum() / N)


def shd(seq: str) -> float:
    """
    Sequence Hydropathy Decoration (Zheng et al. 2020).
    SHD = (1/N) * sum_{i<j} (h_i + h_j) * |j - i|^(-1)
    Uses normalized Kyte-Doolittle hydropathy.

    Higher SHD => hydrophobic residues clustered (drives compaction)
    """
    s = _clean(seq)
    N = len(s)
    if N < 2:
        return np.nan
    h = np.array([KD_NORM[c] for c in s], dtype=float)
    idx = np.arange(N)
    inv_sep = np.where(idx[:, None] != idx[None, :],
                        1.0 / np.maximum(np.abs(idx[:, None] - idx[None, :]), 1),
                        0.0)
    np.fill_diagonal(inv_sep, 0.0)
    h_sum = h[:, None] + h[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    return float((h_sum * inv_sep)[mask].sum() / N)


# ---- ALBATROSS-friendly null reference (AFRC) ---------------------------
# Useful for downstream analyses even though the user opted out of running
# ALBATROSS. The AFRC Rg gives us a length-aware "null model" so domains of
# different sizes can be compared on a common axis if desired later.

def afrc_rg(length: int) -> float:
    """Analytical Flory random coil Rg (Alston et al. 2023). Approximation:
    Rg = 2.49 * N^0.5 (Angstroms). Useful as a length-normalized reference."""
    if length is None or length < 1:
        return np.nan
    return 2.49 * float(length) ** 0.5


# ---- registry ------------------------------------------------------------
# Maps a human-readable property name -> (callable, category)
# Used by the driver to compute everything in one pass.

SEQUENCE_PROPERTIES = {
    # Composition
    "FCR":              (fcr,                       "composition"),
    "NCPR":             (ncpr,                      "composition"),
    "abs_NCPR":         (lambda s: abs(ncpr(s)),    "composition"),
    "frac_aromatic":    (fraction_aromatic,         "composition"),
    "frac_R":           (fraction_R,                "composition"),
    "frac_K":           (fraction_K,                "composition"),
    "frac_D":           (fraction_D,                "composition"),
    "frac_E":           (fraction_E,                "composition"),
    "frac_positive":    (fraction_positive,         "composition"),
    "frac_negative":    (fraction_negative,         "composition"),
    "frac_proline":     (fraction_proline,          "composition"),
    "frac_glycine":     (fraction_glycine,          "composition"),
    "frac_hydrophobic": (fraction_hydrophobic,      "composition"),
    "frac_polar":       (fraction_polar,            "composition"),
    "mean_hydropathy":  (mean_hydropathy,           "composition"),
    # Patterning
    "kappa":            (kappa,                     "patterning"),
    "omega":            (omega,                     "patterning"),
    "SCD":              (scd,                       "patterning"),
    "SHD":              (shd,                       "patterning"),
}


def compute_all(seq: str) -> dict:
    """Compute every registered sequence property for one sequence."""
    return {name: fn(seq) for name, (fn, _cat) in SEQUENCE_PROPERTIES.items()}
