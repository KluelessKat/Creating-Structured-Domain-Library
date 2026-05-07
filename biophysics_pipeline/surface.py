"""
surface.py
==========

Surface-exposed property calculations using locally cached AlphaFold PDB files.

For each protein we compute per-residue SASA (solvent accessible surface area)
with freesasa, classify residues as "surface" if their relative SASA exceeds a
threshold (default 0.20, a common cutoff), then compute compositional metrics
restricted to those surface residues.

Outputs (when called on a full protein PDB):
    - surface_FCR, surface_NCPR, surface_abs_NCPR
    - surface_frac_aromatic, surface_frac_positive, surface_frac_negative

For domain-level surface metrics, this module supports two modes:
    1. "in_context" — use the full protein PDB and restrict residue indices to
       the domain's start..end. This reflects the surface the domain presents
       in the context of the full protein.
    2. "isolated" — use a domain-only PDB (excised, see your existing
       3_alphaFoldDomainInteractions.exciseDomainPDB). This reflects surface
       in isolation, ignoring intra-protein burial.

Both are useful and tell different stories. The driver records "in_context"
by default since that's what the cell actually sees.
"""

from __future__ import annotations
import os
import numpy as np

try:
    import freesasa
    from Bio import PDB
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

# Maximum SASA (Tien et al. 2013, "empirical" theoretical values, Å^2).
# Used to compute relative SASA = SASA / max_SASA.
MAX_SASA_TIEN = {
    'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 'C': 167.0,
    'Q': 225.0, 'E': 223.0, 'G':  104.0, 'H': 224.0, 'I': 197.0,
    'L': 201.0, 'K': 236.0, 'M': 224.0, 'F': 240.0, 'P': 159.0,
    'S': 155.0, 'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0,
}

# Three-letter -> one-letter
THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}

POSITIVE = set("RK")
NEGATIVE = set("DE")
AROMATIC = set("FWY")


def _per_residue_sasa(pdb_file: str):
    """Return (sequence_str, per_residue_sasa_array, residue_numbers_array).
    Residue numbers come from the PDB and may not be contiguous."""
    if not _DEPS_OK:
        raise ImportError("freesasa and biopython are required for surface metrics")

    structure = freesasa.Structure(pdb_file)
    result = freesasa.calc(structure)
    residue_areas = result.residueAreas()  # dict: chain -> dict of resnum_str -> ResidueArea

    # AlphaFold structures have a single chain
    if not residue_areas:
        return "", np.array([]), np.array([])
    chain_id = next(iter(residue_areas.keys()))
    chain_residues = residue_areas[chain_id]

    seq_chars = []
    sasa_vals = []
    res_nums = []
    # Sort by integer residue number
    for resnum_str in sorted(chain_residues.keys(), key=lambda x: int(x)):
        ra = chain_residues[resnum_str]
        aa3 = ra.residueType
        if aa3 not in THREE_TO_ONE:
            continue
        aa1 = THREE_TO_ONE[aa3]
        seq_chars.append(aa1)
        sasa_vals.append(ra.total)
        res_nums.append(int(resnum_str))

    return "".join(seq_chars), np.array(sasa_vals), np.array(res_nums)


def _surface_mask(seq: str, sasa: np.ndarray, threshold: float = 0.20) -> np.ndarray:
    """Boolean mask of length len(seq), True where residue is surface-exposed."""
    rel = np.array([
        sasa[i] / MAX_SASA_TIEN.get(seq[i], 200.0) if i < len(sasa) else 0.0
        for i in range(len(seq))
    ])
    return rel >= threshold


def _comp_metrics(residues: str) -> dict:
    """Compositional metrics for a residue list (already filtered to surface)."""
    n = len(residues)
    if n == 0:
        return {
            "surface_FCR": np.nan,
            "surface_NCPR": np.nan,
            "surface_abs_NCPR": np.nan,
            "surface_frac_aromatic": np.nan,
            "surface_frac_positive": np.nan,
            "surface_frac_negative": np.nan,
            "n_surface_residues": 0,
        }
    pos = sum(1 for c in residues if c in POSITIVE)
    neg = sum(1 for c in residues if c in NEGATIVE)
    aro = sum(1 for c in residues if c in AROMATIC)
    return {
        "surface_FCR":           (pos + neg) / n,
        "surface_NCPR":          (pos - neg) / n,
        "surface_abs_NCPR":      abs(pos - neg) / n,
        "surface_frac_aromatic": aro / n,
        "surface_frac_positive": pos / n,
        "surface_frac_negative": neg / n,
        "n_surface_residues":    n,
    }


def surface_metrics_full_protein(pdb_file: str, threshold: float = 0.20) -> dict:
    """Compute surface metrics for a full-protein AlphaFold PDB."""
    seq, sasa, _ = _per_residue_sasa(pdb_file)
    if len(seq) == 0:
        return _comp_metrics("")
    mask = _surface_mask(seq, sasa, threshold)
    return _comp_metrics("".join(c for c, m in zip(seq, mask) if m))


def surface_metrics_domain_in_context(
    pdb_file: str, domain_start: int, domain_end: int, threshold: float = 0.20
) -> dict:
    """
    Surface metrics for a domain in the context of the full protein.

    domain_start and domain_end are 1-indexed inclusive (UniProt convention).
    """
    seq, sasa, resnums = _per_residue_sasa(pdb_file)
    if len(seq) == 0:
        return _comp_metrics("")
    # Restrict to residues whose PDB residue number lies in [start, end]
    in_domain = (resnums >= domain_start) & (resnums <= domain_end)
    dom_seq = "".join(c for c, m in zip(seq, in_domain) if m)
    dom_sasa = sasa[in_domain]
    if len(dom_seq) == 0:
        return _comp_metrics("")
    mask = _surface_mask(dom_seq, dom_sasa, threshold)
    return _comp_metrics("".join(c for c, m in zip(dom_seq, mask) if m))


# Convenience for batched runs --------------------------------------------

def af_pdb_path(uniprot_id: str, pdb_dir: str, version: int = 6) -> str | None:
    """Standard AlphaFold filename convention used in the user's pipeline."""
    p = os.path.join(pdb_dir, f"{uniprot_id}_model.pdb")
    if os.path.exists(p):
        return p
    # also try the AlphaFold direct naming convention as fallback
    p2 = os.path.join(pdb_dir, f"AF-{uniprot_id}-F1-model_v{version}.pdb")
    if os.path.exists(p2):
        return p2
    return None
