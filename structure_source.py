#!/usr/bin/env python3
"""
structure_source.py

Resolves the best available structural model for a protein domain, choosing
between experimentally-determined PDB structures and AlphaFold predictions.

Used by:
    - 3_alphaFoldDomainInteractions.py (interaction metrics)
    - 4_physicalPropertyDomainStruct.py (physical properties)

Public entry point
------------------
    resolve_domain_structure(uniprot_id, domain_start, domain_end,
                             cache_dir, mode="experimental_preferred",
                             min_domain_coverage=0.8)
        -> StructureChoice

Modes
-----
    experimental_only        Only use a published PDB. If none covers the
                             domain (with min_domain_coverage), return a
                             StructureChoice with .available == False.
    experimental_preferred   Use the best published PDB if available,
                             otherwise fall back to AlphaFold.
    alphafold_only           Always use AlphaFold (current pipeline default).

The returned StructureChoice tells the caller which file path to use, what
the source is, and (when relevant) AF<->PDB similarity for trust calibration.

Caveats
-------
- Experimental PDBs are usually published with PDB residue numbering that
  may differ from UniProt numbering. We always renumber the saved PDB to
  UniProt numbering using the SIFTS mapping, so callers can use the same
  [domain_start, domain_end] range regardless of source.
- Experimental PDBs often contain multiple chains/copies. We pick the chain
  whose UniProt mapping covers the most of the requested domain, and write
  out only that chain.
- TM-score validation requires `tmtools`. If unavailable, the helper still
  works but tm_score will be None.

Network endpoints used
----------------------
    SIFTS best_structures:  https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot}
    SIFTS uniprot mapping:  https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}
    RCSB PDB file:          https://files.rcsb.org/download/{pdb_id}.pdb
    AlphaFoldDB PDB:        https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v6.pdb
    AlphaFoldDB PAE:        https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-predicted_aligned_error_v6.json
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from Bio import PDB


# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------

ALPHAFOLD_VERSION = "v6"  # bump if AlphaFoldDB ever serves a newer model

# Method ranking — higher is better. Used in ranking PDB candidates.
METHOD_RANK = {
    "X-ray diffraction": 3,
    "Electron Microscopy": 3,
    "Solution NMR": 1,
    "Solid-state NMR": 1,
    "Neutron Diffraction": 2,
}

# Cache TTL for SIFTS lookups (seconds). 7 days is plenty for a research run.
SIFTS_CACHE_TTL = 7 * 24 * 3600


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class StructureChoice:
    """The resolved structure for one domain.

    Attributes
    ----------
    uniprot_id : str
    domain_start, domain_end : int      UniProt-numbered residue range
    available : bool                    False means no usable structure found
    source : str                        "experimental" | "alphafold" | "none"
    pdb_path : Optional[Path]           file path of the picked structure,
                                        renumbered to UniProt residues
    full_protein_pdb_path : Optional[Path]
                                        AlphaFold full-protein file (always
                                        downloaded when available, used by
                                        interaction metrics regardless of
                                        primary source)
    pae_path : Optional[Path]           AlphaFold PAE JSON (anchoringIndex)
    pdb_id : Optional[str]              e.g. "7MRJ" (when experimental)
    chain_id : Optional[str]            chain selected from the PDB
    resolution : Optional[float]        Å (None for NMR / AF)
    method : Optional[str]              experimental method
    domain_coverage : Optional[float]   fraction of domain residues present
                                        in the picked structure
    domain_purity : Optional[float]     fraction of structure residues that
                                        are within the domain range
    tm_score : Optional[float]          TM-score AF vs experimental (when both
                                        exist and tmtools available)
    tm_flag : Optional[str]             "ok" / "low_similarity" / "unavailable"
    notes : list[str]                   anything worth flagging
    """
    uniprot_id: str
    domain_start: int
    domain_end: int
    available: bool = False
    source: str = "none"
    pdb_path: Optional[Path] = None
    full_protein_pdb_path: Optional[Path] = None
    pae_path: Optional[Path] = None
    pdb_id: Optional[str] = None
    chain_id: Optional[str] = None
    resolution: Optional[float] = None
    method: Optional[str] = None
    domain_coverage: Optional[float] = None
    domain_purity: Optional[float] = None
    tm_score: Optional[float] = None
    tm_flag: Optional[str] = None
    notes: list = field(default_factory=list)

    def to_row(self) -> dict:
        """Flat dict suitable for adding to a pandas DataFrame row."""
        return {
            "structureSource": self.source,
            "pdbID": self.pdb_id,
            "pdbChain": self.chain_id,
            "pdbResolution": self.resolution,
            "pdbMethod": self.method,
            "domainCoverage": self.domain_coverage,
            "domainPurity": self.domain_purity,
            "afPdbTMScore": self.tm_score,
            "afPdbTMFlag": self.tm_flag,
            "structureNotes": "; ".join(self.notes) if self.notes else None,
        }


# ---------------------------------------------------------------------------
# HTTP with retries
# ---------------------------------------------------------------------------

def _get_with_retry(url: str, max_tries: int = 3, timeout: int = 30,
                    stream: bool = False) -> Optional[requests.Response]:
    """GET with simple exponential backoff. Returns None on persistent
    failure; the caller decides whether that's fatal."""
    for attempt in range(1, max_tries + 1):
        try:
            r = requests.get(url, timeout=timeout, stream=stream)
            if r.status_code == 404:
                return r  # caller distinguishes 404 from network error
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == max_tries:
                warnings.warn(f"GET failed after {max_tries} tries: {url} ({e})")
                return None
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_dir(cache_dir: Path) -> Path:
    cache_dir = Path(cache_dir)
    (cache_dir / "alphafold").mkdir(parents=True, exist_ok=True)
    (cache_dir / "experimental").mkdir(parents=True, exist_ok=True)
    (cache_dir / "sifts").mkdir(parents=True, exist_ok=True)
    return cache_dir


def _sifts_cached(cache_dir: Path, key: str) -> Optional[dict]:
    p = cache_dir / "sifts" / f"{key}.json"
    if not p.exists():
        return None
    if (time.time() - p.stat().st_mtime) > SIFTS_CACHE_TTL:
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _sifts_cache_write(cache_dir: Path, key: str, data) -> None:
    p = cache_dir / "sifts"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{key}.json").write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# AlphaFold downloads (mirror of script 3's downloader, hardened)
# ---------------------------------------------------------------------------

def download_alphafold(uniprot_id: str, cache_dir: Path
                       ) -> tuple[Optional[Path], Optional[Path]]:
    """Download AF PDB + PAE for a UniProt accession. Returns (pdb_path,
    pae_path); either may be None if the download failed (e.g. AF has no
    prediction for this protein)."""
    cache_dir = _cache_dir(cache_dir)
    af_dir = cache_dir / "alphafold"
    pdb_path = af_dir / f"{uniprot_id}_model.pdb"
    pae_path = af_dir / f"{uniprot_id}_PAE.json"

    if not pdb_path.exists():
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_{ALPHAFOLD_VERSION}.pdb"
        r = _get_with_retry(url, stream=True)
        if r is None or r.status_code == 404:
            pdb_path = None
        else:
            with open(af_dir / f"{uniprot_id}_model.pdb", "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

    if not pae_path.exists():
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-predicted_aligned_error_{ALPHAFOLD_VERSION}.json"
        r = _get_with_retry(url, stream=True)
        if r is None or r.status_code == 404:
            pae_path = None
        else:
            with open(af_dir / f"{uniprot_id}_PAE.json", "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

    return (pdb_path if pdb_path and pdb_path.exists() else None,
            pae_path if pae_path and pae_path.exists() else None)


# ---------------------------------------------------------------------------
# SIFTS / RCSB lookups
# ---------------------------------------------------------------------------

def list_pdbs_for_uniprot(uniprot_id: str, cache_dir: Path) -> list[dict]:
    """Use the PDBe SIFTS API to list all PDB structures for a UniProt
    accession with their UniProt residue mappings, resolution, and method.
    Returns a list of dicts with keys: pdb_id, chain_id, unp_start, unp_end,
    pdb_start, pdb_end, resolution, method, coverage."""
    cache_dir = _cache_dir(cache_dir)
    cached = _sifts_cached(cache_dir, f"best_{uniprot_id}")
    if cached is not None:
        return cached

    # best_structures returns a ranked list of PDBs with unp ranges per chain.
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot_id}"
    r = _get_with_retry(url)
    if r is None or r.status_code == 404:
        _sifts_cache_write(cache_dir, f"best_{uniprot_id}", [])
        return []

    try:
        payload = r.json()
    except json.JSONDecodeError:
        return []

    entries = payload.get(uniprot_id, [])
    out = []
    for e in entries:
        # Each entry covers one chain's mapping to UniProt.
        out.append({
            "pdb_id": e.get("pdb_id"),
            "chain_id": e.get("chain_id"),
            "unp_start": e.get("unp_start"),
            "unp_end": e.get("unp_end"),
            "pdb_start": e.get("start"),
            "pdb_end": e.get("end"),
            "resolution": e.get("resolution"),
            "method": e.get("experimental_method"),
            "coverage": e.get("coverage"),
        })
    _sifts_cache_write(cache_dir, f"best_{uniprot_id}", out)
    return out


def get_pdb_full_uniprot_mapping(pdb_id: str, cache_dir: Path) -> dict:
    """Fetch all (chain, UniProt range, PDB range) mappings for a PDB ID.
    Used to compute the precise renumbering offset and to detect
    auth-vs-label residue numbering. Returns {chain_id: {unp_start, unp_end,
    pdb_start, pdb_end, pdb_chain_id}}."""
    pdb_id = pdb_id.lower()
    cache_dir = _cache_dir(cache_dir)
    cached = _sifts_cached(cache_dir, f"unpmap_{pdb_id}")
    if cached is not None:
        return cached

    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    r = _get_with_retry(url)
    if r is None or r.status_code == 404:
        _sifts_cache_write(cache_dir, f"unpmap_{pdb_id}", {})
        return {}

    try:
        payload = r.json()
    except json.JSONDecodeError:
        return {}

    out: dict = {}
    entry = payload.get(pdb_id, {}).get("UniProt", {})
    for unp_acc, info in entry.items():
        for mapping in info.get("mappings", []):
            chain = mapping.get("chain_id")
            if chain not in out:
                out[chain] = []
            out[chain].append({
                "uniprot": unp_acc,
                "unp_start": mapping.get("unp_start"),
                "unp_end": mapping.get("unp_end"),
                "pdb_start": (mapping.get("start") or {}).get("residue_number"),
                "pdb_end": (mapping.get("end") or {}).get("residue_number"),
                "pdb_auth_start": (mapping.get("start") or {}).get("author_residue_number"),
                "pdb_auth_end": (mapping.get("end") or {}).get("author_residue_number"),
                "pdb_chain_id": chain,
            })
    _sifts_cache_write(cache_dir, f"unpmap_{pdb_id}", out)
    return out


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _domain_overlap(unp_start: int, unp_end: int,
                    dstart: int, dend: int) -> int:
    """Number of residues in [unp_start..unp_end] that lie within [dstart..dend]."""
    if unp_start is None or unp_end is None:
        return 0
    lo = max(unp_start, dstart)
    hi = min(unp_end, dend)
    return max(0, hi - lo + 1)


def rank_pdb_candidates(uniprot_id: str, dstart: int, dend: int,
                        cache_dir: Path, min_domain_coverage: float = 0.8
                        ) -> list[dict]:
    """Find PDB candidates covering the domain, score each, return ranked.

    Score components per (pdb_id, chain) candidate:
        domain_coverage = domain_residues_in_pdb / domain_length
        domain_purity   = domain_residues_in_pdb / total_residues_in_chain
        method_rank     = 3 (X-ray/EM) > 2 > 1 (NMR)
        resolution      = lower is better (None for NMR -> sort last)

    Returns a list sorted best-first. Each dict has the original SIFTS
    fields plus 'domain_coverage', 'domain_purity', 'rank_score'.
    """
    domain_length = dend - dstart + 1
    candidates = list_pdbs_for_uniprot(uniprot_id, cache_dir)

    scored = []
    for c in candidates:
        unp_start = c.get("unp_start")
        unp_end = c.get("unp_end")
        if unp_start is None or unp_end is None:
            continue

        overlap = _domain_overlap(unp_start, unp_end, dstart, dend)
        if overlap == 0:
            continue

        chain_length = unp_end - unp_start + 1
        coverage = overlap / domain_length
        purity = overlap / max(chain_length, 1)

        if coverage < min_domain_coverage:
            continue

        method = c.get("method") or ""
        method_rank = METHOD_RANK.get(method, 0)
        # Some method strings vary slightly; do a lenient check.
        if method_rank == 0:
            mlow = method.lower()
            if "x-ray" in mlow or "electron" in mlow:
                method_rank = 3
            elif "nmr" in mlow:
                method_rank = 1
            elif "neutron" in mlow:
                method_rank = 2

        resolution = c.get("resolution")
        # For NMR, resolution is None — replace with a sentinel that ranks
        # them after low-resolution X-ray but before failures.
        res_for_sort = resolution if resolution is not None else 99.0

        c2 = dict(c)
        c2["domain_coverage"] = coverage
        c2["domain_purity"] = purity
        c2["method_rank"] = method_rank
        c2["res_for_sort"] = res_for_sort
        scored.append(c2)

    # Sort: highest purity first, then lowest resolution, then highest method rank,
    # then highest coverage as a final tiebreaker.
    scored.sort(key=lambda x: (-x["domain_purity"],
                                x["res_for_sort"],
                                -x["method_rank"],
                                -x["domain_coverage"]))
    return scored


# ---------------------------------------------------------------------------
# PDB downloading + per-domain extraction
# ---------------------------------------------------------------------------

def download_rcsb_pdb(pdb_id: str, cache_dir: Path) -> Optional[Path]:
    """Download a PDB file from RCSB. Returns the local path, or None on failure."""
    cache_dir = _cache_dir(cache_dir)
    out = cache_dir / "experimental" / f"{pdb_id.upper()}.pdb"
    if out.exists():
        return out
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    r = _get_with_retry(url, stream=True)
    if r is None or r.status_code == 404:
        return None
    with open(out, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return out


def extract_chain_renumbered(pdb_path: Path, chain_id: str, cache_dir: Path,
                             pdb_id: str, uniprot_id: str,
                             dstart: int, dend: int,
                             out_path: Path) -> Optional[Path]:
    """Extract one chain from a PDB and renumber its residues to UniProt
    numbering. Restricts to the [dstart, dend] domain range. Saves to
    out_path and returns it. Returns None if the chain or mapping can't be
    resolved.

    PDBe SIFTS gives us the mapping in residue-pair form (unp_start ->
    pdb_start, unp_end -> pdb_end). For most cases the offset is constant
    within a chain, so we apply: new_resnum = old_resnum + (unp_start - pdb_auth_start).
    Where the mapping has gaps (rare, but possible for chimeras or fusion
    tags), residues outside any mapping segment are dropped.
    """
    if out_path.exists():
        return out_path

    mapping = get_pdb_full_uniprot_mapping(pdb_id, cache_dir).get(chain_id)
    if not mapping:
        return None
    # Filter to mappings for this UniProt
    mapping = [m for m in mapping if m["uniprot"] == uniprot_id]
    if not mapping:
        return None

    # Build per-pdb-residue -> uniprot-residue lookup using all mapping segments.
    # PDB files use auth residue numbers; SIFTS gives both 'residue_number'
    # (label_seq_id) and 'author_residue_number'. We use auth numbers since
    # that's what the PDB file indexes by.
    def _resnum_to_uniprot(pdb_auth_resnum: int) -> Optional[int]:
        for seg in mapping:
            ps = seg.get("pdb_auth_start")
            pe = seg.get("pdb_auth_end")
            us = seg.get("unp_start")
            if ps is None or pe is None or us is None:
                continue
            if ps <= pdb_auth_resnum <= pe:
                return us + (pdb_auth_resnum - ps)
        return None

    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("s", str(pdb_path))
    except Exception as e:
        warnings.warn(f"Failed to parse {pdb_path}: {e}")
        return None

    model = structure[0]
    if chain_id not in [c.id for c in model.get_chains()]:
        return None
    src_chain = model[chain_id]

    # Build new structure with renumbered residues, restricted to the domain.
    new_structure = PDB.Structure.Structure("ext")
    new_model = PDB.Model.Model(0)
    new_chain = PDB.Chain.Chain(chain_id)
    new_model.add(new_chain)
    new_structure.add(new_model)

    kept = 0
    for residue in list(src_chain):
        # Only keep standard residues (skip waters, ligands, hetatms).
        hetflag, resseq, icode = residue.id
        if hetflag.strip() and hetflag != " ":
            continue
        unp_resnum = _resnum_to_uniprot(resseq)
        if unp_resnum is None:
            continue
        if not (dstart <= unp_resnum <= dend):
            continue
        new_residue = residue.copy()
        new_residue.id = (" ", unp_resnum, " ")
        new_chain.add(new_residue)
        kept += 1

    if kept == 0:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Optional: TM-score AF vs experimental
# ---------------------------------------------------------------------------

def compute_af_pdb_tm(af_pdb: Path, exp_pdb: Path,
                      dstart: int, dend: int) -> Optional[float]:
    """Compute TM-score between the AF prediction (restricted to the domain)
    and the experimental structure (already domain-restricted, UniProt-numbered).

    Requires `tmtools`. If not available, returns None silently.
    """
    try:
        from tmtools import tm_align
        from tmtools.io import get_residue_data
    except ImportError:
        return None

    parser = PDB.PDBParser(QUIET=True)
    try:
        af = parser.get_structure("af", str(af_pdb))
        ex = parser.get_structure("ex", str(exp_pdb))
    except Exception:
        return None

    # Slice AF down to domain residues
    def _coords_seq(structure, restrict_range=None):
        coords, seq = [], ""
        chain = next(structure[0].get_chains())
        for res in chain:
            hetflag, resseq, _ = res.id
            if hetflag.strip() and hetflag != " ":
                continue
            if restrict_range is not None:
                lo, hi = restrict_range
                if not (lo <= resseq <= hi):
                    continue
            if "CA" not in res:
                continue
            coords.append(res["CA"].get_coord())
            try:
                aa = PDB.Polypeptide.three_to_one(res.get_resname())
            except Exception:
                aa = "X"
            seq += aa
        return np.array(coords), seq

    af_coords, af_seq = _coords_seq(af, restrict_range=(dstart, dend))
    ex_coords, ex_seq = _coords_seq(ex)

    if len(af_coords) < 5 or len(ex_coords) < 5:
        return None

    try:
        result = tm_align(af_coords, ex_coords, af_seq, ex_seq)
        # Use the larger TM-score (chain1 or chain2 normalization).
        # For our purposes — "do these represent the same fold?" — max is fine.
        return float(max(result.tm_norm_chain1, result.tm_norm_chain2))
    except Exception as e:
        warnings.warn(f"tm_align failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def resolve_domain_structure(uniprot_id: str,
                             dstart: int,
                             dend: int,
                             cache_dir: Path,
                             mode: str = "experimental_preferred",
                             min_domain_coverage: float = 0.8,
                             tm_low_threshold: float = 0.7,
                             ) -> StructureChoice:
    """Pick the best structure for a domain.

    Parameters
    ----------
    uniprot_id : str
    dstart, dend : int           UniProt residue range (1-indexed, inclusive)
    cache_dir : Path             local cache root for downloads + sifts
    mode : str                   "experimental_only" | "experimental_preferred" | "alphafold_only"
    min_domain_coverage : float  fraction of domain that must be present in
                                 a candidate PDB to accept it (default 0.8)
    tm_low_threshold : float     TM-score below this triggers "low_similarity"
                                 flag in the output

    Returns
    -------
    StructureChoice
    """
    if mode not in ("experimental_only", "experimental_preferred", "alphafold_only"):
        raise ValueError(f"Unknown mode: {mode}")

    cache_dir = _cache_dir(cache_dir)
    choice = StructureChoice(uniprot_id=uniprot_id,
                             domain_start=dstart, domain_end=dend)

    # Always try to get AF — needed for PAE, anchoringIndex, full-protein
    # context, and TM-validation. Skipping AF entirely is reserved for
    # downstream "save bandwidth" knobs we're not exposing here.
    af_pdb_path, pae_path = download_alphafold(uniprot_id, cache_dir)
    choice.full_protein_pdb_path = af_pdb_path
    choice.pae_path = pae_path

    # --- Try experimental first if mode allows -----------------------------
    picked_exp = None
    if mode in ("experimental_only", "experimental_preferred"):
        ranked = rank_pdb_candidates(uniprot_id, dstart, dend, cache_dir,
                                     min_domain_coverage=min_domain_coverage)
        for cand in ranked:
            pdb_id = cand["pdb_id"]
            chain_id = cand["chain_id"]
            raw_pdb = download_rcsb_pdb(pdb_id, cache_dir)
            if raw_pdb is None:
                continue
            ext_out = (cache_dir / "experimental" /
                       f"{uniprot_id}_{pdb_id}_{chain_id}_dom_{dstart}_{dend}.pdb")
            extracted = extract_chain_renumbered(
                raw_pdb, chain_id, cache_dir, pdb_id, uniprot_id,
                dstart, dend, ext_out)
            if extracted is None:
                continue
            picked_exp = (cand, extracted)
            break  # ranked best-first; stop at first success

    # --- Decide what to return ---------------------------------------------
    if picked_exp is not None:
        cand, extracted = picked_exp
        choice.available = True
        choice.source = "experimental"
        choice.pdb_path = extracted
        choice.pdb_id = cand["pdb_id"]
        choice.chain_id = cand["chain_id"]
        choice.resolution = cand.get("resolution")
        choice.method = cand.get("method")
        choice.domain_coverage = cand.get("domain_coverage")
        choice.domain_purity = cand.get("domain_purity")

        # TM-score validation (when AF also available)
        if af_pdb_path is not None:
            tm = compute_af_pdb_tm(af_pdb_path, extracted, dstart, dend)
            if tm is None:
                choice.tm_flag = "unavailable"
            else:
                choice.tm_score = tm
                choice.tm_flag = "low_similarity" if tm < tm_low_threshold else "ok"
        else:
            choice.tm_flag = "unavailable"
            choice.notes.append("AF prediction unavailable; cannot compute TM-score")

        return choice

    # No experimental — decide based on mode.
    if mode == "experimental_only":
        choice.available = False
        choice.source = "none"
        choice.notes.append("No experimental PDB covers the domain "
                            f"with coverage >= {min_domain_coverage}; "
                            "row will be dropped per experimental_only mode.")
        return choice

    # alphafold_only or experimental_preferred (with no exp available)
    if af_pdb_path is None:
        choice.available = False
        choice.source = "none"
        choice.notes.append("AlphaFold prediction unavailable.")
        return choice

    choice.available = True
    choice.source = "alphafold"
    choice.pdb_path = af_pdb_path  # the full-protein AF file IS the source
    return choice


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Smoke-test structure_source.")
    ap.add_argument("uniprot_id")
    ap.add_argument("dstart", type=int)
    ap.add_argument("dend", type=int)
    ap.add_argument("--cache", type=Path, default=Path("./struct_cache"))
    ap.add_argument("--mode", default="experimental_preferred",
                    choices=["experimental_only", "experimental_preferred",
                             "alphafold_only"])
    args = ap.parse_args()
    res = resolve_domain_structure(args.uniprot_id, args.dstart, args.dend,
                                   args.cache, mode=args.mode)
    import pprint
    pprint.pprint(asdict(res))


if __name__ == "__main__":
    _cli()
