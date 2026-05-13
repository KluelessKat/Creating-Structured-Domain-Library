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
# AlphaFold file resolution (local cache + downloads, multi-fragment aware)
# ---------------------------------------------------------------------------

# Regex to recognize AF filenames in any of the standard forms:
#   AF-{uniprot}-F{n}-model_v6.pdb
#   AF-{uniprot}-F{n}-model_v6.pdb.gz
#   AF-{uniprot}-F{n}-predicted_aligned_error_v6.json(.gz)
import re as _re
_AF_FILE_RE = _re.compile(
    r"^AF-([A-Z0-9]+)-F(\d+)-(model|predicted_aligned_error)_(v\d+)\."
    r"(pdb|cif|json)(\.gz)?$"
)


def _list_local_af_fragments(uniprot_id: str, local_dir: Path,
                             kind: str) -> dict[int, Path]:
    """Return {fragment_number: path} for AF files matching this UniProt and
    kind ('pdb' or 'pae'). Searches `local_dir` non-recursively. Files may
    be either gzipped or not.
    """
    if local_dir is None or not Path(local_dir).is_dir():
        return {}
    expected_ftype = "model" if kind == "pdb" else "predicted_aligned_error"
    expected_ext = "pdb" if kind == "pdb" else "json"
    out: dict[int, Path] = {}
    for entry in Path(local_dir).iterdir():
        m = _AF_FILE_RE.match(entry.name)
        if not m:
            continue
        upid, fnum, ftype, _ver, ext, _gz = m.groups()
        if upid != uniprot_id:
            continue
        if ftype != expected_ftype:
            continue
        if ext != expected_ext:
            continue
        # If we already have a non-gz version, prefer it; otherwise take whatever.
        existing = out.get(int(fnum))
        if existing is None or (existing.suffix == ".gz" and not entry.name.endswith(".gz")):
            out[int(fnum)] = entry
    return out


def _read_pdb_residue_range(pdb_or_gz: Path) -> Optional[tuple[int, int]]:
    """Quickly scan a PDB file's ATOM lines and return (min_resnum, max_resnum)
    for the first chain. Used to figure out which fragment of a multi-fragment
    AF protein contains a given domain. Handles gzipped files transparently.
    """
    import gzip
    opener = gzip.open if str(pdb_or_gz).endswith(".gz") else open
    lo = None
    hi = None
    try:
        with opener(pdb_or_gz, "rt") as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                # PDB residue number is columns 23-26 (1-indexed, 4 chars)
                try:
                    resnum = int(line[22:26])
                except ValueError:
                    continue
                if lo is None or resnum < lo:
                    lo = resnum
                if hi is None or resnum > hi:
                    hi = resnum
    except (OSError, EOFError) as e:
        warnings.warn(f"Could not read {pdb_or_gz}: {e}")
        return None
    if lo is None or hi is None:
        return None
    return (lo, hi)


def _decompress_to_cache(src: Path, dst: Path) -> Path:
    """Copy/decompress an AF file into the cache. dst should NOT have .gz."""
    import gzip, shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    if str(src).endswith(".gz"):
        with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    else:
        shutil.copyfile(src, dst)
    return dst


def _download_af_fragment(uniprot_id: str, fragment: int, kind: str,
                          out_path: Path) -> Optional[Path]:
    """Download one AF file (model PDB or PAE JSON) for a specific fragment.
    Returns the local path on success, None on failure."""
    if kind == "pdb":
        url = (f"https://alphafold.ebi.ac.uk/files/"
               f"AF-{uniprot_id}-F{fragment}-model_{ALPHAFOLD_VERSION}.pdb")
    elif kind == "pae":
        url = (f"https://alphafold.ebi.ac.uk/files/"
               f"AF-{uniprot_id}-F{fragment}-predicted_aligned_error_{ALPHAFOLD_VERSION}.json")
    else:
        raise ValueError(f"kind must be 'pdb' or 'pae', got {kind!r}")
    r = _get_with_retry(url, stream=True)
    if r is None or r.status_code == 404:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return out_path


def _pick_best_fragment(uniprot_id: str,
                        local_dir: Optional[Path],
                        cache_dir: Path,
                        dstart: int, dend: int,
                        allow_download: bool = True
                        ) -> Optional[int]:
    """Decide which fragment number contains the domain.

    Strategy:
      1. If we have local fragments, pick the one whose residue range
         covers [dstart, dend] best.
      2. If no local fragments, default to F1 (correct for ~99% of human
         proteins, which fit in a single fragment).
      3. If local has F1 only and the domain end > residue range of F1,
         that means there are more fragments we don't have. Try F2, F3...
         by downloading and probing the residue ranges (rare path).

    Returns the fragment number, or None if nothing usable can be found.
    """
    local = _list_local_af_fragments(uniprot_id, local_dir, "pdb") if local_dir else {}

    if local:
        # Score each fragment by overlap with the domain
        best_frag = None
        best_overlap = -1
        for fnum, fpath in local.items():
            r = _read_pdb_residue_range(fpath)
            if r is None:
                continue
            lo, hi = r
            overlap = max(0, min(hi, dend) - max(lo, dstart) + 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_frag = fnum
        if best_frag is not None and best_overlap > 0:
            return best_frag
        # If no local fragment covers the domain, fall through to download.

    if not allow_download:
        return None

    # No local fragment covers the domain (or we have nothing locally).
    # Try F1 first — covers ~99% of proteins.
    cache_dir = _cache_dir(cache_dir)
    af_cache = cache_dir / "alphafold"
    f1_pdb = af_cache / f"{uniprot_id}_F1_model.pdb"
    if not f1_pdb.exists():
        if _download_af_fragment(uniprot_id, 1, "pdb", f1_pdb) is None:
            return None
    r = _read_pdb_residue_range(f1_pdb)
    if r is None:
        return None
    lo, hi = r
    if lo <= dstart and dend <= hi:
        return 1
    # Domain extends beyond F1. Probe further fragments.
    for fnum in range(2, 50):  # AF caps at ~25 fragments for the longest proteins
        fpdb = af_cache / f"{uniprot_id}_F{fnum}_model.pdb"
        if not fpdb.exists():
            if _download_af_fragment(uniprot_id, fnum, "pdb", fpdb) is None:
                return None  # ran out of fragments
        r = _read_pdb_residue_range(fpdb)
        if r is None:
            continue
        lo, hi = r
        if lo <= dstart and dend <= hi:
            return fnum
    return None


def resolve_alphafold(uniprot_id: str, cache_dir: Path,
                      dstart: int, dend: int,
                      local_alphafold_dir: Optional[Path] = None,
                      allow_download: bool = True
                      ) -> tuple[Optional[Path], Optional[Path]]:
    """Resolve the AF PDB + PAE for a domain. Picks the right fragment if
    the protein is multi-fragment. If local_alphafold_dir is given, looks
    there first (handles gzipped files); otherwise downloads.

    Returns (pdb_path, pae_path), each pointing into the cache dir as a
    decompressed file. Either may be None if unavailable.
    """
    cache_dir = _cache_dir(cache_dir)
    af_cache = cache_dir / "alphafold"

    fragment = _pick_best_fragment(uniprot_id, local_alphafold_dir, cache_dir,
                                   dstart, dend, allow_download=allow_download)
    if fragment is None:
        return None, None

    # PDB resolution — prefer local, fall back to cache, fall back to download
    pdb_dst = af_cache / f"{uniprot_id}_F{fragment}_model.pdb"
    if not pdb_dst.exists():
        local_pdbs = _list_local_af_fragments(uniprot_id, local_alphafold_dir, "pdb") \
            if local_alphafold_dir else {}
        if fragment in local_pdbs:
            _decompress_to_cache(local_pdbs[fragment], pdb_dst)
        elif allow_download:
            _download_af_fragment(uniprot_id, fragment, "pdb", pdb_dst)
    final_pdb = pdb_dst if pdb_dst.exists() else None

    # PAE — same flow
    pae_dst = af_cache / f"{uniprot_id}_F{fragment}_PAE.json"
    if not pae_dst.exists():
        local_paes = _list_local_af_fragments(uniprot_id, local_alphafold_dir, "pae") \
            if local_alphafold_dir else {}
        if fragment in local_paes:
            _decompress_to_cache(local_paes[fragment], pae_dst)
        elif allow_download:
            _download_af_fragment(uniprot_id, fragment, "pae", pae_dst)
    final_pae = pae_dst if pae_dst.exists() else None

    return final_pdb, final_pae


# Backwards-compat wrapper kept so older callers still work.
def download_alphafold(uniprot_id: str, cache_dir: Path,
                       local_alphafold_dir: Optional[Path] = None
                       ) -> tuple[Optional[Path], Optional[Path]]:
    """Deprecated: kept as a thin wrapper that picks F1. Use resolve_alphafold
    when you know the domain range — it picks the right fragment.
    """
    cache_dir = _cache_dir(cache_dir)
    return resolve_alphafold(uniprot_id, cache_dir, dstart=1, dend=1400,
                             local_alphafold_dir=local_alphafold_dir)


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


# ---------------------------------------------------------------------------
# UniProt sequence lookup (for alignment-based renumbering fallback)
# ---------------------------------------------------------------------------

# Module-level cache: {tsv_path -> {accession -> sequence}}
_TSV_SEQUENCE_CACHE: dict[str, dict[str, str]] = {}


def load_uniprot_sequences(tsv_path: Path) -> dict[str, str]:
    """Parse a UniProt-export TSV with 'Entry' and 'Sequence' columns and
    return {accession: sequence}. Cached in memory keyed by absolute path,
    so repeated calls during a batch only read the file once.

    The caller is responsible for passing this dict to resolve_domain_structure
    (via uniprot_sequences=...), so we read the file exactly once per run.
    """
    tsv_path = Path(tsv_path).resolve()
    key = str(tsv_path)
    if key in _TSV_SEQUENCE_CACHE:
        return _TSV_SEQUENCE_CACHE[key]
    if not tsv_path.exists():
        raise FileNotFoundError(f"UniProt TSV not found: {tsv_path}")
    import pandas as pd
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)
    if "Entry" not in df.columns or "Sequence" not in df.columns:
        raise ValueError(
            f"TSV missing 'Entry' or 'Sequence' column: {list(df.columns)}")
    out = {}
    for _, row in df.iterrows():
        acc = row["Entry"]
        seq = row["Sequence"]
        if pd.isna(acc) or pd.isna(seq):
            continue
        out[acc] = str(seq).strip()
    _TSV_SEQUENCE_CACHE[key] = out
    return out



# ---------------------------------------------------------------------------
# Biopython compatibility — Bio.PDB.Polypeptide.three_to_one was removed in
# recent versions, so we build our own three-letter -> one-letter map.
# ---------------------------------------------------------------------------
def _build_three_to_one() -> dict:
    """Return a dict mapping three-letter amino-acid codes (uppercase) to
    one-letter codes. Includes the 20 standard amino acids plus a few
    common non-standard residues seen in PDB files."""
    try:
        from Bio.Data.IUPACData import protein_letters_3to1_extended as src
    except ImportError:
        from Bio.Data.IUPACData import protein_letters_3to1 as src
    out = {k.upper(): v for k, v in src.items()}
    # Common modified residues mapped to their parent
    out.update({
        "MSE": "M",  # selenomethionine
        "SEC": "U",  # selenocysteine
        "PYL": "O",  # pyrrolysine
        "HYP": "P",  # hydroxyproline
        "PCA": "Q",  # pyroglutamic acid
    })
    return out


_THREE_TO_ONE = _build_three_to_one()


def _aa_three_to_one(resname: str) -> str:
    """Return one-letter code for a residue name; 'X' for unknown."""
    return _THREE_TO_ONE.get((resname or "").upper(), "X")


def extract_chain_renumbered(pdb_path: Path, chain_id: str, cache_dir: Path,
                             pdb_id: str, uniprot_id: str,
                             dstart: int, dend: int,
                             out_path: Path,
                             notes_out: Optional[list] = None,
                             uniprot_sequence: Optional[str] = None,
                             alignment_identity_threshold: float = 0.85,
                             ) -> Optional[Path]:
    """Extract one chain from a PDB and renumber its residues to UniProt
    numbering. Restricts to [dstart, dend].

    Mapping strategy
    ----------------
    1. SIFTS-based numbering (PRIMARY).
       Uses pdb_auth_start/end from SIFTS to map auth_seq_id -> uniprot
       residue. When SIFTS gives complete information (no nulls), this is
       the most direct and reliable path.
    2. Sequence alignment (FALLBACK).
       Only runs when SIFTS has null auth values that force the SIFTS path
       to guess. Local-aligns the chain sequence to the UniProt sequence
       using BLOSUM62. If the alignment passes identity + coverage checks,
       its mapping is used instead of SIFTS' inferred values. Robust to
       unmodeled N/C-terminal residues, His-tags, fusion constructs, point
       mutations.

    The chosen path is recorded in notes_out for downstream auditing.
    """
    if out_path.exists():
        return out_path

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

    # Build the (auth_resnum, AA, residue_obj) list for the chain.
    chain_residues: list[tuple[int, str, "PDB.Residue.Residue"]] = []
    for residue in src_chain:
        hetflag, auth_resnum, _icode = residue.id
        if hetflag.strip() and hetflag != " ":
            continue
        aa = _aa_three_to_one(residue.get_resname())
        chain_residues.append((auth_resnum, aa, residue))
    if not chain_residues:
        return None

    # ---- Path 1: SIFTS first ------------------------------------------------
    sifts_result = _sifts_based_renumbering(
        chain_residues, cache_dir, pdb_id, chain_id, uniprot_id)
    auth_to_unp: Optional[dict[int, int]] = None

    if sifts_result is not None:
        sifts_map, sifts_used_fallback = sifts_result
        if not sifts_used_fallback:
            # Clean SIFTS data — use it directly, no alignment needed.
            auth_to_unp = sifts_map
            if notes_out is not None:
                notes_out.append(
                    f"Renumbered via SIFTS ({pdb_id}/{chain_id}, clean).")
        else:
            # SIFTS has nulls. Try alignment as a more reliable alternative.
            if uniprot_sequence:
                aln_map = _alignment_based_renumbering(
                    chain_residues, uniprot_sequence,
                    pdb_id, chain_id, dstart, dend,
                    alignment_identity_threshold, notes_out)
                if aln_map is not None:
                    auth_to_unp = aln_map
            # If alignment didn't work, fall back to SIFTS-with-fallback.
            if auth_to_unp is None:
                auth_to_unp = sifts_map
                # _sifts_based_renumbering already appended the fallback note(s)
                # below — replay them now that we've decided to use this path.
                if notes_out is not None:
                    notes_out.append(
                        f"Used SIFTS with fallback for {pdb_id}/{chain_id} "
                        "(alignment unavailable or rejected). "
                        "Verify with TM-score.")
    else:
        # SIFTS has no mapping at all for this (pdb, chain). Try alignment.
        if uniprot_sequence:
            aln_map = _alignment_based_renumbering(
                chain_residues, uniprot_sequence,
                pdb_id, chain_id, dstart, dend,
                alignment_identity_threshold, notes_out)
            if aln_map is not None:
                auth_to_unp = aln_map
        if auth_to_unp is None:
            if notes_out is not None:
                notes_out.append(
                    f"No SIFTS mapping and alignment unavailable for "
                    f"{pdb_id}/{chain_id}; cannot renumber.")
            return None

    # ---- Build the output structure ----------------------------------------
    new_structure = PDB.Structure.Structure("ext")
    new_model = PDB.Model.Model(0)
    new_chain = PDB.Chain.Chain(chain_id)
    new_model.add(new_chain)
    new_structure.add(new_model)

    seen: set[int] = set()
    for auth_resnum, _aa, residue in chain_residues:
        unp_resnum = auth_to_unp.get(auth_resnum)
        if unp_resnum is None:
            continue
        if not (dstart <= unp_resnum <= dend):
            continue
        if unp_resnum in seen:
            continue
        seen.add(unp_resnum)
        new_residue = residue.copy()
        new_residue.id = (" ", unp_resnum, " ")
        new_chain.add(new_residue)

    if not seen:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(str(out_path))
    return out_path


def _alignment_based_renumbering(chain_residues: list,
                                 uniprot_sequence: str,
                                 pdb_id: str, chain_id: str,
                                 dstart: int, dend: int,
                                 identity_threshold: float,
                                 notes_out: Optional[list]
                                 ) -> Optional[dict[int, int]]:
    """Local-align the chain sequence to the UniProt sequence and return
    auth->uniprot map. Returns None if alignment fails or identity/coverage
    is below threshold.
    """
    chain_seq = "".join(aa for _resnum, aa, _res in chain_residues)
    domain_length = dend - dstart + 1
    # Coverage minimum: at least 20 residues OR 30% of the requested domain.
    # This rejects spurious 1-residue matches without rejecting fusion
    # constructs (where the matched region is small relative to the chain
    # but covers the requested domain).
    min_aligned = max(20, int(0.3 * domain_length))

    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError:
        if notes_out is not None:
            notes_out.append(
                "Bio.Align unavailable; cannot do alignment-based mapping.")
        return None

    try:
        aligner = PairwiseAligner()
        aligner.mode = "local"
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -1
        alns = aligner.align(uniprot_sequence, chain_seq)
        if len(alns) == 0:
            return None
        best = alns[0]
        target_blocks, query_blocks = best.aligned

        tentative_map: dict[int, int] = {}
        matches = 0
        aligned_total = 0
        for (t_lo, t_hi), (q_lo, q_hi) in zip(target_blocks, query_blocks):
            block_len = t_hi - t_lo
            for offset in range(block_len):
                unp_pos_0 = t_lo + offset
                chain_pos = q_lo + offset
                unp_resnum_1 = unp_pos_0 + 1
                auth_resnum, aa, _res = chain_residues[chain_pos]
                tentative_map[auth_resnum] = unp_resnum_1
                if uniprot_sequence[unp_pos_0] == aa:
                    matches += 1
                aligned_total += 1

        if aligned_total == 0:
            return None

        identity = matches / aligned_total
        if identity >= identity_threshold and aligned_total >= min_aligned:
            if notes_out is not None:
                notes_out.append(
                    f"Renumbered via sequence alignment "
                    f"({pdb_id}/{chain_id}, identity={identity:.3f}, "
                    f"matched {aligned_total}/{len(chain_residues)} residues).")
            return tentative_map
        else:
            if notes_out is not None:
                reason = []
                if identity < identity_threshold:
                    reason.append(
                        f"identity={identity:.3f}<{identity_threshold}")
                if aligned_total < min_aligned:
                    reason.append(
                        f"aligned_residues={aligned_total}<{min_aligned}")
                notes_out.append(
                    f"Alignment rejected for {pdb_id}/{chain_id} "
                    f"({', '.join(reason)}).")
            return None
    except Exception as e:
        if notes_out is not None:
            notes_out.append(
                f"Alignment failed for {pdb_id}/{chain_id}: "
                f"{type(e).__name__}: {e}.")
        return None


def _sifts_based_renumbering(chain_residues: list,
                             cache_dir: Path,
                             pdb_id: str, chain_id: str, uniprot_id: str,
                             ) -> Optional[tuple[dict[int, int], bool]]:
    """Derive auth->uniprot map from SIFTS segments. Returns
    (map, used_fallback) where used_fallback is True if any segment had
    null auth values that we had to infer.

    Returns None when SIFTS has no mapping at all for this (pdb, chain,
    uniprot). When SIFTS has data, always returns a map (may be partial),
    plus a flag indicating data quality.
    """
    mapping = get_pdb_full_uniprot_mapping(pdb_id, cache_dir).get(chain_id)
    if not mapping:
        return None
    mapping = [m for m in mapping if m["uniprot"] == uniprot_id]
    if not mapping:
        return None

    chain_resnums = [r[0] for r in chain_residues]
    if not chain_resnums:
        return None
    first_chain_resnum = min(chain_resnums)

    fixed_segments = []
    used_fallback = False
    for seg in mapping:
        unp_start = seg.get("unp_start")
        unp_end = seg.get("unp_end")
        auth_start = seg.get("pdb_auth_start")
        auth_end = seg.get("pdb_auth_end")
        if unp_start is None or unp_end is None:
            continue
        seg_length = unp_end - unp_start + 1

        if auth_start is None:
            used_fallback = True
            if auth_end is None:
                auth_start = first_chain_resnum
            else:
                auth_start = auth_end - seg_length + 1

        if auth_end is None:
            used_fallback = True
            auth_end = auth_start + seg_length - 1

        fixed_segments.append({
            "unp_start": unp_start, "unp_end": unp_end,
            "auth_start": auth_start, "auth_end": auth_end,
        })

    if not fixed_segments:
        return None

    out_map: dict[int, int] = {}
    for auth_resnum in chain_resnums:
        for seg in fixed_segments:
            if seg["auth_start"] <= auth_resnum <= seg["auth_end"]:
                out_map[auth_resnum] = seg["unp_start"] + (auth_resnum - seg["auth_start"])
                break
    return (out_map, used_fallback) if out_map else None


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
            seq += _aa_three_to_one(res.get_resname())
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
                             local_alphafold_dir: Optional[Path] = None,
                             allow_af_download: bool = True,
                             uniprot_sequence: Optional[str] = None,
                             alignment_identity_threshold: float = 0.85,
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
    local_alphafold_dir : Path   directory containing pre-downloaded AF files
                                 (gzipped or plain). Helper looks here first.
    allow_af_download : bool     If True, missing AF files are downloaded.
    uniprot_sequence : str       Canonical UniProt sequence for this protein
                                 (caller looks it up via load_uniprot_sequences).
                                 Used as the alignment fallback when SIFTS
                                 has null auth values. If None, no fallback.
    alignment_identity_threshold : float
                                 Minimum identity for the alignment-based
                                 fallback to be accepted (default 0.85).

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
    # context, and TM-validation. Picks the right fragment for long proteins.
    af_pdb_path, pae_path = resolve_alphafold(
        uniprot_id, cache_dir, dstart, dend,
        local_alphafold_dir=local_alphafold_dir,
        allow_download=allow_af_download)
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
            extraction_notes: list = []
            extracted = extract_chain_renumbered(
                raw_pdb, chain_id, cache_dir, pdb_id, uniprot_id,
                dstart, dend, ext_out,
                notes_out=extraction_notes,
                uniprot_sequence=uniprot_sequence,
                alignment_identity_threshold=alignment_identity_threshold)
            if extracted is None:
                continue
            picked_exp = (cand, extracted, extraction_notes)
            break  # ranked best-first; stop at first success

    # --- Decide what to return ---------------------------------------------
    if picked_exp is not None:
        cand, extracted, extraction_notes = picked_exp
        choice.available = True
        choice.source = "experimental"
        choice.pdb_path = extracted
        choice.pdb_id = cand["pdb_id"]
        choice.chain_id = cand["chain_id"]
        choice.resolution = cand.get("resolution")
        choice.method = cand.get("method")
        choice.domain_coverage = cand.get("domain_coverage")
        choice.domain_purity = cand.get("domain_purity")
        # Surface any auth-numbering fallback notes for downstream auditing.
        for note in extraction_notes:
            choice.notes.append(note)

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

def _cli_build_uniprot_to_auth_map(pdb_id: str, chain_id: str,
                                   uniprot_id: str,
                                   cache_dir: Path) -> dict[int, int]:
    """Return {uniprot_resnum: pdb_auth_resnum} for one (pdb, chain, uniprot)
    using the same fallback logic as _sifts_based_renumbering."""
    chain_mappings = get_pdb_full_uniprot_mapping(pdb_id, cache_dir).get(chain_id, [])
    chain_mappings = [m for m in chain_mappings if m["uniprot"] == uniprot_id]
    out: dict[int, int] = {}
    for seg in chain_mappings:
        unp_start = seg.get("unp_start")
        unp_end   = seg.get("unp_end")
        auth_start = seg.get("pdb_auth_start")
        auth_end   = seg.get("pdb_auth_end")
        if unp_start is None or unp_end is None:
            continue
        seg_length = unp_end - unp_start + 1
        if auth_start is None and auth_end is not None:
            auth_start = auth_end - seg_length + 1
        elif auth_end is None and auth_start is not None:
            auth_end = auth_start + seg_length - 1
        elif auth_start is None and auth_end is None:
            continue
        for offset in range(seg_length):
            out[unp_start + offset] = auth_start + offset
    return out


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
    ap.add_argument("--local-af-dir", type=Path, default=None,
                    help="Directory of pre-downloaded AlphaFold .pdb(.gz) "
                         "and PAE .json(.gz) files. Searched before "
                         "downloading.")
    ap.add_argument("--no-af-download", action="store_true",
                    help="Don't download AF files that are missing locally.")
    ap.add_argument("--min-coverage", type=float, default=0.8,
                    help="Minimum fraction of domain that must be in the PDB "
                         "for the PDB to be accepted (default 0.8).")
    ap.add_argument("--uniprot-tsv", type=Path, default=None,
                    help="UniProt TSV with 'Entry' and 'Sequence' columns. "
                         "If provided, the UniProt sequence is looked up and "
                         "passed to the alignment-based renumbering fallback.")
    ap.add_argument("--align-identity", type=float, default=0.85,
                    help="Min alignment identity to accept alignment-based "
                         "renumbering (default 0.85).")
    args = ap.parse_args()

    # Look up the UniProt sequence if a TSV was given.
    uniprot_seq = None
    if args.uniprot_tsv:
        seqs = load_uniprot_sequences(args.uniprot_tsv)
        uniprot_seq = seqs.get(args.uniprot_id)
        if uniprot_seq is None:
            print(f"WARNING: {args.uniprot_id} not found in {args.uniprot_tsv}; "
                  "alignment fallback disabled.")

    res = resolve_domain_structure(
        args.uniprot_id, args.dstart, args.dend,
        args.cache, mode=args.mode,
        min_domain_coverage=args.min_coverage,
        local_alphafold_dir=args.local_af_dir,
        allow_af_download=not args.no_af_download,
        uniprot_sequence=uniprot_seq,
        alignment_identity_threshold=args.align_identity,
    )
    import pprint
    pprint.pprint(asdict(res))

    # --- Residue mapping table -------------------------------------------
    if res.source == "experimental" and res.pdb_id and res.chain_id:
        unp_to_auth = _cli_build_uniprot_to_auth_map(
            res.pdb_id, res.chain_id, args.uniprot_id, args.cache)

        print(f"\nResidue map  "
              f"(UniProt → PDB {res.pdb_id.upper()}/{res.chain_id} auth "
              f"→ AlphaFold F1, both already in UniProt numbering)")
        print(f"  {'UniProt':>8}  {'PDB auth':>8}  {'AF F1':>6}")
        print(f"  {'-------':>8}  {'--------':>8}  {'------':>6}")
        for unp_res in range(args.dstart, args.dend + 1):
            auth_res = unp_to_auth.get(unp_res, "–")
            print(f"  {unp_res:>8}  {str(auth_res):>8}  {unp_res:>6}")

        auth_residues = sorted(
            v for k, v in unp_to_auth.items()
            if args.dstart <= k <= args.dend and isinstance(v, int))

        print(f"\nPyMOL selections:")
        if auth_residues:
            amin, amax = auth_residues[0], auth_residues[-1]
            print(f"  # Load the original PDB (auth numbering):")
            print(f"  fetch {res.pdb_id.upper()}")
            print(f"  select domain, chain {res.chain_id} and resi {amin}-{amax}")
        if res.full_protein_pdb_path:
            print(f"  # Load the AlphaFold model (UniProt numbering):")
            print(f"  load {res.full_protein_pdb_path}")
            print(f"  select domain_af, resi {args.dstart}-{args.dend}")

    elif res.source == "alphafold" and res.full_protein_pdb_path:
        print(f"\nResidue map  (AlphaFold only — residues already in UniProt numbering)")
        print(f"  {'UniProt':>8}  {'AF F1':>6}")
        print(f"  {'-------':>8}  {'------':>6}")
        for unp_res in range(args.dstart, args.dend + 1):
            print(f"  {unp_res:>8}  {unp_res:>6}")
        print(f"\nPyMOL selection:")
        print(f"  load {res.full_protein_pdb_path}")
        print(f"  select domain_af, resi {args.dstart}-{args.dend}")


if __name__ == "__main__":
    _cli()
