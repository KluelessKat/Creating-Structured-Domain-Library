#!/usr/bin/env python3
"""
structure_source_bioassembly.py

Variant of structure_source_fixedmapping.py that resolves the BIOLOGICAL
ASSEMBLY for an experimental PDB rather than the asymmetric unit (ASU).

Why a separate module?
----------------------
The ASU module (`structure_source_fixedmapping.py`) downloads the plain
`{pdb_id}.pdb` file, which contains only the asymmetric unit. For
homo-oligomers and many heterocomplexes the ASU does NOT contain the full
biological complex — instead REMARK 350 carries BIOMT transformation
matrices that the consumer must apply to generate the other copies. This
matters because freeSASA / contactDensity computed on the ASU will see a
"free" domain surface that is actually buried in the real complex.

This module sidesteps the BIOMT-application problem by downloading
pre-expanded biological assembly files from RCSB:

    https://files.rcsb.org/download/{PDB_ID}-assembly{N}.pdb.gz

These files contain ALL chains of the assembly with explicit coordinates,
so BioPython parses them directly and freeSASA sees the full oligomeric
context.

Key differences vs. the ASU module
----------------------------------
1. SIFTS chain IDs do not apply (assembly files renumber chains). We rely
   on **sequence alignment only** for chain → UniProt residue mapping, so
   `uniprot_sequence` (or a `--uniprot-tsv` at the CLI) is REQUIRED.
2. The full assembly file is kept around (in addition to the renumbered
   domain extraction) so downstream SASA/contact calculations get
   full-complex context. Pre-existing freeSASA code only needs the full
   assembly path; the renumbered domain file is for selection / TM-score.
3. A size guard skips pathological assemblies (viral capsids, ribosomes,
   etc.) — freeSASA on a 120-chain icosahedron is impractical and the
   biological question for a small domain library is not meaningful there.
   When the guard trips, the row falls back to ASU (and we record why).
4. PDBe summary API is used to pick the *preferred* assembly id (most
   structures have multiple). Crystal-contact-only assemblies (PISA-only,
   no author support) are de-prioritised.

Public entry point
------------------
    resolve_domain_bioassembly(uniprot_id, dstart, dend, cache_dir,
                               uniprot_sequence=..., ...) -> BioAssemblyChoice

Cache layout (under cache_dir/)
-------------------------------
    bioassembly/{PDB_ID}-assembly{N}.pdb           # pre-expanded download
    bioassembly/{uniprot}_{pdb}_ba{N}_{chain}_dom_{dstart}_{dend}.pdb
                                                    # extracted, renumbered domain
    alphafold/{uniprot}_F{n}_model.pdb              # shared with ASU module
    alphafold/{uniprot}_F{n}_PAE.json
    sifts/best_{uniprot}.json                       # shared with ASU module
    pdbe_assembly/{pdb_id}.json                     # new (assembly metadata)
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import time
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import requests
from Bio import PDB

# Reuse helpers from the ASU module to avoid duplicating code that did not
# change (AF resolution, SIFTS PDB-list lookup, ranking, AA tables, …).
from structure_source_fixedmapping import (
    _get_with_retry,
    _cache_dir as _base_cache_dir,
    _sifts_cached,
    _sifts_cache_write,
    list_pdbs_for_uniprot,
    rank_pdb_candidates,
    resolve_alphafold,
    _aa_three_to_one,
    _THREE_TO_ONE,
    StructureChoice,  # imported for type hints / interop only
    SIFTS_CACHE_TTL,
    ALPHAFOLD_VERSION,
)

# ---------------------------------------------------------------------------
# Constants / size guard
# ---------------------------------------------------------------------------

# Maximum number of chains / residues we are willing to process in a single
# assembly. Viral capsids (60-mers, 120-mers) and ribosomes vastly exceed
# these; SASA on those is impractical AND meaningless for a small-domain
# library. When tripped we fall back to the ASU PDB for this row.
MAX_ASSEMBLY_CHAINS = 24
MAX_ASSEMBLY_RESIDUES = 20_000

# Methods that do NOT have a meaningful biological assembly concept (no
# crystal symmetry → REMARK 350 either absent or trivial). We skip the
# bioassembly path for these and let the caller use the ASU file instead.
_METHODS_WITHOUT_BIOASSEMBLY = {
    "Solution NMR",
    "Solid-state NMR",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BioAssemblyChoice:
    """The resolved structure for one domain, biological-assembly variant.

    The fields up to `notes` mirror StructureChoice; the rest are
    bioassembly-specific. `pdb_path` is the renumbered domain extraction
    (one chain only); `full_assembly_pdb_path` is the entire expanded
    assembly file, which is what you feed to freeSASA for the proper
    oligomeric-context buried-surface calculation.
    """
    uniprot_id: str
    domain_start: int
    domain_end: int
    available: bool = False
    source: str = "none"     # "bioassembly" | "asu_fallback" | "alphafold" | "none"
    pdb_path: Optional[Path] = None              # renumbered domain only
    full_assembly_pdb_path: Optional[Path] = None  # whole expanded assembly (raw)
    full_protein_pdb_path: Optional[Path] = None   # AlphaFold full protein
    pae_path: Optional[Path] = None
    pdb_id: Optional[str] = None
    chain_id: Optional[str] = None               # chain in the assembly carrying our domain
    asu_chain_id: Optional[str] = None           # corresponding chain in the ASU (if known)
    resolution: Optional[float] = None
    method: Optional[str] = None
    domain_coverage: Optional[float] = None
    domain_purity: Optional[float] = None
    assembly_id: Optional[int] = None
    assembly_form: Optional[str] = None          # e.g. "monomer", "homo 2-mer", "hetero 4-mer"
    assembly_n_chains: Optional[int] = None
    assembly_n_residues: Optional[int] = None
    assembly_is_preferred: Optional[bool] = None
    skipped_reason: Optional[str] = None         # populated when bioassembly path was rejected
    loose_align_fragment: Optional[int] = None
    notes: list = field(default_factory=list)

    def to_row(self) -> dict:
        """Flat dict suitable for adding to a pandas DataFrame row."""
        return {
            "structureSource": self.source,
            "pdbID": self.pdb_id,
            "pdbChain": self.chain_id,
            "asuChainID": self.asu_chain_id,
            "pdbResolution": self.resolution,
            "pdbMethod": self.method,
            "domainCoverage": self.domain_coverage,
            "domainPurity": self.domain_purity,
            "assemblyID": self.assembly_id,
            "assemblyForm": self.assembly_form,
            "assemblyNChains": self.assembly_n_chains,
            "assemblyNResidues": self.assembly_n_residues,
            "assemblyIsPreferred": self.assembly_is_preferred,
            "skippedReason": self.skipped_reason,
            "loose_align_fragment": self.loose_align_fragment,
            "structureNotes": "; ".join(self.notes) if self.notes else None,
        }


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_dir(cache_dir: Path) -> Path:
    """Extend the shared cache layout with bioassembly + pdbe_assembly dirs."""
    base = _base_cache_dir(cache_dir)
    (base / "bioassembly").mkdir(parents=True, exist_ok=True)
    (base / "pdbe_assembly").mkdir(parents=True, exist_ok=True)
    return base


def _assembly_meta_cached(cache_dir: Path, pdb_id: str) -> Optional[list]:
    p = cache_dir / "pdbe_assembly" / f"{pdb_id.lower()}.json"
    if not p.exists():
        return None
    if (time.time() - p.stat().st_mtime) > SIFTS_CACHE_TTL:
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _assembly_meta_write(cache_dir: Path, pdb_id: str, data) -> None:
    p = cache_dir / "pdbe_assembly"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{pdb_id.lower()}.json").write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# PDBe assembly metadata
# ---------------------------------------------------------------------------

def get_assembly_info(pdb_id: str, cache_dir: Path) -> list[dict]:
    """Fetch the list of biological assemblies for `pdb_id` from PDBe.

    Returns a list of dicts (one per assembly), each containing keys:
        assembly_id : int
        form        : str  (e.g. "monomer", "homo dimer", "hetero tetramer")
        preferred   : bool
        provenance  : str  (e.g. "author_and_software_defined_assembly")
        composition : str  (e.g. "homomer", "heteromer", "monomer")

    Results are cached on disk for SIFTS_CACHE_TTL seconds.
    Returns [] on 404 / network failure.
    """
    pdb_id = pdb_id.lower()
    cached = _assembly_meta_cached(cache_dir, pdb_id)
    if cached is not None:
        return cached

    url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/{pdb_id}"
    r = _get_with_retry(url)
    if r is None or r.status_code == 404:
        _assembly_meta_write(cache_dir, pdb_id, [])
        return []

    try:
        payload = r.json()
    except json.JSONDecodeError:
        return []

    entries = payload.get(pdb_id, [])
    if not entries:
        _assembly_meta_write(cache_dir, pdb_id, [])
        return []

    out = []
    for entry in entries:
        for a in entry.get("assemblies", []):
            try:
                aid = int(a.get("assembly_id"))
            except (TypeError, ValueError):
                continue
            out.append({
                "assembly_id": aid,
                "form": a.get("form"),                # "monomer"/"hetero 4-mer"/etc.
                "preferred": bool(a.get("preferred", False)),
                "provenance": a.get("assembly_composition") or a.get("provenance"),
                "name": a.get("name"),
            })
    # Sort: preferred first, then by assembly_id ascending.
    out.sort(key=lambda x: (not x["preferred"], x["assembly_id"]))
    _assembly_meta_write(cache_dir, pdb_id, out)
    return out


def pick_preferred_assembly(assemblies: list[dict]) -> Optional[dict]:
    """Return the preferred assembly dict (or the first one, or None)."""
    if not assemblies:
        return None
    for a in assemblies:
        if a.get("preferred"):
            return a
    return assemblies[0]


# ---------------------------------------------------------------------------
# Bioassembly file download
# ---------------------------------------------------------------------------

def download_bioassembly(pdb_id: str, assembly_id: int,
                         cache_dir: Path) -> Optional[Path]:
    """Download a pre-expanded biological assembly PDB from RCSB.

    Tries the modern naming first, then the legacy naming:
        {PDB_ID}-assembly{N}.pdb.gz   (current)
        {pdb_id}.pdb{N}.gz            (legacy, some pre-2014 entries)

    Decompresses to {cache_dir}/bioassembly/{PDB_ID}-assembly{N}.pdb and
    returns that path. Returns None on persistent failure (e.g. 404 on
    both URL forms).
    """
    cache_dir = _cache_dir(cache_dir)
    out = cache_dir / "bioassembly" / f"{pdb_id.upper()}-assembly{assembly_id}.pdb"
    if out.exists():
        return out

    urls = [
        f"https://files.rcsb.org/download/{pdb_id.upper()}-assembly{assembly_id}.pdb.gz",
        f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb{assembly_id}.gz",
    ]
    tmp_gz = cache_dir / "bioassembly" / f"{pdb_id.upper()}-assembly{assembly_id}.pdb.gz"
    got = False
    for url in urls:
        r = _get_with_retry(url, stream=True)
        if r is None or r.status_code == 404:
            continue
        with open(tmp_gz, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        got = True
        break
    if not got:
        return None

    try:
        with gzip.open(tmp_gz, "rb") as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    except (OSError, EOFError) as e:
        warnings.warn(f"Failed to decompress {tmp_gz}: {e}")
        return None
    finally:
        try:
            tmp_gz.unlink()
        except OSError:
            pass
    return out


# ---------------------------------------------------------------------------
# Size guard
# ---------------------------------------------------------------------------

def count_chains_and_residues(pdb_path: Path) -> tuple[int, int]:
    """Count distinct (model_id, chain_id) pairs and total protein residues
    in a PDB file. Uses a streaming scan rather than BioPython for speed —
    we want to abort BEFORE loading a 120-chain capsid into memory.

    Counts ATOM CA records only (protein backbone marker). HETATM is
    ignored — water/ions/ligands are not part of the assembly size we
    care about for SASA.
    """
    chains: set[tuple[str, str]] = set()
    residues: set[tuple[str, str, int, str]] = set()
    current_model = "1"
    try:
        with open(pdb_path, "r", errors="ignore") as fh:
            for line in fh:
                if line.startswith("MODEL"):
                    current_model = line[10:14].strip() or "1"
                    continue
                if not line.startswith("ATOM"):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                chain_id = line[21:22]
                try:
                    resnum = int(line[22:26])
                except ValueError:
                    continue
                icode = line[26:27]
                chains.add((current_model, chain_id))
                residues.add((current_model, chain_id, resnum, icode))
    except OSError as e:
        warnings.warn(f"Failed to scan {pdb_path}: {e}")
        return (0, 0)
    return (len(chains), len(residues))


# ---------------------------------------------------------------------------
# Alignment-based chain finder
# ---------------------------------------------------------------------------

def _chain_residues_seq(chain) -> list[tuple[int, str, "PDB.Residue.Residue"]]:
    """Return [(auth_resnum, aa_1letter, residue_obj), ...] for protein
    residues in a chain, skipping HETATM and waters."""
    out = []
    for residue in chain:
        hetflag, auth_resnum, _icode = residue.id
        if hetflag.strip() and hetflag != " ":
            continue
        aa = _aa_three_to_one(residue.get_resname())
        out.append((auth_resnum, aa, residue))
    return out


def _align_chain_to_uniprot(chain_residues: list,
                            uniprot_sequence: str,
                            dstart: int, dend: int,
                            ) -> Optional[dict]:
    """Local-align a chain's amino-acid string to the canonical UniProt
    sequence with BLOSUM62. Returns a dict:

        {
          "auth_to_unp":     {auth_resnum: uniprot_resnum},  # 1-indexed
          "identity":        float,           # fraction of aligned positions matching
          "aligned_total":   int,             # number of aligned residues
          "domain_overlap":  int,             # how many [dstart,dend] residues were aligned
        }

    Or None if Bio.Align is unavailable or no alignment exists.
    """
    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError:
        return None
    chain_seq = "".join(aa for _resnum, aa, _res in chain_residues)
    if not chain_seq:
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
    except Exception:
        return None

    auth_to_unp: dict[int, int] = {}
    matches = 0
    aligned_total = 0
    domain_overlap = 0
    for (t_lo, t_hi), (q_lo, q_hi) in zip(target_blocks, query_blocks):
        block_len = t_hi - t_lo
        for offset in range(block_len):
            unp_pos_0 = t_lo + offset
            chain_pos = q_lo + offset
            unp_resnum_1 = unp_pos_0 + 1
            auth_resnum, aa, _res = chain_residues[chain_pos]
            auth_to_unp[auth_resnum] = unp_resnum_1
            if uniprot_sequence[unp_pos_0] == aa:
                matches += 1
            aligned_total += 1
            if dstart <= unp_resnum_1 <= dend:
                domain_overlap += 1

    if aligned_total == 0:
        return None
    return {
        "auth_to_unp": auth_to_unp,
        "identity": matches / aligned_total,
        "aligned_total": aligned_total,
        "domain_overlap": domain_overlap,
    }


def find_domain_chain_in_assembly(assembly_pdb: Path,
                                  uniprot_sequence: str,
                                  dstart: int, dend: int,
                                  identity_threshold: float = 0.85,
                                  ) -> Optional[dict]:
    """Scan every chain of an expanded assembly file and pick the one whose
    local alignment to `uniprot_sequence` covers the most of [dstart, dend].

    Returns a dict:
        {
          "model_id":     str,    # "1" for single-model assemblies
          "chain_id":     str,
          "auth_to_unp":  dict[int, int],
          "identity":     float,
          "aligned_total": int,
          "domain_overlap": int,
          "n_chain_residues": int,
        }

    Returns None if no chain produced a usable alignment (identity below
    threshold, or all chains failed). For homo-oligomers all chains tie —
    we pick the first by (model_id, chain_id) ordering for determinism.
    """
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("ba", str(assembly_pdb))
    except Exception as e:
        warnings.warn(f"Failed to parse {assembly_pdb}: {e}")
        return None

    best: Optional[dict] = None
    for model in structure:
        for chain in model:
            chain_residues = _chain_residues_seq(chain)
            if len(chain_residues) < 5:
                continue
            aln = _align_chain_to_uniprot(chain_residues, uniprot_sequence,
                                          dstart, dend)
            if aln is None:
                continue
            if aln["identity"] < identity_threshold:
                continue
            score = (aln["domain_overlap"], aln["aligned_total"], aln["identity"])
            if best is None or score > (best["domain_overlap"],
                                         best["aligned_total"],
                                         best["identity"]):
                best = {
                    "model_id": str(model.id),
                    "chain_id": chain.id,
                    "auth_to_unp": aln["auth_to_unp"],
                    "identity": aln["identity"],
                    "aligned_total": aln["aligned_total"],
                    "domain_overlap": aln["domain_overlap"],
                    "n_chain_residues": len(chain_residues),
                }
    if best is None or best["domain_overlap"] == 0:
        return None
    return best


# ---------------------------------------------------------------------------
# Domain extraction (renumbered to UniProt, single chain)
# ---------------------------------------------------------------------------

def extract_domain_from_assembly(assembly_pdb: Path,
                                 model_id: str, chain_id: str,
                                 auth_to_unp: dict[int, int],
                                 dstart: int, dend: int,
                                 out_path: Path) -> Optional[Path]:
    """Pull the one chain identified by `chain_id` out of the assembly,
    restrict to UniProt residues in [dstart, dend], renumber, and save."""
    if out_path.exists():
        return out_path

    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("ba", str(assembly_pdb))
    except Exception as e:
        warnings.warn(f"Failed to parse {assembly_pdb}: {e}")
        return None

    src_chain = None
    for model in structure:
        if str(model.id) != model_id:
            continue
        for chain in model:
            if chain.id == chain_id:
                src_chain = chain
                break
        break
    if src_chain is None:
        return None

    new_structure = PDB.Structure.Structure("dom")
    new_model = PDB.Model.Model(0)
    new_chain = PDB.Chain.Chain(chain_id)
    new_model.add(new_chain)
    new_structure.add(new_model)

    seen: set[int] = set()
    for residue in src_chain:
        hetflag, auth_resnum, _icode = residue.id
        if hetflag.strip() and hetflag != " ":
            continue
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


# ---------------------------------------------------------------------------
# Heuristic: is this assembly form a homomer / hetero / monomer?
# ---------------------------------------------------------------------------

def _classify_form(form: Optional[str]) -> str:
    if not form:
        return "unknown"
    f = form.lower()
    if "monomer" in f:
        return "monomer"
    if "hetero" in f:
        return "hetero"
    if "homo" in f:
        return "homo"
    return "unknown"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def resolve_domain_bioassembly(uniprot_id: str,
                               dstart: int, dend: int,
                               cache_dir: Path,
                               uniprot_sequence: Optional[str],
                               min_domain_coverage: float = 0.8,
                               local_alphafold_dir: Optional[Path] = None,
                               allow_af_download: bool = True,
                               alignment_identity_threshold: float = 0.85,
                               max_assembly_chains: int = MAX_ASSEMBLY_CHAINS,
                               max_assembly_residues: int = MAX_ASSEMBLY_RESIDUES,
                               accept_only_preferred: bool = False,
                               accept_heteromer: bool = True,
                               ) -> BioAssemblyChoice:
    """Pick the best biological-assembly structure for one domain.

    Parameters
    ----------
    uniprot_id, dstart, dend :  domain range in UniProt numbering
    cache_dir                :  local cache root
    uniprot_sequence         :  canonical UniProt sequence (REQUIRED — used
                                for alignment-based chain identification).
                                Pass None to disable the bioassembly path
                                entirely (will fall back to AF).
    min_domain_coverage      :  passed to rank_pdb_candidates; PDBs covering
                                less than this fraction of the domain are
                                skipped.
    local_alphafold_dir      :  for AF PDB + PAE (anchoringIndex remains AF).
    max_assembly_chains      :  size guard — abandon bioassembly when the
                                expanded file has more chains than this.
    max_assembly_residues    :  size guard — same for total residue count.
    accept_only_preferred    :  if True, only use the PDBe-flagged preferred
                                assembly; skip otherwise. Default False:
                                fall through to other assemblies if the
                                preferred one is unusable.
    accept_heteromer         :  if False, only homomers / monomers are used
                                (a heteromer assembly inflates fb/cd with
                                non-parent-protein contacts). Default True.

    Returns
    -------
    BioAssemblyChoice
    """
    cache_dir = _cache_dir(cache_dir)
    choice = BioAssemblyChoice(uniprot_id=uniprot_id,
                               domain_start=dstart, domain_end=dend)

    # ----- 1. AF context (anchoringIndex, full protein, TM-score) -----------
    af_pdb_path, pae_path, loose_align_fragment = resolve_alphafold(
        uniprot_id, cache_dir, dstart, dend,
        local_alphafold_dir=local_alphafold_dir,
        allow_download=allow_af_download,
        return_fragment=True)
    choice.full_protein_pdb_path = af_pdb_path
    choice.pae_path = pae_path
    choice.loose_align_fragment = loose_align_fragment

    # If the caller has no UniProt sequence we cannot do alignment-based
    # chain identification — fall through to AF immediately.
    if not uniprot_sequence:
        choice.skipped_reason = "no_uniprot_sequence"
        choice.notes.append("uniprot_sequence not supplied; "
                            "bioassembly path requires it.")
        return _finalize_with_af(choice, af_pdb_path)

    # ----- 2. Rank PDB candidates (same logic as ASU module) ----------------
    ranked = rank_pdb_candidates(uniprot_id, dstart, dend, cache_dir,
                                  min_domain_coverage=min_domain_coverage)
    if not ranked:
        choice.skipped_reason = "no_pdb_candidates"
        return _finalize_with_af(choice, af_pdb_path)

    # ----- 3. Walk the ranked list, looking for a usable assembly ----------
    for cand in ranked:
        pdb_id = cand["pdb_id"]
        method = cand.get("method") or ""

        # NMR / other no-symmetry methods → no meaningful bioassembly.
        if method in _METHODS_WITHOUT_BIOASSEMBLY:
            choice.notes.append(f"{pdb_id}: method '{method}' has no "
                                "bioassembly concept; skipping.")
            continue

        # Determine which assembly to download.
        assemblies = get_assembly_info(pdb_id, cache_dir)
        target = pick_preferred_assembly(assemblies)
        if target is None:
            choice.notes.append(f"{pdb_id}: no PDBe assembly metadata.")
            continue
        if accept_only_preferred and not target.get("preferred"):
            choice.notes.append(f"{pdb_id}: only preferred assemblies "
                                "accepted; preferred flag missing.")
            continue
        form_class = _classify_form(target.get("form"))
        if form_class == "hetero" and not accept_heteromer:
            choice.notes.append(
                f"{pdb_id}: assembly is heteromer ('{target.get('form')}') "
                "and accept_heteromer=False; skipping.")
            continue

        # Download expanded assembly.
        ba_path = download_bioassembly(pdb_id, target["assembly_id"], cache_dir)
        if ba_path is None:
            choice.notes.append(f"{pdb_id}: assembly{target['assembly_id']} "
                                "download failed.")
            continue

        # Size guard — count first, refuse if too big.
        n_chains, n_residues = count_chains_and_residues(ba_path)
        if n_chains > max_assembly_chains or n_residues > max_assembly_residues:
            choice.notes.append(
                f"{pdb_id}/assembly{target['assembly_id']}: size guard "
                f"({n_chains} chains / {n_residues} residues) — skipping.")
            continue

        # Find the chain containing our domain.
        match = find_domain_chain_in_assembly(
            ba_path, uniprot_sequence, dstart, dend,
            identity_threshold=alignment_identity_threshold)
        if match is None:
            choice.notes.append(
                f"{pdb_id}/assembly{target['assembly_id']}: no chain matched "
                f"the UniProt domain (identity<{alignment_identity_threshold}).")
            continue

        # Extract renumbered domain.
        dom_out = (cache_dir / "bioassembly"
                   / f"{uniprot_id}_{pdb_id}_ba{target['assembly_id']}"
                     f"_{match['chain_id']}_dom_{dstart}_{dend}.pdb")
        extracted = extract_domain_from_assembly(
            ba_path, match["model_id"], match["chain_id"],
            match["auth_to_unp"], dstart, dend, dom_out)
        if extracted is None:
            choice.notes.append(
                f"{pdb_id}/assembly{target['assembly_id']}: domain extraction "
                "produced empty PDB.")
            continue

        # Coverage / purity in the assembly chain we picked.
        dom_len = dend - dstart + 1
        coverage = match["domain_overlap"] / dom_len
        purity = match["domain_overlap"] / max(match["n_chain_residues"], 1)

        # Success — populate and return.
        choice.available = True
        choice.source = "bioassembly"
        choice.pdb_path = extracted
        choice.full_assembly_pdb_path = ba_path
        choice.pdb_id = pdb_id
        choice.chain_id = match["chain_id"]
        choice.asu_chain_id = cand.get("chain_id")
        choice.resolution = cand.get("resolution")
        choice.method = method
        choice.domain_coverage = coverage
        choice.domain_purity = purity
        choice.assembly_id = target["assembly_id"]
        choice.assembly_form = target.get("form")
        choice.assembly_n_chains = n_chains
        choice.assembly_n_residues = n_residues
        choice.assembly_is_preferred = target.get("preferred")
        choice.notes.append(
            f"Renumbered via alignment ({pdb_id}/assembly{target['assembly_id']}"
            f"/{match['chain_id']}, identity={match['identity']:.3f}, "
            f"domain_overlap={match['domain_overlap']}/{dom_len}, "
            f"chain_residues={match['n_chain_residues']}).")
        return choice

    # No candidate produced a usable bioassembly.
    if choice.skipped_reason is None:
        choice.skipped_reason = "all_candidates_skipped"
    return _finalize_with_af(choice, af_pdb_path)


def _finalize_with_af(choice: BioAssemblyChoice,
                      af_pdb_path: Optional[Path]) -> BioAssemblyChoice:
    """Fallback path: no usable bioassembly found, return whatever AF we have."""
    if af_pdb_path is None:
        choice.available = False
        choice.source = "none"
        choice.notes.append("AlphaFold prediction also unavailable.")
        return choice
    choice.available = True
    choice.source = "alphafold"
    choice.pdb_path = af_pdb_path
    return choice


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    import pprint
    from structure_source_fixedmapping import load_uniprot_sequences

    ap = argparse.ArgumentParser(
        description="Smoke-test structure_source_bioassembly.")
    ap.add_argument("uniprot_id")
    ap.add_argument("dstart", type=int)
    ap.add_argument("dend", type=int)
    ap.add_argument("--cache", type=Path, default=Path("./struct_cache"))
    ap.add_argument("--uniprot-tsv", type=Path, required=True,
                    help="UniProt TSV with 'Entry' and 'Sequence' columns. "
                         "REQUIRED — bioassembly path uses alignment for "
                         "chain mapping, which needs the canonical sequence.")
    ap.add_argument("--local-af-dir", type=Path, default=None)
    ap.add_argument("--no-af-download", action="store_true")
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--align-identity", type=float, default=0.85)
    ap.add_argument("--max-chains", type=int, default=MAX_ASSEMBLY_CHAINS)
    ap.add_argument("--max-residues", type=int, default=MAX_ASSEMBLY_RESIDUES)
    ap.add_argument("--only-preferred", action="store_true",
                    help="Reject assemblies not flagged 'preferred' by PDBe.")
    ap.add_argument("--no-heteromer", action="store_true",
                    help="Reject heteromer assemblies "
                         "(contacts with partner proteins would inflate fb/cd).")
    args = ap.parse_args()

    seqs = load_uniprot_sequences(args.uniprot_tsv)
    uniprot_seq = seqs.get(args.uniprot_id)
    if uniprot_seq is None:
        print(f"ERROR: {args.uniprot_id} not in {args.uniprot_tsv}.")
        return

    res = resolve_domain_bioassembly(
        args.uniprot_id, args.dstart, args.dend, args.cache,
        uniprot_sequence=uniprot_seq,
        min_domain_coverage=args.min_coverage,
        local_alphafold_dir=args.local_af_dir,
        allow_af_download=not args.no_af_download,
        alignment_identity_threshold=args.align_identity,
        max_assembly_chains=args.max_chains,
        max_assembly_residues=args.max_residues,
        accept_only_preferred=args.only_preferred,
        accept_heteromer=not args.no_heteromer,
    )
    pprint.pprint(asdict(res))

    if res.source == "bioassembly":
        print(f"\nPyMOL selections:")
        print(f"  load {res.full_assembly_pdb_path}")
        print(f"  # Whole assembly loaded; domain chain = '{res.chain_id}'")
        print(f"  select domain, chain {res.chain_id} and resi "
              f"({_format_auth_range(res, args.dstart, args.dend)})")
        print(f"  # Renumbered, domain-only file (UniProt numbering):")
        print(f"  load {res.pdb_path}, domain_only")


def _format_auth_range(res: BioAssemblyChoice, dstart: int, dend: int) -> str:
    """Print a PyMOL-friendly residue range. In the assembly PDB the chain
    still uses author (auth) numbering — we'd need the auth→unp map to
    translate, but for a quick smoke test we just print the UniProt range
    and let the user remember that the assembly file is in auth numbers.
    The renumbered domain file IS in UniProt numbering."""
    return f"{dstart}-{dend}"


if __name__ == "__main__":
    _cli()
