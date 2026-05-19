#!/usr/bin/env python3
"""
2_disorderedPredictions.py

Run disorder prediction on domain sequences using metapredict and AlphaFold
pLDDT by default (AIUPred and IUPred3 are skipped unless explicitly enabled),
then filter the library based on a chosen predictor (or combination) in a
single pass.

All predictors are always run. The output TSV carries per-predictor columns
side-by-side plus derived consensus columns so you can compare post-hoc.
Filtering uses one of the predictor/consensus columns of your choice.

Output columns added per predictor P ('metapredict', 'aiupred', 'iupred3'):
    P_mean_disorder              mean per-residue disorder score
    P_fraction_disordered        fraction of residues with score > 0.5

pLDDT column (fetched from AlphaFold EBI; streamed in memory, no disk write):
    plddt_mean_domain            mean pLDDT over domain residues (0–100 scale)
                                 higher = more confident structure

Consensus columns (computed from all predictors that succeeded):
    mean_ensemble_mean_disorder         mean of the three *_mean_disorder
    mean_ensemble_fraction_disordered   mean of the three *_fraction_disordered
    intersection_passes_filter          1 if EVERY available predictor AND
                                        pLDDT independently passes the filter,
                                        else 0

Usage:
    python 2_disorderedPredictions.py \\
        --input 1_domainLibraryRaw.tsv \\
        --output 2_domainLibraryStructuredSeq_all.tsv \\
        --filter-on mean_ensemble \\
        --iupred3-dir /path/to/iupred3

--filter-on choices (which column set the mean<=0.5 / fraction<=0.2 filter is
applied to):
    metapredict_and_plddt [DEFAULT]          metapredict disorder AND pLDDT must
                                             both pass their thresholds
    metapredict, aiupred, iupred3            single predictor
    mean_ensemble                            ensemble mean
    intersection                             domain must pass under all predictors
                                             AND pLDDT (if available)
    plddt                                    pLDDT alone
    none                                     do not filter; just write the
                                             full annotated library

Filter mode choices (--filter-mode):
    structured   keep rows with mean<=0.5 AND fraction<=0.2  (default)
                 (for --filter-on plddt: keep rows with plddt >= threshold)
    disordered   keep rows with mean>0.5  AND fraction>0.2
                 (for --filter-on plddt: keep rows with plddt < threshold)
    none         do not filter even if --filter-on is set (useful to produce
                 an annotated superset and run the filter elsewhere)
"""

from __future__ import annotations

import argparse
import gzip
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# AlphaFold fragment helpers
# ---------------------------------------------------------------------------
# AF DB splits proteins >2700 AA into overlapping 1400-AA fragments (step=200).
# Proteins <=2700 AA are never fragmented — F1 covers the full chain.
# Fragment PDB files use LOCAL residue numbering (1 to ~1400).
# F1: global 1-1400 (offset 0), F2: global 201-1600 (offset 200), etc.

FRAG_STEP = 200  # AA step between consecutive fragment start positions

# Standard 3-letter → 1-letter amino acid lookup (used for sequence verification)
_AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'SEC':'U','PYL':'O','ASX':'B','GLX':'Z','XLE':'J','UNK':'X',
}

def getAFFragment(domainStart: int, domainEnd: int, proteinLength: int) -> Optional[int]:
    '''Return the AF fragment number containing [domainStart, domainEnd].
    Proteins <=2700 AA are not fragmented (always F1, local = global).
    For fragmented proteins: step=200, F1=1-1400, F2=201-1600, ...
    Returns None if the domain is too long to fit in any single fragment.'''
    if proteinLength <= 2700:
        return 1
    # Smallest n where (n-1)*200 + 1400 >= domainEnd
    n = max(1, math.ceil((domainEnd - 1400) / FRAG_STEP) + 1)
    # Verify domainStart also fits inside fragment n
    if (n - 1) * FRAG_STEP + 1 > domainStart:
        return None  # domain spans a fragment boundary; cannot fit in one fragment
    return n

def globalToLocal(pos: int, fragment: int) -> int:
    '''Convert a global UniProt residue position to a local fragment position.
    F1: offset 0 (local = global), F2: offset 200, F3: offset 400, ...'''
    return pos - (fragment - 1) * FRAG_STEP

def _verifyDomainSeq(resnames: dict, localStart: int,
                     domainSeq: str, nCheck: int = 5) -> bool:
    '''Check first nCheck residues of domainSeq against already-parsed resnames dict.
    Returns True on match, False (with a printed warning) on mismatch.'''
    pdbSeq   = ''.join(resnames.get(r, '?') for r in range(localStart, localStart + nCheck))
    expected = domainSeq[:nCheck].upper()
    if '?' not in pdbSeq and pdbSeq == expected:
        return True
    print(f"  Sequence mismatch at local {localStart}: PDB '{pdbSeq}' vs domain '{expected}'")
    return False


# ---------------------------------------------------------------------------
# USER CONFIGURATION  -->  edit this section
# ---------------------------------------------------------------------------
# Hardcoded default directory containing iupred3_lib.py and its data/ folder.
# Set this to your local IUPred3 download directory so you don't have to pass
# --iupred3-dir on the command line every run.
#
# The CLI flag --iupred3-dir, if provided, still wins over this value. Set
# to None to require the flag and refuse to guess.

IUPRED3_DIR_DEFAULT = '/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/iupred3'

# ---------------------------------------------------------------------------
# Predictor loading
# ---------------------------------------------------------------------------
# Each predictor is loaded lazily and wrapped in a uniform interface:
#     predict(sequence: str) -> np.ndarray of per-residue scores in [0, 1]
# If a predictor can't be loaded, we record why and skip it at runtime; its
# output columns will be filled with NaN.

@dataclass
class Predictor:
    name: str
    predict: Optional[Callable[[str], np.ndarray]]
    error: Optional[str] = None  # populated if the predictor failed to load

    @property
    def available(self) -> bool:
        return self.predict is not None


def load_metapredict() -> Predictor:
    try:
        import metapredict as meta
        def predict(seq: str) -> np.ndarray:
            return np.asarray(meta.predict_disorder(seq), dtype=float)
        return Predictor('metapredict', predict)
    except Exception as e:
        return Predictor('metapredict', None, error=f'{type(e).__name__}: {e}')


def load_aiupred() -> Predictor:
    try:
        from aiupred import AIUPred
        instance = AIUPred(force_cpu=True)
        def predict(seq: str) -> np.ndarray:
            return np.asarray(instance.predict_disorder(seq), dtype=float)
        return Predictor('aiupred', predict)
    except Exception as e:
        return Predictor('aiupred', None, error=f'{type(e).__name__}: {e}')


def load_iupred3(iupred3_dir: Optional[Path],
                 mode: str = 'long',
                 smoothing: str = 'medium') -> Predictor:
    """IUPred3 lives as a downloaded folder with iupred3_lib.py + data/ next
    to it. We put that folder on sys.path so 'import iupred3_lib' works, then
    rely on the module's PATH = dirname(__file__) to find the data files."""
    try:
        if iupred3_dir is not None:
            d = Path(iupred3_dir).resolve()
            if not d.is_dir():
                raise FileNotFoundError(f'--iupred3-dir does not exist: {d}')
            if not (d / 'iupred3_lib.py').exists():
                raise FileNotFoundError(f'iupred3_lib.py not found in {d}')
            if not (d / 'data').is_dir():
                raise FileNotFoundError(
                    f'{d}/data not found — IUPred3 needs its data folder next to iupred3_lib.py')
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
        import iupred3_lib
        def predict(seq: str) -> np.ndarray:
            scores, _ = iupred3_lib.iupred(seq, mode=mode, smoothing=smoothing)
            return np.asarray(scores, dtype=float)
        return Predictor('iupred3', predict)
    except Exception as e:
        return Predictor('iupred3', None, error=f'{type(e).__name__}: {e}')


# ---------------------------------------------------------------------------
# pLDDT scoring (AlphaFold, no disk I/O)
# ---------------------------------------------------------------------------
# In AlphaFold PDB files the pLDDT score for each residue is stored in the
# B-factor column. We stream the file over HTTP and parse only the CA ATOM
# lines, so nothing is written to disk. Results are cached in memory keyed by
# UniProt ID so each protein is fetched at most once per run.

# Cache keyed by (uniprot_id, fragment).
# Value: None  → network failure
#        (bfactors, resnames) tuple  → success (may be empty dicts for 404)
_plddt_cache: dict[tuple[str, int], Optional[tuple[dict, dict]]] = {}


def _parse_pdb_ca_bfactors(lines) -> tuple[dict, dict]:
    '''Parse CA ATOM lines from an iterable of PDB lines (str or bytes).
    Returns ({local_resnum: pLDDT}, {local_resnum: 1-letter AA}).'''
    bfactors: dict[int, float] = {}
    resnames: dict[int, str]   = {}
    for raw in lines:
        line = raw.decode("ascii", errors="ignore") if isinstance(raw, bytes) else raw
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            resnum  = int(line[22:26])
            bfactor = float(line[60:66])
            resname = _AA3TO1.get(line[17:20].strip(), 'X')
        except ValueError:
            continue
        if resnum not in bfactors:
            bfactors[resnum] = bfactor
            resnames[resnum] = resname
    return bfactors, resnames


def _fetch_plddt_for_fragment(uniprot_id: str, fragment: int,
                               af_dir: Optional[Path] = None
                               ) -> Optional[tuple[dict, dict]]:
    """Return (bfactors, resnames) dicts for the given entry + fragment.
    Checks af_dir first (unzipped or .gz), then falls back to AlphaFold EBI.
    Returns None on network failure, ({}, {}) on 404 (no AF model).
    """
    cache_key = (uniprot_id, fragment)
    if cache_key in _plddt_cache:
        return _plddt_cache[cache_key]

    # --- 1. Check local af_dir first ----------------------------------------
    if af_dir and af_dir.is_dir():
        candidates: list[Path] = []
        if fragment == 1:                          # step-3 naming convention
            candidates += [
                af_dir / f"{uniprot_id}_model.pdb",
                af_dir / f"{uniprot_id}_model.pdb.gz",
            ]
        candidates += [
            af_dir / f"AF-{uniprot_id}-F{fragment}-model_v6.pdb",
            af_dir / f"AF-{uniprot_id}-F{fragment}-model_v6.pdb.gz",
        ]
        for path in candidates:
            if path.exists():
                if path.suffix == '.gz':
                    with gzip.open(path, 'rt', errors='ignore') as fh:
                        result = _parse_pdb_ca_bfactors(fh)
                else:
                    with open(path, 'r', errors='ignore') as fh:
                        result = _parse_pdb_ca_bfactors(fh)
                _plddt_cache[cache_key] = result
                return result

    # --- 2. Fall back to AlphaFold EBI URL ----------------------------------
    url = (f"https://alphafold.ebi.ac.uk/files/"
           f"AF-{uniprot_id}-F{fragment}-model_{ALPHAFOLD_VERSION}.pdb")
    try:
        r = requests.get(url, timeout=30, stream=True)
        if r.status_code == 404:
            _plddt_cache[cache_key] = ({}, {})
            return ({}, {})
        r.raise_for_status()
    except requests.RequestException as e:
        warnings.warn(f"pLDDT fetch failed for {uniprot_id} F{fragment}: {e}")
        _plddt_cache[cache_key] = None
        return None

    result = _parse_pdb_ca_bfactors(r.iter_lines())
    _plddt_cache[cache_key] = result
    return result


def score_plddt_domain(uniprot_id: str, dstart: int, dend: int,
                        proteinLength: int,
                        domainSeq: Optional[str] = None,
                        af_dir: Optional[Path] = None) -> Optional[float]:
    """Return mean pLDDT over domain residues [dstart, dend] (global, 1-indexed).
    Picks the correct AF fragment based on protein length, converts to local coords,
    and optionally verifies the first 5 residues against domainSeq.
    Returns None if the fragment cannot be fetched or no CA atoms fall in range.
    """
    fragment = getAFFragment(dstart, dend, proteinLength)
    if fragment is None:
        print(f"  Warning: domain {dstart}-{dend} spans a fragment boundary; skipping pLDDT.")
        return None
    localStart = globalToLocal(dstart, fragment)
    localEnd   = globalToLocal(dend,   fragment)

    frag_result = _fetch_plddt_for_fragment(uniprot_id, fragment, af_dir)
    if frag_result is None:
        return None
    bfactors, resnames = frag_result
    if not bfactors:
        return None

    # Sequence verification: confirm first 5 residues match before trusting coords
    if domainSeq and resnames:
        _verifyDomainSeq(resnames, localStart, domainSeq)

    scores = [bfactors[r] for r in range(localStart, localEnd + 1) if r in bfactors]
    if not scores:
        return None
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Per-domain scoring
# ---------------------------------------------------------------------------

def score_sequence(seq: str, predictors: list[Predictor]) -> dict[str, float]:
    """Run every available predictor on one sequence and return a dict of
    per-predictor summary stats (mean score, fraction disordered). Missing
    or failed predictions come back as NaN."""
    out: dict[str, float] = {}
    for p in predictors:
        mean_col = f'{p.name}_mean_disorder'
        frac_col = f'{p.name}_fraction_disordered'
        if not p.available or not isinstance(seq, str) or not seq.strip():
            out[mean_col] = np.nan
            out[frac_col] = np.nan
            continue
        try:
            scores = p.predict(seq)
            if scores.size == 0:
                out[mean_col] = np.nan
                out[frac_col] = np.nan
            else:
                out[mean_col] = float(np.mean(scores))
                out[frac_col] = float(np.sum(scores > 0.5) / len(scores))
        except Exception as e:
            warnings.warn(f'{p.name} failed on a sequence of length {len(seq)}: {e}')
            out[mean_col] = np.nan
            out[frac_col] = np.nan
    return out


# ---------------------------------------------------------------------------
# Consensus columns
# ---------------------------------------------------------------------------

# Thresholds for the structured-domain filter — exposed as module constants
# so they're easy to find and tweak.
MEAN_THRESHOLD = 0.5
FRACTION_THRESHOLD = 0.2

# pLDDT threshold: domains with mean pLDDT >= this are considered structured.
# 70 is AlphaFold's "confident" cutoff (light blue); 80 is stricter.
PLDDT_THRESHOLD = 70.0
ALPHAFOLD_VERSION = "v6"


def add_consensus_columns(df: pd.DataFrame, predictor_names: list[str],
                          plddt_threshold: float = PLDDT_THRESHOLD) -> pd.DataFrame:
    """Add the mean-ensemble and intersection-of-filters columns derived from
    the per-predictor columns. Uses only predictors that produced values.
    pLDDT is included in the intersection check when the column is present.
    """
    mean_cols = [f'{n}_mean_disorder'       for n in predictor_names if f'{n}_mean_disorder'       in df.columns]
    frac_cols = [f'{n}_fraction_disordered' for n in predictor_names if f'{n}_fraction_disordered' in df.columns]

    if mean_cols:
        df['mean_ensemble_mean_disorder']       = df[mean_cols].mean(axis=1, skipna=True)
    if frac_cols:
        df['mean_ensemble_fraction_disordered'] = df[frac_cols].mean(axis=1, skipna=True)

    # Intersection: a row passes only if EVERY predictor independently passes
    # the structured filter. Predictors that returned NaN are treated as
    # "didn't pass" so we don't accidentally keep a row that two predictors
    # rejected just because the third crashed.
    passes_per_predictor = []
    for n in predictor_names:
        mc = f'{n}_mean_disorder'
        fc = f'{n}_fraction_disordered'
        if mc in df.columns and fc in df.columns:
            passes = ((df[mc] <= MEAN_THRESHOLD) & (df[fc] <= FRACTION_THRESHOLD)).fillna(False)
            passes_per_predictor.append(passes)

    # Include pLDDT in intersection when available (NaN = no AF model → fails).
    if 'plddt_mean_domain' in df.columns:
        plddt_passes = df['plddt_mean_domain'].notna() & (df['plddt_mean_domain'] >= plddt_threshold)
        passes_per_predictor.append(plddt_passes)

    if passes_per_predictor:
        intersection = passes_per_predictor[0].copy()
        for p in passes_per_predictor[1:]:
            intersection &= p
        df['intersection_passes_filter'] = intersection.astype(int)

    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

FILTER_TARGETS = {
    # target name -> (mean_col, fraction_col)   OR   sentinel string
    'metapredict':          ('metapredict_mean_disorder',        'metapredict_fraction_disordered'),
    'aiupred':              ('aiupred_mean_disorder',            'aiupred_fraction_disordered'),
    'iupred3':              ('iupred3_mean_disorder',            'iupred3_fraction_disordered'),
    'mean_ensemble':        ('mean_ensemble_mean_disorder',      'mean_ensemble_fraction_disordered'),
    'intersection':         ('intersection', None),
    'plddt':                ('plddt', None),         # special-cased in apply_filter
    'metapredict_and_plddt': ('metapredict_and_plddt', None),  # special-cased in apply_filter
    'none':                 (None, None),
}


def apply_filter(df: pd.DataFrame, target: str, mode: str,
                 plddt_threshold: float = PLDDT_THRESHOLD) -> pd.DataFrame:
    if mode == 'none' or target == 'none':
        return df
    mean_col, frac_col = FILTER_TARGETS[target]

    if target == 'intersection':
        # 'intersection' is a pre-computed 0/1 column already encoding the
        # structured-domain filter. For 'disordered' mode we don't have an
        # equivalent intersection column (all predictors agreeing a domain is
        # disordered is a different computation) — so we refuse that combo.
        if mode == 'disordered':
            sys.exit("ERROR: --filter-on intersection only works with "
                     "--filter-mode structured. For disordered-mode "
                     "consensus, use --filter-on mean_ensemble.")
        if 'intersection_passes_filter' not in df.columns:
            sys.exit("ERROR: intersection column missing — no predictors "
                     "produced values, so there's nothing to intersect.")
        return df[df['intersection_passes_filter'] == 1].copy()

    if target == 'plddt':
        if 'plddt_mean_domain' not in df.columns:
            sys.exit("ERROR: plddt_mean_domain column missing — pLDDT scoring "
                     "was skipped (pass --skip plddt to suppress this error "
                     "or remove --filter-on plddt).")
        valid = df['plddt_mean_domain'].notna()
        if mode == 'structured':
            return df[valid & (df['plddt_mean_domain'] >= plddt_threshold)].copy()
        elif mode == 'disordered':
            return df[valid & (df['plddt_mean_domain'] < plddt_threshold)].copy()
        else:
            sys.exit(f"ERROR: unknown --filter-mode '{mode}'")

    if target == 'metapredict_and_plddt':
        mc = 'metapredict_mean_disorder'
        fc = 'metapredict_fraction_disordered'
        for col in (mc, fc):
            if col not in df.columns:
                sys.exit(f"ERROR: column '{col}' missing — metapredict probably "
                         "failed to load.")
        if df[mc].isna().all():
            sys.exit("ERROR: metapredict_and_plddt filter selected but all "
                     "metapredict scores are NaN — metapredict failed to load "
                     "or produced no output. Install it with:\n"
                     "  pip install metapredict\n"
                     "Or use --filter-on plddt to filter on pLDDT alone.")
        if 'plddt_mean_domain' not in df.columns:
            sys.exit("ERROR: plddt_mean_domain column missing — pLDDT scoring "
                     "was skipped (pass --skip plddt to disable pLDDT or "
                     "choose a different --filter-on).")
        meta_ok  = (df[mc] <= MEAN_THRESHOLD) & (df[fc] <= FRACTION_THRESHOLD)
        plddt_ok = df['plddt_mean_domain'].notna() & (df['plddt_mean_domain'] >= plddt_threshold)
        if mode == 'structured':
            return df[meta_ok & plddt_ok].copy()
        elif mode == 'disordered':
            meta_dis  = (df[mc] > MEAN_THRESHOLD) & (df[fc] > FRACTION_THRESHOLD)
            plddt_dis = df['plddt_mean_domain'].notna() & (df['plddt_mean_domain'] < plddt_threshold)
            return df[meta_dis & plddt_dis].copy()
        else:
            sys.exit(f"ERROR: unknown --filter-mode '{mode}'")

    if mean_col not in df.columns or frac_col not in df.columns:
        sys.exit(f"ERROR: filter target '{target}' requires columns "
                 f"'{mean_col}' and '{frac_col}', which are missing — the "
                 f"relevant predictor probably failed to load.")

    if mode == 'structured':
        return df[(df[mean_col] <= MEAN_THRESHOLD) &
                  (df[frac_col] <= FRACTION_THRESHOLD)].copy()
    elif mode == 'disordered':
        return df[(df[mean_col] >  MEAN_THRESHOLD) &
                  (df[frac_col] >  FRACTION_THRESHOLD)].copy()
    else:
        sys.exit(f"ERROR: unknown --filter-mode '{mode}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input',  required=True, type=Path,
                    help='Input TSV (typically 1_domainLibraryRaw.tsv)')
    ap.add_argument('--output', required=True, type=Path,
                    help='Output TSV path')
    ap.add_argument('--output-all', type=Path, default=None,
                    help='Output TSV path for the FULL annotated library '
                         '(every domain, filtered or not, with all predictor '
                         'columns). Defaults to the --output filename with '
                         '"_all" inserted before the extension, e.g. '
                         'foo.tsv -> foo_all.tsv. Pass explicitly to '
                         'override, or pass an empty string to skip writing '
                         'the full library.')
    ap.add_argument('--iupred3-dir', type=Path, default=None,
                    help='Directory containing iupred3_lib.py and its data/ '
                         'folder. If omitted, falls back to '
                         'IUPRED3_DIR_DEFAULT at the top of this script. '
                         'Leave both unset if iupred3_lib is already on your '
                         'PYTHONPATH.')
    ap.add_argument('--iupred3-mode', choices=['long', 'short', 'glob'],
                    default='long', help='IUPred3 prediction mode')
    ap.add_argument('--iupred3-smoothing', choices=['no', 'medium', 'strong'],
                    default='medium', help='IUPred3 smoothing')
    ap.add_argument('--filter-on', choices=list(FILTER_TARGETS.keys()),
                    default='metapredict_and_plddt',
                    help='Which predictor/consensus column set to filter on '
                         '(default: metapredict_and_plddt)')
    ap.add_argument('--filter-mode',
                    choices=['structured', 'disordered', 'none'],
                    default='structured',
                    help='Keep structured domains (default), disordered '
                         'regions, or skip filtering entirely')
    ap.add_argument('--skip', action='append', default=None,
                    choices=['metapredict', 'aiupred', 'iupred3', 'plddt'],
                    help='Skip a predictor (repeatable). By default aiupred and '
                         'iupred3 are skipped (only metapredict + pLDDT run). '
                         'Passing any --skip flag replaces the default entirely, '
                         'so --skip plddt runs all three disorder predictors '
                         'without pLDDT, and --skip aiupred --skip iupred3 '
                         'restores the default explicitly.')
    ap.add_argument('--af-dir', type=Path, default=None, metavar='DIR',
                    help='Optional directory of cached AlphaFold PDB files '
                         '(e.g. the --af-dir used in steps 3 & 4). Checked '
                         'before downloading from AlphaFold EBI. Supports '
                         'both .pdb and .pdb.gz files. Handles proteins with '
                         '>1400 AA by selecting the correct fragment.')
    ap.add_argument('--plddt-threshold', type=float, default=PLDDT_THRESHOLD,
                    help=f'Mean pLDDT threshold for the structured-domain '
                         f'filter (default: {PLDDT_THRESHOLD}). Domains with '
                         f'mean pLDDT >= this pass. AlphaFold color boundaries: '
                         f'70 = confident, 80 = high confidence.')
    args = ap.parse_args()

    # If no --skip flags were given, apply the default: run only metapredict + pLDDT.
    # Any explicit --skip flag(s) replace this default entirely.
    skip = args.skip if args.skip is not None else ['aiupred', 'iupred3']

    # Apply the hardcoded default from the top of the script when the CLI
    # flag was not explicitly provided.
    iupred3_dir = args.iupred3_dir
    if iupred3_dir is None and IUPRED3_DIR_DEFAULT is not None:
        iupred3_dir = Path(IUPRED3_DIR_DEFAULT)

    # --- Load predictors ---------------------------------------------------
    all_predictors: list[Predictor] = []
    if 'metapredict' not in skip:
        all_predictors.append(load_metapredict())
    if 'aiupred' not in skip:
        all_predictors.append(load_aiupred())
    if 'iupred3' not in skip:
        all_predictors.append(load_iupred3(iupred3_dir,
                                           mode=args.iupred3_mode,
                                           smoothing=args.iupred3_smoothing))

    run_plddt = 'plddt' not in skip

    print('Predictor status:')
    for p in all_predictors:
        if p.available:
            print(f'  [OK]     {p.name}')
        else:
            print(f'  [SKIP]   {p.name}   ({p.error})')
    if run_plddt:
        print(f'  [OK]     plddt  (AlphaFold EBI, streamed; threshold={args.plddt_threshold})')
    else:
        print('  [SKIP]   plddt  (--skip plddt)')
    available = [p for p in all_predictors if p.available]
    if not available and not run_plddt:
        sys.exit('ERROR: no predictors could be loaded. Nothing to do.')

    # --- Load input --------------------------------------------------------
    df = pd.read_csv(args.input, sep='\t')
    if 'Domain Sequence' not in df.columns:
        sys.exit("ERROR: input TSV has no 'Domain Sequence' column.")
    if run_plddt:
        for col in ('Entry', 'Start', 'End'):
            if col not in df.columns:
                sys.exit(f"ERROR: pLDDT scoring requires column '{col}' in the "
                         f"input TSV. Use --skip plddt to disable pLDDT.")
    print(f'\nLoaded {len(df):,} domain sequences from {args.input}')

    # --- Score every sequence (disorder predictors) -----------------------
    # We collect rows as list-of-dicts and attach them to df at the end; this
    # is much faster than repeated df.at[idx, col] = ... assignments.
    results = []
    for i, seq in enumerate(df['Domain Sequence'], start=1):
        if i % 200 == 0 or i == len(df):
            print(f'  scored {i:,}/{len(df):,}')
        results.append(score_sequence(seq, all_predictors))
    scores_df = pd.DataFrame(results, index=df.index)
    df = pd.concat([df, scores_df], axis=1)

    # --- Score pLDDT per domain -------------------------------------------
    if run_plddt:
        af_dir = args.af_dir
        src = f"local cache ({af_dir}) + AlphaFold EBI" if af_dir else "AlphaFold EBI"
        print(f'\nFetching pLDDT scores ({src}) ...')
        plddt_scores = []
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            if i % 200 == 0 or i == len(df):
                print(f'  pLDDT {i:,}/{len(df):,}')
            plddt_scores.append(
                score_plddt_domain(str(row['Entry']),
                                   int(row['Start']), int(row['End']),
                                   int(row['Length']),
                                   domainSeq=str(row.get('Domain Sequence', '')),
                                   af_dir=af_dir))
        df['plddt_mean_domain'] = plddt_scores

    # --- Derived consensus columns ----------------------------------------
    predictor_names = [p.name for p in available]
    df = add_consensus_columns(df, predictor_names,
                               plddt_threshold=args.plddt_threshold)

    # --- Filter ------------------------------------------------------------
    n_before = len(df)
    df_filtered = apply_filter(df, args.filter_on, args.filter_mode,
                               plddt_threshold=args.plddt_threshold)
    n_after = len(df_filtered)
    print(f'\nFilter: --filter-on {args.filter_on} --filter-mode {args.filter_mode}')
    print(f'  {n_before:,} -> {n_after:,} domain sequences')

    # --- Write -------------------------------------------------------------

    # Two outputs: the filtered library (--output) and the full annotated
    # library with every domain + predictor column (--output-all).
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(args.output, sep='\t', index=False)
    print(f'\nWrote filtered library:  {args.output}  ({len(df_filtered):,} rows)')

    # Eliminated library: rows that did not pass the filter, same columns.
    if args.filter_mode != 'none' and args.filter_on != 'none':
        df_eliminated = df[~df.index.isin(df_filtered.index)].copy()
        elim_output = args.output.with_name(
            args.output.stem + '_eliminated' + args.output.suffix)
        elim_output.parent.mkdir(parents=True, exist_ok=True)
        df_eliminated.to_csv(elim_output, sep='\t', index=False)
        print(f'Wrote eliminated library: {elim_output}  ({len(df_eliminated):,} rows)')
 
    # Decide where the "all" file goes.
    if args.output_all is None:
        # Insert "_all" before the extension:
        #   foo/bar.tsv  -> foo/bar_all.tsv
        #   foo/bar      -> foo/bar_all   (no extension case)
        out_all = args.output.with_name(args.output.stem + '_all' + args.output.suffix)
    elif str(args.output_all) == '':
        out_all = None   # user explicitly suppressed the all-file
    else:
        out_all = args.output_all
 
    # Skip writing the all-file when the filter wasn't actually applied —
    # otherwise --output and --output-all would be identical.
    if args.filter_mode == 'none':
        print('Filter mode is "none"; full library already lives in --output, '
              'skipping --output-all.')
    elif out_all is not None:
        out_all.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_all, sep='\t', index=False)
        print(f'Wrote full library:      {out_all}  ({len(df):,} rows)')


if __name__ == '__main__':
    main()
