import argparse
import gzip
import math
import requests
import os
import sys
import json
from pathlib import Path
import numpy as np
import freesasa
# Silence FreeSASA's per-atom warnings about UNK / modified residues.
# These are informational ("guessing radius for unknown atom") and would
# otherwise flood the log with thousands of lines for experimental PDBs
# that contain UNK backbones or non-standard residues. Errors still print.
freesasa.setVerbosity(freesasa.nowarnings)
from Bio import PDB
from scipy.spatial.distance import cdist
import pandas as pd

# structure_source_fixedmapping.py lives one directory above code/.
# Adding the parent dir to sys.path lets us import it without packaging.
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))
from structure_source_fixedmapping import (
    resolve_domain_structure, load_uniprot_sequences
)

# Domain purity threshold: when an experimental PDB covers most of the
# domain residues AND most of those residues belong to the domain (i.e. the
# structure is essentially "the domain alone"), we set highPurityFlag = True
# so step 5 can treat the row specially. fb/cd are still computed from the AF
# PDB so cross-row comparisons stay consistent.
HIGH_PURITY_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Fragment helpers
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

def getAFFragment(domainStart, domainEnd, proteinLength):
    '''Return the AF fragment number containing [domainStart, domainEnd].
    Proteins <=2700 AA are not fragmented (always F1, local = global).
    For fragmented proteins: step=200, F1=1-1400, F2=201-1600, ...
    Returns None if the domain spans a boundary and cannot fit in one fragment.'''
    if proteinLength <= 2700:
        return 1
    # Smallest n where (n-1)*200 + 1400 >= domainEnd
    n = max(1, math.ceil((domainEnd - 1400) / FRAG_STEP) + 1)
    # Verify domainStart also fits inside fragment n
    if (n - 1) * FRAG_STEP + 1 > domainStart:
        return None  # domain spans a fragment boundary
    return n

def globalToLocal(pos, fragment):
    '''Convert a global UniProt residue position to a local fragment position.
    F1: offset 0 (local = global), F2: offset 200, F3: offset 400, ...'''
    return pos - (fragment - 1) * FRAG_STEP

def verifyDomainInFragment(pdbFile, localStart, domainSeq, nCheck=5):
    '''Confirm first nCheck residues of domainSeq match PDB residues at localStart.
    Uses BioPython. Returns True on match or if verification cannot be performed.'''
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("verify", pdbFile)
        chain = next(structure[0].get_chains())
        pdbResidues = {}
        for res in chain:
            rid = res.id[1]
            if localStart <= rid < localStart + nCheck:
                pdbResidues[rid] = _AA3TO1.get(res.resname, 'X')
        pdbSeqStr   = ''.join(pdbResidues.get(r, '?') for r in range(localStart, localStart + nCheck))
        domainCheck = domainSeq[:nCheck].upper()
        if '?' not in pdbSeqStr and pdbSeqStr == domainCheck:
            return True
        print(f"  Sequence mismatch at local {localStart}: PDB '{pdbSeqStr}' vs domain '{domainCheck}'")
        return False
    except Exception as e:
        print(f"  Warning: sequence verification failed ({e}); proceeding without check")
        return True

def _ensureUnzipped(gzPath, destPath):
    '''Decompress gzPath → destPath (skipped if destPath already exists). Returns destPath.'''
    if not os.path.exists(destPath):
        with gzip.open(gzPath, 'rb') as gz, open(destPath, 'wb') as out:
            out.write(gz.read())
    return destPath

# ---------------------------------------------------------------------------
# AlphaFold file retrieval  (local cache first, then URL)
# ---------------------------------------------------------------------------

def getAlphaFoldFiles(uniprotID, fragment, outputDir):
    '''Return paths to an unzipped PDB and PAE JSON for the given entry + fragment.
    Checks outputDir for cached files (unzipped or .gz) before downloading.
    Returns ("", "") if either file cannot be obtained.'''

    # Determine canonical output paths for unzipped files
    if fragment == 1:
        pdbDest = os.path.join(outputDir, f"{uniprotID}_F1_model.pdb")
        paeDest = os.path.join(outputDir, f"{uniprotID}_F1_PAE.json")
    else:
        pdbDest = os.path.join(outputDir, f"AF-{uniprotID}-F{fragment}-model_v6.pdb")
        paeDest = os.path.join(outputDir, f"AF-{uniprotID}-F{fragment}-PAE.json")

    def _resolveFile(destPath, urlPath, label):
        '''Return destPath if already present; decompress .gz if available; else download.'''
        if os.path.exists(destPath):
            print(f"  Using local {label}: {destPath}")
            return destPath
        # Check .gz variants in af-dir (both canonical and AF DB naming)
        gzCandidates = [
            destPath + ".gz",
            os.path.join(outputDir, f"AF-{uniprotID}-F{fragment}-model_v6.pdb.gz")
                if label == "PDB" else
            os.path.join(outputDir, f"AF-{uniprotID}-F{fragment}-predicted_aligned_error_v6.json.gz"),
        ]
        for gzPath in gzCandidates:
            if os.path.exists(gzPath):
                print(f"  Decompressing local {label}: {gzPath}")
                return _ensureUnzipped(gzPath, destPath)
        # Fall back to URL download
        try:
            response = requests.get(urlPath, stream=True)
            response.raise_for_status()
            with open(destPath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  Downloaded {label} for {uniprotID} F{fragment}")
            return destPath
        except requests.exceptions.RequestException as e:
            print(f"  Failed to obtain {label} for {uniprotID} F{fragment}: {e}")
            return ""

    pdbUrl = f"https://alphafold.ebi.ac.uk/files/AF-{uniprotID}-F{fragment}-model_v6.pdb"
    paeUrl = f"https://alphafold.ebi.ac.uk/files/AF-{uniprotID}-F{fragment}-predicted_aligned_error_v6.json"

    pdbPath = _resolveFile(pdbDest, pdbUrl, "PDB")
    paePath = _resolveFile(paeDest, paeUrl, "PAE")
    return paePath, pdbPath

# ---------------------------------------------------------------------------
# Domain metric functions  (all accept LOCAL residue coords)
# ---------------------------------------------------------------------------

def exciseDomainPDB(inputPDB, outputPDB, domainStart, domainEnd):
    '''Extract and save only the structural information of a specific domain.
    domainStart/domainEnd must be LOCAL fragment residue numbers.'''
    if os.path.exists(outputPDB):
        print(f"Domain PDB already exists: {outputPDB}")
        return outputPDB

    parser=PDB.PDBParser(QUIET=True)
    fullProtein=parser.get_structure("protein", inputPDB)
    proteinModel=fullProtein[0]
    proteinChain=next(proteinModel.get_chains())

    domain=PDB.Structure.Structure("domain")
    domainModel=PDB.Model.Model(0)
    domain.add(domainModel)
    domainChain=PDB.Chain.Chain(proteinChain.id)

    for residue in proteinChain:
        residueNum=residue.id[1]
        if domainStart <= residueNum <= domainEnd:
            domainChain.add(residue.copy())

    domainModel.add(domainChain)
    io=PDB.PDBIO()
    io.set_structure(domain)
    io.save(outputPDB)
    print(f"New domain saved to: {outputPDB}")

def anchoringIndex(paeFile, domainStart, domainEnd):
    '''Uses the PAE matrix to measure how constrained the domain is by its parent fragment.
    domainStart/domainEnd must be LOCAL fragment residue numbers.

    Returns NaN (rather than raising) when the domain extends past the PAE
    matrix — this happens when the input TSV's protein length / domain
    boundaries are stale relative to the AlphaFold DB v6 entry (e.g. a
    UniProt isoform change). The caller decides what to do with NaN.
    '''
    with open(paeFile, "r") as f:
        data=json.load(f)
    paeMatrix=np.array(data[0]["predicted_aligned_error"])
    matrix_size = paeMatrix.shape[0]

    # Defensive: domain residues must fit within the PAE matrix.
    if domainEnd > matrix_size or domainStart < 1:
        print(f"  WARNING: PAE matrix is {matrix_size}x{matrix_size} but domain "
              f"is {domainStart}-{domainEnd} — likely input/AF length mismatch. "
              f"anchoringIndex set to NaN.")
        return float('nan')

    domainResidues=np.arange(domainStart-1, domainEnd)        # 0-indexed local
    proteinResidues=np.setdiff1d(np.arange(matrix_size), domainResidues)
    if len(proteinResidues) == 0:
        return float('nan')
    paeBetween=paeMatrix[np.ix_(domainResidues, proteinResidues)]
    lowPAE=paeBetween<=5

    perResidueAnchoring=np.sum(lowPAE, axis=1)/len(proteinResidues)
    return np.mean(perResidueAnchoring)

def fractionBuried(pdbFile, domainPDBFile, domainStart, domainEnd):
    '''Calculates the fraction of a domain buried within its parent fragment.
    domainStart/domainEnd must be LOCAL fragment residue numbers.'''
    fullProtein=freesasa.Structure(pdbFile)
    fpResult=freesasa.calc(fullProtein)
    domainInProtein=freesasa.selectArea(
        [f'r{domainStart}_{domainEnd}, resi {domainStart}-{domainEnd}'],
        fullProtein, fpResult)
    domainInProteinSASA=domainInProtein[f'r{domainStart}_{domainEnd}']

    domainOnly=freesasa.Structure(domainPDBFile)
    doResult=freesasa.calc(domainOnly)
    domainOnlySASA=doResult.totalArea()
    deltaSASA=domainOnlySASA-domainInProteinSASA
    return deltaSASA/domainOnlySASA

def contactDensity(pdbFile, domainStart, domainEnd):
    '''Quantifies interactions between the domain and the rest of its parent fragment.
    domainStart/domainEnd must be LOCAL fragment residue numbers.'''
    parser=PDB.PDBParser(QUIET=True)
    fullProtein=parser.get_structure("protein", pdbFile)
    proteinChain=next(fullProtein[0].get_chains())

    domainResidues=[res for res in proteinChain if domainStart <= res.get_id()[1] <= domainEnd]
    otherResidues=[res for res in proteinChain if res.get_id()[1] < domainStart or res.get_id()[1] > domainEnd]

    if not domainResidues or not otherResidues:
        print(f"  contactDensity: no {'domain' if not domainResidues else 'non-domain'} residues found "
              f"(local {domainStart}-{domainEnd}); returning NaN")
        return float('nan')

    cutoff=4.0
    totalContacts=0
    for domainResidue in domainResidues:
        domainCoordinates=np.array([atom.get_coord() for atom in domainResidue])
        for otherResidue in otherResidues:
            otherCoordinates=np.array([atom.get_coord() for atom in otherResidue])
            if np.any(cdist(domainCoordinates, otherCoordinates) < cutoff):
                totalContacts+=1

    #return totalContacts/(len(domainResidues)*len(otherResidues))
    return totalContacts/len(domainResidues)

def plddtMean(domainPDBFile):
    '''Returns the mean pLDDT (B-factor) of all residues in the domain PDB.'''
    parser=PDB.PDBParser(QUIET=True)
    domain=parser.get_structure("domain", domainPDBFile)
    domainChain=next(domain[0].get_chains())

    plddtValues=[]
    for residue in domainChain:
        atomBfactors=[atom.get_bfactor() for atom in residue]
        plddtValues.append(sum(atomBfactors)/len(atomBfactors))
    return round(sum(plddtValues)/len(plddtValues), 2)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ap = argparse.ArgumentParser(description='Step 3: Compute AlphaFold-based domain interaction metrics.')
_ap.add_argument('--input',
                 default='/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/kat_output_library_files/04212026_metapredict/2_domainLibraryStructuredSeq_meta.tsv',
                 help='Input TSV (step 2 output)')
_ap.add_argument('--output',
                 default='/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/kat_output_library_files/04282026_metapredict/3_domainLibraryInteractions_meta.tsv',
                 help='Output TSV path')
_ap.add_argument('--af-dir',
                 default='/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/alphaFold/dbFiles',
                 help='Directory for AlphaFold PDB/PAE files (checked before downloading)')
_ap.add_argument('--struct-cache-dir',
                 default='/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/struct_cache',
                 help='Local cache for structure_source (experimental PDBs, SIFTS JSON, etc.)')
_ap.add_argument('--experimental-mode',
                 default='experimental_preferred',
                 choices=['experimental_preferred', 'experimental_only', 'alphafold_only'],
                 help='Structure source policy passed to resolve_domain_structure '
                      '(default: experimental_preferred — use the best PDB if any '
                      'covers the domain, else fall back to AlphaFold).')
_ap.add_argument('--min-domain-coverage', type=float, default=0.8,
                 help='Min fraction of domain residues that must be present in '
                      'a candidate PDB for it to be accepted (default 0.8).')
_ap.add_argument('--uniprot-tsv', default=None,
                 help='Optional UniProt proteome TSV with Entry+Sequence columns. '
                      'Enables the alignment fallback inside structure_source when '
                      'SIFTS has null auth values.')
_args = _ap.parse_args()
input        = _args.input
output       = _args.output
outputDir    = _args.af_dir
structCache  = Path(_args.struct_cache_dir)
expMode      = _args.experimental_mode
minDomainCov = _args.min_domain_coverage
uniprotTSV   = Path(_args.uniprot_tsv) if _args.uniprot_tsv else None
os.makedirs(outputDir, exist_ok=True)
structCache.mkdir(parents=True, exist_ok=True)

# Pre-load UniProt sequences once (for the alignment fallback inside
# structure_source). load_uniprot_sequences is internally cached by path,
# so it's safe to call multiple times.
_uniprot_seqs = load_uniprot_sequences(uniprotTSV) if uniprotTSV else {}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

df=pd.read_csv(input, sep="\t")
# --- AlphaFold-derived metrics (always populated when AF PDB available) ----
df["anchoringIndex"]=None
df["fractionBuried"]=None
df["contactDensity"]=None
df["interactionIndex"]=None
# --- Experimental-PDB-derived metrics (populated only when a usable
#     experimental structure exists AND highPurityFlag is False).
#     exp_interactionIndex uses the SAME formula as interactionIndex
#     (anchoringIndex from AF PAE, fb/cd from the experimental full chain) so
#     the two indices are directly comparable per-row. Downstream analyses
#     should default to exp_* when available and fall back to AF otherwise.
df["exp_fractionBuried"]=None
df["exp_contactDensity"]=None
df["exp_interactionIndex"]=None
df["pdbID"]=None             # experimental PDB id when one is picked; else None
df["structureInfo"]=None     # JSON string: {method, resolution, domainCoverage, domainPurity}
                             # or '{"source": "AF"}' when only AlphaFold is used.
df["highPurityFlag"]=False   # True iff experimental was picked AND domainPurity > HIGH_PURITY_THRESHOLD
df["loose_align_fragment"]=None  # Fragment chosen by structure_source's residue-range scan
                                 # (for comparison against hardcoded_fragment from step 2)
df["structureNotes"]=None    # Audit trail from structure_source (alignment used, etc.)

# ---------------------------------------------------------------------------
# hardcoded_fragment backfill
# ---------------------------------------------------------------------------
# Step 2 writes a `hardcoded_fragment` column with the AF fragment that
# contains each domain. When the user skips step 2 (e.g. running step 3
# directly on step-1 output) or the column exists but has NaN values, we
# need to compute the fragment here using the same algorithm.
#
# Doing this upfront — rather than per-row in the main loop — means:
#   - the column is always populated in the output TSV (even when step 2
#     was skipped), keeping downstream consumers consistent
#   - a single log line tells the user how many rows had to be filled
#   - the main loop simplifies to a single `int(sequence["hardcoded_fragment"])`

def backfillHardcodedFragments(frame):
    """Ensure `hardcoded_fragment` exists and is populated for every row.
    Rows whose domain spans a fragment boundary keep NaN (getAFFragment
    returns None for those — they get skipped in the main loop)."""
    if 'hardcoded_fragment' not in frame.columns:
        print("No 'hardcoded_fragment' column found (step 2 was skipped?). "
              "Computing on-the-fly using step 2's algorithm.")
        frame['hardcoded_fragment'] = None

    missing = frame['hardcoded_fragment'].isna()
    n_missing = int(missing.sum())
    if n_missing == 0:
        return frame  # nothing to do

    print(f"Backfilling hardcoded_fragment for {n_missing:,} rows "
          f"({n_missing/len(frame)*100:.1f}% of input).")
    for idx in frame.index[missing]:
        s = int(frame.at[idx, 'Start'])
        e = int(frame.at[idx, 'End'])
        L = int(frame.at[idx, 'Length'])
        frame.at[idx, 'hardcoded_fragment'] = getAFFragment(s, e, L)
    n_boundary = int(frame['hardcoded_fragment'].isna().sum())
    if n_boundary:
        print(f"  {n_boundary:,} of those still NaN (domain spans a fragment "
              f"boundary; will be skipped in the main loop).")
    return frame

df = backfillHardcodedFragments(df)

# Quick lookup of the lengthMismatch column from the UniProt↔AF audit
# (audit_uniprot_vs_af.py + tag_length_mismatch.py). Rows where this is
# True will be skipped: the AF structure is for a different (older/shorter)
# isoform than the input TSV's domain coordinates, so any computed metric
# is silently wrong. Skipping leaves all metric columns NaN for those rows
# but preserves the row itself for traceability downstream.
HAS_LEN_MISMATCH_COL = 'lengthMismatch' in df.columns
if HAS_LEN_MISMATCH_COL:
    n_tagged = int(df['lengthMismatch'].fillna(False).astype(bool).sum())
    print(f"Input has lengthMismatch column — will skip {n_tagged:,} tagged rows.")

# ---------------------------------------------------------------------------
# Checkpoint / resumability
# ---------------------------------------------------------------------------
# Streaming append: every row's state is flushed to disk as soon as that
# row's computation finishes (or the per-row try/finally fires). If the
# script crashes 17 hours in, the next invocation reads the checkpoint and
# skips already-completed rows — no compute is lost.
#
# Cost: one TSV line write per row (~50 microseconds vs. ~seconds of
# per-row computation, so <0.01% overhead).
#
# The checkpoint lives next to the final output and is deleted on
# successful completion. Don't manually delete it mid-run; if the script
# crashes, leave it alone so the next run can resume.
outputPath = Path(output)
checkpointPath = outputPath.with_name(
    outputPath.stem + '_inProgress' + outputPath.suffix)

EXPECTED_COLUMNS = list(df.columns)
completed_keys: set[tuple[str, int, int]] = set()

if checkpointPath.exists() and checkpointPath.stat().st_size > 0:
    try:
        prev = pd.read_csv(checkpointPath, sep='\t')
    except Exception as e:
        print(f"ERROR: failed to read existing checkpoint {checkpointPath}: {e}\n"
              f"Delete the checkpoint and re-run to start fresh.")
        sys.exit(1)
    if list(prev.columns) != EXPECTED_COLUMNS:
        print(f"ERROR: checkpoint {checkpointPath} has a different column "
              f"schema than this version of the script expects.\n"
              f"  checkpoint columns: {list(prev.columns)}\n"
              f"  expected columns:   {EXPECTED_COLUMNS}\n"
              "Delete the checkpoint and re-run, or restore the previous "
              "version of the script.")
        sys.exit(1)
    completed_keys = {(str(r.Entry), int(r.Start), int(r.End))
                      for r in prev.itertuples(index=False)}
    print(f"Resuming — {len(completed_keys):,} rows already in checkpoint "
          f"{checkpointPath}.")

# Open checkpoint in append+line-buffered mode. Line buffering means each
# row's TSV line is flushed to the OS as soon as it ends with '\n'.
need_header = (not checkpointPath.exists()
               or checkpointPath.stat().st_size == 0)
ck_file = open(checkpointPath, 'a', buffering=1)
if need_header:
    # Write header only — use an empty slice of df so the columns are
    # written but no rows are.
    df.iloc[:0].to_csv(ck_file, sep='\t', header=True, index=False)


def writeRowToCheckpoint(row_idx):
    """Append row `row_idx`'s current state to the checkpoint TSV.
    Called before every `continue` and at the end of each iteration so
    every row that we visited (whether fully computed, skipped, or errored)
    is recorded — that's what makes the run resumable. Pandas handles
    quoting/escaping for JSON-string columns automatically."""
    df.iloc[[row_idx]].to_csv(ck_file, sep='\t', header=False, index=False)


for idx, sequence in df.iterrows():
    domainStart=int(sequence["Start"])
    domainEnd=int(sequence["End"])
    proteinLength=int(sequence["Length"])
    entry=sequence["Entry"]

    # ----- Checkpoint: skip rows already completed in a prior run ----------
    row_key = (str(entry), domainStart, domainEnd)
    if row_key in completed_keys:
        continue   # already in checkpoint — don't recompute, don't rewrite

    # ----- Skip rows the audit flagged as UniProt/AF length-mismatched -----
    # These would either crash (when domain extends past AF coverage) or
    # silently miscompute (when residue indices happen to land in-range
    # but refer to a different isoform's amino acids). Either way the
    # metrics aren't trustworthy, so leave them NaN.
    if HAS_LEN_MISMATCH_COL and bool(sequence.get("lengthMismatch", False)):
        print(f"  Skipping {entry} {domainStart}-{domainEnd}: "
              f"lengthMismatch=True (AF model is for a different sequence "
              f"version than the input TSV).")
        writeRowToCheckpoint(idx)
        continue

    # ----- Fragment selection ------------------------------------------------
    # The hardcoded_fragment column is guaranteed populated by the backfill
    # above (whether step 2 wrote it or we computed it). NaN only remains
    # for domains that span a fragment boundary, which we skip.
    raw_frag = sequence.get("hardcoded_fragment")
    if raw_frag is None or pd.isna(raw_frag):
        print(f"  Skipping {entry} domain {domainStart}-{domainEnd}: "
              f"spans a fragment boundary (domain may be longer than 1200 AA).")
        writeRowToCheckpoint(idx)
        continue
    fragment = int(raw_frag)
    localStart=globalToLocal(domainStart, fragment)
    localEnd  =globalToLocal(domainEnd,   fragment)
    print(f"Processing {entry} domain {domainStart}-{domainEnd} "
          f"(protein length {proteinLength}, F{fragment}, local {localStart}-{localEnd})")

    # ----- Resolve structure source via structure_source --------------------
    # Always called (even in alphafold_only) so we get the loose_align_fragment
    # value + audit trail for free.
    uniprot_seq = _uniprot_seqs.get(entry)
    try:
        choice = resolve_domain_structure(
            entry, domainStart, domainEnd, structCache,
            mode=expMode,
            min_domain_coverage=minDomainCov,
            local_alphafold_dir=Path(outputDir),
            uniprot_sequence=uniprot_seq,
        )
    except Exception as e:
        print(f"  structure_source lookup failed for {entry}: {e}; continuing with AF only.")
        choice = None

    if choice is not None:
        df.at[idx, "loose_align_fragment"] = choice.loose_align_fragment
        if choice.notes:
            df.at[idx, "structureNotes"] = "; ".join(choice.notes)

        if choice.source == "experimental":
            df.at[idx, "pdbID"] = choice.pdb_id
            info = {
                "method": choice.method,
                "resolution": choice.resolution,
                "domainCoverage": choice.domain_coverage,
                "domainPurity": choice.domain_purity,
                "chain": choice.chain_id,
            }
            df.at[idx, "structureInfo"] = json.dumps(info)
            if (choice.domain_purity is not None
                    and choice.domain_purity > HIGH_PURITY_THRESHOLD):
                df.at[idx, "highPurityFlag"] = True
                print(f"  Experimental match {choice.pdb_id}/{choice.chain_id} "
                      f"(purity={choice.domain_purity:.2f}>HIGH_PURITY); "
                      f"flagging row but still computing fb/cd from AF.")
            else:
                purity_str = (f"{choice.domain_purity:.2f}"
                              if choice.domain_purity is not None else "n/a")
                print(f"  Experimental match {choice.pdb_id}/{choice.chain_id} "
                      f"(purity={purity_str}).")
        else:
            df.at[idx, "structureInfo"] = json.dumps({"source": "AF"})
    else:
        df.at[idx, "structureInfo"] = json.dumps({"source": "AF"})

    # ----- AlphaFold metrics: always computed for cross-row comparability ---
    paeFile, PDBFile=getAlphaFoldFiles(entry, fragment, outputDir)
    if not PDBFile:
        writeRowToCheckpoint(idx)
        continue  # cannot proceed without the AF structure file
    if not paeFile:
        print(f"  PAE file unavailable for {entry} F{fragment} "
              f"(expected for F2+ of proteins >2700 AA). "
              f"anchoringIndex will be NaN; other metrics will still be computed.")

    parser=PDB.PDBParser(QUIET=True)
    fullProtein=parser.get_structure("protein", PDBFile)
    proteinChain=next(fullProtein[0].get_chains())
    fragmentLength=len(list(proteinChain))

    if localStart < 1 or localEnd > fragmentLength:
        print(f"  Skipping: local coords {localStart}-{localEnd} out of fragment range "
              f"(1-{fragmentLength}). Check FRAG_STEP constant.")
        writeRowToCheckpoint(idx)
        continue

    domainPDBFile=os.path.join(outputDir,
        f"{entry}_F{fragment}_{localStart}_{localEnd}_domain.pdb")
    exciseDomainPDB(PDBFile, domainPDBFile, localStart, localEnd)

    # Verify that the first 5 residues of the domain sequence match the PDB
    if not verifyDomainInFragment(PDBFile, localStart, str(sequence["Domain Sequence"])):
        print(f"  WARNING: Sequence mismatch for {entry}. "
              f"Check fragment/length logic. Proceeding with caution.")

    fb=fractionBuried(PDBFile, domainPDBFile, localStart, localEnd)
    cd=contactDensity(PDBFile, localStart, localEnd)
    df.at[idx, "fractionBuried"]=fb
    df.at[idx, "contactDensity"]=cd
    ai = None
    if paeFile:
        ai_val = anchoringIndex(paeFile, localStart, localEnd)
        # anchoringIndex returns NaN when the PAE matrix and the input
        # protein length disagree (stale TSV vs. AF DB v6 isoform).
        # Treat that as "PAE unavailable" for the interactionIndex formula.
        if not (isinstance(ai_val, float) and np.isnan(ai_val)):
            ai = ai_val
            df.at[idx, "anchoringIndex"] = ai
    if ai is not None:
        # Full formula: weights sum to 1.0 (0.247 + 0.565 + 0.187 ≈ 0.999)
        # Retrained and reweighted on 5/20/2026
        df.at[idx, "interactionIndex"] = (0.263*ai) + (0.579*fb) + (0.158*cd)
    else:
        # PAE unavailable or mismatched: reweight over fb and cd only.
        # Retrained and reweighted on 05/20/2026 to these:
        df.at[idx, "interactionIndex"] = (0.745*fb) + (0.255*cd)

    # ----- Experimental-PDB metrics (parallel to the AF block above) --------
    # Computed only when:
    #   1. A usable experimental structure was picked (choice.source ==
    #      "experimental").
    #   2. The full chain renumbered to UniProt is on disk
    #      (choice.full_chain_pdb_path) — needed for fractionBuried /
    #      contactDensity, which require non-domain context.
    #   3. highPurityFlag is False — when purity > HIGH_PURITY_THRESHOLD the
    #      chain is essentially the domain itself, so there are too few
    #      non-domain residues to make fb/cd meaningful.
    # Residue numbers in the experimental files are GLOBAL UniProt numbers
    # (structure_source renumbered them), so we pass domainStart/domainEnd
    # directly — no localStart/localEnd translation.
    if (choice is not None
            and choice.source == "experimental"
            and not df.at[idx, "highPurityFlag"]
            and choice.full_chain_pdb_path is not None
            and choice.pdb_path is not None):
        full_chain_path = str(choice.full_chain_pdb_path)
        domain_only_path = str(choice.pdb_path)
        if os.path.exists(full_chain_path) and os.path.exists(domain_only_path):
            try:
                exp_fb = fractionBuried(full_chain_path, domain_only_path,
                                        domainStart, domainEnd)
                exp_cd = contactDensity(full_chain_path, domainStart, domainEnd)
                df.at[idx, "exp_fractionBuried"] = exp_fb
                df.at[idx, "exp_contactDensity"] = exp_cd
                # Use the AF anchoringIndex (PAE-derived; experimental data
                # has no equivalent) so the two indices are directly
                # comparable per-row.
                if ai is not None:
                    df.at[idx, "exp_interactionIndex"] = (
                        (0.247*ai) + (0.565*exp_fb) + (0.187*exp_cd))
                else:
                    df.at[idx, "exp_interactionIndex"] = (
                        (0.751*exp_fb) + (0.249*exp_cd))
                print(f"  exp metrics: fb={exp_fb:.3f}, cd={exp_cd:.3f}, "
                      f"II={df.at[idx, 'exp_interactionIndex']:.3f}")
            except Exception as e:
                print(f"  WARNING: exp metric computation failed for {entry} "
                      f"({choice.pdb_id}/{choice.chain_id}): {e}")
        else:
            missing = [p for p in (full_chain_path, domain_only_path)
                       if not os.path.exists(p)]
            print(f"  exp metrics skipped — files missing: {missing}")
    elif (choice is not None
          and choice.source == "experimental"
          and df.at[idx, "highPurityFlag"]):
        print(f"  exp metrics skipped (highPurityFlag — too few non-domain residues).")

    # End-of-iteration checkpoint write. This is the success path: full
    # row got computed (or partially computed but didn't hit a `continue`)
    # and we persist whatever state was reached.
    writeRowToCheckpoint(idx)

# ---------------------------------------------------------------------------
# Loop done — finalise outputs
# ---------------------------------------------------------------------------
# Close the checkpoint file before reading it back as the source of truth.
ck_file.close()

# Read the checkpoint as the complete row inventory (in-memory df may have
# only the in-this-run rows populated; the checkpoint has everything from
# previous runs too).
print(f"Reading completed rows from checkpoint {checkpointPath} ...")
final_df = pd.read_csv(checkpointPath, sep='\t')
print(f"  {len(final_df):,} rows total.")

# Split on interactionIndex: domains where it could not be computed are eliminated
df_kept = final_df[final_df["interactionIndex"].notna()].copy()
df_eliminated = final_df[~final_df.index.isin(df_kept.index)].copy()

elimOutput = str(outputPath.with_name(outputPath.stem + '_eliminated' + outputPath.suffix))
df_eliminated.to_csv(elimOutput, sep='\t', index=False)

print(f"{len(df_kept)} domain sequences kept after interaction index filter")
print(f"{len(df_eliminated)} domain sequences eliminated; saved to {elimOutput}")
df_kept.to_csv(output, sep='\t', index=False)
print(f"Saved domain sequences to {output}")

# Clean up the checkpoint only when the full run completes successfully.
# If the script crashed before this point, the checkpoint survives so the
# next run can resume.
try:
    checkpointPath.unlink()
    print(f"Removed checkpoint {checkpointPath} (run completed cleanly).")
except OSError:
    pass
