import argparse
import gzip
import math
import requests
import os
import json
from pathlib import Path
import numpy as np
import freesasa
from Bio import PDB
from scipy.spatial.distance import cdist
import pandas as pd

# ---------------------------------------------------------------------------
# Fragment helpers
# ---------------------------------------------------------------------------
# AlphaFold DB splits proteins > ~1400 AA into overlapping fragments.
# Each fragment is 1400 AA long with a 200 AA overlap → step = 1200 AA.
# Fragment PDB files use LOCAL residue numbering (1 to ~1400), NOT global
# UniProt positions.  Convert before any residue-based calculation.
#   F1: global  1–1400  → local offset 0
#   F2: global  1201–2600 → local offset 1200
#   F3: global  2401–3800 → local offset 2400  …
FRAG_STEP = 1200

def getAFFragment(domainStart):
    '''Return the AlphaFold fragment number that contains the given global residue.'''
    if domainStart <= 1400:
        return 1
    return math.ceil(domainStart / FRAG_STEP)

def globalToLocal(pos, fragment):
    '''Convert a global UniProt residue position to a local fragment position.'''
    return pos - (fragment - 1) * FRAG_STEP

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
        pdbDest = os.path.join(outputDir, f"{uniprotID}_model.pdb")
        paeDest = os.path.join(outputDir, f"{uniprotID}_PAE.json")
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
    domainStart/domainEnd must be LOCAL fragment residue numbers.'''
    with open(paeFile, "r") as f:
        data=json.load(f)
    paeMatrix=np.array(data[0]["predicted_aligned_error"])

    domainResidues=np.arange(domainStart-1, domainEnd)        # 0-indexed local
    proteinResidues=np.setdiff1d(np.arange(paeMatrix.shape[0]), domainResidues)
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

    cutoff=4.0
    totalContacts=0
    for domainResidue in domainResidues:
        domainCoordinates=np.array([atom.get_coord() for atom in domainResidue])
        for otherResidue in otherResidues:
            otherCoordinates=np.array([atom.get_coord() for atom in otherResidue])
            if np.any(cdist(domainCoordinates, otherCoordinates) < cutoff):
                totalContacts+=1

    return totalContacts/(len(domainResidues)*len(otherResidues))

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
_args = _ap.parse_args()
input     = _args.input
output    = _args.output
outputDir = _args.af_dir
os.makedirs(outputDir, exist_ok=True)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

df=pd.read_csv(input, sep="\t")
df["anchoringIndex"]=None
df["fractionBuried"]=None
df["contactDensity"]=None
df["interactionIndex"]=None
df["meanDomainplddt"]=None

for idx, sequence in df.iterrows():
    domainStart=sequence["Start"]
    domainEnd=sequence["End"]

    # Determine which AF fragment contains this domain and convert to local coords
    fragment=getAFFragment(domainStart)
    localStart=globalToLocal(domainStart, fragment)
    localEnd=globalToLocal(domainEnd, fragment)
    print(f"Processing {sequence['Entry']} domain {domainStart}-{domainEnd} "
          f"(F{fragment}, local {localStart}-{localEnd})")

    paeFile, PDBFile=getAlphaFoldFiles(sequence["Entry"], fragment, outputDir)
    if not paeFile or not PDBFile:
        continue

    parser=PDB.PDBParser(QUIET=True)
    fullProtein=parser.get_structure("protein", PDBFile)
    proteinChain=next(fullProtein[0].get_chains())
    fragmentLength=len(list(proteinChain))

    if localStart < 1 or localEnd > fragmentLength:
        print(f"  Skipping: local coords {localStart}-{localEnd} out of fragment range "
              f"(1-{fragmentLength}). Check FRAG_STEP constant.")
        continue

    domainPDBFile=os.path.join(outputDir, f"{sequence['Entry']}_domain_model.pdb")
    exciseDomainPDB(PDBFile, domainPDBFile, localStart, localEnd)

    plddt=plddtMean(domainPDBFile)
    df.at[idx, "meanDomainplddt"]=plddt
    if plddt>=80:
        df.at[idx, "anchoringIndex"]=anchoringIndex(paeFile, localStart, localEnd)
        df.at[idx, "fractionBuried"]=fractionBuried(PDBFile, domainPDBFile, localStart, localEnd)
        df.at[idx, "contactDensity"]=contactDensity(PDBFile, localStart, localEnd)
        df.at[idx, "interactionIndex"]=(0.247*df.at[idx, "anchoringIndex"])+(0.565*df.at[idx, "fractionBuried"])+(0.187*df.at[idx, "contactDensity"])

df_kept=df[(df["meanDomainplddt"]>=80)].copy()
df_eliminated=df[~df.index.isin(df_kept.index)].copy()

outputPath=Path(output)
elimOutput=str(outputPath.with_name(outputPath.stem + '_eliminated' + outputPath.suffix))
df_eliminated.to_csv(elimOutput, sep='\t', index=False)

print(f"{len(df_kept)} domain sequences kept after pLDDT filtering (>= 80)")
print(f"{len(df_eliminated)} domain sequences eliminated; saved to {elimOutput}")
df_kept.to_csv(output, sep='\t', index=False)
print(f"Saved domain sequences to {output}")
