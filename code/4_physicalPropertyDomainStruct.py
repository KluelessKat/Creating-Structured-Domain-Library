import argparse
import gzip
import math
import requests
import numpy as np
#from neurosnap.protein import * #https://neurosnap.ai/blog/post/understanding-the-radius-of-gyration-in-protein-design-bioinformatics/66fb0ecc5a941760bf4b5138
import freesasa
import pandas as pd
import os
from Bio import PDB
from neurosnap.io.pdb import parse_pdb

# ---------------------------------------------------------------------------
# Fragment helpers (mirrors step 3 — AF fragments use local residue numbering)
# ---------------------------------------------------------------------------
FRAG_STEP = 1200

def getAFFragment(domainStart):
    if domainStart <= 1400:
        return 1
    return math.ceil(domainStart / FRAG_STEP)

def globalToLocal(pos, fragment):
    return pos - (fragment - 1) * FRAG_STEP

def _ensureUnzipped(gzPath, destPath):
    if not os.path.exists(destPath):
        with gzip.open(gzPath, 'rb') as gz, open(destPath, 'wb') as out:
            out.write(gz.read())
    return destPath

def fetchFragmentPDB(entry, fragment, outputDir):
    '''Return a path to an unzipped fragment PDB. Checks outputDir first, then downloads.'''
    if fragment == 1:
        candidates = [os.path.join(outputDir, f"{entry}_model.pdb")]
    else:
        candidates = []
    candidates.append(os.path.join(outputDir, f"AF-{entry}-F{fragment}-model_v6.pdb"))

    # Check unzipped files
    for path in candidates:
        if os.path.exists(path):
            return path

    # Check .gz files and decompress
    for path in candidates:
        gzPath = path + ".gz"
        if os.path.exists(gzPath):
            print(f"  Decompressing {gzPath}")
            return _ensureUnzipped(gzPath, path)

    # Download from AlphaFold DB
    destPath = os.path.join(outputDir, f"AF-{entry}-F{fragment}-model_v6.pdb")
    url = f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F{fragment}-model_v6.pdb"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destPath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Downloaded fragment PDB for {entry} F{fragment}")
        return destPath
    except requests.exceptions.RequestException as e:
        print(f"  Failed to fetch PDB for {entry} F{fragment}: {e}")
        return None

def exciseDomain(fullPDBPath, domainPDBPath, localStart, localEnd):
    '''Extract domain residues from a fragment PDB and save as an isolated domain PDB.'''
    parser = PDB.PDBParser(QUIET=True)
    fullProtein = parser.get_structure("protein", fullPDBPath)
    proteinChain = next(fullProtein[0].get_chains())

    domain = PDB.Structure.Structure("domain")
    domainModel = PDB.Model.Model(0)
    domain.add(domainModel)
    domainChain = PDB.Chain.Chain(proteinChain.id)
    for residue in proteinChain:
        if localStart <= residue.id[1] <= localEnd:
            domainChain.add(residue.copy())
    domainModel.add(domainChain)

    io = PDB.PDBIO()
    io.set_structure(domain)
    io.save(domainPDBPath)
    print(f"  Extracted domain ({localStart}-{localEnd}) → {domainPDBPath}")

def calcRg(pdbFile):
    ensemble = parse_pdb(pdbFile, return_type="ensemble")
    structure = ensemble.first()
    return structure.calculate_rog()

# def calcRg(pdbFile):
#     '''Calculate the radius of gyration for a given structure, reflecting how compact it is.'''

#     structure = Protein(pdbFile)
#     distances_from_com=structure.distances_from_com()  
#     Rg=np.sqrt(np.sum(distances_from_com**2)/len(distances_from_com))
#     return Rg #In Å

def calcSASAMetrics(pdbFile, SASACutoff):
    '''Calculates the surface properties of a given structure by looking at types of surface residues'''

    domainStructure=freesasa.Structure(pdbFile)#Load the domain structure
    result=freesasa.calc(domainStructure)#Calculate the solvent accessible surface area (SASA) of each domain residue
    domainData=result.residueAreas()

    perResidueSASA=[]
    for sequenceID, sequenceData in domainData.items():#Go through each domain residue and store its total SASA to a list
        for residueNum, residueData in sequenceData.items():
            perResidueSASA.append((residueNum, residueData.residueType, residueData.total))
    
    totalResidues=len(perResidueSASA)
    surfaceResidues=[(residueNum, residueType) for residueNum, residueType, residueSASA in perResidueSASA if residueSASA>=SASACutoff] #Collect all residues that are solvent accesible, which is defined as having more than 20 Å^2 of solvent accessible surface area
    surfaceFraction=len(surfaceResidues)/totalResidues

    #Note the type of residues that are solvent accessible
    aromatic={"PHE", "TYR", "TRP", "HIS"}
    positive={"ARG", "LYS", "HIS"}
    negative={"ASP", "GLU"}
    totalAromatic=sum(residueType in aromatic for residueNum, residueType in surfaceResidues)
    totalPositive=sum(residueType in positive for residueNum, residueType in surfaceResidues)
    totalNegative=sum(residueType in negative for residueNum, residueType in surfaceResidues)

    metrics={ #Record surface properties based on the type of residues on the surface. 
        "totalResidues": totalResidues,
        "totalSurfaceResidues": len(surfaceResidues),
        "surfaceFraction": surfaceFraction,
        "aromaticFraction": totalAromatic/totalResidues,
        "positiveFraction": totalPositive/totalResidues,
        "negativeFraction": totalNegative/totalResidues,
        "perResidueSASA": {resideNum: residueSASA for resideNum, residueType, residueSASA in perResidueSASA}}
    return metrics


_ap = argparse.ArgumentParser(description='Step 4: Calculate physical properties of domain structures.')
_ap.add_argument('--input',
                 default='/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/kat_output_library_files/04282026_metapredict/3_domainLibraryInteractions_meta.tsv',
                 help='Input TSV (step 3 output)')
_ap.add_argument('--output',
                 default='/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/kat_output_library_files/04282026_metapredict/4_domainLibraryPhysicalProperties_meta.tsv',
                 help='Output TSV path')
_ap.add_argument('--af-dir',
                 default='/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/alphaFold/dbFiles',
                 help='Directory containing domain PDB files from step 3')
_args = _ap.parse_args()
input      = _args.input
output     = _args.output
pdbFileDir = _args.af_dir

#Prepare a dataframe to store physical properties for domain sequences
df=pd.read_csv(input, sep="\t")
df["Rg(Compactness)"]=None
df["surfaceFraction"]=None
df["aromaticSurfaceFraction"]=None
df["positiveSurfaceFraction"]=None
df["negativeSurfaceFraction"]=None

for idx, sequence in df.iterrows():#Go through each domain sequence and calculate physical properties
    pdbFile=os.path.join(pdbFileDir, f"{sequence['Entry']}_domain_model.pdb")

    if not os.path.exists(pdbFile):
        # Domain PDB not found — try to build it from the correct AF fragment
        fragment=getAFFragment(sequence["Start"])
        localStart=globalToLocal(sequence["Start"], fragment)
        localEnd=globalToLocal(sequence["End"], fragment)
        print(f"Domain PDB missing for {sequence['Entry']}; fetching F{fragment} to extract domain {sequence['Start']}-{sequence['End']} (local {localStart}-{localEnd})")
        fragmentPDB=fetchFragmentPDB(sequence["Entry"], fragment, pdbFileDir)
        if fragmentPDB:
            exciseDomain(fragmentPDB, pdbFile, localStart, localEnd)
        if not os.path.exists(pdbFile):
            print(f"  Skipping {sequence['Entry']}: could not obtain domain PDB")
            continue

    print(f"Calculating Physical Properties for {sequence['Entry']}")
    metrics=calcSASAMetrics(pdbFile, 20)
    df.at[idx, "Rg(Compactness)"]=calcRg(pdbFile)
    df.at[idx, "surfaceFraction"]=metrics['surfaceFraction']
    df.at[idx, "aromaticSurfaceFraction"]=metrics['aromaticFraction']
    df.at[idx, "positiveSurfaceFraction"]=metrics['positiveFraction']
    df.at[idx, "negativeSurfaceFraction"]=metrics['negativeFraction']

df.to_csv(output, sep='\t', index=False)
print(f"Saved domain sequences to {output}")
