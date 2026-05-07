import numpy as np
#from neurosnap.protein import * #https://neurosnap.ai/blog/post/understanding-the-radius-of-gyration-in-protein-design-bioinformatics/66fb0ecc5a941760bf4b5138
import freesasa
import pandas as pd
import os
from neurosnap.io.pdb import parse_pdb

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


pdbFileDir="/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/alphaFold/dbFiles" #Specify where your domain PDB files are stored (should be a folder)

#File Pathnames. Change them to match yours.  
input= \
'/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/kat_output_library_files/04282026_metapredict/3_domainLibraryInteractions_meta.tsv'
output= \
'/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/kat_output_library_files/04282026_metapredict/4_domainLibraryPhysicalProperties_meta.tsv'

#Prepare a dataframe to store physical properties for domain sequences
df=pd.read_csv(input, sep="\t")
df["Rg(Compactness)"]=None
df["surfaceFraction"]=None
df["aromaticSurfaceFraction"]=None
df["positiveSurfaceFraction"]=None
df["negativeSurfaceFraction"]=None

for idx, sequence in df.iterrows():#Go through each domain sequence and calculate physical properties
    pdbFile=f"/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/alphaFold/dbFiles/{sequence['Entry']}_domain_model.pdb"
    if not os.path.exists(pdbFile):#If there is no PDB file for the domain, move onto the next sequence
        continue
    
    print(f"Calculating Physical Properties for {sequence}")
    metrics=calcSASAMetrics(pdbFile, 20)
    df.at[idx, "Rg(Compactness)"]=calcRg(pdbFile)
    df.at[idx, "surfaceFraction"]=metrics['surfaceFraction']
    df.at[idx, "aromaticSurfaceFraction"]=metrics['aromaticFraction']
    df.at[idx, "positiveSurfaceFraction"]=metrics['positiveFraction']
    df.at[idx, "negativeSurfaceFraction"]=metrics['negativeFraction']

df.to_csv(output, sep='\t', index=False)
print(f"Saved domain sequences to {output}")
