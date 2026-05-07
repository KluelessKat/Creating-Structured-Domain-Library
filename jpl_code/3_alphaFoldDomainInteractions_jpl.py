import requests
import os
import json
import numpy as np
import freesasa
from Bio import PDB
from scipy.spatial.distance import cdist
import pandas as pd

def downloadAlphaFoldFiles(uniprotID, outputDir):
    '''Uses a protein's uniprot accession number to download its predicted structure's PAE and PDB files from AlphaFoldDB'''
    
    #The pathnames used for downloading and storing the files. You likely don't have to change anything here. However, it is possible that some protein stuctures may have been predicted with a different AlphaFold model than v6; if so, you can change the version number to match.
    pdbDownloadUrl=f"https://alphafold.ebi.ac.uk/files/AF-{uniprotID}-F1-model_v6.pdb"
    pdbOutputPath=os.path.join(outputDir, f"{uniprotID}_model.pdb")
    paeDownloadUrl=f"https://alphafold.ebi.ac.uk/files/AF-{uniprotID}-F1-predicted_aligned_error_v6.json"
    paeOutputPath=os.path.join(outputDir, f"{uniprotID}_PAE.json")

    try:#Download the PAE and PDB files if you have not already
        if not os.path.exists(pdbOutputPath):
            response=requests.get(pdbDownloadUrl, stream=True)
            response.raise_for_status()
            with open(pdbOutputPath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):#Write the PDB file contents into a new file saved to your system
                    f.write(chunk)
            print(f"Downloaded PDB file for {uniprotID}")
        else:
            print(f"PDB file already exists for {uniprotID}, skipping download.")

        if not os.path.exists(paeOutputPath):
            response=requests.get(paeDownloadUrl, stream=True)
            response.raise_for_status()
            with open(paeOutputPath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded PAE file for {uniprotID}")
        else:
            print(f"PAE file already exists for {uniprotID}, skipping download.")

        return paeOutputPath, pdbOutputPath

    except requests.exceptions.RequestException as e:
        print(f"Failed to download files for {uniprotID}. Error: {e}")
        return "", ""

def exciseDomainPDB(inputPDB, outputPDB, domainStart, domainEnd):
    '''Extract and save only the structural information of a specific domain into a PDB file using data from its parent structure'''
    
    if os.path.exists(outputPDB):#Only excise the domain if we have not already
        print(f"Domain PDB already exists: {outputPDB}")
        return outputPDB
    
    #PDB files hold info in the following hierarchy: structure -> model -> chain -> residue -> atom
    parser=PDB.PDBParser(QUIET=True) #Load the protein structure
    fullProtein=parser.get_structure("protein", inputPDB)
    proteinModel=fullProtein[0] #AlphaFold structures are single models, so we just take the first and only one
    proteinChain=next(proteinModel.get_chains()) #From that model, we take the first chain, as AlphaFold only outputs a single chain

    domain=PDB.Structure.Structure("domain") #Create a structured object to store the domain model
    domainModel=PDB.Model.Model(0) #Create a model to hold the domain chain
    domain.add(domainModel) #Add the domain model to the structure
    domainChain=PDB.Chain.Chain(proteinChain.id) #Create a chain to hold the domain residues

    for residue in proteinChain: #Only add domain residues to the domain chain
        residueNum=residue.id[1]
        if domainStart <= residueNum <= domainEnd:
            domainChain.add(residue.copy())

    domainModel.add(domainChain) #Add the domain chain to the domain model
    io=PDB.PDBIO()
    io.set_structure(domain)
    io.save(outputPDB)
    print(f"New domain saved to: {outputPDB}")

def anchoringIndex(paeFile, domainStart, domainEnd):
    '''Uses the PAE file of a protein to measure how constrained one of its domains is by it. Helps us determine if domain B has to be around residues X-Y. Note that this is not a measure of interaction strength'''

    with open(paeFile, "r") as f: #Open and load PAE data
        data=json.load(f)
    paeMatrix=np.array(data[0]["predicted_aligned_error"]) #A matrix of PAE values for every possible residue pair

    domainResidues=np.arange(domainStart-1, domainEnd) #Define domain residues
    proteinResidues=np.setdiff1d(np.arange(paeMatrix.shape[0]), domainResidues) #Define non-domain residues
    paeBetween=paeMatrix[np.ix_(domainResidues, proteinResidues)] #Extract PAE values between domain residues and all non-domain residues
    lowPAE=paeBetween<=5 #Any PAE value <= 5 Å  is converted to a 1, while any value above > 5 Å is turned to a 0. A value of 1 indicates the model is confident in the relative positioning of that residue pair; 0 indicates uncertainty.

    perResidueAnchoring=np.sum(lowPAE, axis=1)/len(proteinResidues) #For each domain residue, compute the fraction of non-domain residues it is confidently positioned relative to.
    anchoringIndex=np.mean(perResidueAnchoring) #Take the mean across all domain residues to obtain a score reflecting how constrained the domain is by its parent protein
    return anchoringIndex
   
def fractionBuried(pdbFile, domainPDBFile, domainStart, domainEnd):
    '''Calculates the fraction of a domain that is buried within its parent protein'''
    
    fullProtein=freesasa.Structure(pdbFile) #Load the protein structure.
    fpResult=freesasa.calc(fullProtein) #Calculates the surface area of each residue in the protein that is accessible to solvent in Å^2. 
    domainInProtein=freesasa.selectArea([f'r{domainStart}_{domainEnd}, resi {domainStart}-{domainEnd}'], fullProtein, fpResult)
    domainInProteinSASA=domainInProtein[f'r{domainStart}_{domainEnd}'] #The amount of surface area of the domain that is accessible to solvent while it is in its parent protein.

    domainOnly=freesasa.Structure(domainPDBFile)#Load the isolated domain structure
    doResult=freesasa.calc(domainOnly) #Calculates the surface area of each residue in the isolated domain structure that is accessible to solvent. 
    domainOnlySASA=doResult.totalArea()
    deltaSASA=domainOnlySASA-domainInProteinSASA

    return deltaSASA/domainOnlySASA #Fraction of the domain that is buried in its parent protein

def contactDensity(pdbFile, domainStart, domainEnd):
    '''Quantifies the number of interactions a domain has with its parent protein by returning the fraction of total possible contacts found between the domain and protein'''
    
    parser=PDB.PDBParser(QUIET=True) #Load the protein structure
    fullProtein=parser.get_structure("protein", pdbFile)
    proteinModel=fullProtein[0] #Load the protein model 
    proteinChain=next(proteinModel.get_chains()) #Load the protein chain
    
    domainResidues=[res for res in proteinChain if domainStart <= res.get_id()[1] <= domainEnd] #Define domain residues
    otherResidues=[res for res in proteinChain if res.get_id()[1] < domainStart or res.get_id()[1] > domainEnd] #Define non-domain residues

    cutoff=4.0 #A domain residue and non-domain residue are considered to be interacting if any atom in one is within 4 Å of the other
    totalContacts=0
    for domainResidue in domainResidues: #Go through every possible domain and non-domain residue pair. For each pair, note any interactions
        domainCoordinates=np.array([atom.get_coord() for atom in domainResidue])
        for otherResidue in otherResidues:
            otherCoordinates=np.array([atom.get_coord() for atom in otherResidue])
            if np.any(cdist(domainCoordinates, otherCoordinates) < cutoff):
                totalContacts+=1

    contactDensity=totalContacts/(len(domainResidues)*len(otherResidues)) #Divide the observed interactions by the number of all possible contacts
    return contactDensity

def plddtMean(domainPDBFile):
    '''pLDDT is a measure of how confident the AlphaFold model is in its prediction. Here, the mean pLDDT of the domain is measured'''

    parser=PDB.PDBParser(QUIET=True) #Load the domain structure
    domain=parser.get_structure("domain", domainPDBFile)
    domainModel=domain[0] #Load the domain model
    domainChain=next(domainModel.get_chains()) #Load the domain chain

    plddtValues=[]
    for residue in domainChain:#Obtain the pLDDT value of every residue in the domain
        atomBfactors=[atom.get_bfactor() for atom in residue]
        plddtValues.append(sum(atomBfactors)/len(atomBfactors))

    return round(sum(plddtValues)/len(plddtValues), 2) #Return the mean pLDDT value for the domain

outputDir='/Users/joseparedes/Desktop/kappelLab/alphaFold/dbFiles' #Specify where you want (should be a folder) to store all of the PAE and PDB files we will be downloading. Note that each one is typically less than a MB in size. 

#File Pathnames. Change them to match yours.  
input='/Users/joseparedes/Desktop/kappelLab/structuredDomainLibrary/2_domainLibraryStructuredSeq.tsv'
output='/Users/joseparedes/Desktop/kappelLab/structuredDomainLibrary/3_domainLibraryInteractions.tsv'

#Prepare a dataframe to store interaction metrics for domain sequences
df=pd.read_csv(input, sep="\t")
df["anchoringIndex"]=None
df["fractionBuried"]=None
df["contactDensity"]=None
df["interactionIndex"]=None
df["meanDomainplddt"]=None

for idx, sequence in df.iterrows(): #Go through each domain sequence
    domainStart=sequence["Start"]
    domainEnd=sequence["End"]
    
    paeFile, PDBFile=downloadAlphaFoldFiles(sequence["Entry"], outputDir)
    if not paeFile or not PDBFile:#If the PAE and PDB files could not be downloaded for a sequence, skip to the next one
        continue

    parser=PDB.PDBParser(QUIET=True)
    fullProtein=parser.get_structure("protein", PDBFile) #Load the protein structure
    proteinChain=next(fullProtein[0].get_chains())
    proteinLength=len(list(proteinChain))

    if domainStart<1 or domainEnd>proteinLength: #Skip if domain coordinates are out of bounds
        print(f"Skipping {sequence['Entry']} domain {domainStart}-{domainEnd}: out of protein range (1-{proteinLength})")
        continue

    domainPDBFile=os.path.join(outputDir, f"{sequence["Entry"]}_domain_model.pdb")
    exciseDomainPDB(PDBFile, domainPDBFile, domainStart, domainEnd) #Extract the isolated domain strucuture
    
    plddt=plddtMean(PDBFile)
    df.at[idx, "meanDomainplddt"]=plddt #Calculate the mean pLDDT of a domain
    if plddt>=80: #Only compute interaction metrics for confident domain structures
        df.at[idx, "anchoringIndex"]=anchoringIndex(paeFile, domainStart, domainEnd)
        df.at[idx, "fractionBuried"]=fractionBuried(PDBFile, domainPDBFile, domainStart, domainEnd)
        df.at[idx, "contactDensity"]=contactDensity(PDBFile, domainStart, domainEnd)
        df.at[idx, "interactionIndex"]=(0.247*df.at[idx, "anchoringIndex"])+(0.565*df.at[idx, "fractionBuried"])+(0.187*df.at[idx, "contactDensity"]) #Take a weighted average of the interaction metrics to get a single score reflecting how much a domain is interacting with its parent. Weights were found with a logistic regression model

df=df[(df["meanDomainplddt"]>=80)].copy()#Filter out low confidence domain structures.
print(f"{len(df)} domain sequences after mean domain plddt filtering")
df.to_csv(output, sep='\t', index=False)
print(f"Saved domain sequences to {output}")
