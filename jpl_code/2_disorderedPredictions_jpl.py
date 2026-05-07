import sys
import numpy as np
import pandas as pd

#Ensures related AIUPred files are in the same system path you are running the code in to allow the library to be imported. 
AIUPredPath="/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/jpl_code/Creating-Structured-Domain-Library/AIUPred_old"
if AIUPredPath not in sys.path:
    sys.path.insert(0, AIUPredPath)
import aiupred_lib

embeddingModel, regressionModel, device=aiupred_lib.init_models("disorder")#The AIUPred model we will use to make predictions

#File Pathnames. Change them to match yours.   
input = '/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/jpl_code/Creating-Structured-Domain-Library/1_domainLibraryRaw.tsv'
output = '/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/jpl_code/Creating-Structured-Domain-Library/2_domainLibraryStructuredSeq_old.tsv'
df=pd.read_csv(input, sep="\t")

meanScoresForSequences=[]#For a given sequence, each residue is assigned a predicted disordered score. Here, we store the mean predicted disordered value across all residues in a sequence.
fracOfDisResidues=[]#Here we store the fraction of residues in a sequence that have a predicted disordered score >0.5

for sequence in df["Domain Sequence"]:#Go through your domain sequences to make disordered predictions for each one
    print(f"Running Disordered Predictions on {sequence}")

    if pd.isna(sequence) or sequence==0:#Skip empty sequences
        meanScoresForSequences.append(np.nan)
        continue

    predictions=aiupred_lib.predict_disorder(sequence, embeddingModel, regressionModel, device, smoothing=True)#Make a disordered prediction for each residue. There is also a low memory version of this function if your computer is running out of memory when running this.
    meanScoresForSequences.append(float(np.mean(predictions)))
    fracOfDisResidues.append(np.sum(np.array(predictions)>0.5)/len(predictions))

df["Mean Disorder"]=meanScoresForSequences
df["Fraction Of Disordered Residues"]=fracOfDisResidues

#Here, we filter to keep structured domains (1) or disordered domains (2). 
df=df[(df["Mean Disorder"]<=0.5) & (df["Fraction Of Disordered Residues"]<=0.2)].copy() #Filter 1: gathering structured domains here.
#df=df[(df["Mean Disorder"]>0.5) & (df["Fraction Of Disordered Residues"]>0.2)].copy() #Filter 2: gathering disordered sequences here. 

print(f"{len(df)} domain sequences after disordered prediction filtering")
df.to_csv(output, sep="\t", index=False)
print(f"Saved domain sequences to {output}")