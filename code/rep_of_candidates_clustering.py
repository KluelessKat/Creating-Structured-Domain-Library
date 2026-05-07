import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from umap import UMAP
import plotly.express as px

def embedSequences(seqs, batchSize):#BatchSize allows us to embed our sequences into chunks
    finalEmbeddings=[]

    for i in range(0, len(seqs), batchSize):#Chunk our sequences and embed each one
        batch=seqs[i:i+batchSize]
        encoded=tokenizer(batch, return_tensors="pt", padding=True)#Tokeninze the sequences so that our model understands them. We have to pad our sequences so that we feed the model a rectangular tensor
        encoded={k: v.to(device) for k, v in encoded.items()}#Move our tokenized sequences to the GPU. 

        with torch.no_grad():#We are not training the model, so there is no need to update the model weights with backpropagation
            output=model(**encoded)#Provide the model our tokenized sequences
            tokenEmbeddings=output.last_hidden_state#Embed every token

        attentionMask=encoded["attention_mask"].unsqueeze(-1)#Create a mask to ignore any embeddings originating from padding
        maskedEmbeddings=tokenEmbeddings*attentionMask

        sumEmbeddings=maskedEmbeddings.sum(dim=1)#Do mean pooling to represent each sequence with a single vector
        lengths=attentionMask.sum(dim=1)#We divide each sum embedding by the number of real residues. That is, padding does not contribute to length
        meanEmbeddings=sumEmbeddings/lengths

        finalEmbeddings.append(meanEmbeddings.cpu().numpy())

    return np.vstack(finalEmbeddings)#Return a single array containing all of our embeddings

def getRepSeqs(embeddings):
    normalizedEmbeddings=normalize(embeddings, axis=1)#Normalize to prevent magnitude from dominating
    cosineDistances=cosine_distances(normalizedEmbeddings)#Obtain the cosine distance of every embedding pair
    condensedDistances=squareform(cosineDistances, checks=False)#Condense our ditance matrix into a 1D vector
    avgLinkage=linkage(condensedDistances, method="average")#Contruct a dendrogram. Note that this is a UPGMA-style hierarchical clustering
    clusters=fcluster(avgLinkage, t=0.20, criterion="distance")#Cut the dendrogram horizontally at cosine distance=0.2. That is, any sequences or merged clusters that connect below this height are grouped together

    representatives={}
    for cluster in np.unique(clusters):#Go through the clusters to find a representative of each
        idxs=np.where(clusters == cluster)[0]#Get indicies for each sequence in the cluster

        if len(idxs) == 1:#If there is only one sequence in the cluster, that is the rep by default
            representatives[cluster]=idxs[0]
            continue

        clusterDistances=cosineDistances[np.ix_(idxs, idxs)]#Find the medoid in the cluster
        meanDistance=clusterDistances.mean(axis=1)
        medoidClusterIdx=np.argmin(meanDistance)
        medoidGlobalIdx=idxs[medoidClusterIdx]
        representatives[cluster]=medoidGlobalIdx
    return representatives, clusters

def getSequences(inputFile):
    df=pd.read_csv(inputFile, sep="\t")
    domainSeqs={}
    for _, row in df.iterrows():
        interproID=row["interProList"]
        domainSeq=row["Domain Sequence"]

        if pd.isna(interproID) or pd.isna(domainSeq):
            continue

        if interproID not in domainSeqs:
            domainSeqs[interproID]=[]

        domainSeqs[interproID].append({
            "Entry": row["Entry"],
            "Start": row["Start"],
            "End": row["End"],
            "Sequence": domainSeq})

    return domainSeqs

def plotClustersWithReps(embeddings, clusters, representatives, sequences):
    umap2d=UMAP(n_components=2, random_state=0)
    umapEmbeddings=umap2d.fit_transform(embeddings)
    df=pd.DataFrame({"UMAP1": umapEmbeddings[:, 0], "UMAP2": umapEmbeddings[:, 1], "Label": clusters.astype(str)})

    fig=px.scatter(df, x="UMAP1", y="UMAP2", color="Label", title="Sequence Embeddings UMAP", width=800, height=600)
    
    repIDXs=list(representatives.values())
    repHover=[f"Cluster: {c}<br>Sequence:<br>{sequences[i]}" for c, i in representatives.items()]
    fig.add_scatter(x=df.loc[repIDXs, "UMAP1"], y=df.loc[repIDXs, "UMAP2"], mode="markers",
        marker=dict(size=16, symbol="star", color="black"), name="Cluster Representative", hovertext=repHover, hoverinfo="text")
    
    fig.show()

def createRepSeqLibrary(input, output, batchSize=8):
    domainSeqs=getSequences(input)
    rows=[]
    run=1

    for interproID, records in domainSeqs.items():
        print(f"Current Run: {run}")
        if len(records)==0:
            continue
        sequences=[r["Sequence"] for r in records]
        embeddings=embedSequences(sequences, batchSize)
        representatives, clusters=getRepSeqs(embeddings)
        for clusterID, repIDX in representatives.items():
            repRecord=records[repIDX]
            rows.append({
            "Entry": repRecord["Entry"],
            "InterProID": interproID,
            "ClusterID": clusterID,
            "Start": repRecord["Start"],
            "End": repRecord["End"],
            "RepresentativeSequence": repRecord["Sequence"]})

        #plotClustersWithReps(embeddings, clusters, representatives, sequences)
        #if run>5:
            #break
        run+=1
    repSeqLibrary=pd.DataFrame(rows)
    repSeqLibrary.to_csv(output, sep="\t", index=False)

device=torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device being used: {device}")
modelName="facebook/esm2_t12_35M_UR50D"
tokenizer=AutoTokenizer.from_pretrained(modelName, do_lower_case=False)
model=AutoModel.from_pretrained(modelName)
model=model.to(device)
model.eval()

input="/Users/joseparedes/Desktop/kappelLab/structuredDomainLibrary/domainLibraryFiltered.tsv"
output="/Users/joseparedes/Desktop/kappelLab/structuredDomainLibrary/domainLibraryRepresentatives.tsv"
createRepSeqLibrary(input, output)
print("Done")