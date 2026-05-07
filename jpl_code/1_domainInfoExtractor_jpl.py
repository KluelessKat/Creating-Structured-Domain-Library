import pandas as pd
import re

#File Pathnames. Change them to match yours.   
input="/Users/joseparedes/Desktop/kappelLab/structuredDomainLibrary/humanProteome.tsv"
output="/Users/joseparedes/Desktop/kappelLab/structuredDomainLibrary/1_domainLibraryRaw.tsv"
df=pd.read_csv(input, sep="\t")

outputEntries=[]#Where we will store domain sequences and their relevant info
for idx, entry in df.iterrows():#Go through each protein in the proteome to collect all domains
    uniprotID=entry["Entry"]#Here, you can add more lines to capture more information. For example, if your uniprot proteome download includes column X, add infoX=entry["X"]
    geneName=entry["Gene Names"]
    length=entry["Length"]
    sequence=entry["Sequence"]

    domainInfo=entry.get("Domain [FT]", "")#Get the info for a given domain
    if pd.isna(domainInfo) or domainInfo.strip()=="":#Skip proteins with no domain info
        continue

    #Uniprot stores domain info in a messy string of text. Here, we extract relevant info from the string
    domainFeatures=re.finditer(r'([A-Z0-9_]+)\s+(\d+)\.\.(\d+)(.*?)(?=(?:[A-Z0-9_]+\s+\d+\.\.)|$)', domainInfo)#Capture relevant info.
    for feature in domainFeatures:#Go through the captured info to extract domain info.
        domainType=feature.group(1)
        start=int(feature.group(2))-1 #-1 is for indexing reasons
        end=int(feature.group(3))
        potDomainName=feature.group(4).strip().rstrip(';')
        nameSearch=re.search(r'/note="([^"]+)"', potDomainName)
        domainName=nameSearch.group(1) if nameSearch else domainType
        domainSeq=sequence[start:end]

        outputEntries.append({
            "Entry": uniprotID,
            "Gene Name": geneName,
            "Length": length,
            "Domain": domainName,
            "Start": start+1,
            "End": end,
            "Domain Length": end-start,
            "Domain Sequence": domainSeq})

dfOutput=pd.DataFrame(outputEntries)
print(f"{len(dfOutput)} domain sequences before filtering")
dfOutput=dfOutput[(dfOutput["Domain Length"]<=66)].copy() #Because of experimental limitations, we filter out any domain sequence longer than 66 aa. 
print(f"{len(dfOutput)} domain sequences after domain sequence length filtering")
dfOutput['Domain']=dfOutput['Domain'].apply(lambda s: re.sub(r'\s*\d+$', '', s))#Clean domain names by removing trailing numbers

dfOutput.to_csv(output, sep="\t", index=False)
print(f"Saved domain sequences to {output}")
