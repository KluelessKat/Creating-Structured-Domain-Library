from pymol import cmd
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os

def takePymolImage(pdbFile, start, end, output):
    '''Take an image of a given protein with a specified domain highlighted in pymol'''
    
    #Load pymol and the protein
    cmd.reinitialize()
    cmd.load(pdbFile, "fullProtein")
    cmd.color("green", "fullProtein")

    #Highlight the domain red and center and zoom in on it
    cmd.select("domain", f"resi {start}-{end}")
    cmd.color("red", "domain")
    cmd.orient("domain")
    cmd.zoom("domain", buffer=100)

    #Take an image
    cmd.set("ray_opaque_background", 0)
    cmd.ray(1200, 1200)
    cmd.png(output, dpi=300)

def tileImages(imagePaths, output, maxCols, annotations, fontSize=50):
    '''Annotate indvidual domain images and tile them together'''
    
    images=[Image.open(f) for f in imagePaths] #A list of images of our candidate sequences
    
    #Specify the dimensions of the tiled image
    width, height=images[0].size
    cols=min(maxCols, len(images))
    rows=(len(images)+cols-1)//cols
    tiledImage=Image.new("RGB", (cols*width, rows*height), (255, 255, 255))

    font=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", fontSize)
    draw=ImageDraw.Draw(tiledImage)

    for idx, image in enumerate(images):#For each candidate domain image, add it to the tiled image and annotate it.
        imageCoordinateX=(idx%cols)*width
        imageCoordinateY=(idx//cols)*height
        tiledImage.paste(image, (imageCoordinateX, imageCoordinateY))#Add the image to the tiled image.

        annotation=annotations[idx]#Create an annotation for the image
        lines=[f"Entry: {annotation.get('Entry','')}",
               f"Domain: {annotation.get('Domain','')}",
               f"Start: {annotation.get('Start','')}",
               f"End: {annotation.get('End','')}"]
        margin=5
        for i, line in enumerate(lines):#Add the annotation to the image
            lineCoordinateX=imageCoordinateX+margin
            lineCoordinateY=imageCoordinateY+height - (len(lines)-i)*(fontSize+2)-margin
            draw.text((lineCoordinateX, lineCoordinateY), line, fill="white", font=font)
    tiledImage.save(output)

#File Pathnames. Change them to match yours.
input = '/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/kat_output_library_files/04282026_metapredict/5_finalCandidateSequences_meta.tsv'
#'/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/kat_output_library_files/04212026_iupred3/5_finalCandidateSequences_iupred3.tsv'
baseOutputDir='/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/kat_output_library_files/04282026_metapredict'
imagesDir=os.path.join(baseOutputDir, 'images')
if os.path.exists(imagesDir):
    i=1
    while os.path.exists(f"{imagesDir}_{i}"):
        i+=1
    imagesDir=f"{imagesDir}_{i}"
os.makedirs(imagesDir)
tileOutput=os.path.join(imagesDir, '1_allFilters.png')
df=pd.read_csv(input, sep="\t")

entriesDictionary={}
for entry, group in df.groupby('Entry'):#Record relevant info for each domain sequence to help create annotations
    entriesDictionary[entry]=group[['Domain', 'Domain Sequence', 'Start', 'End']].values.tolist()

    #entriesDictionary[entry]=group[['Domain', 'Domain Sequence', 'Start', 'End', 'candidateSequence']].values.tolist()
 
imagePaths=[]
annotations=[]
for entry, rows in entriesDictionary.items():#Go through each protein 
    for row in rows: #Go through each domain of a protein
        domain, domainSeq, start, end = row#, driverClass=row
        print(f"Imaging {entry}")
        pdbFile=f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F1-model_v6.pdb"
        output=os.path.join(imagesDir, f"{entry}_{start}_{end}.png") #Specify the folder that will hold individual images
        takePymolImage(pdbFile, start, end, output)
        imagePaths.append(output)
        annotation={
                "Entry": entry,
                "Domain": domain,
                "Domain Sequence": domainSeq,
                "Start": start,
                "End": end
                #"driverClass": driverClass
                }
        annotations.append(annotation)
        
        #if driverClass!="Neither":#If the sequence is a candidate sequence, capture an image of it and make an annotation for it
            # print(f"Imaging {entry}")
            # pdbFile=f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F1-model_v6.pdb"
            # output=f"/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/kat_output_library_files/04212026_iupred3/2_images/{entry}_{start}_{end}.png" #Specify the folder that will hold individual images
            # takePymolImage(pdbFile, start, end, output)
            # imagePaths.append(output)
            # annotation={
            #     "Entry": entry,
            #     "Domain": domain,
            #     "Domain Sequence": domainSeq,
            #     "Start": start,
            #     "End": end
            #     #"driverClass": driverClass
            #     }
            # annotations.append(annotation)

tileImages(imagePaths, tileOutput, 5, annotations) #Create a tiled image of all candidate sequences
print(f"Saved tiled image to {tileOutput}")
