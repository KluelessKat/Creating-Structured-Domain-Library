import argparse
import gzip
import math
import tempfile
from pymol import cmd
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os

def findLocalPDB(entry, fragment, afDir):
    '''Search afDir for a cached AlphaFold PDB before hitting the web.
    Checks both uncompressed (.pdb) and gzipped (.pdb.gz) files.
    Naming conventions checked (in order):
      - {entry}_model.pdb/.gz        (step 3 download convention, F1 only)
      - AF-{entry}-F{N}-model_v6.pdb/.gz  (original AlphaFold DB naming)
    Returns (path, is_temp): path is usable by PyMOL; is_temp=True means it is
    a decompressed temp file that the caller must delete after use.'''
    if not afDir or not os.path.isdir(afDir):
        return None, False

    candidates = []
    if fragment == 1:  # step 3 only ever downloads F1 under this name
        candidates += [
            os.path.join(afDir, f"{entry}_model.pdb"),
            os.path.join(afDir, f"{entry}_model.pdb.gz"),
        ]
    candidates += [
        os.path.join(afDir, f"AF-{entry}-F{fragment}-model_v6.pdb"),
        os.path.join(afDir, f"AF-{entry}-F{fragment}-model_v6.pdb.gz"),
    ]

    for path in candidates:
        if os.path.exists(path):
            if path.endswith('.gz'):
                tmp = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
                with gzip.open(path, 'rb') as gz:
                    tmp.write(gz.read())
                tmp.close()
                print(f"  Using local (decompressed gz): {path}")
                return tmp.name, True
            else:
                print(f"  Using local: {path}")
                return path, False

    return None, False

def getAFFragment(start):
    '''Return the AlphaFold fragment number that contains the given residue.
    AF fragments are 1400 AA long with 200 AA overlap (step = 1200 AA).
    F1: 1-1400, F2: 1201-2600, F3: 2401-3800, etc.'''
    if start <= 1400:
        return 1
    return math.ceil(start / 1200)

def takePymolImage(pdbFile, start, end, output, fragment=1):
    '''Take an image of a given protein with a specified domain highlighted in pymol.
    AF fragment PDB files use local residue numbering (1 to ~1400), not global UniProt
    positions. Convert global coords to local before selecting. Fragment step = 1200 AA.'''

    #Load pymol and the protein
    cmd.reinitialize()
    cmd.load(pdbFile, "fullProtein")
    cmd.color("green", "fullProtein")

    #Convert global domain coords to local fragment coords (F1: no change, F2+: subtract offset)
    fragmentOffset = (fragment - 1) * 1200
    localStart = start - fragmentOffset
    localEnd   = end   - fragmentOffset

    #Highlight the domain red and center and zoom in on it
    cmd.select("domain", f"resi {localStart}-{localEnd}")
    if cmd.count_atoms("domain") == 0:
        raise ValueError(f"Domain residues {start}-{end} (local {localStart}-{localEnd}) "
                         f"not found in fragment F{fragment}. Check fragment step size.")
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

    # Try common font locations across macOS and Linux
    fontCandidates=[
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux (RHEL/CentOS)
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux (Debian/Ubuntu)
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    font=None
    for fontPath in fontCandidates:
        if os.path.exists(fontPath):
            font=ImageFont.truetype(fontPath, fontSize)
            break
    if font is None:
        print("Warning: no TrueType font found, falling back to default font (no size control).")
        font=ImageFont.load_default()
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

_ap = argparse.ArgumentParser(description='Step 6: Render PyMOL images of candidate domain structures.')
_ap.add_argument('--input',
                 default='/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/kat_output_library_files/04282026_metapredict/5_finalCandidateSequences_meta.tsv',
                 help='Input TSV (step 5 output)')
_ap.add_argument('--output-dir',
                 default='/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/kat_output_library_files/04282026_metapredict',
                 help='Directory where the images/ subfolder will be created')
_ap.add_argument('--af-dir', default=None, metavar='DIR',
                 help='Optional directory of cached AlphaFold PDB files (from step 3). '
                      'Checked before downloading from the web. Supports both .pdb and .pdb.gz files.')
_ap.add_argument('--sample', type=int, default=None, metavar='N',
                 help='Randomly select N rows from the input before imaging. '
                      'If N exceeds the total number of rows, all rows are used. '
                      'Selection uses a fixed random seed (42) for reproducibility.')
_args = _ap.parse_args()
input         = _args.input
baseOutputDir = _args.output_dir
imagesDir=os.path.join(baseOutputDir, 'images')
if os.path.exists(imagesDir):
    i=1
    while os.path.exists(f"{imagesDir}_{i}"):
        i+=1
    imagesDir=f"{imagesDir}_{i}"
os.makedirs(imagesDir)
tileOutput=os.path.join(imagesDir, '1_allFilters.png')
sep = "," if input.endswith(".csv") else "\t"
df=pd.read_csv(input, sep=sep)

if _args.sample is not None:
    total = len(df)
    if _args.sample >= total:
        print(f"Warning: --sample {_args.sample} >= total rows ({total}); using all rows.")
    else:
        df = df.sample(n=_args.sample, random_state=42).reset_index(drop=True)
        print(f"Randomly sampled {_args.sample} rows from {total} total (seed=42).")
print(f"Imaging {len(df)} domain(s).")

entriesDictionary={}
for entry, group in df.groupby('Entry'):#Record relevant info for each domain sequence to help create annotations
    entriesDictionary[entry]=group[['Domain', 'Domain Sequence', 'Start', 'End']].values.tolist()

    #entriesDictionary[entry]=group[['Domain', 'Domain Sequence', 'Start', 'End', 'candidateSequence']].values.tolist()
 
imagePaths=[]
annotations=[]
for entry, rows in entriesDictionary.items():#Go through each protein 
    for row in rows: #Go through each domain of a protein
        domain, domainSeq, start, end = row#, driverClass=row
        fragment=getAFFragment(start)
        print(f"Imaging {entry} (domain {start}-{end}, AF fragment F{fragment})")
        localPath, isTemp=findLocalPDB(entry, fragment, _args.af_dir)
        pdbFile=localPath if localPath else f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F{fragment}-model_v6.pdb"
        output=os.path.join(imagesDir, f"{entry}_{start}_{end}.png") #Specify the folder that will hold individual images
        try:
            takePymolImage(pdbFile, start, end, output, fragment=fragment)
        except Exception as e:
            print(f"  WARNING: Skipping {entry} ({start}-{end}) — {e}")
            if isTemp and os.path.exists(pdbFile):
                os.unlink(pdbFile)
            continue
        finally:
            if isTemp and os.path.exists(pdbFile):
                os.unlink(pdbFile)
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
