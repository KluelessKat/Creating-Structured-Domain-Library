import argparse
import gzip
import math
import tempfile
from pymol import cmd
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os

# ---------------------------------------------------------------------------
# AlphaFold fragment constants
# ---------------------------------------------------------------------------
# AF DB splits proteins >2700 AA into overlapping 1400-AA fragments (step=200).
# Proteins <=2700 AA use F1 only (local = global, no offset).
# F1: global 1-1400, F2: global 201-1600, F3: global 401-1800, ...
FRAG_STEP = 200

# Standard 3-letter → 1-letter amino acid lookup
_AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'SEC':'U','PYL':'O','ASX':'B','GLX':'Z','XLE':'J','UNK':'X',
}

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

def getAFFragment(domainStart, domainEnd, proteinLength):
    '''Return the AF fragment number containing [domainStart, domainEnd].
    Proteins <=2700 AA are not fragmented (always F1, local = global).
    For fragmented proteins: step=200, F1=1-1400, F2=201-1600, ...
    Returns None if the domain spans a boundary and cannot fit in one fragment.'''
    if proteinLength <= 2700:
        return 1
    n = max(1, math.ceil((domainEnd - 1400) / FRAG_STEP) + 1)
    if (n - 1) * FRAG_STEP + 1 > domainStart:
        return None
    return n

def globalToLocal(pos, fragment):
    '''Convert a global UniProt residue position to a local fragment position.
    F1: offset 0 (local = global), F2: offset 200, F3: offset 400, ...'''
    return pos - (fragment - 1) * FRAG_STEP

def _quickVerifySeq(pdbPath, localStart, domainSeq, nCheck=5):
    '''Scan a local PDB file and confirm the first nCheck residues at localStart
    match domainSeq. Pure line parsing — no BioPython required.
    Non-blocking: returns True on read errors or for URL paths.'''
    if pdbPath.startswith('http'):
        return True  # cannot verify without downloading the full file
    resnames = {}
    try:
        opener = gzip.open if pdbPath.endswith('.gz') else open
        with opener(pdbPath, 'rt', errors='ignore') as fh:
            for line in fh:
                if not line.startswith('ATOM'):
                    continue
                if line[12:16].strip() != 'CA':
                    continue
                try:
                    resnum = int(line[22:26])
                except ValueError:
                    continue
                if localStart <= resnum < localStart + nCheck:
                    resnames[resnum] = _AA3TO1.get(line[17:20].strip(), 'X')
                elif resnum >= localStart + nCheck:
                    break
    except Exception:
        return True  # non-blocking on read failure
    pdbSeq   = ''.join(resnames.get(r, '?') for r in range(localStart, localStart + nCheck))
    expected = domainSeq[:nCheck].upper()
    if '?' not in pdbSeq and pdbSeq == expected:
        return True
    print(f"  WARNING: Seq check failed — PDB local {localStart}–{localStart+nCheck-1} "
          f"= '{pdbSeq}', domain = '{expected}'")
    return False

def takePymolImage(pdbFile, localStart, localEnd, globalStart, globalEnd, output, fragment=1):
    '''Render a PyMOL image of the protein with the domain highlighted.
    localStart/localEnd are the LOCAL fragment residue numbers (1 to ~1400).
    globalStart/globalEnd are shown in the error message for debugging only.'''

    cmd.reinitialize()
    cmd.load(pdbFile, "fullProtein")
    cmd.color("green", "fullProtein")

    cmd.select("domain", f"resi {localStart}-{localEnd}")
    if cmd.count_atoms("domain") == 0:
        raise ValueError(
            f"Domain residues (global {globalStart}-{globalEnd}, local {localStart}-{localEnd}) "
            f"not found in fragment F{fragment}. Sequence verification may have failed.")
    cmd.color("red", "domain")
    cmd.orient("domain")
    cmd.zoom("domain", buffer=100)

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
    entriesDictionary[entry]=group[['Domain', 'Domain Sequence', 'Start', 'End', 'Length']].values.tolist()

imagePaths=[]
annotations=[]
for entry, rows in entriesDictionary.items():#Go through each protein
    for row in rows: #Go through each domain of a protein
        domain, domainSeq, start, end, length = row

        # Determine fragment and convert to local coords using protein length
        fragment=getAFFragment(int(start), int(end), int(length))
        if fragment is None:
            print(f"  Skipping {entry} domain {start}-{end}: spans a fragment boundary.")
            continue
        localStart=globalToLocal(int(start), fragment)
        localEnd  =globalToLocal(int(end),   fragment)
        print(f"Imaging {entry} (protein length {int(length)}, domain {start}-{end}, "
              f"AF F{fragment}, local {localStart}-{localEnd})")

        localPath, isTemp=findLocalPDB(entry, fragment, _args.af_dir)
        pdbFile=localPath if localPath else f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F{fragment}-model_v6.pdb"

        # Safety check: verify first 5 residues match before rendering
        if localPath and not _quickVerifySeq(pdbFile, localStart, str(domainSeq)):
            print(f"  WARNING: Sequence mismatch for {entry} F{fragment} — "
                  f"check fragment/length logic. Proceeding with caution.")

        output=os.path.join(imagesDir, f"{entry}_{start}_{end}.png")
        try:
            takePymolImage(pdbFile, localStart, localEnd, int(start), int(end),
                           output, fragment=fragment)
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
