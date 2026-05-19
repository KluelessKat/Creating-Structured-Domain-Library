#!/usr/bin/env python3
"""
5_obtainFinalCandidateSequences.py

Classify structured domains by the mechanism driving condensate formation
(aromatic interactions, charge interactions, both, or neither) using
percentile-based thresholds, then save the final candidate sequence library.

Produces:
  - <output>              TSV of all scored domains with candidateSequence label
  - <output_dir>/plots.pdf  Three diagnostic scatter plots (mirrors Rplots.pdf)

Usage (standalone):
    python 5_obtainFinalCandidateSequences.py input.tsv output.tsv

Usage (via run_pipeline.py):
    Automatically called with positional args: <input> <output>
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe on cluster nodes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


# ---------------------------------------------------------------------------
# Hardcoded defaults (overridden by positional args when called from pipeline)
# ---------------------------------------------------------------------------
_DEFAULT_INPUT  = '/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/jpl_exampleLibraryFiles/4_domainLibraryPhysicalProperties_jpl.tsv'
_DEFAULT_OUTPUT = '/Users/katherinezhang/Downloads/Kappel_2026SpringRotation/Creating-Structured-Domain-Library/jpl_exampleLibraryFiles/5_finalCandidateSequences_jpl2.tsv'

ap = argparse.ArgumentParser(description='Step 5: Classify domains and select final candidates.')
ap.add_argument('input',  nargs='?', default=_DEFAULT_INPUT,
                help='Input TSV (step 4 output)')
ap.add_argument('output', nargs='?', default=_DEFAULT_OUTPUT,
                help='Output TSV path')
args = ap.parse_args()

inputPath  = Path(args.input)
outputPath = Path(args.output)
outputPath.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv(inputPath, sep='\t', na_values=['', 'NA'])

# Ensure metric columns are numeric
for col in ['interactionIndex', 'aromaticSurfaceFraction', 'Rg(Compactness)',
            'fractionBuried', 'positiveSurfaceFraction', 'negativeSurfaceFraction']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Compute additional charge metrics
df['netSurfaceCharge']    = df['positiveSurfaceFraction'] - df['negativeSurfaceFraction']
df['totalChargeFraction'] = df['positiveSurfaceFraction'] + df['negativeSurfaceFraction']

# Keep only rows with all required values (mirrors R's filter(!is.na(...)))
required = ['interactionIndex', 'aromaticSurfaceFraction', 'Rg(Compactness)',
            'totalChargeFraction', 'netSurfaceCharge']
dfPlot = df.dropna(subset=required).copy()
print(f"Rows with all required values: {len(dfPlot):,} / {len(df):,}")

# ---------------------------------------------------------------------------
# Percentile-based thresholds (mirrors R quantile() calls)
# ---------------------------------------------------------------------------
lowInteractionThreshold = np.quantile(dfPlot['interactionIndex'],        0.30)
highAromaticThreshold   = np.quantile(dfPlot['aromaticSurfaceFraction'], 0.75)
highChargeThreshold     = np.quantile(dfPlot['totalChargeFraction'],     0.75)
rgLower                 = np.quantile(dfPlot['Rg(Compactness)'],         0.25)
rgUpper                 = np.quantile(dfPlot['Rg(Compactness)'],         0.75)

print(f"Thresholds:")
print(f"  interactionIndex      < {lowInteractionThreshold:.4f}  (30th pct)")
print(f"  aromaticSurfaceFraction > {highAromaticThreshold:.4f}  (75th pct)")
print(f"  totalChargeFraction   > {highChargeThreshold:.4f}  (75th pct)")
print(f"  Rg(Compactness) in ({rgLower:.2f}, {rgUpper:.2f})  (IQR)")

# ---------------------------------------------------------------------------
# Classification (mirrors R case_when logic)
# ---------------------------------------------------------------------------
rg = dfPlot['Rg(Compactness)']
rg_ok = (rg > rgLower) & (rg < rgUpper)

aromaticCandidate = (
    (dfPlot['interactionIndex']        < lowInteractionThreshold) &
    (dfPlot['aromaticSurfaceFraction'] > highAromaticThreshold)   &
    rg_ok
)
chargeCandidate = (
    (dfPlot['interactionIndex']   < lowInteractionThreshold) &
    (dfPlot['totalChargeFraction'] > highChargeThreshold)    &
    rg_ok
)

conditions = [
    aromaticCandidate & chargeCandidate,
    aromaticCandidate & ~chargeCandidate,
    ~aromaticCandidate & chargeCandidate,
]
choices = ['Both', 'Aromatic-driven', 'Charge-driven']
dfPlot['candidateSequence'] = np.select(conditions, choices, default='Neither')

counts = dfPlot['candidateSequence'].value_counts()
print(f"\nCandidate breakdown:\n{counts.to_string()}")

# ---------------------------------------------------------------------------
# Plots  (mirrors R ggplot2 figures → saved to plots.pdf)
# ---------------------------------------------------------------------------
pdfPath = outputPath.parent / 'plots.pdf'

def _scale_sizes(series, min_size=10, max_size=120):
    '''Scale a numeric series to point sizes for scatter plots.'''
    rng = series.max() - series.min()
    if rng == 0:
        return np.full(len(series), (min_size + max_size) / 2)
    return (series - series.min()) / rng * (max_size - min_size) + min_size

def _candidate_region_rect(ax, xmax, ymin, color):
    '''Draw a semi-transparent rectangle from x=-inf..xmax, y=ymin..+inf.'''
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    rect = mpatches.Rectangle(
        (xlim[0], ymin),
        xmax - xlim[0],
        ylim[1] - ymin,
        facecolor=color, alpha=0.1, edgecolor='none', zorder=0)
    ax.add_patch(rect)
    # Restore limits — adding a patch can expand them
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

with PdfPages(pdfPath) as pdf:
    plt.rcParams.update({'font.size': 12})

    # --- Plot 1: Aromatic surface vs interaction index ----------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = _scale_sizes(dfPlot['Rg(Compactness)'])
    sc = ax.scatter(dfPlot['interactionIndex'], dfPlot['aromaticSurfaceFraction'],
                    c=dfPlot['fractionBuried'], s=sizes,
                    cmap='plasma', alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Fraction Buried')
    ax.set_xlabel('Interaction Index with Parent Protein')
    ax.set_ylabel('Aromatic Surface Fraction')
    ax.set_title('Aromatic Surface vs Interaction Index')
    _candidate_region_rect(ax, lowInteractionThreshold, highAromaticThreshold, 'red')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Plot 2: Charge surface vs interaction index ------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = _scale_sizes(dfPlot['Rg(Compactness)'])
    vmax = dfPlot['netSurfaceCharge'].abs().max()
    sc = ax.scatter(dfPlot['interactionIndex'], dfPlot['totalChargeFraction'],
                    c=dfPlot['netSurfaceCharge'], s=sizes,
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Net Surface Charge')
    ax.set_xlabel('Interaction Index')
    ax.set_ylabel('Total Charged Surface Fraction')
    ax.set_title('Charge Surface vs Interaction Index')
    _candidate_region_rect(ax, lowInteractionThreshold, highChargeThreshold, 'blue')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Plot 3: Final candidate sequences ----------------------------------
    colorMap = {
        'Aromatic-driven': 'purple',
        'Charge-driven':   'blue',
        'Both':            'red',
        'Neither':         'grey',
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in ['Neither', 'Aromatic-driven', 'Charge-driven', 'Both']:
        group = dfPlot[dfPlot['candidateSequence'] == label]
        if group.empty:
            continue
        sizes = _scale_sizes(dfPlot['Rg(Compactness)'])  # consistent scale
        group_sizes = sizes.loc[group.index]
        ax.scatter(group['interactionIndex'], group['aromaticSurfaceFraction'],
                   s=group_sizes, c=colorMap[label], label=label, alpha=0.8)
    ax.set_xlabel('Interaction Index')
    ax.set_ylabel('Aromatic Surface Fraction')
    ax.set_title('Final Candidate Sequences')
    ax.legend(title='Driver Class')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved plots to {pdfPath}")

# ---------------------------------------------------------------------------
# Save output TSV
# ---------------------------------------------------------------------------
finalCandidates = (dfPlot[['Entry', 'Domain', 'Domain Sequence', 'Start', 'End',
                             'Length', 'candidateSequence']]
                   .copy()
                   .sort_values('candidateSequence'))
finalCandidates.to_csv(outputPath, sep='\t', index=False)
print(f"Saved {len(finalCandidates):,} domain sequences to {outputPath}")
