#!/usr/bin/env python3
"""
analyze_fragment_coverage.py

Reads a step-1 domain library TSV and reports which domains fall outside
AlphaFold fragment F1 (i.e. their Start position > 1400 AA).

Outputs:
  - Summary statistics printed to stdout
  - A TSV of all outside-F1 domains saved to <output>

Usage:
    python analyze_fragment_coverage.py                         # uses defaults
    python analyze_fragment_coverage.py --input 1_domainLibraryRaw.tsv \\
                                        --output domains_outside_F1.tsv
"""

import argparse
import csv
import math
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# AlphaFold fragment helpers (mirrors steps 2–4)
# ---------------------------------------------------------------------------
# AF DB splits proteins >2700 AA into overlapping 1400-AA fragments (step=200).
# Proteins <=2700 AA are never fragmented — F1 covers the full chain.
# F1: global 1-1400 (offset 0), F2: global 201-1600 (offset 200), etc.
FRAG_STEP = 200  # AA step between consecutive fragment start positions

def getAFFragment(domainStart, domainEnd, proteinLength):
    '''Return the AF fragment number containing [domainStart, domainEnd].
    Proteins <=2700 AA are not fragmented (always F1).
    For fragmented proteins: step=200, F1=1-1400, F2=201-1600, ...
    Returns None if the domain spans a boundary and cannot fit in one fragment.'''
    if proteinLength <= 2700:
        return 1
    n = max(1, math.ceil((domainEnd - 1400) / FRAG_STEP) + 1)
    if (n - 1) * FRAG_STEP + 1 > domainStart:
        return None  # domain spans a fragment boundary
    return n

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_BASE = Path(__file__).parent.parent / 'kat_output_library_files'
_DEFAULT_INPUT  = str(_BASE / '1_domainLibraryRaw.tsv')
_DEFAULT_OUTPUT = str(_BASE / 'domains_outside_F1.tsv')

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description='Report domains that fall outside AlphaFold fragment F1.')
ap.add_argument('--input',  default=_DEFAULT_INPUT,
                help='Step-1 domain library TSV (default: kat_output_library_files/1_domainLibraryRaw.tsv)')
ap.add_argument('--output', default=_DEFAULT_OUTPUT,
                help='Output TSV for outside-F1 domains (default: kat_output_library_files/domains_outside_F1.tsv)')
args = ap.parse_args()

input_path  = args.input
output_path = args.output

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found: {input_path}")

os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
outside_f1       = []
frag_counts      = {}
total_domains    = 0
all_entries      = set()
multi_frag_entries = set()

with open(input_path, newline='') as fh:
    reader = csv.DictReader(fh, delimiter='\t')
    for row in reader:
        total_domains += 1
        start  = int(row['Start'])
        end    = int(row['End'])
        length = int(row['Length'])
        entry  = row['Entry']
        frag   = getAFFragment(start, end, length)
        all_entries.add(entry)
        if frag is None:
            frag = 'boundary'  # domain spans a fragment boundary
        if frag != 1:
            outside_f1.append({
                'Entry':    entry,
                'Domain':   row['Domain'],
                'Start':    row['Start'],
                'End':      row['End'],
                'fragment': frag,
            })
            frag_counts[frag] = frag_counts.get(frag, 0) + 1
            multi_frag_entries.add(entry)

outside_f1.sort(key=lambda r: (r['Entry'], int(r['Start'])))

# ---------------------------------------------------------------------------
# Save output TSV
# ---------------------------------------------------------------------------
with open(output_path, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=['Entry', 'Domain', 'Start', 'End', 'fragment'],
                            delimiter='\t')
    writer.writeheader()
    writer.writerows(outside_f1)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
total_genes = len(all_entries)
n_outside   = len(outside_f1)
n_multi     = len(multi_frag_entries)

print("=" * 50)
print("  AlphaFold Fragment Coverage Analysis")
print("=" * 50)
print(f"  Input  : {input_path}")
print(f"  Output : {output_path}")
print()
print(f"  Total domains           : {total_domains:,}")
print(f"  Domains outside F1      : {n_outside:,}  ({n_outside / total_domains * 100:.2f}%)")
print()
print(f"  Total unique genes      : {total_genes:,}")
print(f"  Genes with >1 fragment  : {n_multi:,}  ({n_multi / total_genes * 100:.2f}%)")
print()
print("  Fragment distribution of outside-F1 domains:")
int_frags    = sorted(k for k in frag_counts if isinstance(k, int))
other_frags  = sorted(k for k in frag_counts if not isinstance(k, int))
for frag in int_frags:
    print(f"    F{frag}: {frag_counts[frag]:,} domains")
for frag in other_frags:
    print(f"    {frag}: {frag_counts[frag]:,} domains")
print("=" * 50)
