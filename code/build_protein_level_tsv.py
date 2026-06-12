#!/usr/bin/env python3
"""
build_protein_level_tsv.py

Convert a UniProt-style proteome TSV (one row per protein) into a TSV that
steps 3 and 4 can consume as if each protein were a single "full-length
domain". Useful when you want protein-level metrics (surface composition,
Rg, structure source) to compare against the domain-level distribution in
histograms_and_weights.py.

Input columns required:
    Entry      UniProt accession
    Length     full protein length
    Sequence   full protein sequence

Optionally preserved if present:
    Gene Names / Gene Name

Output columns (matches the schema steps 3 and 4 expect):
    Entry, Gene Name, Length, Domain, Start, End, Domain Length, Domain Sequence

Each row has Domain="full_protein", Start=1, End=Length, and
Domain Sequence equal to the full protein sequence.

By default, proteins longer than --max-length (2700, the AF DB
single-fragment limit) are dropped — steps 3 and 4 would skip them anyway
because their "domain" can't fit in one AlphaFold fragment.

USAGE
-----
    python build_protein_level_tsv.py PROTEOME_TSV OUTPUT_TSV [--max-length 2700]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_tsv",
                    help="UniProt-style proteome TSV with Entry / Length / "
                         "Sequence columns (e.g. humanProteome_KZ.tsv).")
    ap.add_argument("output_tsv",
                    help="Where to write the protein-as-domain TSV.")
    ap.add_argument("--max-length", type=int, default=2700,
                    help="Drop proteins longer than this. Default 2700 (AF "
                         "DB single-fragment limit). Pass 0 to keep all "
                         "proteins — but >2700 AA rows will be silently "
                         "skipped by steps 3 and 4 downstream.")
    args = ap.parse_args()

    print(f"Loading proteome: {args.input_tsv}")
    df = pd.read_csv(args.input_tsv, sep="\t")
    for col in ("Entry", "Length", "Sequence"):
        if col not in df.columns:
            sys.exit(f"ERROR: input missing required column '{col}'. "
                     f"Got: {list(df.columns)}")
    before = len(df)
    print(f"  {before:,} input rows.")

    # Clean: drop rows missing any required field, coerce Length to int.
    df = df.dropna(subset=["Entry", "Length", "Sequence"]).copy()
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
    df = df.dropna(subset=["Length"]).copy()
    df["Length"] = df["Length"].astype(int)
    df["Sequence"] = df["Sequence"].astype(str)
    n_clean = len(df)
    if n_clean < before:
        print(f"  Dropped {before - n_clean:,} rows with missing Entry/"
              f"Length/Sequence.")

    # Optional length filter — by default drop proteins >2700 AA since steps
    # 3 and 4 would skip them anyway.
    if args.max_length and args.max_length > 0:
        before_len = len(df)
        df = df[df["Length"] <= args.max_length].copy()
        n_dropped = before_len - len(df)
        if n_dropped:
            print(f"  Dropped {n_dropped:,} proteins with Length > "
                  f"{args.max_length} (would be skipped downstream).")

    # Pick a Gene Name column if available; UniProt exports use "Gene Names".
    gene_col = "Gene Names" if "Gene Names" in df.columns else (
               "Gene Name" if "Gene Name" in df.columns else None)
    gene_vals = df[gene_col] if gene_col else ""

    # Build the protein-as-domain rows.
    out = pd.DataFrame({
        "Entry":           df["Entry"].astype(str),
        "Gene Name":       gene_vals,
        "Length":          df["Length"],
        "Domain":          "full_protein",
        "Start":           1,
        "End":             df["Length"],
        "Domain Length":   df["Length"],
        "Domain Sequence": df["Sequence"],
    })
    # Ensure the output directory exists so the to_csv open() doesn't fail.
    out_path = Path(args.output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)

    print(f"\nWrote {out_path}")
    print(f"  {len(out):,} protein-level 'domain' rows.")
    if args.max_length and args.max_length > 0:
        print(f"  All proteins ≤ {args.max_length} AA.")


if __name__ == "__main__":
    main()
