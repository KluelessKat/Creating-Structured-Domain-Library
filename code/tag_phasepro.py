#!/usr/bin/env python3
"""
tag_phasepro.py

Annotate a domain / proteome TSV with condensate observations from PhaSePro
(https://phasepro.elte.hu). For each row, looks up its UniProt accession
(`Entry` column) in the PhaSePro export and adds three columns:

    experimental    nullable int (pandas Int64)
                    # of biomolecular (natural) condensate observations.
                    NaN  → entry not in PhaSePro (unknown / untested)
                    0    → entry in PhaSePro but no experimental obs reported
                    > 0  → confirmed experimental condensate observations
    synthetic       nullable int (pandas Int64) — same semantics for
                    synthetic / engineered condensate observations.
    phasepro_in_db  bool — True iff the entry appears in PhaSePro at all.
                    Convenience flag equivalent to experimental.notna().

Why NaN for missing entries?
----------------------------
PhaSePro is a CURATED database, not a screen. An entry's absence does NOT
mean "tested and found negative" — it means "no one has reported a
condensate observation for this protein yet". Collapsing absence to 0
would erase the distinction between "in DB with zero obs in this
category" (rare but informative) and "not in DB" (the dominant ~95% of
the proteome). NaN preserves the distinction:

    df[df['experimental'] > 0]      → confirmed positive observations
    df[df['experimental'] == 0]     → in DB, zero obs in this category
    df[df['experimental'].notna()]  → entry is in PhaSePro at all
    df[df['experimental'].isna()]   → entry NOT in PhaSePro (unknown)

USAGE
-----
    python tag_phasepro.py INPUT_TSV PHASEPRO_CSV OUTPUT_TSV [--species "Homo sapiens"]

Default --species "Homo sapiens" restricts the PhaSePro lookup to human
entries (relevant for a human-proteome library). Pass --species '' to
disable the filter and accept all species.
"""

from __future__ import annotations
import argparse
import sys
import pandas as pd


def load_phasepro(csv_path: str, species: str | None) -> pd.DataFrame:
    """Read PhaSePro CSV, optionally filter to one species, return a deduped
    lookup table with Entry / experimental / synthetic / phasepro_in_db."""
    pp = pd.read_csv(csv_path)
    required = ["Uniprot ID",
                "Biomolecular condensate count",
                "Synthetic condensate count"]
    for col in required:
        if col not in pp.columns:
            sys.exit(f"ERROR: PhaSePro CSV missing column '{col}'. "
                     f"Got: {list(pp.columns)}")
    if species:
        if "Species" not in pp.columns:
            sys.exit("ERROR: --species filter requested but no 'Species' "
                     "column in PhaSePro CSV.")
        before = len(pp)
        pp = pp[pp["Species"] == species].copy()
        print(f"  PhaSePro: filtered to '{species}' "
              f"({before} → {len(pp)} rows).")
    pp = pp.rename(columns={
        "Uniprot ID": "Entry",
        "Biomolecular condensate count": "experimental",
        "Synthetic condensate count":    "synthetic",
    })
    # Coerce counts to integers; default 0 for blanks.
    for col in ("experimental", "synthetic"):
        pp[col] = pd.to_numeric(pp[col], errors="coerce").fillna(0).astype(int)
    # Some accessions may appear multiple times (rare); collapse by sum.
    pp = (pp.groupby("Entry", as_index=False)
            .agg({"experimental": "sum", "synthetic": "sum"}))
    pp["phasepro_in_db"] = True
    return pp[["Entry", "experimental", "synthetic", "phasepro_in_db"]]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_tsv",
                    help="Input TSV with an 'Entry' (UniProt accession) column.")
    ap.add_argument("phasepro_csv",
                    help="PhaSePro export CSV (download from "
                         "https://phasepro.elte.hu).")
    ap.add_argument("output_tsv",
                    help="Where to write the annotated TSV.")
    ap.add_argument("--species", default="Homo sapiens",
                    help="Restrict PhaSePro lookup to this species "
                         "(default 'Homo sapiens'). Pass '' to disable.")
    args = ap.parse_args()

    print(f"Loading input: {args.input_tsv}")
    df = pd.read_csv(args.input_tsv, sep="\t")
    if "Entry" not in df.columns:
        sys.exit("ERROR: input TSV has no 'Entry' column.")
    print(f"  {len(df):,} rows, {df['Entry'].nunique():,} unique entries.")

    print(f"Loading PhaSePro: {args.phasepro_csv}")
    species = args.species if args.species else None
    pp = load_phasepro(args.phasepro_csv, species)
    print(f"  {len(pp):,} unique condensate-annotated entries available "
          f"for lookup.")

    # Drop pre-existing experimental/synthetic columns to avoid name clash.
    for col in ("experimental", "synthetic", "phasepro_in_db"):
        if col in df.columns:
            print(f"  Warning: dropping existing '{col}' column.")
            df = df.drop(columns=col)

    merged = df.merge(pp, on="Entry", how="left")
    # Preserve NaN for entries not in PhaSePro — see module docstring.
    # pandas nullable Int64 dtype keeps integer display while allowing NaN.
    merged["experimental"]    = merged["experimental"].astype("Int64")
    merged["synthetic"]       = merged["synthetic"].astype("Int64")
    merged["phasepro_in_db"]  = merged["phasepro_in_db"].fillna(False).astype(bool)

    merged.to_csv(args.output_tsv, sep="\t", index=False)
    print(f"\nWrote {args.output_tsv}")
    print(f"  total rows:                          {len(merged):,}")

    n_in_db   = int(merged["phasepro_in_db"].sum())
    n_not_db  = len(merged) - n_in_db
    n_uniq_in_db = int(merged.loc[merged["phasepro_in_db"], "Entry"].nunique())
    n_any_exp = int((merged["experimental"] > 0).fillna(False).sum())
    n_zero_exp = int(((merged["experimental"] == 0)).fillna(False).sum())
    n_any_syn = int((merged["synthetic"]    > 0).fillna(False).sum())
    n_any_obs = int((((merged["experimental"] > 0)
                       | (merged["synthetic"] > 0))).fillna(False).sum())

    print(f"  rows in PhaSePro:                    {n_in_db:,}  "
          f"({n_in_db/len(merged)*100:.2f}%)")
    print(f"    unique entries:                    {n_uniq_in_db:,}")
    print(f"  rows NOT in PhaSePro (NaN):          {n_not_db:,}  "
          f"({n_not_db/len(merged)*100:.2f}%)   ← absence ≠ negative")
    print(f"  rows with experimental > 0:          {n_any_exp:,}")
    print(f"  rows with experimental == 0 (in DB): {n_zero_exp:,}")
    print(f"  rows with synthetic    > 0:          {n_any_syn:,}")
    print(f"  rows with ANY observation > 0:       {n_any_obs:,}")


if __name__ == "__main__":
    main()
