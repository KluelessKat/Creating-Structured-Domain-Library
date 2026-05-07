"""
compute_distributions.py
========================

Driver: read the human proteome TSV and the structured-domain TSV, compute
all biophysical properties for each sequence in each, and write two tidy
CSVs ready for plotting.

USAGE
-----
    python compute_distributions.py \\
        --proteome /path/to/humanProteome_KZ.tsv \\
        --domains  /path/to/1_domainLibraryRaw.tsv \\
        --pdb-dir  /path/to/alphaFold/dbFiles \\
        --outdir   ./output

The --pdb-dir flag is optional. If omitted, surface metrics are skipped.

OUTPUTS
-------
    output/properties_full_proteins.tsv
    output/properties_structured_domains.tsv
    output/property_summary.tsv         <- per-property statistics, both groups
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Optional
import pandas as pd
import numpy as np

# Local modules
from properties import compute_all, SEQUENCE_PROPERTIES
from disorder import disorder_fraction, mean_disorder_score, _HAS_METAPREDICT
try:
    import surface as surface_mod
    _HAS_SURFACE = True
except ImportError:
    _HAS_SURFACE = False


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_proteome(path: str) -> pd.DataFrame:
    """Load the UniProt human proteome TSV. Expects columns:
       Entry, Entry Name, Protein names, Gene Names, Length, Domain [FT], Sequence."""
    df = pd.read_csv(path, sep="\t")
    needed = {"Entry", "Sequence", "Length"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Proteome TSV missing columns: {missing}")
    # drop rows without sequence
    df = df.dropna(subset=["Sequence"]).copy()
    df["Sequence"] = df["Sequence"].astype(str)
    return df


def load_domains(path: str) -> pd.DataFrame:
    """Load the processed domain library TSV. Expects:
       Entry, Gene Name, Length, Domain, Start, End, Domain Length, Domain Sequence."""
    df = pd.read_csv(path, sep="\t")
    needed = {"Entry", "Domain Sequence", "Start", "End", "Domain"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Domain TSV missing columns: {missing}")
    df = df.dropna(subset=["Domain Sequence"]).copy()
    df["Domain Sequence"] = df["Domain Sequence"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Per-row property computation
# ---------------------------------------------------------------------------

def _properties_for_seq(seq: str, *, with_disorder: bool = True) -> dict:
    out = compute_all(seq)
    if with_disorder:
        out["disorder_fraction"] = disorder_fraction(seq)
        out["mean_disorder"] = mean_disorder_score(seq)
    return out


def compute_full_proteins(prot_df: pd.DataFrame,
                          pdb_dir: Optional[str] = None,
                          progress_every: int = 500) -> pd.DataFrame:
    """For every protein in the proteome, compute sequence + (optional) surface metrics."""
    print(f"[full proteins] N = {len(prot_df)}")
    rows = []
    t0 = time.time()
    for i, (_, row) in enumerate(prot_df.iterrows()):
        seq = row["Sequence"]
        rec = {
            "Entry":       row["Entry"],
            "Gene_Names":  row.get("Gene Names"),
            "length":      len(seq),
        }
        rec.update(_properties_for_seq(seq, with_disorder=True))

        # Surface metrics if PDB available
        if pdb_dir and _HAS_SURFACE:
            pdb_path = surface_mod.af_pdb_path(row["Entry"], pdb_dir)
            if pdb_path:
                try:
                    rec.update(surface_mod.surface_metrics_full_protein(pdb_path))
                except Exception as e:
                    print(f"  surface failed for {row['Entry']}: {e}", file=sys.stderr)

        rows.append(rec)
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (len(prot_df) - i - 1) / rate
            print(f"  {i+1}/{len(prot_df)}  ({rate:.1f}/s, ETA {eta/60:.1f} min)")

    df = pd.DataFrame(rows)
    df["group"] = "full_protein"
    return df


def compute_domains(dom_df: pd.DataFrame,
                    pdb_dir: Optional[str] = None,
                    progress_every: int = 500) -> pd.DataFrame:
    """For every structured domain, compute sequence + (optional) surface metrics."""
    print(f"[domains] N = {len(dom_df)}")
    rows = []
    t0 = time.time()
    for i, (_, row) in enumerate(dom_df.iterrows()):
        seq = row["Domain Sequence"]
        rec = {
            "Entry":         row["Entry"],
            "Gene_Name":     row.get("Gene Name"),
            "Domain":        row["Domain"],
            "Start":         row["Start"],
            "End":           row["End"],
            "domain_length": row.get("Domain Length", len(seq)),
            "length":        len(seq),
        }
        # We don't run metapredict on short folded domains — uninformative & slow
        rec.update(_properties_for_seq(seq, with_disorder=False))

        if pdb_dir and _HAS_SURFACE:
            pdb_path = surface_mod.af_pdb_path(row["Entry"], pdb_dir)
            if pdb_path:
                try:
                    rec.update(surface_mod.surface_metrics_domain_in_context(
                        pdb_path, int(row["Start"]), int(row["End"])))
                except Exception as e:
                    print(f"  surface failed for {row['Entry']} "
                          f"{row['Start']}-{row['End']}: {e}", file=sys.stderr)

        rows.append(rec)
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (len(dom_df) - i - 1) / rate
            print(f"  {i+1}/{len(dom_df)}  ({rate:.1f}/s, ETA {eta/60:.1f} min)")

    df = pd.DataFrame(rows)
    df["group"] = "structured_domain"
    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize(full_df: pd.DataFrame, dom_df: pd.DataFrame) -> pd.DataFrame:
    """Per-property summary table comparing the two distributions."""
    # Numeric columns common to both
    num_cols = [c for c in full_df.columns
                if c in dom_df.columns
                and pd.api.types.is_numeric_dtype(full_df[c])
                and pd.api.types.is_numeric_dtype(dom_df[c])
                and c not in ("Start", "End", "domain_length", "length")]
    rows = []
    for c in num_cols:
        a = full_df[c].dropna()
        b = dom_df[c].dropna()
        rec = {
            "property": c,
            "full_protein_n":      len(a),
            "full_protein_mean":   a.mean()   if len(a) else np.nan,
            "full_protein_median": a.median() if len(a) else np.nan,
            "full_protein_std":    a.std()    if len(a) else np.nan,
            "domain_n":            len(b),
            "domain_mean":         b.mean()   if len(b) else np.nan,
            "domain_median":       b.median() if len(b) else np.nan,
            "domain_std":          b.std()    if len(b) else np.nan,
        }
        # Effect size: Cohen's d between full and domain distributions
        if len(a) > 1 and len(b) > 1:
            sd_pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var())
                                / (len(a) + len(b) - 2))
            rec["cohen_d_domain_minus_full"] = (b.mean() - a.mean()) / sd_pooled \
                                                if sd_pooled > 0 else np.nan
        else:
            rec["cohen_d_domain_minus_full"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("property").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteome", required=True)
    ap.add_argument("--domains",  required=True)
    ap.add_argument("--pdb-dir",  default=None,
                    help="Directory holding AlphaFold PDB files named "
                         "<Entry>_model.pdb (the convention from your "
                         "downloadAlphaFoldFiles function). Optional.")
    ap.add_argument("--outdir",   default="output")
    ap.add_argument("--max-proteome", type=int, default=None,
                    help="If set, only process the first N proteins (debug).")
    ap.add_argument("--max-domains", type=int, default=None,
                    help="If set, only process the first N domains (debug).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from properties import _HAS_LOCALCIDER
    print(f"localcider available: {_HAS_LOCALCIDER}")
    print(f"metapredict available: {_HAS_METAPREDICT}")
    print(f"surface module available: {_HAS_SURFACE}")
    print(f"PDB dir: {args.pdb_dir}")

    prot = load_proteome(args.proteome)
    dom  = load_domains(args.domains)
    if args.max_proteome:
        prot = prot.head(args.max_proteome)
    if args.max_domains:
        dom  = dom.head(args.max_domains)

    full_df = compute_full_proteins(prot, args.pdb_dir)
    dom_df  = compute_domains(dom, args.pdb_dir)

    full_path = os.path.join(args.outdir, "properties_full_proteins.tsv")
    dom_path  = os.path.join(args.outdir, "properties_structured_domains.tsv")
    sum_path  = os.path.join(args.outdir, "property_summary.tsv")

    full_df.to_csv(full_path, sep="\t", index=False)
    dom_df.to_csv(dom_path, sep="\t", index=False)
    summarize(full_df, dom_df).to_csv(sum_path, sep="\t", index=False)

    print(f"\nWrote:\n  {full_path}\n  {dom_path}\n  {sum_path}")


if __name__ == "__main__":
    main()
