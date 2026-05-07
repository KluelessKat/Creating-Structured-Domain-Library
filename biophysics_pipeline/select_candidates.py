"""
select_candidates.py
====================

Helper for picking a diverse set of candidate domains for CondenSeq from
your library, based on the property distributions you computed.

Three strategies, each useful for a different goal:

1. EXTREMES  — pick domains at the extremes of single properties.
               "Top 50 most surface-aromatic + top 50 with highest |NCPR|."
               Goal: maximize chance of seeing a phenotype.

2. STRATIFIED — bin the property space into a grid and sample one (or k)
               candidate per bin.
               Goal: maximize coverage of property combinations so
               regression analyses (Kappel-style) have power.

3. PCA-DIVERSE — k-medoids in PCA space.
               Goal: a small (~50–200) maximally non-redundant panel.

USAGE
-----
    python select_candidates.py --domain-tsv output/properties_structured_domains.tsv \\
                                --strategy stratified --n 200 \\
                                --axes surface_NCPR surface_frac_aromatic \\
                                --out candidates_stratified.tsv
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

ID_COLS = ["Entry", "Gene_Name", "Domain", "Start", "End", "domain_length"]


def _ensure_id_cols(df):
    return [c for c in ID_COLS if c in df.columns]


def extremes(df: pd.DataFrame, axes: list[str], k_per_axis: int = 50) -> pd.DataFrame:
    """
    Top-k and bottom-k along each axis. Returns a dedup'd DataFrame with
    a 'selected_for' column listing which axis pulled each candidate in.
    """
    picks = {}  # row index -> set of reasons
    for ax in axes:
        if ax not in df.columns:
            print(f"  axis '{ax}' not in dataframe, skipping")
            continue
        s = pd.to_numeric(df[ax], errors="coerce")
        valid = s.dropna()
        if len(valid) < 2 * k_per_axis:
            print(f"  axis '{ax}' has only {len(valid)} valid rows; reducing k")
        top_idx = s.nlargest(k_per_axis).index.tolist()
        bot_idx = s.nsmallest(k_per_axis).index.tolist()
        for i in top_idx:
            picks.setdefault(i, []).append(f"high_{ax}")
        for i in bot_idx:
            picks.setdefault(i, []).append(f"low_{ax}")

    rows = df.loc[list(picks.keys())].copy()
    rows["selected_for"] = [";".join(picks[i]) for i in rows.index]
    return rows.reset_index(drop=True)


def stratified(df: pd.DataFrame, axes: list[str], n: int = 200,
               n_bins: int = 6, seed: int = 0) -> pd.DataFrame:
    """
    Grid-bin in `axes` and sample evenly from filled bins.
    n_bins per axis (default 6) -> up to n_bins**len(axes) cells.
    """
    rng = np.random.default_rng(seed)
    sub = df.dropna(subset=axes).copy()
    if len(sub) == 0:
        return sub
    # Quantile-based bin edges so populated regions get more granularity
    bin_cols = []
    for ax in axes:
        edges = np.unique(np.quantile(sub[ax], np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            sub[f"_bin_{ax}"] = 0
        else:
            sub[f"_bin_{ax}"] = pd.cut(sub[ax], bins=edges,
                                       labels=False, include_lowest=True)
        bin_cols.append(f"_bin_{ax}")
    grouped = sub.groupby(bin_cols, dropna=False)
    n_groups = grouped.ngroups
    per_bin = max(1, n // n_groups)
    picks = []
    for _, g in grouped:
        if len(g) <= per_bin:
            picks.append(g)
        else:
            picks.append(g.sample(per_bin, random_state=int(rng.integers(1e9))))
    out = pd.concat(picks, ignore_index=True)
    # If we undershot the budget, top up with random unselected rows
    if len(out) < n:
        leftover = sub.drop(index=out.index, errors="ignore")
        if len(leftover):
            extra = leftover.sample(min(n - len(out), len(leftover)),
                                    random_state=seed)
            out = pd.concat([out, extra], ignore_index=True)
    drop_cols = [c for c in out.columns if c.startswith("_bin_")]
    return out.drop(columns=drop_cols)


def pca_diverse(df: pd.DataFrame, axes: list[str] | None = None,
                n: int = 100, seed: int = 0) -> pd.DataFrame:
    """
    Greedy farthest-point sampling in standardized property space.
    Picks an initial random point, then repeatedly adds the row maximally
    distant from the already-selected set. Lightweight stand-in for k-medoids.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    if axes is None:
        # use every numeric column
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        axes = [c for c in num if c not in ("Start", "End", "domain_length",
                                             "length", "n_surface_residues")]
    sub = df.dropna(subset=axes).copy()
    if len(sub) <= n:
        return sub.reset_index(drop=True)
    Xs = StandardScaler().fit_transform(sub[axes].values)
    # Project to ≤8 PCs to keep distances meaningful
    k_pcs = min(8, Xs.shape[1])
    pcs = PCA(n_components=k_pcs).fit_transform(Xs)

    chosen = [int(rng.integers(0, len(pcs)))]
    min_d = np.linalg.norm(pcs - pcs[chosen[0]], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        d = np.linalg.norm(pcs - pcs[nxt], axis=1)
        min_d = np.minimum(min_d, d)
    return sub.iloc[chosen].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain-tsv", required=True,
                    help="properties_structured_domains.tsv from compute_distributions.py")
    ap.add_argument("--strategy",
                    choices=["extremes", "stratified", "pca-diverse"],
                    required=True)
    ap.add_argument("--axes", nargs="+",
                    default=["NCPR", "surface_frac_aromatic"],
                    help="Property columns to use as the selection space")
    ap.add_argument("--n", type=int, default=200,
                    help="Total panel size (stratified, pca-diverse)")
    ap.add_argument("--k-per-axis", type=int, default=50,
                    help="Top-k AND bottom-k per axis (extremes)")
    ap.add_argument("--n-bins", type=int, default=6,
                    help="Bins per axis (stratified)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True,
                    help="Output TSV path for selected candidates")
    args = ap.parse_args()

    df = pd.read_csv(args.domain_tsv, sep="\t")
    print(f"loaded {len(df)} domains")

    if args.strategy == "extremes":
        out = extremes(df, args.axes, k_per_axis=args.k_per_axis)
    elif args.strategy == "stratified":
        out = stratified(df, args.axes, n=args.n,
                         n_bins=args.n_bins, seed=args.seed)
    else:
        out = pca_diverse(df, axes=args.axes, n=args.n, seed=args.seed)

    print(f"selected {len(out)} candidates")
    out.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
