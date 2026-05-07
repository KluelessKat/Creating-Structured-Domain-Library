"""
plot_distributions.py
=====================

Visualizations on top of compute_distributions.py output.

Produces:
1.  Paired histograms (full proteins vs structured domains) for every numeric
    property, with KDE overlay and shaded percentile markers.
2.  Diagram-of-states style 2D plots (NCPR vs FCR; surface_NCPR vs
    surface_frac_aromatic) — these are far more informative than 1D histograms
    for picking diverse candidates.
3.  Correlation heatmap of all properties on the structured-domain set, with
    hierarchical clustering. Tells you which metrics carry independent
    information vs. which are redundant — directly answering "how much weight
    should I put on each".
4.  PCA biplot of the domain library colored by property of interest. Lets you
    visually identify clusters and pick candidates that span the space.
5.  Percentile-rank table for each domain — for any candidate you consider,
    you can look up where it lies in each distribution at a glance.

Run after compute_distributions.py:

    python plot_distributions.py \\
        --full-tsv   output/properties_full_proteins.tsv \\
        --domain-tsv output/properties_structured_domains.tsv \\
        --outdir     output/figures
"""

from __future__ import annotations
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
from scipy.cluster.hierarchy import linkage, leaves_list

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Properties grouped for figure layout
PROPERTY_ORDER = [
    # composition
    "FCR", "NCPR", "abs_NCPR",
    "frac_aromatic", "frac_R", "frac_K", "frac_D", "frac_E",
    "frac_positive", "frac_negative",
    "frac_proline", "frac_glycine", "frac_hydrophobic", "frac_polar",
    "mean_hydropathy",
    # patterning
    "kappa", "omega", "SCD", "SHD",
    # surface (may be absent if no PDB dir was supplied)
    "surface_FCR", "surface_NCPR", "surface_abs_NCPR",
    "surface_frac_aromatic", "surface_frac_positive", "surface_frac_negative",
    # disorder
    "disorder_fraction", "mean_disorder",
]

# Properties whose values scale with sequence length — direct full-protein
# vs short-domain comparisons of these are length-confounded.
LENGTH_DEPENDENT = {"SCD", "SHD"}


# --------------------------------------------------------------------------
# 1. Paired histograms
# --------------------------------------------------------------------------

def paired_histograms(full: pd.DataFrame, dom: pd.DataFrame,
                      outpath: str,
                      properties: list = None):
    """Grid of histograms, one per property, full vs domain overlay."""
    if properties is None:
        properties = [p for p in PROPERTY_ORDER
                      if p in full.columns and p in dom.columns]

    n = len(properties)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, prop in zip(axes, properties):
        a = pd.to_numeric(full[prop], errors="coerce").dropna()
        b = pd.to_numeric(dom[prop],  errors="coerce").dropna()
        if len(a) == 0 and len(b) == 0:
            ax.set_visible(False)
            continue

        # Common bin edges so distributions are directly comparable
        combined = np.concatenate([a.values, b.values])
        if len(combined) < 2:
            ax.set_visible(False)
            continue
        lo, hi = np.nanpercentile(combined, [0.5, 99.5])
        if lo == hi:
            lo, hi = combined.min(), combined.max() + 1e-9
        bins = np.linspace(lo, hi, 50)

        # Convert counts to "% of population" so heights are directly
        # interpretable (a bar at 8 means 8% of that group falls in this bin).
        # Using weights= rather than density= avoids x-unit-dependent y values.
        wa = np.full_like(a.values, 100.0 / len(a), dtype=float)
        wb = np.full_like(b.values, 100.0 / len(b), dtype=float)
        ax.hist(a, bins=bins, weights=wa, alpha=0.45,
                color="#4C72B0", label=f"full proteins (n={len(a)})")
        ax.hist(b, bins=bins, weights=wb, alpha=0.55,
                color="#DD8452", label=f"domains (n={len(b)})")

        # Median markers
        if len(a):
            ax.axvline(a.median(), color="#4C72B0", lw=1.5, ls="--", alpha=0.9)
        if len(b):
            ax.axvline(b.median(), color="#DD8452", lw=1.5, ls="--", alpha=0.9)

        # KS test as a one-line annotation
        if len(a) > 5 and len(b) > 5:
            ks = ks_2samp(a, b)
            ax.text(0.02, 0.97, f"KS D={ks.statistic:.2f}",
                    transform=ax.transAxes, fontsize=8,
                    va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax.set_title(prop + (" ⚠ length-dep" if prop in LENGTH_DEPENDENT else ""),
                     fontsize=10,
                     color="darkred" if prop in LENGTH_DEPENDENT else "black")
        ax.set_xlabel("")
        ax.set_ylabel("% of population")
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for ax in axes[len(properties):]:
        ax.set_visible(False)

    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Biophysical property distributions:\n"
                 "human full proteins vs. structured domain library",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


# --------------------------------------------------------------------------
# 2. Diagram-of-states style 2D maps
# --------------------------------------------------------------------------

def diagram_of_states(dom: pd.DataFrame, outpath: str):
    """
    Two side-by-side 2D density plots that are particularly useful for
    CondenSeq candidate selection:
      Left:  bulk NCPR vs FCR  (Das & Pappu / Kappel et al. landscape)
      Right: surface NCPR vs surface fraction aromatic (CondenSeq drivers)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    def _hex(ax, x, y, xl, yl, title):
        x = pd.to_numeric(x, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() < 5:
            ax.text(0.5, 0.5, "not enough data",
                    transform=ax.transAxes, ha="center", va="center")
            return
        hb = ax.hexbin(x[m], y[m], gridsize=40, mincnt=1,
                       cmap="viridis", norm=LogNorm())
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=11)
        cb = plt.colorbar(hb, ax=ax, shrink=0.8)
        cb.set_label("domain count (log)")

    _hex(axes[0], dom.get("NCPR"), dom.get("FCR"),
         "NCPR", "FCR (fraction charged residues)",
         "Bulk charge landscape\n(Das–Pappu diagram of states)")

    if "surface_frac_aromatic" in dom.columns and "surface_NCPR" in dom.columns:
        _hex(axes[1], dom["surface_NCPR"], dom["surface_frac_aromatic"],
             "surface NCPR", "surface fraction aromatic",
             "Surface property landscape\n(CondenSeq-relevant drivers)")
    else:
        axes[1].text(0.5, 0.5, "surface metrics not computed\n"
                              "(re-run with --pdb-dir)",
                     transform=axes[1].transAxes, ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


# --------------------------------------------------------------------------
# 3. Correlation heatmap with hierarchical clustering
# --------------------------------------------------------------------------

def correlation_heatmap(dom: pd.DataFrame, outpath: str):
    """Spearman correlations among domain properties, ordered by clustering."""
    cols = [p for p in PROPERTY_ORDER if p in dom.columns]
    # drop columns that are all-NaN or constant
    sub = dom[cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(axis=1, how="all")
    sub = sub.loc[:, sub.std(numeric_only=True) > 0]
    if sub.shape[1] < 2:
        print("  correlation heatmap: not enough numeric columns")
        return

    corr = sub.corr(method="spearman")
    # cluster ordering
    dist = 1 - np.abs(corr.values)
    np.fill_diagonal(dist, 0.0)
    # condensed upper triangle for linkage
    iu = np.triu_indices_from(dist, k=1)
    condensed = dist[iu]
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    corr_o = corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(0.45 * len(corr_o) + 2,
                                    0.45 * len(corr_o) + 2))
    im = ax.imshow(corr_o.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(corr_o))); ax.set_yticks(range(len(corr_o)))
    ax.set_xticklabels(corr_o.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr_o.columns, fontsize=8)
    ax.set_title("Spearman correlation (structured domains)\n"
                 "clustered — use to pick non-redundant filtering axes",
                 fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.7, label="ρ")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


# --------------------------------------------------------------------------
# 4. PCA biplot
# --------------------------------------------------------------------------

def pca_plot(dom: pd.DataFrame, outpath: str, color_by: str = "FCR"):
    """PC1/PC2 of all numeric domain properties, colored by chosen property."""
    cols = [p for p in PROPERTY_ORDER if p in dom.columns]
    X = dom[cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    keep_rows = X.notna().all(axis=1)
    X = X[keep_rows]
    if X.shape[0] < 10 or X.shape[1] < 2:
        print("  pca: insufficient complete rows")
        return
    Xs = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    color_vals = pd.to_numeric(dom.loc[keep_rows, color_by], errors="coerce") \
                  if color_by in dom.columns else None
    sc = axes[0].scatter(pcs[:, 0], pcs[:, 1], c=color_vals, cmap="viridis",
                          s=8, alpha=0.6)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].set_title(f"Structured domains in property PC space "
                      f"(n={X.shape[0]})\ncolor: {color_by}")
    if color_vals is not None:
        plt.colorbar(sc, ax=axes[0], label=color_by, shrink=0.8)

    # Loadings as arrows
    loadings = pca.components_.T  # (features, 2)
    feat_names = X.columns.tolist()
    scale = np.abs(pcs).max() * 0.95
    for i, name in enumerate(feat_names):
        axes[1].arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale,
                      head_width=0.05 * scale, alpha=0.7,
                      color="black", length_includes_head=True)
        axes[1].text(loadings[i, 0] * scale * 1.08,
                     loadings[i, 1] * scale * 1.08,
                     name, fontsize=8, ha="center", va="center")
    axes[1].set_xlim(-scale * 1.3, scale * 1.3)
    axes[1].set_ylim(-scale * 1.3, scale * 1.3)
    axes[1].axhline(0, color="gray", lw=0.5); axes[1].axvline(0, color="gray", lw=0.5)
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].set_title("PCA loadings\n(properties pointing in the same direction\n"
                      "are correlated — pick from different directions for diversity)")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


# --------------------------------------------------------------------------
# 5. Percentile-rank table
# --------------------------------------------------------------------------

def percentile_table(dom: pd.DataFrame, outpath: str):
    """For every domain, percentile rank in each numeric property within the
    domain library. Useful: when you pick a candidate, this tells you 'this
    domain is at the 95th percentile for surface aromaticity', etc."""
    cols = [p for p in PROPERTY_ORDER if p in dom.columns]
    nums = dom[cols].apply(pd.to_numeric, errors="coerce")
    pct = nums.rank(pct=True) * 100  # 0–100 percentile within the domain set

    id_cols = [c for c in ("Entry", "Gene_Name", "Domain", "Start", "End",
                            "domain_length")
               if c in dom.columns]
    out = pd.concat([dom[id_cols].reset_index(drop=True),
                     pct.add_suffix("_pct").reset_index(drop=True)], axis=1)
    out.to_csv(outpath, sep="\t", index=False)
    print(f"  wrote {outpath}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-tsv",   required=True,
                    help="properties_full_proteins.tsv from compute_distributions.py")
    ap.add_argument("--domain-tsv", required=True,
                    help="properties_structured_domains.tsv from compute_distributions.py")
    ap.add_argument("--outdir",     default="output/figures")
    ap.add_argument("--pca-color",  default="FCR",
                    help="Property to color the PCA plot by")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    full = pd.read_csv(args.full_tsv, sep="\t")
    dom  = pd.read_csv(args.domain_tsv, sep="\t")
    print(f"loaded full proteins: {len(full)}  domains: {len(dom)}")

    paired_histograms(full, dom,
                      os.path.join(args.outdir, "01_paired_histograms.png"))
    diagram_of_states(dom,
                      os.path.join(args.outdir, "02_diagram_of_states.png"))
    correlation_heatmap(dom,
                        os.path.join(args.outdir, "03_correlation_heatmap.png"))
    pca_plot(dom,
             os.path.join(args.outdir, "04_pca.png"),
             color_by=args.pca_color)
    percentile_table(dom,
                     os.path.join(args.outdir, "05_domain_percentiles.tsv"))


if __name__ == "__main__":
    main()
