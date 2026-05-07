"""
histograms_and_weights.py
=========================

Single-script analysis tool for your structured-domain library pipeline.

Takes:
  1. The full UniProt human proteome TSV (humanProteome_KZ.tsv)
  2. Your pipeline's stage-4 output with all biophysical metrics
     (4_domainLibraryPhysicalProperties_meta.tsv)
  3. Optionally, your stage-5 output with candidate labels
     (5_finalCandidateSequences_meta.tsv) — when provided, the script also
     does feature-importance analysis to guide your weighting in step 5.

Produces:
  Section A (always)   — full proteome vs all structured domains
  Section B (always)   — by-domain-family panels (EF-hand, EGF-like, etc.)
  Section C (if step-5 file given) — candidate-vs-non-candidate analysis +
                                     suggested feature weights with
                                     standardized logistic-regression
                                     coefficients.

USAGE
-----
    python histograms_and_weights.py \\
        --proteome  humanProteome_KZ.tsv \\
        --domains   4_domainLibraryPhysicalProperties_meta.tsv \\
        --candidates 5_finalCandidateSequences_meta.tsv \\
        --outdir    histograms_output

The --candidates flag is optional; without it, Section C is skipped.

NOTES
-----
- Full-proteome metrics are computed on full sequences using simple
  composition + hydropathy. Surface metrics (which require AlphaFold PDBs
  per protein) are NOT computed for the full proteome here — they are
  domain-specific. Section A therefore only compares composition metrics
  for the full proteome vs domain library; surface metrics are shown for
  domains only.
- For each property, Cohen's d is reported alongside the histogram so you
  can immediately see which properties differ most between the groups.
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
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===========================================================================
#  Sequence-based property functions (kept self-contained so this file is
#  one drop-in script with no local-module imports).
# ===========================================================================

POSITIVE = set("RK")
NEGATIVE = set("DE")
CHARGED  = POSITIVE | NEGATIVE
AROMATIC = set("FWY")
HYDROPHOBIC_ALIPHATIC = set("AILMV")
POLAR = set("STNQH")
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
      'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
      'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}


def _clean(seq):
    if not isinstance(seq, str):
        return ""
    return "".join(c for c in seq.upper() if c in VALID_AA)


def seq_properties(seq):
    """Composition + hydropathy properties for a single sequence."""
    s = _clean(seq)
    if not s:
        return {k: np.nan for k in [
            "FCR","NCPR","abs_NCPR","frac_aromatic","frac_R","frac_K",
            "frac_D","frac_E","frac_W", "frac_Y", "frac_F", "frac_proline","frac_glycine",
            "frac_hydrophobic","frac_polar","mean_hydropathy"
        ]}
    n = len(s)
    nR=s.count('R'); nK=s.count('K'); nD=s.count('D'); nE=s.count('E')
    nP=s.count('P'); nG=s.count('G')
    nW=s.count('W'); nY=s.count('Y'); nF=s.count('F')
    nAro=sum(1 for c in s if c in AROMATIC)
    nHyd=sum(1 for c in s if c in HYDROPHOBIC_ALIPHATIC)
    nPol=sum(1 for c in s if c in POLAR)
    pos = nR+nK; neg = nD+nE
    return {
        "FCR":              (pos+neg)/n,
        "NCPR":             (pos-neg)/n,
        "abs_NCPR":         abs(pos-neg)/n,
        "frac_aromatic":    nAro/n,
        "frac_R":           nR/n,
        "frac_K":           nK/n,
        "frac_D":           nD/n,
        "frac_E":           nE/n,
        "frac_W":           nW/n,
        "frac_Y":           nY/n,
        "frac_F":           nF/n,
        "frac_proline":     nP/n,
        "frac_glycine":     nG/n,
        "frac_hydrophobic": nHyd/n,
        "frac_polar":       nPol/n,
        "mean_hydropathy":  np.mean([KD[c] for c in s]),
    }


# ===========================================================================
#  Loading
# ===========================================================================

def load_inputs(proteome_path, domains_path, candidates_path=None):
    print(f"Loading proteome: {proteome_path}")
    prot = pd.read_csv(proteome_path, sep="\t")
    prot = prot.dropna(subset=["Sequence"]).copy()
    prot["Sequence"] = prot["Sequence"].astype(str)
    print(f"  {len(prot)} proteins")

    print(f"Loading domains: {domains_path}")
    dom = pd.read_csv(domains_path, sep="\t")
    dom = dom.dropna(subset=["Domain Sequence"]).copy()
    dom["Domain Sequence"] = dom["Domain Sequence"].astype(str)
    print(f"  {len(dom)} domains, {dom['Domain'].nunique()} unique families")

    cand = None
    if candidates_path and os.path.exists(candidates_path):
        print(f"Loading candidates: {candidates_path}")
        cand = pd.read_csv(candidates_path, sep="\t")
        print(f"  {len(cand)} candidates  (classes: "
              f"{dict(cand['candidateSequence'].value_counts())})")
    return prot, dom, cand


def add_seq_properties(df, seq_col):
    """Append composition properties as new columns."""
    rows = [seq_properties(s) for s in df[seq_col]]
    feats = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, feats], axis=1)


# ===========================================================================
#  Plot helpers
# ===========================================================================

def cohen_d(a, b):
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                 / (len(a)+len(b)-2))
    return (b.mean() - a.mean()) / s if s > 0 else np.nan


def _hist_panel(ax, a, b, label_a, label_b, title, color_a="#4C72B0",
                color_b="#DD8452", show_ks=True, show_d=True):
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) == 0 and len(b) == 0:
        ax.set_visible(False); return
    combined = np.concatenate([a.values, b.values])
    if len(combined) < 2:
        ax.set_visible(False); return
    lo, hi = np.nanpercentile(combined, [0.5, 99.5])
    if lo == hi:
        lo, hi = combined.min(), combined.max()+1e-9
    bins = np.linspace(lo, hi, 50)
    if len(a):
        ax.hist(a, bins=bins, weights=np.full(len(a), 100.0/len(a)),
                alpha=0.45, color=color_a, label=f"{label_a} (n={len(a)})")
        ax.axvline(a.median(), color=color_a, ls="--", lw=1.3, alpha=0.9)
    if len(b):
        ax.hist(b, bins=bins, weights=np.full(len(b), 100.0/len(b)),
                alpha=0.55, color=color_b, label=f"{label_b} (n={len(b)})")
        ax.axvline(b.median(), color=color_b, ls="--", lw=1.3, alpha=0.9)
    annot = []
    if show_ks and len(a) > 5 and len(b) > 5:
        annot.append(f"KS D={ks_2samp(a, b).statistic:.2f}")
    if show_d:
        d = cohen_d(a, b)
        if not np.isnan(d):
            annot.append(f"d={d:+.2f}")
    if annot:
        ax.text(0.02, 0.97, "  ".join(annot), transform=ax.transAxes,
                fontsize=8, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("% of population")
    ax.tick_params(labelsize=8)


# ===========================================================================
#  Section A: full proteome vs all structured domains
# ===========================================================================

# Sequence-based metrics (computable from sequences only — for both groups).
COMP_PROPS = ["FCR","NCPR","abs_NCPR","frac_aromatic","frac_R","frac_K", 
              "frac_W", "frac_Y", "frac_F",
              "frac_D","frac_E","frac_proline","frac_glycine",
              "frac_hydrophobic","frac_polar","mean_hydropathy"]

# Surface/structural metrics (domains only — already in your TSV).
SURFACE_PROPS = ["surfaceFraction","aromaticSurfaceFraction",
                 "positiveSurfaceFraction","negativeSurfaceFraction",
                 "Rg(Compactness)"]

# Structural-context metrics (domains only).
CONTEXT_PROPS = ["anchoringIndex","fractionBuried","contactDensity",
                 "interactionIndex","meanDomainplddt"]


def section_A_proteome_vs_domains(prot, dom, outdir):
    """Full-proteome composition vs domain library composition + domain-only
    surface / structural panels."""
    print("\n--- Section A: full proteome vs all structured domains ---")
    props = COMP_PROPS + SURFACE_PROPS + ["interactionIndex"]

    n = len(props); ncols = 5; nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = axes.flatten()

    for ax, prop in zip(axes, props):
        if prop in COMP_PROPS:
            a = prot[prop]; b = dom[prop]
            _hist_panel(ax, a, b, "full proteins", "domains", prop)
        else:
            # surface metrics — domain-only, plot as single distribution
            b = dom[prop]
            b = pd.to_numeric(b, errors="coerce").dropna()
            if len(b) == 0:
                ax.set_visible(False); continue
            lo, hi = np.nanpercentile(b, [0.5, 99.5])
            if lo == hi: lo, hi = b.min(), b.max()+1e-9
            bins = np.linspace(lo, hi, 50)
            ax.hist(b, bins=bins, weights=np.full(len(b), 100.0/len(b)),
                    alpha=0.6, color="#DD8452",
                    label=f"domains (n={len(b)})")
            ax.axvline(b.median(), color="#DD8452", ls="--", lw=1.3)
            ax.set_title(prop + "  (domains only)", fontsize=10)
            ax.set_ylabel("% of population")
            ax.tick_params(labelsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Section A — full human proteome vs all structured domains\n"
                 "(blue = proteome, orange = domains; surface metrics are "
                 "domain-only because they require structures)",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = os.path.join(outdir, "A_proteome_vs_domains.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")

# ===========================================================================
#  Section B helpers — Rg normalization, sequence variability, patterning
# ===========================================================================
 
# ---- B helper 1: Rg normalization ----------------------------------------
# For IDRs, ALBATROSS uses an "analytical Flory random-coil" (AFRC) null model:
#   Rg_expected ~ 2.49 * N^0.50  (Gaussian chain; Lotthammer et al. 2024)
#
# That model assumes the chain has NO preferred structure, so it is the
# *worst-case* baseline for compactness. Folded domains are much more compact
# than random coils, so dividing by the AFRC would give normalized Rg << 1
# for nearly everything and compress all the interesting variation.
#
# A better null model for STRUCTURED domains is a COMPACT GLOBULE, where:
#   Rg ~ R0 * N^(1/3)        (constant density approximation)
#
# R0 is fitted empirically from your own data so it reflects your domain
# library rather than a theoretical ideal. We then report:
#   Rg_norm = Rg_observed / (R0 * N^(1/3))
#
# Rg_norm > 1  → domain is MORE expanded than average-for-its-size
# Rg_norm = 1  → domain compactness matches the typical globular scaling
# Rg_norm < 1  → domain is MORE compact than average-for-its-size
#
# This is directly analogous to what ALBATROSS does (Fig. 4c,f), but the
# exponent is 1/3 instead of 0.5 to reflect folded rather than disordered
# chain physics. The fitted R0 absorbs unit/scaling differences.
 
def add_rg_norm(dom):
    """Add Rg_norm column (length-normalized compactness for folded domains)."""
    rg = pd.to_numeric(dom["Rg(Compactness)"], errors="coerce")
    n  = dom["Domain Length"].apply(pd.to_numeric, errors="coerce")
    n_third = n ** (1/3)
 
    # Fit R0 by least squares: Rg = R0 * N^(1/3)  →  R0 = mean(Rg / N^(1/3))
    ratio = rg / n_third
    valid = ratio.dropna()
    if len(valid) == 0:
        dom["Rg_norm"] = np.nan
        return dom, np.nan
    R0 = float(valid.mean())
    dom["Rg_norm"] = rg / (R0 * n_third)
    print(f"  Rg normalization: fitted R0 = {R0:.3f} Å  "
          f"(Rg_norm = Rg / ({R0:.2f} × N^1/3))")
    return dom, R0
 
 
# ---- B helper 2: sequence variability via multiple-sequence alignment ------
# For each domain family we build a pairwise multiple-sequence alignment using
# Biopython's PairwiseAligner, then compute per-column Shannon entropy.
#
# Shannon entropy H at column c:
#   H(c) = -Σ_aa  p(aa, c) * log2(p(aa, c))
#
# where p(aa, c) is the fraction of sequences that have amino acid aa at
# position c (gaps counted separately).
#
# H = 0      → every sequence has the same residue here (fully conserved)
# H = log2(21) ≈ 4.39  → all 20 amino acids + gap equally represented
#                         (maximally variable)
#
# We report two numbers per family:
#   mean_entropy   — average conservation across all aligned positions.
#                    Tells you "how variable is this family overall?"
#   frac_conserved — fraction of alignment columns with H < 1 bit
#                    (a position is "conserved" if ≤ 2 amino acids dominate).
#                    Tells you "how much of the sequence is really invariant?"
#
# A family with high mean_entropy and low frac_conserved has very diverse
# sequences — any property differences between members reflect genuine
# biochemical variation, not sampling noise. A family with low mean_entropy
# has conserved sequences — the properties you measure are likely
# domain-intrinsic rather than driven by outliers.
 
try:
    from Bio import pairwise2
    from Bio.pairwise2 import format_alignment
    _HAS_BIOPYTHON = True
except ImportError:
    _HAS_BIOPYTHON = False
 
try:
    from Bio.Align import PairwiseAligner
    _HAS_PAIRWISE_ALIGNER = True
except ImportError:
    _HAS_PAIRWISE_ALIGNER = False
 
 
def _align_sequences(seqs):
    """
    Build a simple multiple-sequence alignment by progressively aligning
    each sequence to a growing consensus using Biopython's PairwiseAligner.
    Returns a list of equal-length strings (the aligned sequences).
    """
    if not _HAS_BIOPYTHON and not _HAS_PAIRWISE_ALIGNER:
        return None
 
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
 
    if len(seqs) == 1:
        return seqs
 
    # Use the longest sequence as the anchor to minimise ragged ends
    anchor = max(seqs, key=len)
    aligned = [anchor]
    for seq in seqs:
        if seq == anchor:
            continue
        try:
            aln = aligner.align(anchor, seq)[0]
            # Extract aligned versions of both sequences
            a_aln = aln.format("fasta").split("\n")
            # Biopython format gives us the full alignment string
            target = str(aln).split("\n")[0]
            query  = str(aln).split("\n")[2]
            if len(target) == len(query):
                aligned.append(query)
            else:
                # Fallback: pad shorter sequence with gaps
                aligned.append(seq + "-" * (len(anchor) - len(seq)))
        except Exception:
            aligned.append(seq + "-" * max(0, len(anchor) - len(seq)))
    return aligned
 
 
def _shannon_entropy(col):
    """Shannon entropy (bits) for one alignment column (string of chars)."""
    counts = {}
    for c in col:
        counts[c] = counts.get(c, 0) + 1
    total = len(col)
    h = 0.0
    for c_count in counts.values():
        p = c_count / total
        if p > 0:
            h -= p * np.log2(p)
    return h
 
 
def compute_family_variability(dom, min_family_size=10):
    """
    For each domain family, align sequences and return a DataFrame with:
      Domain, n_seqs, mean_entropy, frac_conserved, mean_length, cv_length
    """
    counts = dom["Domain"].value_counts()
    big = counts[counts >= min_family_size].index.tolist()
    rows = []
    for fam in big:
        sub = dom[dom["Domain"] == fam]
        seqs = sub["Domain Sequence"].dropna().tolist()
        lens = [len(s) for s in seqs]
        mean_len = np.mean(lens)
        cv_len   = np.std(lens) / mean_len if mean_len > 0 else np.nan
 
        row = {"Domain": fam, "n_seqs": len(seqs),
               "mean_length": mean_len, "cv_length": cv_len,
               "mean_entropy": np.nan, "frac_conserved": np.nan}
 
        if not (_HAS_BIOPYTHON or _HAS_PAIRWISE_ALIGNER):
            rows.append(row); continue
 
        try:
            aligned = _align_sequences(seqs)
            if aligned is None or len(aligned) < 2:
                rows.append(row); continue
            # Pad all to same length
            max_len = max(len(a) for a in aligned)
            padded  = [a.ljust(max_len, "-") for a in aligned]
            entropies = [_shannon_entropy([s[i] for s in padded])
                         for i in range(max_len)]
            row["mean_entropy"]   = float(np.mean(entropies))
            row["frac_conserved"] = float(np.mean([e < 1.0 for e in entropies]))
        except Exception as exc:
            print(f"  alignment failed for {fam}: {exc}")
        rows.append(row)
    return pd.DataFrame(rows)
 
 
# ---- B helper 3: patterning metrics (kappa, omega, SCD) -------------------
# kappa  — charge patterning (Das & Pappu 2013). Range 0-1.
#           0 = perfectly alternating +/- charges
#           1 = fully segregated (all + together, all - together)
#          Requires localcider. Returns NaN if not installed.
#
# omega  — generalized patterning of charged + Pro residues vs others
#          (Martin et al. 2020). Range 0-1. Same interpretation as kappa.
#          Requires localcider. Returns NaN if not installed.
#
# SCD    — Sequence Charge Decoration (Sawle & Ghosh 2015). No dependency.
#          More negative = more alternating (compact / well-mixed)
#          Less negative / positive = more blocky (expanded)
#          Note: unlike kappa/omega, SCD has a mild length dependence,
#          so within-family comparisons (similar lengths) are reliable;
#          cross-family comparisons should be interpreted cautiously.
 
try:
    from localcider.sequenceParameters import SequenceParameters as _SeqParams
    _HAS_LOCALCIDER = True
except ImportError:
    _HAS_LOCALCIDER = False
 
 
def _kappa(seq):
    if not _HAS_LOCALCIDER or len(seq) < 10:
        return np.nan
    try:
        return _SeqParams(seq).get_kappa()
    except Exception:
        return np.nan
 
 
def _omega(seq):
    if not _HAS_LOCALCIDER or len(seq) < 10:
        return np.nan
    try:
        return _SeqParams(seq).get_Omega()
    except Exception:
        return np.nan
 
 
def _scd(seq):
    """Sequence Charge Decoration — closed-form, no dependency."""
    s = "".join(c for c in seq.upper() if c in VALID_AA)
    n = len(s)
    if n < 2:
        return np.nan
    q = np.array([1 if c in POSITIVE else (-1 if c in NEGATIVE else 0)
                  for c in s], dtype=float)
    idx = np.arange(n)
    sep = np.sqrt(np.abs(idx[:, None] - idx[None, :]))
    qq  = np.outer(q, q)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float((qq * sep)[mask].sum() / n)
 
 
def add_patterning(dom):
    """Append kappa, omega, SCD columns to the domain dataframe."""
    print("  Computing patterning metrics (SCD always; kappa/omega if "
          f"localcider installed: {_HAS_LOCALCIDER}) ...")
    seqs = dom["Domain Sequence"].astype(str)
    dom["kappa"] = seqs.apply(_kappa)
    dom["omega"] = seqs.apply(_omega)
    dom["SCD"]   = seqs.apply(_scd)
    return dom
 
 
# ---- B helper 4: shared violin-plot renderer --------------------------------
 
def _violin_plot(dom, big, prop, prop_label, outdir, suffix,
                 subtitle="", ylabel_note=""):
    """
    One violin-per-family figure for `prop`. Saves to outdir.
    big       — ordered list of family names to include
    suffix    — filename suffix (e.g. 'FCR', 'kappa', 'Rg_norm')
    subtitle  — extra text appended to the figure title
    """
    data, labels = [], []
    valid_all = pd.to_numeric(dom[prop], errors="coerce")
    for fam in big:
        v = pd.to_numeric(dom.loc[dom["Domain"] == fam, prop],
                          errors="coerce").dropna()
        if len(v) >= 3:
            data.append(v.values)
            labels.append(f"{fam}\n(n={len(v)})")
    if not data:
        return
 
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(data) + 2), 5))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.85)
    for pc in parts["bodies"]:
        pc.set_facecolor("#DD8452"); pc.set_alpha(0.45); pc.set_edgecolor("black")
 
    # All-domain IQR reference band
    med = float(valid_all.median()) if valid_all.dropna().any() else np.nan
    q25, q75 = np.nanpercentile(valid_all.dropna(), [25, 75]) \
               if valid_all.dropna().any() else (np.nan, np.nan)
    if not np.isnan(q25):
        ax.axhspan(q25, q75, color="gray", alpha=0.15,
                   label=f"all-domain IQR (median={med:.3f})")
        ax.axhline(med, color="gray", ls="--", lw=1)
 
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(prop_label + (f"\n{ylabel_note}" if ylabel_note else ""))
    title = f"Section B — {prop_label}: by domain family"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title)
    if not np.isnan(q25):
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fname = f"B_byFamily_{suffix.replace('/', '_').replace(' ', '_')}.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
# ===========================================================================
#  Section B: by-domain-family panels
# ===========================================================================
 
def section_B_by_family(dom, outdir, min_family_size=10):
    """
    For each sufficiently large domain family:
      B.1 — Standard biophysical properties (composition + surface + context)
      B.2 — Length-normalized Rg (Rg_norm) — folded-domain globule scaling
      B.3 — Sequence variability via MSA + Shannon entropy
      B.4 — Patterning: kappa, omega (localcider), SCD (always)
    """
    print(f"\n--- Section B: per-domain-family analysis "
          f"(families with >= {min_family_size} members) ---")
    counts = dom["Domain"].value_counts()
    big = counts[counts >= min_family_size].index.tolist()
    print(f"  {len(big)} families, covering "
          f"{counts.loc[big].sum()}/{len(dom)} domains")
 
    # ---- B.1: standard properties -----------------------------------------
    print("  B.1: standard property violins ...")
    props_b1 = COMP_PROPS + SURFACE_PROPS + ["interactionIndex"]
    for prop in props_b1:
        if prop not in dom.columns:
            continue
        if pd.to_numeric(dom[prop], errors="coerce").dropna().empty:
            continue
        _violin_plot(dom, big, prop, prop, outdir, prop,
                     subtitle="gray band = IQR over all domains")
    print(f"    wrote {len(props_b1)} property figures")
 
    # ---- B.2: Rg normalization --------------------------------------------
    print("  B.2: Rg normalization (folded-domain globule scaling) ...")
    if "Rg(Compactness)" in dom.columns:
        dom, R0 = add_rg_norm(dom)
        if not np.isnan(dom["Rg_norm"].dropna().iloc[0] if len(dom["Rg_norm"].dropna()) else np.nan):
            _violin_plot(
                dom, big, "Rg_norm",
                prop_label="Rg_norm  (observed Rg / expected for globule of same length)",
                outdir=outdir, suffix="Rg_norm",
                subtitle=(f"Null model: Rg_expected = {R0:.2f} × N^(1/3)   |   "
                           "Rg_norm > 1 = more expanded than average, < 1 = more compact"),
                ylabel_note="Rg_norm = 1  →  typical globular compactness for this length")
            print(f"    wrote B_byFamily_Rg_norm.png")
    else:
        print("    Rg(Compactness) column not found — skipping B.2")
 
    # ---- B.3: sequence variability ----------------------------------------
    print("  B.3: sequence variability (MSA + Shannon entropy) ...")
    var_df = compute_family_variability(dom, min_family_size)
 
    # Plot: mean_entropy and frac_conserved side-by-side as bars
    var_clean = var_df.dropna(subset=["mean_entropy"]).sort_values(
        "mean_entropy", ascending=False)
    if len(var_clean) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.55 * len(var_clean) + 4), 5))
        x = np.arange(len(var_clean))
 
        axes[0].bar(x, var_clean["mean_entropy"], color="#4C72B0", alpha=0.75)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(
            [f"{r['Domain']}\n(n={int(r['n_seqs'])})"
             for _, r in var_clean.iterrows()],
            rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("Mean Shannon entropy per alignment column (bits)")
        axes[0].set_title("Sequence diversity within each family\n"
                           "Higher = more variable sequences\n"
                           "(max possible ≈ 4.4 bits = all AAs equally represented)")
        axes[0].axhline(var_clean["mean_entropy"].mean(), color="gray",
                        ls="--", lw=1, label="cross-family mean")
        axes[0].legend(fontsize=8)
 
        axes[1].bar(x, var_clean["frac_conserved"], color="#DD8452", alpha=0.75)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            [f"{r['Domain']}\n(n={int(r['n_seqs'])})"
             for _, r in var_clean.iterrows()],
            rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("Fraction of alignment columns with H < 1 bit")
        axes[1].set_title("Fraction of positions that are conserved\n"
                           "Higher = more positions invariant across members\n"
                           "(H < 1 bit: ≤2 amino acids dominate that column)")
        axes[1].axhline(var_clean["frac_conserved"].mean(), color="gray",
                        ls="--", lw=1, label="cross-family mean")
        axes[1].legend(fontsize=8)
 
        fig.suptitle(
            "Section B.3 — Sequence variability by domain family\n"
            "Low entropy + high frac_conserved → properties are domain-intrinsic\n"
            "High entropy + low frac_conserved → properties may reflect outliers",
            fontsize=11, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "B3_sequence_variability.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("    wrote B3_sequence_variability.png")
 
        # Also save the table
        var_df.to_csv(os.path.join(outdir, "B3_sequence_variability.tsv"),
                      sep="\t", index=False)
        print("    wrote B3_sequence_variability.tsv")
    else:
        print("    not enough families with alignment data — skipping B.3 plot")
 
    # ---- B.4: patterning (kappa, omega, SCD) ------------------------------
    print("  B.4: patterning metrics ...")
    dom = add_patterning(dom)
 
    patterning_props = [
        ("kappa", "kappa — charge patterning (0=alternating, 1=blocky)",
         "kappa",
         "Requires localcider" if not _HAS_LOCALCIDER else
         "0 = perfectly alternating +/- charges  |  1 = fully segregated blocks"),
        ("omega", "omega — general patterning: charged+Pro vs others (0–1)",
         "omega",
         "Requires localcider" if not _HAS_LOCALCIDER else
         "0 = well-mixed  |  1 = strongly patterned"),
        ("SCD", "SCD — Sequence Charge Decoration (Sawle & Ghosh 2015)",
         "SCD",
         "More negative = alternating charges  |  less negative/positive = blocky charges\n"
         "Caution: mild length dependence — within-family comparisons most reliable"),
    ]
    for prop, label, suffix, note in patterning_props:
        if pd.to_numeric(dom[prop], errors="coerce").dropna().empty:
            print(f"    {prop}: all NaN (install localcider for kappa/omega) — skipping")
            continue
        _violin_plot(dom, big, prop, label, outdir, suffix,
                     subtitle="gray band = IQR over all domains",
                     ylabel_note=note)
        print(f"    wrote B_byFamily_{suffix}.png")
 
    if not _HAS_LOCALCIDER:
        print("    NOTE: install localcider (pip install localcider) to get "
              "kappa and omega. SCD is always available.")
 
#  ##########################OLD VERSION HERE
# # ===========================================================================
# #  Section B: by-domain-family panels
# # ===========================================================================

# def section_B_by_family(dom, outdir, min_family_size=10):
#     """For each large family, show its distribution of every metric overlaid
#     on the full domain library."""
#     print(f"\n--- Section B: per-domain-family histograms "
#           f"(families with >= {min_family_size} members) ---")
#     counts = dom["Domain"].value_counts()
#     big = counts[counts >= min_family_size].index.tolist()
#     print(f"  {len(big)} families pass threshold (covering "
#           f"{counts.loc[big].sum()}/{len(dom)} domains)")

#     props_to_plot = COMP_PROPS + SURFACE_PROPS + ["interactionIndex"]
#     # We make ONE figure per metric showing all families side-by-side
#     # (cleaner than one figure per family).

#     for prop in props_to_plot:
#         if prop not in dom.columns: continue
#         valid_dom = pd.to_numeric(dom[prop], errors="coerce")
#         if valid_dom.dropna().empty: continue

#         fig, ax = plt.subplots(figsize=(max(8, 0.5*len(big)+2), 5))
#         # Collect arrays
#         data = []
#         labels = []
#         for fam in big:
#             v = pd.to_numeric(dom.loc[dom["Domain"] == fam, prop],
#                                errors="coerce").dropna()
#             if len(v) >= 3:
#                 data.append(v.values); labels.append(f"{fam}\n(n={len(v)})")

#         if not data:
#             plt.close(fig); continue

#         # Violin + scatter for clarity
#         parts = ax.violinplot(data, showmeans=False, showmedians=True,
#                               widths=0.85)
#         for pc in parts['bodies']:
#             pc.set_facecolor("#DD8452"); pc.set_alpha(0.45)
#             pc.set_edgecolor("black")
#         # Background reference: full domain distribution as a horizontal band
#         # (median ± IQR)
#         med = valid_dom.median()
#         q25, q75 = np.nanpercentile(valid_dom, [25, 75])
#         ax.axhspan(q25, q75, color="gray", alpha=0.15,
#                    label=f"all-domain IQR (median={med:.2f})")
#         ax.axhline(med, color="gray", ls="--", lw=1)

#         ax.set_xticks(range(1, len(labels)+1))
#         ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
#         ax.set_ylabel(prop)
#         ax.set_title(f"Section B — {prop}: by domain family\n"
#                      f"(gray band = IQR over all domains; orange = each family)")
#         ax.legend(loc="upper right", fontsize=8)
#         fig.tight_layout()
#         out = os.path.join(outdir, f"B_byFamily_{prop.replace('/', '_')}.png")
#         fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
#     print(f"  wrote {len(props_to_plot)} per-metric figures to {outdir}")


# ===========================================================================
#  Section C: candidate-vs-non-candidate + feature weighting
# ===========================================================================

def section_C_candidates(dom, cand, outdir):
    """Candidate-vs-non-candidate distributions + standardized logistic
    regression coefficients to suggest property weights."""
    print("\n--- Section C: candidate vs Neither — feature importance ---")
    # Merge candidate labels onto the full domain table
    merge_keys = ["Entry", "Start", "End"]
    merged = dom.merge(cand[merge_keys + ["candidateSequence"]],
                        on=merge_keys, how="left")
    merged["candidateSequence"] = merged["candidateSequence"].fillna("Neither")
    merged["is_candidate"] = (merged["candidateSequence"] != "Neither").astype(int)
    n_pos = merged["is_candidate"].sum(); n_neg = len(merged) - n_pos
    print(f"  candidates: {n_pos}, neither: {n_neg}")

    # ---- C.1: histograms candidate vs neither ----
    feat_cols = [c for c in (COMP_PROPS + SURFACE_PROPS + CONTEXT_PROPS
                              + ["Domain Length"])
                 if c in merged.columns]
    n = len(feat_cols); ncols = 5; nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = axes.flatten()
    neither = merged[merged["is_candidate"] == 0]
    cands   = merged[merged["is_candidate"] == 1]
    for ax, prop in zip(axes, feat_cols):
        _hist_panel(ax, neither[prop], cands[prop],
                    "Neither", "candidates", prop,
                    color_a="#888888", color_b="#C44E52")
    for ax in axes[n:]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Section C — candidate (any class) vs Neither\n"
                 "Cohen's d quantifies how much each property separates the "
                 "two groups\n(|d|>0.5 = moderate, >0.8 = large)",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = os.path.join(outdir, "C1_candidate_vs_neither_hists.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")

    # ---- C.1b-d: per-category breakdowns vs Neither ----
    category_specs = [
        ("Aromatic-driven", "#9467BD", "C1b_aromatic_vs_neither_hists.png",
         "Section C — Aromatic-driven only vs Neither"),
        ("Charge-driven",   "#2196F3", "C1c_charge_vs_neither_hists.png",
         "Section C — Charge-driven only vs Neither"),
        ("Both",            "#D62728", "C1d_both_vs_neither_hists.png",
         "Section C — Both (aromatic + charge) vs Neither"),
    ]
    for category, color, fname, title in category_specs:
        subset = merged[merged["candidateSequence"] == category]
        if len(subset) == 0:
            print(f"  no domains labeled '{category}' — skipping {fname}")
            continue
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
        axes = axes.flatten()
        for ax, prop in zip(axes, feat_cols):
            _hist_panel(ax, neither[prop], subset[prop],
                        "Neither", category, prop,
                        color_a="#888888", color_b=color)
        for ax in axes[n:]:
            ax.set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"{title}\nCohen's d quantifies separation from Neither\n"
                     "(|d|>0.5 = moderate, >0.8 = large)",
                     fontsize=12, y=1.0)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        out = os.path.join(outdir, fname)
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out}")

    # ---- C.2: standardized logistic regression for weighting ----
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  sklearn not available — skipping logistic-regression weights")
        return merged

    # Build feature matrix; drop rows with any NaN in chosen features
    candidate_features = [c for c in feat_cols if c not in ("Domain Length",)]
    X = merged[candidate_features].apply(pd.to_numeric, errors="coerce")
    keep = X.notna().all(axis=1)
    X = X[keep].values
    y = merged.loc[keep, "is_candidate"].values
    if y.sum() < 5:
        print("  too few positive candidates after NaN filtering — skipping LR")
        return merged

    Xs = StandardScaler().fit_transform(X)
    # Modest L2 regularization; balanced class weights since classes very
    # imbalanced.
    lr = LogisticRegression(class_weight="balanced", max_iter=2000,
                             C=1.0, solver="lbfgs")
    lr.fit(Xs, y)
    coefs = pd.Series(lr.coef_[0], index=candidate_features) \
              .sort_values(key=lambda s: s.abs(), ascending=False)
    # Convert to a "suggested weight" suite by taking absolute value and
    # normalizing to sum to 1.
    abs_coefs = coefs.abs()
    weights = abs_coefs / abs_coefs.sum()

    # Cohen's d per feature — model-free counterpart to LR coefficients
    cohen = {}
    for c in candidate_features:
        cohen[c] = cohen_d(neither[c], cands[c])
    cohen = pd.Series(cohen)

    # Save table
    summary = pd.DataFrame({
        "logreg_coef_standardized": coefs,
        "abs_coef":                 abs_coefs,
        "suggested_weight":         weights,
        "cohen_d":                  cohen.reindex(coefs.index),
    })
    csv_out = os.path.join(outdir, "C2_feature_weights.tsv")
    summary.to_csv(csv_out, sep="\t")
    print(f"  wrote {csv_out}")

    # Bar plot
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.3*len(coefs))))
    pos = np.arange(len(coefs))
    bar_colors = ["#C44E52" if v > 0 else "#4C72B0" for v in coefs.values]
    axes[0].barh(pos, coefs.values, color=bar_colors)
    axes[0].set_yticks(pos); axes[0].set_yticklabels(coefs.index)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="black", lw=0.5)
    axes[0].set_xlabel("Standardized logistic-regression coefficient\n"
                       "(positive → property pushes a domain toward 'candidate')")
    axes[0].set_title("Per-feature direction & importance")

    axes[1].barh(pos, weights.values, color="#888888")
    axes[1].set_yticks(pos); axes[1].set_yticklabels(weights.index)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Suggested weight (|coef| normalized to sum=1)")
    axes[1].set_title("Weighting suggestion for step 5\n"
                       "(use as a starting point, not gospel)")

    fig.suptitle(
        "Section C.2 — what should the step-5 weights be?\n"
        f"trained on {y.sum()} candidates vs {len(y)-y.sum()} 'Neither' "
        f"domains, balanced-class L2 logistic regression",
        fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(outdir, "C2_feature_weights.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")

    print("\nTOP 5 SUGGESTED WEIGHTS:")
    for name, w in weights.head(5).items():
        d = cohen.get(name, np.nan)
        direction = "↑" if coefs[name] > 0 else "↓"
        print(f"  {name:30s} weight={w:.3f}  direction={direction}  d={d:+.2f}")
    print(
        "\nInterpret: 'direction ↑' means the property is HIGHER in "
        "candidates than in Neither (push it up to favor candidates).\n"
        "          'direction ↓' means the property is LOWER in candidates "
        "(push it down to favor candidates).\n"
        "Compare these directions to your current step-5 logic to spot any "
        "places where your hard cutoffs disagree with the data."
    )
    return merged


# ===========================================================================
#  CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--proteome",   required=True,
                    help="UniProt full proteome TSV with a Sequence column")
    ap.add_argument("--domains",    required=True,
                    help="Stage-4 domain library TSV with all biophysical metrics")
    ap.add_argument("--candidates", default=None,
                    help="Optional stage-5 TSV with candidateSequence column")
    ap.add_argument("--outdir",     default="histograms_output")
    ap.add_argument("--min-family-size", type=int, default=10,
                    help="Minimum members for a domain family to be plotted "
                         "individually (default 10)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    prot, dom, cand = load_inputs(args.proteome, args.domains, args.candidates)

    # Compute composition properties for both groups
    print("\nComputing composition properties for full proteome ...")
    prot = add_seq_properties(prot, "Sequence")
    print("Computing composition properties for domains ...")
    dom = add_seq_properties(dom, "Domain Sequence")

    section_A_proteome_vs_domains(prot, dom, args.outdir)
    section_B_by_family(dom, args.outdir, min_family_size=args.min_family_size)
    if cand is not None:
        section_C_candidates(dom, cand, args.outdir)
    else:
        print("\n(no candidates file provided — skipping Section C)")

    print(f"\nDone. All output in: {args.outdir}")


if __name__ == "__main__":
    main()
