# Biophysical Property Distribution Pipeline

A pipeline for computing and visualizing the distributions of biophysical
properties across (a) the full human proteome and (b) your structured
domain library, to support CondenSeq candidate selection.

Inspired by the analyses in Kappel et al. (Nature Methods 2025, CondenSeq)
and Lotthammer et al. (Nature Methods 2024, ALBATROSS).

## What it does

For every full protein and every structured domain, computes:

**Composition** — FCR, NCPR, |NCPR|, fraction R/K/D/E, fraction aromatic,
fraction proline, fraction glycine, fraction hydrophobic (aliphatic),
fraction polar, mean Kyte-Doolittle hydropathy.

**Patterning** — kappa (Das & Pappu charge patterning), omega (general
patterning), SCD (sequence charge decoration, Sawle & Ghosh),
SHD (sequence hydropathy decoration, Zheng et al.).

**Surface (using your local AlphaFold PDBs)** — surface FCR, surface NCPR,
|surface NCPR|, surface fraction aromatic / positive / negative.
For domains, computed *in context of the full protein* (i.e. the surface
the domain actually presents in the cell, accounting for intra-protein
burial).

**Disorder (full proteins only)** — fraction disordered residues and mean
disorder score from metapredict V2-FF.

Then produces:

1. Paired histograms (full proteome vs domain library) for every metric
2. 2D density maps for the diagrams-of-states most relevant to condensate
   formation
3. Spearman correlation heatmap (hierarchically clustered) showing which
   metrics are redundant — your guide for filtering weights
4. PCA biplot of the domain library
5. Per-domain percentile lookup TSV

## Install

```bash
pip install pandas numpy scipy scikit-learn matplotlib biopython freesasa
pip install localcider          # for kappa, omega
pip install metapredict         # for disorder fraction
```

`localcider` and `metapredict` are optional but recommended — without them,
kappa/omega/disorder come back as NaN and those panels stay blank, but
everything else runs.

## Run

### Step 1: compute property tables

```bash
python compute_distributions.py \
    --proteome /path/to/humanProteome_KZ.tsv \
    --domains  /path/to/1_domainLibraryRaw.tsv \
    --pdb-dir  /path/to/alphaFold/dbFiles \
    --outdir   ./output
```

Omit `--pdb-dir` to skip surface metrics (much faster — useful for a first
pass).

For a debug run on a small slice:

```bash
python compute_distributions.py \
    --proteome humanProteome_KZ.tsv --domains 1_domainLibraryRaw.tsv \
    --outdir output --max-proteome 100 --max-domains 100
```

### Step 2: generate figures

```bash
python plot_distributions.py \
    --full-tsv   output/properties_full_proteins.tsv \
    --domain-tsv output/properties_structured_domains.tsv \
    --outdir     output/figures \
    --pca-color  FCR              # property to color the PCA scatter by
```

## Outputs

```
output/
├── properties_full_proteins.tsv      # one row per protein, all metrics
├── properties_structured_domains.tsv # one row per domain, all metrics
├── property_summary.tsv              # per-metric stats + Cohen's d
└── figures/
    ├── 01_paired_histograms.png
    ├── 02_diagram_of_states.png
    ├── 03_correlation_heatmap.png
    ├── 04_pca.png
    └── 05_domain_percentiles.tsv
```

## How to read the figures for filtering decisions

**`01_paired_histograms.png`** shows where your domain library's
distributions sit relative to the proteome-wide background. The KS D
statistic on each panel quantifies the gap. Properties where domains
look very different from full proteins are properties for which "domain
candidates" already represent a biased sample of human sequences — keep
that bias in mind when you generalize.

**`02_diagram_of_states.png`** is the most directly actionable plot.
The left panel (NCPR vs FCR) is the Das-Pappu landscape; the right
panel (surface NCPR vs surface fraction aromatic) is the closest 2D
analogue of what Kappel et al. (Figs. 3e, 4d) identified as the dominant
drivers of nuclear condensate formation. Pick candidates that span this
2D space.

**`03_correlation_heatmap.png`** answers your weighting question
directly. Properties in the same cluster (high |ρ|) carry redundant
information — pick **one** representative from each cluster as a filter
axis. Properties from different clusters are independently informative
and should each get their own filter axis.

**`04_pca.png`** shows the natural axes of variation in your domain
library. The PCA arrows tell you which property combinations capture
the most variance. For a diverse library, sample candidates that span
PC1 and PC2 broadly rather than clustering in one corner.

**`05_domain_percentiles.tsv`** — for any domain you're considering, this
gives its rank within the library on every metric. Example workflow:
sort by `surface_frac_aromatic_pct` descending, then within the top
10% sort by `NCPR_pct` to find domains that are simultaneously
high-aromatic AND extreme-charged — the "interesting corner" candidates.

## Project file map

```
properties.py            sequence-based metrics (composition + patterning)
surface.py               surface SASA metrics from AlphaFold PDBs
disorder.py              metapredict wrapper
compute_distributions.py driver — reads TSVs, writes TSVs
plot_distributions.py    visualization
test_smoke.py            quick correctness check
```

## Caveats

- **Length-dependent metrics**: SCD and SHD both have a built-in length
  dependence — they scale with sequence length even for chemically
  identical sequences. This means **direct full-protein vs domain
  comparisons of SCD and SHD are not meaningful** (full proteins are
  much longer than your ≤66 aa domains, so SHD/SCD will look very
  different even if chemistry is identical). The fix is one of:
    - Compare SCD/SHD only *within* the domain library (where lengths
      are similar), not across the two distributions, OR
    - Run the full proteome SCD/SHD on length-matched fragments rather
      than whole proteins.
  Composition metrics (FCR, NCPR, fraction-of-X) are length-invariant
  by construction and can be compared freely. kappa and omega are
  length-invariant by design too.
- kappa requires both + and - residues and length ≥ ~10. Short or
  monosign domains return NaN for kappa.
- Surface metrics use the AlphaFold predicted structure as ground truth.
  Low-pLDDT regions will give noisy SASA — use your existing pLDDT
  filter (≥ 80) before trusting domain surface metrics.
- "Surface" is defined as relative SASA ≥ 0.20 (a common cutoff). You
  can change this in `surface.py` if you want a different definition.
- The full-proteome run with surface metrics is slow (~1–3 sec per
  protein × 20k proteins ≈ a few hours). The domain run with surface
  metrics is fast (≤ 30 min for 6.7k domains because PDBs are already
  cached). For a first pass, run without `--pdb-dir` to inspect bulk
  metrics in minutes.
