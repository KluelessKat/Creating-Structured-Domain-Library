# Creating a Structured Domain Library

## Table of Contents

1. [Scientific Background](#scientific-background)
2. [Pipeline Overview](#pipeline-overview)
3. [Big-Picture Design Decisions](#big-picture-design-decisions)
4. [Dependencies & Setup](#dependencies--setup)
5. [Core Pipeline Steps](#core-pipeline-steps)
   - [Step 0 — Download the Human Proteome](#step-0--download-the-human-proteome)
   - [Step 1 — Extract Domains](#step-1--extract-domains-1_domaininfoextractorpy)
   - [Step 2 — Filter Disordered Domains](#step-2--filter-disordered-domains-2_disorderedpredictions_3in1py)
   - [Step 2.5 — Tag Length Mismatches](#step-25--tag-length-mismatches-25_taglengthismatchpy)
   - [Step 3 — Quantify Domain Interactions](#step-3--quantify-domain-interactions-3_alphafolddomain-interactionspy)
   - [Step 4 — Compute Physical Properties](#step-4--compute-physical-properties-4_physicalpropertydomainstructpy)
   - [Step 5 — Select Final Candidates](#step-5--select-final-candidates-5_obtainfinalcandidatesequencespy)
   - [Step 6 — PyMOL Imaging](#step-6--pymol-imaging-6_pymolimagespy)
6. [Helper / Analysis Scripts](#helper--analysis-scripts)
   - [audit_uniprot_vs_af.py](#audit_uniprot_vs_afpy)
   - [build_protein_level_tsv.py](#build_protein_level_tsvpy)
   - [tag_phasepro.py](#tag_phasepropy)
   - [train_isBuried_classifier.py](#train_isburied_classifierpy)
   - [compare_domain_libraries.py](#compare_domain_librariespy)
   - [rep_of_candidates_clustering.py](#rep_of_candidates_clusteringpy)
7. [Supporting Modules](#supporting-modules)
   - [structure_source_fixedmapping.py](#structure_source_fixedmappingpy)
   - [structure_source_bioassembly.py](#structure_source_bioassemblypy)
8. [Biophysics Pipeline (`biophysics_pipeline/`)](#biophysics-pipeline-biophysics_pipeline)
   - [histograms_and_weights.py](#histograms_and_weightspy)
   - [surface.py](#surfacepy)
   - [compute_distributions.py / plot_distributions.py](#compute_distributionspy--plot_distributionspy)
9. [End-to-End Commands (Hoffman2 / SLURM)](#end-to-end-commands-hoffman2--slurm)
10. [Output Column Glossary](#output-column-glossary)

---

## Scientific Background

Biological condensates are membrane-less compartments that concentrate biomolecules to regulate transcription, RNA processing, signal transduction, and stress responses. Because their dysregulation is implicated in cancer and neurodegeneration, understanding how protein sequences encode condensation propensity is fundamentally important.

[CondenSeq (Kappel et al., 2025)](https://www.nature.com/articles/s41592-025-02726-y) screened thousands of intrinsically disordered region (IDR) sequences to map sequence–condensation relationships, finding that aromatic residues and high net charge promote condensation. **This project extends that work to structured domains.** Certain structured domains are known to influence condensation; by generating a curated library of structured domains from the entire human proteome and screening it with CondenSeq, we can determine how domains contribute to condensate formation independently of their sequence composition.

The core experimental constraint is **domain length ≤ 66 amino acids** (the maximum insert size tolerated by the CondenSeq construct).

---

## Pipeline Overview

```
UniProt Human Proteome (TSV)
        │
        ▼
[Step 0]  Download & decompress
        │
        ▼
[Step 1]  1_domainInfoExtractor.py
          Extract all annotated domains → 1_domainLibraryRaw.tsv
        │
        ▼
[Step 2]  2_disorderedPredictions_3in1.py
          Disorder scoring (metapredict + AIUPred + IUPred3 + pLDDT)
          Filter to structured domains → 2_domainLibraryStructuredSeq.tsv
        │
        ▼
[Step 2.5] 2.5_tagLengthMismatch.py          ← run audit_uniprot_vs_af.py first (one-time)
           Tag rows where UniProt seq ≠ AF DB v6 sequence
           → 2.5_domainLibraryTagged.tsv
        │
        ▼
[Step 3]  3_alphaFoldDomainInteractions.py
          Fetch AlphaFold PDB + PAE (+ optional experimental PDB via SIFTS)
          Compute anchoringIndex, fractionBuried, contactDensity, interactionIndex
          → 3_domainLibraryInteractions.tsv
        │
        ▼
[Step 4]  4_physicalPropertyDomainStruct.py
          Compute Rg, surfaceFraction, aromaticSurfaceFraction,
          positiveSurfaceFraction, negativeSurfaceFraction
          → 4_domainLibraryPhysicalProperties.tsv
        │
        ▼
[Step 5]  5_obtainFinalCandidateSequences.py
          Percentile-based multi-criteria selection
          → 5_finalCandidateSequences.tsv
        │
        ▼
[Step 6]  6_pymolImages.py
          PyMOL visualization of each candidate in its parent protein
          → tiled PNG image
```

**Optional analysis branches** (not part of the primary filter):

```
[build_protein_level_tsv.py]  → protein-as-domain TSV for full-proteome baseline
[tag_phasepro.py]             → annotate with PhaSePro condensate observations
[train_isBuried_classifier.py]→ logistic regression to predict buriedInside
[histograms_and_weights.py]   → proteome-level vs domain-level metric distributions
[audit_uniprot_vs_af.py]      → identify entries where UniProt and AF DB disagree
```

---

## Big-Picture Design Decisions

### Which structure source: AlphaFold vs experimental PDB?

Most human domains lack experimental structures, so **[AlphaFold DB v6](https://alphafold.ebi.ac.uk)** is the primary structure source. AlphaFold provides a predicted structure for virtually every human protein and, crucially, a **Predicted Aligned Error (PAE)** matrix — the inter-residue confidence estimate required to compute the anchoring index.

When an experimental PDB is available and of sufficient quality, the pipeline can optionally use it for the SASA-based metrics (`fractionBuried`, `contactDensity`) while still using the AF PAE for the anchoring index. This is controlled by `--experimental-mode` in step 3.

### PDB candidate ranking (when experimental mode is active)

The helper `structure_source_fixedmapping.py` queries **[PDBe SIFTS](https://www.ebi.ac.uk/pdbe/pdbe-kb/api/mappings/)** to enumerate all experimental PDB entries that cover a given UniProt domain. Candidates are ranked by the following ordered criteria (higher = better):

1. **Domain coverage ≥ 0.80** — the PDB chain must cover at least 80% of the requested domain residues (configurable via `--min-domain-coverage`).
2. **Highest domain purity** — fraction of the extracted chain that belongs to the target domain (low purity = the domain is a small slice of a large multidomain structure).
3. **Lowest resolution (Å)** — for crystallographic and cryo-EM entries; lower Å = better model.
4. **Method quality** — X-ray / Electron Microscopy (rank 3) > Neutron Diffraction (rank 2) > NMR (rank 1).

When no experimental structure meets the coverage threshold, the pipeline falls back to AlphaFold. A `highPurityFlag` column is set to `True` when the experimental PDB covers nearly the full domain with very few extraneous residues (threshold: purity ≥ 0.70).

### Why not use biological assemblies?

The standard pipeline uses the **asymmetric unit (ASU)** PDB, where residue numbering is reconciled with UniProt via SIFTS. A separate module (`structure_source_bioassembly.py`) downloads **pre-expanded biological assemblies** from RCSB (relevant when the functional form is a homo-dimer, etc.), but it is not wired into the default pipeline because biological assembly PDBs do not carry SIFTS chain IDs, requiring alignment-only chain identification.

### AlphaFold fragment system

Proteins **> 2700 AA** are split by AlphaFold DB into overlapping **1400-AA fragments** with a step of 200 AA:

| Fragment | Global residues (UniProt numbering) |
|---------|--------------------------------------|
| F1 | 1 – 1400 |
| F2 | 201 – 1600 |
| F3 | 401 – 1800 |
| … | … |

Each domain is mapped to the single fragment that fully contains it. A domain that **spans two fragment boundaries is skipped** (it cannot fit in any one AlphaFold model). Fragment PDB files use *local* residue numbering (1 → up to 1400), so global UniProt positions must be converted before using them as PDB residue indices.

Two naming conventions exist in the codebase:

- **Step-3 download convention:** `{UniProt}_F{N}_model.pdb` (e.g. `P12345_F2_model.pdb`)
- **AF-DB original naming:** `AF-{UniProt}-F{N}-model_v6.pdb` (e.g. `AF-P12345-F2-model_v6.pdb`)

Both are recognized by all scripts.

### UniProt ↔ AlphaFold sequence mismatch

UniProt curates canonical sequences more frequently than AlphaFold DB rebuilds. For ~0.7% of human proteins, the canonical sequence changed after the AF DB v6 release. This causes:

- PAE matrix size mismatches (step 3 `anchoringIndex` fails).
- Incorrect domain coordinate mapping (domain at positions 900–960 may actually sit outside the shorter AF model).

The solution is to run `audit_uniprot_vs_af.py` once (one-time, ~5 min for the whole proteome) and then tag affected rows with `2.5_tagLengthMismatch.py`. Tagged rows (`lengthMismatch=True`) are written to the checkpoint in step 3 but skipped for all metric computation.

### Interaction metrics (step 3)

Three complementary metrics quantify how embedded a domain is within its parent protein:

| Metric | What it measures | Calculation |
|--------|-----------------|-------------|
| **anchoringIndex** | How constrained the domain is by surrounding residues on average; reflects positional rigidity, not interaction strength | Mean PAE from domain residues to all non-domain residues (AlphaFold PAE matrix, clipped at ≤ 5 Å = high confidence) |
| **fractionBuried** | Fraction of domain residues that are buried when the full protein is present vs when the domain is isolated | freeSASA on full protein chain vs isolated domain PDB; residue is "buried" if SASA in context < SASA in isolation |
| **contactDensity** | Density of inter-domain contacts (domain ↔ non-domain heavy atom contacts within 5 Å) | Fraction of all possible domain–nondomain residue pairs that are in contact (BioPython / SciPy cdist) |
| **interactionIndex** | Single weighted score combining all three | `0.4 × anchoringIndex + 0.4 × fractionBuried + 0.2 × contactDensity` |

### Disorder prediction (step 2)

Three predictors are always scored; the default filter uses **metapredict + pLDDT**:

| Predictor | Threshold | Source |
|-----------|-----------|--------|
| [metapredict](https://metapredict.net) | mean disorder ≤ 0.5 AND fraction disordered ≤ 0.2 | pip-installable |
| [AIUPred](https://aiupred.elte.hu) | same | local clone required |
| [IUPred3](https://iupred3.elte.hu) | same | local clone required |
| pLDDT (AlphaFold) | mean pLDDT ≥ 80 | AlphaFold EBI API |

A residue is considered disordered if its score > 0.5. Domains that pass the filter are considered structured.

---

## Dependencies & Setup

### Python packages

```bash
pip install pandas numpy scipy biopython freesasa requests metapredict neurosnap joblib scikit-learn matplotlib
```

Optional (for disorder comparisons):
- [AIUPred](https://github.com/doszilab/AIUPred) — clone locally, pass `--aiupred-dir`
- [IUPred3](https://iupred3.elte.hu) — clone locally, pass `--iupred3-dir`

PyMOL is required only for step 6 (imaging):

```bash
conda install -c conda-forge pymol-open-source
```

### AlphaFold DB files

Download the human AF DB v6 PDB files from [AlphaFold DB](https://alphafold.ebi.ac.uk/download) and point `--af-dir` at the directory. The scripts can also download on-demand (slower but functional for small runs).

---

## Core Pipeline Steps

---

### Step 0 — Download the Human Proteome

**Source:** [UniProt Human Proteome](https://www.uniprot.org/proteomes/UP000005640)

**Settings when downloading:**
- Filter: Reviewed (Swiss-Prot) proteins only
- Format: TSV
- Columns to include: `Entry`, `Entry Name`, `Length`, `Sequence`, `Gene Names`, `Domain [FT]`
- Download the compressed file

**Decompress:**

```bash
gunzip -c humanProteome_compressed.tsv.gz > humanProteome_KZ.tsv
```

---

### Step 1 — Extract Domains (`1_domainInfoExtractor.py`)

**Rationale:** UniProt stores domain annotations in a single messy free-text `Domain [FT]` column (e.g. `DOMAIN 45..120 /note="EF-hand 1"; DOMAIN 200..278 /note="EF-hand 2"`). This script parses that string to produce one row per domain.

**Input:**
- `humanProteome_KZ.tsv` — UniProt proteome TSV

**Output:**
- `1_domainLibraryRaw.tsv` — one row per domain, columns: `Entry`, `Gene Name`, `Length`, `Domain`, `Start`, `End`, `Domain Length`, `Domain Sequence`

**Usage:**

```bash
python code/1_domainInfoExtractor.py \
    --input  /path/to/humanProteome_KZ.tsv \
    --output /path/to/1_domainLibraryRaw.tsv
```

**Notes:**
- The length filter (`Domain Length ≤ 66`) is present in the code but commented out — apply it at step 5 instead to retain full distributions for analysis.
- Trailing numbers in domain names are stripped (e.g. `EF-hand 1` → `EF-hand`).

---

### Step 2 — Filter Disordered Domains (`2_disorderedPredictions_3in1.py`)

**Rationale:** UniProt domain annotations are not curated for structural content — many annotated "domains" are actually disordered linkers or low-complexity regions. We score each domain with multiple disorder predictors and retain only those that pass a structuredness threshold. Multiple predictors are used because no single tool is universally best; running all three and using a consensus provides a more robust filter.

Additionally, this step writes a `hardcoded_fragment` column (the AlphaFold fragment number for each domain), which downstream steps (3, 4, 6) rely on to avoid re-deriving fragment assignment from scratch.

**Input:**
- `1_domainLibraryRaw.tsv` (or `2.5_domainLibraryTagged.tsv`)
- AlphaFold EBI API (for pLDDT values, streamed; no disk write)
- Optionally local AIUPred and IUPred3 directories

**Output:**
- Annotated TSV with disorder score columns per predictor, consensus columns, and `hardcoded_fragment`
- Only the passing (structured) rows by default

**New columns added:**
| Column | Description |
|--------|-------------|
| `metapredict_mean_disorder` | Mean per-residue disorder (metapredict) |
| `metapredict_fraction_disordered` | Fraction of residues > 0.5 (metapredict) |
| `aiupred_mean_disorder` / `_fraction_disordered` | Same, AIUPred |
| `iupred3_mean_disorder` / `_fraction_disordered` | Same, IUPred3 |
| `plddt_mean_domain` | Mean pLDDT over domain residues (0–100) |
| `mean_ensemble_mean_disorder` | Mean of all three predictor means |
| `intersection_passes_filter` | 1 iff all predictors AND pLDDT pass |
| `hardcoded_fragment` | AF fragment number for the domain |

**Usage:**

```bash
python code/2_disorderedPredictions_3in1.py \
    --input      1_domainLibraryRaw.tsv \
    --output     2_domainLibraryStructuredSeq.tsv \
    --af-dir     /path/to/af_pdb_files \
    --filter-on  metapredict_and_plddt \
    --filter-mode structured
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--filter-on` | `metapredict_and_plddt` | Which predictor(s) gate the filter |
| `--filter-mode` | `structured` | `structured` keeps low-disorder rows; `disordered` inverts |
| `--aiupred-dir` | *(skip AIUPred)* | Path to local AIUPred clone |
| `--iupred3-dir` | *(skip IUPred3)* | Path to local IUPred3 clone |
| `--af-dir` | *(skip pLDDT)* | Directory with AF PDB files (used to read pLDDT without downloading) |

---

### Step 2.5 — Tag Length Mismatches (`2.5_tagLengthMismatch.py`)

**Rationale:** ~0.7% of human UniProt entries have a canonical sequence that was revised after AlphaFold DB v6 was built. For these entries, the domain coordinate system in our TSV (based on the current UniProt sequence) does not match the AF model (based on an older sequence). Running step 3 on these rows would produce silently wrong metrics. This script tags them so step 3 can skip them cleanly.

**Prerequisite:** Run `audit_uniprot_vs_af.py` first (one-time, ~5 minutes for the full proteome). See [audit_uniprot_vs_af.py](#audit_uniprot_vs_afpy).

**Input:**
- `1_domainLibraryRaw.tsv` or `2_domainLibraryStructuredSeq.tsv` — any step-1/2 output
- `af_uniprot_mismatch_audit.tsv` — audit output

**Output:**
- Same TSV with three additional columns:

| Column | Description |
|--------|-------------|
| `lengthMismatch` | `True` if this entry should be excluded from structure metrics |
| `af_length` | Length that AF DB v6 actually modeled (0 = no AF model) |
| `af_mismatch_reason` | `length_revision` / `no_af_model` / `not_in_audit` / *(empty)* |

**Usage:**

```bash
python code/2.5_tagLengthMismatch.py \
    2_domainLibraryStructuredSeq.tsv \
    af_uniprot_mismatch_audit.tsv \
    2.5_domainLibraryTagged.tsv
```

---

### Step 3 — Quantify Domain Interactions (`3_alphaFoldDomainInteractions.py`)

**Rationale:** The goal of this step is to identify domains that fold and behave independently of their parent protein context. A domain that is deeply buried in the parent or has many contacts with non-domain residues is unlikely to fold correctly when isolated. Three complementary structural metrics are computed (see [Big-Picture Design Decisions](#big-picture-design-decisions)).

This is the most computationally expensive step (~18 hours for the full human proteome). It downloads AlphaFold PDB and PAE files, optionally fetches experimental PDB structures via SIFTS, and computes all interaction metrics.

**Crash-resumable checkpoint:** Every completed row is streamed to `<output>_inProgress.tsv` immediately. If the job crashes, re-running the same command resumes from where it left off (already-completed rows are skipped). On clean completion the checkpoint file is deleted.

**Input:**
- `2.5_domainLibraryTagged.tsv` (or `2_domainLibraryStructuredSeq.tsv`)
- AlphaFold PDB/PAE files (downloaded on demand or from `--af-dir`)
- UniProt FASTA sequences (via `--uniprot-tsv`) for sequence verification
- Optional: experimental PDB structures fetched via SIFTS / RCSB

**Output:**
- `3_domainLibraryInteractions.tsv` — all input columns plus interaction metrics

**New columns added:**
| Column | Description |
|--------|-------------|
| `anchoringIndex` | Mean PAE from domain to non-domain (lower = more anchored) |
| `fractionBuried` | Fraction of domain residues buried in full-protein context |
| `contactDensity` | Fraction of possible domain–nondomain contact pairs within 5 Å |
| `interactionIndex` | Weighted composite (0.4·AI + 0.4·FB + 0.2·CD) |
| `pdbID` | Experimental PDB accession used (if any; blank = AF only) |
| `structureInfo` | JSON with PDB chain, resolution, method, coverage, purity |
| `structureNotes` | Human-readable log of how the structure was resolved |
| `highPurityFlag` | `True` if experimental structure covers domain with purity ≥ 0.70 |
| `loose_align_fragment` | `True` if chain was found by sequence alignment (not SIFTS) |
| `exp_fractionBuried` | `fractionBuried` computed from the experimental PDB (when available) |
| `exp_contactDensity` | `contactDensity` computed from the experimental PDB |
| `exp_interactionIndex` | Weighted score using experimental fb/cd + AF anchoringIndex |
| `hardcoded_fragment` | AF fragment number (filled here if step 2 didn't write it) |
| `lengthMismatch` | Carried through from step 2.5; skipped rows still appear in output |

**Usage:**

```bash
python code/3_alphaFoldDomainInteractions.py \
    --input           2.5_domainLibraryTagged.tsv \
    --output          3_domainLibraryInteractions.tsv \
    --af-dir          /path/to/af_pdb_files \
    --struct-cache-dir /path/to/struct_cache \
    --uniprot-tsv     humanProteome_KZ.tsv \
    --experimental-mode experimental_preferred \
    --min-domain-coverage 0.80
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--experimental-mode` | `alphafold_only` | `alphafold_only`, `experimental_preferred`, or `experimental_only` |
| `--min-domain-coverage` | `0.80` | Minimum fraction of domain that must be covered by an experimental PDB |
| `--struct-cache-dir` | `./struct_cache` | Where to cache downloaded experimental PDB files |
| `--uniprot-tsv` | *(required for seq verification)* | UniProt proteome TSV |

**Running as a SLURM job (Hoffman2 example):**

```bash
#!/bin/bash
#$ -l h_rt=24:00:00
#$ -l h_data=16G
#$ -pe shared 4
#$ -N step3_domains

source /u/local/Modules/default/init/modules.sh
module load python/3.11

python /path/to/code/3_alphaFoldDomainInteractions.py \
    --input            /path/to/2.5_domainLibraryTagged.tsv \
    --output           /path/to/3_domainLibraryInteractions.tsv \
    --af-dir           /path/to/af_pdb_files \
    --struct-cache-dir /path/to/struct_cache \
    --uniprot-tsv      /path/to/humanProteome_KZ.tsv
```

---

### Step 4 — Compute Physical Properties (`4_physicalPropertyDomainStruct.py`)

**Rationale:** Surface residues drive condensation — they are the ones that physically contact RNA, other proteins, or the condensate environment. Step 4 uses [freeSASA](https://freesasa.github.io) to compute the solvent-accessible surface area (SASA) of each domain in **isolation** and classifies surface vs buried residues using an absolute SASA cutoff of ≥ 20 Å².

The radius of gyration (Rg) is also computed to filter out domains that are extremely compact (likely core structural units) or too extended (IDR-like).

When step 3 assigned an experimental PDB (`pdbID` column is set), step 4 preferentially uses the experimental PDB for SASA calculations. Override with `--ignore-pdbid` for full-protein runs.

**Input:**
- `3_domainLibraryInteractions.tsv`
- AlphaFold domain PDB files (excised by step 3, or re-excised on demand)
- Experimental domain PDB files in `--struct-cache-dir/experimental/` (if pdbID is set)

**Output:**
- `4_domainLibraryPhysicalProperties.tsv` — all prior columns plus physical properties

**New columns added:**
| Column | Description |
|--------|-------------|
| `Rg(Compactness)` | Radius of gyration (Å) |
| `surfaceFraction` | Fraction of domain residues with SASA ≥ 20 Å² |
| `aromaticSurfaceFraction` | Fraction of all residues that are aromatic AND surface-exposed |
| `positiveSurfaceFraction` | Fraction that are positively charged AND surface-exposed |
| `negativeSurfaceFraction` | Fraction that are negatively charged AND surface-exposed |
| `structureSource` | `"experimental"` or `"alphafold"` — which PDB was used for SASA |

**Usage:**

```bash
python code/4_physicalPropertyDomainStruct.py \
    --input           3_domainLibraryInteractions.tsv \
    --output          4_domainLibraryPhysicalProperties.tsv \
    --af-dir          /path/to/af_pdb_files \
    --struct-cache-dir /path/to/struct_cache

# For full-protein runs (avoid mixing experimental partial-chain SASA
# with full-protein AF SASA across rows):
python code/4_physicalPropertyDomainStruct.py \
    --input  protein_level.tsv \
    --output protein_level_4.tsv \
    --af-dir /path/to/af_pdb_files \
    --ignore-pdbid
```

---

### Step 5 — Select Final Candidates (`5_obtainFinalCandidateSequences.py`)

**Rationale:** After steps 1–4, we have structural and biophysical metrics for every domain. Step 5 applies percentile-based thresholds to select the subset most likely to drive condensate formation when isolated:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| `interactionIndex` | Bottom 30% | Domain must not rely heavily on parent protein context |
| `aromaticSurfaceFraction` | Top 25% | Aromatic surface drives condensation via π-interactions |
| Net charge surface fraction | Top 25% | Electrostatic interactions also promote condensation |
| `Rg` | Middle 50% | Not too compact (no surface area) or too loose (disordered) |

Domains passing all four filters are labeled `candidateSequence = True`.

**Input:**
- `4_domainLibraryPhysicalProperties.tsv`

**Output:**
- `5_finalCandidateSequences.tsv` — all rows with `candidateSequence` column; scatter plots in a companion PDF

**Usage:**

```bash
python code/5_obtainFinalCandidateSequences.py \
    input.tsv \
    output.tsv
```

---

### Step 6 — PyMOL Imaging (`6_pymolImages.py`)

**Rationale:** Visual QC. Each candidate domain is rendered in the context of its full parent protein using PyMOL, with the domain highlighted in a distinct color. Images are tiled into a single PNG for quick review of whether the flagged domain looks structurally sensible.

**Input:**
- `5_finalCandidateSequences.tsv` (or any TSV with `Entry`, `Start`, `End`, `Domain Length`, `hardcoded_fragment`)
- AlphaFold PDB files (same `--af-dir` as steps 3 and 4)

**Output:**
- Individual PNG images per domain
- A single tiled PNG combining all images

**Usage:**

```bash
python code/6_pymolImages.py \
    --input      5_finalCandidateSequences.tsv \
    --image-dir  /path/to/individual_images/ \
    --output     /path/to/tiled_image.png \
    --af-dir     /path/to/af_pdb_files
```

**Note:** PyMOL must be importable in the Python environment. Use the open-source build (`conda install -c conda-forge pymol-open-source`).

---

## Helper / Analysis Scripts

---

### `audit_uniprot_vs_af.py`

**Rationale:** Before tagging mismatch rows (step 2.5), we need to know which entries disagree. This one-time audit queries the [AlphaFold EBI API](https://alphafold.ebi.ac.uk/api/) for every UniProt entry in the proteome and compares the modeled sequence length to the TSV's `Length` column.

**Input:** UniProt proteome TSV, any step-1/2 output with an `Entry` column  
**Output:** `af_uniprot_mismatch_audit.tsv` with columns `Entry`, `tsv_length`, `af_length`, `mismatch` (`no` / `YES` / `no_af_model`), `af_seq_version_date`, `error`

**Usage:**

```bash
python code/audit_uniprot_vs_af.py \
    humanProteome_KZ.tsv \
    af_uniprot_mismatch_audit.tsv \
    --workers 8
```

Runs with 8 concurrent HTTP workers; the full human proteome (~20K entries) completes in ~5 minutes. **Resumable** — re-running skips already-audited entries.

---

### `build_protein_level_tsv.py`

**Rationale:** For baseline comparison, it is useful to know the biophysical properties of *full proteins* (not just their domains). This script converts the UniProt proteome TSV (one row per protein) into the same format that steps 3 and 4 expect (one row per "domain"), treating each protein as its own single domain (`Domain=full_protein`, `Start=1`, `End=Length`).

The output can then be fed through steps 3 and 4 (with `--ignore-pdbid` in step 4) to compute interaction and surface metrics for full proteins, enabling proteome-vs-domain comparisons in `histograms_and_weights.py`.

By default, proteins > 2700 AA are dropped (they would be fragmented by AlphaFold and cannot be treated as a single domain).

**Input:** `humanProteome_KZ.tsv`  
**Output:** `protein_level_domains.tsv`

**Usage:**

```bash
python code/build_protein_level_tsv.py \
    humanProteome_KZ.tsv \
    protein_level_domains.tsv \
    --max-length 2700
```

---

### `tag_phasepro.py`

**Rationale:** [PhaSePro](https://phasepro.elte.hu) is a curated database of proteins with experimentally observed condensate behavior. Annotating the domain library with PhaSePro observations allows downstream analysis to ask whether domains from known condensate-forming proteins have different structural properties.

Critically, absent entries are stored as **NaN, not 0**. PhaSePro is curated, not screened — an absent entry means "no one has reported this protein in a condensate", not "tested and found negative". Collapsing to 0 would erase that distinction.

**Input:** Any domain TSV with an `Entry` column; `phasepro_human-data.csv` downloaded from [PhaSePro](https://phasepro.elte.hu)  
**Output:** Same TSV with three new columns:

| Column | Description |
|--------|-------------|
| `experimental` | # biomolecular condensate observations (NaN if not in DB) |
| `synthetic` | # synthetic/engineered condensate observations (NaN if not in DB) |
| `phasepro_in_db` | `True` iff the entry appears in PhaSePro at all |

**Usage:**

```bash
python code/tag_phasepro.py \
    4_domainLibraryPhysicalProperties.tsv \
    phasepro_human-data.csv \
    4_domainLibraryPhysicalProperties_phasepro.tsv \
    --species "Homo sapiens"
```

Pass `--species ''` to include all species in the lookup.

---

### `train_isBuried_classifier.py`

**Rationale:** Step 5's percentile-based `interactionIndex` threshold is a blunt instrument. A logistic regression model trained on a manually curated truth set of buried vs exposed domains can learn the optimal decision boundary across all three interaction metrics simultaneously, potentially with better recall. The trained model can then score the entire proteome library.

**Features:** `anchoringIndex`, `fractionBuried`, `contactDensity`  
**Target:** `buriedInside` (1 = buried, 0 = not buried, manually curated in truth set)

**Pipeline:** `StandardScaler → LogisticRegression(L2, class_weight="balanced")` inside an sklearn `Pipeline` (standardization is inside the pipeline to prevent data leakage).

**Hyperparameter search:** 5-fold StratifiedKFold GridSearchCV over `C ∈ {0.01, 0.1, 1, 3, 10, 30, 100}`, scoring by ROC-AUC.

**Diagnostic output (8 plots):** Feature distributions, pairwise scatter colored by class, correlation heatmap, ROC curve, Precision-Recall curve, calibration curve, confusion matrices at two thresholds, decision-threshold sweep, CV-fold boxplot, logistic regression coefficients, permutation feature importance.

**Input:**
- Truth-set TSV that has been run through step 3 (to have `anchoringIndex`, `fractionBuried`, `contactDensity`) and has a `buriedInside` label column

**Output:**
- `isBuried_logreg.joblib` — trained pipeline (joblib bundle: model + feature list + best C + threshold dict)
- `isBuried_training_report/` — directory of diagnostic PNGs + metrics JSON

**Usage:**

```bash
# Train
python code/train_isBuried_classifier.py \
    --truth-tsv   truthset_with_metrics.tsv \
    --model-path  isBuried_logreg.joblib \
    --report-dir  isBuried_training_report

# Score a new library
python code/train_isBuried_classifier.py \
    --truth-tsv  truthset_with_metrics.tsv \
    --model-path isBuried_logreg.joblib \
    --predict-on 3_domainLibraryInteractions.tsv
```

`--predict-on` adds `isBuried_prob`, `isBuried_pred_t050`, and `isBuried_pred_tBest` columns to a new file (`<stem>_isBuried_pred.tsv`).

---

### `compare_domain_libraries.py`

**Rationale:** When comparing two runs of the pipeline (e.g. metapredict vs AIUPred filter, or different pLDDT thresholds), this script produces a side-by-side diff showing which domains are unique to each run, which are shared, and how metric values changed.

See [`compare_domain_libraries_DOCS.md`](code/compare_domain_libraries_DOCS.md) for full documentation.

---

### `rep_of_candidates_clustering.py`

**Rationale:** After selecting final candidates, cluster them by sequence identity or structural metric space to ensure the library is diverse rather than redundant (i.e. not 30 EF-hand variants with identical biophysics).

---

## Supporting Modules

---

### `structure_source_fixedmapping.py`

The core structure-resolution module used by steps 3 and 4. Not run directly — imported by the step scripts.

**What it does:**

1. Queries **[PDBe SIFTS](https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot})** to enumerate experimental PDB entries covering the requested UniProt domain.
2. Ranks candidates by coverage → purity → resolution → method (see [PDB candidate ranking](#pdb-candidate-ranking-when-experimental-mode-is-active)).
3. Downloads the winning PDB from [RCSB](https://files.rcsb.org/download/) and renumbers it to UniProt residue numbering using the SIFTS UniProt→PDB residue mapping.
4. Falls back to the AlphaFold model (fetched from [AlphaFold EBI](https://alphafold.ebi.ac.uk/files/)) if no experimental structure meets the coverage threshold.
5. Caches all downloaded files in `--struct-cache-dir/experimental/` and `--struct-cache-dir/alphafold/`.

**Public entry point:**

```python
from structure_source_fixedmapping import resolve_domain_structure

choice = resolve_domain_structure(
    uniprot_id="P12345",
    domain_start=45,
    domain_end=120,
    cache_dir=Path("./struct_cache"),
    mode="experimental_preferred",   # or alphafold_only / experimental_only
    min_domain_coverage=0.80,
)
# choice.source    → "experimental" or "alphafold"
# choice.pdb_path  → path to the PDB file (renumbered to UniProt coords)
# choice.pdb_id    → e.g. "1CLL"
# choice.chain     → e.g. "A"
```

**APIs used:**
- `https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot}` — SIFTS best structures
- `https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}` — SIFTS residue mapping
- `https://files.rcsb.org/download/{pdb_id}.pdb` — PDB coordinate file
- `https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F{N}-model_v6.pdb` — AF model
- `https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F{N}-predicted_aligned_error_v6.json` — AF PAE

---

### `structure_source_bioassembly.py`

An alternative structure resolution module that downloads **pre-expanded biological assembly** PDB files from RCSB (`{PDB_ID}-assembly{N}.pdb.gz`) instead of the asymmetric unit. Useful when the functionally relevant form is a homo-dimer or other oligomer.

**Key difference from `structure_source_fixedmapping.py`:** SIFTS chain IDs do not apply to biological assembly PDBs (they contain symmetry-expanded copies with modified chain IDs). Chain identification is done by **sequence alignment only**.

**Size guard:** Assemblies with > 24 chains or > 20,000 residues are skipped to prevent memory issues.

**Not wired into the default pipeline** — use directly if you need assembly-level analysis:

```python
from structure_source_bioassembly import resolve_bioassembly_structure
```

---

## Biophysics Pipeline (`biophysics_pipeline/`)

---

### `histograms_and_weights.py`

**Rationale:** After running the full pipeline, this script produces distribution plots comparing the full human proteome against the structured domain library, and optionally against step-5 candidates. It helps you understand which biophysical properties distinguish your library from the background proteome, and quantifies which features drive candidate selection (Section C).

**Sections:**

| Section | What it shows | Requires |
|---------|---------------|----------|
| **A** | Full proteome vs all structured domains: distributions of fractionBuried, contactDensity, interactionIndex, surfaceFraction, aromaticFraction, etc. Dual AF vs experimental distributions when both are available | steps 3–4 output |
| **A2** | AlphaFold vs experimental metric comparison (paired per domain) | step 3 with `--experimental-mode experimental_preferred` |
| **B** | Per-domain-family breakdown (EF-hand, EGF-like, etc.) | step 4 output |
| **C** | Candidate vs non-candidate analysis; logistic-regression feature importance suggesting step-5 weights | step 5 output |

**`_best` columns:** For each interaction metric, a `{col}_best` column is computed: uses `exp_{col}` when available (experimental PDB was used), falls back to the AlphaFold value. These best-available values are what appear in Section A plots.

**Input:**
- `--proteome` — UniProt proteome TSV **or** a step-3/4 full-protein TSV (the script auto-detects `Sequence` vs `Domain Sequence`)
- `--domains` — step-4 output TSV
- `--candidates` — step-5 output TSV *(optional; enables Section C)*

**Usage:**

```bash
python biophysics_pipeline/histograms_and_weights.py \
    --proteome   humanProteome_KZ.tsv \
    --domains    4_domainLibraryPhysicalProperties.tsv \
    --candidates 5_finalCandidateSequences.tsv \
    --outdir     histograms_output/
```

---

### `surface.py`

**Rationale:** Standalone surface analysis script. Computes **relative** SASA (normalized by the theoretical maximum SASA of each residue type) for a domain PDB. This differs from step 4's `calcSASAMetrics`, which uses an **absolute** 20 Å² cutoff. `surface.py` is used in `biophysics_pipeline/` when you want residue-type-normalized surface fractions.

---

### `compute_distributions.py` / `plot_distributions.py`

Supporting scripts for `histograms_and_weights.py`. `compute_distributions.py` pre-computes sequence-based property distributions from the proteome TSV; `plot_distributions.py` renders them. Usually invoked indirectly by `histograms_and_weights.py`.

---

## End-to-End Commands (Hoffman2 / SLURM)

The commands below are for the full human proteome run. Adjust paths to your Hoffman2 directory structure.

```bash
# Set base paths
KAT=/u/home/k/kathyzyx/project-kappel/kathyzyx/scripts/kat_output
CODE=/u/home/k/kathyzyx/project-kappel/kathyzyx/scripts/Creating-Structured-Domain-Library/code
AF_DIR=/path/to/af_pdb_files
STRUCT_CACHE=/path/to/struct_cache
PROT=$KAT/humanProteome_KZ.tsv

# Step 0: decompress (run once)
gunzip -c humanProteome_compressed.tsv.gz > $PROT

# Step 1: extract domains
python $CODE/1_domainInfoExtractor.py \
    --input $PROT \
    --output $KAT/1_domainLibraryRaw.tsv

# One-time audit (run before step 2.5)
python $CODE/audit_uniprot_vs_af.py \
    $PROT \
    $KAT/af_uniprot_mismatch_audit.tsv \
    --workers 8

# Step 2: disorder filter
python $CODE/2_disorderedPredictions_3in1.py \
    --input      $KAT/1_domainLibraryRaw.tsv \
    --output     $KAT/2_domainLibraryStructuredSeq.tsv \
    --af-dir     $AF_DIR \
    --filter-on  metapredict_and_plddt \
    --filter-mode structured

# Step 2.5: tag mismatches
python $CODE/2.5_tagLengthMismatch.py \
    $KAT/2_domainLibraryStructuredSeq.tsv \
    $KAT/af_uniprot_mismatch_audit.tsv \
    $KAT/2.5_domainLibraryTagged.tsv

# Step 3 (submit as SLURM/SGE job — takes ~18 hrs):
qsub step3_job.sh   # see example job script above

# Step 4:
python $CODE/4_physicalPropertyDomainStruct.py \
    --input           $KAT/3_domainLibraryInteractions.tsv \
    --output          $KAT/4_domainLibraryPhysicalProperties.tsv \
    --af-dir          $AF_DIR \
    --struct-cache-dir $STRUCT_CACHE

# Step 5:
python $CODE/5_obtainFinalCandidateSequences.py \
    $KAT/4_domainLibraryPhysicalProperties.tsv \
    $KAT/5_finalCandidateSequences.tsv

# Step 6 (imaging — needs PyMOL):
python $CODE/6_pymolImages.py \
    --input    $KAT/5_finalCandidateSequences.tsv \
    --image-dir $KAT/images/ \
    --output   $KAT/tiled_candidates.png \
    --af-dir   $AF_DIR
```

**Full-protein baseline (optional):**

```bash
python $CODE/build_protein_level_tsv.py $PROT $KAT/protein_level.tsv
# Run step 3 on protein_level.tsv, then:
python $CODE/4_physicalPropertyDomainStruct.py \
    --input            $KAT/protein_level_step3.tsv \
    --output           $KAT/protein_level_step4.tsv \
    --af-dir           $AF_DIR \
    --ignore-pdbid        # <-- force AF SASA for consistency
```

---

## Output Column Glossary

| Column | Step added | Description |
|--------|-----------|-------------|
| `Entry` | 1 | UniProt accession |
| `Gene Name` | 1 | Gene name(s) |
| `Length` | 1 | Full protein length (AA) |
| `Domain` | 1 | Domain name (from UniProt `Domain [FT]`) |
| `Start` / `End` | 1 | UniProt global residue positions (1-based) |
| `Domain Length` | 1 | `End - Start + 1` |
| `Domain Sequence` | 1 | Amino acid sequence of the domain |
| `hardcoded_fragment` | 2 | AlphaFold fragment number (1 = no fragmentation) |
| `metapredict_mean_disorder` | 2 | Mean per-residue disorder score (metapredict) |
| `plddt_mean_domain` | 2 | Mean AlphaFold pLDDT over domain residues |
| `intersection_passes_filter` | 2 | 1 iff all predictors AND pLDDT pass thresholds |
| `lengthMismatch` | 2.5 | True if UniProt sequence ≠ AF DB v6 sequence |
| `af_length` | 2.5 | Length that AF DB actually modeled |
| `af_mismatch_reason` | 2.5 | Why the mismatch was flagged |
| `anchoringIndex` | 3 | Mean PAE domain→non-domain (lower = more constrained) |
| `fractionBuried` | 3 | Fraction of domain residues buried in parent context |
| `contactDensity` | 3 | Fraction of possible domain–nondomain contact pairs within 5 Å |
| `interactionIndex` | 3 | 0.4·AI + 0.4·FB + 0.2·CD |
| `pdbID` | 3 | Experimental PDB accession (blank = AF only) |
| `structureInfo` | 3 | JSON: chain, resolution, method, coverage, purity |
| `structureNotes` | 3 | Human-readable resolution log |
| `highPurityFlag` | 3 | True if experimental PDB purity ≥ 0.70 |
| `exp_fractionBuried` | 3 | fractionBuried from experimental PDB |
| `exp_contactDensity` | 3 | contactDensity from experimental PDB |
| `exp_interactionIndex` | 3 | interactionIndex using experimental fb/cd + AF AI |
| `Rg(Compactness)` | 4 | Radius of gyration (Å) |
| `surfaceFraction` | 4 | Fraction of domain residues with SASA ≥ 20 Å² |
| `aromaticSurfaceFraction` | 4 | Fraction of all residues aromatic AND surface-exposed |
| `positiveSurfaceFraction` | 4 | Fraction positively charged AND surface-exposed |
| `negativeSurfaceFraction` | 4 | Fraction negatively charged AND surface-exposed |
| `structureSource` | 4 | `"experimental"` or `"alphafold"` (which PDB SASA used) |
| `candidateSequence` | 5 | True if domain passed all step-5 filters |
| `experimental` | tag_phasepro | # PhaSePro biomolecular condensate observations (NaN if absent) |
| `synthetic` | tag_phasepro | # PhaSePro synthetic condensate observations |
| `phasepro_in_db` | tag_phasepro | True iff entry appears in PhaSePro |
| `isBuried_prob` | train_isBuried | Logistic regression P(buriedInside=1) |
| `isBuried_pred_t050` | train_isBuried | Predicted class at threshold 0.50 |
| `isBuried_pred_tBest` | train_isBuried | Predicted class at CV-optimized threshold |
