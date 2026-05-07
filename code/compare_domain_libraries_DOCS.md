# `compare_domain_libraries.py` — Documentation

A command-line tool for comparing TSV outputs from your protein-domain library
pipeline across different runs (e.g. different disorder-prediction algorithms
or parameter choices) and emitting an Excel workbook that highlights what
changed.

---

## Contents

1. [Quick start](#quick-start)
2. [Requirements](#requirements)
3. [Conceptual overview](#conceptual-overview)
4. [How a single comparison works](#how-a-single-comparison-works)
5. [Pair mode](#pair-mode)
6. [Batch mode](#batch-mode)
7. [Output structure](#output-structure)
8. [Adapting the script to new pipeline steps](#adapting-the-script-to-new-pipeline-steps)
9. [Tunable parameters inside the script](#tunable-parameters-inside-the-script)
10. [Troubleshooting / caveats](#troubleshooting--caveats)

---

## Quick start

```bash
# Compare two TSV files (one comparison, one workbook)
python compare_domain_libraries.py pair OLD.tsv NEW.tsv \
    --label1 old --label2 new -o diff.xlsx

# Compare every matching step file across two directories (one workbook,
# many steps)
python compare_domain_libraries.py batch runs/old/ runs/new/ \
    --label1 old --label2 new -o pipeline_diff.xlsx
```

## Requirements

- Python 3.9+
- `pandas`
- `openpyxl`

```bash
pip install pandas openpyxl
```

---

## Conceptual overview

The script answers two kinds of question:

1. *"I changed the disorder-prediction algorithm. Which domain calls stayed
   the same, which shifted, which got added, and which got dropped?"* —
   answered in **pair mode**, one TSV vs one TSV.

2. *"I changed something upstream and want to see how those changes ripple
   through every stage of my pipeline."* — answered in **batch mode**, which
   finds matching step files in two directories and runs a pair comparison on
   each.

Either way, the unit of comparison is a **row** in the TSV. For your pipeline
a row is a single domain annotation inside a single gene, identified by
`(Entry, Domain, Start, End)`. The output groups every row into one of seven
statuses (see [Row statuses](#row-statuses)).

---

## How a single comparison works

This is the core of the script — everything in pair and batch mode eventually
calls `compare_pair()`, which runs the following four stages.

### Stage 1 — Schema detection (`detect_schema`)

Given the two input DataFrames, the script automatically classifies every
column it sees into one of five categories:

| Category | Meaning | How it's detected |
|---|---|---|
| **Identity** | Used to label rows in the output | Columns in `IDENTITY_CANDIDATES` (`Entry`, `Gene Name`, `Domain`) that exist in **both** files |
| **Position** | Used for pairing rows and detecting boundary shifts | Columns in `POSITION_CANDIDATES` (`Start`, `End`) that exist in both files |
| **Numeric** | Gets `file1__X / file2__X / delta__X` triplet in output | Shared columns with a numeric dtype in **both** files |
| **Text** | Gets `file1__X / file2__X` side-by-side (no delta) | Shared columns that aren't numeric in both files (e.g. `Domain Sequence`) |
| **File-unique** | Reported in the Summary, but not compared | Columns that exist in only one of the two files |

Critically, the script **does not hard-code your column names** beyond the
identity and position candidates. If you add a new metric column like
`aromaticSurfaceFraction`, and it's numeric, it automatically gets compared
and picks up a `delta__` column. This is what makes the script work across
all your pipeline stages (2, 3, 4, 5), each of which has a different schema.

The script also picks a **pairing key** here: `(Entry, Domain)` if both
columns exist, otherwise just `(Entry)`. This determines how rows get grouped
before pairing.

> **Requirement:** both files must contain an `Entry` column. Without it the
> script exits with an error.

### Stage 2 — Pairing rows (`build_diff` → `pair_group`)

Rows are grouped by the pairing key detected in Stage 1. Then, within each
group, the script performs a **greedy nearest-neighbor pairing by Start
position**.

The reason this matters: a gene can contain many copies of the same domain
type — e.g. `NOTCH2` has dozens of `EGF-like` repeats. If you matched just on
`(Entry, Domain)`, you'd have a many-to-many collision. Instead, within each
`(Entry, Domain)` group the script:

1. Computes `|Start₁ − Start₂|` for every possible pair of rows across the
   two files.
2. Sorts those candidate pairs by absolute distance.
3. Walks the sorted list and locks in pairs greedily, skipping any whose
   row index has already been used.
4. Whatever remains unpaired on either side becomes an `only_in_*` row.

If neither file has a `Start` column (unusual but possible for custom
pipeline outputs), rows are paired in the order they appear.

### Stage 3 — Classifying each row

Every row in the output gets one of seven statuses:

| Status | Meaning |
|---|---|
| `identical` | Paired row, every compared value equal within `FLOAT_TOL` |
| `changed_boundaries` | Paired row where `Start` and/or `End` differ (other columns may differ too) |
| `changed_values` | Paired row where `Start`/`End` match but at least one other value differs |
| `only_in_<label1>` | Row exists only in file 1 — but the gene also appears in file 2 |
| `only_in_<label2>` | Row exists only in file 2 — but the gene also appears in file 1 |
| `gene_only_in_<label1>` | Row exists only in file 1, and **the entire gene** is absent from file 2 |
| `gene_only_in_<label2>` | Row exists only in file 2, and **the entire gene** is absent from file 1 |

The gene-level vs row-level distinction is useful because during triage you
often want to know *"which genes got dropped entirely"* separately from
*"which individual domains got dropped within genes that still survived"*.

Float comparisons use `FLOAT_TOL = 1e-9` by default, which is strict enough
that any genuine algorithm-induced change will register as different. Raise
this if you want small numerical drift treated as "identical."

### Stage 4 — Assembling the diff DataFrame

For each classified row the script emits one output record with:

- `status` (one of the seven above)
- All identity columns (copied from whichever side has the row, or file1 if both)
- For every **numeric** column: three columns
  `<label1>__<col>`, `<label2>__<col>`, `delta__<col>` where `delta = file2 − file1`
- For every **text** column: two columns `<label1>__<col>`, `<label2>__<col>`

Rows are sorted so `identical` appears first, followed by `changed_boundaries`,
`changed_values`, the `gene_only_*` categories, and finally the `only_in_*`
categories. Within each status, rows are sorted by identity columns.

### Stage 5 — Per-numeric-column summary stats (`compute_numeric_stats`)

Across all paired rows (any status that involves a pair), the script computes:

- `n_paired` — number of paired rows with non-null deltas
- `mean_delta` — mean of `file2 − file1`
- `median_delta` — median
- `max_abs_delta` — the biggest absolute shift seen
- `n_unchanged` — rows where the absolute delta is within `FLOAT_TOL`

This appears in the Summary sheet and is the fastest way to answer *"which
metric moved the most between the two runs?"*

---

## Pair mode

```bash
python compare_domain_libraries.py pair FILE1.tsv FILE2.tsv \
    [--label1 LABEL1] [--label2 LABEL2] [--step-name STEP_NAME] \
    [-o OUT.xlsx]
```

| Argument | Default | Purpose |
|---|---|---|
| `FILE1`, `FILE2` | — | The two TSV files to compare |
| `--label1`, `--label2` | filename stems | Short strings used as column prefixes (`old__Start`, `meta__Start`). Shorter is better — keep them to a handful of characters |
| `--step-name` | inferred from filename | Shown in the Summary sheet header |
| `-o` | `diff.xlsx` | Output workbook path |

If the two inferred labels collide (e.g. both files have the same stem),
`_1` and `_2` are appended.

---

## Batch mode

```bash
python compare_domain_libraries.py batch DIR1 DIR2 \
    [--label1 LABEL1] [--label2 LABEL2] [--step-pattern REGEX] \
    [-o OUT.xlsx]
```

Batch mode lets you run every applicable pairwise comparison in one command.
Instead of taking two files, it takes two directories and finds matching
pairs automatically.

### How steps are detected and paired

The pairing is purely filename-based. For every `*.tsv` file in each
directory, the script computes a **step key** from the filename using a
regex, then pairs files that produce the same step key.

The default regex is:

```
^(?:\d+_)?(?P<step>[^/]+?)(?:_[A-Za-z0-9]+)?\.tsv$
```

Which means:

1. **Strip an optional leading `N_` prefix** (your `1_`, `2_`, `3_` ordering numbers).
2. **Capture the middle** as the step key.
3. **Strip an optional trailing `_variant` suffix** (the alphanumeric tag before `.tsv`).

Examples of what the default regex produces:

| Filename | Step key |
|---|---|
| `2_domainLibraryStructuredSeq_meta.tsv` | `domainLibraryStructuredSeq` |
| `2_domainLibraryStructuredSeq_old.tsv`  | `domainLibraryStructuredSeq` |
| `3_domainLibraryInteractions.tsv`       | `domainLibraryInteractions` |
| `4_domainLibraryPhysicalProperties.tsv` | `domainLibraryPhysicalProperties` |
| `5_finalCandidateSequences.tsv`         | `finalCandidateSequences` |
| `6_myBrandNewStep_variantA.tsv`         | `myBrandNewStep` |

The step key doesn't need to be in any registry — it's computed fresh from
each filename at runtime. So as long as two files across the two directories
produce the same step key, they get paired and compared.

### Rules and edge cases

- Only `.tsv` files are considered.
- Files that don't match the regex are silently ignored.
- If a step key appears in only one directory, a `NOTE:` line is printed to
  stderr and the file is skipped.
- If a step key matches multiple files within the same directory, the first
  one (alphabetically) is used and a `NOTE:` is printed. Rename or move the
  extras if that's wrong.
- The step key becomes the label for that step in the Overall_summary sheet
  and is embedded in per-step sheet names (truncated to fit Excel's 31-char
  sheet name limit).

### Using a custom step pattern

If your naming convention is different, supply your own regex via
`--step-pattern`. Your regex **must** contain a `(?P<step>...)` named group.

Examples:

```bash
# Files named like: run3_interactions.v2.tsv -> step key "interactions"
--step-pattern 'run\d+_(?P<step>\w+)\.v\d+\.tsv$'

# Files named like: step_02_structuredSeq.tsv -> step key "structuredSeq"
--step-pattern 'step_\d+_(?P<step>\w+)\.tsv$'
```

### What happens per step

Each paired step goes through the same `compare_pair()` flow described above,
using the labels you passed in (same `--label1`/`--label2` for every step).
That means every step in the batch workbook shares a consistent labeling
scheme, which is important — batch mode assumes the label distinction
(e.g. "old vs new") applies uniformly across all steps.

---

## Output structure

### Pair mode workbook

| Sheet | Contents |
|---|---|
| `Summary` | Step name, row/gene counts, status breakdown, columns unique to each file, per-numeric-column delta statistics |
| `Full_diff` | Every output row from the comparison in one sheet, sorted by status |
| `identical` | Rows where every value matched within tolerance |
| `changed_boundaries` | Paired rows with shifted `Start`/`End` |
| `changed_values` | Paired rows with same boundaries but different values |
| `gene_only_in_<label1>` | Genes present only in file 1 |
| `gene_only_in_<label2>` | Genes present only in file 2 |
| `only_in_<label1>` | Domains present only in file 1 (gene exists in both) |
| `only_in_<label2>` | Domains present only in file 2 (gene exists in both) |

Every data sheet has:

- Frozen header row + frozen identity columns so you can scroll horizontally
  without losing context.
- Autofilter on the header row so you can filter any column quickly.
- Conditional formatting on every `delta__X` column: red for negative
  deltas, white at zero, blue for positive — large shifts jump out visually.
- Floats formatted to 4 decimals.

### Batch mode workbook

| Sheet | Contents |
|---|---|
| `Overall_summary` | One row per step showing all the status counts side-by-side — the fastest way to see "step 2 changed a lot, step 5 barely moved" |
| `sum_<step_name>` | One per step: the same summary you'd get in pair mode |
| `diff_<step_name>` | One per step: the full diff for that step (no per-status splits in batch mode — that would produce too many sheets) |

Sheet names are truncated to 31 characters because Excel's limit; in practice
this rarely causes collisions but if your step keys are long you may want
shorter keys.

---

## Adapting the script to new pipeline steps

**Most of the time you don't need to change anything.** The script's design
is schema-agnostic in the ways that matter:

✅ **Adding a new metric column** (any numeric value per domain) — detected
automatically and compared with a delta column. Zero changes needed.

✅ **Adding a new pipeline step with a different set of columns** — as long
as both run directories contain a file for the step with the same step key,
batch mode will find and compare them automatically. Zero changes needed.

✅ **Step files that drop columns present in earlier steps** (e.g. your step
5 has no `Gene Name`) — handled via the "identity columns that exist in
both files" rule. Zero changes needed.

### When you *do* need to change something

There are only a few cases that require editing the script:

1. **Your new step has no `Entry` column.** The script currently requires
   `Entry` in both files. If you ever produce a step output keyed on something
   else entirely (e.g. a different accession system), you'd change the check in
   `detect_schema()` and the `key_cols` logic.

2. **You want a non-numeric column treated as "worth tracking changes on"
   with a fancier comparator** (e.g. sequence-level diffs of `Domain
   Sequence` rather than a bare string equality check). Currently text
   columns only contribute to `changed_values` via equality — no delta.
   Would require extending `classify_pair()`.

3. **You rename your naming convention** (e.g. stop using the `N_step_variant`
   pattern). Use `--step-pattern` to supply a new regex; no code change.

4. **You want to add a new identity or position column** (say you start
   tagging rows with a `Chain` column and want it to be part of the identity).
   Add its name to `IDENTITY_CANDIDATES` or `POSITION_CANDIDATES` at the top
   of the script.

In short: the script is designed so that **growing the pipeline horizontally
(more steps) or vertically (more columns per step) costs you nothing**. The
only structural changes that require code edits are the ones that break the
assumptions about identity/position columns.

---

## Tunable parameters inside the script

These live at the top of the file under `# Configuration`:

| Name | Default | Effect |
|---|---|---|
| `IDENTITY_CANDIDATES` | `['Entry', 'Gene Name', 'Domain']` | Columns treated as row labels |
| `POSITION_CANDIDATES` | `['Start', 'End']` | Columns used for pairing and boundary-shift detection |
| `FLOAT_TOL` | `1e-9` | Below this, float differences are considered "identical". Bump to `1e-4` or similar if small numerical drift should be ignored |
| `FONT` | `'Arial'` | Font used in the Excel output |

---

## Troubleshooting / caveats

### "I get a lot of `changed_boundaries` rows with huge Start shifts"

The greedy nearest-neighbor pairing will always pair every row it can, even
when the nearest partner is hundreds of residues away. If one algorithm
merges two repeats into one (or splits one into two), the pairing can look
odd — one row gets paired (with a large `delta__Start`) and the extras get
labeled `only_in_*`.

Mitigations:

- Sort `changed_boundaries` by `|delta__Start|` and inspect the extreme cases.
- If it's a systematic issue, we can add a `--max-shift` flag that refuses
  to pair rows whose Start shift exceeds a threshold (they'd become `only_in_*`
  on both sides instead).

### "I see `0 identical` even though most rows look the same"

Floating-point numeric columns (`Mean Disorder`, etc.) are the usual culprit
— even tiny algorithm tweaks propagate enough noise to push every paired row
into `changed_values`. Check the per-column delta stats in the Summary sheet
— if `max_abs_delta` is 1e-5 or smaller you probably want to raise `FLOAT_TOL`.

### "`NOTE: no match in DIR2 for step X`"

That step file exists in one directory but not the other. Either:

- You forgot to produce that file in the second run — rerun that stage.
- The filenames don't share a step key — check what the default regex produces,
  or pass `--step-pattern`.

### "Sheet names look truncated / collide"

Excel caps sheet names at 31 characters and forbids `[]:*?/\`. Long step
names get cut. If you're getting collisions, rename the upstream files to
use shorter step identifiers.

### "A column is numeric in one file but not the other"

It gets classified as a **text** column (no delta), because the script only
treats a column as numeric when both sides agree. Common cause: missing
values that pandas reads as strings. Clean the upstream data, or coerce with
`pd.to_numeric(..., errors='coerce')` before handing to the script.
