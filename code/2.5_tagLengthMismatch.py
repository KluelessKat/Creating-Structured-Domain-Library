#!/usr/bin/env python3
"""
2.5_tagLengthMismatch.py

Tag a domain TSV with UniProt ↔ AlphaFold-DB sequence-mismatch flags so
step 3 (and any downstream consumer) can skip entries whose AF model is
for a different (older / shorter) sequence version than the input TSV.

Position in the pipeline
------------------------
Runs AFTER step 1 (or step 2) and BEFORE step 3. Inputs:
  - The TSV produced by step 1 (1_domainLibraryRaw.tsv) or step 2
    (2_domainLibraryStructuredSeq_meta.tsv). Only required columns are
    'Entry' and 'Length'; everything else passes through untouched.
  - The audit TSV produced by audit_uniprot_vs_af.py.

Output columns added
--------------------
  lengthMismatch        bool   True iff this Entry should be excluded from
                               structure-based metrics. Covers both the
                               length-revision case and the no-AF-model case.
  af_length             int    Length AF DB v6 actually modeled (0 = no model).
  af_mismatch_reason    str    'length_revision' / 'no_af_model' / ''
                               Human-readable why the row was tagged.

Usage
-----
    # Run the audit first (one-time, ~5 min for the whole proteome):
    python audit_uniprot_vs_af.py humanProteome.tsv af_audit.tsv

    # Then tag step-1 or step-2 output:
    python 2.5_tagLengthMismatch.py 1_domainLibraryRaw.tsv \\
                                    af_audit.tsv \\
                                    2.5_domainLibraryTagged.tsv

    # Now step 3 can read the tagged file and skip tagged rows automatically.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_tsv",
                    help="Step-1 or step-2 output TSV (must have 'Entry' "
                         "and 'Length' columns).")
    ap.add_argument("audit_tsv",
                    help="Audit TSV from audit_uniprot_vs_af.py.")
    ap.add_argument("output_tsv",
                    help="Where to write the tagged TSV.")
    args = ap.parse_args()

    df = pd.read_csv(args.input_tsv, sep="\t")
    if "Entry" not in df.columns:
        sys.exit("ERROR: input TSV missing 'Entry' column.")

    audit = pd.read_csv(args.audit_tsv, sep="\t")
    for col in ("Entry", "mismatch", "af_length"):
        if col not in audit.columns:
            sys.exit(f"ERROR: audit TSV missing required column '{col}'.")

    # Build per-Entry lookup tables from the audit.
    # mismatch column has values: 'no' / 'YES' / 'no_af_model' / ''
    bad_set = set(audit.loc[
        audit["mismatch"].isin(["YES", "no_af_model"]), "Entry"
    ].astype(str))

    af_length_lookup = dict(
        zip(audit["Entry"].astype(str),
            pd.to_numeric(audit["af_length"], errors="coerce").fillna(-1).astype(int))
    )

    reason_lookup = {}
    for _, row in audit.iterrows():
        e = str(row["Entry"])
        m = str(row.get("mismatch", ""))
        if m == "YES":
            reason_lookup[e] = "length_revision"
        elif m == "no_af_model":
            reason_lookup[e] = "no_af_model"
        else:
            reason_lookup[e] = ""

    entries = df["Entry"].astype(str)
    df["lengthMismatch"]     = entries.isin(bad_set)
    df["af_length"]          = entries.map(af_length_lookup).fillna(-1).astype(int)
    df["af_mismatch_reason"] = entries.map(reason_lookup).fillna("")

    # Audit entries not present in the input TSV would map to NaN; the
    # fillna handles those. Entries in the input but missing from the audit
    # also land at af_length=-1 with empty reason — flag them so the user
    # knows to re-run the audit.
    df.loc[~entries.isin(audit["Entry"].astype(str)),
           "af_mismatch_reason"] = "not_in_audit"
    not_audited = int((df["af_mismatch_reason"] == "not_in_audit").sum())

    out_path = Path(args.output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)

    # Summary.
    n      = len(df)
    n_bad  = int(df["lengthMismatch"].sum())
    n_rev  = int((df["af_mismatch_reason"] == "length_revision").sum())
    n_no   = int((df["af_mismatch_reason"] == "no_af_model").sum())
    print(f"Wrote {args.output_tsv}")
    print(f"  total rows:                       {n:,}")
    print(f"  lengthMismatch=True rows:         {n_bad:,}  "
          f"({n_bad/n*100:.2f}%)")
    print(f"    of which length_revision:       {n_rev:,}")
    print(f"    of which no_af_model:           {n_no:,}")
    print(f"  unique affected UniProt entries:  "
          f"{df.loc[df['lengthMismatch'], 'Entry'].nunique():,}")
    if not_audited:
        print(f"  WARNING: {not_audited:,} input rows belong to entries not "
              f"present in the audit. They are NOT tagged. Re-run "
              f"audit_uniprot_vs_af.py to refresh.")


if __name__ == "__main__":
    main()
