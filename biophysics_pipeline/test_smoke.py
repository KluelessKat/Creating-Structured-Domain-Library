"""
test_smoke.py
=============

Quick correctness check on a small slice of the real data. Run this once
before kicking off the full proteome computation to make sure parsing and
metrics behave as expected.

    python test_smoke.py /path/to/humanProteome_KZ.tsv /path/to/1_domainLibraryRaw.tsv
"""

import sys
import pandas as pd
from properties import compute_all, fcr, ncpr, fraction_aromatic, scd, shd

def main(proteome_path, domain_path):
    print("=" * 60)
    print("PROTEOME (first 3 rows)")
    print("=" * 60)
    p = pd.read_csv(proteome_path, sep="\t").head(3)
    for _, r in p.iterrows():
        seq = r["Sequence"]
        if not isinstance(seq, str): continue
        props = compute_all(seq)
        print(f"\n{r['Entry']}  ({r.get('Gene Names')})  len={len(seq)}")
        for k, v in props.items():
            if v is None: continue
            try:
                print(f"  {k:22s} {v: .4f}")
            except (TypeError, ValueError):
                print(f"  {k:22s} {v}")

    print("\n" + "=" * 60)
    print("DOMAINS (first 3 rows)")
    print("=" * 60)
    d = pd.read_csv(domain_path, sep="\t").head(3)
    for _, r in d.iterrows():
        seq = r["Domain Sequence"]
        if not isinstance(seq, str): continue
        props = compute_all(seq)
        print(f"\n{r['Entry']}  {r['Domain']}  ({r['Start']}-{r['End']})  len={len(seq)}")
        print(f"  seq: {seq}")
        for k, v in props.items():
            if v is None: continue
            try:
                print(f"  {k:22s} {v: .4f}")
            except (TypeError, ValueError):
                print(f"  {k:22s} {v}")

    # spot-check one well-known sequence: a heavily charged synthetic stretch
    print("\n" + "=" * 60)
    print("SANITY CHECKS on synthetic sequences")
    print("=" * 60)
    s_alt = "EKEKEKEKEKEKEKEKEKEK"   # alternating -> low SCD (very negative)
    s_blk = "EEEEEEEEEEKKKKKKKKKK"   # block      -> less negative SCD
    print(f"alternating {s_alt}: FCR={fcr(s_alt):.2f} NCPR={ncpr(s_alt):.2f} SCD={scd(s_alt):.3f}")
    print(f"blocky      {s_blk}: FCR={fcr(s_blk):.2f} NCPR={ncpr(s_blk):.2f} SCD={scd(s_blk):.3f}")
    print("(For a net-neutral polyampholyte, BLOCKY should have a MORE-NEGATIVE")
    print(" SCD than alternating because long-range opposite-sign pairs dominate")
    print(" — this is the diagnostic from Sawle & Ghosh 2015.)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
