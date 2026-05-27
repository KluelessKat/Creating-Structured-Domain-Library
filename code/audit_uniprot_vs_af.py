#!/usr/bin/env python3
"""
audit_uniprot_vs_af.py

Cross-check a UniProt-derived domain TSV against AlphaFold DB to find
entries where the TSV's `Length` disagrees with the length AlphaFold actually
modeled (e.g. UniProt revised the canonical sequence after AF DB built v6).

Why this exists
---------------
UniProt updates faster than AlphaFold DB rebuilds. For entries where the
canonical sequence changed since the AF v6 release, the input TSV's
`Length` (and the domain start/end coords) refer to a sequence the AF model
no longer reflects. step 3's `anchoringIndex` will fail (PAE matrix size
mismatch) for those rows, and the SASA-based metrics may be misaligned.

This script enumerates the affected entries so you can:
  - Tag rows in your input TSV (lengthMismatch column) for downstream
    filtering / supplementary analysis.
  - Get a clean scope statement for your methods write-up ("X% of human
    UniProt entries have a sequence-revision mismatch with AF DB v6 as of
    <date>").

Usage
-----
    python audit_uniprot_vs_af.py INPUT_TSV OUTPUT_TSV [--workers N]

Concurrent (default 8 workers) so a ~20K-protein run finishes in minutes
rather than hours. Resumable: re-running picks up where it left off using
the partial output file as a checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
HTTP_TIMEOUT = 20
HTTP_MAX_TRIES = 3
HTTP_BACKOFF = 2.0


def fetch_af_metadata(acc: str) -> dict:
    """Return a dict with af_length / dates / mismatch info for one accession.
    On persistent error returns a dict with `af_length = -1` and an error
    string in `error`. Never raises.
    """
    last_err = None
    for attempt in range(1, HTTP_MAX_TRIES + 1):
        try:
            r = requests.get(AF_API.format(acc=acc), timeout=HTTP_TIMEOUT)
            if r.status_code == 404:
                return {
                    "Entry": acc, "af_length": 0,
                    "af_seq_version_date": "", "af_model_created": "",
                    "error": "AF_404_no_model",
                }
            r.raise_for_status()
            data = r.json()
            if not data:
                return {
                    "Entry": acc, "af_length": 0,
                    "af_seq_version_date": "", "af_model_created": "",
                    "error": "AF_empty_response",
                }
            d = data[0]
            seq = d.get("sequence") or ""
            return {
                "Entry": acc,
                "af_length": len(seq),
                "af_seq_version_date": (d.get("sequenceVersionDate") or "")[:10],
                "af_model_created":    (d.get("modelCreatedDate")     or "")[:10],
                "error": "",
            }
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < HTTP_MAX_TRIES:
                time.sleep(HTTP_BACKOFF * attempt)
    return {
        "Entry": acc, "af_length": -1,
        "af_seq_version_date": "", "af_model_created": "",
        "error": last_err or "unknown",
    }


def load_checkpoint(out_path: Path) -> set[str]:
    """Return set of Entry accessions already audited (resumable runs)."""
    if not out_path.exists():
        return set()
    try:
        prev = pd.read_csv(out_path, sep="\t")
        return set(prev["Entry"].astype(str).tolist())
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_tsv",
                    help="Input TSV with at least Entry and Length columns "
                         "(your step 1 / step 2 output, or the raw proteome).")
    ap.add_argument("output_tsv",
                    help="Where to write the audit table. Re-running with the "
                         "same path resumes from the existing rows.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent AF API requests (default 8). "
                         "Don't push much higher — AF EBI rate-limits.")
    ap.add_argument("--limit", type=int, default=None,
                    help="(debug) only process the first N unique entries.")
    args = ap.parse_args()

    in_path  = Path(args.input_tsv)
    out_path = Path(args.output_tsv)

    print(f"Reading {in_path} ...")
    df = pd.read_csv(in_path, sep="\t")
    for col in ("Entry", "Length"):
        if col not in df.columns:
            sys.exit(f"ERROR: input TSV missing required column '{col}'.")

    entries = (df[["Entry", "Length"]]
               .drop_duplicates(subset=["Entry"])
               .reset_index(drop=True))

    # Some proteome exports have rows with blank Length (e.g. obsolete
    # entries kept for historical reasons). Drop them so the int cast below
    # doesn't choke; they can't be audited anyway without a length to compare.
    before = len(entries)
    entries = entries.dropna(subset=["Length"])
    # Also drop rows where Length isn't a parseable integer.
    entries["Length"] = pd.to_numeric(entries["Length"], errors="coerce")
    entries = entries.dropna(subset=["Length"])
    dropped = before - len(entries)
    if dropped:
        print(f"  Dropped {dropped:,} entries with missing/non-numeric Length.")

    if args.limit:
        entries = entries.head(args.limit)
    print(f"{len(entries):,} unique UniProt entries to audit.")

    # Resume: skip entries already in the output file
    done = load_checkpoint(out_path)
    if done:
        print(f"Resuming — {len(done):,} entries already done in {out_path}.")
    todo = entries[~entries["Entry"].isin(done)].copy()
    if todo.empty:
        print("Nothing to do.")
        return
    print(f"{len(todo):,} entries to fetch.")

    # Append-mode write: stream rows as they come back so a crash/timeout
    # still leaves a usable partial result.
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    f_out = open(out_path, "a", buffering=1)  # line-buffered
    cols = ["Entry", "tsv_length", "af_length",
            "af_seq_version_date", "af_model_created",
            "mismatch", "error"]
    if write_header:
        f_out.write("\t".join(cols) + "\n")

    tsv_len_lookup = dict(zip(todo["Entry"].astype(str), todo["Length"].astype(int)))
    total = len(todo)
    done_count = 0
    start = time.time()
    mismatches = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_af_metadata, acc): acc
                   for acc in tsv_len_lookup}
        for fut in as_completed(futures):
            acc = futures[fut]
            res = fut.result()
            tsv_len = tsv_len_lookup[acc]
            af_len  = res["af_length"]
            if af_len < 0:
                errors += 1
                is_mismatch = ""
            elif af_len == 0:
                is_mismatch = "no_af_model"
                mismatches += 1
            elif af_len != tsv_len:
                is_mismatch = "YES"
                mismatches += 1
            else:
                is_mismatch = "no"
            f_out.write("\t".join([
                acc,
                str(tsv_len),
                str(af_len) if af_len >= 0 else "",
                res["af_seq_version_date"],
                res["af_model_created"],
                is_mismatch,
                res["error"],
            ]) + "\n")

            done_count += 1
            if done_count % 100 == 0 or done_count == total:
                elapsed = time.time() - start
                rate = done_count / elapsed if elapsed > 0 else 0
                eta_s = (total - done_count) / rate if rate > 0 else 0
                print(f"  {done_count:>5,}/{total:,}  "
                      f"({rate:5.1f} entries/sec, "
                      f"ETA {eta_s/60:5.1f} min, "
                      f"mismatches={mismatches:,}, errors={errors:,})",
                      flush=True)

    f_out.close()
    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min.")
    print(f"Wrote {out_path}")

    # ----- Summary --------------------------------------------------------
    summary = pd.read_csv(out_path, sep="\t")
    n = len(summary)
    n_ok          = int((summary["mismatch"] == "no").sum())
    n_mismatch    = int((summary["mismatch"] == "YES").sum())
    n_no_model    = int((summary["mismatch"] == "no_af_model").sum())
    n_err         = int(summary["error"].fillna("").astype(str).ne("").sum()
                        - n_no_model)  # errors that aren't 404s
    print("\nSummary:")
    print(f"  total entries audited:        {n:,}")
    print(f"  matched (tsv_length=af_length): {n_ok:,}  "
          f"({n_ok/n*100:.1f}%)")
    print(f"  length mismatched:            {n_mismatch:,}  "
          f"({n_mismatch/n*100:.1f}%)")
    print(f"  no AF model (AF 404):         {n_no_model:,}  "
          f"({n_no_model/n*100:.1f}%)")
    if n_err:
        print(f"  network/other errors:         {n_err:,}  "
              f"(re-run to retry — script is resumable)")


if __name__ == "__main__":
    main()
