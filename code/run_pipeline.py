#!/usr/bin/env python3
"""
run_pipeline.py  —  Orchestrate the structured domain library pipeline.

MODES
-----
pipeline  Run a consecutive range of steps, automatically chaining each
          step's output into the next step's input.

          Step 1's input is always the hardcoded UniProt proteome TSV (set
          inside 1_domainInfoExtractor.py). All outputs land in --output-dir.

          # Full pipeline, steps 1–6
          python run_pipeline.py pipeline --output-dir ./runs/my_run

          # Resume at step 2 using an existing step-1 output
          python run_pipeline.py pipeline --start 2 --end 4 \\
              --input ./runs/my_run/1_domainLibraryRaw.tsv \\
              --output-dir ./runs/my_run

step      Run a single step with paths you supply explicitly. Useful for
          re-running one step without touching the rest.

          python run_pipeline.py step --step 3 \\
              --input  ./runs/my_run/2_domainLibraryStructured.tsv \\
              --output ./runs/my_run/3_domainLibraryInteractions.tsv \\
              --af-dir ./runs/my_run/alphafold_files

PASSING EXTRA FLAGS TO A STEP
------------------------------
Append -- followed by any flags that the underlying script understands.
These are forwarded verbatim to every step that is run.

  # Run pipeline but skip pLDDT in step 2
  python run_pipeline.py pipeline --output-dir ./runs/r1 -- --skip plddt

  # Use a stricter pLDDT threshold
  python run_pipeline.py pipeline --output-dir ./runs/r1 -- --plddt-threshold 80

NOTES
-----
- Steps 1–4 and 6 are Python; step 5 is R (requires Rscript on PATH).
- Steps 3 and 4 download / read AlphaFold PDB and PAE files. Use --af-dir
  to point at an existing cache and avoid re-downloading. In pipeline mode,
  --af-dir defaults to <output-dir>/alphafold_files.
- Intermediate file names inside --output-dir follow the convention:
    1_domainLibraryRaw.tsv
    2_domainLibraryStructured.tsv
    3_domainLibraryInteractions.tsv
    4_domainLibraryPhysicalProperties.tsv
    5_finalCandidateSequences.tsv
    images/  (step 6 output directory)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEP_NAMES = {
    1: "Domain Info Extraction",
    2: "Disorder Prediction + pLDDT Filter",
    3: "AlphaFold Domain Interactions",
    4: "Physical Property Calculation",
    5: "Final Candidate Sequences (R)",
    6: "PyMOL Structure Images",
}

# Output filename for each step inside --output-dir (step 6 uses a directory)
STEP_OUTPUT_FILES = {
    1: "1_domainLibraryRaw.tsv",
    2: "2_domainLibraryStructured.tsv",
    3: "3_domainLibraryInteractions.tsv",
    4: "4_domainLibraryPhysicalProperties.tsv",
    5: "5_finalCandidateSequences.tsv",
}

CODE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    width = 62
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")


def _step_header(step: int) -> None:
    _banner(f"Step {step}: {STEP_NAMES[step]}")
    print(f"  Started at {time.strftime('%H:%M:%S')}")


def _step_done(step: int, elapsed: float, output: Optional[str] = None) -> None:
    print(f"\n  [DONE] Step {step} finished in {elapsed:.1f}s")
    if output:
        print(f"  Output → {output}")


def _step_failed(step: int, returncode: int) -> None:
    print(f"\n  [FAIL] Step {step} exited with code {returncode}. Stopping.")


# ---------------------------------------------------------------------------
# Per-step subprocess runners
# ---------------------------------------------------------------------------

def _python(script: str, *flags: str) -> list[str]:
    return [sys.executable, str(CODE_DIR / script), *flags]


def _run(cmd: list[str]) -> int:
    result = subprocess.run(cmd)
    return result.returncode


def run_step1(output: Path, extra: list[str]) -> int:
    return _run(_python("1_domainInfoExtractor.py",
                        "--output", str(output),
                        *extra))


def run_step2(inp: Path, output: Path, af_dir: Optional[Path],
              extra: list[str]) -> int:
    cmd = ["--input", str(inp), "--output", str(output)]
    if af_dir is not None:
        cmd += ["--af-dir", str(af_dir)]
    return _run(_python("2_disorderedPredictions_3in1.py", *cmd, *extra))


def run_step3(inp: Path, output: Path, af_dir: Path, extra: list[str]) -> int:
    return _run(_python("3_alphaFoldDomainInteractions.py",
                        "--input",  str(inp),
                        "--output", str(output),
                        "--af-dir", str(af_dir),
                        *extra))


def run_step4(inp: Path, output: Path, af_dir: Path, extra: list[str]) -> int:
    return _run(_python("4_physicalPropertyDomainStruct.py",
                        "--input",  str(inp),
                        "--output", str(output),
                        "--af-dir", str(af_dir),
                        *extra))


def run_step5(inp: Path, output: Path, extra: list[str]) -> int:
    return _run(["Rscript",
                 str(CODE_DIR / "5_obtainFinalCandidateSequences.R"),
                 str(inp), str(output),
                 *extra])


def run_step6(inp: Path, output_dir: Path, extra: list[str]) -> int:
    return _run(_python("6_pymolImages.py",
                        "--input",      str(inp),
                        "--output-dir", str(output_dir),
                        *extra))


# ---------------------------------------------------------------------------
# Mode: pipeline
# ---------------------------------------------------------------------------

def mode_pipeline(args: argparse.Namespace, extra: list[str]) -> None:
    start     = args.start
    end       = args.end
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    af_dir = Path(args.af_dir) if args.af_dir else out_dir / "alphafold_files"
    af_dir.mkdir(parents=True, exist_ok=True)

    # The first step in the range needs an explicit --input when start > 1
    if start > 1:
        if not args.input:
            sys.exit(f"ERROR: --input is required when --start {start} "
                     f"(provide the output of step {start - 1})")
        current_input: Optional[Path] = Path(args.input)
        if not current_input.exists():
            sys.exit(f"ERROR: --input not found: {current_input}")
    else:
        current_input = None   # step 1 uses its hardcoded proteome input

    _banner(f"Pipeline: steps {start}–{end}  |  output dir: {out_dir}")

    total_t0 = time.time()

    for step in range(start, end + 1):
        _step_header(step)
        t0 = time.time()

        # Resolve output path for this step
        if step < 6:
            step_output: Path = out_dir / STEP_OUTPUT_FILES[step]
        else:
            step_output = out_dir   # step 6 writes into an images/ subdir

        # Dispatch
        if step == 1:
            rc = run_step1(step_output, extra)
        elif step == 2:
            rc = run_step2(current_input, step_output, af_dir, extra)
        elif step == 3:
            rc = run_step3(current_input, step_output, af_dir, extra)
        elif step == 4:
            rc = run_step4(current_input, step_output, af_dir, extra)
        elif step == 5:
            rc = run_step5(current_input, step_output, extra)
        elif step == 6:
            rc = run_step6(current_input, step_output, extra)

        elapsed = time.time() - t0

        if rc != 0:
            _step_failed(step, rc)
            sys.exit(rc)

        out_label = str(step_output / "images") if step == 6 else str(step_output)
        _step_done(step, elapsed, out_label)

        # Chain: this step's output becomes next step's input
        if step < 6:
            current_input = step_output

    total_elapsed = time.time() - total_t0
    _banner(f"Pipeline complete!  Total time: {total_elapsed:.1f}s")
    print(f"  All outputs in: {out_dir}\n")


# ---------------------------------------------------------------------------
# Mode: step
# ---------------------------------------------------------------------------

def mode_step(args: argparse.Namespace, extra: list[str]) -> None:
    step     = args.step
    inp      = Path(args.input)      if args.input      else None
    output   = Path(args.output)     if args.output     else None
    out_dir  = Path(args.output_dir) if args.output_dir else None
    af_dir   = Path(args.af_dir)     if args.af_dir     else None

    # Validate inputs
    if step != 1 and (inp is None or not inp.exists()):
        sys.exit(f"ERROR: --input is required for step {step} and must exist.")

    _step_header(step)
    t0 = time.time()

    if step == 1:
        if output is None:
            sys.exit("ERROR: --output is required for step mode.")
        output.parent.mkdir(parents=True, exist_ok=True)
        rc = run_step1(output, extra)

    elif step == 2:
        if output is None:
            sys.exit("ERROR: --output is required for step mode.")
        output.parent.mkdir(parents=True, exist_ok=True)
        rc = run_step2(inp, output, af_dir, extra)

    elif step == 3:
        if output is None:
            sys.exit("ERROR: --output is required for step mode.")
        if af_dir is None:
            sys.exit("ERROR: --af-dir is required for step 3.")
        output.parent.mkdir(parents=True, exist_ok=True)
        af_dir.mkdir(parents=True, exist_ok=True)
        rc = run_step3(inp, output, af_dir, extra)

    elif step == 4:
        if output is None:
            sys.exit("ERROR: --output is required for step mode.")
        if af_dir is None:
            sys.exit("ERROR: --af-dir is required for step 4.")
        output.parent.mkdir(parents=True, exist_ok=True)
        rc = run_step4(inp, output, af_dir, extra)

    elif step == 5:
        if output is None:
            sys.exit("ERROR: --output is required for step mode.")
        output.parent.mkdir(parents=True, exist_ok=True)
        rc = run_step5(inp, output, extra)

    elif step == 6:
        if out_dir is None:
            sys.exit("ERROR: --output-dir is required for step 6.")
        out_dir.mkdir(parents=True, exist_ok=True)
        rc = run_step6(inp, out_dir, extra)

    elapsed = time.time() - t0

    if rc != 0:
        _step_failed(step, rc)
        sys.exit(rc)

    out_label = str(out_dir / "images") if step == 6 else str(output)
    _step_done(step, elapsed, out_label)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # Split argv on '--' so anything after it is forwarded to the child script
    argv = sys.argv[1:]
    if "--" in argv:
        split     = argv.index("--")
        main_argv = argv[:split]
        extra     = argv[split + 1:]
    else:
        main_argv = argv
        extra     = []

    ap = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True,
                            metavar="{pipeline, step}")

    # --- pipeline -----------------------------------------------------------
    pp = sub.add_parser(
        "pipeline",
        help="Run a consecutive range of steps, chaining outputs to inputs.")
    pp.add_argument("--start", type=int, default=1,
                    choices=range(1, 7), metavar="{1-6}",
                    help="First step to run (default: 1)")
    pp.add_argument("--end", type=int, default=6,
                    choices=range(1, 7), metavar="{1-6}",
                    help="Last step to run (default: 6)")
    pp.add_argument("--output-dir", required=True, metavar="DIR",
                    help="Directory where all pipeline outputs are written")
    pp.add_argument("--input", default=None, metavar="PATH",
                    help="Input file for the first step "
                         "(required when --start > 1)")
    pp.add_argument("--af-dir", default=None, metavar="DIR",
                    help="AlphaFold PDB/PAE cache directory (steps 2, 3 & 4). "
                         "Step 2 uses it for pLDDT lookup; steps 3 & 4 use it "
                         "for structure files. Default: <output-dir>/alphafold_files")

    # --- step ---------------------------------------------------------------
    sp = sub.add_parser(
        "step",
        help="Run a single step with explicitly provided paths.")
    sp.add_argument("--step", type=int, required=True,
                    choices=range(1, 7), metavar="{1-6}",
                    help="Step number to run")
    sp.add_argument("--input", default=None, metavar="PATH",
                    help="Input file (not needed for step 1)")
    sp.add_argument("--output", default=None, metavar="PATH",
                    help="Output file (steps 1–5)")
    sp.add_argument("--output-dir", default=None, metavar="DIR",
                    help="Output directory (step 6 only)")
    sp.add_argument("--af-dir", default=None, metavar="DIR",
                    help="AlphaFold files directory (required for steps 3 & 4)")

    args = ap.parse_args(main_argv)

    if args.mode == "pipeline":
        if args.start > args.end:
            ap.error(f"--start ({args.start}) must be <= --end ({args.end})")
        mode_pipeline(args, extra)
    else:
        mode_step(args, extra)


if __name__ == "__main__":
    main()
