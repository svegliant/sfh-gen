#!/usr/bin/env python3
"""Put the eight computed Hamiltonians beside the published ones.

Reads each `*/H_SF.dat` with the library and compares it with the matching
table of Xie et al. (2019) in reference_xie2019.yaml.

    python3 compare_with_reference.py            # one table per dimer
    python3 compare_with_reference.py --summary  # one line per dimer

Two conventions have to be fixed before any comparison means anything.

Orbital phases are arbitrary, so every off-diagonal element may come out with
either sign; magnitudes are compared. Which monomer is called A is also a
convention, so exchanging LE1 with LE2 and CT1 with CT2 at the same time is a
relabelling of the same result, not a different one. Both labellings of the
*published* table are tried and the closer one is used; the computed matrix is
always printed exactly as H_SF.dat holds it, row for row, and the output says
when the published A and B were exchanged.

The two calculations do not share an energy zero, so the diagonal is judged
by its spread: the largest minus the smallest of the five differences, which
no uniform shift can remove.
"""

import argparse as arg
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

from sfh_gen.build_SF_H import SF_STATES

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "reference_xie2019.yaml")

LABELS = ["LE1", "LE2", "CT1", "CT2", "TT"]

#: Exchanging the two monomers: LE1 <-> LE2 and CT1 <-> CT2 together.
SWAP_AB = np.array([1, 0, 3, 2, 4])


def read_matrix(path: str) -> np.ndarray:
    """The 5x5 block of an H_SF.dat, in eV."""
    return np.loadtxt(path, comments="#", usecols=range(5))


def load_reference(path: str = REFERENCE) -> Dict[str, dict]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def align(h: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, bool]:
    """`ref` under the A/B labelling that sits closest to `h`, and whether
    its A and B were exchanged. `h` itself is never relabelled.

    A nearly symmetric dimer has nearly equal LE energies and the couplings
    alone barely separate the two labellings, so the whole matrix is scored:
    couplings by magnitude, the diagonal signed.
    """
    best, swapped, score = ref, False, np.inf
    for perm, is_swap in ((np.arange(5), False), (SWAP_AB, True)):
        candidate = ref[np.ix_(perm, perm)]
        d = np.abs(np.abs(candidate) - np.abs(h))
        total = d[np.triu_indices(5, 1)].sum() + np.abs(
            np.diag(candidate) - np.diag(h)).sum()
        if total < score:
            best, swapped, score = candidate, is_swap, total
    return best, swapped


def couplings(h: np.ndarray) -> np.ndarray:
    """The ten off-diagonal magnitudes, upper triangle, row by row."""
    return np.abs(h[np.triu_indices(5, 1)])


def report(run: str, dimer: str, source: str, h: np.ndarray, ref: np.ndarray):
    ref, swapped = align(h, ref)
    diag = np.diag(h) - np.diag(ref)

    print(f"{run}  —  {dimer}, {source}")
    if swapped:
        print("  (published A and B exchanged to match this work)")
    print(f"  {'element':12s}{'this work':>11s}{'Xie 2019':>11s}{'diff':>9s}")
    for i in range(5):
        print(f"  {'E(' + LABELS[i] + ')':12s}{h[i, i]:11.3f}{ref[i, i]:11.3f}"
              f"{diag[i]:+9.3f}")
    for i in range(5):
        for j in range(i + 1, 5):
            a, b = abs(h[i, j]), abs(ref[i, j])
            pair = f"{LABELS[i]}/{LABELS[j]}"
            print(f"  {pair:12s}{a:11.3f}{b:11.3f}{a - b:+9.3f}")
    worst = np.abs(couplings(h) - couplings(ref)).max()
    print(f"  largest coupling deviation {worst * 1000:5.0f} meV;"
          f"  diagonal spread after a uniform shift"
          f" {diag.max() - diag.min():.3f} eV")
    print()


def summarise(run: str, dimer: Optional[str], h: np.ndarray,
              ref: Optional[np.ndarray]):
    strongest = couplings(h).max()
    if ref is None:
        print(f"  {run:14s} {'—':>14s} {strongest * 1000:9.0f} "
              f"{'—':>9s} {'—':>11s}")
        return
    ref, _ = align(h, ref)
    diag = np.diag(h) - np.diag(ref)
    worst = np.abs(couplings(h) - couplings(ref)).max()
    print(f"  {run:14s} {dimer:>14s} {strongest * 1000:9.0f}"
          f" {worst * 1000:9.0f} {diag.max() - diag.min():11.3f}")


def main(argv=None) -> int:
    parser = arg.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", action="store_true",
                        help="one line per dimer instead of the full tables")
    args = parser.parse_args(argv)

    reference = load_reference()
    by_run = {entry["run"]: (dimer, entry) for dimer, entry in reference.items()
              if entry["run"]}

    runs = sorted(d for d in os.listdir(HERE)
                  if os.path.exists(os.path.join(HERE, d, "H_SF.dat")))
    if not runs:
        print(f"No */H_SF.dat under {HERE}", file=sys.stderr)
        return 1

    print("states: " + ", ".join(f"{lab} = {name}"
                                 for lab, (name, _) in zip(LABELS, SF_STATES)))
    print()
    if args.summary:
        print(f"  {'run':14s} {'published as':>14s} {'strongest':>9s}"
              f" {'worst err':>9s} {'diag spread':>11s}")
        print(f"  {'':14s} {'':>14s} {'meV':>9s} {'meV':>9s} {'eV':>11s}")
    for run in runs:
        h = read_matrix(os.path.join(HERE, run, "H_SF.dat"))
        dimer, entry = by_run.get(run, (None, None))
        ref = np.array(entry["matrix"]) if entry else None
        if args.summary:
            summarise(run, dimer, h, ref)
        elif ref is not None:
            report(run, dimer, entry["source"], h, ref)
    if not args.summary:
        unmatched = [r for r in runs if r not in by_run]
        if unmatched:
            print("No published table to compare with: " + ", ".join(unmatched))
            print("See README.md for why.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
