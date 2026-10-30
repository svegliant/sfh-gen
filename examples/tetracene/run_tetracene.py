#!/usr/bin/env python3
"""Run the tetracene case through the library and check it against the reference.

Does what `sfh-gen config.yaml` does, but from Python, and then compares the
result with the committed matrix element by element.

    python3 run_tetracene.py [config.yaml]

Completed steps are skipped, so re-running only redoes what changed.
"""

import logging
import os
import sys

import numpy as np

from sfh_gen.config_parser import load_config, print_config_summary
from sfh_gen.instrumentation import setup_logger
from sfh_gen.run_all import run_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "H_SF.dat.reference")

STATES = ["|A* B >", "|A  B*>", "|A+ B->", "|A- B+>", "|AT BT>"]


def read_matrix(path):
    """The 5x5 block of an H_SF.dat, in eV."""
    return np.loadtxt(path, comments="#", usecols=range(5))


def main():
    config = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "config.yaml")

    cfg = load_config(config)          # absolute paths, no side effects
    print_config_summary(cfg)
    os.chdir(cfg.paths.workdir)        # the caller enters the working directory

    logger = setup_logger(cfg.run.verbose)
    run_pipeline(cfg, logger, start_from=None)

    h = read_matrix(cfg.hamiltonian.output)
    ref = read_matrix(REFERENCE)

    # The phase of each local orbital is arbitrary, so a diabatic state may
    # come out with either sign and the Hamiltonian with it: H'_ij = s_i s_j
    # H_ij for some s_i = +-1. That leaves the diagonal and every magnitude
    # untouched, so those are what is compared. Comparing signed off-diagonals
    # would report a spurious failure whenever the phases happened to differ.
    print()
    print(f"{'element':14s} {'this run':>10s} {'reference':>10s} {'diff / meV':>11s}")
    worst = 0.0
    flipped = 0
    for i in range(5):
        for j in range(i, 5):
            if i == j:
                d, tag = h[i, j] - ref[i, j], STATES[i]
            else:
                d, tag = abs(h[i, j]) - abs(ref[i, j]), f"{i + 1}/{j + 1}"
                if h[i, j] * ref[i, j] < 0:
                    flipped += 1
                    tag += " *"
            worst = max(worst, abs(d))
            print(f"{tag:14s} {h[i, j]:10.6f} {ref[i, j]:10.6f} {d * 1000:11.3f}")
    print()
    if flipped:
        print(f"* {flipped} element(s) differ in sign only: the orbital phases came")
        print("  out the other way round. Magnitudes are what carries meaning.")
    print(f"largest deviation from the reference: {worst * 1000:.3f} meV"
          "   (diagonal signed, off-diagonal by magnitude)")
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    # MDAnalysis and h5py emit DeprecationWarnings that would break up the
    # element-by-element comparison table. Routed to the logging machinery,
    # which has no handler for them, rather than printed to stderr.
    logging.captureWarnings(True)
    sys.exit(main())
