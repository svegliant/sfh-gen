#!/usr/bin/env python3
"""
Split a dimer .xyz into its two monomers, using MDAnalysis bond guessing to
find the fragments. Used by steps 01/02 when paths.monomer1 and
paths.monomer2 are absent from config.yaml, and standalone for checking.

    from .split_dimer import split_dimer

    mono1, mono2 = split_dimer(
        geometry_path="geometry/dimer.xyz",
        project_name=cfg.system.name,
        output_dir="geometry",
    )

Use from the command line:
    python split_dimer.py geometry/dimer.xyz --project dimer --output-dir geometry
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np


# ── Exceptions ─────────────────────────────────────────────────────────────

class SplitDimerError(Exception):
    """Failure while splitting the dimer into monomers."""
    pass


# ── Fragment validation ────────────────────────────────────────────────────

def check_fragments_contiguous(fragment_indices, geometry_path=""):
    """Require the fragments to be consecutive blocks, in order, from atom 0.

    local_orbitals.py locates each monomer as a contiguous block of the dimer
    and maps basis functions by index, so an .xyz interleaving the two
    monomers cannot be embedded. Takes plain index arrays, so it is testable
    without MDAnalysis.
    """
    expected = 0
    for i, idx in enumerate(fragment_indices, start=1):
        idx = np.asarray(idx)
        # A fragment with a hole has the right min and max but too few atoms,
        # so the length is checked as well as the index range.
        span = int(idx.max()) - int(idx.min()) + 1
        contiguous = (len(idx) == span and
                      np.array_equal(idx, np.arange(idx.min(), idx.max() + 1)))
        if idx.min() != expected or not contiguous:
            where = f" in '{geometry_path}'" if geometry_path else ""
            raise SplitDimerError(
                f"The atoms of monomer {i} do not form a contiguous block"
                f"{where} (indices {idx.min()}-{idx.max()}, expected to "
                f"start at {expected}).\n"
                "Reorder the .xyz file so that it lists all the atoms of one "
                "monomer first and all those of the other second."
            )
        expected = int(idx.max()) + 1


# ── Main function ──────────────────────────────────────────────────────────

def split_dimer(
    geometry_path: str,
    project_name: str,
    output_dir: Optional[str] = None,
) -> tuple[str, str]:
    """
    Split `geometry_path` (a dimer geometry) into two .xyz files, one per
    monomer, identifying the fragments through bond guessing.

    Parameters
    ----------
    geometry_path : str
        Path to the dimer .xyz file (typically cfg.paths.geometry).
    project_name : str
        System name (cfg.system.name), prefix of the output
        files: "{project_name}_mono1.xyz", "{project_name}_mono2.xyz".
    output_dir : str, optional
        Directory to write the monomer files into.
        If None, the directory containing `geometry_path` is used.

    Returns
    -------
    (mono1_path, mono2_path) : tuple[str, str]
        Absolute paths to the two monomer .xyz files, ordered deterministically
        by the index of each fragment's first atom.

    Raises
    ------
    SplitDimerError
        If MDAnalysis is not installed, the geometry file does not exist, the
        number of fragments found is not exactly two, or the fragments are not
        contiguous blocks of the dimer.
    """
    try:
        import MDAnalysis as mda
    except ImportError as e:
        raise SplitDimerError(
            "MDAnalysis is not installed. It is only needed to split the dimer "
            "automatically; alternatively set paths.monomer1 / paths.monomer2 in "
            "the config. Install it with: pip install MDAnalysis"
        ) from e

    if not os.path.isfile(geometry_path):
        raise SplitDimerError(f"Geometry file not found: '{geometry_path}'")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(geometry_path))
    os.makedirs(output_dir, exist_ok=True)

    # to_guess is the current spelling; guess_bonds is deprecated in
    # MDAnalysis 2.x and removed in 3.x. The modern form is tried first so a
    # current version does not warn on every run.
    try:
        try:
            universe = mda.Universe(geometry_path, to_guess=("types", "bonds"))
        except TypeError:
            universe = mda.Universe(geometry_path, guess_bonds=True)
        fragments = universe.atoms.fragments
    except Exception as e:
        raise SplitDimerError(
            f"Failed to load/analyse '{geometry_path}' with MDAnalysis: {e}\n"
            f"Give paths.monomer1 and paths.monomer2 in the config to skip the "
            f"automatic split entirely."
        ) from e
    if len(fragments) != 2:
        raise SplitDimerError(
            f"Expected exactly 2 fragments (the two monomers) in '{geometry_path}', "
            f"found {len(fragments)}. Check the geometry: if the monomers are "
            "covalently bound automatic bond guessing cannot separate them, and "
            "any extra molecules (solvent, counterions) must be removed first."
        )

    # Deterministic order: the fragment with the lowest first atom index is "mono1".
    fragments = sorted(fragments, key=lambda frag: frag.indices.min())

    check_fragments_contiguous([f.indices for f in fragments], geometry_path)

    output_paths = []
    for i, frag in enumerate(fragments, start=1):
        mono_path = os.path.join(output_dir, f"{project_name}_mono{i}.xyz")
        try:
            frag.write(mono_path)
        except Exception as e:
            raise SplitDimerError(f"Failed to write '{mono_path}': {e}") from e
        output_paths.append(os.path.abspath(mono_path))

    return output_paths[0], output_paths[1]


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Split a dimer geometry into its two monomers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("geometry", help="Dimer .xyz file")
    parser.add_argument(
        "--project", "-p", default="dimer",
        help="Project name used to name the output files",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory (default: the geometry's own directory)",
    )
    args = parser.parse_args()

    try:
        mono1, mono2 = split_dimer(
            geometry_path=args.geometry,
            project_name=args.project,
            output_dir=args.output_dir,
        )
    except SplitDimerError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Monomer 1: {mono1}")
    print(f"Monomer 2: {mono2}")


if __name__ == "__main__":
    main()
