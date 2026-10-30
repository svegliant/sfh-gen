#!/usr/bin/env python3
"""
Local (diabatic) dimer orbitals from two monomer RASSCF calculations and a
dimer .h5 supplying the shared AO overlap. The monomers are run without ghost
atoms, each in its own AO basis.

Two stages, both by symmetric (Loewdin) orthogonalization, C' = C S_mo^-1/2
with S_mo = C^T S C:

  1. Inactive + active orbitals of both monomers are embedded in the dimer AO
     basis and orthogonalized together. Loewdin gives the orthonormal set of
     least square deviation from the input, so localized monomer orbitals stay
     localized; this fixes the diabatic LE/CT/TT configurations.

  2. The remaining secondary orbitals are embedded, projected orthogonal to
     stage 1, and orthogonalized among themselves. Stage 1 is unaffected.

The output is a full nbas x nbas RasOrb grouped inactive / active / secondary,
monomer 1 before monomer 2 within each group, giving the active order
HOMO_A, LUMO_A, HOMO_B, LUMO_B assumed by the CAS(4,4)CI CSF labels.

Use the root-specific natural orbitals <project>.RasOrb.N, not the
state-averaged <project>.RasOrb. State averaging a CAS(2,2) over equally
weighted roots gives an active density of exactly I/2, whose natural orbitals
are an arbitrary rotation of {HOMO, LUMO}. The CASCI is invariant to that
rotation, so the adiabatic energies are unaffected while the diabatic basis is
destroyed. check_active_occupations rejects it.

Usage:
    python local_orbitals.py rasorb_1 h5_1 rasorb_2 h5_2 h5_dimer -o out.RasOrb
"""

import argparse
import os
import re
import sys
from collections import Counter

import h5py
import numpy as np

try:
    from .build_SF_H import GROUND_WEIGHT_MIN
except ImportError:      # standalone: python local_orbitals.py ...
    from build_SF_H import GROUND_WEIGHT_MIN


TYPE_ORDER = ["F", "I", "1", "2", "3", "S", "D"]

# Mulliken population on one fragment below which an active orbital is not
# localized enough for its diabatic label to mean anything.
LOCALIZATION_MIN = 0.95


# ── RasOrb I/O ─────────────────────────────────────────────────────────────

def _section(text, name):
    """Return the body of a #NAME section of an INPORB file, or None."""
    if f"#{name}" not in text:
        return None
    body = text.split(f"#{name}", 1)[1]
    return re.split(r"\n#\w+", body, maxsplit=1)[0]


def _numbers(body):
    """Floats in an INPORB section body, ignoring its '*' comment lines."""
    return [float(x)
            for line in body.splitlines() if not line.lstrip().startswith("*")
            for x in line.split()]


def parse_rasorb(path, nbas):
    """Parse a #INPORB 2.2 RasOrb file.

    Returns (C, occ, types):
      C      (nbas, norb) coefficients, columns = MOs, in the file's AO order
      occ    (norb,) occupation numbers from the #OCC section
      types  list of norb type letters ('I', '2', 'S', ...) from #INDEX
    """
    with open(path) as f:
        text = f.read()

    orb_section = _section(text, "ORB")
    if orb_section is None:
        raise ValueError(f"{path}: no #ORB section found.")
    blocks = re.split(r"\*\s*ORBITAL\s+\d+\s+\d+", orb_section)[1:]
    if not blocks:
        raise ValueError(f"{path}: no '* ORBITAL i j' blocks found.")
    coeffs = []
    for i, b in enumerate(blocks):
        nums = _numbers(b)
        if len(nums) != nbas:
            raise ValueError(f"{path}: orbital block {i+1} has {len(nums)} "
                             f"coefficients, expected nbas={nbas}.")
        coeffs.append(nums)
    C = np.array(coeffs).T
    norb = C.shape[1]

    occ_section = _section(text, "OCC")
    if occ_section is None:
        raise ValueError(f"{path}: no #OCC section found.")
    occ = np.array(_numbers(occ_section)[:norb])
    if occ.size != norb:
        raise ValueError(f"{path}: #OCC holds {occ.size} values, expected {norb}.")

    idx_section = _section(text, "INDEX")
    if idx_section is None:
        raise ValueError(f"{path}: no #INDEX section found.")
    types = []
    for line in idx_section.splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        # each row is a leading group digit followed by up to 10 type letters
        m = re.match(r"^\d\s*([a-zA-Z123]+)$", line.replace(" ", ""))
        if m:
            types.extend(m.group(1).upper())
    if len(types) != norb:
        raise ValueError(f"{path}: #INDEX holds {len(types)} type letters, "
                         f"expected {norb}.")

    return C, occ, types


def format_number(x):
    """Match OpenMolcas RasOrb numeric formatting (width 22, 14 decimals)."""
    return "{:22.14E}".format(x)


def write_rasorb(path, C, occ, nbas, type_letters, header_comment):
    norb = C.shape[1]
    letter_map = {"F": "f", "I": "i", "1": "1", "2": "2", "3": "3", "S": "s", "D": "d"}
    lines = ["#INPORB 2.2", "#INFO", f"* {header_comment}",
             "       0       1       0", f"    {nbas}", f"    {norb}", "#ORB"]
    for j in range(norb):
        lines.append(f"* ORBITAL    1    {j+1}")
        col = C[:, j]
        for i in range(0, nbas, 5):
            lines.append("".join(format_number(x) for x in col[i:i+5]))
    lines += ["#OCC", "* OCCUPATION NUMBERS"]
    for i in range(0, norb, 5):
        lines.append("".join(format_number(x) for x in occ[i:i+5]))
    lines += ["#INDEX", "* 1234567890"]
    row_letters = [letter_map.get(t, t.lower()) for t in type_letters]
    for row_start in range(0, norb, 10):
        row = row_letters[row_start:row_start + 10]
        lines.append(f"{(row_start // 10) % 10}" + "".join(row))
    # Temporary file plus rename: the presence of this file marks the step as
    # complete, so a partial write must never appear under the final name.
    tmp = path + ".part"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


# ── HDF5 / geometry mapping ────────────────────────────────────────────────

def load_monomer_h5(path):
    """Geometry and AO bookkeeping only -- the orbitals come from the RasOrb."""
    with h5py.File(path, "r") as f:
        nbas = int(f.attrs["NBAS"][0])
        return dict(
            nbas=nbas,
            bfids=f["BASIS_FUNCTION_IDS"][:],
            coords=f["CENTER_COORDINATES"][:],
        )


def format_warning_banner(warnings, tail: str = "") -> str:
    """Format the end-of-run "check by hand" block. Shared with run_all."""
    if not warnings:
        return ""
    lines = ["", "  " + "!" * 70, "  CHECK THIS RUN BY HAND:"]
    lines += [f"    - {w}" for w in warnings]
    lines += [
        "  The orbital order decides which CSF is which diabatic state, so a",
        "  mistake here is silent. Confirm that the weight of CSF |2020> in",
        f"  root 1 exceeds {GROUND_WEIGHT_MIN}, and that the two LE diagonal",
        "  elements come out close to each other.",
    ]
    if tail:
        lines.append(f"  {tail}")
    lines += ["  " + "!" * 70, ""]
    return "\n".join(lines)


class LocalOrbitalsError(Exception):
    """A monomer orbital set the local basis cannot be built from."""


def find_atom_offset(mono_coords, dimer_coords, label, tol=1e-6):
    """0-based starting atom index of mono_coords within dimer_coords."""
    n_mono = mono_coords.shape[0]
    for start in range(dimer_coords.shape[0] - n_mono + 1):
        # rtol=0: the default rtol=1e-5 would scale the tolerance with the
        # coordinate, loosening it far beyond the stated atol.
        if np.allclose(dimer_coords[start:start + n_mono], mono_coords,
                       rtol=0.0, atol=tol):
            return start
    raise LocalOrbitalsError(
        f"could not locate {label}'s geometry as a contiguous block within the "
        f"dimer geometry (tolerance {tol} Bohr).\n"
        f"Check that every OpenMolcas run used 'NoMove' in &GATEWAY -- without "
        f"it the monomer and dimer frames differ.")


def build_ao_index_map(dimer_bfids, atom_start, n_atoms):
    """0-based dimer AO indices belonging to atoms [atom_start, atom_start+n_atoms)."""
    center_ids = dimer_bfids[:, 0]
    lo, hi = atom_start + 1, atom_start + n_atoms
    return np.where((center_ids >= lo) & (center_ids <= hi))[0]


def verify_ao_ordering(dimer_bfids, dimer_idx, mono_bfids, atom_start, label):
    shifted = dimer_bfids[dimer_idx].copy()
    shifted[:, 0] -= atom_start
    if not np.array_equal(shifted, mono_bfids):
        raise LocalOrbitalsError(
            f"basis function ordering for {label} does not match the "
            f"corresponding block of the dimer basis -- cannot embed by simple "
            f"index scatter.")


def embed(C_sub, ao_idx, nbas_dimer):
    C = np.zeros((nbas_dimer, C_sub.shape[1]))
    C[ao_idx, :] = C_sub
    return C


# ── Orthogonalization ──────────────────────────────────────────────────────

def vec_lowdin(C, S):
    """Symmetric (Loewdin) orthonormalization of the columns of C in the S metric.

    C' = C S_mo^-1/2, S_mo = C^T S C. Minimizes sum_i |C'_i - C_i|^2 over all
    orthonormal sets and is independent of column order, unlike Gram-Schmidt.
    """
    S_mo = C.T @ S @ C
    w, V = np.linalg.eigh(0.5 * (S_mo + S_mo.T))
    return C @ (V / np.sqrt(w)) @ V.T


def check_conditioning(C, S, label, threshold):
    """Reject a set whose S-metric overlap is near-singular.

    A small eigenvalue of C^T S C means two input orbitals are nearly the same
    vector; S_mo^-1/2 then amplifies noise and the Loewdin set is meaningless.
    """
    S_mo = C.T @ S @ C
    eigvals = np.linalg.eigvalsh(0.5 * (S_mo + S_mo.T))
    print(f"  {label}: overlap eigenvalues min={eigvals.min():.6f}, max={eigvals.max():.6f}")
    if np.any(eigvals < threshold):
        bad = eigvals[eigvals < threshold]
        raise LocalOrbitalsError(
            f"{label} has eigenvalue(s) below threshold {threshold}: {bad}")


def project_out(A, F, S):
    """Remove any component of A's columns lying in F's column space (S-metric)."""
    return A - F @ (F.T @ S @ A)


# ── Diagnostics ────────────────────────────────────────────────────────────

def check_orbital_types(types, label):
    """Reject orbital types that would be dropped from the dimer basis."""
    unknown = sorted(set(types) - {"I", "2", "S"})
    if unknown:
        raise LocalOrbitalsError(
            f"{label} contains orbital type(s) {unknown}, which this script "
            f"does not handle (only inactive 'i', active '2' and secondary "
            f"'s'). Frozen or deleted orbitals would be dropped and the dimer "
            f"AO basis would be incomplete.")


def check_active_occupations(occ, types, label, tol=1e-3):
    """Validate the monomer active pair; return (indices, swapped), HOMO first.

    Degenerate occupations are fatal: the active density is then proportional
    to the identity and its eigenvectors are an arbitrary rotation of
    HOMO/LUMO, which no reordering recovers.

    Ascending occupations are not. OpenMolcas transforms to root-specific
    natural orbitals in place without guaranteeing descending order, so the
    LUMO may be written first; the assignment stays unambiguous and the two
    columns are swapped.
    """
    act = [i for i, t in enumerate(types) if t == "2"]
    if len(act) != 2:
        raise LocalOrbitalsError(
            f"{label} has {len(act)} active orbitals, expected 2 "
            f"(a CAS(2,2) HOMO/LUMO pair).")
    n_first, n_second = occ[act[0]], occ[act[1]]
    print(f"  {label}: active occupations {n_first:.6f} / {n_second:.6f}")

    if abs(n_first - n_second) < tol:
        raise LocalOrbitalsError(
            f"{label} active occupations are {n_first:.6f} and {n_second:.6f} "
            f"-- degenerate.\n"
            f"       The active orbitals are an arbitrary rotation of HOMO/LUMO and "
            f"the diabatic basis would be meaningless.\n"
            f"       Use the root-specific natural orbitals (<project>.RasOrb.N), "
            f"not the state-averaged <project>.RasOrb."
        )

    if n_first < n_second:
        print(f"  {label}: LUMO is stored before HOMO -- swapping the two active "
              f"columns so the block reads HOMO then LUMO.")
        return [act[1], act[0]], True

    return act, False


def report_localization(C_act, S, ao_idx_1, ao_idx_2, orbital_names):
    """Print the Mulliken population of each active orbital on each fragment."""
    SC = S @ C_act
    print("  Active orbital localization (Mulliken population per fragment):")
    ok = True
    for j, name in enumerate(orbital_names):
        q = C_act[:, j] * SC[:, j]
        pA, pB = q[ao_idx_1].sum(), q[ao_idx_2].sum()
        localized = max(pA, pB) > LOCALIZATION_MIN
        flag = "" if localized else "   <-- NOT LOCALIZED"
        print(f"    {name:8s} monomer 1 = {pA:+.4f}   monomer 2 = {pB:+.4f}{flag}")
        ok &= localized
    return ok


# ── Main ───────────────────────────────────────────────────────────────────

def _map_monomers_into_dimer(h5_1, h5_2, h5_dimer):
    """Read the dimer AO data and locate each monomer's AO block within it.

    Returns the dimer basis size, the AO overlap matrix, the dimer AO indices
    belonging to each monomer, and the two monomer basis sizes. The monomer and
    dimer runs must share a frame (&GATEWAY NoMove) and a basis ordering, both
    of which are verified here.
    """
    with h5py.File(h5_dimer, "r") as f:
        nbas_d = int(f.attrs["NBAS"][0])
        S_d = f["AO_OVERLAP_MATRIX"][:].reshape(nbas_d, nbas_d)
        bfids_d = f["BASIS_FUNCTION_IDS"][:]
        coords_d = f["CENTER_COORDINATES"][:]

    m1, m2 = load_monomer_h5(h5_1), load_monomer_h5(h5_2)
    start_1 = find_atom_offset(m1["coords"], coords_d, "monomer 1")
    start_2 = find_atom_offset(m2["coords"], coords_d, "monomer 2")
    idx_1 = build_ao_index_map(bfids_d, start_1, m1["coords"].shape[0])
    idx_2 = build_ao_index_map(bfids_d, start_2, m2["coords"].shape[0])
    verify_ao_ordering(bfids_d, idx_1, m1["bfids"], start_1, "monomer 1")
    verify_ao_ordering(bfids_d, idx_2, m2["bfids"], start_2, "monomer 2")

    return nbas_d, S_d, idx_1, idx_2, m1["nbas"], m2["nbas"]


def build_local_orbitals(rasorb_1, h5_1, rasorb_2, h5_2, h5_dimer, output,
                         eigval_threshold=1e-6):
    """Assemble the local dimer orbital set and write it as a RasOrb file.

    Returns (output_path, warnings); warnings lists recoverable problems that
    were worked around and should be checked by hand.
    """
    nbas_d, S_d, idx_1, idx_2, nbas_1, nbas_2 = _map_monomers_into_dimer(
        h5_1, h5_2, h5_dimer)

    C1, occ1, types1 = parse_rasorb(rasorb_1, nbas_1)
    C2, occ2, types2 = parse_rasorb(rasorb_2, nbas_2)

    # Validate the active pairs before any work.
    check_orbital_types(types1, "monomer 1")
    check_orbital_types(types2, "monomer 2")
    act_1, swap_1 = check_active_occupations(occ1, types1, "monomer 1")
    act_2, swap_2 = check_active_occupations(occ2, types2, "monomer 2")

    warnings = []
    for label, swapped in (("monomer 1", swap_1), ("monomer 2", swap_2)):
        if swapped:
            warnings.append(
                f"{label}: the two active orbitals were stored LUMO-first in the "
                f"RasOrb and have been swapped."
            )

    # Stage 1: inactive + active, active ordered HOMO then LUMO.
    sel_1 = [i for i, t in enumerate(types1) if t == "I"] + act_1
    sel_2 = [i for i, t in enumerate(types2) if t == "I"] + act_2
    types_occact = [types1[i] for i in sel_1] + [types2[i] for i in sel_2]
    occ_occact = list(occ1[sel_1]) + list(occ2[sel_2])

    C_occact_raw = np.hstack([embed(C1[:, sel_1], idx_1, nbas_d),
                              embed(C2[:, sel_2], idx_2, nbas_d)])
    print(f"Stage 1: {C_occact_raw.shape[1]} occupied+active orbitals "
          f"({len(sel_1)} + {len(sel_2)})")
    check_conditioning(C_occact_raw, S_d, "Stage 1", eigval_threshold)
    C_occact = vec_lowdin(C_occact_raw, S_d)

    # Stage 2: secondary orbitals fitted around stage 1.
    sec_1 = [i for i, t in enumerate(types1) if t == "S"]
    sec_2 = [i for i, t in enumerate(types2) if t == "S"]
    types_sec = ["S"] * (len(sec_1) + len(sec_2))
    occ_sec = [0.0] * (len(sec_1) + len(sec_2))

    C_sec_raw = np.hstack([embed(C1[:, sec_1], idx_1, nbas_d),
                           embed(C2[:, sec_2], idx_2, nbas_d)])
    print(f"Stage 2: {C_sec_raw.shape[1]} secondary orbitals "
          f"({len(sec_1)} + {len(sec_2)})")
    C_sec_proj = project_out(C_sec_raw, C_occact, S_d)
    check_conditioning(C_sec_proj, S_d, "Stage 2 (post-projection)", eigval_threshold)
    C_sec = vec_lowdin(C_sec_proj, S_d)

    # Assemble as inactive, active, secondary. sorted() is stable, so within
    # the active block the order stays HOMO_A, LUMO_A, HOMO_B, LUMO_B.
    C_all = np.hstack([C_occact, C_sec])
    types_all = types_occact + types_sec
    occ_all = np.array(occ_occact + occ_sec)
    order = sorted(range(C_all.shape[1]), key=lambda j: TYPE_ORDER.index(types_all[j]))
    C_full = C_all[:, order]
    ordered_types = [types_all[j] for j in order]
    ordered_occ = occ_all[order]

    act = [j for j, t in enumerate(ordered_types) if t == "2"]
    if len(act) != 4:
        raise LocalOrbitalsError(
            f"assembled {len(act)} active orbitals, expected 4.")
    if not report_localization(C_full[:, act], S_d, idx_1, idx_2,
                               ["HOMO_A", "LUMO_A", "HOMO_B", "LUMO_B"]):
        warnings.append("an active orbital is not localized on a single monomer; "
                        "the diabatic labels do not mean what they say.")

    full_check = C_full.T @ S_d @ C_full
    max_dev = np.max(np.abs(full_check - np.eye(C_full.shape[1])))
    print(f"Final set: {C_full.shape[1]} orbitals, type counts: {dict(Counter(ordered_types))}")
    print(f"Full-set orthonormality check: max|C^T S C - I| = {max_dev:.2e}")

    write_rasorb(output, C_full, ordered_occ, nbas_d, ordered_types,
                 "Local orbitals: Loewdin(occupied+active), secondary fitted around them")
    print(f"Wrote {C_full.shape[1]} orbitals to {output}")

    if warnings:
        print(format_warning_banner(warnings))

    return output, warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rasorb_1", help="RasOrb.N file for monomer 1 (root-specific NOs)")
    ap.add_argument("h5_1", help="Matching RASSCF .h5 file for monomer 1")
    ap.add_argument("rasorb_2", help="RasOrb.N file for monomer 2 (root-specific NOs)")
    ap.add_argument("h5_2", help="Matching RASSCF .h5 file for monomer 2")
    ap.add_argument("h5_dimer", help="Dimer .h5 file (AO_OVERLAP_MATRIX, geometry, basis IDs)")
    ap.add_argument("-o", "--output", default="local.RasOrb")
    ap.add_argument("--eigval-threshold", type=float, default=1e-6)
    args = ap.parse_args()

    try:
        _, warnings = build_local_orbitals(args.rasorb_1, args.h5_1, args.rasorb_2,
                                           args.h5_2, args.h5_dimer, args.output,
                                           args.eigval_threshold)
    except LocalOrbitalsError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
