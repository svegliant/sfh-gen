#!/usr/bin/env python3

"""
5x5 diabatic singlet fission Hamiltonian from a CAS(4,4)CI run.

The CASCI eigenvectors expand the adiabatic states in the CSF basis,
Phi_a = sum_i C_ai Psi_i, so in that basis

    H = C^T diag(E) C .

Exact only for a complete root set, where C is square and orthogonal. With
fewer roots the result is a rank-deficient projection; build_hamiltonian
refuses it.

The five diabatic states are single CSFs, identified by their GUGA occupation
string. The strings assume the active order HOMO_A, LUMO_A, HOMO_B, LUMO_B
produced by local_orbitals.py.
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import constants as _const

try:
    from . import __version__
except ImportError:        # standalone execution
    __version__ = "unknown"

try:
    from .instrumentation import block
except ImportError:        # standalone: python3 build_SF_H.py ...
    from instrumentation import block

AU2EV = _const.physical_constants["Hartree energy in eV"][0]

# Weight of |2020> in root 1 below which the localization is suspect.
GROUND_WEIGHT_MIN = 0.95

# Tolerance on max|C C^T - I|. C must be orthogonal for the transform above.
ORTHONORMALITY_TOL = 1e-8


class SFHamiltonianError(Exception):
    """A CASCI result the Hamiltonian cannot be built from."""


# CAS(4,4) singlet CSFs in OpenMolcas GUGA order (cf. the RASSCF wave function
# printout). Step numbers: 0 empty, u/d spin coupling +-1/2, 2 doubly occupied.
# Positions are the active orbitals in file order.
CSF_LABELS = [
    "2200", "2ud0", "2u0d", "u2d0", "u20d", "2020", "20ud", "2002", "ud20", "udud",
    "ud02", "0220", "02ud", "0202", "uudd", "u02d", "u0d2", "0u2d", "0ud2", "0022",
]

# Diabatic state -> CSF string, active order (h_A, l_A, h_B, l_B).
SF_STATES = [
    ("|A* B >", "ud20"),   # LE on A
    ("|A  B*>", "20ud"),   # LE on B
    ("|A+ B->", "u02d"),   # CT, hole on A
    ("|A- B+>", "2ud0"),   # CT, hole on B
    ("|AT BT>", "uudd"),   # singlet-coupled triplet pair
]

# Local closed-shell configuration; the diabatic ground state.
GROUND_CSF = "2020"


def csf_index(label):
    """0-based index of a CSF occupation string in the OpenMolcas GUGA order."""
    try:
        return CSF_LABELS.index(label)
    except ValueError:
        raise ValueError(f"'{label}' is not a CAS(4,4) singlet CSF. "
                         f"Known: {', '.join(CSF_LABELS)}")


def load_ci_data(filepath):
    """Load CI vectors and root energies from a RASSCF .h5 file."""
    try:
        with h5py.File(filepath, "r") as f:
            return f["CI_VECTORS"][:], f["ROOT_ENERGIES"][:]
    except KeyError as e:
        raise SFHamiltonianError(
            f"{filepath} does not contain {e}. It must be the .h5 of a RASSCF "
            f"run that computed CI vectors; an SCF .h5 will not do.") from e
    except OSError as e:
        raise SFHamiltonianError(f"Cannot read {filepath}: {e}") from e


def build_hamiltonian(ci_vectors, energies):
    """Hamiltonian in the CSF basis, in Hartree. Requires one root per CSF."""
    n_roots, n_csf = ci_vectors.shape
    if n_roots != n_csf:
        raise SFHamiltonianError(
            f"the CASCI produced {n_roots} roots for {n_csf} CSFs.\n"
            f"H = C^T diag(E) C is the Hamiltonian only when every root is\n"
            f"computed, so that C is square and orthogonal. With {n_roots} roots\n"
            f"it is a rank-{n_roots} projection whose diagonal is not a state\n"
            f"energy. Set the number of roots to {n_csf} and re-run.")
    if n_csf != len(CSF_LABELS):
        raise SFHamiltonianError(
            f"{n_csf} CSFs found, expected {len(CSF_LABELS)} for a CAS(4,4) "
            f"singlet. The SF Hamiltonian is defined for that active space only.")

    # Orthogonality of C makes the transform a change of basis rather than an
    # arbitrary congruence.
    deviation = np.abs(ci_vectors @ ci_vectors.T - np.eye(n_roots)).max()
    if deviation > ORTHONORMALITY_TOL:
        raise SFHamiltonianError(
            f"the CI vectors are not orthonormal: max|C C^T - I| = "
            f"{deviation:.2e}, above the {ORTHONORMALITY_TOL:.0e} tolerance.\n"
            f"H = C^T diag(E) C is a change of basis only for orthogonal C, so "
            f"the resulting Hamiltonian would be meaningless. The .h5 is "
            f"probably truncated or from an interrupted run.")

    return ci_vectors.T @ np.diag(energies) @ ci_vectors


def report_diagnostics(ci_vectors, hamiltonian, energies):
    """Report the sanity checks on the diabatic basis; return any warnings.

    The weight of |2020> in root 1 is the diagnostic of record. A rotation
    within the active space leaves the adiabatic energies invariant, so a
    delocalized active set shows up here and nowhere else.
    """
    warnings = []
    g = csf_index(GROUND_CSF)
    weight = ci_vectors[0, g] ** 2
    print(f"Ground-state check: weight of CSF |{GROUND_CSF}> in root 1 = {weight:.4f}"
          f"  (want > {GROUND_WEIGHT_MIN})")
    if weight < GROUND_WEIGHT_MIN:
        warnings.append(
            f"weight of CSF |{GROUND_CSF}> in root 1 is {weight:.4f}, below the "
            f"{GROUND_WEIGHT_MIN} expected for a weakly coupled dimer. The active "
            f"orbitals are probably not localized HOMO/LUMO pairs: check that the "
            f"monomer orbitals came from a root-specific RasOrb.N and that the "
            f"final RASSCF used CIOnly.")
        print(f"  WARNING: {warnings[-1]}")

    gap = (hamiltonian[g, g] - energies.min()) * AU2EV
    print(f"Diabatic ground state |{GROUND_CSF}> lies {gap:.4f} eV above the lowest "
          f"adiabatic root")
    return warnings


def extract_sf_states(hamiltonian, csf_labels, zero):
    """5x5 sub-block in eV, with `zero` (in Hartree) subtracted from the diagonal."""
    idx = np.array([csf_index(lab) for lab in csf_labels])
    h_sf = hamiltonian[np.ix_(idx, idx)].copy()
    h_sf[np.diag_indices_from(h_sf)] -= zero
    return h_sf * AU2EV


def matrix_lines(h_sf, state_labels):
    """The matrix as a header of column labels and one line per state.

    One formatting for the file and for the log, so the two cannot drift.
    """
    lines = ["".join(f"{lab:>12s}" for lab in state_labels) + "   | state"]
    lines += ["".join("%12.6f" % v for v in row) + f"   | {lab}"
              for row, lab in zip(h_sf, state_labels)]
    return lines


def format_hamiltonian(h_sf, state_labels, csf_labels, adiabatic_diagonal=None,
                       zero_shift=None):
    """The result block printed at the end of a run.

    The matrix itself is also written to H_SF.dat; it is repeated here so that
    one job log holds the result as well as the run that produced it.
    `adiabatic_diagonal` and `zero_shift`, when given, add the same diagonal
    referenced to the lowest adiabatic root, which is not written out.
    """
    states = ", ".join(f"{name} = {csf}"
                       for name, csf in zip(state_labels, csf_labels))
    lines = [
        f"energy zero : eps(|{GROUND_CSF}>), the diabatic ground state",
        f"states      : {states}",
        "              (active order HOMO_A, LUMO_A, HOMO_B, LUMO_B)",
        "",
    ]
    lines += matrix_lines(h_sf, state_labels)
    if adiabatic_diagonal is not None:
        lines += [
            "",
            "the same diagonal referenced to the lowest adiabatic root, which",
            "lies %.3f eV below eps(|%s>), for comparison only:"
            % (zero_shift, GROUND_CSF),
            "".join(f"{v:12.3f}" for v in adiabatic_diagonal),
        ]
    return block("Singlet Fission diabatic Hamiltonian / eV", lines)


def save_hamiltonian(h_sf, state_labels, output_file, source=None):
    """Write the 5x5 matrix with its unit, energy zero, source and state order.

    Written to a temporary file and renamed, so an interrupted run leaves the
    previous result intact.
    """
    head = [
        f"Singlet Fission diabatic Hamiltonian ({len(state_labels)}x{len(state_labels)})",
        f"produced by  : sfh-gen {__version__}",
        "units        : eV",
        f"energy zero  : eps(|{GROUND_CSF}>), the diabatic ground state",
    ]
    if source:
        head.append(f"source       : {os.path.abspath(source)}")
    rows = matrix_lines(h_sf, state_labels)
    head += ["", "row/col order: " + ", ".join(state_labels), "", rows[0]]

    text = "".join(f"# {ln}\n" if ln else "#\n" for ln in head)
    text += "".join(f"{row}\n" for row in rows[1:])

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".",
                    exist_ok=True)
        tmp = output_file + ".part"
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, output_file)       # atomic within a filesystem
        print(f"Hamiltonian saved to {output_file}")
    except OSError as e:
        raise SFHamiltonianError(f"Cannot write {output_file}: {e}") from e


def compute_sf_hamiltonian(h5_path, output):
    """CASCI .h5 -> H_SF.dat. Returns the 5x5 matrix in eV and any warnings.

    The diagonal is referenced to eps(|2020>), which lies in the same CSF basis
    as the five states. The lowest adiabatic root is a combination of CSFs and
    sits below every diabatic state by the stabilization energy; it is printed
    for comparison but not used.
    """
    print(f"Loading CI data from {h5_path}")
    ci_vectors, energies = load_ci_data(h5_path)
    print(f"Found {len(energies)} roots, {ci_vectors.shape[1]} CSFs")

    hamiltonian = build_hamiltonian(ci_vectors, energies)
    warnings = report_diagnostics(ci_vectors, hamiltonian, energies)

    state_names = [name for name, _ in SF_STATES]
    csf_labels = [csf for _, csf in SF_STATES]

    diabatic_zero  = hamiltonian[csf_index(GROUND_CSF), csf_index(GROUND_CSF)]
    adiabatic_zero = energies.min()

    h_sf = extract_sf_states(hamiltonian, csf_labels, diabatic_zero)

    # Reported for comparison only.
    diag_adia = np.diag(extract_sf_states(hamiltonian, csf_labels, adiabatic_zero))
    print(format_hamiltonian(h_sf, state_names, csf_labels,
                             adiabatic_diagonal=diag_adia,
                             zero_shift=(diabatic_zero - adiabatic_zero) * AU2EV),
          flush=True)
    save_hamiltonian(h_sf, state_names, output, source=h5_path)
    return h_sf, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Build the singlet fission Hamiltonian from a CAS(4,4)CI run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", default="dimer.rasscf.h5",
                        help="Input RASSCF HDF5 file")
    parser.add_argument("--output", "-o", default="H_SF.dat",
                        help="Output Hamiltonian file")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found", file=sys.stderr)
        sys.exit(1)

    try:
        compute_sf_hamiltonian(args.input, args.output)
    except SFHamiltonianError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
