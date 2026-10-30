"""
OpenMolcas input generator. Each step assembles the blocks it needs, in the
order the modules are to run.

    from .gen_inp import InputBuilder

    builder = InputBuilder()
    builder.add_gateway(geometry_path="mono1.xyz", basis="6-31G")
    builder.add_seward(cutoff=1.0e-20)
    builder.add_scf()
    builder.add_rasscf(charge=0, n_electrons=2, n_orbitals=2, n_roots=3)
    builder.write("01_mono1/input.inp")
"""

import os
from typing import Optional


# ── Exceptions ─────────────────────────────────────────────────────────────

class InputBuilderError(Exception):
    """Failure while building the OpenMolcas input."""
    pass


# ── Helper ─────────────────────────────────────────────────────────────────

def _fortran_float(x: float) -> str:
    """
    Format a float in the Fortran notation OpenMolcas expects
    (e.g. 1.0e-20 -> '1.0D-20').
    """
    mantissa, exponent = f"{float(x):.1E}".split("E")
    return f"{mantissa}D{int(exponent):+d}"


# ── Main builder ───────────────────────────────────────────────────────────

class InputBuilder:
    """
    Builds an OpenMolcas input file one block at a time. The order in
    which blocks are added is the order in which the modules run.
    """

    def __init__(self):
        self._blocks: list[str] = []

    # ── Available blocks ───────────────────────────────────────────────────

    def add_gateway(
        self,
        geometry_path: str,
        basis: Optional[str],
        basis_per_element: Optional[dict] = None,
        group: str = "NoSym",
        ricd: bool = True,
    ) -> "InputBuilder":
        """
        Add the &GATEWAY block (geometry + basis set).

        Parameters
        ----------
        geometry_path : str
            Path to the .xyz file.
        basis : str, optional
            One basis set for every atom (e.g. "6-31G"). None when
            basis_per_element is given.
        basis_per_element : dict, optional
            One basis set per element, {"C": "6-31G*", "H": "6-31G"}, used
            instead of `basis` and not as an override of it: the two are
            alternatives and giving both is a configuration error.
        group : str
            Point group. Left at "NoSym" by the pipeline: symmetry-adapted
            basis functions would not map between the monomer and dimer runs
            by index, which is what local_orbitals.py relies on.
        ricd : bool
            Atomic Cholesky decomposition of the two-electron integrals.

        NoMove is always emitted. Without it GATEWAY may translate or rotate
        the coordinates while looking for symmetry, leaving the monomer and
        dimer frames different; the AO embedding in local_orbitals.py requires
        a common frame.
        """
        if not os.path.isfile(geometry_path):
            raise InputBuilderError(
                f"[GATEWAY] Geometry file not found: '{geometry_path}'"
            )

        lines = [
            "&GATEWAY",
            f"  Coord = {os.path.abspath(geometry_path)}",
            f"  Group = {group}",
        ]

        # Basis set: uniform or per-element
        if basis_per_element:
            for element, b in basis_per_element.items():
                lines.append("  Basis set")
                lines.append(f"  {element}.{b}")
                lines.append("  End of basis")
        else:
            lines.append(f"  Basis = {basis}")

        if ricd:
            lines.append("  RICD")
        lines.append("  NoMove")

        self._blocks.append("\n".join(lines))
        return self

    def add_seward(self, cutoff: Optional[float] = None) -> "InputBuilder":
        """
        Add the &SEWARD block (integral evaluation).

        Parameters
        ----------
        cutoff : float, optional
            Integral cutoff (CutOff keyword). The same value in every step,
            otherwise the final CASCI runs on different integrals from those
            its orbitals came from.
        """
        lines = ["&SEWARD"]
        if cutoff is not None:
            lines.append(f"  CutOff = {_fortran_float(cutoff)}")
        self._blocks.append("\n".join(lines))
        return self

    def add_scf(self) -> "InputBuilder":
        """
        Add the &SCF block (Hartree-Fock), giving the RASSCF start orbitals.
        """
        self._blocks.append("&SCF")
        return self

    def add_rasscf(
        self,
        charge: int,
        n_electrons: int,
        n_orbitals: int,
        n_roots: int,
        spin: int = 1,
        orbital_file: Optional[str] = None,
        ci_only: bool = False,
        check_orbital_file: bool = True,
    ) -> "InputBuilder":
        """
        Add the &RASSCF block.

        Parameters
        ----------
        charge : int
            Total charge.
        n_electrons : int
            Electrons in the RAS2 active space.
        n_orbitals : int
            Orbitals in the RAS2 active space.
        n_roots : int
            Number of roots (state-averaged with equal weights).
        spin : int
            Multiplicity (1 = singlet).
        orbital_file : str, optional
            Path to the starting orbital file (File/FileOrb keyword).
        check_orbital_file : bool
            Verify that orbital_file exists. Off for a dry run, where the CASCI
            input is written before the local orbitals exist.
        ci_only : bool
            CIOnly: no orbital optimization, only the CI diagonalization at
            fixed orbitals. Required in the final CASCI: otherwise RASSCF
            re-optimizes the orbitals, losing the localization, and rewrites
            the CI vectors in the state-averaged natural orbital basis, which
            invalidates the CSF indices.
        """
        lines = ["&RASSCF"]

        if orbital_file is not None:
            abs_path = os.path.abspath(orbital_file)
            if check_orbital_file and not os.path.isfile(abs_path):
                raise InputBuilderError(
                    f"[RASSCF] Orbital file not found: '{orbital_file}'"
                )
            lines.append(f"  File   = {abs_path}")

        lines += [
            f"  Charge = {charge}",
            f"  Spin   = {spin}",
            f"  nActEl = {n_electrons} 0 0",
            f"  Ras2   = {n_orbitals}",
            f"  CIRoot = {n_roots} {n_roots} 1",
        ]

        if ci_only:
            lines.append("  CIOnly")

        self._blocks.append("\n".join(lines))
        return self

    # ── Assembly ──────────────────────────────────────────────────────────

    def build(self) -> str:
        """
        Assemble every added block into a string ready to be written
        to an .inp file.
        """
        if not self._blocks:
            raise InputBuilderError("No block added: the input is empty.")
        return "\n\n".join(self._blocks) + "\n"

    def write(self, path: str) -> str:
        """
        Write the input to file, creating intermediate directories as
        needed. Returns the absolute path of the file written.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(self.build())
        return os.path.abspath(path)
