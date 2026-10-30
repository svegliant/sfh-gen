"""
Runs OpenMolcas on an input file and handles the log, the return code and the
output files. Generic: each step calls it with its own input and directory.

    from .run_molcas import run_molcas, MolcasError

    result = run_molcas(
        input_file = "01_mono1/input.inp",
        output_dir = "01_mono1",
        openmolcas = "/path/to/pymolcas",
        project    = "tetracene_mono1",
        logger     = logger,
    )
    # result.success, result.log_path
"""

import os
import shutil
import subprocess
import logging
from dataclasses import dataclass
from typing import Optional


# OpenMolcas prints this on the last line of a run that completed.
HAPPY_LANDING = "Happy landing"


# ── Result of a run ────────────────────────────────────────────────────────

@dataclass
class MolcasResult:
    """Status of one OpenMolcas run.

    Minimal: steps locate each other's products by name through
    run_all._locate rather than by passing paths around.
    """
    success:  bool
    log_path: str


# ── Exception ──────────────────────────────────────────────────────────────

class MolcasError(Exception):
    """Failure while running OpenMolcas."""
    pass


# ── Main function ──────────────────────────────────────────────────────────

def run_molcas(
    input_file:  str,
    output_dir:  str,
    openmolcas:  str,
    project:     str,
    logger:      logging.Logger,
    dry_run:     bool = False,
    scratch_dir: Optional[str] = None,
    keep_scratch: bool = False,
    nprocs: Optional[int] = None,
) -> MolcasResult:
    """
    Run OpenMolcas on `input_file` and collect the results into `output_dir`.

    Parameters
    ----------
    input_file : str
        Path to the .inp file to run.
    output_dir : str
        Directory to copy the log and output files into after the run.
    openmolcas : str
        Path to the pymolcas executable (or its name in PATH).
    project : str
        Project name, which OpenMolcas uses to name its output files.
    logger : logging.Logger
        Logger of the main pipeline.
    dry_run : bool
        If True, build the command but do not run it.
    scratch_dir : str, optional
        Scratch/working directory for OpenMolcas (MOLCAS_WORKDIR).
        If None, a subdirectory of output_dir is used.
    nprocs : int, optional
        Number of MPI processes for this run, exported as MOLCAS_NPROCS.
        Leave it as None for a serial OpenMolcas build: pymolcas treats any
        value above 1 as a parallel run and wraps the binaries with RUNBINARY,
        which on a serial build is not an MPI launcher.
    keep_scratch : bool
        If False (the default) the scratch directory is removed once the run
        succeeds. It holds only integrals and temporary files — on the order
        of a GB for a dimer — while everything the later steps need has
        already been copied into output_dir. Set it to True to inspect the
        intermediate files.

    Returns
    -------
    MolcasResult
        success and the path of the log file.

    Raises
    ------
    MolcasError
        If the executable cannot be found.
    """
    input_file = os.path.abspath(input_file)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if scratch_dir is None:
        scratch_dir = os.path.join(output_dir, "scratch")

    cmd = [openmolcas, input_file]
    env = _build_env(project=project, scratch_dir=scratch_dir, nprocs=nprocs)

    logger.info("Command: %s", " ".join(cmd))
    logger.debug("Project         = %s", env.get("Project"))
    logger.debug("MOLCAS_PROJECT  = %s", env.get("MOLCAS_PROJECT"))
    logger.debug("MOLCAS_WORKDIR  = %s", env.get("MOLCAS_WORKDIR"))
    logger.debug("MOLCAS_OUTPUT   = %s", env.get("MOLCAS_OUTPUT"))
    if nprocs is not None:
        logger.info("MOLCAS_NPROCS   = %d", nprocs)

    # ── dry_run: generate the input, touch nothing ────────────────────────
    # Returns before the scratch directory is cleared, which run.keep_scratch
    # may have preserved on purpose.
    if dry_run:
        logger.info("[DRY RUN] Execution skipped.")
        return MolcasResult(success=True, log_path="(dry_run)")

    # Emptied before every run: the steps hand results to each other by file
    # name, so a leftover .h5 or .RasOrb would be collected as the new one.
    if os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)
    os.makedirs(scratch_dir, exist_ok=True)

    # ── Execution ─────────────────────────────────────────────────────────
    log_path = os.path.join(output_dir, f"{project}.log")

    try:
        with open(log_path, "w") as log_fh:
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,   # stderr -> same log file
                cwd=output_dir,
            )
    except FileNotFoundError:
        raise MolcasError(
            f"OpenMolcas executable not found: '{openmolcas}'. "
            "Check paths.openmolcas in the config."
        )

    logger.debug("Return code: %d", proc.returncode)

    # ── Log analysis ──────────────────────────────────────────────────────
    success, error_msg = _parse_log(log_path)

    if proc.returncode != 0:
        success = False
        error_msg = f"pymolcas returned code {proc.returncode}. {error_msg}"

    if not success:
        logger.error("OpenMolcas reported an error in the log.")
        logger.error("Detail: %s", error_msg)
        logger.error("Full log: %s", log_path)
        return MolcasResult(success=False, log_path=log_path)

    # ── Collect the output files ──────────────────────────────────────────
    result = _collect_outputs(
        scratch_dir=scratch_dir,
        output_dir=output_dir,
        project=project,
        log_path=log_path,
        logger=logger,
    )

    if not keep_scratch:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    logger.info("OpenMolcas finished successfully.")
    logger.info("Log: %s", result.log_path)

    return result


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_env(project: str, scratch_dir: str,
               nprocs: Optional[int] = None) -> dict:
    """
    Build the environment variables OpenMolcas needs, starting from the
    current environment and overriding the specific ones.

    Note: OpenMolcas reads the project name (used to name .RasOrb / .JobIph /
    .rasscf.h5) from the "Project" variable, not from "MOLCAS_PROJECT" — it is
    set explicitly here, overriding any "Project" already in the environment
    (e.g. exported by a SLURM submit script), otherwise runs with different
    projects would all write files under the same name. Likewise
    "MOLCAS_WORKDIR" (not "MOLCAS_SCRATCH") is the real OpenMolcas working
    directory: it has to be isolated per calculation so that successive runs
    in the same job do not share or contaminate the same scratch.
    """
    env = os.environ.copy()
    env["Project"]        = project
    env["MOLCAS_PROJECT"] = project
    env["MOLCAS_WORKDIR"] = scratch_dir
    env["MOLCAS_OUTPUT"]  = scratch_dir
    if nprocs is not None:
        env["MOLCAS_NPROCS"] = str(nprocs)
    return env


def _parse_log(log_path: str) -> tuple[bool, str]:
    """
    Read the OpenMolcas log and decide whether the calculation succeeded.

    OpenMolcas prints "Happy landing" at the end when all went well. Explicit
    error patterns are looked for as well.

    Returns
    -------
    (success: bool, message: str)
    """
    happy_landing = False
    error_lines   = []
    # Matched case-sensitively: "error" occurs in ordinary OpenMolcas output
    # ("error estimate", "no errors"). STOP only at the start of a line, which
    # is how OpenMolcas emits it.
    error_patterns = (
        "MOLCAS have detected an error",
        "Fatal error",
        "Segmentation fault",
        "out of memory",
    )

    try:
        with open(log_path, "r", errors="replace") as fh:
            for line in fh:
                if HAPPY_LANDING in line:
                    happy_landing = True
                stripped = line.strip()
                if stripped.startswith("STOP") or any(p in line for p in error_patterns):
                    error_lines.append(stripped)
    except OSError as e:
        return False, f"Could not read the log: {e}"

    # The last matches, not the first: earlier ones are usually from a module
    # that recovered.
    if error_lines and not happy_landing:
        return False, " | ".join(error_lines[-3:])

    if not happy_landing:
        return False, f"The log ended without '{HAPPY_LANDING}'."

    return True, "OK"


def _collect_outputs(
    scratch_dir: str,
    output_dir:  str,
    project:     str,
    log_path:    str,
    logger:      logging.Logger,
) -> MolcasResult:
    """
    Copy from scratch_dir into output_dir only the files the later steps
    need, and build the MolcasResult.

    The whole scratch is not copied: it also holds the integrals and the
    temporary files, which are large and which no step reads.

    Files collected, and nothing else:
      - <project>.rasscf.h5         -> CI vectors, energies, AO overlap, geometry
      - <project>.RasOrb*           -> averaged (.RasOrb) and per-root (.RasOrb.N)
      - <project>.scf.h5 / .ScfOrb  -> SCF orbitals

    .JobIph is large and unread, and is left in scratch.

    Each file is copied under a .part name and renamed into place, so a run
    killed during collection leaves either the complete file or none, never a
    truncated .h5 that the next run would take for a finished step.
    """
    keep = (".rasscf.h5", ".scf.h5", ".ScfOrb")

    for fname in sorted(os.listdir(scratch_dir)):
        src = os.path.join(scratch_dir, fname)
        if not os.path.isfile(src):
            continue
        # Only this run's files; a shared scratch may hold others.
        if not fname.startswith(project):
            continue
        if not (fname.endswith(keep) or ".RasOrb" in fname):
            continue
        dst = os.path.join(output_dir, fname)
        tmp = dst + ".part"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        logger.debug("Copied: %s -> %s", src, dst)

    return MolcasResult(success=True, log_path=log_path)
