"""
Where the pipeline's files live, and how a step decides it has already run.

This module owns the on-disk vocabulary -- which directory each step writes
into, what each one must leave behind, and where to look for it -- and the
stamp machinery that records the configuration a completed step was run with.
Nothing here invokes OpenMolcas or knows what any step computes; it depends
only on the configuration and on the filesystem.

A step is skipped when its products are present AND the stamp beside them
matches the configuration in force now. The two halves are separable: a run
restarted in a fresh directory has the products copied in without a stamp, and
is reused with a warning rather than refused.
"""

import hashlib
import json
import logging
import os

from . import __version__
from .config_parser import PipelineConfig
from .run_molcas import HAPPY_LANDING


# ── On-disk layout ─────────────────────────────────────────────────────────

STEPS = ["01", "02", "03", "04", "05"]

STEP_DIR = {
    "01": "01_mono1",
    "02": "02_mono2",
    "03": "03_dimer",
    "04": "04_local",
    "05": "05_casci",
}


def project_name(cfg: PipelineConfig, step: str) -> str:
    """OpenMolcas project name (prefix of the files it writes) for a step.

    Defined for the four OpenMolcas steps only; step 04 runs no OpenMolcas and
    has no project name.
    """
    return {
        "01": f"{cfg.system.name}_mono1",
        "02": f"{cfg.system.name}_mono2",
        "03": f"{cfg.system.name}_dimer",
        "05": f"{cfg.system.name}_casci",
    }[step]


# ── Finding the files a step needs ─────────────────────────────────────────
#
# A step writes into its own directory, but that layout survives only a restart
# in place. A restart on a cluster happens in a new scratch directory, where
# only the files the remaining steps read are copied over. Each file is
# therefore looked for in the step directory first, then in the working
# directory.

def search_paths(step: str, basename: str) -> list[str]:
    """Where a file belonging to `step` may be found, in order of preference."""
    return [os.path.join(STEP_DIR[step], basename), basename]


def locate(step: str, basename: str) -> str | None:
    """First existing copy of `basename`, or None if there is none."""
    for path in search_paths(step, basename):
        if os.path.isfile(path):
            return path
    return None


def local_orb_output(cfg: PipelineConfig) -> str:
    """Where step 04 writes the local orbitals."""
    return os.path.join(STEP_DIR["04"], f"{cfg.system.name}_local.RasOrb")


def local_orb_input(cfg: PipelineConfig) -> str | None:
    """Where step 05 reads them from -- possibly a copy made for a restart."""
    return locate("04", f"{cfg.system.name}_local.RasOrb")


def _scaffold_suffixes(cfg: PipelineConfig) -> tuple[str, str]:
    """The two step-03 file suffixes, the configured one first."""
    preferred = "scf" if cfg.dimer.scaffold == "scf" else "rasscf"
    return (preferred, "rasscf" if preferred == "scf" else "scf")


def dimer_scaffold_h5(cfg: PipelineConfig) -> str | None:
    """The step-03 file supplying the AO overlap, geometry and basis IDs.

    An .scf.h5 and a .rasscf.h5 of the same dimer carry identical copies of all
    three, so either is accepted whatever dimer.scaffold asked for.
    """
    proj = project_name(cfg, "03")
    for suffix in _scaffold_suffixes(cfg):
        found = locate("03", f"{proj}.{suffix}.h5")
        if found is not None:
            return found
    return None


def _warnings_name(cfg: PipelineConfig) -> str:
    """Project-scoped: a bare "warnings.txt" in the working directory could
    belong to anything."""
    return f"{cfg.system.name}_local.warnings"


def warnings_output(cfg: PipelineConfig) -> str:
    """Where the problems a run worked around are recorded. On disk, so the
    reminder survives a re-run in which that step is skipped."""
    return os.path.join(STEP_DIR["04"], _warnings_name(cfg))


def warnings_input(cfg: PipelineConfig) -> str | None:
    return locate("04", _warnings_name(cfg))


def append_warnings(cfg: PipelineConfig, new: list[str]):
    """Add to the warnings file restated at the end of the run."""
    if not new:
        return
    existing = []
    path = warnings_input(cfg)
    if path:
        with open(path) as fh:
            existing = [ln.strip() for ln in fh if ln.strip()]
    merged = existing + [w for w in new if w not in existing]
    os.makedirs(STEP_DIR["04"], exist_ok=True)
    with open(warnings_output(cfg), "w") as fh:
        fh.write("\n".join(merged) + "\n")


# ── Filesystem state ───────────────────────────────────────────────────────

def _log_has_happy_landing(log_path: str) -> bool:
    try:
        with open(log_path, "r", errors="replace") as fh:
            return any(HAPPY_LANDING in line for line in fh)
    except OSError:
        return False


# ── Settings a step was run with ───────────────────────────────────────────
#
# File names cannot say which configuration produced them, so each completed
# step writes a stamp holding the config subset it consumed. The next run
# compares it and re-runs the step if anything it depends on changed.
#
# Signatures are cumulative downstream: step 04 consumes the orbitals of 01-03
# and 05 consumes 04, so a change to the basis re-runs the whole chain.

STAMP_NAME = ".sfh_stamp.json"


def _file_digest(path: str) -> str:
    """Short content hash, or a marker if unreadable. Applied to geometry
    files, which no config field would otherwise distinguish."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        return "unreadable"


def _geometry_signature(cfg: PipelineConfig) -> dict:
    """Digests of the geometry files named in the config.

    The monomer geometries derive from the dimer when absent, so hashing what
    the config names suffices and needs neither the split nor MDAnalysis.
    """
    sig = {"dimer": _file_digest(cfg.paths.geometry)}
    for key in ("monomer1", "monomer2"):
        path = getattr(cfg.paths, key)
        if path:
            sig[key] = _file_digest(path)
    return sig


def _step_signature(cfg: PipelineConfig, step: str) -> dict:
    """The configuration a step depends on, as a comparable dict."""
    common = {
        "code_version": __version__,
        "geometry": _geometry_signature(cfg),
        "basis": {"name": cfg.basis.name, "per_element": cfg.basis.per_element},
        "integrals": {"ricd": cfg.integrals.ricd, "cutoff": cfg.integrals.cutoff},
        "system": {"charge": cfg.system.charge, "spin": cfg.system.spin,
                   "name": cfg.system.name},
    }
    monomer = {"n_electrons": cfg.monomer.n_electrons,
               "n_orbitals": cfg.monomer.n_orbitals,
               "n_roots": cfg.monomer.n_roots,
               "natural_orbital_root": cfg.monomer.natural_orbital_root}
    scaffold = {"scaffold": cfg.dimer.scaffold}
    if cfg.dimer.scaffold != "scf":
        scaffold["cas"] = [cfg.dimer.n_electrons, cfg.dimer.n_orbitals,
                           cfg.dimer.n_roots]
    casci = {"n_electrons": cfg.dimer.n_electrons,
             "n_orbitals": cfg.dimer.n_orbitals,
             "n_roots": cfg.dimer.n_roots}

    if step in ("01", "02"):
        return {**common, "monomer": monomer}
    if step == "03":
        return {**common, "dimer_scaffold": scaffold}
    if step == "04":
        return {**common, "monomer": monomer, "dimer_scaffold": scaffold}
    return {**common, "monomer": monomer, "dimer_scaffold": scaffold,
            "casci": casci}


def _stamp_output(step: str) -> str:
    return os.path.join(STEP_DIR[step], STAMP_NAME)


def write_stamp(cfg: PipelineConfig, step: str):
    """Record the settings of a completed step. Never fatal: a missing stamp
    costs a recomputation, which is the safe direction."""
    os.makedirs(STEP_DIR[step], exist_ok=True)
    try:
        with open(_stamp_output(step), "w") as fh:
            json.dump(_step_signature(cfg, step), fh, indent=2, sort_keys=True)
    except OSError as e:
        logging.getLogger("sfh_gen").warning(
            "Could not write the stamp for step %s (%s). The step will be "
            "re-run next time rather than reused.", step, e)


def _stamp_mismatch(cfg: PipelineConfig, step: str) -> list[str] | None:
    """Settings that changed since the step ran.

    [] if the stamp matches, the differing keys if not, None if there is no
    stamp -- the usual case for files copied in by hand.
    """
    path = locate(step, STAMP_NAME)
    if path is None:
        return None
    try:
        with open(path) as fh:
            saved = json.load(fh)
    except (OSError, ValueError):
        return ["<unreadable stamp>"]

    want = _step_signature(cfg, step)
    return sorted({k for k in set(saved) | set(want)
                   if saved.get(k) != want.get(k)})


def step_products(cfg: PipelineConfig, step: str) -> list[str]:
    """Basenames a step must leave behind. Copying these into the working
    directory is enough to skip it."""
    if step in ("01", "02"):
        proj = project_name(cfg, step)
        return [f"{proj}.RasOrb.{cfg.monomer.natural_orbital_root}",
                f"{proj}.rasscf.h5"]
    if step == "03":
        # Either file will do; the configured one is listed first.
        proj = project_name(cfg, "03")
        preferred, other = _scaffold_suffixes(cfg)
        return [f"{proj}.{preferred}.h5", f"{proj}.{other}.h5"]
    if step == "04":
        return [f"{cfg.system.name}_local.RasOrb"]
    return [f"{project_name(cfg, '05')}.rasscf.h5"]


def step_done(cfg: PipelineConfig, step: str,
               logger: logging.Logger | None = None) -> bool:
    """True if the step can be skipped: outputs present and settings unchanged.

    The "Happy landing" test applies only when a log is found beside the
    outputs; it rejects a run that died half way, not a hand-copied restart,
    which arrives without one. Likewise, outputs without a stamp cannot be
    attributed to a configuration and are accepted with a warning.
    """
    if step == "03":
        if dimer_scaffold_h5(cfg) is None:
            return False
    elif any(locate(step, b) is None for b in step_products(cfg, step)):
        return False

    if step != "04":
        log = locate(step, f"{project_name(cfg, step)}.log")
        if log is not None and not _log_has_happy_landing(log):
            return False

    changed = _stamp_mismatch(cfg, step)
    if changed is None:
        if logger:
            logger.warning(
                "Step %s: no %s beside its output, so the settings it was run "
                "with cannot be checked against this config. Reusing it as is.",
                step, STAMP_NAME)
        return True
    if changed:
        if logger:
            logger.info("Step %s must be re-run: %s changed since it last ran.",
                        step, ", ".join(changed))
        return False
    return True


def print_restart_files(cfg: PipelineConfig):
    """Print the restart contract for this configuration, file by file."""
    print()
    print("  Files each step must leave behind for the later ones to run.")
    print("  They are looked for in the step directory first, then in the")
    print(f"  working directory ({os.getcwd()}).")
    print()
    for step in STEPS:
        print(f"  {STEP_DIR[step]}/")
        if step == "03":
            found = dimer_scaffold_h5(cfg)
            names = " or ".join(step_products(cfg, step))
            print(f"    [{'found  ' if found else 'MISSING'}] {names}"
                  + (f"   -> {found}" if found else ""))
        else:
            for basename in step_products(cfg, step):
                found = locate(step, basename)
                print(f"    [{'found  ' if found else 'MISSING'}] {basename}"
                      + (f"   -> {found}" if found and found != basename else ""))

        changed = _stamp_mismatch(cfg, step)
        if changed is None:
            print(f"    [       ] {STAMP_NAME} — absent; settings cannot be "
                  f"checked, outputs reused as is")
        elif changed:
            print(f"    [stale  ] {STAMP_NAME} — {', '.join(changed)} changed "
                  f"since this step ran")
        else:
            print(f"    [found  ] {STAMP_NAME} — settings match this config")

        if not cfg.run.skip_existing:
            print(f"    => step {step} will be RE-RUN anyway: "
                  f"run.skip_existing is false")
        else:
            print(f"    => step {step} "
                  + ("is complete and would be skipped"
                     if step_done(cfg, step) else "would be run"))
        print()
    print("  Copy the files of the steps you want to skip into the working")
    print("  directory, then run again. Either dimer scaffold file is accepted")
    print("  for step 03: both carry the AO overlap, the geometry and the")
    print("  basis-function IDs, which is all step 04 reads from it.")
    print()
