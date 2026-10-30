"""
Driver for the singlet fission Hamiltonian pipeline, after
Xie et al., JCTC 2019 (10.1021/acs.jctc.9b00122).

Five steps:

  01_mono1/  CASSCF(2,2), monomer 1       -> root-specific natural orbitals
  02_mono2/  CASSCF(2,2), monomer 2       -> root-specific natural orbitals
  03_dimer/  SCF or CASSCF(4,4), dimer    -> AO overlap, geometry, basis IDs
                                             (the MOs are not used)
  04_local/  local_orbitals.py            -> local set in the dimer AO basis,
                                             HOMO_A, LUMO_A, HOMO_B, LUMO_B
  05_casci/  CASCI(4,4), fixed orbitals   -> 20 roots, H = C^T diag(E) C,
                                             5x5 block -> H_SF.dat

Steps 01 and 02 are independent and run concurrently by default, sharing the
allocated processes; steps 03 and 05 each take the whole allocation.

The command line that drives this is cli.py; as a library, call
run_pipeline(cfg, logger, start_from).

Restarting
----------
Each file is looked for in its step directory and then in the working
directory, so a run resumes either in place or in a fresh directory holding
only the files still needed. For the latter, copy the products of the steps to
be skipped:

    01  <name>_mono1.RasOrb.<N> , <name>_mono1.rasscf.h5
    02  <name>_mono2.RasOrb.<N> , <name>_mono2.rasscf.h5
    03  <name>_dimer.scf.h5     (or <name>_dimer.rasscf.h5)
    04  <name>_local.RasOrb
    05  <name>_casci.rasscf.h5

Those steps are then skipped on their own; --from only forces a redo.
`--restart-files` prints the list for the config at hand.
"""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .config_parser import PipelineConfig, split_nprocs
from .instrumentation import banner, TaggedLogger, TimingSummary
from .layout import (STEPS, STEP_DIR, project_name, search_paths, locate,
                     local_orb_output, local_orb_input, dimer_scaffold_h5,
                     warnings_output, warnings_input, append_warnings,
                     write_stamp, step_products, step_done)
from .gen_inp       import InputBuilder
from .run_molcas    import run_molcas, MolcasResult
from .build_SF_H    import compute_sf_hamiltonian
from .local_orbitals import build_local_orbitals, format_warning_banner
from .split_dimer   import split_dimer, SplitDimerError


# ── Monomer geometries ─────────────────────────────────────────────────────

def _monomer_geometries(cfg: PipelineConfig, logger: logging.Logger) -> tuple[str, str]:
    """The two monomer .xyz files, from the config or split from the dimer.

    local_orbitals.py locates each monomer as a contiguous block of the dimer,
    so the two files must partition it without reordering the atoms.
    """
    if cfg.paths.monomer1 and cfg.paths.monomer2:
        logger.info("Monomers from the config: %s , %s",
                    cfg.paths.monomer1, cfg.paths.monomer2)
        return os.path.abspath(cfg.paths.monomer1), os.path.abspath(cfg.paths.monomer2)

    try:
        mono1, mono2 = split_dimer(geometry_path=cfg.paths.geometry,
                                   project_name=cfg.system.name)
    except SplitDimerError as e:
        logger.error("Failed to split the dimer: %s", e)
        sys.exit(1)

    logger.info("Monomer 1: %s", mono1)
    logger.info("Monomer 2: %s", mono2)
    return mono1, mono2


# ── OpenMolcas steps ───────────────────────────────────────────────────────

def _run_molcas_step(cfg: PipelineConfig, step: str, builder: InputBuilder,
                     logger: logging.Logger,
                     nprocs: int | None = None) -> MolcasResult:
    out_dir = STEP_DIR[step]
    project = project_name(cfg, step)
    os.makedirs(out_dir, exist_ok=True)
    inp = builder.write(os.path.join(out_dir, "input.inp"))
    logger.info("Input written to: %s", inp)

    result = run_molcas(input_file=inp, output_dir=out_dir,
                        openmolcas=cfg.paths.openmolcas, project=project,
                        logger=logger, dry_run=cfg.run.dry_run,
                        keep_scratch=cfg.run.keep_scratch,
                        nprocs=nprocs)

    if not result.success:
        logger.error("Step %s FAILED. Check the log: %s", step, result.log_path)
        sys.exit(1)
    if not cfg.run.dry_run:
        write_stamp(cfg, step)
    return result


def _gateway_seward(cfg: PipelineConfig, geometry: str) -> InputBuilder:
    """&GATEWAY + &SEWARD, identical in every step.

    The basis and the integral cutoff must match across steps, or the final
    CASCI runs on different integrals from the ones its orbitals came from.
    """
    builder = InputBuilder()
    builder.add_gateway(geometry_path=geometry, basis=cfg.basis.name,
                        basis_per_element=cfg.basis.per_element,
                        ricd=cfg.integrals.ricd)
    builder.add_seward(cutoff=cfg.integrals.cutoff)
    return builder


def _casscf_builder(cfg: PipelineConfig, geometry: str,
                    n_electrons: int, n_orbitals: int, n_roots: int) -> InputBuilder:
    """GATEWAY -> SEWARD -> SCF -> RASSCF, common to the first three steps."""
    builder = _gateway_seward(cfg, geometry)
    builder.add_scf()
    builder.add_rasscf(charge=cfg.system.charge, spin=cfg.system.spin,
                       n_electrons=n_electrons, n_orbitals=n_orbitals,
                       n_roots=n_roots)
    return builder


def step_monomer(cfg: PipelineConfig, step: str, geometry: str,
                 logger: logging.Logger, nprocs: int | None = None):
    builder = _casscf_builder(cfg, geometry, cfg.monomer.n_electrons,
                              cfg.monomer.n_orbitals, cfg.monomer.n_roots)
    _run_molcas_step(cfg, step, builder, logger, nprocs=nprocs)


# Steps 01, 02 and 03 each read only their own geometry and write into their
# own directory, so none of them depends on the others and all three can
# overlap. 04 consumes all three and starts the dependent chain.
INDEPENDENT_STEPS = ("01", "02", "03")

STEP_TAG = {"01": "mono1", "02": "mono2", "03": "dimer"}


def _step_title(cfg: PipelineConfig, step: str) -> str:
    """What the step does, for its banner and for the timing table."""
    if step in ("01", "02"):
        cas = f"CASSCF({cfg.monomer.n_electrons},{cfg.monomer.n_orbitals})"
        return f"{cas} on monomer {step[-1]}"
    if step == "03":
        if cfg.dimer.scaffold == "scf":
            return "SCF on the dimer (AO overlap + geometry only)"
        return (f"CASSCF({cfg.dimer.n_electrons},{cfg.dimer.n_orbitals}) "
                f"on the dimer (scaffold)")
    if step == "04":
        return "Local orbitals (Loewdin)"
    return (f"CASCI({cfg.dimer.n_electrons},{cfg.dimer.n_orbitals}), "
            f"{cfg.dimer.n_roots} roots")


def _timing_label(cfg: PipelineConfig, step: str) -> str:
    """Row name in the timing table; the parenthetical of a title is detail
    the table does not need."""
    return f"{step}  {_step_title(cfg, step).split(' (')[0]}"


HAMILTONIAN_LABEL = "H   Hamiltonian and output"


def _run_independent_step(cfg: PipelineConfig, step: str, geometries: dict,
                          logger: logging.Logger, nprocs: int | None,
                          timing: TimingSummary):
    with timing.measure(_timing_label(cfg, step)):
        if step == "03":
            step_dimer_scaffold(cfg, logger, nprocs=nprocs)
        else:
            step_monomer(cfg, step, geometries[step], logger, nprocs=nprocs)


def run_independent_steps(cfg: PipelineConfig, logger: logging.Logger,
                          steps: list[str], geometries: dict,
                          timing: TimingSummary | None = None) -> list[str]:
    """Run the pending steps among 01, 02 and 03, overlapping what is allowed.

    `parallel.concurrent_steps` decides which of them share the allocation:
    "all" overlaps the three, "monomers" only the two CASSCF runs, "none" runs
    each with the whole allocation in turn. Overlapping costs nothing in
    processes but multiplies peak memory by the number of concurrent runs.

    Returns the steps that did overlap: their durations are wall clock each,
    so they add up to more than the run took, which the timing table says.
    """
    timing = timing if timing is not None else TimingSummary()
    if not steps:
        return []

    mode = cfg.parallel.concurrent_steps
    if mode == "all":
        group = list(steps)
    elif mode == "monomers":
        group = [s for s in steps if s in ("01", "02")]
    else:
        group = []
    if len(group) < 2:
        group = []                      # nothing left to overlap

    sequential = [s for s in steps if s not in group]

    if group:
        shares = dict(zip(group, split_nprocs(cfg.parallel.nprocs, len(group))))
        detail = (f" ({'+'.join(str(shares[s]) for s in group)} processes)"
                  if cfg.parallel.nprocs else "")
        banner(logger, f"STEPS {'+'.join(group)} — concurrently{detail}")
        with ThreadPoolExecutor(max_workers=len(group)) as pool:
            futures = {}
            for step in group:
                tagged = TaggedLogger(logger, {"tag": STEP_TAG[step]})
                tagged.info("%s", _step_title(cfg, step))
                futures[step] = pool.submit(_run_independent_step, cfg, step,
                                            geometries, tagged, shares[step],
                                            timing)
            # Re-raise from any thread; another's success must not mask it.
            for step, fut in futures.items():
                fut.result()

    for step in sequential:
        banner(logger, f"STEP {step} — {_step_title(cfg, step)}")
        _run_independent_step(cfg, step, geometries, logger,
                              cfg.parallel.nprocs, timing)

    return group


def step_dimer_scaffold(cfg: PipelineConfig, logger: logging.Logger,
                        nprocs: int | None = None):
    """Produce the dimer AO overlap, geometry and basis-function IDs.

    Step 04 reads nothing else from this calculation, so GATEWAY + SEWARD + SCF
    suffices and gives identical local orbitals. dimer.scaffold = "rasscf"
    restores the full state-averaged CASSCF of the reference procedure.
    """
    if cfg.dimer.scaffold == "scf":
        builder = _gateway_seward(cfg, cfg.paths.geometry)
        builder.add_scf()
    else:
        builder = _casscf_builder(cfg, cfg.paths.geometry, cfg.dimer.n_electrons,
                                  cfg.dimer.n_orbitals, cfg.dimer.n_roots)
    _run_molcas_step(cfg, "03", builder, logger, nprocs=nprocs)


def step_local_orbitals(cfg: PipelineConfig, logger: logging.Logger) -> list[str]:
    banner(logger, f"STEP 04 — {_step_title(cfg, '04')}")

    root = cfg.monomer.natural_orbital_root
    rasorb, h5, missing = {}, {}, []
    for step in ("01", "02"):
        proj = project_name(cfg, step)
        for store, basename in ((rasorb, f"{proj}.RasOrb.{root}"),
                                (h5, f"{proj}.rasscf.h5")):
            found = locate(step, basename)
            if found is None:
                missing.append((step, basename))
            store[step] = found

    h5_dimer = dimer_scaffold_h5(cfg)
    if h5_dimer is None:
        missing.append(("03", f"{project_name(cfg, '03')}.{cfg.dimer.scaffold}.h5"))

    if missing:
        logger.error("Step 04 cannot run: %d input file(s) not found.", len(missing))
        for step, basename in missing:
            logger.error("  %s — looked in: %s", basename,
                         " , ".join(search_paths(step, basename)))
        logger.error("Copy them into the working directory (%s) to restart here.",
                     os.getcwd())
        sys.exit(1)

    out = local_orb_output(cfg)
    os.makedirs(STEP_DIR["04"], exist_ok=True)
    logger.info("Monomer orbitals: RasOrb.%d (natural orbitals of root %d)",
                root, root)
    # Name the copy actually used: more than one may exist, and the wrong
    # orbitals would not announce themselves.
    for step in ("01", "02"):
        logger.info("  monomer %s: %s", step[-1], rasorb[step])
    logger.info("  dimer scaffold: %s", h5_dimer)
    _, warnings = build_local_orbitals(rasorb["01"], h5["01"], rasorb["02"],
                                       h5["02"], h5_dimer, out)
    for w in warnings:
        logger.warning("Step 04: %s", w)

    write_stamp(cfg, "04")

    if warnings:
        # Overwrites rather than appending (unlike append_warnings): step 04
        # has just re-derived the orbitals, so its previous warnings no longer
        # describe the file on disk.
        with open(warnings_output(cfg), "w") as fh:
            fh.write("\n".join(warnings) + "\n")
    else:
        # Remove whichever copy would be read back, not only the one in the
        # step directory, or a stale warning reappears after a clean re-run.
        stale = warnings_input(cfg)
        if stale:
            os.remove(stale)

    return warnings


def step_casci(cfg: PipelineConfig, logger: logging.Logger):
    banner(logger, f"STEP 05 — {_step_title(cfg, '05')}")
    builder = _gateway_seward(cfg, cfg.paths.geometry)
    builder.add_rasscf(charge=cfg.system.charge, spin=cfg.system.spin,
                       n_electrons=cfg.dimer.n_electrons,
                       n_orbitals=cfg.dimer.n_orbitals,
                       n_roots=cfg.dimer.n_roots,
                       orbital_file=(local_orb_input(cfg)
                                     or local_orb_output(cfg)),
                       ci_only=True,
                       check_orbital_file=not cfg.run.dry_run)
    _run_molcas_step(cfg, "05", builder, logger, nprocs=cfg.parallel.nprocs)


def step_hamiltonian(cfg: PipelineConfig, logger: logging.Logger):
    banner(logger, "Singlet Fission Hamiltonian")
    basename = f"{project_name(cfg, '05')}.rasscf.h5"
    h5 = locate("05", basename)
    if h5 is None:
        logger.error("Cannot build H_SF: %s not found. Looked in: %s",
                     basename, " , ".join(search_paths("05", basename)))
        sys.exit(1)
    _, warnings = compute_sf_hamiltonian(h5, cfg.hamiltonian.output)
    logger.info("Output: %s", os.path.abspath(cfg.hamiltonian.output))
    # Persisted like the step-04 warnings and restated after the results.
    append_warnings(cfg, warnings)


# ── Driver ─────────────────────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig, logger: logging.Logger,
                 start_from: str | None):
    """Run the pipeline.

    start_from, if given, forces that step and everything after it to run
    regardless of what is on disk. Otherwise each step is skipped when its
    outputs exist and match the current configuration.
    """
    timing = TimingSummary()
    wall_start = time.perf_counter()

    forced = start_from is not None
    todo = STEPS[STEPS.index(start_from or STEPS[0]):]

    if forced:
        logger.info("--from %s: steps %s will be run whatever is already on "
                    "disk.", start_from, "+".join(todo))
        pending = list(todo)
    else:
        pending = [s for s in todo
                   if not (cfg.run.skip_existing and not cfg.run.dry_run
                           and step_done(cfg, s, logger))]
    for step in todo:
        if step in pending:
            continue
        found = (dimer_scaffold_h5(cfg) if step == "03"
                 else locate(step, step_products(cfg, step)[0]))
        logger.info("Step %s already done — skipped, reusing %s", step, found)

    # Resolved only when a monomer step will run: splitting the dimer needs
    # MDAnalysis, which a restart past step 02 should not require.
    independent = [s for s in pending if s in INDEPENDENT_STEPS]
    geometries = {"01": None, "02": None}
    if any(s in ("01", "02") for s in independent):
        geometries["01"], geometries["02"] = _monomer_geometries(cfg, logger)
    overlapped = run_independent_steps(cfg, logger, independent, geometries,
                                       timing)

    for step in pending:
        if step in INDEPENDENT_STEPS:
            continue
        if step == "04":
            if cfg.run.dry_run:
                logger.info("[DRY RUN] Step 04 skipped (it needs the .h5 files of "
                            "the first three steps).")
                continue
            with timing.measure(_timing_label(cfg, step)):
                step_local_orbitals(cfg, logger)
        elif step == "05":
            with timing.measure(_timing_label(cfg, step)):
                step_casci(cfg, logger)

    if cfg.run.dry_run:
        logger.info("[DRY RUN] Inputs generated; nothing was executed.")
        return

    with timing.measure(HAMILTONIAN_LABEL):
        step_hamiltonian(cfg, logger)

    # Before the warning banner, which stays the last thing printed.
    print(timing.format_summary(wall=time.perf_counter() - wall_start,
                                concurrent=len(overlapped) > 1), flush=True)
    _report_pending_warnings(cfg, logger)


def _report_pending_warnings(cfg: PipelineConfig, logger: logging.Logger):
    """Restate the warnings after the results, as the last thing printed.

    The orbital order decides which CSF is which diabatic state, so an error
    there is silent. Read back from disk, so the reminder survives a re-run in
    which the step that raised it was skipped.
    """
    path = warnings_input(cfg)
    if path is None:
        return
    with open(path) as fh:
        warnings = [line.strip() for line in fh if line.strip()]
    if not warnings:
        return

    for h in logger.handlers:
        h.flush()
    print(format_warning_banner(warnings, tail=f"(recorded in {path})"),
          flush=True)
