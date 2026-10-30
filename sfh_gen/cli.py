"""
Command line for sfh-gen: argument parsing, locating or generating the
configuration, and turning an error into an exit code.

The pipeline itself is run_all.py. Keeping the two apart means a caller using
the package as a library never inherits the CLI's decisions -- in particular
that it may chdir, write a file into the working directory, or call sys.exit.

    sfh-gen [config.yaml]
    python3 -m sfh_gen [config.yaml]     # without installing
"""

import argparse
import os
import sys
from importlib import resources

from . import __version__
from .build_SF_H import SFHamiltonianError
from .config_parser import load_config, print_config_summary, ConfigError
from .gen_inp import InputBuilderError
from .local_orbitals import LocalOrbitalsError
from .instrumentation import setup_logger
from .layout import STEPS, print_restart_files
from .run_all import run_pipeline
from .run_molcas import MolcasError
from .split_dimer import SplitDimerError

DEFAULT_CONFIG_NAME = "config.yaml"


# ── Default config ─────────────────────────────────────────────────────────

def default_config_text() -> str:
    """The commented config.yaml written into an empty working directory.

    Kept as `default_config.yaml` beside this module rather than as a string
    literal, so that it is syntax-highlighted, diffable, and checkable against
    the schema in config_parser (tests/test_pure.py does exactly that).
    """
    return resources.files(__package__).joinpath("default_config.yaml").read_text()

def _generate_default_config(path: str):
    with open(path, "w") as fh:
        fh.write(default_config_text())
    print()
    print(f"  No config found: '{path}' generated.")
    print("  Edit the fields (geometry, openmolcas, basis, ...) and run again.")
    print()


def _resolve_config(config_arg: str | None) -> str:
    if config_arg is not None:
        if not os.path.isfile(config_arg):
            print(f"Error: Configuration file not found: '{config_arg}'",
                  file=sys.stderr)
            sys.exit(1)
        return config_arg

    default = os.path.join(os.getcwd(), DEFAULT_CONFIG_NAME)
    if os.path.isfile(default):
        return default

    _generate_default_config(default)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Singlet Fission Hamiltonian pipeline — OpenMolcas"
    )
    parser.add_argument("--version", action="version",
                        version=f"sfh-gen {__version__}")
    parser.add_argument("config", nargs="?", default=None,
                        help="Path to config.yaml (default: ./config.yaml).")
    parser.add_argument("--from", dest="start_from", choices=STEPS, default=None,
                        help="Redo from this step: it and everything after it "
                             "are run again whatever is already on disk. "
                             "Earlier steps are skipped outright, so their "
                             "products must already be in the step directories "
                             "or in the working directory. Without this flag "
                             "each step is skipped only when its outputs exist "
                             "and match the current config.")
    parser.add_argument("--restart-files", action="store_true",
                        help="List the files each step must leave behind for "
                             "the later ones, and say whether each is already "
                             "in place. Copy the ones you want to reuse into "
                             "the working directory and run again.")
    parser.add_argument("--nprocs", type=int, default=None,
                        help="Total MPI processes available, overriding "
                             "parallel.nprocs in the config. In a batch script "
                             "pass the scheduler's own count, e.g. "
                             "--nprocs $SLURM_NTASKS.")
    args = parser.parse_args()

    config_path = os.path.abspath(_resolve_config(args.config))

    try:
        cfg = load_config(config_path)
    except (FileNotFoundError, ConfigError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # load_config leaves the process where it was. The chdir is process-global
    # and belongs here; step directories resolve against it.
    os.chdir(cfg.paths.workdir)

    if args.nprocs is not None:
        if args.nprocs < 1:
            print(f"Error: --nprocs must be >= 1, got {args.nprocs}", file=sys.stderr)
            sys.exit(1)
        cfg.parallel.nprocs = args.nprocs

    if args.restart_files:
        print_restart_files(cfg)
        return

    logger = setup_logger(cfg.run.verbose)
    logger.info("sfh-gen %s", __version__)
    logger.info("Configuration loaded from: %s", config_path)
    print_config_summary(cfg)

    try:
        run_pipeline(cfg, logger, args.start_from)
    except (MolcasError, InputBuilderError, SFHamiltonianError,
            LocalOrbitalsError, SplitDimerError) as e:
        logger.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted from the keyboard.")
        sys.exit(130)


