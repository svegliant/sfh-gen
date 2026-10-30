"""
Reading and validation of config.yaml.

    from .config_parser import load_config
    cfg = load_config("config.yaml")
"""

import math
import os
import yaml
from dataclasses import dataclass
from typing import Optional


# ── Dataclasses (one per section) ──────────────────────────────────────────

@dataclass
class PathsConfig:
    geometry:   str
    openmolcas: str
    workdir:    str
    monomer1:   Optional[str] = None   # if None, the dimer is split automatically
    monomer2:   Optional[str] = None


@dataclass
class SystemConfig:
    name:   str
    charge: int
    spin:   int


@dataclass
class BasisConfig:
    name:         Optional[str] = None
    per_element:  Optional[dict] = None   # alternative to `name`, not an override


@dataclass
class IntegralsConfig:
    ricd:   bool = True
    cutoff: Optional[float] = 1.0e-20


@dataclass
class MonomerConfig:
    n_electrons: int = 2
    n_orbitals:  int = 2
    n_roots:     int = 3
    natural_orbital_root: int = 1
    # Which <project>.RasOrb.N to read. Not the state-averaged
    # <project>.RasOrb: with equally weighted roots its active density is
    # degenerate and the orbitals are an arbitrary HOMO/LUMO rotation.


@dataclass
class DimerConfig:
    n_electrons: int = 4
    n_orbitals:  int = 4
    n_roots:     Optional[int] = None   # if None -> number of CSFs (Weyl formula)
    scaffold:    str = "scf"
    # Which calculation step 03 runs to supply the AO overlap matrix, the
    # geometry and the basis-function IDs. Only those are read, so "scf"
    # (GATEWAY + SEWARD + SCF) is enough and its .scf.h5 gives byte-identical
    # local orbitals. "rasscf" additionally runs the full state-averaged
    # CASSCF of the reference procedure, which costs several minutes and whose
    # orbitals are then discarded.


@dataclass
class HamiltonianConfig:
    output: str = "H_SF.dat"
    # No choice of energy zero: the diagonal is referenced to the diabatic
    # ground state eps(|2020>), so the whole matrix lies in one basis. The
    # adiabatic reference is printed for comparison only.


@dataclass
class ParallelConfig:
    nprocs: Optional[int] = None
    # Total number of MPI processes available to the job, i.e. what SLURM
    # allocated. Passed to OpenMolcas as MOLCAS_NPROCS. None leaves the
    # variable untouched, which is what a serial OpenMolcas build wants.
    concurrent_steps: str = "all"
    # Which of the independent steps overlap, sharing nprocs between them.
    #   all      01, 02 and 03 together; nprocs split three ways
    #   monomers 01 and 02 together, 03 alone with the whole allocation
    #   none     one after the other
    # Steps 01, 02 and 03 each read only their own geometry, so none of them
    # depends on the others. Overlapping them costs nothing in processes but
    # multiplies peak memory by the number of concurrent runs.


@dataclass
class RunConfig:
    skip_existing: bool = True
    verbose:       bool = True
    dry_run:       bool = False
    keep_scratch:  bool = False


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    paths:       PathsConfig
    system:      SystemConfig
    basis:       BasisConfig
    integrals:   IntegralsConfig
    monomer:     MonomerConfig
    dimer:       DimerConfig
    hamiltonian: HamiltonianConfig
    parallel:    ParallelConfig
    run:         RunConfig


# ── Validators ─────────────────────────────────────────────────────────────

class ConfigError(Exception):
    """Configuration error: missing field or invalid value."""
    pass


def _require(d: dict, key: str, section: str):
    """Raise ConfigError if the key is missing from the dictionary."""
    if key not in d:
        raise ConfigError(f"[{section}] Missing required field: '{key}'")
    return d[key]


def _as(kind, value, section: str, key: str):
    """Coerce a config value, reporting failure as a ConfigError."""
    if value is None:
        return None
    try:
        return kind(value)
    except (TypeError, ValueError):
        raise ConfigError(
            f"[{section}] '{key}' must be {kind.__name__}, "
            f"found {value!r}") from None


def _as_dict(raw, section: str) -> dict:
    """A section must be a mapping; an empty one reads as None."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"[{section}] must be a block of key: value pairs, found {raw!r}")
    return raw


def n_csf(n_electrons: int, n_orbitals: int, spin: int) -> int:
    """Number of CSFs of a CAS space, by Weyl's formula

        f(N, n, S) = (2S+1)/(n+1) C(n+1, N/2 - S) C(n+1, n - N/2 - S) .

    20 for a CAS(4,4) singlet, which is the root count the final CASCI needs.
    """
    S = (spin - 1) / 2                        # total spin
    a = n_electrons / 2 - S                   # N/2 - S, first binomial argument
    b = n_orbitals - n_electrons / 2 - S      # n - N/2 - S, second
    if a != int(a) or b != int(b) or a < 0 or b < 0:
        raise ConfigError(
            f"Active space ({n_electrons},{n_orbitals}) is incompatible with "
            f"multiplicity {spin}."
        )
    return round(spin / (n_orbitals + 1)
                 * math.comb(n_orbitals + 1, int(a))
                 * math.comb(n_orbitals + 1, int(b)))


def _validate_paths(raw: dict, config_dir: str) -> PathsConfig:
    """Validate the paths section and make every path absolute.

    Relative paths resolve against workdir, and workdir against the directory
    holding the config file. Resolved here rather than by chdir, so parsing has
    no effect on the process.
    """
    section    = "paths"
    geometry   = _require(raw, "geometry",   section)
    openmolcas = _require(raw, "openmolcas", section)
    workdir    = os.path.abspath(os.path.join(config_dir,
                                              raw.get("workdir", ".") or "."))

    if not os.path.isdir(workdir):
        raise ConfigError(f"[paths] workdir is not a directory: '{workdir}'")

    def resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(workdir, path)

    geometry = resolve(geometry)
    if not os.path.isfile(geometry):
        raise ConfigError(f"[paths] Geometry file not found: '{geometry}'")

    monomers = []
    for key in ("monomer1", "monomer2"):
        path = raw.get(key, None)
        if path is not None:
            path = resolve(path)
            if not os.path.isfile(path):
                raise ConfigError(
                    f"[paths] Monomer geometry file not found: '{path}'")
        monomers.append(path)

    if (monomers[0] is None) != (monomers[1] is None):
        raise ConfigError(
            "[paths] monomer1 and monomer2 must be given together, or neither "
            "of them (in which case the dimer is split automatically)."
        )

    return PathsConfig(geometry=geometry, openmolcas=openmolcas, workdir=workdir,
                       monomer1=monomers[0], monomer2=monomers[1])


def _validate_system(raw: dict) -> SystemConfig:
    section = "system"
    name   = str(_require(raw, "name", section))
    charge = _as(int, raw.get("charge", 0), section, "charge")
    spin   = _as(int, raw.get("spin",   1), section, "spin")

    if spin != 1:
        # The CSF table and the five state labels are specific to the CAS(4,4)
        # singlet space; another multiplicity would run to completion and give
        # a meaningless matrix.
        raise ConfigError(
            f"[system] spin must be 1, found {spin}. The SF Hamiltonian and "
            f"the CSF table it is built from are defined for the singlet space "
            f"only.")

    if not name.strip():
        raise ConfigError("[system] 'name' must not be empty: it is the prefix "
                          "of every file the pipeline writes.")

    return SystemConfig(name=name, charge=charge, spin=spin)


def _validate_basis(raw: dict) -> BasisConfig:
    """Validate the basis section.

    name and per_element are alternatives, not a default plus overrides:
    gen_inp emits one or the other.
    """
    per_element = raw.get("per_element", None)
    if per_element is not None:
        if not isinstance(per_element, dict) or not per_element:
            raise ConfigError(
                "[basis] per_element must be a non-empty mapping of element "
                "symbol to basis set, e.g. {C: '6-31G*', H: '6-31G'}.")
        if "name" in raw:
            raise ConfigError(
                "[basis] give either 'name' or 'per_element', not both: "
                "per_element replaces the uniform basis rather than overriding "
                "it, so 'name' would be silently ignored.")
        return BasisConfig(name=None, per_element=per_element)
    return BasisConfig(name=str(_require(raw, "name", "basis")), per_element=None)


def _validate_integrals(raw: dict) -> IntegralsConfig:
    cutoff = _as(float, raw.get("cutoff", 1.0e-20), "integrals", "cutoff")
    if cutoff is not None and cutoff <= 0:
        raise ConfigError(f"[integrals] cutoff must be positive, found {cutoff}")
    return IntegralsConfig(ricd=bool(raw.get("ricd", True)), cutoff=cutoff)


def _validate_monomer(raw: dict) -> MonomerConfig:
    cfg = MonomerConfig(
        n_electrons=_as(int, raw.get("n_electrons", 2), "monomer", "n_electrons"),
        n_orbitals=_as(int, raw.get("n_orbitals", 2), "monomer", "n_orbitals"),
        n_roots=_as(int, raw.get("n_roots", 3), "monomer", "n_roots"),
        natural_orbital_root=_as(int, raw.get("natural_orbital_root", 1),
                                 "monomer", "natural_orbital_root"),
    )
    if (cfg.n_electrons, cfg.n_orbitals) != (2, 2):
        raise ConfigError(
            f"[monomer] The active space must be (2,2) — one HOMO/LUMO pair per "
            f"monomer — found ({cfg.n_electrons},{cfg.n_orbitals})."
        )
    if cfg.natural_orbital_root < 1 or cfg.natural_orbital_root > cfg.n_roots:
        raise ConfigError(
            f"[monomer] natural_orbital_root = {cfg.natural_orbital_root} is "
            f"outside the range 1..{cfg.n_roots}."
        )
    return cfg


def _validate_dimer(raw: dict, spin: int) -> DimerConfig:
    cfg = DimerConfig(
        n_electrons=_as(int, raw.get("n_electrons", 4), "dimer", "n_electrons"),
        n_orbitals=_as(int, raw.get("n_orbitals", 4), "dimer", "n_orbitals"),
        n_roots=_as(int, raw.get("n_roots", None), "dimer", "n_roots"),
        scaffold=raw.get("scaffold", "scf"),
    )
    allowed = ("scf", "rasscf")
    if cfg.scaffold not in allowed:
        raise ConfigError(
            f"[dimer] scaffold '{cfg.scaffold}' not recognized. "
            f"Allowed values: {list(allowed)}."
        )
    if (cfg.n_electrons, cfg.n_orbitals) != (4, 4):
        raise ConfigError(
            f"[dimer] The active space must be (4,4) — the HOMO/LUMO of both "
            f"monomers — found ({cfg.n_electrons},{cfg.n_orbitals}). The SF "
            f"Hamiltonian is defined for that space only."
        )

    required = n_csf(cfg.n_electrons, cfg.n_orbitals, spin)
    if cfg.n_roots is None:
        cfg.n_roots = required
    elif cfg.n_roots != required:
        raise ConfigError(
            f"[dimer] n_roots = {cfg.n_roots} but the CAS"
            f"({cfg.n_electrons},{cfg.n_orbitals}) space has {required} CSFs.\n"
            f"         H = C^T diag(E) C is the Hamiltonian only if every root is "
            f"computed; otherwise C is not square and the result is a rank-"
            f"{cfg.n_roots} projection."
        )
    return cfg


def _validate_hamiltonian(raw: dict, workdir: str) -> HamiltonianConfig:
    output = raw.get("output", "H_SF.dat")
    # Resolved against workdir, like every other relative path.
    if not os.path.isabs(output):
        output = os.path.join(workdir, output)
    parent = os.path.dirname(output) or "."
    # Checked at load time rather than at the last line of a long run.
    if not os.path.isdir(parent):
        raise ConfigError(
            f"[hamiltonian] cannot write '{output}': the directory '{parent}' "
            f"does not exist.")
    if not os.access(parent, os.W_OK):
        raise ConfigError(
            f"[hamiltonian] cannot write '{output}': the directory '{parent}' "
            f"is not writable.")
    return HamiltonianConfig(output=output)


def _validate_parallel(raw: dict) -> ParallelConfig:
    nprocs = raw.get("nprocs", None)
    if nprocs is not None:
        nprocs = int(nprocs)
        if nprocs < 1:
            raise ConfigError(f"[parallel] nprocs must be >= 1, found: {nprocs}")
    allowed = ("all", "monomers", "none")
    concurrent = str(raw.get("concurrent_steps", "all"))
    if concurrent not in allowed:
        raise ConfigError(
            f"[parallel] concurrent_steps '{concurrent}' not recognized. "
            f"Allowed values: {list(allowed)}.")

    return ParallelConfig(nprocs=nprocs, concurrent_steps=concurrent)


def split_nprocs(nprocs: Optional[int], n_ways: int) -> tuple:
    """Divide the job's processes between n_ways concurrent OpenMolcas runs.

    Processes left over by an uneven division go to the earlier runs. A total
    smaller than n_ways cannot be divided, so each run gets 1 and they share
    the node.
    """
    if n_ways < 1:
        raise ConfigError(f"[parallel] n_ways must be >= 1, found: {n_ways}")
    if nprocs is None:
        return (None,) * n_ways
    if nprocs <= n_ways:
        return (1,) * n_ways
    base, extra = divmod(nprocs, n_ways)
    return tuple(base + (1 if i < extra else 0) for i in range(n_ways))


def _validate_run(raw: dict) -> RunConfig:
    return RunConfig(
        skip_existing=bool(raw.get("skip_existing", True)),
        verbose=bool(raw.get("verbose", True)),
        dry_run=bool(raw.get("dry_run", False)),
        keep_scratch=bool(raw.get("keep_scratch", False)),
    )


# ── Public entry point ─────────────────────────────────────────────────────

# Every key the config may contain. Anything else is a typo and is refused.
_SCHEMA = {
    "paths":       {"geometry", "monomer1", "monomer2", "openmolcas", "workdir"},
    "system":      {"name", "charge", "spin"},
    "basis":       {"name", "per_element"},
    "integrals":   {"ricd", "cutoff"},
    "monomer":     {"n_electrons", "n_orbitals", "n_roots", "natural_orbital_root"},
    "dimer":       {"n_electrons", "n_orbitals", "n_roots", "scaffold"},
    "hamiltonian": {"output"},
    "parallel":    {"nprocs", "concurrent_steps"},
    "run":         {"skip_existing", "verbose", "dry_run", "keep_scratch"},
}


def _suggest(word: str, options) -> str:
    """' (did you mean X?)' when a near match exists, else ''."""
    import difflib
    close = difflib.get_close_matches(word, sorted(options), n=1, cutoff=0.7)
    return f" (did you mean '{close[0]}'?)" if close else ""


def _check_keys(raw: dict):
    """Reject unknown sections and unknown keys within a section."""
    unknown = set(raw) - set(_SCHEMA)
    if unknown:
        bad = sorted(unknown)[0]
        raise ConfigError(
            f"Unknown configuration section '{bad}'{_suggest(bad, _SCHEMA)}. "
            f"Known sections: {', '.join(sorted(_SCHEMA))}.")
    for section, allowed in _SCHEMA.items():
        body = raw.get(section)
        if not isinstance(body, dict):
            continue
        extra = set(body) - allowed
        if extra:
            bad = sorted(extra)[0]
            raise ConfigError(
                f"[{section}] Unknown key '{bad}'{_suggest(bad, allowed)}. "
                f"Known keys: {', '.join(sorted(allowed))}.")


def load_config(config_path: str) -> PipelineConfig:
    """
    Read config.yaml, validate every section and return a ready-to-use
    PipelineConfig object.

    Has no effect on the process; in particular it does not chdir. Every path
    on the returned object is absolute.

    Raises ConfigError if a field is missing or invalid, FileNotFoundError if
    config_path does not exist.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: '{config_path}'")

    try:
        with open(config_path, "r") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        raise ConfigError(f"'{config_path}' is not valid YAML:\n{e}") from None
    except OSError as e:
        raise ConfigError(f"Cannot read '{config_path}': {e}") from None

    if raw is None:
        raise ConfigError("The configuration file is empty.")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"'{config_path}' must contain a mapping of sections, found "
            f"{type(raw).__name__}.")

    _check_keys(raw)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    system = _validate_system(_as_dict(raw.get("system"), "system"))
    paths  = _validate_paths(_as_dict(raw.get("paths"), "paths"), config_dir)

    return PipelineConfig(
        paths=       paths,
        system=      system,
        basis=       _validate_basis(      _as_dict(raw.get("basis"), "basis")),
        integrals=   _validate_integrals(  _as_dict(raw.get("integrals"), "integrals")),
        monomer=     _validate_monomer(    _as_dict(raw.get("monomer"), "monomer")),
        dimer=       _validate_dimer(      _as_dict(raw.get("dimer"), "dimer"),
                                           system.spin),
        hamiltonian= _validate_hamiltonian(_as_dict(raw.get("hamiltonian"), "hamiltonian"),
                                           paths.workdir),
        parallel=    _validate_parallel(   _as_dict(raw.get("parallel"), "parallel")),
        run=         _validate_run(        _as_dict(raw.get("run"), "run")),
    )


def print_config_summary(cfg: PipelineConfig):
    """Print a readable summary of the loaded configuration."""
    sep = "─" * 60
    print(sep)
    print("  Pipeline Configuration Summary")
    print(sep)
    print(f"  System         : {cfg.system.name}  (charge {cfg.system.charge}, spin {cfg.system.spin})")
    print(f"  Geometry       : {cfg.paths.geometry}")
    if cfg.paths.monomer1:
        print(f"  Monomers       : {cfg.paths.monomer1} , {cfg.paths.monomer2}")
    else:
        print("  Monomers       : split automatically from the dimer")
    print(f"  OpenMolcas     : {cfg.paths.openmolcas}")
    if cfg.basis.per_element:
        detail = ", ".join(f"{el}: {b}" for el, b in cfg.basis.per_element.items())
        print(f"  Basis set      : per element ({detail})")
    else:
        print(f"  Basis set      : {cfg.basis.name}")
    cutoff = "—" if cfg.integrals.cutoff is None else f"{cfg.integrals.cutoff:.1e}"
    print(f"  Integrals      : RICD={cfg.integrals.ricd}  CutOff={cutoff}")
    print(f"  Monomer        : CAS({cfg.monomer.n_electrons},{cfg.monomer.n_orbitals}) "
          f"-> {cfg.monomer.n_roots} roots, orbitals from RasOrb.{cfg.monomer.natural_orbital_root}")
    print(f"  Dimer (CASCI)  : CAS({cfg.dimer.n_electrons},{cfg.dimer.n_orbitals}) "
          f"-> {cfg.dimer.n_roots} roots")
    print(f"  Step 03        : {cfg.dimer.scaffold}"
          + ("  (AO overlap + geometry only)" if cfg.dimer.scaffold == "scf" else ""))
    print(f"  H_SF output    : {cfg.hamiltonian.output}  (diabatic, zero = eps(|2020>))")
    overlap = {"all": ("01", "02", "03"), "monomers": ("01", "02"),
               "none": ()}[cfg.parallel.concurrent_steps]
    if cfg.parallel.nprocs is None:
        procs = "not set (MOLCAS_NPROCS left as the environment has it)"
    else:
        procs = f"{cfg.parallel.nprocs} total"
        if overlap:
            shares = split_nprocs(cfg.parallel.nprocs, len(overlap))
            procs += ("  -> " + "+".join(overlap) + " concurrently, "
                      + "+".join(str(n) for n in shares) + " processes")
    print(f"  Processes      : {procs}")
    if overlap:
        print(f"                   steps {'+'.join(overlap)} overlap; peak memory "
              f"is {len(overlap)}x one run")
    else:
        print("                   steps 01, 02 and 03 run one after the other")
    print(f"  Dry run        : {cfg.run.dry_run}   Skip existing: {cfg.run.skip_existing}")
    print(sep)
