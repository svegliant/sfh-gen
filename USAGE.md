# Using `sfh-gen`

What it computes and why it is right is in [README.md](README.md). This is how
to run it.

## 1. Install

```bash
bash install.sh --check   # check dependencies, install nothing
bash install.sh           # install
bash install.sh --dev     # editable install
```

`pip install .` does the same thing. The script is a wrapper that names the
environment it is about to write to — pip does not, and on a cluster the
interpreter first on `PATH` is often not the one you meant — and reports
missing dependencies before anything is written. It registers the command
`sfh-gen`.

To run from the source tree without installing anything:

```bash
PYTHONPATH=. python3 -m sfh_gen config.yaml
```

Uninstall with `bash install.sh --uninstall` or `pip uninstall sfh-gen`.

> `pip show` and `pip list` report the *source tree* when run from inside one
> that has an `egg-info` directory. Check an installation from somewhere else,
> or with `which sfh-gen`.

## 2. Command line

```bash
sfh-gen                           # empty dir: writes a commented config.yaml
sfh-gen config.yaml               # run
sfh-gen config.yaml --from 04     # redo step 04 onward, ignoring what is on disk
sfh-gen config.yaml --restart-files   # what each step needs, and what is in place
sfh-gen config.yaml --nprocs 12   # total MPI processes for this job
sfh-gen --version
```

| flag | effect |
|---|---|
| `--from {01..05}` | that step and all later ones run regardless of disk state |
| `--restart-files` | report the restart contract; changes nothing |
| `--nprocs N` | overrides `parallel.nprocs`; use `$SLURM_NTASKS` in a batch script |
| `--version` | version, for the record in a log |

### The input file

Run `sfh-gen` in an empty directory and it writes a commented `config.yaml`.
Every key:

```yaml
paths:
  geometry:   "dimer.xyz"      # required
  monomer1:   null             # optional: both or neither. If absent the
  monomer2:   null             # dimer is split by connectivity (MDAnalysis)
  openmolcas: "pymolcas"       # required: the executable, in PATH or absolute
  workdir:    "."              # everything below resolves against this

system:
  name:   "dimer"              # prefix of every file written
  charge: 0
  spin:   1                    # must be 1

basis:
  name: "6-31G"                # or, as an alternative, not an override:
  # per_element: {C: "6-31G*", H: "6-31G"}

integrals:
  ricd:   true                 # atomic Cholesky of the two-electron integrals
  cutoff: 1.0e-20              # &SEWARD CutOff, the same in every step

monomer:
  n_electrons: 2               # must be (2,2)
  n_orbitals:  2
  n_roots:     3               # state-averaged, which relaxes the LUMO
  natural_orbital_root: 1      # reads <project>.RasOrb.N

dimer:
  n_electrons: 4               # must be (4,4)
  n_orbitals:  4
  n_roots:     null            # null -> every CSF (20 here)
  scaffold:    "scf"           # or "rasscf"

hamiltonian:
  output: "H_SF.dat"

parallel:
  nprocs: null                 # MOLCAS_NPROCS; null leaves it unset
  concurrent_steps: "all"      # all | monomers | none

run:
  skip_existing: true
  verbose:       true
  dry_run:       false         # generate the inputs, run nothing
  keep_scratch:  false         # scratch is GB per dimer
```

Paths are resolved against `paths.workdir`, and `workdir` against the directory
holding the config file. Loading a config has no effect on the process; the
working directory is entered by the CLI, not by the parser, so the package is
usable as a library.

Unknown keys and unknown sections are refused, with a suggestion:

```
[CONFIG ERROR] [monomer] Unknown key 'natural_orbital_rooot'
(did you mean 'natural_orbital_root'?).
```

## 3. When a step is skipped

A step is skipped when **both** its outputs are present **and** they were
produced with the configuration in force now. Each completed step writes a
`.sfh_stamp.json` beside its outputs holding the config subset it consumed; the
next run compares and re-runs what changed, naming it:

```
Step 01 must be re-run: basis, integrals changed since it last ran.
```

Signatures are cumulative downstream, so the propagation is not all-or-nothing:

| edit | re-runs |
|---|---|
| `basis`, `integrals.*`, `system.charge` | 01 02 03 04 05 |
| `monomer.n_roots`, `natural_orbital_root` | 01 02 04 05 |
| `dimer.scaffold` | 03 04 05 |
| the `.xyz` file, same config | 01 02 03 04 05 |

The geometry is caught because its contents are hashed into the stamp, which no
config field would otherwise detect.

Outputs with **no** stamp — copied in by hand, or from an older version — are
reused with a warning that their settings could not be checked.

`--from N` is the override: step N and everything after it run regardless.

## 4. Restarting, and restarting on a cluster

Every file is looked for in its step directory first and in the working
directory second. A run therefore resumes either in place or in a brand-new
directory containing only the files still needed — the normal case on a cluster,
where each job gets its own scratch directory and the numbered subdirectories of
the previous job are not there.

To resume elsewhere, copy the products of the steps to be skipped in flat:

| step | copy across |
|---|---|
| 01 | `<name>_mono1.RasOrb.<N>`, `<name>_mono1.rasscf.h5` |
| 02 | `<name>_mono2.RasOrb.<N>`, `<name>_mono2.rasscf.h5` |
| 03 | `<name>_dimer.scf.h5` *or* `<name>_dimer.rasscf.h5` |
| 04 | `<name>_local.RasOrb` |
| 05 | `<name>_casci.rasscf.h5` |

```bash
sfh-gen config.yaml --restart-files   # prints this list, and what is in place
```

Copy each step's `.sfh_stamp.json` too, to keep the settings check working.
Either scaffold file is accepted for step 03 whatever `dimer.scaffold` says:
step 04 reads only the AO overlap, the geometry and the basis-function IDs, and
both files carry identical copies of all three.

## 5. Parallelism

```yaml
parallel:
  nprocs: null
  concurrent_steps: "all"
```

Steps 01, 02 and 03 each read only their own geometry, so none depends on the
others and all three run at the same time, sharing the allocation. Step 05 then
takes all of it. With `nprocs` unset they still run as three concurrent serial
processes, which is the whole gain on a serial OpenMolcas build.

| `concurrent_steps` | overlaps | `--nprocs 12` gives |
|---|---|---|
| `all` (default) | 01, 02, 03 | 4 + 4 + 4 |
| `monomers` | 01, 02 | 6 + 6, then 12 for 03 |
| `none` | nothing | 12 each, in turn |

Leftover processes go to the earlier steps, so 13 splits 5 + 4 + 4. A total
smaller than the number of concurrent runs cannot be divided, and each gets 1.

**Overlapping costs nothing in processes but multiplies peak memory**: three
concurrent runs hold three times `MOLCAS_MEM`, and the dimer is the largest of
the three. Drop to `monomers` or `none` if the node cannot take it, or where a
site forbids nested launches.

> Whether 6+6 concurrent beats 12+12 sequential depends on how OpenMolcas scales
> on your machine. RASSCF often scales poorly past a few processes, in which
> case the split wins twice over. Measure it on the first real job.

## 6. Recipes

**First run on a new dimer.** Put the `.xyz` in an empty directory, run
`sfh-gen` with no arguments to get a config, edit `paths` and `system.name`, run
it. Check the `|2020⟩` weight before reading anything else.

**Check the inputs without computing.** `run.dry_run: true` writes all five
OpenMolcas decks and runs nothing. Nothing on disk is modified, including
existing scratch directories.

**Reproduce the reference procedure exactly.** `dimer.scaffold: rasscf`. Four
times the cost for identical local orbitals, but it is what a referee may ask
for.

**Use the package as a library.**

```python
from sfh_gen.config_parser import load_config
from sfh_gen.build_SF_H import compute_sf_hamiltonian

cfg = load_config("config.yaml")          # no chdir, no side effects
H, warnings = compute_sf_hamiltonian("05_casci/x.rasscf.h5", "H_SF.dat")
```

**Build local orbitals on their own**, from files produced elsewhere:

```bash
python -m sfh_gen.local_orbitals m1.RasOrb.1 m1.rasscf.h5 \
                                 m2.RasOrb.1 m2.rasscf.h5 \
                                 dimer.scf.h5 -o local.RasOrb
```

## 7. When it stops

Library code raises (`ConfigError`, `SFHamiltonianError`, `LocalOrbitalsError`,
`MolcasError`, `SplitDimerError`); only `main()` turns an error into an exit
code. Common stops:

| message | cause |
|---|---|
| `active occupations are … degenerate` | a state-averaged `RasOrb` was used; use `RasOrb.N` |
| `the CASCI produced N roots for 20 CSFs` | `dimer.n_roots` is not the full set |
| `the CI vectors are not orthonormal` | truncated or interrupted `.h5` |
| `could not locate … as a contiguous block` | `NoMove` missing, or frames differ |
| `Step 04 cannot run: N input file(s) not found` | lists both places searched |
| `spin must be 1` | the CSF table is singlet-only |

Warnings that need a human are written to `04_local/<name>_local.warnings` and
restated in a banner after the results, so they survive a re-run in which the
step that raised them was skipped.

## 8. Gotchas

- **Log and results share stdout.** A cluster job redirecting both streams to
  one file keeps them in order. A run closes by printing the 5x5 matrix, the
  same one written to `H_SF.dat`, then a table of the time each step took, and
  last of all the warning banner if there is one.
- **Scratch is deleted after each successful step** (GB per dimer). Keep it with
  `run.keep_scratch: true`.
- **`run.verbose` sets this package's logger only**, not the root logger.
- **`.JobIph` is not collected.** Nothing reads it and it is large.
- **MDAnalysis is optional and capped `<3`.** It is needed only when
  `paths.monomer1/2` are absent, and only when a monomer step actually runs.

## 9. Tests

```bash
pytest tests/        # 93 tests, under a second, no OpenMolcas
```

They cover the logic whose failure is silent: the CSF index table,
`H = CᵀEC` reproducing its own eigenvalues, the rank-deficient and
non-orthonormal refusals, Löwdin's minimal-change and order-independence
properties, the degeneracy and LUMO-first guards, and config validation.

Anything needing OpenMolcas is run by hand; `examples/tetracene/` is the
reference case and `H_SF.dat.reference` the matrix it must reproduce.

## 10. Module map

| module | holds |
|---|---|
| `run_all.py` | the driver, the five steps, restart logic, CLI |
| `config_parser.py` | `load_config`, the dataclass sections, `n_csf`, `monomer_nprocs` |
| `gen_inp.py` | `InputBuilder` — the OpenMolcas decks |
| `run_molcas.py` | `run_molcas`, scratch handling, output collection |
| `local_orbitals.py` | `vec_lowdin`, `build_local_orbitals`, RasOrb I/O |
| `build_SF_H.py` | `build_hamiltonian`, `compute_sf_hamiltonian`, the CSF table |
| `split_dimer.py` | `split_dimer` (MDAnalysis) |
| `instrumentation.py` | `setup_logger`, `banner`, `block`, `Timer`, `TimingSummary` |
