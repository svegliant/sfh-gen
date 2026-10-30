# sfh-gen

Diabatic Hamiltonians for singlet fission, from a dimer geometry.

## What this computes

How fast a molecular crystal splits one singlet excitation into two triplets
depends on the couplings between a small set of states: the excitation sitting
on either molecule (LE), an electron having moved from one to the other (CT),
and the two triplets born but still spin-correlated (TT). These are *diabatic*
states — chemically recognisable configurations, not eigenstates of anything.

No quantum chemistry program returns them. A CASCI returns adiabatic states,
which are mixtures of all of them, and the quantities you want are hidden
inside that mixture. They have to be recovered by writing the Hamiltonian in a
basis you construct yourself.

That basis is the whole problem. It is defined by molecular orbitals localised
on each monomer — HOMO and LUMO of A, HOMO and LUMO of B — and a state-averaged
CASSCF does not give you them. For a CAS(2,2) averaged over equally weighted
roots the active density is exactly half the identity, so its natural orbitals
are an arbitrary rotation of the HOMO/LUMO pair, fixed only by numerical noise.
A CASCI is invariant to that rotation, so **the adiabatic energies stay perfect
while every coupling becomes meaningless, and nothing in the output says so.**

`sfh-gen` builds the basis correctly, checks it, and returns the 5×5 matrix
over

| state | | CSF |
|---|---|---|
| LE1 | \|A\*B⟩ | `ud20` |
| LE2 | \|AB\*⟩ | `20ud` |
| CT1 | \|A⁺B⁻⟩ | `u02d` |
| CT2 | \|A⁻B⁺⟩ | `2ud0` |
| TT | \|¹(AᵀBᵀ)⟩ | `uudd` |

The procedure is that of Xie *et al.*, *J. Chem. Theory Comput.* **15**, 3721
(2019), [10.1021/acs.jctc.9b00122](https://doi.org/10.1021/acs.jctc.9b00122).
`sfh-gen` drives existing OpenMolcas modules (`gateway`, `seward`, `scf`,
`rasscf`) and modifies no OpenMolcas source.

## How it works

Five steps, run for you:

| step | directory | what it does |
|---|---|---|
| 01 | `01_mono1/` | CASSCF(2,2) on monomer 1, 3 roots |
| 02 | `02_mono2/` | CASSCF(2,2) on monomer 2, 3 roots |
| 03 | `03_dimer/` | one SCF on the dimer, for the AO overlap matrix |
| 04 | `04_local/` | localised orbitals (Python only) |
| 05 | `05_casci/` | CASCI(4,4), all 20 roots → `H_SF.dat` |

Step 04 takes each monomer's inactive and active orbitals, embeds them in the
dimer AO basis, and orthogonalises them symmetrically (Löwdin),
`C' = C S_mo^-1/2` with `S_mo = CᵀSC`. Symmetric orthogonalisation is the one
that gives the orthonormal set closest to the input and does not depend on the
order of the columns; Gram–Schmidt would leave the first monomer untouched and
push the whole distortion onto the second.

Step 05 diagonalises the CI at fixed orbitals over **all** 20 CSFs of the
CAS(4,4) singlet space. That makes the coefficient matrix **C** square and
orthogonal, so `H = Cᵀ diag(E) C` is exact and the 5×5 block can be read off by
CSF index. With fewer roots the same expression gives a rank-deficient
projection that still looks like a Hamiltonian; `sfh-gen` refuses it.

Steps 01 and 02 are independent and run at the same time.

## Installation

```bash
pip install .             # or:
bash install.sh --check   # dependencies and target environment, installs nothing
bash install.sh           # install, registers the command sfh-gen
bash install.sh --dev     # editable
```

`install.sh` wraps pip to name the environment it is about to write to, which
on a shared machine is the thing most worth seeing before it happens.

To run from a checkout without installing it, put the checkout on
`PYTHONPATH` and call the module:

```bash
PYTHONPATH=. python3 -m sfh_gen config.yaml
```

| package | needed for |
|---|---|
| `numpy`, `h5py`, `pyyaml` | everything |
| `scipy` | the Hartree-to-eV conversion, taken from `scipy.constants` rather than hand-typed |
| `MDAnalysis` (`<3`) | optional — splitting a dimer into its monomers automatically |
| `pytest` | optional — the test suite |

OpenMolcas is installed separately; `paths.openmolcas` points at its `pymolcas`.

## First run

```bash
cd somewhere_empty
sfh-gen                   # writes a commented config.yaml
# edit paths.geometry, paths.openmolcas, system.name
sfh-gen config.yaml
```

A minimal configuration is three blocks:

```yaml
paths:  {geometry: dimer.xyz, openmolcas: pymolcas}
system: {name: tetracene, charge: 0, spin: 1}
basis:  {name: 6-31G}
```

Every key is documented in [USAGE.md](USAGE.md).

### As a library

```python
from sfh_gen.config_parser import load_config
from sfh_gen.build_SF_H import compute_sf_hamiltonian

cfg = load_config("config.yaml")     # absolute paths, no chdir, no side effects
H, warnings = compute_sf_hamiltonian("05_casci/x.rasscf.h5", "H_SF.dat")
```

[examples/tetracene/run_tetracene.py](examples/tetracene/run_tetracene.py) is a
worked script.

## Which options matter

Everything else can stay at its default. Timings are
[examples/tetracene/](examples/tetracene/): 60 atoms, 6-31G, 24 cores.

| decision | key | cost | choose it when |
|---|---|---|---|
| SCF dimer scaffold | `dimer.scaffold: scf` (default) | 194 s | always |
| CASSCF dimer scaffold | `dimer.scaffold: rasscf` | 421 s | only to mirror the published procedure; the orbitals come out byte-identical |
| overlap 01, 02 and 03 | `parallel.concurrent_steps: all` (default) | — | always, if the node has the memory |
| overlap the monomers only | `monomers` | 32 s + 194 s | three concurrent runs do not fit in memory |
| no overlap | `none` | 60 s + 194 s | your site forbids nested job launches |
| give the monomer geometries | `paths.monomer1`, `monomer2` | — | preferred; no MDAnalysis needed |
| let it split the dimer | omit both | — | convenient; needs MDAnalysis |

Step 03 supplies only the AO overlap matrix, the geometry and the
basis-function numbering, so an SCF is enough. That is why it is the default
rather than the CASSCF of the original procedure.

## Reading the result

`H_SF.dat` carries enough to be interpreted years later:

```
# Singlet Fission diabatic Hamiltonian (5x5)
# produced by  : sfh-gen 0.1.0
# units        : eV
# energy zero  : eps(|2020>), the diabatic ground state
# source       : /…/05_casci/tetracene_casci.rasscf.h5
#
# row/col order: |A* B >, |A  B*>, |A+ B->, |A- B+>, |AT BT>
```

| field | meaning |
|---|---|
| `units` | always eV; the CASCI works in Hartree and converts at the end |
| `energy zero` | the diagonal is measured from ε(\|2020⟩), the dimer's local closed-shell configuration |
| `source` | the CASCI `.h5` the matrix was built from |
| `row/col order` | the five states, in the order of the rows |

The same matrix is printed at the end of the run, followed by a table of how
long each step took, so a redirected job log holds the result as well as the
run that produced it.

Diagonal entries are diabatic state energies, off-diagonal entries the
couplings between them. Signs are not physical — orbital phases are arbitrary —
so compare magnitudes. Read it back with
`numpy.loadtxt("H_SF.dat", comments="#", usecols=range(5))`.

**Why that energy zero.** ε(\|2020⟩) is the diabatic ground state and lives in
the same CSF basis as the five states, so the whole matrix stays in one
picture. The lowest adiabatic root is a mixture of CSFs and lies below every
diabatic state by however much the coupling stabilises it. Both diagonals are
printed each run, because they differ by a few tenths of an eV and published
tables do not always say which they used.

## The check that matters

Every run prints:

```
Ground-state check: weight of CSF |2020> in root 1 = 0.9543  (want > 0.95)
```

**Below 0.95, stop and look before using the couplings.** This is the only
number that detects a rotated active space, because such a rotation leaves
everything else looking correct. In the bug that motivated this package it read
0.41 while the energies looked fine and the couplings were wrong by up to a
factor of four.

The threshold is calibrated on tetracene, and a more open-shell monomer eats
into the weight before the dimer exists. A CAS(2,2) ground state is
c₁\|20⟩ + c₂\|02⟩, so its LUMO natural occupation is n_L = 2c₂², and two
uncoupled monomers give w(\|2020⟩) = (1 − n_L(A)/2)(1 − n_L(B)/2). The eight
runs in [examples/crystal_dimers/](examples/crystal_dimers/) all sit within
0.004 of that product, and the three pentacene pairs land at 0.935–0.937 with a
localisation that is demonstrably clean. So when the number is a little low,
read the step 04 Mulliken populations in the log: each active orbital must carry
more than 0.95 of its population on its own monomer. If it does, the shortfall
is the monomer's own static correlation and not a broken basis. The failure this
check exists for reads 0.4, not 0.94.

Three things must hold for the basis to mean anything, and all three are
enforced:

- **Root-specific natural orbitals**, `<project>.RasOrb.N`, never the
  state-averaged `<project>.RasOrb`. A degenerate active pair is fatal.
- **`NoMove` in `&GATEWAY`**, so the monomer and dimer frames coincide and the
  orbitals can be embedded. Always emitted.
- **`CIOnly` in the final RASSCF**, so the orbitals are not re-optimised and the
  CI vectors are not rewritten in another basis. Always emitted.

A fourth case is handled rather than enforced: OpenMolcas may write the active
pair LUMO first. That is detected from the occupations, corrected, and reported
both at the time and again after the results.

## Validation

Xie *et al.* tabulate three tetracene dimers — AB in Table 1, AC and BC in the
Supporting Information — whose couplings span an order of magnitude. All three
were run here at CASCI(4,4)/6-31G\*. Magnitudes are compared, since the signs
are conventions. Which monomer is called A is a convention too: our values are
given exactly as `H_SF.dat` holds them, and where the paper names the two
monomers the other way round, its values are relabelled to match ours.

| dimer | pair | strongest coupling | worst coupling error |
|---|---|---|---|
| AB, Table 1 | herringbone, 4.77 Å | 107 meV | 25 meV |
| BC, Table S2 | herringbone, 5.13 Å | 105 meV | 14 meV |
| AC, Table S1 | translation by 6.0565 Å | 96 meV | 10 meV |

**Every coupling reproduces, to 25 meV or better, on all three pairs.** Dimer
AB element by element, in eV, rows in the order of
[tetracene_1/H_SF.dat](examples/crystal_dimers/tetracene_1/H_SF.dat); the
paper's A and B are exchanged for this pair:

| | this work | Xie 2019 | diff |
|---|---|---|---|
| E(LE1), \|A\* B⟩ | 3.894 | 4.070 | −0.176 |
| E(LE2), \|A B\*⟩ | 4.014 | 4.144 | −0.130 |
| E(CT1), \|A⁺ B⁻⟩ | 4.963 | 4.992 | −0.029 |
| E(CT2), \|A⁻ B⁺⟩ | 4.362 | 4.509 | −0.147 |
| E(TT), \|Aᵀ Bᵀ⟩ | 4.188 | 4.065 | +0.123 |
| LE1/LE2 | 0.026 | 0.026 | +0.000 |
| LE1/CT1 | 0.100 | 0.112 | −0.012 |
| LE1/CT2 | 0.104 | 0.090 | +0.014 |
| LE2/CT1 | 0.107 | 0.092 | +0.015 |
| LE2/CT2 | 0.095 | 0.109 | −0.014 |
| LE1/TT, LE2/TT | 0.000 | 0.000 | +0.000 |
| CT1/CT2 | 0.000 | 0.000 | +0.000 |
| CT1/TT | 0.062 | 0.087 | −0.025 |
| CT2/TT | 0.088 | 0.070 | +0.018 |

The run behind this table, its log and the other two pairs are in
[examples/crystal_dimers/](examples/crystal_dimers/), where
`compare_with_reference.py` regenerates the comparison from the committed
output.

**The diagonal does not reproduce.** After a uniform shift, a spread of 0.299,
0.305 and 0.350 eV remains on the three pairs, and in each case that spread is
one single gap: the TT energy against the lowest LE state. Two explanations are
ruled out and one is not.

- *Not the energy zero.* The reference states none, but a spread is invariant
  under any choice of one. The lowest adiabatic root lies 0.267–0.283 eV below
  ε(\|2020⟩) on these three pairs, so referencing the diagonal to it instead
  raises every entry by the same amount and changes no gap.
- *Not the choice between the two uniform bases.* The reference used
  CASSCF/6-31G for the monomers and CASCI/6-31G(d) for the dimer. `sfh-gen`
  maps basis functions by index when it embeds the monomer orbitals, so one
  basis has to run throughout, and both were tried: on dimer AB the worst
  coupling error is 23 meV with 6-31G and 25 meV with 6-31G\*, and the diagonal
  spread 0.313 against 0.299 eV.
- *Unresolved: the geometry, and the mixed-basis scheme itself.* Both
  calculations start from experimental crystal structures (the reference's
  refs. 55 and 56; its DFTB work, ref. 51, supplies phonons, not geometries),
  so this is a weaker candidate than it looks, but the exact structure,
  temperature and hydrogen treatment have not been matched. Neither has the
  reference's two-basis combination, which `sfh-gen` cannot reproduce.

The couplings are what this tool is for, and what the comparison tests. The
diagonal is not validated, and no dynamic correlation is included here or in
the reference.

Beyond tetracene, [examples/crystal_dimers/](examples/crystal_dimers/) also
holds three pentacene pairs and two rubrene pairs. **Neither is a validation:**
the reference studies no pentacene, and the two rubrene pairs are not the one
it tabulates — their couplings come out below 10 meV against the published 86
and 168 meV, because in both the two tetracene cores, which carry the frontier
orbitals, stay 5.8–5.9 Å apart.

## Running on a cluster

```bash
sfh-gen config.yaml --nprocs $SLURM_NTASKS
```

Steps 01, 02 and 03 are mutually independent and overlap by default, splitting
the allocation three ways (12 → 4 + 4 + 4); step 05 then takes all of it.
Overlapping multiplies peak memory by three, so `concurrent_steps: monomers` or
`none` is the fallback on a tight node: three OpenMolcas runs hold three times
`MOLCAS_MEM`, and `sfh-gen` must be free to launch them, so give the batch step
one task and the whole allocation rather than one task per process.

Whether splitting beats running the monomers one after the other with
everything depends on how OpenMolcas scales on your machine — RASSCF often
scales poorly past a few processes. Measure it on the first real job.

### Restarting, including in a new scratch directory

Every file is looked for in its step directory and then in the working
directory, so a job resumes either in place or in a fresh directory holding
only the files still needed — the usual case when each job gets its own
scratch.

```bash
sfh-gen config.yaml --restart-files   # what each step needs, and what is present
```

Copy the products of the steps to be skipped in flat, together with their
`.sfh_stamp.json`, and run normally. A step is skipped only when its outputs
exist **and** the recorded settings match the current configuration, so editing
a parameter re-runs what depends on it instead of silently returning the old
answer. `--from 04` overrides that and redoes everything from step 04 on.

## Examples

| example | shows |
|---|---|
| [examples/tetracene/](examples/tetracene/) | one worked run: input, geometries, a library script, and the matrix the run must reproduce |
| [examples/crystal_dimers/](examples/crystal_dimers/) | eight finished cluster runs — the three tetracene pairs of the validation above, plus three pentacene and two rubrene pairs with no published counterpart |

## Package layout

| module | role |
|---|---|
| `run_all.py` | the driver: five steps, restart logic, CLI |
| `config_parser.py` | reads and validates the YAML, rejects unknown keys |
| `gen_inp.py` | writes the OpenMolcas input decks |
| `run_molcas.py` | runs OpenMolcas, manages scratch, collects outputs |
| `local_orbitals.py` | the localisation (step 04) and RasOrb file I/O |
| `build_SF_H.py` | the Hamiltonian, the CSF table, the 5×5 block |
| `split_dimer.py` | splits a dimer by connectivity, when asked to |
| `instrumentation.py` | the logger, the framed result blocks, the step timers |

## When something goes wrong

| message | cause |
|---|---|
| `Unknown key 'x' (did you mean …)` | a typo in `config.yaml`; the nearest real key is suggested |
| `Geometry file not found` | paths resolve against `paths.workdir`, not the shell's directory |
| `Step 04 cannot run: N input file(s) not found` | a restart is missing files; the message lists both places searched |
| `active occupations are … degenerate` | a state-averaged `RasOrb` reached step 04; use `RasOrb.N` |
| `the CASCI produced N roots for 20 CSFs` | `dimer.n_roots` is not the complete set |
| `the CI vectors are not orthonormal` | a truncated or interrupted `.h5` |
| `could not locate … as a contiguous block` | monomer and dimer frames differ, or the monomers are not contiguous in the `.xyz` |
| `spin must be 1` | the CSF table is singlet-only |

Warnings that need a human are written to `04_local/<name>_local.warnings` and
repeated after the results, so they survive a re-run in which that step was
skipped. OpenMolcas failures quote the last error lines and the path to the
full log.

## Tests

```bash
pytest tests/        # 93 tests, under a second, no OpenMolcas needed
```

They cover the logic that fails silently: the CSF index table, `H = CᵀEC`
reproducing its own eigenvalues, the refusal of rank-deficient and
non-orthonormal CI matrices, the minimal-change and order-independence
properties of the Löwdin step, the degeneracy and LUMO-first guards, and
configuration validation.

Nothing there touches OpenMolcas. The physics path is tested by running
[examples/tetracene/](examples/tetracene/) and comparing with its committed
reference matrix.

## Scope

- Singlets only. The CSF table is specific to the CAS(4,4) singlet space, and
  another multiplicity is refused when the configuration is read.
- One frontier pair per monomer: the monomer space is (2,2), the dimer (4,4).
- The two monomers must occupy contiguous blocks of the dimer geometry, in
  order. The embedding maps basis functions by index.
- No dynamic correlation, as in the reference. The couplings are what this is
  built for; the diagonal energies are CASCI quality.

Further reading: [USAGE.md](USAGE.md) for every configuration key.

## Citing and licence

The procedure is that of Xie, Santana-Bonilla, Fang, Liu, Troisi and Ma,
*J. Chem. Theory Comput.* **15**, 3721 (2019),
[10.1021/acs.jctc.9b00122](https://doi.org/10.1021/acs.jctc.9b00122). Please
cite it alongside this package.

`sfh-gen` is released under the GNU General Public License v3 or later; see
[LICENSE](LICENSE). It drives OpenMolcas as a separate program and links none
of it.
