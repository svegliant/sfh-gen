# Tetracene dimer

Two tetracene molecules in the relative orientation they take in the crystal.
Tetracene is the material singlet fission was first characterised in, and this
nearest-neighbour pair is the smallest unit that can show it: one molecule
carries the excitation, and the pair has to decide whether to pass it as an
exciton, as an electron, or as two triplets.

The run answers which of those couplings actually exist. It is also the
regression test for the package — the matrix it produces is committed here, and
any change to the physics must reproduce it.

## Running it

Copy the directory somewhere else first. A run writes five step directories
plus results, and `examples/` is meant to stay clean.

```bash
cp -r examples/tetracene /scratch/tetracene && cd /scratch/tetracene
# set paths.openmolcas in config.yaml to your pymolcas
sfh-gen config.yaml                 # the command line
python3 run_tetracene.py            # the same thing through the library
```

| file | what it is |
|---|---|
| `config.yaml` | the input; monomer geometries are given explicitly, so MDAnalysis is not needed |
| `tetracene_dimer.xyz` | 60 atoms, monomer 1 then monomer 2, contiguous |
| `tetracene_mono1.xyz`, `tetracene_mono2.xyz` | the two halves |
| `run_tetracene.py` | runs the pipeline through the library, then compares with the reference element by element |
| `H_SF.dat.reference` | the committed result |

About 5.5 minutes on 24 cores: 32 s for the two monomers together, 194 s for
the dimer SCF, the rest in the final CASCI.

## What it found

Diabatic Hamiltonian in eV, diagonal referenced to ε(|2020⟩):

| | LE1 | LE2 | CT1 | CT2 | TT |
|---|---|---|---|---|---|
| **LE1** | 4.011 | | | | |
| **LE2** | 0.028 | 4.126 | | | |
| **CT1** | −0.102 | −0.108 | 4.998 | | |
| **CT2** | 0.105 | 0.097 | 0.000 | 4.388 | |
| **TT** | −0.000 | −0.000 | −0.064 | −0.091 | 4.257 |

Three groups, spanning three orders of magnitude:

| pair | magnitude |
|---|---|
| LE–CT | 97 to 108 meV |
| CT–TT | 64 and 91 meV |
| LE–LE | 28 meV |
| CT–CT | 0.1 meV |
| LE–TT | 0.05 meV |

**The LE states are not coupled to the triplet pair at all; they are coupled to
the CT states, and the CT states are coupled to the triplet pair.** Fission
therefore proceeds through the charge-transfer configurations rather than
directly, and since both CT states sit 0.3–1.0 eV above the LE states, they act
as virtual intermediates — a superexchange mechanism, not a sequential one.
That the direct LE–TT coupling comes out at 0.05 meV rather than exactly zero
is numerical noise, not a small real coupling.

## What this does not establish

The couplings above agree with Xie *et al.* (2019) Table 1 to within 23 meV.
The diagonal does not: TT is 192 meV too high and CT1 121 meV too low.

That difference changes the qualitative picture of the driving force. Here
E(TT) − E(LE1) = +0.247 eV, making fission look clearly endothermic; the
reference values give −0.005 eV, essentially isoergic, which is what tetracene
is known for.

Two explanations have since been ruled out: the energy zero, which shifts the
whole diagonal and so cannot change a gap between two entries, and the basis,
since this same dimer at 6-31G\* leaves the couplings within 25 meV and the gap
at +0.294 eV. Both calculations start from experimental crystal structures, so
the geometry is a weaker candidate than it looks, though the exact structure,
temperature and hydrogen treatment have not been matched. The same disagreement
appears on all three published tetracene pairs — see
[../crystal_dimers/](../crystal_dimers/).

Use this case as evidence about couplings and about the code being
self-consistent. Do not quote its state energies.

## Pitfalls

- **Do not run in place.** The step directories and `H_SF.dat` would land in
  the repository.
- **`paths.openmolcas` is `pymolcas`**, which is right generically and wrong on
  most machines. Set it to your build.
- **Signs are not reproducible, magnitudes are.** Each local orbital's phase
  is arbitrary, so a diabatic state can come out with either sign and the
  Hamiltonian with it, `H'_ij = s_i s_j H_ij` with `s_i = ±1`. That leaves the
  diagonal and every magnitude untouched. `run_tetracene.py` compares on that
  basis and marks sign-flipped elements with `*`; a run whose phases differ is
  not a failure. Splitting the dimer automatically rather than naming the
  monomer files is enough to flip them.
- **Do not regenerate `H_SF.dat.reference`** to make a failing comparison pass.
  It is the regression target; if the magnitudes no longer match, something
  changed.
- The monomer atoms must stay contiguous and in order in the dimer `.xyz`. The
  orbital embedding maps basis functions by index, and reordering the file
  silently breaks the correspondence.

## Output files

| file | contents |
|---|---|
| `H_SF.dat` | the 5×5 matrix, with units, energy zero and source in the header |
| `01_mono1/`, `02_mono2/` | monomer CASSCF, including the `RasOrb.1` the localisation reads |
| `03_dimer/` | the dimer SCF; only its AO overlap and geometry are used |
| `04_local/` | `tetracene_local.RasOrb`, the localised set |
| `05_casci/` | the final CASCI, whose `.h5` holds the CI vectors and root energies |
| `04_local/tetracene_local.warnings` | written only if something needs checking by hand |
