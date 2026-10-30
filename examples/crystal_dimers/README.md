# Eight crystal dimers

Three tetracene pairs, three pentacene pairs and two rubrene pairs, taken from
the crystal structures and run end to end on a cluster. Each directory holds
the configuration, the geometry, the full job log and the resulting
Hamiltonian, exactly as they came off the machine.

These are the package's evidence that it reproduces a published calculation
and its record of what happens on systems the publication does not cover. The
three tetracene pairs are the three studied by Xie *et al.*, *J. Chem. Theory
Comput.* **15**, 3721 (2019), so their matrices can be compared element by
element. Pentacene is not in that paper, and neither rubrene pair here is the
one it tabulates, so those five stand on their own.

For the single worked run with a committed reference matrix, see
[../tetracene/](../tetracene/). This directory is a result set, not a tutorial.

## Running the comparison

```bash
python3 compare_with_reference.py            # one table per dimer
python3 compare_with_reference.py --summary  # one line per dimer
```

It reads every `*/H_SF.dat` and the published tables in
`reference_xie2019.yaml`, and needs neither OpenMolcas nor a re-run.

To reproduce one of the dimers:

```bash
cd tetracene_1
sfh-gen config.yaml --nprocs 12
```

Each `config.yaml` is self-contained apart from `paths.openmolcas`, which must
point at your `pymolcas`.

## What the dimers are

Geometry alone distinguishes them. A pair related by a pure lattice
translation has two identical molecules in identical orientations, which forces
the two LE energies and the two CT energies to be equal; a herringbone pair has
neither.

| run | atoms | monomers related by | separation | closest heavy-atom contact |
|---|---|---|---|---|
| tetracene_1 | 60 | herringbone | 4.773 Å | 3.71 Å |
| tetracene_2 | 60 | herringbone | 5.125 Å | 3.58 Å |
| tetracene_3 | 60 | pure translation | 6.0565 Å | 3.69 Å |
| pentacene_1 | 72 | pure translation | 6.2753 Å | 3.63 Å |
| pentacene_2 | 72 | herringbone | 4.734 Å | 3.65 Å |
| pentacene_3 | 72 | herringbone | 5.199 Å | 3.62 Å |
| rubrene_1 | 140 | pure translation | 8.7397 Å | 3.65 Å |
| rubrene_2 | 140 | neither | 9.313 Å | 3.45 Å |

Separation is between the heavy-atom centroids, and equals the translation
where there is one.

## What it found

All eight ran to completion with the committed code, at CASCI(4,4)/6-31G\* on
12 cores:

| run | wall clock | w(\|2020⟩) | strongest coupling | published as | worst coupling error |
|---|---|---|---|---|---|
| tetracene_1 | 12 m 46 s | 0.9563 | 107 meV | dimer AB, Table 1 | 25 meV |
| tetracene_2 | 39 m 27 s | 0.9566 | 105 meV | dimer BC, Table S2 | 14 meV |
| tetracene_3 | 13 m 06 s | 0.9598 | 96 meV | dimer AC, Table S1 | 10 meV |
| pentacene_1 | 39 m 24 s | 0.9371 | 91 meV | — | — |
| pentacene_2 | 36 m 16 s | 0.9353 | 130 meV | — | — |
| pentacene_3 | 22 m 43 s | 0.9358 | 128 meV | — | — |
| rubrene_1 | 3 h 07 m | 0.9502 | 6 meV | — | — |
| rubrene_2 | 3 h 27 m | 0.9501 | 10 meV | — | — |

**The three tetracene pairs reproduce the published couplings to 25 meV or
better, across pairs whose couplings differ by an order of magnitude.** The
translational pair comes out weakly coupled (LE/CT 4 and 19 meV against 9 and
14 meV published) and the two herringbone pairs strongly coupled (largest
LE/CT 107 and 105 meV against 112 and 119 meV), which is the part of the
result that matters for singlet fission rates.

**The diagonal does not reproduce.** The two calculations do not share an
energy zero, and after a uniform shift a spread of 0.299, 0.305 and 0.350 eV
remains. In all three pairs that spread is one single gap: the TT energy
against the lowest LE state. No dynamic correlation is included here or in the
reference, and the diagonal is not what this package is built to deliver.

### Why the rubrene pairs are not the published one

Table 2 of the paper gives LE/CT couplings of 86 and 168 meV for rubrene dimer
AB. Both pairs here come out at 10 meV or less, a 164 meV miss, so they are
different pairs rather than a failed reproduction. The geometry says why: the
frontier orbitals of rubrene sit on the fused tetracene core, not on the four
phenyl side chains, and in both of these pairs the **closest contact between the
two cores is 5.84 Å** (rubrene_2) and **5.89 Å** (rubrene_1). The 3.4–3.6 Å
contacts in the table above are phenyl to phenyl. Neither the paper nor its SI
tabulates a Hamiltonian for any rubrene pair but AB.

### Pentacene and the ground-state check

All three pentacene runs print the warning

```
Ground-state check: weight of CSF |2020> in root 1 = 0.9353  (want > 0.95)
  WARNING: ... The active orbitals are probably not localized HOMO/LUMO pairs
```

**On these runs that diagnosis is wrong, and the results are usable.** The
localisation is clean: in all three runs the log reports Mulliken populations
of 1.0001 or 1.0002 for each active orbital on its own fragment, −0.0001 or
−0.0002 on the other, and `max|CᵀSC − I|` between 3.2 and 4.8 × 10⁻¹³.

What the number actually measures is partly set before the dimer exists. A
CAS(2,2) monomer ground state is c₁\|20⟩ + c₂\|02⟩, so its LUMO natural
occupation is n_L = 2c₂², and two uncoupled monomers give

    w(|2020⟩) = (1 − n_L(A)/2)(1 − n_L(B)/2) .

Every run here sits within 0.004 of that product:

| run | n_L(A) | n_L(B) | predicted | observed |
|---|---|---|---|---|
| tetracene_3 | 0.0405 | 0.0405 | 0.9599 | 0.9598 |
| tetracene_1 | 0.0446 | 0.0405 | 0.9579 | 0.9563 |
| rubrene_1 | 0.0503 | 0.0503 | 0.9504 | 0.9502 |
| pentacene_2 | 0.0633 | 0.0617 | 0.9385 | 0.9353 |
| pentacene_1 | 0.0633 | 0.0633 | 0.9377 | 0.9371 |

Pentacene's smaller HOMO–LUMO gap costs it 0.024 in weight before the two
molecules are put together, which is most of its distance below the threshold.
The 0.95 cut is calibrated on tetracene and is a reliable alarm there; on a
more open-shell monomer, check the Mulliken fragment populations in the step 04
log before discarding a run. The failure it was built to catch read 0.41, not
0.94.



## Files in each directory

| file | what it is |
|---|---|
| `config.yaml` | the configuration the run used, verbatim |
| `dimer_N.xyz` | the dimer geometry; the monomers were split from it by connectivity |
| `run.log` | the whole job log: configuration summary, the five steps, diagnostics, the matrix, the timing table |
| `H_SF.dat` | the 5×5 diabatic Hamiltonian in eV, zero at ε(\|2020⟩) |
