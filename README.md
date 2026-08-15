# Do descriptors of the bound complex predict ligand potency better than descriptors of the ligand?

Ligand potency is almost always predicted from descriptors of the isolated molecule.
That is a practical choice, not a physical one: potency is a property of a complex,
and a molecule that is potent against one protein may be inactive against another,
yet a ligand-only description assigns it one vector and therefore one prediction.

This repository tests whether descriptors computed from a molecular dynamics
trajectory of the **bound complex** predict potency better, on 122 ligand-protein
systems across four cancer targets, under an evaluation strict enough for the answer
to mean something.

## The comparison

| representation | descriptors | median held-out R2 |
| --- | ---: | ---: |
| Simulation derived | 89 | **0.781** |
| PaDEL | 1 444 | 0.579 |
| **Protein identity alone** | **4** | **0.475** |
| PyDescriptor | 62 571 | 0.250 |
| Quantum chemical | 109 | 0.149 |

Ten structure-disjoint partitions, hyperparameters chosen by grouped cross
validation inside each training partition, twelve learners in two preprocessing
variants, 960 fits. The simulation-derived set ranks first in all 24 learner
configurations.

**The four-column protein-identity baseline is the reference point that makes the
rest readable.** It knows only which of the four proteins a row belongs to and
nothing about the molecule. Two of the three comparison families score below it, so
they provide no predictive information beyond knowing the target.

## Representation reduction

Each step asks whether the previous representation carried anything the next one
does not. Verdicts come from a bootstrap interval on the median paired difference
against a margin of 0.05 fixed in advance, not from significance alone.

| step | median R2 | paired difference | interval | conclusion |
| --- | ---: | ---: | --- | --- |
| Full corrected set (272) | 0.782 | | | |
| Core plus MACCS (256) | 0.790 | +0.0146 | -0.049 to 0.037 | numerical only, no support |
| Simulation-derived core (89) | 0.781 | +0.0014 vs 272 | -0.019 to 0.043 | equivalent within the margin |
| Compact trajectory subset (56) | 0.773 | -0.0018 vs 89 | -0.075 to 0.026 | consistent with equivalence, not established |

Adding a verified structural fingerprint to the core changes nothing the data can
resolve, so the conclusion is **redundancy and parsimony, not interference**. The
compact subset differs from the core by less than two thousandths in the median while
dropping 37 per cent of the columns, which points to predictive signal being
concentrated rather than spread evenly, though ten partitions cannot establish formal
equivalence.

Descriptor count is a poor guide to quality throughout: the largest family here is
among the weakest, and four indicator columns beat two of the four families.

## Transfer to an unseen protein

The hardest test holds out a whole target, trains on the other three, and removes
from training every structure that also occurs in the held-out target. Transfer is
three separate questions with different answers.

Simulation-derived representation, target by target:

| held-out target | R2 | R2 with the offset known | rank correlation | p |
| --- | ---: | ---: | ---: | ---: |
| MAPK1 | **+0.353** | +0.478 | 0.543 | 0.002 |
| TDP1 | -1.141 | +0.178 | 0.559 | <0.001 |
| ESR1 | -3.053 | -0.101 | 0.457 | 0.019 |
| TP53 | -1.979 | -1.253 | -0.217 | 0.32 |

Absolute prediction is generally poor and strongly target dependent, with MAPK1 the
exception. Much of the failure elsewhere is **calibration**: supplying the
between-target offset returns TDP1 and ESR1 to near zero, but not TP53. **Ranking**
within the unseen target survives best.

Across representations, by median rank correlation on an unseen target:

| representation | rank correlation | targets with a significant correlation |
| --- | ---: | --- |
| Simulation derived | 0.500 | **3 of 4** |
| PaDEL | 0.343 | 0 of 4 |
| Quantum chemical | 0.149 | 0 of 4 |
| PyDescriptor | 0.023 | 0 of 4 |

Only the representation computed from the bound complex carries information that
demonstrably reaches a protein it was not trained on. Note that the bottom two
exchange places relative to the resampling ranking: a representation can rank third
on interpolation and carry nothing to a new target.

**Prioritising compounds and predicting their potencies are different problems.**
This work supports the first and not the second.

## Descriptor variable lists

`data/descriptors/` gives the exact membership of every representation, so the
reduction can be checked rather than taken on trust.

| file | contents |
| --- | --- |
| `full_272_variables.csv` | all 272 columns with their block |
| `core_89_variables.csv` | the 89 simulation-derived columns, block, range, and whether each survives into the compact subset with the reason for removal |
| `compact_56_variables.csv` | the 56 purely trajectory-derived columns |
| `representation_summary.csv` | one row per representation |

The 56-column subset is 89 minus 30 constant columns, minus 2 array-shape artefacts,
minus the docking score. **Every criterion is a property of the feature or its
provenance; none uses the outcome**, so the subset needs no correction for selection.

## Descriptor computation

The per-residue interaction block is the part of the representation with no
ligand-only counterpart. For every simulation frame, each contact between the ligand
and a protein residue is classified into one of five types, weighted by a
representative energy for that type, and accumulated by the amino-acid identity of
the residue:

```
F(r) = (1/N) * sum over frames f, sum over interaction types k of  e(k) * n(r,k,f)
```

with N the number of frames, k running over hydrogen bonds, cation pi contacts,
hydrophobic contacts, ionic interactions and water bridges, e(k) the adopted energy
in kcal per mole, and n(r,k,f) the number of distinct residues of type r making a
contact of type k in frame f. Counting distinct residues rather than atom pairs
stops one contact involving several nearby atoms from being counted several times.
Geometric criteria and adopted energies are in
`docs/INTERACTION_ENERGY_REFERENCES.md`; the extraction is
`src/feature_extraction/ligand_protein_interaction_forces.py`.

Structural keys are computed with `src/maccs_fix.py`, which reads each molecule from
its mol2 file, falls back through a documented parse chain for structures that resist
full sanitisation, and verifies that the keys are identical for every pair of rows
holding the same canonical structure before returning. The verified block is in
`results/benchmark/maccs_recomputed.csv` and is the block used in every analysis
here.

## Reproducing

```bash
pip install -r requirements.txt

python src/robust.py --datasets md_core89 padel mol2desc dft --inner group --out grouped/robust.csv
python src/ablation.py                 # the representation progression and baselines
python src/loto.py --share drop        # transfer to an unseen target
python src/ceiling.py                  # the bound imposed by recurring structures
python src/permute.py                  # y scrambling with the pipeline held fixed
python src/descriptor_lists.py         # the variable lists in data/descriptors
python src/validate_final.py           # completeness, pairing, leakage, determinism
```

`src/validate_final.py` is the gate: it checks that every cell is present exactly
once, that partitions are identical across representations, that grouped inner folds
leak zero rows, and that repeated runs reproduce exactly. Run it before believing any
number.

## Evaluation notes

- Partitions are disjoint by canonical structure at **every** level, including the
  inner folds used for hyperparameter selection. With ungrouped inner folds, 26 to 31
  validation rows per partition shared a structure with their own training fold.
- 122 rows contain only 94 distinct structures; 19 molecules were simulated against
  more than one target.
- Medians are reported throughout: 6 of 960 fits diverged below minus ten, all from
  linear and polynomial kernels, none from tree ensembles.
- Paired comparisons use the Wilcoxon signed-rank test with Holm correction, and the
  Nadeau and Bengio correction for the dependence between overlapping partitions.
- The permutation test holds the pipeline fixed at its selected hyperparameters;
  learner selection is not inside the null, which is stated rather than implied.

## Limitations

122 rows, 94 structures, four targets, one dominant chemical series. The design varies
representation and information together, since the simulation-derived set has access
to the protein and the comparison sets do not; each is evaluated as its authors
deliver it, which is the choice a practitioner faces. Claims are limited to
interpolation among these four systems and to relative ranking within a target.
