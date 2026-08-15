# Descriptor variable lists

Exact membership of every representation used in the benchmark, so that the reduction from 272 columns to 89 and then to 56 can be checked rather than taken on trust.

| File | Contents |
| --- | --- |
| `full_272_variables.csv` | all 272 columns with their block |
| `core_89_variables.csv` | the 89 simulation-derived columns, with block, range, and whether each survives into the compact subset |
| `compact_56_variables.csv` | the 56 purely trajectory-derived columns |
| `representation_summary.csv` | one row per representation |

## How the 56-column subset is defined

89 minus 30 constant columns, minus 2 array-shape artefacts, minus the docking score. Every criterion is a property of the feature or its provenance. None uses the outcome, so the subset needs no correction for selection.

## Blocks of the simulation-derived core

| Block | Columns | Constant | In the 56 |
| --- | --- | --- | --- |
| docking score | 1 | 0 | 0 |
| interaction energy decomposition | 30 | 21 | 9 |
| ligand conformational motion | 6 | 1 | 3 |
| per-residue interaction | 20 | 8 | 12 |
| structural stability | 32 | 0 | 32 |

The eight per-residue descriptors that are constant are ASN, CYS, GLN, GLY, HIS, PRO, SER and THR. They are exactly the residues reachable only by the three interaction types that the extraction code never evaluated; see `docs/RESIDUE_BLOCK_AUDIT.md`.
