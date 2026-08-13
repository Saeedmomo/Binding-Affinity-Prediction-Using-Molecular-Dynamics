# Ligand–Protein Interaction Modelling Using Molecular Dynamics and Hybrid Machine Learning

Code, data and results for the prediction of pIC50 from molecular dynamics simulation
descriptors, benchmarked against conventional two-dimensional descriptors, pose-derived
three-dimensional descriptors and quantum-chemical descriptors on identical molecules,
partitions and metrics.

Targets: ERα (ESR1, PDB 1SJ0), ERK2 (MAPK1, 8AOJ), p53 Y220C (TP53, 6GGB) and TDP1
(6N0D). 122 ligand–protein systems.

---

## Headline result

Hybrid model, test-set R², both partitioning regimes:

| Descriptor set | Descriptors | Random split | Structure-disjoint split |
|---|---|---|---|
| **Molecular dynamics core** | **89** | **0.655** | **0.795** |
| MD ⊕ PaDEL | 1549 | 0.534 | 0.616 |
| MD + AlphaFold + MACCS | 272 | 0.622 | 0.566 |
| PaDEL 2D/3D | 1444 | 0.502 | 0.559 |
| MACCS only | 167 | 0.417 | 0.473 |
| AlphaFold only | 16 | 0.391 | 0.411 |
| DFT quantum + MD | 381 | 0.500 | 0.145 |
| DFT quantum | 109 | 0.060 | −0.574 |
| Pose-derived 3D | 62571 | 0.030 | −1.120 |

Simulation-derived descriptors outperform every alternative tested. The 89-descriptor
molecular dynamics core is the strongest single block, and performs best under the
stricter structure-disjoint partition, approaching the theoretical ceiling of R² = 0.887
imposed by the dataset's repeated structures.

---

## Repository layout

```
src/        analysis code (see "Reproducing the results")
data/
  inputs/   the descriptor matrices the models are trained on
  dft/      quantum-chemistry geometries, per-molecule outputs and descriptors
  ligands_122_*.csv   the join key: identifiers, SMILES, pIC50, target
results/    every metric, per-molecule prediction, and a run manifest
tables/     publication tables (CSV and DOCX)
figures/    publication figures (PNG and PDF, 600 dpi)
docs/       verified record of the data, and the corrections applied
```

`docs/GROUND_TRUTH.md` documents the provenance of every input file and the checks
performed on it. Read it before modifying anything.

---

## Reproducing the results

```bash
pip install -r requirements.txt

python src/validate.py                 # reproduces the published Nu-SVR result
python src/run_benchmarks.py --datasets md272,md272_clean,padel,fused --regimes random,structure_disjoint
python src/run_benchmarks.py --datasets mol2desc --regimes random,structure_disjoint
python src/run_benchmarks.py --datasets md_core89,maccs167,alphafold16 --regimes random,structure_disjoint
python src/rerun_dnn_variants.py       # the four deep-network architectures, 5 seeds each
python src/shap_md_core.py             # SHAP attribution of the MD core
python src/paired_tests.py             # paired Wilcoxon comparisons between descriptor sets
python src/make_tables.py && python src/make_figures.py
```

Quantum-chemical descriptors are generated on Linux (or WSL) and require PySCF and xtb:

```bash
python src/dft_01_export_geometries.py
THREADS_PER_WORKER=4 python src/dft_02_run_qc.py --workers 5
python src/dft_02b_solvation.py
python src/dft_03_assemble.py
```

`results/run_manifest.json` records package versions, random seeds, resolved
hyperparameters and a SHA-256 prefix for every input file.

---

## Methods

**Models.** Nu-SVR (RBF kernel, hyperparameters by grid search over nu, C and kernel,
principal components swept over 15 to 22); a deep neural network; and a Ridge
meta-learner stacking the two. Standardisation and principal-component analysis are
fitted inside cross-validation folds. Partitions are 70:20:10 train/test/holdout.

**Two partitioning regimes.** A random split, and a structure-disjoint split grouped on
canonical SMILES. The latter is necessary because the 122 rows contain only 94 distinct
chemical structures: the same ligand appears against several targets with different
pIC50, so a random split leaks structures between training and test.

**Validation.** Bootstrap 95 % confidence intervals on R² (1000 resamples), y-scrambling
against 100 label permutations, leave-one-out Q², and paired Wilcoxon signed-rank tests
on per-molecule absolute errors with Holm correction.

**Quantum chemistry.** GFN2-xTB geometry optimisation from the docked pose, then
B3LYP/def2-SVP single-point energies in PySCF with the matching effective core potential
for iodine. Frontier orbitals, HOMO-LUMO gap, dipole, traceless quadrupole, Mulliken and
meta-Löwdin charges, conceptual-DFT reactivity indices, and an ALPB aqueous solvation
free energy. All 122 calculations converged.

---

## Notes for anyone rebuilding this

Three implementation details cost significant time and are worth knowing:

- **`NuSVR` has no iteration limit by default.** Grid points at C = 100 with the
  polynomial and sigmoid kernels do not converge on this data and will spin indefinitely.
  `max_iter` is capped in `src/hybrid_pipeline.py`.
- **PySCF must be limited to a few threads per process**, set before import. Unpinned it
  took 295 s for a 264-basis-function molecule against 48 s on four threads.
  Parallelise across molecules, not within one.
- **Density fitting is not used.** With several concurrent workers it spilled gigabytes
  of integral cache to disk and was slower than direct SCF at this molecule size.

---

## Citation

If you use this code or data, please cite the associated publication. Correspondence:
Said Moshawih.
