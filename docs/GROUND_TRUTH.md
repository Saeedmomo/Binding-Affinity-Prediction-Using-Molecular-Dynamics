# Ground truth for the fourth-paper benchmark work

Everything below was verified directly against the files on disk (see `src/forensics/`).
Do not re-derive it; do not contradict it without new evidence.

## 1. The study

Manuscript: *Ligand–Protein Interaction Modelling Using Molecular Dynamics and Hybrid
Machine Learning*. Predicts pIC50 for ligands against four cancer targets
(ESR1/1SJ0, MAPK1/8AOJ, TP53-Y220C/6GGB, TDP1/6N0D) from a 272-feature block
combining MD dynamics, system energies, ligand-PCA, per-residue interaction forces,
AlphaFold pLDDT/PAE confidence, and 167-bit MACCS fingerprints.

Manuscript is split across three documents in `D:\Chua files\fourth paper\`:

| File | Content |
|---|---|
| `Modelling_manuscript110425_Mod_updated230425_mod.docx` | main manuscript (intro, methods, results, refs) |
| `Hybrid model.docx` | DNN + hybrid methods/results section |
| `Interpretation of SHAP Summary Plot.docx` | PCA-SHAP interpretation + PC table |

## 2. Canonical row order

`merged_descriptors.csv['mol_name']` (alphabetical) is the canonical study order.
It is shared, row-for-row, by **all** of these:

- `Said_mol2_files\merged_descriptors.csv` / `merged_descriptors_with_pic50.csv`
- `MDS_analysis\data_model\2.csv`, `JDes_output2.csv`, `JDes_output2_imputed.csv`, `5_imputed.csv`
- `fourth paper\JDes_output3.csv` (its `Name` column is generic `Mol_1 … Mol_122`)

The PIC50 vector is byte-identical across `2.csv`, `JDes_output2.csv` and
`merged_descriptors_with_pic50.csv` (verified with `np.allclose`). Mean 5.6232,
sd 1.3479, range [3.950, 9.721], n = 122.

`data/ligands_122_annotated.csv` is the authoritative join key:
`row_index, mol_id (Mol_N), mol_name, PIC50, smiles, n_heavy, formal_charge, af_group, target`.

## 3. Datasets available

| Key | File | rows × features | Target col | Notes |
|---|---|---|---|---|
| `md272` | `data_model\2.csv` | 122 × 272 | PIC50 | **the manuscript's feature set** |
| `md272_clean` | `data_model\2_cleaned.csv` | 115 × 272 | PIC50 | IsolationForest(contamination=0.05) removed 7 rows; **published results were trained on this** |
| `padel` | `fourth paper\JDes_output3.csv` | 122 × 1444 | *none* | PaDEL 2D/3D; has `Name`; 63 columns × 3 molecules are NaN |
| `padel_v2` | `data_model\JDes_output2.csv` | 122 × 1444 | PIC50 | same descriptors, same row order, near-identical values |
| `padel_v2_imp` | `data_model\JDes_output2_imputed.csv` | 122 × 1444 | PIC50 | NaNs filled (not mean/median/zero — an iterative/KNN imputer) |
| `fused` | `data_model\5_imputed.csv` | 122 × 1549 | PIC50 | MD block ⊕ PaDEL block |
| `mol2desc` | `Said_mol2_files\merged_descriptors_with_pic50.csv` | 122 × **112 194** | PIC50 | 3D pose-derived (PyDescriptor-style); p ≫ n by 3 orders of magnitude |

### JDes_output3 vs JDes_output2

Identical descriptor column **set and order**; identical NaN pattern; same row order
(nearest-neighbour matching returns the identity permutation). 99.39 % of cells are
bit-identical, mean column-wise r = 0.9998. The only substantive difference is row 102
(`CHEMBL178559`), whose structure was re-prepared: `CrippenLogP` 1.328 vs −0.532,
`naasC` 8 vs 6. Rows with NaN descriptors: 102 `CHEMBL178559`, 103 `CHEMBL178857`,
116 `CHEMBL5199803` (missing BCUT / chain / path / VABC descriptors).

**PIC50 for the PaDEL benchmark must be attached by row index** from any of the
three identical sources — JDes_output3 itself carries no target.

## 4. Published results to reproduce and beat

Nu-SVR, from `Classical_ML_model_results_data_2.docx` (17 PCs, RBF, C = 1.0, nu = 0.7,
`random_state=1`, trained on `2_cleaned.csv`):

```
R2 train 0.8582 | CV(5) 0.5423 | test 0.6532 | holdout 0.6668
MSE 0.8591 | RMSE 0.9269 | MAE 0.6265 | W_new 0.306816
```

Hybrid Nu-SVR + DNN + Ridge: `R2 train 0.8943 | CV 0.6546 | test 0.6560 | holdout 0.6680`.

`W_new` (verbatim from the source code):

```python
base = (r2_train + r2_cv + r2_test) / (mse + rmse + mae)
pen  = (1 - abs(r2_train - r2_cv)) / (1 + abs(r2_train - r2_cv))
w_new = base * pen / (1 + base * pen)
```

### Split protocol (published)

```python
X_temp, X_unseen, y_temp, y_unseen = train_test_split(X, y, test_size=0.1,    random_state=RS)
X_train, X_test, y_train, y_test   = train_test_split(X_temp, y_temp, test_size=0.2222, random_state=RS)
```

→ 70 / 20 / 10. **`RS = 1` for Nu-SVR but `RS = 42` for all four DNNs.** The two base
learners of the published "hybrid" were therefore fitted on *different* splits before
being stacked. New work must use one split protocol for every learner.

### Documented inconsistencies in the manuscript (fix during integration)

1. MSE columns are shifted between documents for the Hybrid row:
   main manuscript Table 4 says CV MSE 0.6571 / test 0.4602 / unseen 0.9752;
   `Hybrid model.docx` Table 1 says CV 0.4602 / test 0.9752 / unseen 0.1921.
2. `Hybrid model.docx` narrative quotes DNN-1 R2 train 0.8044 / CV 0.7373 and
   DNN-2 0.9532 / 0.9323 — numbers that appear nowhere in its own Table 1.
3. Nu-SVR's 0.8591 is the **test-set** MSE in the source code but is printed under
   "Mean CV MSE" in manuscript Table 4.
4. Manuscript Table 3 total reads "~270–280 272"; DNN-1/DNN-3 and DNN-2/DNN-4 rows in
   `Hybrid model.docx` Table 1 are exact duplicates.

## 5. The prior, undocumented PaDEL run

`MDS_analysis\data_model\padel_model\` (26 Jul 2025, 15:21–15:24) contains
`scaler.pkl` (`n_features_in_ = 1444`), `pca.pkl` (1444 → **34 components**, 95.01 %
variance), `meta_model_ridge.pkl` (`coef_ = [0.3979, 0.3979]`, alpha 1.0),
`best_dnn_model.keras`, and four scatter plots whose titles read:

```
Train R2 0.455 | 5-fold CV R2 0.411 +/- 0.081 | Test R2 0.434 | Holdout R2 0.371
```

So a PaDEL hybrid **was** run once, on the 1444-descriptor block. But: it was run on
`JDes_output2_imputed` (not `JDes_output3`), no script survives, no MSE/RMSE/MAE/W_new
was recorded, there is no Nu-SVR baseline, and none of it reached the manuscript.
Treat those four numbers as a provenance cross-check for the new run, not as results.

The GitHub repo (`github_repo/`) does **not** contain the training pipeline — its
README advertises `scripts/classical_ml_pipeline.py`, `hybrid_model_training.py`,
`data/`, `models/`, `outputs/`, none of which exist. Only three feature-extraction
scripts, a `.keras` file and a Ridge pickle (`coef_ = [0.803, 0.300]`,
`intercept_ = −0.325`) are present.

## 6. Structural composition of the 122 rows — critical

Parsed from the 122 `.mol2` files with RDKit (all 122 parsed; 98 clean, 19 partial,
5 unsanitised; all single-fragment; charges 0 ×119, +1 ×2, +2 ×1; 16–42 heavy atoms,
median 20).

**Only 94 unique canonical SMILES across 122 rows.** 47 rows sit in multi-row
structure groups: the *same ligand* simulated against *different targets*, with
different pIC50. Worst case, one anthraquinone appears four times spanning
pIC50 4.700 (TP53) → 9.097 (TDP1), a 4.4-log spread on identical chemistry.

Consequences, all verified numerically:

- **Ligand-only descriptor sets have a hard ceiling.** The best possible predictor that
  sees only structure is the within-structure mean: **R2 ≤ 0.8875, RMSE ≥ 0.4503**.
  This bounds PaDEL and DFT benchmarks in principle, and must be drawn on every
  comparison figure.
- **The published random splits leak structures.** At `random_state=1`, 10 of 25 test
  rows and 4 of 13 holdout rows share a SMILES with the training set (at 42: 8/25 and
  5/13). Every benchmark must therefore report a **structure-disjoint (`GroupShuffleSplit`
  on canonical SMILES)** evaluation alongside the published random split.

### Target identity is partly confounded with the label

The AlphaFold pLDDT/PAE block is constant within a protein, so it functions as a
target-identity indicator. There are **8** distinct AlphaFold signatures, not the 4
proteins the manuscript describes (groups 5/6/7 differ only in the third decimal;
groups 1 and 2 are singletons). Group means of pIC50 range 3.95 → 9.60, and:

- AlphaFold-group dummies **alone**: in-sample R2 0.5115, 5-fold CV R2 0.3398.
- All 17 CHEMBL reference actives plus `Co-cryst_5O1C/6GGB/6GGC` carry the *same*
  signature (group 4) and hold the top of the pIC50 range (5.66–9.72).

This reframes the manuscript's SHAP result — PC1 (mean |SHAP| = 0.807) is loaded on
`Fraction_Hi, pLDDT_Q50/Q75/Max/Q25`, i.e. largely on target identity rather than on
chemistry. Report it as such: substantial but not dominant (~1/3 of variance).

### Feature-block ablation (Nu-SVR recipe, all 122 rows, RS = 1)

| Block | n feat | train | test | holdout |
|---|---|---|---|---|
| all 272 (published recipe) | 272 | 0.8283 | 0.5641 | 0.8085 |
| **MD + energy + ligand-PCA + residue only** | **91** | **0.8705** | **0.6753** | **0.7668** |
| everything except AlphaFold | 258 | 0.8128 | 0.5903 | 0.7519 |
| MACCS only | 167 | 0.6717 | 0.4052 | 0.6475 |
| AlphaFold only | 14 | 0.4457 | 0.1560 | 0.4533 |

The 91-feature MD/energetics core **outperforms** the full 272-feature set on test.
Structure-disjoint split, all 272: train 0.8264 / test 0.6198 / holdout 0.7848.

## 6b. Results obtained in this work

Reproduction gate (`src/validate.py`, `md272_clean`, RS = 1, W_new selection):
training R² **0.8581** against 0.8582 published — the study reproduces.

Hybrid test / holdout R², both regimes:

| Descriptor set | n feat | random test | random holdout | disjoint test | disjoint holdout |
|---|---|---|---|---|---|
| md272 (manuscript) | 272 | 0.6217 | 0.8356 | 0.5658 | 0.5747 |
| md272_clean | 272 | 0.5668 | 0.4470 | 0.4129 | 0.6038 |
| fused (MD ⊕ PaDEL) | 1549 | 0.5345 | 0.7564 | 0.6161 | 0.7102 |
| padel | 1444 | 0.5017 | 0.6885 | 0.5595 | 0.5626 |

Nu-SVR ablation (the headline finding):

| Block | n feat | random test | random holdout | **disjoint test** | **disjoint holdout** |
|---|---|---|---|---|---|
| md272 | 272 | 0.5751 | 0.8144 | 0.6267 | 0.6849 |
| **md_core89** | **91** | 0.6505 | 0.7625 | **0.7910** | **0.8382** |
| maccs167 | 167 | 0.3925 | 0.6246 | 0.5773 | 0.2244 |
| alphafold16 | 14 | 0.3276 | 0.4632 | 0.5045 | 0.7157 |

The 89-descriptor MD/energetics core is the best model in the entire study, and it is
best under the *stricter* partition, at 0.791 / 0.838 against a ceiling of 0.8875.
Trimming MACCS and AlphaFold improves the model.

Paired Wilcoxon on per-molecule absolute error (`src/paired_tests.py`), Holm-corrected:
**11 of 90 comparisons are significant.** The 89-descriptor MD core beats the PyDescriptor set (p_holm 1.2e-3), the MACCS block (4.8e-2) and the quantum descriptors (2.1e-7) under the structure-disjoint split. Most remaining pairs are "not distinguishable on these data". The DNN
reproduces its published failure independently (test R² −0.32 here, −0.39 published).

Bootstrap 95 % CIs on test R² are roughly ±0.5 wide on 23–25 test molecules, so the
published hybrid-vs-Nu-SVR gap of 0.0028 is not resolvable.

## 7. Environment

| Tool | Status |
|---|---|
| Windows Python | 3.13.7 — pandas 2.3.3, numpy 2.5.1, sklearn 1.9.0, lightgbm, torch 2.13+cpu, statsmodels, matplotlib 3.11, rdkit 2026.03.5, shap, xgboost, seaborn |
| Node | v24.19.0 |
| WSL | Ubuntu-24.04, 24 cores, 15 GB RAM, 954 GB free. **No pip, no sudo password.** Userspace stack installed at `~/.local/bin/micromamba` + env `qc` (pyscf, xtb, rdkit, ase, openbabel) |
| Saved pickles | written by scikit-learn 1.3.0; loading under 1.9.0 raises `InconsistentVersionWarning` — do not rely on them for numbers |

Files must not be written while open in Word/Excel — if a `.docx`/`.xlsx` write fails,
the file is open, so stop and say so.

## V3b findings (13 Aug 2026)

Three facts established during the V3b build that change what the paper may claim.

**1. `md_core89` is not purely trajectory-derived.** Column 5 is `Docking score`.
88 of the 89 columns come from the trajectory; the docking score belongs to the
pose that seeded the simulation. Methods already disclosed it inside the
structural-stability block, but the summary sentence said the representation was
"generated entirely from the trajectories", which was false. Corrected in V3b.
Methods also said "Four blocks are involved" when there are five.

**2. The residue descriptors are summed across contact types.**
`ligand_protein_interaction_forces.py:179` does
`totals[resname] += len(resids) * energy[kind]`, accumulating all five contact
types into one value per amino-acid type, then aggregating over every residue of
that type in the protein. So a PHE value may be hydrophobic or cation-pi, an ASP
value may be ionic, hydrogen-bonded or water-mediated, and no residue position is
identified. Any reading of SHAP weights as evidence for a specific interaction or
binding mode is unsupportable. The V3 sentence claiming a descriptor "points at a
residue a medicinal chemist can act on" was too strong and has been replaced.

**3. The model does not transfer to an unseen target.** Leave-one-target-out with
the same Nu-SVR pipeline (`results/leave_one_target_out.csv`):

| held-out target | n | R2 | RMSE | RMSE of training mean |
|---|---|---|---|---|
| ESR1 | 26 | -0.135 | 0.468 | 0.950 |
| MAPK1 | 30 | -0.067 | 0.676 | 1.061 |
| TDP1 | 43 | -1.579 | 2.407 | 2.431 |
| TP53 | 23 | -2.641 | 0.708 | 0.993 |

R2 is negative for every target, so the model cannot place an unseen target on the
absolute potency scale. RMSE is nevertheless below the training-mean baseline in
all four cases, so some within-target ordering survives. The structure-disjoint
split prevents ligand leakage but not target leakage: all four targets appear on
both sides of it. Claims are therefore limited to interpolation among these four
target systems.

**Also corrected in V3b:** references were numbered by list position, producing
citations such as (32, 31, 33); they are now numbered by order of first citation,
and an uncited reference raises at build time. The main-text seed-spread figure
was `max SD * 3.9` = 3.11, a theoretical range; the observed within-architecture
spread is 1.99.

**Paired tests, restated precisely.** Structure-disjoint, MD core versus:
PyDescriptor p_holm 1.2e-3, quantum 2.1e-7, MACCS 4.8e-2 are significant;
**PaDEL 0.759 and AlphaFold 0.759 are not.** The Abstract and Conclusions
previously claimed superiority over conventional two-dimensional descriptors
without that qualification.
