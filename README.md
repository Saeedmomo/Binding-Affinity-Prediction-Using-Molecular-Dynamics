# 🧪 Ligand–Protein Interaction Modeling Using Molecular Dynamics and Hybrid Machine Learning

This repository contains the full pipeline for a scientific study that develops and evaluates a **hybrid machine learning model** for predicting **PIC50** values from **molecular dynamics (MD)** simulation data, **ligand features**, **AlphaFold structural confidence**, and **interaction energy descriptors**.

## 📌 Project Objective

To design and validate a predictive pipeline that accurately estimates **ligand–protein binding affinity (PIC50)** using a robust hybrid model that combines:
- Classical machine learning (Nu-SVR)
- Deep neural networks (DNN)
- Ensemble meta-learning (Ridge regression)

---

## 📂 Repository Structure

├── data/ │ ├── features/ # Final feature CSV files │ └── json/ # AlphaFold JSON structure files ├── scripts/ │ ├── extract_alphaFold_features.py │ ├── ligand_pca_feature_extraction.py │ ├── interaction_energy_calculator.py │ ├── classical_ml_pipeline.py │ ├── hybrid_model_training.py ├── models/ │ ├── saved_dnn_model.h5 │ ├── saved_svr_model.pkl ├── outputs/ │ ├── pca_plots/ │ ├── predictions/ │ └── evaluation_metrics/ ├── Interaction_Force_Feature_Engineering.docx └── README.md

---

## 🧰 Feature Engineering Pipeline

Each ligand–protein complex was represented using **multiple categories of descriptors**, extracted from MD simulations and precomputed data:

### ✅ 1. MD-Based Structural & Energetic Features
- Protein–Ligand RMSD, RMSF
- Ligand RMSD, RMSF
- Docking Score (Glide)
- SASA (Solvent Accessible Surface Area)
- Total, Potential, Kinetic, Coulombic Energies

### ✅ 2. Interaction Force Features
- Hydrogen Bonds, π-Cation, Hydrophobic, Ionic, Water Bridges
- Computed using custom scripts based on geometric and angular cutoffs
- Average force per residue per interaction type

### ✅ 3. Per-Residue Interaction Summary
- 20 amino acid features (ALA, ARG, ..., VAL)
- Represent interaction density across protein surface

### ✅ 4. Ligand Dynamics via PCA
- Principal component analysis (PC1–PC3) of ligand-only trajectories
- Eigenvector shape descriptors
- Average conformer shape

### ✅ 5. MACCS Keys (Ligand Substructure Fingerprints)
- 167-bit fingerprint from RDKit
- Encodes substructure presence/absence for each ligand

### ✅ 6. AlphaFold Structural Confidence
- pLDDT scores: mean, std, min, max, quartiles, high/low confidence fractions
- PAE matrix: mean, std, min, max, quartiles

---

## 🧠 Machine Learning Pipeline

### 🔹 Classical ML Models
- Evaluated 10 regressors including Linear, SVR, Decision Tree, Random Forest, XGBoost
- **Best model**: Nu-SVR (`C=1.0`, `nu=0.7`, `kernel='rbf'`)

### 🔹 Deep Neural Networks
- 4 DNN variants with dropout, L2 regularization, Adam/AdamW
- Captured nonlinear patterns but prone to overfitting on small datasets

### 🔹 Hybrid Ensemble Model
- Combined Nu-SVR and DNN predictions
- Final prediction made by Ridge Regression meta-learner
- Achieved best R² and lowest error on holdout data

---

## 📊 Model Evaluation

| **Model**   | **Train R²** | **CV R²** | **Test R²** | **Holdout R²** | **MSE** |
|-------------|--------------|-----------|-------------|----------------|----------|
| Nu-SVR      | 0.8582       | 0.5423    | 0.6532      | 0.6668         | 0.8591   |
| Hybrid Model| **0.8943**   | **0.6546**| **0.6560**  | **0.6680**     | **0.4602** |

---

## 📦 Requirements

Install all dependencies via:

```bash
conda create -n md_hybrid_model python=3.9
conda activate md_hybrid_model
pip install -r requirements.txt
