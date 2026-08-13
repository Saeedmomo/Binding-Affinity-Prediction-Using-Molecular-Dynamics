import json
import numpy as np
import pandas as pd

# === Load AlphaFold JSON Output ===
json_file = 'path_to_your_json_file.json'  # Specify the path to your AlphaFold JSON file

with open(json_file, "r") as f:
    data = json.load(f)

# === Extract pLDDT Scores (atom-level) ===
plddt_scores = np.array(data["atom_plddts"])
residue_ids = np.array(data["token_res_ids"])

# === Compute Mean pLDDT Per Residue ===
unique_residues = np.unique(residue_ids)
residue_plddt_map = {res: [] for res in unique_residues}

for atom_idx, residue_id in enumerate(residue_ids):
    residue_plddt_map[residue_id].append(plddt_scores[atom_idx])

residue_avg_plddt = {res: np.mean(scores) for res, scores in residue_plddt_map.items()}
df_residue_plddt = pd.DataFrame(list(residue_avg_plddt.items()), columns=["Residue_Index", "pLDDT_Score"])

# === Compute Statistical Features for pLDDT ===
plddt_mean = df_residue_plddt["pLDDT_Score"].mean()
plddt_std = df_residue_plddt["pLDDT_Score"].std()
plddt_min = df_residue_plddt["pLDDT_Score"].min()
plddt_max = df_residue_plddt["pLDDT_Score"].max()
plddt_q25 = df_residue_plddt["pLDDT_Score"].quantile(0.25)
plddt_q50 = df_residue_plddt["pLDDT_Score"].median()
plddt_q75 = df_residue_plddt["pLDDT_Score"].quantile(0.75)
frac_high_confidence = (df_residue_plddt["pLDDT_Score"] > 90).mean()
frac_low_confidence = (df_residue_plddt["pLDDT_Score"] < 70).mean()

# === Extract and Analyze PAE Matrix ===
pae_matrix = np.array(data["pae"])
pae_mean = pae_matrix.mean()
pae_std = pae_matrix.std()
pae_min = pae_matrix.min()
pae_max = pae_matrix.max()
pae_q25 = np.percentile(pae_matrix, 25)
pae_q50 = np.percentile(pae_matrix, 50)
pae_q75 = np.percentile(pae_matrix, 75)

# === Display Results ===
print("\n=== AlphaFold Extracted Features ===")
print(f"pLDDT Mean: {plddt_mean:.2f}")
print(f"pLDDT Std: {plddt_std:.2f}")
print(f"pLDDT Min: {plddt_min:.2f}")
print(f"pLDDT Max: {plddt_max:.2f}")
print(f"pLDDT Q25: {plddt_q25:.2f}")
print(f"pLDDT Q50: {plddt_q50:.2f}")
print(f"pLDDT Q75: {plddt_q75:.2f}")
print(f"Fraction High Confidence (pLDDT > 90): {frac_high_confidence:.2%}")
print(f"Fraction Low Confidence (pLDDT < 70): {frac_low_confidence:.2%}")

print("\n=== PAE Extracted Features ===")
print(f"PAE Mean: {pae_mean:.2f}")
print(f"PAE Std: {pae_std:.2f}")
print(f"PAE Min: {pae_min:.2f}")
print(f"PAE Max: {pae_max:.2f}")
print(f"PAE Q25: {pae_q25:.2f}")
print(f"PAE Q50: {pae_q50:.2f}")
print(f"PAE Q75: {pae_q75:.2f}")

print("\nFeature extraction complete.")
