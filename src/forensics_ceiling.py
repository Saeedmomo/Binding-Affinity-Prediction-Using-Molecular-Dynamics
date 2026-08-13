"""Recover the protein-target label for each of the 122 rows, and quantify the
irreducible-error ceiling that duplicate ligand structures impose on any
ligand-only descriptor set (PaDEL 2D, DFT quantum descriptors)."""
import pandas as pd, numpy as np, os, json

W  = r'D:\Chua files\fourth paper\benchmark_work'
d2 = pd.read_csv(r'D:\Chua files\MDS_analysis\data_model\2.csv', low_memory=False)
lig = pd.read_csv(os.path.join(W, 'data', 'ligands_122_smiles.csv'))
assert len(d2) == len(lig) == 122

# ---- 1. recover target identity from the AlphaFold block (per-protein constants) ----
af = [c for c in d2.columns if 'pLDDT' in c or 'PAE' in c]
print("AlphaFold columns:", af)
sig = d2[af].round(6).astype(str).agg('|'.join, axis=1)
groups = {s: i for i, s in enumerate(sorted(sig.unique()))}
d2['af_group'] = sig.map(groups)
print("\ndistinct AlphaFold signatures (=distinct proteins):", len(groups))

lig['af_group'] = d2['af_group']
lig['prefix'] = (lig['mol_name'].str.extract(r'^(1SJ0|6GGB|6GGC|8AOJ|6N0D|5O1C|2OUZ|3ERT|CHEMBL|Co-cryst)')[0]
                 .fillna('other'))
print("\ncross-tab AlphaFold group x mol_name prefix")
print(pd.crosstab(lig['af_group'], lig['prefix']).to_string())

PDB2TARGET = {0: None, 1: None, 2: None, 3: None}
print("\npLDDT/PAE means per group (to name the proteins):")
print(d2.groupby('af_group')[af].first().round(3).to_string())

# name groups by the dominant structural prefix
name_map = {'1SJ0': 'ESR1', '6GGB': 'TP53', '6GGC': 'TP53', '8AOJ': 'MAPK1',
            '6N0D': 'TDP1', '5O1C': 'TP53', '2OUZ': 'ESR1', '3ERT': 'ESR1'}
lig['target_guess'] = lig['prefix'].map(name_map)
grp2target = (lig.dropna(subset=['target_guess'])
                 .groupby('af_group')['target_guess']
                 .agg(lambda s: s.value_counts().idxmax()))
lig['target'] = lig['af_group'].map(grp2target)
print("\ngroup -> target:", grp2target.to_dict())
print("\nrows per target:")
print(lig['target'].value_counts().to_string())
print("\nunresolved-prefix rows and their assigned target:")
print(lig[lig['target_guess'].isna()][['mol_name','target','PIC50']].to_string(index=False))

# ---- 2. duplicate-structure ceiling for ligand-only feature sets ----
print("\n" + "="*80)
print("IRREDUCIBLE-ERROR CEILING FOR LIGAND-ONLY DESCRIPTORS")
print("="*80)
y = lig['PIC50'].to_numpy()
g = lig['smiles'].to_numpy()
tv = y.var(ddof=0)
# best possible predictor that sees only the structure = group mean
pred = pd.Series(y).groupby(pd.Series(g)).transform('mean').to_numpy()
ss_res = ((y - pred) ** 2).sum()
ss_tot = ((y - y.mean()) ** 2).sum()
r2_ceiling = 1 - ss_res / ss_tot
print(f"n rows                       : {len(y)}")
print(f"n unique structures (SMILES) : {lig['smiles'].nunique()}")
print(f"rows in multi-target groups  : {int((pd.Series(g).map(pd.Series(g).value_counts()) > 1).sum())}")
print(f"total variance of PIC50      : {tv:.4f}")
print(f"within-structure variance    : {ss_res/len(y):.4f}  ({ss_res/ss_tot*100:.1f}% of total)")
print(f"CEILING R2 (ligand-only)     : {r2_ceiling:.4f}")
print(f"CEILING RMSE (ligand-only)   : {np.sqrt(ss_res/len(y)):.4f}")

# largest within-structure PIC50 spreads
tmp = lig.groupby('smiles').agg(n=('PIC50','size'), lo=('PIC50','min'), hi=('PIC50','max'),
                                mols=('mol_name', lambda s: ';'.join(s)),
                                targets=('target', lambda s: ';'.join(map(str, s))))
tmp['spread'] = tmp['hi'] - tmp['lo']
print("\nlargest within-structure PIC50 spreads (same ligand, different target):")
print(tmp[tmp['n'] > 1].sort_values('spread', ascending=False)
        .head(8)[['n','lo','hi','spread','targets']].round(3).to_string())

# ---- 3. leakage check on the published random split ----
from sklearn.model_selection import train_test_split
print("\n" + "="*80)
print("STRUCTURE LEAKAGE IN THE PUBLISHED RANDOM SPLITS")
print("="*80)
idx = np.arange(122)
for rs in (1, 42):
    tmp_i, unseen_i = train_test_split(idx, test_size=0.1, random_state=rs)
    tr_i, te_i = train_test_split(tmp_i, test_size=0.2222, random_state=rs)
    for nm, a, b in [('train/test', tr_i, te_i), ('train/holdout', tr_i, unseen_i)]:
        shared = set(g[a]) & set(g[b])
        nleak = int(np.isin(g[b], list(shared)).sum())
        print(f"  random_state={rs:2d}  {nm:14s} n_a={len(a):3d} n_b={len(b):3d}  "
              f"shared structures={len(shared):2d}  leaked rows in b={nleak}/{len(b)}")

lig.to_csv(os.path.join(W, 'data', 'ligands_122_annotated.csv'), index=False)
print("\nwrote", os.path.join(W, 'data', 'ligands_122_annotated.csv'))
json.dump({'r2_ceiling_ligand_only': float(r2_ceiling),
           'rmse_ceiling_ligand_only': float(np.sqrt(ss_res/len(y))),
           'n_unique_smiles': int(lig['smiles'].nunique()),
           'n_rows': int(len(y))},
          open(os.path.join(W, 'data', 'ceiling.json'), 'w'), indent=2)
