"""Extract canonical SMILES + PIC50 for the 122 study ligands from their .mol2 files.

Row order of merged_descriptors.csv['mol_name'] is the canonical study order and is
shared by JDes_output2/3, 2.csv and the PIC50 vector (established by forensics).
"""
import os, glob, sys
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

MOL2 = r'D:\Chua files\Said_mol2_files'
OUT  = r'D:\Chua files\fourth paper\benchmark_work\data'
os.makedirs(OUT, exist_ok=True)

order = pd.read_csv(os.path.join(MOL2, 'merged_descriptors.csv'),
                    usecols=['mol_name'], low_memory=False)['mol_name'].tolist()
pic50 = pd.read_csv(os.path.join(MOL2, 'merged_descriptors_with_pic50.csv'),
                    usecols=['PIC50'], low_memory=False)['PIC50'].tolist()
names_j3 = pd.read_csv(r'D:\Chua files\fourth paper\JDes_output3.csv',
                       usecols=['Name'], low_memory=False)['Name'].tolist()
assert len(order) == len(pic50) == len(names_j3) == 122

rows, fails = [], []
for i, (nm, y, mid) in enumerate(zip(order, pic50, names_j3)):
    p = os.path.join(MOL2, nm + '.mol2')
    rec = dict(row_index=i, mol_id=mid, mol_name=nm, PIC50=y, mol2_path=p)
    mol = Chem.MolFromMol2File(p, sanitize=True, removeHs=False)
    how = 'sanitized'
    if mol is None:
        mol = Chem.MolFromMol2File(p, sanitize=False, removeHs=False)
        how = 'unsanitized'
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^
                                      Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                how = 'partial_sanitize'
            except Exception as e:
                rec['note'] = f'sanitize failed: {e}'
    if mol is None:
        fails.append(nm); rec.update(smiles=None, parse=how); rows.append(rec); continue

    noH = Chem.RemoveHs(mol, sanitize=False)
    try:
        smi = Chem.MolToSmiles(noH)
    except Exception:
        smi = None
    rec.update(
        smiles=smi, parse=how,
        n_atoms_withH=mol.GetNumAtoms(),
        n_heavy=noH.GetNumHeavyAtoms(),
        formal_charge=Chem.GetFormalCharge(mol),
        has_3d=int(mol.GetNumConformers() > 0),
        n_frags=len(Chem.GetMolFrags(noH)),
    )
    try:
        rec['mw'] = round(Descriptors.MolWt(noH), 3)
        rec['n_rot'] = rdMolDescriptors.CalcNumRotatableBonds(noH)
        rec['n_ring'] = rdMolDescriptors.CalcNumRings(noH)
    except Exception:
        pass
    rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'ligands_122_smiles.csv'), index=False)

print("wrote", os.path.join(OUT, 'ligands_122_smiles.csv'))
print("total:", len(df), " smiles ok:", df['smiles'].notna().sum(), " failures:", fails)
print("\nparse modes:", df['parse'].value_counts().to_dict())
print("\nheavy-atom count: min=%s med=%s max=%s  (DFT cost driver)"
      % (df['n_heavy'].min(), df['n_heavy'].median(), df['n_heavy'].max()))
print("atoms incl. H:   min=%s med=%s max=%s"
      % (df['n_atoms_withH'].min(), df['n_atoms_withH'].median(), df['n_atoms_withH'].max()))
print("formal charges:", df['formal_charge'].value_counts().to_dict())
print("fragment counts:", df['n_frags'].value_counts().to_dict())
print("has 3D conformer:", df['has_3d'].sum(), "/", len(df))
print("\nunique canonical SMILES:", df['smiles'].nunique(), "of", df['smiles'].notna().sum())
dup = df[df.duplicated('smiles', keep=False) & df['smiles'].notna()]
if len(dup):
    print("DUPLICATE structures (same SMILES, different PIC50?):")
    for s, g in dup.groupby('smiles'):
        print("   ", list(zip(g['mol_name'], g['PIC50'].round(3))))
print("\nfirst 5 rows:")
print(df[['row_index','mol_id','mol_name','PIC50','n_heavy','smiles']].head().to_string(index=False))
