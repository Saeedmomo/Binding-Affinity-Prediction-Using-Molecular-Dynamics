"""Dataset registry for the fourth-paper benchmarks.

Every loader returns (X: DataFrame, y: Series, meta: dict) with rows in the canonical
canonical study order. Source files are read-only.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUA = r'D:\Chua files'

PATHS = {
    'md272':       os.path.join(CHUA, 'MDS_analysis', 'data_model', '2.csv'),
    'md272_clean': os.path.join(CHUA, 'MDS_analysis', 'data_model', '2_cleaned.csv'),
    'padel':       os.path.join(CHUA, 'fourth paper', 'JDes_output3.csv'),
    'fused':       os.path.join(CHUA, 'MDS_analysis', 'data_model', '5_imputed.csv'),
    'mol2desc':    os.path.join(CHUA, 'Said_mol2_files', 'merged_descriptors_with_pic50.csv'),
    # produced by src/dft_01..03; present only after the WSL quantum-chemistry run
    'dft':         os.path.join(ROOT, 'data', 'dft', 'quantum_descriptors.csv'),
}

# feature-block ablations carved out of md272, and the DFT+MD fusion. These are derived
# in-code rather than read from a file.
DERIVED = ('md_core89', 'maccs167', 'alphafold16', 'dft_plus_md')

LABELS = {
    'md272':       'MD + AlphaFold + MACCS (272)',
    'md272_clean': 'MD + AlphaFold + MACCS, outliers removed (272)',
    'padel':       'PaDEL 2D/3D descriptors (1444)',
    'fused':       'MD block + PaDEL block fused (1549)',
    'mol2desc':    'Pose-derived 3D descriptors (112194)',
    'dft':         'DFT/xTB quantum descriptors',
    'dft_plus_md': 'DFT quantum + MD block',
}

TARGET = 'PIC50'


def sha256(path: str, cap: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        read = 0
        while read < cap:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            read += len(b)
    return h.hexdigest()[:16]


def annotations() -> pd.DataFrame:
    """The 122-row join key: row_index, mol_name, PIC50, smiles, target, af_group."""
    df = pd.read_csv(os.path.join(ROOT, 'data', 'ligands_122_annotated.csv'))
    assert len(df) == 122, f'expected 122 annotated ligands, got {len(df)}'
    return df


def _canonical_pic50() -> pd.Series:
    """PIC50 in canonical row order. Identical in 2.csv, JDes_output2 and merged_*,
    verified with np.allclose."""
    return pd.read_csv(PATHS['md272'], usecols=[TARGET], low_memory=False)[TARGET]


def _md_blocks() -> dict[str, list[str]]:
    """Split the 272 md272 columns into their feature blocks.

    'Fraction Hi' and 'Fraction Lo' are the fractions of residues with pLDDT above 90
    and below 70. They are AlphaFold confidence descriptors despite not carrying pLDDT
    in their column name, and matching on the name alone put them in the MD core. That
    also reconciles the block with the manuscript, which describes a 16-dimensional
    AlphaFold vector: 14 pLDDT/PAE summary statistics plus these two fractions.
    """
    X, _y, _m = load('md272')
    af = [c for c in X.columns
          if 'pLDDT' in c or 'PAE' in c or c.strip().lower().startswith('fraction ')]
    maccs = [c for c in X.columns if c.startswith('MACCS')]
    core = [c for c in X.columns if c not in af and c not in maccs]
    assert len(af) == 16, f'expected a 16-column AlphaFold block, got {len(af)}: {af}'
    assert len(maccs) == 167, f'expected 167 MACCS bits, got {len(maccs)}'
    assert len(core) == 89, f'expected an 89-column MD core, got {len(core)}'
    return dict(alphafold16=af, maccs167=maccs, md_core89=core)


def load(key: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    if key in DERIVED:
        return _load_derived(key)
    if key not in PATHS:
        raise KeyError(f'unknown dataset {key!r}; known: {sorted(PATHS) + list(DERIVED)}')
    path = PATHS[key]
    if key == 'dft' and not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found - run src/dft_01_export_geometries.py, '
            f'src/dft_02_run_qc.py (in WSL) and src/dft_03_assemble.py first')
    meta = dict(key=key, label=LABELS[key], path=path, sha256_16=sha256(path))

    if key == 'padel':
        # PaDEL output carries no target; attach PIC50 by row index.
        df = pd.read_csv(path, low_memory=False)
        assert len(df) == 122
        names = df['Name'].tolist()
        assert names[0] == 'Mol_1' and names[-1] == 'Mol_122', 'unexpected Name ordering'
        X = df.drop(columns=['Name'])
        y = _canonical_pic50()
        meta['target_source'] = 'attached by row index from 2.csv'
        meta['row_ids'] = np.arange(122)
    elif key == 'dft':
        df = pd.read_csv(path, low_memory=False)
        assert TARGET in df.columns, 'dft: no PIC50 column'
        meta['row_ids'] = df['row_index'].to_numpy()
        y = df[TARGET].reset_index(drop=True)
        X = df.drop(columns=[TARGET, 'row_index', 'mol_id', 'mol_name', 'target'],
                    errors='ignore')
        meta['target_source'] = 'in-file'
        meta['n_molecules_with_qc'] = int(len(df))
    else:
        df = pd.read_csv(path, low_memory=False)
        assert TARGET in df.columns, f'{key}: no {TARGET} column'
        y = df[TARGET].reset_index(drop=True)
        X = df.drop(columns=[TARGET])
        meta['target_source'] = 'in-file'
        if key == 'md272_clean':
            meta['row_ids'] = _match_clean_rows(X)
        else:
            assert len(df) == 122, f'{key}: expected 122 rows, got {len(df)}'
            meta['row_ids'] = np.arange(122)

    # drop non-numeric / identifier columns if any slipped through
    nonnum = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if nonnum:
        meta['dropped_nonnumeric'] = nonnum[:20]
        X = X.drop(columns=nonnum)

    if key == 'mol2desc':
        # 44% of the 112194 pose descriptors are all-zero for every molecule. Dropping
        # them here is exactly equivalent to the variance gate inside the pipeline -- a
        # column constant across all 122 rows is constant within every fold too -- and it
        # is label-free, so no information leaks. Doing it once at load time rather than
        # per fold, and holding the matrix as float32, is what makes this set tractable:
        # 20 joblib workers each copying the float64 matrix exhausted memory and the run
        # died silently.
        v = X.to_numpy(np.float32).var(axis=0)
        keep = v > 1e-12
        meta['dropped_constant_columns'] = int((~keep).sum())
        X = X.loc[:, keep]
        X = X.astype(np.float32)

    X = X.reset_index(drop=True)
    meta.update(n_rows=int(len(X)), n_features=int(X.shape[1]),
                n_nan_cells=int(X.isna().to_numpy().sum()),
                n_cols_with_nan=int((X.isna().sum() > 0).sum()))
    return X, y, meta


def _load_derived(key: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Feature-block ablations of md272, and the DFT + MD fusion."""
    if key in ('md_core89', 'maccs167', 'alphafold16'):
        X, y, meta = load('md272')
        cols = _md_blocks()[key]
        X = X[cols]
        meta = dict(meta, key=key, label={
            'md_core89': 'MD + energetics + ligand-PCA + residue forces',
            'maccs167': 'MACCS fingerprints only',
            'alphafold16': 'AlphaFold pLDDT/PAE block only'}[key],
            n_features=int(X.shape[1]), derived_from='md272')
        return X, y, meta

    # dft_plus_md: quantum descriptors joined to the MD block on canonical row index
    Xd, yd, md_ = load('dft')
    Xm, _ym, mm = load('md272')
    ids = md_['row_ids']
    Xm = Xm.iloc[ids].reset_index(drop=True)
    Xm.columns = [f'md__{c}' for c in Xm.columns]
    Xd = Xd.reset_index(drop=True)
    Xd.columns = [f'qc__{c}' for c in Xd.columns]
    X = pd.concat([Xd, Xm], axis=1)
    meta = dict(key=key, label='DFT quantum descriptors + MD block',
                path='(derived)', sha256_16='-', target_source='in-file',
                row_ids=ids, n_rows=int(len(X)), n_features=int(X.shape[1]),
                n_nan_cells=int(X.isna().to_numpy().sum()),
                n_cols_with_nan=int((X.isna().sum() > 0).sum()),
                derived_from='dft + md272')
    return X, yd.reset_index(drop=True), meta


def _match_clean_rows(Xc: pd.DataFrame) -> np.ndarray:
    """Map the 115 rows of 2_cleaned.csv back to canonical row indices by value,
    not by position (IsolationForest removed 7 rows)."""
    full = pd.read_csv(PATHS['md272'], low_memory=False).drop(columns=[TARGET])
    cols = [c for c in Xc.columns if c in full.columns]
    A = full[cols].to_numpy(float)
    B = Xc[cols].to_numpy(float)
    ids, used = [], set()
    for i in range(len(B)):
        d = np.nansum((A - B[i]) ** 2, axis=1)
        for j in np.argsort(d):
            if j not in used:
                assert d[j] < 1e-6, f'row {i} of 2_cleaned has no exact match (d={d[j]:.3g})'
                ids.append(int(j))
                used.add(int(j))
                break
    ids = np.array(ids)
    assert len(ids) == len(B) == 115, f'matched {len(ids)} of {len(B)} cleaned rows'
    return ids


def groups_for(meta: dict) -> np.ndarray:
    """Canonical-SMILES group labels aligned to the dataset's rows, for
    structure-disjoint splitting."""
    ann = annotations()
    return ann['smiles'].to_numpy()[meta['row_ids']]


def names_for(meta: dict) -> pd.DataFrame:
    ann = annotations()
    return ann.iloc[meta['row_ids']][['row_index', 'mol_name', 'smiles', 'target']] \
              .reset_index(drop=True)


if __name__ == '__main__':
    for k in list(PATHS) + list(DERIVED):
        try:
            X, y, m = load(k)
        except FileNotFoundError as e:
            print(f'{k:13s} not available yet ({e})')
            continue
        print(f"{k:13s} rows={m['n_rows']:4d} feat={m['n_features']:7d} "
              f"nan_cells={m['n_nan_cells']:5d} y[mean={y.mean():.4f} sd={y.std():.4f}] "
              f"sha={m['sha256_16']}  {m['target_source']}")
        g = groups_for(m)
        print(f"{'':13s} unique structures in this dataset: {len(set(g))}")

