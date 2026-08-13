"""Extract the 16-dimensional AlphaFold confidence vector from AlphaFold JSON output.

Reads the atom-level pLDDT scores, averages them per residue, and summarises those
together with the predicted aligned error matrix. Produces the descriptor block used in
the model:

    pLDDT Mean, Std, Min, Max, Q25, Q50, Q75:
    Fraction Hi, Fraction Lo
    PAE Mean, Std, Min, Max, Q25, Q50, Q75

Column names are written exactly as they appear in the modelling matrices so the output
can be joined without renaming. "Fraction Hi" and "Fraction Lo" are the fractions above
pLDDT 90 and below pLDDT 70 respectively; they belong to this block even though their
names do not say so.

CORRECTION relative to the original script. AlphaFold 3 full-data output gives
`atom_plddts` per atom and `token_res_ids` per token, with no array linking the two: for
1SJ0 that is 1978 atom scores against 248 residues. The original script zipped them,

    for atom_idx, residue_id in enumerate(residue_ids):
        residue_plddt_map[residue_id].append(plddt_scores[atom_idx])

which walks only the first 248 entries of `atom_plddts`, i.e. the atoms of roughly the
first thirty residues, and labels them as if they were one residue each. The remaining
87 percent of the atoms are discarded and every statistic describes the wrong part of
the structure.

Because the file carries no atom-to-residue mapping, per-residue means cannot be
recovered from it alone. This version therefore summarises the pLDDT distribution over
all atoms, which is the correct use of the data available, and says so in the output.
Pass --legacy to reproduce the original behaviour for continuity with published matrices.
PAE statistics are unaffected: that matrix is token-level and was always read correctly.

Usage
    python alphafold_feature_extraction.py --json fold_8aoj_full_data_1.json
    python alphafold_feature_extraction.py --json-dir path/to/folds --out alphafold.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

COLUMNS = ['pLDDT Mean', 'pLDDT Std', 'pLDDT Min', 'pLDDT Max',
           'pLDDT Q25', 'pLDDT Q50', 'pLDDT Q75:', 'Fraction Hi', 'Fraction Lo',
           'PAE Mean', 'PAE Std', 'PAE Min', 'PAE Max',
           'PAE Q25', 'PAE Q50', 'PAE Q75']


def features_from_json(path: str, high: float = 90.0, low: float = 70.0,
                       legacy: bool = False) -> dict:
    with open(path) as f:
        data = json.load(f)

    for key in ('atom_plddts', 'token_res_ids', 'pae'):
        if key not in data:
            raise KeyError(f'{os.path.basename(path)} has no "{key}" field; this does '
                           f'not look like AlphaFold full-data output')

    plddt = np.asarray(data['atom_plddts'], float)
    res_ids = np.asarray(data['token_res_ids'])

    if legacy:
        # the original mapping: walk the residue ids and index atom scores positionally,
        # which uses only the first len(res_ids) atoms
        n = min(res_ids.size, plddt.size)
        buckets: dict = {}
        for atom_idx in range(n):
            buckets.setdefault(res_ids[atom_idx], []).append(plddt[atom_idx])
        per_residue = np.array([np.mean(v) for v in buckets.values()])
        basis = f'legacy positional zip, {n} of {plddt.size} atoms used'
    else:
        per_residue = plddt
        basis = f'all {plddt.size} atoms'

    pae = np.asarray(data['pae'], float)

    return {
        'pLDDT Mean': per_residue.mean(),
        'pLDDT Std': per_residue.std(ddof=1) if per_residue.size > 1 else 0.0,
        'pLDDT Min': per_residue.min(),
        'pLDDT Max': per_residue.max(),
        'pLDDT Q25': np.percentile(per_residue, 25),
        'pLDDT Q50': np.percentile(per_residue, 50),
        'pLDDT Q75:': np.percentile(per_residue, 75),
        'Fraction Hi': float((per_residue > high).mean()),
        'Fraction Lo': float((per_residue < low).mean()),
        'PAE Mean': pae.mean(),
        'PAE Std': pae.std(ddof=1) if pae.size > 1 else 0.0,
        'PAE Min': pae.min(),
        'PAE Max': pae.max(),
        'PAE Q25': np.percentile(pae, 25),
        'PAE Q50': np.percentile(pae, 50),
        'PAE Q75': np.percentile(pae, 75),
        '_n_values': int(per_residue.size),
        '_plddt_basis': basis,
        '_pae_shape': f'{pae.shape[0]}x{pae.shape[-1]}',
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--json', nargs='+', help='one or more AlphaFold JSON files')
    src.add_argument('--json-dir', help='directory searched for *.json')
    ap.add_argument('--out', default='alphafold_features.csv')
    ap.add_argument('--high', type=float, default=90.0,
                    help='pLDDT above which a residue counts as high confidence')
    ap.add_argument('--low', type=float, default=70.0,
                    help='pLDDT below which a residue counts as low confidence')
    ap.add_argument('--legacy', action='store_true',
                    help='reproduce the original positional atom-to-residue zip, which '
                         'uses only the first len(token_res_ids) atom scores')
    a = ap.parse_args()

    paths = (sorted(glob.glob(os.path.join(a.json_dir, '*.json')))
             if a.json_dir else a.json)
    if not paths:
        print('no JSON files found', file=sys.stderr)
        return 1

    rows, failed = [], []
    for p in paths:
        try:
            rec = {'structure': os.path.splitext(os.path.basename(p))[0]}
            rec.update(features_from_json(p, a.high, a.low, a.legacy))
            rows.append(rec)
            print(f'  [ok] {rec["structure"]:30s} '
                  f'{rec["_n_values"]:5d} values  '
                  f'pLDDT mean {rec["pLDDT Mean"]:6.2f}  '
                  f'PAE mean {rec["PAE Mean"]:6.2f}  '
                  f'FracHi {rec["Fraction Hi"]:.3f}')
        except Exception as e:
            failed.append((p, str(e)))
            print(f'  [FAIL] {os.path.basename(p)}: {e}', file=sys.stderr)

    if not rows:
        return 1
    df = pd.DataFrame(rows)[['structure'] + COLUMNS +
                            ['_n_values', '_plddt_basis', '_pae_shape']]
    df.to_csv(a.out, index=False)
    print(f'\nwrote {a.out}: {len(df)} structures x {len(COLUMNS)} descriptors')
    if failed:
        print(f'{len(failed)} file(s) failed', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
