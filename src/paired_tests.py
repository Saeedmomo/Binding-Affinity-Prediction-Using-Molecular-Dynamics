"""Paired comparison of descriptor sets on identical molecules.

Comparing headline R2 values across descriptor sets ignores that every set was scored
on the *same* held-out molecules. A paired test on the per-molecule absolute errors is
both more powerful and more honest than comparing two point estimates whose bootstrap
intervals overlap almost completely.

Wilcoxon signed-rank on |error| (two-sided), plus the median paired difference and a
rank-biserial effect size. Holm correction across the comparisons within each regime.
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
PRED = os.path.join(RES, 'predictions')
TAB = os.path.join(ROOT, 'tables')

LABEL = {
    'md272': 'MD+AF+MACCS (272)', 'md272_clean': 'MD+AF+MACCS, no outliers (272)',
    'md_core89': 'MD core (89)', 'maccs167': 'MACCS only (167)',
    'alphafold16': 'AlphaFold only (16)', 'padel': 'PaDEL (1444)',
    'fused': 'MD+PaDEL (1549)', 'mol2desc': 'Pose 3D (112194)',
    'dft': 'DFT quantum', 'dft_plus_md': 'DFT + MD',
}
ORDER = list(LABEL)


def holm(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (n - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def load(regime, model='hybrid'):
    out = {}
    for f in os.listdir(PRED):
        if not f.endswith(f'__{regime}.csv'):
            continue
        ds = f.split('__')[0]
        d = pd.read_csv(os.path.join(PRED, f))
        d = d[d.subset.isin(['test', 'holdout'])]
        d = d.set_index('row_index')
        out[ds] = (d[f'pred_{model}'] - d['y_true']).abs()
    return out


def main():
    rows = []
    for regime in ('random', 'structure_disjoint'):
        err = load(regime)
        keys = [k for k in ORDER if k in err]
        if len(keys) < 2:
            continue
        recs = []
        for a, b in combinations(keys, 2):
            ea, eb = err[a], err[b]
            common = ea.index.intersection(eb.index)
            if len(common) < 8:
                continue
            x, y = ea.loc[common].to_numpy(), eb.loc[common].to_numpy()
            d = x - y
            nz = d[d != 0]
            if len(nz) < 6:
                continue
            stat, p = stats.wilcoxon(x, y, zero_method='wilcox',
                                     alternative='two-sided')
            # rank-biserial effect size
            # d = |err_A| - |err_B|. Ranks of positive d count against A, negative for A.
            pos = np.sum(stats.rankdata(np.abs(nz))[nz > 0])
            neg = np.sum(stats.rankdata(np.abs(nz))[nz < 0])
            tot = pos + neg
            # rank-biserial correlation: positive => A has the lower absolute error
            rbc = (neg - pos) / tot if tot else np.nan
            recs.append(dict(
                Split={'random': 'Random',
                       'structure_disjoint': 'Structure-disjoint'}[regime],
                A=LABEL[a], B=LABEL[b], n=len(common),
                median_abs_err_A=round(float(np.median(x)), 4),
                median_abs_err_B=round(float(np.median(y)), 4),
                median_diff=round(float(np.median(d)), 4),
                effect_size_rbc=round(float(rbc), 3),
                W=float(stat), p_raw=float(p)))
        if recs:
            df = pd.DataFrame(recs)
            df['p_holm'] = holm(df['p_raw'].to_numpy())
            df['significant'] = np.where(df['p_holm'] < 0.05, 'yes', 'no')
            df['p_raw'] = df['p_raw'].map(lambda v: f'{v:.4g}')
            df['p_holm'] = df['p_holm'].map(lambda v: f'{v:.4g}')
            rows.append(df)

    if not rows:
        print('not enough prediction files yet')
        return
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(TAB, exist_ok=True)
    out.to_csv(os.path.join(TAB, 'TableS4_paired_tests.csv'), index=False)

    from make_tables import write
    write(out, 'TableS4_paired_tests',
          'Table S4. Paired comparison of descriptor sets on identical held-out '
          'molecules (Wilcoxon signed-rank on absolute error, hybrid model).',
          'The paired difference is |error(A)| - |error(B)| per molecule, so a NEGATIVE '
          'median difference and a POSITIVE rank-biserial effect size both indicate that '
          'descriptor set A predicts more accurately. p-values are Holm-corrected within '
          'each split regime; comparisons not marked significant should be read as "not '
          'distinguishable on these data" rather than as evidence of equivalence.')

    print(out[['Split', 'A', 'B', 'n', 'median_diff', 'effect_size_rbc',
               'p_raw', 'p_holm', 'significant']].to_string(index=False))
    sig = out[out.significant == 'yes']
    print(f'\n{len(sig)} of {len(out)} pairwise comparisons are significant after Holm '
          f'correction.')


if __name__ == '__main__':
    main()
