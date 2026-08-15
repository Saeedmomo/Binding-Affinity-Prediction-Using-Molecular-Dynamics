"""Rank the swept learners by W(new), the author's own composite criterion, and set
that ranking beside the leakage-free cross-validated ranking.

W(new) is reproduced from benchmark_work/src/hybrid_pipeline.py:60 and verified against
the published md_core89 row: recomputing it from the training, cross-validated and test
coefficients of determination together with the TEST error terms returns 0.572506, which
is the value recorded in results/metrics_all.csv. So W(new) reads the test partition.
That is why the paper selects hyperparameters on cross-validated performance instead, and
why both rankings are reported here rather than only one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
DATA_LABELS = {'md_core89': 'MD core 89', 'padel': 'PaDEL',
               'mol2desc': 'PyDescriptor', 'dft': 'Quantum chemical'}
MODEL_LABELS = {'nusvr': 'Nu-SVR', 'svr': 'SVR', 'ridge': 'Ridge', 'enet': 'Elastic Net',
                'knn': 'K nearest neighbours', 'pls': 'PLS', 'rf': 'Random forest',
                'et': 'Extra trees', 'gbr': 'Gradient boosting',
                'hgb': 'Histogram gradient boosting', 'xgb': 'XGBoost',
                'lgbm': 'LightGBM'}


def w_new(r2_train, r2_cv, r2_test, mse, rmse, mae):
    """Verbatim from the authors' Nu-SVR script."""
    denom = mse + rmse + mae
    if denom <= 0:
        return float('nan')
    base = (r2_train + r2_cv + r2_test) / denom
    gap = abs(r2_train - r2_cv)
    pen = (1 - gap) / (1 + gap)
    return base * pen / (1 + base * pen)


def main():
    raw_path = RES / 'sweep_raw.csv'
    if not raw_path.exists():
        print(f'no sweep results yet at {raw_path}')
        return 1
    d = pd.read_csv(raw_path)
    ok = d[d['status'] == 'ok'].copy()
    if ok.empty:
        print('no successful rows')
        return 1

    ok['mse_test'] = ok['rmse_test'] ** 2
    ok['w_new'] = [w_new(r.r2_train, r.r2_cv, r.r2_test, r.mse_test, r.rmse_test,
                         r.mae_test) for r in ok.itertuples()]

    key = ['dataset', 'split_regime']
    ok['rank_wnew'] = ok.groupby(key)['w_new'].rank(ascending=False, method='min')
    ok['rank_cv'] = ok.groupby(key)['r2_cv'].rank(ascending=False, method='min')
    ok.to_csv(RES / 'sweep_with_wnew.csv', index=False)

    rows = []
    for (ds, reg), g in ok.groupby(key):
        bw = g.loc[g['w_new'].idxmax()]
        bc = g.loc[g['r2_cv'].idxmax()]
        base = g[(g['model'] == 'nusvr') & (g['preprocessing_variant'] == 'pca')]
        base = base.iloc[0] if len(base) else None
        rows.append({
            'Dataset': DATA_LABELS.get(ds, ds),
            'Split': reg.replace('_', ' '),
            'Best by W(new)': f"{MODEL_LABELS.get(bw.model, bw.model)} ({bw.preprocessing_variant})",
            'W(new)': round(float(bw.w_new), 4),
            'Its test R2': round(float(bw.r2_test), 3),
            'Best by CV R2': f"{MODEL_LABELS.get(bc.model, bc.model)} ({bc.preprocessing_variant})",
            'Its test R2 ': round(float(bc.r2_test), 3),
            'Nu-SVR pca W(new)': (round(float(base.w_new), 4) if base is not None else None),
            'Nu-SVR pca test R2': (round(float(base.r2_test), 3) if base is not None else None),
            'Same pick': 'yes' if (bw.model, bw.preprocessing_variant) ==
                                  (bc.model, bc.preprocessing_variant) else 'no',
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(RES / 'decision_wnew_vs_cv.csv', index=False)
    print(summary.to_string(index=False))
    print()
    print(f'rows scored: {len(ok)} of {len(d)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
