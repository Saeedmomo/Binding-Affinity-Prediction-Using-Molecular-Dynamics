"""Reproduction gate: can the harness recover the manuscript's published Nu-SVR numbers?

The published Nu-SVR result was produced on 2_cleaned.csv with random_state=1, 17
principal components, RBF kernel, C=1.0, nu=0.7, and hyperparameters ranked by W_new --
a criterion that reads the test set. To reproduce it we must therefore switch the harness
back into its "as published" mode: leaky_preprocessing is not needed for Nu-SVR (the
Pipeline is equivalent when PCA is refitted per fold only for CV), but selection must be
'w_new_test'.

A faithful negative result here is more useful than a passing test obtained by loosening
tolerances. If the numbers do not land, the discrepancy is reported and the gate fails.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import datasets
from hybrid_pipeline import fit_nusvr, make_splits, metrics, w_new

PUBLISHED = dict(r2_train=0.8582, r2_cv=0.5423, r2_test=0.6532, r2_holdout=0.6668,
                 mse=0.8591, rmse=0.9269, mae=0.6265, w_new=0.306816)
TOL_R2 = 0.05
TOL_ERR = 0.15


def main() -> int:
    X, y, meta = datasets.load('md272_clean')
    groups = datasets.groups_for(meta)
    print(f'dataset : {meta["label"]}')
    print(f'          {meta["n_rows"]} rows x {meta["n_features"]} features  '
          f'sha256[:16]={meta["sha256_16"]}')
    print(f'          {len(set(groups))} unique canonical structures\n')

    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    sp = make_splits(len(yv), groups=groups, regime='random', random_state=1)
    print(f'split   : train={len(sp["train"])} test={len(sp["test"])} '
          f'holdout={len(sp["holdout"])} (random_state=1, as published)\n')

    print('Nu-SVR sweep, hyperparameters ranked by W_new (as published):')
    est, pick, trials = fit_nusvr(Xv, yv, sp, selection='w_new_test', cv=5, verbose=True)

    m_tr = metrics(yv[sp['train']], est.predict(Xv[sp['train']]))
    m_te = metrics(yv[sp['test']], est.predict(Xv[sp['test']]))
    m_ho = metrics(yv[sp['holdout']], est.predict(Xv[sp['holdout']]))
    obs = dict(r2_train=m_tr['r2'], r2_cv=pick['r2_cv'], r2_test=m_te['r2'],
               r2_holdout=m_ho['r2'], mse=m_te['mse'], rmse=m_te['rmse'], mae=m_te['mae'])
    obs['w_new'] = w_new(obs['r2_train'], obs['r2_cv'], obs['r2_test'],
                         obs['mse'], obs['rmse'], obs['mae'])

    print(f'\nselected: n_components={pick["n_components"]}  {pick["params"]}')
    print('          published selection was n_components=17, '
          "{'C': 1.0, 'kernel': 'rbf', 'nu': 0.7}\n")

    rows, ok = [], True
    for k, want in PUBLISHED.items():
        got = obs[k]
        tol = TOL_R2 if k.startswith('r2') or k == 'w_new' else TOL_ERR
        passed = abs(got - want) <= tol
        ok &= passed
        rows.append(dict(metric=k, published=want, observed=round(got, 4),
                         delta=round(got - want, 4), tol=tol,
                         status='PASS' if passed else 'FAIL'))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print(f'\nGATE: {"PASS" if ok else "FAIL"}')
    if not ok:
        print("""
Diagnosis of the discrepancy, in order of likelihood:
  1. The published scripts fitted StandardScaler and PCA on the FULL dataset before
     splitting (see NN_Model.docx and the Nu-SVR listing). This harness fits them inside
     a Pipeline on training folds only. The published CV R2 is therefore optimistic and
     its train/test R2 shift slightly.
  2. GridSearchCV tie-breaking differs between scikit-learn 1.3 (used originally, per the
     pickled artefacts) and 1.9 (installed here).
  3. The published run swept n_components in {15,17,19,20,21,22} and reported the single
     best W_new across the sweep; if several settings tie, the retained one differs.
Any of these shifts R2 by a few hundredths without changing the conclusions.""")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
