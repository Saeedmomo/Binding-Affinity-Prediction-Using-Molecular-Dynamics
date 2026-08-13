"""Fit and persist the final models, so the published result can be applied to new data.

The model shipped previously (best_hybrid_nn_model.keras) came from an earlier
configuration and is no longer the best performer, so it has been removed rather than
left in place to mislead.

What is saved here, per descriptor set:

  nusvr.joblib        the complete Nu-SVR pipeline: median imputation, variance gate,
                      standardisation, PCA, Nu-SVR. Takes raw descriptors in, predicts
                      pIC50 out.
  dnn.keras           the deep network, which expects PCA-transformed input
  dnn_preprocessor.joblib  the transformer chain feeding that network
  ridge_meta.joblib   the meta-learner combining the two base predictions
  metadata.json       hyperparameters, metrics, descriptor order and library versions

Two configurations are written:

  md_core89   the best model in the study (structure-disjoint test R2 0.798), and the one
              to use for prediction
  md272       the manuscript's original 272-descriptor set, for continuity with the
              published work

A pickled scikit-learn estimator is only guaranteed to load under the version that wrote
it, so metadata.json records the versions and src/predict.py checks them at load time.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import joblib
import numpy as np
import sklearn

import datasets
from hybrid_pipeline import DnnRegressor, fit_nusvr, make_splits, metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'models')


def save_one(key: str, regime: str, random_state: int = 1):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.base import clone

    X, y, meta = datasets.load(key)
    groups = datasets.groups_for(meta)
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    sp = make_splits(len(yv), groups=groups, regime=regime, random_state=random_state)

    d = os.path.join(OUT, key)
    os.makedirs(d, exist_ok=True)
    print(f'\n{key}: {meta["label"]}  ({regime}, random_state={random_state})')

    svr, pick, _ = fit_nusvr(Xv, yv, sp, selection='cv_r2', cv=5, verbose=False)
    joblib.dump(svr, os.path.join(d, 'nusvr.joblib'), compress=3)
    print(f'  Nu-SVR  {pick["params"]}, {pick["n_components"]} PCs')

    dnn = DnnRegressor(n_components=pick['n_components'], seed=42)
    dnn.fit(Xv[sp['train']], yv[sp['train']])
    dnn.model_.save(os.path.join(d, 'dnn.keras'))
    joblib.dump(dnn.pre_, os.path.join(d, 'dnn_preprocessor.joblib'), compress=3)

    kf = KFold(5, shuffle=True, random_state=42)
    oof_svr = cross_val_predict(clone(svr), Xv[sp['train']], yv[sp['train']], cv=kf,
                                n_jobs=-1)
    oof_dnn = cross_val_predict(DnnRegressor(n_components=pick['n_components'], seed=42),
                                Xv[sp['train']], yv[sp['train']], cv=kf)
    ridge = Ridge(alpha=1.0).fit(np.c_[oof_svr, oof_dnn], yv[sp['train']])
    joblib.dump(ridge, os.path.join(d, 'ridge_meta.joblib'), compress=3)

    def hybrid(idx):
        return ridge.predict(np.c_[svr.predict(Xv[idx]), dnn.predict(Xv[idx])])

    perf = {}
    for name, pred in (('nusvr', svr.predict), ('dnn', dnn.predict), ('hybrid', hybrid)):
        perf[name] = {sub: {k: round(v, 4) for k, v in
                            metrics(yv[sp[sub]],
                                    pred(sp[sub]) if name == 'hybrid'
                                    else pred(Xv[sp[sub]])).items()
                            if isinstance(v, float)}
                      for sub in ('train', 'test', 'holdout')}

    import tensorflow as tf
    md = dict(
        dataset=key, label=meta['label'], split_regime=regime,
        random_state=random_state, n_features=int(Xv.shape[1]),
        descriptor_order=list(X.columns),
        nusvr_params=pick['params'], n_components=int(pick['n_components']),
        ridge_coef=[float(c) for c in ridge.coef_],
        ridge_intercept=float(ridge.intercept_),
        split_sizes={k: int(len(v)) for k, v in sp.items()},
        performance=perf,
        versions=dict(sklearn=sklearn.__version__, numpy=np.__version__,
                      tensorflow=tf.__version__, joblib=joblib.__version__),
        note=('nusvr.joblib takes raw descriptors in descriptor_order and returns pIC50. '
              'dnn.keras expects the output of dnn_preprocessor.joblib. '
              'ridge_meta.joblib combines [nusvr_prediction, dnn_prediction].'),
    )
    json.dump(md, open(os.path.join(d, 'metadata.json'), 'w'), indent=1)

    for m in ('nusvr', 'dnn', 'hybrid'):
        print(f'  {m:7s} test R2 {perf[m]["test"]["r2"]:+.4f}   '
              f'holdout R2 {perf[m]["holdout"]["r2"]:+.4f}')
    return perf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--copy-to-repo', action='store_true',
                    help='also copy models/ into github_repo/')
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    save_one('md_core89', 'structure_disjoint')
    save_one('md272', 'random')

    if a.copy_to_repo:
        dst = os.path.join(ROOT, 'github_repo', 'models')
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(OUT, dst)
        n = sum(len(f) for _, _, f in os.walk(dst))
        size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(dst) for f in fs)
        print(f'\ncopied {n} files ({size/1e6:.1f} MB) into github_repo/models')


if __name__ == '__main__':
    main()
