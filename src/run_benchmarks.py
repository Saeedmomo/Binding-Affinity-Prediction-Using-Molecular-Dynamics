"""Driver: run the hybrid harness over every feature set x both split regimes."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import datasets
from hybrid_pipeline import run_one, tidy_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
PRED = os.path.join(RES, 'predictions')


def versions():
    import sklearn
    import scipy
    v = dict(python=sys.version.split()[0], platform=platform.platform(),
             numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
             scipy=scipy.__version__)
    try:
        import tensorflow as tf
        import keras
        v['tensorflow'] = tf.__version__
        v['keras'] = keras.__version__
    except Exception as e:
        v['tensorflow'] = f'unavailable: {e}'
    try:
        v['git_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return v


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items() if not k.startswith('_')}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='md272,md272_clean,padel,fused')
    ap.add_argument('--regimes', default='random,structure_disjoint')
    ap.add_argument('--random-state', type=int, default=1)
    ap.add_argument('--selection', default='cv_r2', choices=['cv_r2', 'w_new_test'])
    ap.add_argument('--n-perm', type=int, default=100)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--no-q2', action='store_true')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    os.makedirs(PRED, exist_ok=True)
    keys = [k.strip() for k in a.datasets.split(',') if k.strip()]
    regimes = [r.strip() for r in a.regimes.split(',') if r.strip()]

    tidy, nested, manifest = [], {}, dict(versions=versions(), args=vars(a),
                                          started=time.strftime('%Y-%m-%d %H:%M:%S'),
                                          inputs={})
    for key in keys:
        X, y, meta = datasets.load(key)
        groups = datasets.groups_for(meta)
        names = datasets.names_for(meta)
        manifest['inputs'][key] = {k: v for k, v in meta.items() if k != 'row_ids'}
        Xv, yv = X.to_numpy(float), y.to_numpy(float)
        print(f'\n{"=" * 90}\n{key}: {meta["label"]}  '
              f'({meta["n_rows"]} x {meta["n_features"]}, '
              f'{meta["n_nan_cells"]} NaN cells)\n{"=" * 90}', flush=True)

        for regime in regimes:
            t0 = time.time()
            res = run_one(Xv, yv, groups=groups, regime=regime,
                          random_state=a.random_state, selection=a.selection,
                          n_boot=a.n_boot, n_perm=a.n_perm, do_q2=not a.no_q2,
                          verbose=True)
            tidy.append(tidy_rows(res, key, meta['label']))
            nested[f'{key}__{regime}'] = jsonable(res)

            rows = []
            for sub in ('train', 'test', 'holdout'):
                p = res['_predictions'][sub]
                for j, ridx in enumerate(p['idx']):
                    rows.append(dict(row_index=int(names['row_index'].iloc[ridx]),
                                     mol_name=names['mol_name'].iloc[ridx],
                                     smiles=names['smiles'].iloc[ridx],
                                     target=names['target'].iloc[ridx],
                                     subset=sub, y_true=float(p['y_true'][j]),
                                     pred_nusvr=float(p['nusvr'][j]),
                                     pred_dnn=float(p['dnn'][j]),
                                     pred_hybrid=float(p['hybrid'][j])))
            pd.DataFrame(rows).to_csv(
                os.path.join(PRED, f'{key}__{regime}.csv'), index=False)

            m = res['metrics']
            print(f'  -> {regime:19s} '
                  f'SVR test={m["nusvr"]["test"]["r2"]:+.4f} hold={m["nusvr"]["holdout"]["r2"]:+.4f} | '
                  f'DNN test={m["dnn"]["test"]["r2"]:+.4f} | '
                  f'HYB test={m["hybrid"]["test"]["r2"]:+.4f} hold={m["hybrid"]["holdout"]["r2"]:+.4f} '
                  f'[{time.time() - t0:.0f}s]', flush=True)

    suffix = f'_{a.tag}' if a.tag else ''
    df = pd.concat(tidy, ignore_index=True)
    out_csv = os.path.join(RES, f'metrics_all{suffix}.csv')
    if os.path.exists(out_csv) and a.tag == '':
        prev = pd.read_csv(out_csv)
        keep = ~prev.set_index(['dataset', 'split_regime']).index.isin(
            df.set_index(['dataset', 'split_regime']).index)
        df = pd.concat([prev[keep.tolist()], df], ignore_index=True)
    df.to_csv(out_csv, index=False)

    njson = os.path.join(RES, f'metrics_all{suffix}.json')
    old = json.load(open(njson)) if os.path.exists(njson) and a.tag == '' else {}
    old.update(nested)
    json.dump(old, open(njson, 'w'), indent=1)

    manifest['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    mf = os.path.join(RES, f'run_manifest{suffix}.json')
    oldm = json.load(open(mf)) if os.path.exists(mf) and a.tag == '' else {}
    oldm.update({manifest['started']: manifest})
    json.dump(oldm, open(mf, 'w'), indent=1)

    print(f'\nwrote {out_csv}\n      {njson}\n      {mf}')
    piv = df[(df.subset.isin(['test', 'holdout'])) & (df.model == 'hybrid')].pivot_table(
        index=['dataset', 'split_regime'], columns='subset', values='r2')
    print('\nhybrid R2 summary:')
    print(piv.round(4).to_string())


if __name__ == '__main__':
    main()
