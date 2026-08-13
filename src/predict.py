"""Apply a saved model to new molecules.

    python src/predict.py --input my_descriptors.csv --output predictions.csv

The input must be a CSV whose columns include every descriptor the model was trained on;
the order does not matter, since columns are reindexed to the training order recorded in
metadata.json. Any extra columns are ignored and any missing ones are reported rather
than silently imputed, because a silently absent descriptor is a wrong prediction.

Default model is md_core89, the 89-descriptor molecular dynamics core, which is the best
performing configuration in the study (structure-disjoint test R2 0.798, holdout 0.844).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')


def load(model_dir):
    md = json.load(open(os.path.join(model_dir, 'metadata.json')))

    import sklearn
    if sklearn.__version__ != md['versions']['sklearn']:
        warnings.warn(
            f"models were written with scikit-learn {md['versions']['sklearn']} but "
            f"{sklearn.__version__} is installed. Pickled estimators are not guaranteed "
            f"to load across versions; re-run src/save_models.py if predictions look "
            f"wrong.")

    parts = {'meta': md,
             'nusvr': joblib.load(os.path.join(model_dir, 'nusvr.joblib'))}
    ridge = os.path.join(model_dir, 'ridge_meta.joblib')
    dnn = os.path.join(model_dir, 'dnn.keras')
    pre = os.path.join(model_dir, 'dnn_preprocessor.joblib')
    if os.path.exists(ridge) and os.path.exists(dnn) and os.path.exists(pre):
        from tensorflow import keras
        parts['ridge'] = joblib.load(ridge)
        parts['dnn'] = keras.models.load_model(dnn)
        parts['dnn_pre'] = joblib.load(pre)
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='CSV of descriptors')
    ap.add_argument('--output', default='predictions.csv')
    ap.add_argument('--model', default='md_core89',
                    help='subdirectory of models/ (md_core89 or md272)')
    ap.add_argument('--id-column', default=None,
                    help='column carrying molecule identifiers, copied to the output')
    a = ap.parse_args()

    model_dir = os.path.join(MODELS, a.model)
    if not os.path.isdir(model_dir):
        sys.exit(f'no model at {model_dir}; run src/save_models.py first')
    parts = load(model_dir)
    md = parts['meta']
    need = md['descriptor_order']

    df = pd.read_csv(a.input)
    ids = df[a.id_column] if a.id_column and a.id_column in df.columns else \
        pd.Series([f'mol_{i + 1}' for i in range(len(df))])

    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f'input is missing {len(missing)} required descriptors, '
                 f'first few: {missing[:8]}')
    extra = [c for c in df.columns if c not in need and c != a.id_column]
    if extra:
        print(f'ignoring {len(extra)} column(s) not used by this model')

    X = df.reindex(columns=need).to_numpy(float)
    print(f"model {a.model}: {md['label']}")
    print(f"  {X.shape[0]} molecules x {X.shape[1]} descriptors")

    out = pd.DataFrame({'id': ids.values})
    out['pred_nusvr'] = parts['nusvr'].predict(X)
    if 'dnn' in parts:
        Z = parts['dnn_pre'].transform(X)
        out['pred_dnn'] = parts['dnn'].predict(Z, verbose=0).ravel()
        out['pred_hybrid'] = parts['ridge'].predict(
            np.c_[out['pred_nusvr'], out['pred_dnn']])
        out['pred_pIC50'] = out['pred_nusvr']   # the recommended single estimate
    else:
        out['pred_pIC50'] = out['pred_nusvr']

    out.to_csv(a.output, index=False)
    print(f"wrote {a.output}")
    print(f"  predicted pIC50: mean {out['pred_pIC50'].mean():.3f}, "
          f"range {out['pred_pIC50'].min():.3f} to {out['pred_pIC50'].max():.3f}")
    print('\nNote: the training set spans pIC50 3.95 to 9.72 across four targets. '
          'Predictions for molecules outside that chemical space, or against other '
          'proteins, are extrapolation.')


if __name__ == '__main__':
    main()
