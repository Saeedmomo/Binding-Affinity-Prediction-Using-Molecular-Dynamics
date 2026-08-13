"""Re-run the four deep-network variants (manuscript correction A3).

Why this exists. Table 1 of 'Hybrid model.docx' lists four architectures but only two
distinct sets of numbers: DNN-1 is bit-identical to DNN-3 across all seven columns, and
DNN-2 to DNN-4. Two different architectures cannot produce identical metrics, so the
table cannot be right. The original runs are unrecoverable: no .keras files, no
keras-tuner directories and no logs survive anywhere on this machine, and the source
scripts point at 'D:/chua_trajectory/NN_model/' and 'C:/Users/HUAWEI/Desktop/', i.e. a
different computer.

What this does instead. The four architectures are rebuilt exactly as tabulated in
NN_Model.docx (layers, maximum neurons per layer, dropout range, L2 penalty, learning
rate, optimiser, batch size) and each is trained under the single shared split protocol
used everywhere else in this work. The original study selected these architectures by
Bayesian optimisation with keras-tuner; re-running that search would not reproduce the
published numbers either, and would add a second source of run-to-run variation on top
of weight initialisation. Fixing the architectures at their reported optima is therefore
both more faithful to what was reported and more informative.

Each variant is trained with several random seeds so the table can carry a mean and a
standard deviation. That is the substantive point: it lets the manuscript state whether
the four architectures are genuinely distinguishable on 84 training molecules, rather
than presenting four rows that imply a ranking the data cannot support.

Output: results/dnn_variants.csv and results/dnn_variants_summary.csv
"""
from __future__ import annotations

import argparse
import os
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import pandas as pd

import datasets
from hybrid_pipeline import _SafePCA, make_splits, metrics

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')

# Exactly as tabulated in NN_Model.docx, Table 1.
VARIANTS = {
    'DNN-1 Initial': dict(
        units=(192, 192, 160, 128), dropout=(0.4, 0.5), l2=0.0,
        lr=0.005, optimizer='adam', batch=32, cosine=False),
    'DNN-2 Regularised': dict(
        units=(320, 320, 256, 192, 128), dropout=(0.2, 0.4), l2=1e-4,
        lr=0.001, optimizer='adamw', batch=32, cosine=True),
    'DNN-3 Optimised': dict(
        units=(512, 448, 384, 320, 256, 192), dropout=(0.2, 0.3), l2=1e-4,
        lr=0.001, optimizer='adamw', batch=32, cosine=True),
    'DNN-4 Fine-tuned': dict(
        units=(512, 448, 384, 256, 192), dropout=(0.2, 0.25), l2=1e-3,
        lr=0.0005, optimizer='adamw', batch=16, cosine=True, noise=0.005),
}


def build(spec, n_in, seed):
    import tensorflow as tf
    from tensorflow import keras
    keras.utils.set_random_seed(seed)
    reg = keras.regularizers.l2(spec['l2']) if spec['l2'] else None
    lo, hi = spec['dropout']
    n = len(spec['units'])
    m = keras.Sequential([keras.layers.Input(shape=(n_in,))])
    for i, u in enumerate(spec['units']):
        # dropout ramps linearly across the stack, spanning the tabulated range
        p = lo + (hi - lo) * (i / max(n - 1, 1))
        m.add(keras.layers.Dense(u, activation='relu', kernel_regularizer=reg))
        m.add(keras.layers.BatchNormalization())
        m.add(keras.layers.Dropout(p))
    m.add(keras.layers.Dense(1, activation='linear'))

    lr = spec['lr']
    if spec.get('cosine'):
        lr = keras.optimizers.schedules.CosineDecay(spec['lr'], decay_steps=2000,
                                                    alpha=1e-4)
    opt = (keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5)
           if spec['optimizer'] == 'adamw' else keras.optimizers.Adam(learning_rate=lr))
    m.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return m


def run_variant(name, spec, X, y, sp, seed, n_comp, epochs=200, patience=12):
    from tensorflow import keras
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pre = Pipeline([('imp', SimpleImputer(strategy='median')),
                    ('sc', StandardScaler()),
                    ('pca', _SafePCA(n_components=n_comp, random_state=seed))])
    Ztr = pre.fit_transform(X[sp['train']])
    if spec.get('noise'):
        rng = np.random.default_rng(seed)
        Ztr = Ztr + rng.normal(0, spec['noise'], Ztr.shape)

    model = build(spec, Ztr.shape[1], seed)
    es = keras.callbacks.EarlyStopping(monitor='loss', patience=patience,
                                       restore_best_weights=True)
    model.fit(Ztr, y[sp['train']], epochs=epochs, batch_size=spec['batch'],
              callbacks=[es], verbose=0, shuffle=True)

    out = {'model': name, 'seed': seed,
           'n_params': int(model.count_params()),
           'n_neurons': int(sum(spec['units']))}
    for sub in ('train', 'test', 'holdout'):
        Z = pre.transform(X[sp[sub]])
        m = metrics(y[sp[sub]], model.predict(Z, verbose=0).ravel())
        out[f'r2_{sub}'] = m['r2']
        out[f'rmse_{sub}'] = m['rmse']
    keras.backend.clear_session()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='md272_clean')
    ap.add_argument('--regime', default='random')
    ap.add_argument('--random-state', type=int, default=1)
    ap.add_argument('--n-components', type=int, default=17)
    ap.add_argument('--seeds', type=int, default=5)
    a = ap.parse_args()

    X, y, meta = datasets.load(a.dataset)
    groups = datasets.groups_for(meta)
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    sp = make_splits(len(yv), groups=groups, regime=a.regime,
                     random_state=a.random_state)
    print(f'{meta["label"]}  train={len(sp["train"])} test={len(sp["test"])} '
          f'holdout={len(sp["holdout"])}  {a.n_components} PCs  '
          f'{a.seeds} seeds per architecture\n', flush=True)

    rows = []
    for name, spec in VARIANTS.items():
        for seed in range(a.seeds):
            r = run_variant(name, spec, Xv, yv, sp, seed, a.n_components)
            rows.append(r)
            print(f'  {name:20s} seed={seed}  train={r["r2_train"]:+.4f} '
                  f'test={r["r2_test"]:+.4f} holdout={r["r2_holdout"]:+.4f}', flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'dnn_variants.csv'), index=False)

    g = df.groupby('model')
    summ = pd.DataFrame({
        'Neurons': g['n_neurons'].first(),
        'Parameters': g['n_params'].first(),
        'R2 train': g['r2_train'].mean().round(4),
        'R2 train SD': g['r2_train'].std().round(4),
        'R2 test': g['r2_test'].mean().round(4),
        'R2 test SD': g['r2_test'].std().round(4),
        'R2 holdout': g['r2_holdout'].mean().round(4),
        'R2 holdout SD': g['r2_holdout'].std().round(4),
        'RMSE test': g['rmse_test'].mean().round(4),
    }).reset_index().rename(columns={'model': 'Architecture'})
    summ.to_csv(os.path.join(RES, 'dnn_variants_summary.csv'), index=False)

    print('\n' + summ.to_string(index=False))
    spread = df.groupby('model')['r2_test'].agg(['min', 'max'])
    print('\nrun-to-run spread in test R2 within a single architecture:')
    print((spread['max'] - spread['min']).round(4).to_string())
    print('\nbetween-architecture spread of the means: '
          f'{summ["R2 test"].max() - summ["R2 test"].min():.4f}')


if __name__ == '__main__':
    main()
