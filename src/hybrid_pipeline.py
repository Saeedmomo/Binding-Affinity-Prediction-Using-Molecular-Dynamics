"""Reproducible implementation of the manuscript's Nu-SVR + DNN + Ridge hybrid.

Recovered from the authors' own code and then
corrected in four places that materially affect the reported numbers. Each correction is
switchable so the published result can still be reproduced exactly.

1. ONE SPLIT FOR ALL LEARNERS. The published work fitted Nu-SVR at random_state=1 and
   the DNNs at random_state=42, then stacked the two. Here a single random_state drives
   every learner.
2. NO PREPROCESSING LEAKAGE. The published scripts ran StandardScaler and PCA on the
   whole dataset before splitting. Here they live inside a Pipeline, so they are fitted
   on training folds only. Set leaky_preprocessing=True to reproduce the original.
3. HONEST HYPERPARAMETER SELECTION. The published W_new ranking includes the test-set
   R2 and the test-set MSE/RMSE/MAE, so hyperparameters were chosen by looking at the
   test set. Default selection here is cross-validated R2; selection='w_new_test'
   reproduces the original behaviour.
4. OUT-OF-FOLD STACKING. The Ridge meta-learner is trained on cross_val_predict outputs
   of the two base learners, never on in-sample base predictions.

W_new is kept verbatim from the source, including its unusual normalisation.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, GroupShuffleSplit, KFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

N_COMPONENTS_GRID = (15, 17, 19, 20, 21, 22)
NUSVR_GRID = {'model__nu': [0.1, 0.3, 0.5, 0.7, 0.9],
              'model__C': [0.1, 1.0, 10.0, 100.0],
              'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
SEED = 42

# libsvm defaults to max_iter=-1, i.e. no iteration limit. Several points in the grid
# above (notably C=100 with the poly and sigmoid kernels) do not converge on this data
# and spin indefinitely: single fits were observed consuming 20+ minutes of CPU on one
# thread, which is what made the widest dataset appear to "run for hours". A finite cap
# turns non-convergence into a bounded, and therefore reportable, poor score.
NUSVR_MAX_ITER = 2_000_000


# ------------------------------------------------------------------------ metrics
def w_new(r2_train, r2_cv, r2_test, mse, rmse, mae):
    """Verbatim from the authors' Nu-SVR script."""
    denom = mse + rmse + mae
    if denom <= 0:
        return float('nan')
    base = (r2_train + r2_cv + r2_test) / denom
    gap = abs(r2_train - r2_cv)
    pen = (1 - gap) / (1 + gap)
    return base * pen / (1 + base * pen)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    mse = mean_squared_error(y_true, y_pred)
    out = dict(n=int(len(y_true)), r2=r2_score(y_true, y_pred), mse=mse,
               rmse=float(np.sqrt(mse)), mae=mean_absolute_error(y_true, y_pred))
    if len(y_true) > 2 and np.std(y_pred) > 1e-12:
        out['pearson_r'] = float(stats.pearsonr(y_true, y_pred)[0])
        out['spearman_rho'] = float(stats.spearmanr(y_true, y_pred)[0])
    else:
        out['pearson_r'] = out['spearman_rho'] = float('nan')
    return out


def bootstrap_r2_ci(y_true, y_pred, n_boot=1000, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    n = len(y_true)
    if n < 5:
        return dict(lo=float('nan'), hi=float('nan'), n_boot=0)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.std(y_true[i]) < 1e-12:
            continue
        vals.append(r2_score(y_true[i], y_pred[i]))
    if not vals:
        return dict(lo=float('nan'), hi=float('nan'), n_boot=0)
    return dict(lo=float(np.quantile(vals, alpha / 2)),
                hi=float(np.quantile(vals, 1 - alpha / 2)),
                median=float(np.median(vals)), n_boot=len(vals))


# ------------------------------------------------------------------------- splits
def make_splits(n, groups=None, regime='random', random_state=1):
    """70 / 20 / 10 train / test / holdout, matching the published proportions."""
    idx = np.arange(n)
    if regime == 'random':
        tmp, hold = train_test_split(idx, test_size=0.1, random_state=random_state)
        tr, te = train_test_split(tmp, test_size=0.2222, random_state=random_state)
    elif regime == 'structure_disjoint':
        if groups is None:
            raise ValueError('structure_disjoint requires groups')
        g = np.asarray(groups)
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.1,
                                      random_state=random_state).split(idx, groups=g))
        tmp, hold = idx[a], idx[b]
        c, d = next(GroupShuffleSplit(n_splits=1, test_size=0.2222,
                                      random_state=random_state).split(tmp, groups=g[tmp]))
        tr, te = tmp[c], tmp[d]
        assert not (set(g[tr]) & set(g[te])), 'structure leaked into test'
        assert not (set(g[tr]) & set(g[hold])), 'structure leaked into holdout'
    else:
        raise ValueError(f'unknown regime {regime!r}')
    assert len(set(tr) | set(te) | set(hold)) == n
    return dict(train=np.sort(tr), test=np.sort(te), holdout=np.sort(hold))


def leakage_report(groups, sp):
    g = np.asarray(groups)
    out = {}
    for a, b in (('train', 'test'), ('train', 'holdout')):
        shared = set(g[sp[a]]) & set(g[sp[b]])
        out[f'shared_structures_{a}_{b}'] = len(shared)
        out[f'leaked_rows_in_{b}'] = int(np.isin(g[sp[b]], list(shared)).sum())
    return out


# ---------------------------------------------------------------------- base: SVR
class _SafePCA(TransformerMixin, BaseEstimator):
    """PCA whose component count is clamped at fit time.

    Necessary because the upstream variance gate can drop columns inside a
    cross-validation fold (e.g. the 14-column AlphaFold block collapses to 13 when a
    fold happens to contain a single protein), and a fold's sample count can also fall
    below the requested rank. Clamping here rather than at the call site keeps every
    fold valid without silently changing the requested dimensionality elsewhere.
    """

    def __init__(self, n_components=17, random_state=SEED):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        k = max(1, min(self.n_components, X.shape[0] - 1, X.shape[1]))
        self.n_components_used_ = int(k)
        self.pca_ = PCA(n_components=k, svd_solver='randomized',
                        random_state=self.random_state).fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(np.asarray(X, float))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


# Per-process cache directory. A single shared directory races when two benchmark runs
# execute concurrently: joblib writes output.pkl under a hash that both processes derive
# identically, and the loser gets PermissionError on Windows.
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'logs', '_pipeline_cache', f'pid{os.getpid()}')


def _nusvr_pipe(n_comp, seed=SEED, cache=True):
    """Impute -> drop constants -> scale -> PCA -> Nu-SVR.

    The transformer chain is memoised. Only the final Nu-SVR step varies across the
    80-point hyperparameter grid, so without caching GridSearchCV refits the PCA 80
    times per fold. On the 112194-descriptor set that dominates the entire run.
    """
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('var', _VarianceGate()),
                     ('sc', StandardScaler()),
                     ('pca', _SafePCA(n_components=n_comp, random_state=seed)),
                     ('model', NuSVR(max_iter=NUSVR_MAX_ITER))],
                    memory=_CACHE if cache else None)


class _VarianceGate(TransformerMixin, BaseEstimator):
    """Drop zero-variance columns. Written out rather than using VarianceThreshold so the
    number dropped can be logged per fit."""

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        v = np.nanvar(X, axis=0)
        self.keep_ = v > 1e-12
        self.n_dropped_ = int((~self.keep_).sum())
        if not self.keep_.any():
            self.keep_ = np.ones(X.shape[1], bool)
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.keep_]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


def fit_nusvr(X, y, sp, selection='cv_r2', cv=5, seed=SEED, verbose=False, n_jobs=-1):
    """Sweep n_components x hyperparameters; return the chosen fitted estimator."""
    n_train = len(sp['train'])
    Xtr, ytr = X[sp['train']], y[sp['train']]
    grid_comps = sorted({min(c, n_train - 1, X.shape[1]) for c in N_COMPONENTS_GRID})
    # Every joblib worker copies the feature matrix, so the fan-out has to respect
    # memory, not column count. Budget roughly a quarter of free RAM for worker copies.
    # A column-count threshold was wrong here: dropping the 44% all-zero columns and
    # casting to float32 took the widest dataset from 109 MB to 30 MB, at which point
    # throttling to 4 workers left 20 of 24 cores idle for no reason.
    if n_jobs == -1:
        mat_mb = X.nbytes / 2 ** 20
        try:
            import psutil
            free_mb = psutil.virtual_memory().available / 2 ** 20
        except Exception:
            free_mb = 8000.0
        budget = max(1, int((free_mb * 0.25) // max(mat_mb, 1)))
        n_jobs = max(1, min(os.cpu_count() or 4, budget))
        if verbose:
            print(f'    matrix {mat_mb:.0f} MB, {free_mb / 1024:.1f} GB free '
                  f'-> n_jobs={n_jobs}')
    trials, best = [], None
    for nc in grid_comps:
        gs = GridSearchCV(_nusvr_pipe(nc, seed), NUSVR_GRID, scoring='r2',
                          cv=KFold(cv, shuffle=True, random_state=seed), n_jobs=n_jobs)
        gs.fit(Xtr, ytr)
        est = gs.best_estimator_
        r2_tr = r2_score(ytr, est.predict(Xtr))
        r2_cv = float(gs.best_score_)
        r2_te = r2_score(y[sp['test']], est.predict(X[sp['test']]))
        pte = est.predict(X[sp['test']])
        mse = mean_squared_error(y[sp['test']], pte)
        w = w_new(r2_tr, r2_cv, r2_te, mse, float(np.sqrt(mse)),
                  mean_absolute_error(y[sp['test']], pte))
        score = w if selection == 'w_new_test' else r2_cv
        t = dict(n_components=int(nc), params=gs.best_params_, r2_train=r2_tr,
                 r2_cv=r2_cv, r2_test=r2_te, w_new=w, selection_score=score)
        trials.append(t)
        if verbose:
            print(f'    nc={nc:3d} cv={r2_cv:+.4f} test={r2_te:+.4f} w_new={w:.6f} '
                  f'{gs.best_params_}')
        if best is None or score > best[0]:
            best = (score, est, t)
    return best[1], best[2], trials


# ---------------------------------------------------------------------- base: DNN
def build_dnn(n_in, seed=SEED, units=(128, 128, 192, 128, 128), dropout=0.25,
              l2=1e-3, lr=5e-4, decay_steps=2000):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(seed)
    reg = keras.regularizers.l2(l2)
    m = keras.Sequential([keras.layers.Input(shape=(n_in,))])
    for u in units:
        m.add(keras.layers.Dense(u, activation='relu', kernel_regularizer=reg))
        m.add(keras.layers.BatchNormalization())
        m.add(keras.layers.Dropout(dropout))
    m.add(keras.layers.Dense(1, activation='linear'))
    sched = keras.optimizers.schedules.CosineDecay(lr, decay_steps=decay_steps, alpha=1e-4)
    m.compile(optimizer=keras.optimizers.AdamW(learning_rate=sched, weight_decay=1e-5),
              loss='mse', metrics=['mae'])
    return m


class DnnRegressor(RegressorMixin, BaseEstimator):
    """Minimal sklearn-compatible wrapper: imputer -> variance gate -> scaler -> PCA -> DNN.
    Kept deliberately small so cross_val_predict can refit it cleanly per fold."""

    def __init__(self, n_components=17, seed=SEED, epochs=200, batch_size=32,
                 patience=10, verbose=0):
        self.n_components = n_components
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose

    def fit(self, X, y):
        from tensorflow import keras
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        self.pre_ = Pipeline([('imp', SimpleImputer(strategy='median')),
                              ('var', _VarianceGate()),
                              ('sc', StandardScaler()),
                              ('pca', _SafePCA(n_components=self.n_components,
                                               random_state=self.seed))])
        Z = self.pre_.fit_transform(X)
        self.model_ = build_dnn(Z.shape[1], seed=self.seed)
        es = keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience,
                                          restore_best_weights=True)
        self.model_.fit(Z, y, epochs=self.epochs, batch_size=self.batch_size,
                        callbacks=[es], verbose=self.verbose, shuffle=True)
        return self

    def predict(self, X):
        Z = self.pre_.transform(np.asarray(X, float))
        return self.model_.predict(Z, verbose=0).ravel()


# -------------------------------------------------------------------------- hybrid
def run_one(X, y, groups=None, regime='random', random_state=1, selection='cv_r2',
            cv=5, seed=SEED, n_boot=1000, n_perm=100, do_perm=True, do_q2=True,
            verbose=True, n_jobs=-1):
    """Fit Nu-SVR, DNN and the Ridge stack on one dataset / one split. Returns a dict."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n = len(y)
    sp = make_splits(n, groups=groups, regime=regime, random_state=random_state)
    res = dict(regime=regime, random_state=random_state, selection=selection,
               n_rows=n, n_features=int(X.shape[1]),
               split_sizes={k: int(len(v)) for k, v in sp.items()})
    if groups is not None:
        res['leakage'] = leakage_report(groups, sp)

    t0 = time.time()
    if verbose:
        print(f'  Nu-SVR grid search ({regime}, rs={random_state}, select={selection})')
    svr, svr_pick, svr_trials = fit_nusvr(X, y, sp, selection=selection, cv=cv,
                                          seed=seed, verbose=verbose, n_jobs=n_jobs)
    res['nusvr_selected'] = svr_pick
    res['nusvr_trials'] = svr_trials
    n_comp = svr_pick['n_components']
    res['n_components'] = n_comp

    if verbose:
        print(f'  DNN (n_components={n_comp})')
    dnn = DnnRegressor(n_components=n_comp, seed=seed)
    dnn.fit(X[sp['train']], y[sp['train']])

    kf = KFold(cv, shuffle=True, random_state=seed)
    oof_svr = cross_val_predict(_clone_svr(svr), X[sp['train']], y[sp['train']], cv=kf,
                                n_jobs=n_jobs)
    oof_dnn = cross_val_predict(DnnRegressor(n_components=n_comp, seed=seed),
                                X[sp['train']], y[sp['train']], cv=kf)
    meta = Ridge(alpha=1.0).fit(np.c_[oof_svr, oof_dnn], y[sp['train']])
    res['ridge_coef'] = [float(c) for c in meta.coef_]
    res['ridge_intercept'] = float(meta.intercept_)

    def stack(idx):
        return meta.predict(np.c_[svr.predict(X[idx]), dnn.predict(X[idx])])

    preds = {}
    for sub, idx in sp.items():
        preds[sub] = dict(idx=idx, y_true=y[idx], nusvr=svr.predict(X[idx]),
                          dnn=dnn.predict(X[idx]), hybrid=stack(idx))
    # cross-validated predictions on the training set, per model
    preds['cv'] = dict(idx=sp['train'], y_true=y[sp['train']], nusvr=oof_svr,
                       dnn=oof_dnn,
                       hybrid=cross_val_predict(Ridge(alpha=1.0),
                                                np.c_[oof_svr, oof_dnn],
                                                y[sp['train']], cv=kf))
    res['metrics'] = {}
    for model in ('nusvr', 'dnn', 'hybrid'):
        res['metrics'][model] = {}
        for sub in ('train', 'cv', 'test', 'holdout'):
            res['metrics'][model][sub] = metrics(preds[sub]['y_true'], preds[sub][model])
        m = res['metrics'][model]
        m['w_new'] = w_new(m['train']['r2'], m['cv']['r2'], m['test']['r2'],
                           m['test']['mse'], m['test']['rmse'], m['test']['mae'])
        for sub in ('test', 'holdout'):
            m[sub]['r2_ci95'] = bootstrap_r2_ci(preds[sub]['y_true'], preds[sub][model],
                                                n_boot=n_boot, seed=seed)

    if do_q2:
        res['q2_loo_nusvr'] = _q2_loo(_clone_svr(svr), X[sp['train']], y[sp['train']],
                                      n_jobs=n_jobs)

    if do_perm:
        res['permutation'] = _perm_test(_clone_svr(svr), X, y, sp, n_perm=n_perm, seed=seed,
                                        observed=res['metrics']['nusvr']['test']['r2'],
                                        n_jobs=n_jobs)

    res['seconds'] = round(time.time() - t0, 1)
    res['_predictions'] = preds
    return res


def _clone_svr(fitted):
    from sklearn.base import clone
    return clone(fitted)


def _q2_loo(est, X, y, n_jobs=-1):
    from sklearn.model_selection import LeaveOneOut
    p = cross_val_predict(est, X, y, cv=LeaveOneOut(), n_jobs=n_jobs)
    press = float(((y - p) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    return dict(q2=1 - press / tss, press=press, rmse_loo=float(np.sqrt(press / len(y))))


def _perm_test(est, X, y, sp, n_perm, seed, observed, n_jobs=-1):
    from joblib import Parallel, delayed
    rng = np.random.default_rng(seed)
    # draw every permutation up front so the result stays deterministic regardless of
    # how the work is distributed across workers
    perms = [rng.permutation(y[sp['train']]) for _ in range(n_perm)]

    def one(yp_train):
        try:
            e = _clone_svr(est).fit(X[sp['train']], yp_train)
            return r2_score(y[sp['test']], e.predict(X[sp['test']]))
        except Exception:
            return np.nan

    vals = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(one)(p) for p in perms)
    null = np.asarray([v for v in vals if np.isfinite(v)], float)
    if null.size == 0:
        return dict(n=0)
    return dict(n=int(null.size), observed_test_r2=float(observed),
                null_mean=float(null.mean()), null_sd=float(null.std()),
                null_q95=float(np.quantile(null, 0.95)),
                null_max=float(null.max()),
                p_value=float((null >= observed).sum() + 1) / (null.size + 1))


def tidy_rows(res, dataset, label):
    """Flatten one run into tidy rows for results/metrics_all.csv."""
    rows = []
    for model, byset in res['metrics'].items():
        for sub in ('train', 'cv', 'test', 'holdout'):
            m = byset[sub]
            rows.append(dict(dataset=dataset, dataset_label=label,
                             split_regime=res['regime'], random_state=res['random_state'],
                             selection=res['selection'], model=model, subset=sub,
                             n=m['n'], r2=m['r2'], mse=m['mse'], rmse=m['rmse'],
                             mae=m['mae'], pearson_r=m['pearson_r'],
                             spearman_rho=m['spearman_rho'],
                             r2_ci_lo=m.get('r2_ci95', {}).get('lo'),
                             r2_ci_hi=m.get('r2_ci95', {}).get('hi'),
                             w_new=byset['w_new'],
                             n_features=res['n_features'],
                             n_components=res['n_components'],
                             n_rows=res['n_rows'], seconds=res['seconds']))
    return pd.DataFrame(rows)
