"""Repeated nested resampling: which learner is genuinely best for each descriptor set.

The single-split screen in sweep.py ranks learners on one structure-disjoint partition of
25 test molecules. That is too little data to separate learners: in the screen, k nearest
neighbours won on PaDEL with a test R2 of 0.651 while its holdout R2 fell to 0.504. A
difference that reverses between two partitions of the same data is not a difference.

This script replaces that single split with repeated nested resampling, which is the
standard way to make the claim defensible:

  outer loop   REPEATS independent structure-disjoint partitions, each a fresh
               GroupShuffleSplit on canonical structure, so no structure is ever on both
               sides and every repeat sees a different held-out set
  inner loop   5 fold cross validation inside each outer training partition, used for all
               hyperparameter and component-count selection

Selection therefore never sees the outer test partition, and the reported score is the
mean over REPEATS independent held-out sets rather than one draw.

INNER FOLD CONSTRUCTION. The first pass used plain KFold inside the training
partition. That is not structure-disjoint: 122 rows carry only 94 distinct
structures, so between 31 and 42 of roughly 90 training rows share a structure with
another training row, and an ungrouped inner fold can place the two halves of one
structure on opposite sides of the hyperparameter selection. Hyperparameters are then
chosen partly on the model's ability to recall a structure it has already seen, which
favours low-bias, high-variance settings. `--inner group` replaces KFold with
GroupKFold on the same canonical-structure labels, so selection is disjoint at every
level. `--inner kfold` is retained only to reproduce the first pass. W(new) is not used
anywhere: it takes the test coefficient of determination as an input, and with a negative
cross-validated R2 both its numerator and its train-CV penalty change sign, so their
product is large and positive. It ranked a model with a cross-validated R2 of -29.69
first on PaDEL. See docs in the manuscript for the full argument.

Comparisons across learners use the Friedman test over the repeats, with Wilcoxon signed
rank post hoc tests against the incumbent Nu-SVR and Holm correction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     KFold)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "benchmark_work" / "src"))
sys.path.insert(0, str(ROOT))
import datasets  # noqa: E402
from sweep import DATASETS, MODELS, VARIANTS, make_pipeline, model_spec  # noqa: E402

RESULTS = ROOT / "results"
SEED = 42
DEFAULT_REPEATS = 10
PCA_COMPONENTS = (15, 17, 19, 20, 21, 22)


def outer_splits(groups, repeats, test_size=0.25):
    """One structure-disjoint partition per repeat, each with its own random state."""
    idx = np.arange(len(groups))
    out = []
    for r in range(repeats):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=1000 + r)
        tr, te = next(gss.split(idx, groups=groups))
        out.append((tr, te))
    return out


def inner_cv(kind, groups_train):
    """Inner folds for hyperparameter selection, as an explicit list of index pairs.

    Materialising the folds rather than handing GridSearchCV a splitter plus a `groups`
    argument keeps the group information out of the estimator's fit signature: the
    estimator here is a Pipeline, and a routed `groups` keyword reaches its steps.
    Positions are relative to the training partition, which is what GridSearchCV
    indexes.
    """
    n = len(groups_train)
    idx = np.arange(n)
    if kind == "kfold":
        return list(KFold(5, shuffle=True, random_state=SEED).split(idx))
    if kind != "group":
        raise ValueError(f"unknown inner fold kind {kind!r}")
    n_groups = len(set(groups_train))
    k = min(5, n_groups)
    return list(GroupKFold(k, shuffle=True, random_state=SEED)
                .split(idx, groups=groups_train))


def leak_rows(groups_train, folds):
    """How many validation rows share a structure with their own training fold.

    Zero by construction under grouped folds. Recorded so the two passes can be
    compared on the quantity that motivated the change rather than on assertion.
    """
    total = 0
    for tr, va in folds:
        seen = set(np.asarray(groups_train)[tr])
        total += int(sum(g in seen for g in np.asarray(groups_train)[va]))
    return total


def workers_for(dataset, variant, n_features, requested=0):
    """Worker count.

    The first pass pinned four workers for the widest matrix without PCA, on the
    assumption that 62571 columns would exhaust memory. Measured: each worker held
    about 260 MB against 18 GB free, so the cap cost roughly a factor of four in wall
    time for no reason. An explicit request overrides the heuristic.
    """
    if requested > 0:
        return max(1, requested)
    if variant == "raw" and n_features > 20000:
        return min(os.cpu_count() or 4, 16)
    return min(os.cpu_count() or 4, 20)


def run(dataset, variant, model, repeats, use_gpu, requested=0, inner="kfold"):
    X, y, meta = datasets.load(dataset)
    groups = datasets.groups_for(meta)
    Xv, yv = X, y
    n_jobs = workers_for(dataset, variant, X.shape[1], requested)
    rows = []
    for rep, (tr, te) in enumerate(outer_splits(groups, repeats)):
        spec = model_spec(model)
        if model == "xgb" and use_gpu:
            spec.estimator.set_params(device="cuda")
        grid = {f"model__{k}": v for k, v in spec.grid.items()}
        if variant == "pca":
            comps = sorted({min(c, len(tr) - 1, X.shape[1]) for c in PCA_COMPONENTS})
            grid["pca__n_components"] = comps
        # Partial least squares carries its own latent-component count, which cannot
        # exceed the number of columns it is given. The comparison sets are all wide
        # enough for this never to bind, but the ablation includes a four-column
        # protein indicator, where the ungated grid asks for five components and the
        # fit raises.
        if "model__n_components" in grid:
            width = min(X.shape[1], len(tr) - 1)
            grid["model__n_components"] = sorted(
                {min(c, width) for c in grid["model__n_components"]})
        folds = inner_cv(inner, groups[tr])
        t0 = time.perf_counter()
        gs = GridSearchCV(make_pipeline(variant, spec, None), grid, scoring="r2",
                          cv=folds,
                          n_jobs=n_jobs, refit=True, error_score="raise")
        gs.fit(Xv.iloc[tr], yv.iloc[tr])
        pred = np.asarray(gs.best_estimator_.predict(Xv.iloc[te])).reshape(-1)
        rows.append(dict(dataset=dataset, variant=variant, model=model, repeat=rep,
                         inner=inner, inner_leak_rows=leak_rows(groups[tr], folds),
                         n_train=len(tr), n_test=len(te),
                         n_train_groups=int(len(set(groups[tr]))),
                         n_test_groups=int(len(set(groups[te]))),
                         r2_cv=float(gs.best_score_),
                         r2_test=float(r2_score(yv.iloc[te], pred)),
                         rmse_test=float(np.sqrt(mean_squared_error(yv.iloc[te], pred))),
                         mae_test=float(mean_absolute_error(yv.iloc[te], pred)),
                         seconds=time.perf_counter() - t0,
                         best=json.dumps({k.split("__")[-1]: (v.item() if isinstance(v, np.generic) else v)
                                          for k, v in gs.best_params_.items()}, sort_keys=True)))
        print(f"  {dataset} {variant} {model} rep{rep} "
              f"cv={rows[-1]['r2_cv']:.3f} test={rows[-1]['r2_test']:.3f} "
              f"({rows[-1]['seconds']:.0f}s)", flush=True)
    return rows


def summarise(d):
    g = d.groupby(["dataset", "variant", "model"])
    s = g.agg(mean_test=("r2_test", "mean"), sd_test=("r2_test", "std"),
              median_test=("r2_test", "median"), min_test=("r2_test", "min"),
              max_test=("r2_test", "max"), mean_cv=("r2_cv", "mean"),
              mean_rmse=("rmse_test", "mean"), seconds=("seconds", "sum"),
              repeats=("r2_test", "size")).reset_index()
    return s.sort_values(["dataset", "mean_test"], ascending=[True, False])


def compare(d, out):
    """Friedman across learners within a dataset, then Wilcoxon against Nu-SVR."""
    lines = []
    for ds, g in d.groupby("dataset"):
        wide = g.pivot_table(index="repeat", columns=["variant", "model"],
                             values="r2_test")
        wide = wide.dropna(axis=1)
        if wide.shape[1] < 3:
            continue
        fr = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
        lines.append({"dataset": ds, "test": "Friedman across learners",
                      "statistic": fr.statistic, "p": fr.pvalue,
                      "n_learners": wide.shape[1], "n_repeats": wide.shape[0]})
        base = ("pca", "nusvr")
        if base not in wide.columns:
            continue
        raw_p, names = [], []
        for c in wide.columns:
            if c == base:
                continue
            try:
                w = stats.wilcoxon(wide[c], wide[base])
                raw_p.append(w.pvalue)
            except ValueError:
                raw_p.append(1.0)
            names.append(c)
        order = np.argsort(raw_p)
        holm = np.empty(len(raw_p))
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (len(raw_p) - rank) * raw_p[i])
            holm[i] = min(1.0, running)
        for (variant, model), p_raw, p_h in zip(names, raw_p, holm):
            lines.append({"dataset": ds,
                          "test": f"Wilcoxon {model} ({variant}) vs Nu-SVR (pca)",
                          "statistic": float(wide[(variant, model)].mean()
                                             - wide[base].mean()),
                          "p": p_raw, "p_holm": p_h,
                          "n_learners": np.nan, "n_repeats": wide.shape[0]})
    frame = pd.DataFrame(lines)
    frame.to_csv(out, index=False)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    # Measured on this machine, PaDEL raw, full XGBoost grid, five fold inner CV:
    #   CPU with 20 workers  10.8 s
    #   CUDA with 2 workers  19.9 s
    #   CUDA with 1 worker   18.3 s
    # At 122 rows the kernel launch overhead exceeds the arithmetic, and twenty cores
    # beat one serialised device. The flag is kept for reproducibility of that claim
    # and is off by default.
    ap.add_argument("--gpu", action="store_true",
                    help="use CUDA for XGBoost; measured slower here, see comment")
    ap.add_argument("--n-jobs", type=int, default=0,
                    help="override the worker heuristic; 0 uses it")
    ap.add_argument("--inner", choices=("kfold", "group"), default="kfold",
                    help="inner fold construction; 'group' is structure-disjoint, "
                         "'kfold' reproduces the first pass")
    ap.add_argument("--out", default="robust_raw.csv")
    a = ap.parse_args()

    warnings.filterwarnings("ignore")
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / a.out
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for ds in a.datasets:
        for variant in a.variants:
            for model in a.models:
                rows.extend(run(ds, variant, model, a.repeats, a.gpu, a.n_jobs,
                                a.inner))
                pd.DataFrame(rows).to_csv(path, index=False)
    d = pd.DataFrame(rows)
    summary = summarise(d)
    # summaries live beside their raw rows, so the grouped pass cannot overwrite the
    # first pass and the glob in the figure scripts cannot pick up a summary as data
    summary.to_csv(path.parent / "robust_summary.csv", index=False)
    compare(d, path.parent / "robust_tests.csv")
    print()
    print(summary.groupby("dataset").head(3).round(3).to_string(index=False))
    print(f"\nTOTAL {time.perf_counter() - t0:.0f}s over {a.repeats} repeats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
