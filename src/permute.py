"""Y scrambling, sized so that it can actually be run.

WHAT A PERMUTATION TEST HERE CAN AND CANNOT SAY. The ideal null repeats the entire
procedure inside every permutation, learner selection included, so that the null
distribution describes the thing the paper reports. That procedure costs 6.6 hours
per replicate on this machine and a hundred permutations of it would take a month,
which is why it is not done and why the limitation is stated in the text rather than
buried. What is done instead holds the pipeline fixed: the configuration named for
each descriptor set, with its hyperparameters frozen at the values selected most
often across the ten outer partitions, is refitted on permuted potencies. The
resulting p value therefore tests whether that fitted pipeline has found real signal.
It does not additionally penalise the search that chose the pipeline, and no claim
is made that it does.

TWO NULLS, BECAUSE THEY ASK DIFFERENT QUESTIONS.

  free        potency is permuted freely across all 122 rows. This destroys every
              association, including the association between a molecule and its own
              potency, and is the conventional y scrambling of the QSAR literature.

  structure   the block of potencies belonging to one canonical structure is moved
              as a unit to another structure of the same size. Within-structure
              spread, which is the part of the variance no ligand-only representation
              can reach, is preserved exactly, and only the association between the
              chemistry and the potency level is destroyed. This is the harder and
              more informative null: a representation that passes the free test only
              because it can recognise recurring structures will fail this one.

The observed statistic is the median held-out coefficient of determination over the
same ten structure-disjoint partitions used everywhere else, so the observed value is
directly comparable with the number reported in the results.

    python permute.py --datasets md_core89 padel dft --n-perm 200
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "benchmark_work" / "src"))
sys.path.insert(0, str(ROOT))
import datasets  # noqa: E402
from robust import outer_splits  # noqa: E402
from sweep import make_pipeline, model_spec  # noqa: E402

RESULTS = ROOT / "results"


def selected_configuration(frame, dataset):
    """The configuration the resampling named, with its modal hyperparameters.

    Freezing the hyperparameters is what makes the test affordable: a single fit
    costs roughly one part in fifty of a fit preceded by its grid search. The modal
    value is used rather than the value from one partition so that the frozen
    pipeline is representative of the ten rather than of a draw.
    """
    g = frame[frame.dataset == dataset]
    variant, model = g.groupby(["variant", "model"])["r2_test"].median().idxmax()
    g = g[(g.variant == variant) & (g.model == model)]
    params = [json.loads(b) for b in g["best"]]
    modal = {}
    for k in params[0]:
        vals = [p[k] for p in params]
        modal[k] = Counter(map(repr, vals)).most_common(1)[0][0]
        modal[k] = ast.literal_eval(modal[k])
    return variant, model, modal, float(g["r2_test"].median())


def build(variant, model, modal):
    """Rebuild the selected pipeline with its hyperparameters frozen.

    `n_components` is ambiguous: it names the principal component count on the pca
    step and the latent variable count on partial least squares, and both can appear.
    Routing it by which step actually exposes the parameter avoids setting the wrong
    one silently, which would leave the permutation test measuring a pipeline that is
    not the one being reported.
    """
    spec = model_spec(model)
    pipe = make_pipeline(variant, spec, None)
    valid = pipe.get_params(deep=True)
    settings = {}
    for k, v in modal.items():
        key = f"pca__{k}" if f"pca__{k}" in valid else f"model__{k}"
        assert key in valid, f"{model} ({variant}) has no parameter {k}"
        settings[key] = v
    pipe.set_params(**settings)
    return pipe


def score(pipe, X, y, splits):
    """Median held-out R2 across the outer partitions, the reported statistic."""
    out = []
    for tr, te in splits:
        p = pipe.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        out.append(r2_score(y.iloc[te], np.asarray(p).reshape(-1)))
    return float(np.median(out))


def permute_free(y, groups, rng):
    return y.iloc[rng.permutation(len(y))].reset_index(drop=True)


def permute_structure(y, groups, rng):
    """Move whole structure blocks between structures of equal size.

    Permuting only among equal-sized blocks keeps every row filled exactly once and
    keeps each structure's internal spread attached to a structure, which is the
    property the free permutation destroys and this null is meant to preserve.
    """
    out = np.empty(len(y))
    uniq = pd.unique(groups)
    by_size = {}
    for g in uniq:
        idx = np.flatnonzero(groups == g)
        by_size.setdefault(len(idx), []).append(idx)
    for size, blocks in by_size.items():
        order = rng.permutation(len(blocks))
        for dest, src in enumerate(order):
            out[blocks[dest]] = y.to_numpy()[blocks[src]]
    return pd.Series(out, index=y.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*",
                    default=["md_core89", "padel", "dft", "mol2desc"])
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--source", default="grouped/robust_main.csv",
                    help="resampling results naming the configuration to freeze")
    ap.add_argument("--n-jobs", type=int, default=20)
    ap.add_argument("--out", default="permute/permutation.csv")
    a = ap.parse_args()

    warnings.filterwarnings("ignore")
    src = pd.read_csv(RESULTS / a.source)
    path = RESULTS / a.out
    path.parent.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.default_rng(4242)
    rows = []
    for ds in a.datasets:
        if ds not in set(src.dataset):
            print(f"skipping {ds}: absent from {a.source}", flush=True)
            continue
        X, y, meta = datasets.load(ds)
        groups = datasets.groups_for(meta)
        splits = outer_splits(groups, a.repeats)
        variant, model, modal, reported = selected_configuration(src, ds)
        pipe = build(variant, model, modal)
        observed = score(pipe, X, y, splits)
        print(f"{ds}: {model} ({variant}) frozen at {modal}; observed median "
              f"{observed:.4f}, resampling reported {reported:.4f}", flush=True)

        for null, permute in (("free", permute_free),
                              ("structure", permute_structure)):
            seeds = rng_master.integers(0, 2 ** 31 - 1, a.n_perm)
            t0 = time.perf_counter()
            vals = Parallel(n_jobs=a.n_jobs, verbose=0)(
                delayed(score)(build(variant, model, modal), X,
                               permute(y, groups, np.random.default_rng(s)), splits)
                for s in seeds)
            vals = np.asarray(vals, dtype=float)
            # the conventional estimator, which can never return zero and so never
            # claims more certainty than the number of permutations supports
            p = float((1 + (vals >= observed).sum()) / (1 + a.n_perm))
            rows.append(dict(
                dataset=ds, variant=variant, model=model, null=null,
                n_perm=a.n_perm, observed=observed, reported=reported,
                null_median=float(np.median(vals)), null_mean=float(vals.mean()),
                null_sd=float(vals.std(ddof=1)), null_max=float(vals.max()),
                null_q95=float(np.quantile(vals, 0.95)), p=p,
                p_floor=1 / (1 + a.n_perm),
                frozen=json.dumps(modal, sort_keys=True),
                seconds=time.perf_counter() - t0))
            print(f"  {null:9s} null median {np.median(vals):+.4f}  95th "
                  f"{np.quantile(vals, 0.95):+.4f}  max {vals.max():+.4f}  "
                  f"p = {p:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
    print()
    print(pd.DataFrame(rows)[["dataset", "model", "variant", "null", "observed",
                              "null_median", "null_q95", "p"]]
          .round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
