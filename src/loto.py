"""Leave one protein target out, across every learner and every descriptor set.

The published transfer check held out each of the four targets in turn for a single
configuration: Nu-SVR on the simulation-derived set after principal components. It
found a negative coefficient of determination on all four, and the manuscript
concluded that the model does not place an unseen target on the absolute potency
scale. A referee will ask the obvious follow-up: is that a property of the
representation, or of that one learner. This script answers it by repeating the
same held-out-target protocol for all twelve learners, both preprocessing variants
and all four descriptor sets.

Two design points matter.

STRUCTURES SHARED BETWEEN TARGETS. Nineteen canonical structures appear against
more than one target, so simply removing the rows of the held-out target still
leaves some of its ligands in training under a different target. That is target
transfer contaminated by ligand memory. `--share drop` removes from the training
set every row whose structure also occurs in the held-out target, which is the
honest test; `--share keep` is the contaminated variant, retained so the size of
the effect can be reported rather than asserted. The count of contaminated test
rows is written to every row of the output either way.

WHAT TO MEASURE. Across targets the potency scale shifts, so the coefficient of
determination is dominated by an offset the model has no way to know. It is
reported because it is the quantity the manuscript already quotes, but the
Spearman rank correlation inside the held-out target is reported alongside it:
that is the quantity a medicinal chemist would actually use, since ranking
candidates within one campaign does not require the absolute scale. A
representation that transfers ordering but not offset looks worthless by the
first measure and useful by the second, and the paper should say which it is.

    python loto.py --datasets md_core89 padel dft mol2desc --share drop
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
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "benchmark_work" / "src"))
sys.path.insert(0, str(ROOT))
import datasets  # noqa: E402
from robust import PCA_COMPONENTS, inner_cv, workers_for  # noqa: E402
from sweep import DATASETS, MODELS, VARIANTS, make_pipeline, model_spec  # noqa: E402

RESULTS = ROOT / "results"


def target_labels(meta):
    ann = datasets.annotations()
    return ann["target"].to_numpy()[meta["row_ids"]]


def run(dataset, variant, model, share, requested=0):
    X, y, meta = datasets.load(dataset)
    groups = datasets.groups_for(meta)
    targets = target_labels(meta)
    n_jobs = workers_for(dataset, variant, X.shape[1], requested)
    rows = []
    for held in sorted(set(targets)):
        te = np.flatnonzero(targets == held)
        tr = np.flatnonzero(targets != held)
        shared = set(groups[te]) & set(groups[tr])
        contaminated = int(sum(g in shared for g in groups[te]))
        if share == "drop":
            tr = np.array([i for i in tr if groups[i] not in shared])
        spec = model_spec(model)
        grid = {f"model__{k}": v for k, v in spec.grid.items()}
        if variant == "pca":
            comps = sorted({min(c, len(tr) - 1, X.shape[1]) for c in PCA_COMPONENTS})
            grid["pca__n_components"] = comps
        folds = inner_cv("group", groups[tr])
        t0 = time.perf_counter()
        gs = GridSearchCV(make_pipeline(variant, spec, None), grid, scoring="r2",
                          cv=folds, n_jobs=n_jobs, refit=True, error_score="raise")
        gs.fit(X.iloc[tr], y.iloc[tr])
        pred = np.asarray(gs.best_estimator_.predict(X.iloc[te])).reshape(-1)
        obs = y.iloc[te].to_numpy()
        rho = stats.spearmanr(obs, pred)
        # the baseline a model must beat to be worth anything: predict the training mean
        base_rmse = float(np.sqrt(((obs - y.iloc[tr].mean()) ** 2).mean()))
        rows.append(dict(
            dataset=dataset, variant=variant, model=model, held_out=held, share=share,
            n_train=len(tr), n_test=len(te), contaminated_test_rows=contaminated,
            r2_cv=float(gs.best_score_),
            r2_test=float(r2_score(obs, pred)),
            rmse_test=float(np.sqrt(mean_squared_error(obs, pred))),
            mae_test=float(mean_absolute_error(obs, pred)),
            rmse_training_mean=base_rmse,
            spearman=float(rho.statistic), spearman_p=float(rho.pvalue),
            # the offset the model cannot know, isolated: what R2 would be if the
            # prediction were shifted by the mean error on the held-out target
            r2_after_offset=float(r2_score(obs, pred + (obs - pred).mean())),
            seconds=time.perf_counter() - t0,
            best=json.dumps({k.split("__")[-1]: (v.item() if isinstance(v, np.generic)
                                                 else v)
                             for k, v in gs.best_params_.items()}, sort_keys=True)))
        print(f"  {dataset} {variant} {model} {held} r2={rows[-1]['r2_test']:.3f} "
              f"shift={rows[-1]['r2_after_offset']:.3f} rho={rows[-1]['spearman']:.3f} "
              f"({rows[-1]['seconds']:.0f}s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--share", choices=("drop", "keep"), default="drop")
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--out", default="loto/loto.csv")
    a = ap.parse_args()

    warnings.filterwarnings("ignore")
    path = RESULTS / a.out
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.perf_counter()
    for ds in a.datasets:
        for variant in a.variants:
            for model in a.models:
                rows.extend(run(ds, variant, model, a.share, a.n_jobs))
                pd.DataFrame(rows).to_csv(path, index=False)
    d = pd.DataFrame(rows)
    s = (d.groupby(["dataset", "variant", "model"])
         .agg(median_r2=("r2_test", "median"),
              median_r2_shifted=("r2_after_offset", "median"),
              median_rho=("spearman", "median"),
              targets_beating_mean=("rmse_test", "size"))
         .reset_index())
    beat = (d.assign(win=d.rmse_test < d.rmse_training_mean)
            .groupby(["dataset", "variant", "model"])["win"].sum().reset_index())
    s = s.drop(columns=["targets_beating_mean"]).merge(beat, on=["dataset", "variant",
                                                                 "model"])
    s = s.rename(columns={"win": "targets_beating_training_mean"})
    s.sort_values(["dataset", "median_rho"], ascending=[True, False]).to_csv(
        path.parent / "loto_summary.csv", index=False)
    print()
    print(s.sort_values(["dataset", "median_rho"], ascending=[True, False])
          .groupby("dataset").head(3).round(3).to_string(index=False))
    print(f"\nTOTAL {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
