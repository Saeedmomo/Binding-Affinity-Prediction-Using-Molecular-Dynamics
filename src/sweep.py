"""Leakage-safe learner sweep for the fourth-paper descriptor sets.

The benchmark_work tree is imported read-only. The reproduction gate runs before any
experiment and fails closed: FINDINGS.md is never written after a gate failure.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR

ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = ROOT.parent / "benchmark_work"
BENCHMARK_SRC = BENCHMARK_ROOT / "src"
PUBLISHED_METRICS = BENCHMARK_ROOT / "results" / "metrics_all.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CACHE_ROOT = RESULTS_DIR / "_cache"
# Two layouts are supported. In the working tree the analysis scripts sit beside a
# separate read-only benchmark package; in the published repository every module sits
# in one src directory. Falling back to the script's own directory lets the same file
# run in both without a second copy that could drift from this one.
if BENCHMARK_SRC.exists():
    sys.path.insert(0, str(BENCHMARK_SRC))
elif (ROOT / "datasets.py").exists():
    BENCHMARK_ROOT = ROOT.parent
    PUBLISHED_METRICS = BENCHMARK_ROOT / "results" / "metrics_all.csv"
    sys.path.insert(0, str(ROOT))
else:
    raise FileNotFoundError(
        f"neither {BENCHMARK_SRC} nor {ROOT / 'datasets.py'} exists; "
        f"cannot locate the dataset loaders")
import datasets  # noqa: E402
from hybrid_pipeline import make_splits  # noqa: E402

SEED = 42
SPLIT_SEED = 1
DATASETS = ("md_core89", "padel", "mol2desc", "dft")
REGIMES = ("structure_disjoint", "random")
VARIANTS = ("pca", "raw")
MODELS = ("nusvr", "svr", "ridge", "enet", "knn", "pls", "rf", "et",
          "gbr", "hgb", "xgb", "lgbm")
PCA_COMPONENTS = (15, 17, 19, 20, 21, 22)
EXPECTED_GATE = {"md_core89": 0.798378, "padel": 0.550}
GATE_TOLERANCE = 0.02
DATA_LABELS = {"md_core89": "MD core 89", "padel": "PaDEL",
               "mol2desc": "PyDescriptor", "dft": "Quantum chemical"}
MODEL_LABELS = {"nusvr": "Nu-SVR", "svr": "SVR", "ridge": "Ridge",
                "enet": "Elastic Net", "knn": "K nearest neighbors", "pls": "PLS",
                "rf": "Random forest", "et": "Extra trees", "gbr": "Gradient boosting",
                "hgb": "Histogram gradient boosting", "xgb": "XGBoost", "lgbm": "LightGBM"}
RAW_COLUMNS = ["dataset", "split_regime", "preprocessing_variant", "model",
               "chosen_hyperparameters", "chosen_n_components", "r2_train", "r2_cv", "r2_test", "r2_holdout",
               "rmse_test", "mae_test", "fit_seconds", "status", "skip_reason",
               "n_rows", "n_features", "n_train", "n_test", "n_holdout", "workers"]


from sweep_support import SafePCA, VarianceGate


@dataclass(frozen=True)
class ModelSpec:
    estimator: Any
    grid: dict[str, list[Any]]


def model_spec(name: str, smoke: bool = False) -> ModelSpec:
    if name == "nusvr":
        spec = ModelSpec(NuSVR(kernel="rbf", max_iter=2_000_000),
                         {"nu": [0.1, 0.3, 0.5, 0.7, 0.9],
                          "C": [0.1, 1.0, 10.0, 100.0],
                          "kernel": ["linear", "poly", "rbf", "sigmoid"]})
    elif name == "svr":
        spec = ModelSpec(SVR(kernel="rbf", max_iter=2_000_000),
                         {"C": [0.1, 1.0, 10.0], "epsilon": [0.05, 0.1]})
    elif name == "ridge":
        spec = ModelSpec(Ridge(random_state=SEED), {"alpha": list(np.logspace(-2, 3, 6))})
    elif name == "enet":
        spec = ModelSpec(ElasticNet(random_state=SEED, max_iter=100_000),
                         {"alpha": list(np.logspace(-3, 1, 5)),
                          "l1_ratio": [0.15, 0.5, 0.85]})
    elif name == "knn":
        spec = ModelSpec(KNeighborsRegressor(),
                         {"n_neighbors": [3, 5, 10], "weights": ["uniform", "distance"]})
    elif name == "pls":
        spec = ModelSpec(PLSRegression(max_iter=1_000), {"n_components": [2, 5, 10]})
    elif name == "rf":
        spec = ModelSpec(RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=1),
                         {"max_features": [0.3, "sqrt", 1.0], "min_samples_leaf": [1, 3]})
    elif name == "et":
        spec = ModelSpec(ExtraTreesRegressor(n_estimators=500, random_state=SEED, n_jobs=1),
                         {"max_features": [0.3, "sqrt", 1.0], "min_samples_leaf": [1, 3]})
    elif name == "gbr":
        spec = ModelSpec(GradientBoostingRegressor(random_state=SEED),
                         {"n_estimators": [200, 500], "learning_rate": [0.03, 0.1],
                          "max_depth": [2, 3]})
    elif name == "hgb":
        spec = ModelSpec(HistGradientBoostingRegressor(random_state=SEED),
                         {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 31]})
    elif name == "xgb":
        from xgboost import XGBRegressor
        spec = ModelSpec(XGBRegressor(random_state=SEED, n_jobs=1,
                                     objective="reg:squarederror", subsample=0.8,
                                     colsample_bytree=0.8, verbosity=0),
                         {"n_estimators": [200, 500], "learning_rate": [0.03, 0.1],
                          "max_depth": [2, 4]})
    elif name == "lgbm":
        from lightgbm import LGBMRegressor
        spec = ModelSpec(LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1),
                         {"n_estimators": [200, 500], "learning_rate": [0.03, 0.1],
                          "num_leaves": [15, 31]})
    else:
        raise KeyError(f"Unknown model: {name}")
    if smoke:
        return ModelSpec(spec.estimator, {key: [values[0]] for key, values in spec.grid.items()})
    return spec


def make_pipeline(variant: str, spec: ModelSpec, memory: Memory | None) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median")),
             ("variance", VarianceGate()), ("scale", StandardScaler())]
    if variant == "pca":
        steps.append(("pca", SafePCA(17, SEED)))
    elif variant != "raw":
        raise ValueError(f"Unknown preprocessing variant: {variant}")
    steps.append(("model", spec.estimator))
    return Pipeline(steps, memory=memory)


def available_memory_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 8 * 2**30


def choose_workers(X: pd.DataFrame, dataset: str, variant: str, requested: int) -> int:
    if requested > 0:
        workers = min(requested, 4)
    else:
        matrix_bytes = max(int(X.memory_usage(deep=True).sum()), 1)
        bytes_per_worker = max(matrix_bytes * 5, 256 * 2**20)
        workers = min(os.cpu_count() or 1,
                      max(1, int(available_memory_bytes() * 0.20 / bytes_per_worker)), 4)
    if dataset == "mol2desc" and variant == "raw":
        workers = min(workers, 4)
    return max(1, workers)


def clean_params(params: dict[str, Any]) -> dict[str, Any]:
    clean = {}
    for key, value in params.items():
        if not key.startswith("model__"):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        clean[key.removeprefix("model__")] = value
    return clean


def empty_row(dataset, regime, variant, model, X, splits, workers):
    return {"dataset": dataset, "split_regime": regime, "preprocessing_variant": variant,
            "model": model, "chosen_hyperparameters": "", "chosen_n_components": np.nan,
            "r2_train": np.nan,
            "r2_cv": np.nan, "r2_test": np.nan, "r2_holdout": np.nan,
            "rmse_test": np.nan, "mae_test": np.nan, "fit_seconds": np.nan,
            "status": "failed", "skip_reason": "", "n_rows": len(X),
            "n_features": X.shape[1], "n_train": len(splits["train"]),
            "n_test": len(splits["test"]), "n_holdout": len(splits["holdout"]),
            "workers": workers}


def fit_combination(dataset, regime, variant, model, X, y, groups,
                    requested_workers, smoke=False):
    splits = make_splits(len(y), groups=groups, regime=regime, random_state=SPLIT_SEED)
    workers = choose_workers(X, dataset, variant, requested_workers)
    row = empty_row(dataset, regime, variant, model, X, splits, workers)
    cache_dir = CACHE_ROOT / f"{dataset}_{regime}_{variant}_{model}_{os.getpid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    memory = Memory(cache_dir, verbose=0)
    started = time.perf_counter()
    try:
        spec = model_spec(model, smoke=smoke)
        train, test, holdout = (splits[key] for key in ("train", "test", "holdout"))
        parameter_grid = {f"model__{key}": values for key, values in spec.grid.items()}
        if variant == "pca":
            components = sorted({min(value, len(train) - 1, X.shape[1])
                                 for value in PCA_COMPONENTS})
            parameter_grid["pca__n_components"] = components[:1] if smoke else components
        search = GridSearchCV(
            make_pipeline(variant, spec, memory), parameter_grid,
            scoring="r2", cv=KFold(5, shuffle=True, random_state=SEED),
            n_jobs=workers, refit=True, error_score="raise")
        search.fit(X.iloc[train], y.iloc[train])
        estimator = search.best_estimator_
        pred_train = np.asarray(estimator.predict(X.iloc[train])).reshape(-1)
        pred_test = np.asarray(estimator.predict(X.iloc[test])).reshape(-1)
        pred_holdout = np.asarray(estimator.predict(X.iloc[holdout])).reshape(-1)
        row.update(chosen_hyperparameters=json.dumps(clean_params(search.best_params_),
                                                     sort_keys=True),
                   chosen_n_components=(int(search.best_params_["pca__n_components"])
                                        if variant == "pca" else np.nan),
                   r2_train=float(r2_score(y.iloc[train], pred_train)),
                   r2_cv=float(search.best_score_),
                   r2_test=float(r2_score(y.iloc[test], pred_test)),
                   r2_holdout=float(r2_score(y.iloc[holdout], pred_holdout)),
                   rmse_test=float(np.sqrt(mean_squared_error(y.iloc[test], pred_test))),
                   mae_test=float(mean_absolute_error(y.iloc[test], pred_test)),
                   status="ok", skip_reason="")
        return row, estimator, pred_test
    except Exception as exc:
        row["status"] = "skipped" if isinstance(exc, MemoryError) else "failed"
        row["skip_reason"] = f"{type(exc).__name__}: {exc}"[:1000]
        traceback.print_exc()
        return row, None, None
    finally:
        row["fit_seconds"] = time.perf_counter() - started
        memory.clear(warn=False)
        shutil.rmtree(cache_dir, ignore_errors=True)


def published_gate_values():
    frame = pd.read_csv(PUBLISHED_METRICS)
    selected = frame[(frame["model"] == "nusvr") & (frame["subset"] == "test") &
                     (frame["split_regime"] == "structure_disjoint") &
                     frame["dataset"].isin(EXPECTED_GATE)]
    return dict(zip(selected["dataset"], selected["r2"]))


def reproduction_gate(requested_workers: int):
    published = published_gate_values()
    details = {}
    print("REPRODUCTION CHECK: Nu-SVR + pca + structure_disjoint + random_state=1")
    for dataset in ("md_core89", "padel"):
        X, y, meta = datasets.load(dataset)
        row, _, _ = fit_combination(dataset, "structure_disjoint", "pca", "nusvr",
                                    X, y, datasets.groups_for(meta), requested_workers)
        observed = float(row["r2_test"])
        expected = EXPECTED_GATE[dataset]
        difference = abs(observed - expected)
        component = int(row["chosen_n_components"]) if row["status"] == "ok" else None
        component_ok = dataset != "md_core89" or component == 19
        passed = (row["status"] == "ok" and difference <= GATE_TOLERANCE
                  and component_ok)
        details[dataset] = {"row": row, "observed": observed, "expected": expected,
                            "published_csv": float(published[dataset]),
                            "absolute_difference": difference,
                            "chosen_n_components": component, "passed": passed}
        print(f"  {dataset}: sklearn={sklearn.__version__}, observed={observed:.6f}, "
              f"expected={expected:.6f}, selected_components={component}, "
              f"published_csv={published[dataset]:.6f}, abs_diff={difference:.6f} "
              f"{'PASS' if passed else 'FAIL'}")
    passed = all(item["passed"] for item in details.values())
    print(f"REPRODUCTION GATE: {'PASS' if passed else 'FAIL'}")
    return passed, details


def bootstrap_r2(y_true, y_pred, n_resamples=1000):
    rng = np.random.default_rng(SEED)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    values = []
    for _ in range(n_resamples):
        indices = rng.integers(0, len(y_true), len(y_true))
        if np.var(y_true[indices]) > 1e-12:
            values.append(float(r2_score(y_true[indices], y_pred[indices])))
    if not values:
        return np.nan, np.nan, 0
    return float(np.quantile(values, .025)), float(np.quantile(values, .975)), len(values)


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(path, index=False)


def select_best_and_ci(raw):
    good = raw[raw["status"] == "ok"].copy()
    best = (good.sort_values(["dataset", "split_regime", "r2_cv"],
                             ascending=[True, True, False])
            .groupby(["dataset", "split_regime"], sort=False, as_index=False).head(1).copy())
    best["r2_test_ci_low"], best["r2_test_ci_high"] = np.nan, np.nan
    best["bootstrap_resamples"] = 0
    for index, winner in best.iterrows():
        dataset, regime = winner["dataset"], winner["split_regime"]
        X, y, meta = datasets.load(dataset)
        splits = make_splits(len(y), groups=datasets.groups_for(meta), regime=regime,
                             random_state=SPLIT_SEED)
        spec = model_spec(winner["model"])
        pipe = make_pipeline(winner["preprocessing_variant"], spec, None)
        params = json.loads(winner["chosen_hyperparameters"])
        pipe.set_params(**{f"model__{key}": value for key, value in params.items()})
        if winner["preprocessing_variant"] == "pca":
            pipe.set_params(pca__n_components=int(winner["chosen_n_components"]))
        pipe.fit(X.iloc[splits["train"]], y.iloc[splits["train"]])
        prediction = np.asarray(pipe.predict(X.iloc[splits["test"]])).reshape(-1)
        low, high, count = bootstrap_r2(y.iloc[splits["test"]].to_numpy(), prediction)
        best.loc[index, ["r2_test_ci_low", "r2_test_ci_high", "bootstrap_resamples"]] = [low, high, count]
    return best


def validate_completeness(raw):
    expected = {(d, r, v, m) for d in DATASETS for r in REGIMES for v in VARIANTS for m in MODELS}
    actual = set(raw[["dataset", "split_regime", "preprocessing_variant", "model"]]
                 .itertuples(index=False, name=None))
    if expected != actual:
        raise RuntimeError(f"Combination mismatch: missing={sorted(expected-actual)[:5]}")
    if raw.duplicated(["dataset", "split_regime", "preprocessing_variant", "model"]).any():
        raise RuntimeError("Duplicate combination rows found")
    if ((raw["status"] == "ok") & raw["r2_test"].isna()).any():
        raise RuntimeError("Successful rows with NaN test R2 found")
    if ((raw["status"] != "ok") & raw["skip_reason"].fillna("").str.strip().eq("")).any():
        raise RuntimeError("Non-success rows without a reason found")
    print("COMPLETENESS CHECK: PASS; all 192 combinations are present, all successful "
          "rows have finite test R2, and every non-success has a recorded reason.")


def make_heatmap(raw, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 12), constrained_layout=True)
    image = None
    panels = [(regime, variant) for regime in REGIMES for variant in VARIANTS]
    for axis, (regime, variant) in zip(axes.flat, panels):
        part = raw[(raw["split_regime"] == regime) &
                   (raw["preprocessing_variant"] == variant) & (raw["status"] == "ok")]
        matrix = (part.pivot(index="model", columns="dataset", values="r2_test")
                  .reindex(index=MODELS, columns=DATASETS).to_numpy(float))
        image = axis.imshow(matrix, cmap="RdYlBu", vmin=-1, vmax=1, aspect="auto")
        axis.set_xticks(range(len(DATASETS)), [DATA_LABELS[d] for d in DATASETS], rotation=30)
        axis.set_yticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
        axis.set_title(f"{regime.replace('_', ' ').title()}, {variant}")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                axis.text(j, i, "NA" if np.isnan(value) else f"{value:.2f}",
                          ha="center", va="center", fontsize=7,
                          color="white" if np.isfinite(value) and abs(value) > .65 else "black")
    fig.colorbar(image, ax=axes, shrink=.65, label="Test R2, clipped to minus 1 through 1")
    fig.suptitle("Learner sweep test performance", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def baseline_row(raw, dataset, regime):
    rows = raw[(raw["dataset"] == dataset) & (raw["split_regime"] == regime) &
               (raw["preprocessing_variant"] == "pca") & (raw["model"] == "nusvr") &
               (raw["status"] == "ok")]
    if len(rows) != 1:
        raise RuntimeError(f"Missing unique baseline for {dataset}, {regime}")
    return rows.iloc[0]


def make_findings(raw, best, wall_seconds):
    lines = ["# Learner sweep findings", "",
             "All hyperparameters and all winning learners were selected only by mean five fold "
             "cross validation R2 within the training partition. Test and holdout partitions "
             "were never read during selection.", "", "## Comparison", "",
             "| Dataset | Split | CV selected learner | Preprocessing | CV R2 | Test R2 | Nu-SVR pca test R2 | Difference | Test R2 95 percent interval |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for _, winner in best.sort_values(["dataset", "split_regime"]).iterrows():
        base = baseline_row(raw, winner["dataset"], winner["split_regime"])
        difference = winner["r2_test"] - base["r2_test"]
        lines.append("| {0} | {1} | {2} | {3} | {4:.3f} | {5:.3f} | {6:.3f} | {7:+.3f} | {8:.3f} to {9:.3f} |".format(
            DATA_LABELS[winner["dataset"]], winner["split_regime"].replace("_", " "),
            MODEL_LABELS[winner["model"]], winner["preprocessing_variant"], winner["r2_cv"],
            winner["r2_test"], base["r2_test"], difference,
            winner["r2_test_ci_low"], winner["r2_test_ci_high"]))
    lines.extend(["", "## Answer to the reviewer", ""])
    for dataset in ("padel", "mol2desc", "dft"):
        for regime in REGIMES:
            winner = best[(best["dataset"] == dataset) & (best["split_regime"] == regime)].iloc[0]
            base = baseline_row(raw, dataset, regime)
            md = best[(best["dataset"] == "md_core89") & (best["split_regime"] == regime)].iloc[0]
            gain = winner["r2_test"] - base["r2_test"]
            margin = md["r2_test"] - winner["r2_test"]
            relation = "ahead of" if margin >= 0 else "behind"
            lines.extend([f"{DATA_LABELS[dataset]}, {regime.replace('_', ' ')}: the CV selected "
                          f"learner changes test R2 by {gain:+.3f} relative to Nu-SVR pca. "
                          f"MD core 89 is {relation} by {abs(margin):.3f}.", ""])
    failures = raw[raw["status"] != "ok"]
    lines.extend(["## Run notes", "", f"Total wall clock time was {wall_seconds:.1f} seconds.", ""])
    if failures.empty:
        lines.append("No model and dataset combinations were skipped.")
    else:
        lines.extend(["Skipped or failed combinations:", ""])
        for _, row in failures.iterrows():
            lines.append(f"* {DATA_LABELS[row['dataset']]}, {MODEL_LABELS[row['model']]}, "
                         f"{row['split_regime'].replace('_', ' ')}, {row['preprocessing_variant']}: "
                         f"{str(row['skip_reason']).replace('_', ' ')}")
    text = "\n".join(lines).rstrip() + "\n"
    if any(character in text for character in ("_", "\u2013", "\u2014")):
        raise RuntimeError("Forbidden visible character in FINDINGS.md")
    return text


def require_runtime():
    if sklearn.__version__ != "1.9.0":
        raise RuntimeError(
            "ABORT: scikit-learn 1.9.0 is required for comparison with the paper; "
            f"this interpreter has {sklearn.__version__} at {sys.executable}"
        )


def runtime_provenance():
    versions = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                "pandas": pd.__version__, "numpy": np.__version__}
    for package in ("scipy", "xgboost", "lightgbm"):
        try:
            module = __import__(package)
            versions[package] = module.__version__
        except Exception as exc:
            versions[package] = f"unavailable: {exc}"
    return json.dumps(versions, sort_keys=True)


def smoke_sweep(requested_workers):
    rows = []
    X, y, meta = datasets.load("dft")
    groups = datasets.groups_for(meta)
    for regime in REGIMES:
        for variant in VARIANTS:
            for model in ("nusvr", "ridge", "et"):
                print(f"SMOKE dft {regime} {variant} {model}", flush=True)
                row, _, _ = fit_combination("dft", regime, variant, model, X, y, groups,
                                            requested_workers, smoke=True)
                rows.append(row)
    write_rows(rows, RESULTS_DIR / "smoke_raw.csv")


def full_sweep(requested_workers, gate_details):
    output = RESULTS_DIR / "sweep_raw.csv"
    rows = [detail["row"] for detail in gate_details.values()]
    done = {(row["dataset"], row["split_regime"], row["preprocessing_variant"], row["model"])
            for row in rows}
    write_rows(rows, output)
    for dataset in DATASETS:
        X, y, meta = datasets.load(dataset)
        groups = datasets.groups_for(meta)
        for regime in REGIMES:
            for variant in VARIANTS:
                for model in MODELS:
                    key = (dataset, regime, variant, model)
                    if key in done:
                        continue
                    print("RUN " + " ".join(key), flush=True)
                    row, _, _ = fit_combination(*key, X, y, groups, requested_workers)
                    rows.append(row)
                    done.add(key)
                    write_rows(rows, output)
                    print(f"  status={row['status']} cv={row['r2_cv']} test={row['r2_test']} "
                          f"seconds={row['fit_seconds']:.1f}", flush=True)
    return pd.read_csv(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a reduced DFT sweep")
    parser.add_argument("--n-jobs", type=int, default=0,
                        help="Maximum CV workers, auto when zero and capped at four")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    warnings.filterwarnings("once", category=UserWarning)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"RUNTIME {runtime_provenance()}", flush=True)
    require_runtime()
    passed, gate_details = reproduction_gate(args.n_jobs)
    if not passed:
        print("STOP: reproduction gate failed. No sweep conclusions or FINDINGS.md were produced.",
              flush=True)
        return 2
    if args.smoke:
        smoke_sweep(args.n_jobs)
        elapsed = time.perf_counter() - started
        print(f"SMOKE WALL CLOCK: {elapsed:.3f} seconds")
        print(f"SMOKE TIME GATE: {'PASS' if elapsed < 120 else 'FAIL'}")
        return 0 if elapsed < 120 else 3
    raw = full_sweep(args.n_jobs, gate_details)
    raw = pd.read_csv(RESULTS_DIR / "sweep_raw.csv")
    validate_completeness(raw)
    best = select_best_and_ci(raw)
    best.to_csv(RESULTS_DIR / "sweep_best.csv", index=False)
    make_heatmap(raw, FIGURES_DIR / "sweep_heatmap.png")
    elapsed = time.perf_counter() - started
    (RESULTS_DIR / "FINDINGS.md").write_text(make_findings(raw, best, elapsed), encoding="utf-8")
    print(f"FULL WALL CLOCK: {elapsed:.3f} seconds")
    print("FULL SWEEP: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
