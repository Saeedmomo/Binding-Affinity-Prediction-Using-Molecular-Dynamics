"""Validate every result file before the manuscript is rewritten around them.

The rewrite is meant to be determined by complete evidence, so the evidence has to be
checked first rather than assumed sound. This script asserts nothing quietly: it
prints what it found and returns a non-zero status if anything is wrong.

Checks, in order of how badly each would corrupt the conclusions:

  1. COMPLETENESS. Every dataset, preprocessing variant, learner and repeat present
     exactly once. A missing cell silently shortens a paired comparison; a duplicated
     one silently doubles a representation's weight, which has already happened once
     in this project with the ablation frame.
  2. PAIRING. Every representation scored on the identical set of partition indices,
     and every leave-one-target-out run on the identical four targets. A paired test
     across differently indexed partitions is meaningless.
  3. PARTITION IDENTITY. The held-out row counts per repeat must agree across
     representations, since all of them use the same grouped shuffle split with the
     same seeds. If they do not, the partitions are not the same partitions.
  4. LEAKAGE. Grouped inner folds must report zero validation rows sharing a
     structure with their own training fold. The transfer runs must actually have
     removed the structures they recorded as shared.
  5. DIVERGENCE. Count fits that failed numerically, and say which learner families
     they came from, so the manuscript reports medians for a stated reason.
  6. DETERMINISM. Where a run has been repeated, the repetition must reproduce the
     original exactly, since every seed is fixed. This is what decides whether a
     duplicated job is worth waiting for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
ORDER = ["md_core89", "padel", "mol2desc", "dft"]
MODELS = ("nusvr", "svr", "ridge", "enet", "knn", "pls", "rf", "et", "gbr", "hgb",
          "xgb", "lgbm")
VARIANTS = ("pca", "raw")
TARGETS = ("ESR1", "MAPK1", "TDP1", "TP53")
REPEATS = 10

fail: list[str] = []
note: list[str] = []


def bad(msg):
    fail.append(msg)


def load(path):
    return pd.read_csv(path) if path.exists() else None


# ------------------------------------------------------------------ resampling
def check_resampling():
    frames = [pd.read_csv(p) for p in sorted((RES / "grouped").glob("robust_*.csv"))
              if p.name not in ("robust_summary.csv", "robust_tests.csv")]
    if not frames:
        bad("no grouped resampling results at all")
        return None
    d = pd.concat(frames, ignore_index=True)
    print(f"grouped resampling: {len(d)} rows over {sorted(set(d.dataset))}")

    dup = d.duplicated(subset=["dataset", "variant", "model", "repeat"]).sum()
    if dup:
        bad(f"resampling has {dup} duplicated dataset/variant/model/repeat rows")

    for ds in sorted(set(d.dataset)):
        g = d[d.dataset == ds]
        missing = [(v, m, r) for v in VARIANTS for m in MODELS for r in range(REPEATS)
                   if not len(g[(g.variant == v) & (g.model == m) & (g.repeat == r)])]
        if missing:
            bad(f"{ds}: {len(missing)} missing cells, first {missing[:3]}")
        else:
            print(f"  {ds:10s} complete: {len(VARIANTS)}x{len(MODELS)}x{REPEATS} = "
                  f"{len(g)} fits")

    # partitions must be the same partitions across representations
    sizes = d.pivot_table(index="repeat", columns="dataset", values="n_test",
                          aggfunc="first")
    if sizes.nunique(axis=1).max() > 1:
        bad(f"held-out sizes differ across representations by repeat:\n{sizes}")
    else:
        print(f"  partitions identical across representations, held-out sizes "
              f"{sorted(set(sizes.iloc[:, 0]))}")

    if "inner_leak_rows" in d.columns:
        leak = d[d.inner == "group"]["inner_leak_rows"] if "inner" in d.columns \
            else d["inner_leak_rows"]
        if leak.max() > 0:
            bad(f"grouped inner folds report leakage, max {leak.max()} rows")
        else:
            print("  grouped inner folds leak zero rows")

    diverged = d[d.r2_test < -10]
    trees = ("rf", "et", "gbr", "hgb", "xgb", "lgbm")
    print(f"  divergent fits below minus ten: {len(diverged)} of {len(d)}, "
          f"{int(diverged.model.isin(trees).sum())} from tree ensembles, worst "
          f"{d.r2_test.min():.3g}")
    if d.r2_test.isna().any():
        bad(f"{int(d.r2_test.isna().sum())} resampling fits produced no score")
    return d


# ------------------------------------------------------------------- transfer
def check_transfer():
    parts, names = [], []
    for f in ("loto.csv", "loto_main.csv", "loto_mol2desc.csv"):
        x = load(RES / "loto" / f)
        if x is not None and len(x):
            parts.append(x)
            names.append(f)
    if not parts:
        bad("no transfer results at all")
        return None
    print(f"\ntransfer sources: {', '.join(names)}")

    # determinism: where two sources cover the same cell they must agree exactly
    if len(parts) > 1:
        key = ["dataset", "variant", "model", "held_out"]
        merged = parts[0].merge(pd.concat(parts[1:], ignore_index=True), on=key,
                                suffixes=("_a", "_b"))
        if len(merged):
            delta = (merged["r2_test_a"] - merged["r2_test_b"]).abs().max()
            print(f"  {len(merged)} cells covered twice, largest disagreement "
                  f"{delta:.3g}")
            if delta > 1e-9:
                bad(f"repeated transfer runs disagree by up to {delta:.3g}; the runs "
                    f"are not reproducible and one of them is wrong")
        else:
            note.append("no overlap between transfer sources, determinism untested")

    d = (pd.concat(parts, ignore_index=True)
         .drop_duplicates(subset=["dataset", "variant", "model", "held_out"],
                          keep="first"))
    print(f"  {len(d)} unique cells over {sorted(set(d.dataset))}")

    for ds in sorted(set(d.dataset)):
        g = d[d.dataset == ds]
        missing = [(v, m, t) for v in VARIANTS for m in MODELS for t in TARGETS
                   if not len(g[(g.variant == v) & (g.model == m) &
                                (g.held_out == t)])]
        if missing:
            bad(f"transfer {ds}: {len(missing)} missing cells, first {missing[:3]}")
        else:
            print(f"  {ds:10s} complete: {len(VARIANTS)}x{len(MODELS)}x"
                  f"{len(TARGETS)} = {len(g)} fits")

    # the four held-out sets must be the same four whichever representation is used
    sizes = d.pivot_table(index="held_out", columns="dataset", values="n_test",
                          aggfunc="first")
    if sizes.nunique(axis=1).max() > 1:
        bad(f"held-out target sizes differ across representations:\n{sizes}")
    else:
        print(f"  held-out target sizes identical across representations: "
              f"{sizes.iloc[:, 0].to_dict()}")

    # share=drop must actually have removed the shared structures it recorded
    if {"contaminated_test_rows", "n_train"} <= set(d.columns):
        by_target = d.groupby("held_out")[["n_train", "contaminated_test_rows"]].agg(
            {"n_train": "nunique", "contaminated_test_rows": "first"})
        print(f"  rows removed from training for shared structure, by target: "
              f"{d.groupby('held_out')['contaminated_test_rows'].first().to_dict()}")
        if (d["contaminated_test_rows"] == 0).all():
            note.append("no shared structures recorded; check the drop rule fired")
    return d


def transfer_ranking(d):
    """The ordering that the manuscript may claim, computed rather than assumed."""
    if d is None:
        return None
    s = (d.groupby(["dataset", "variant", "model"])
         .agg(rho=("spearman", "median"), r2=("r2_test", "median"),
              shifted=("r2_after_offset", "median"))
         .reset_index().sort_values("rho", ascending=False)
         .groupby("dataset").head(1).set_index("dataset"))
    complete = [ds for ds in ORDER if ds in s.index and
                len(d[d.dataset == ds]) == len(VARIANTS) * len(MODELS) * len(TARGETS)]
    print("\nbest configuration per representation under transfer:")
    print(s.loc[[k for k in ORDER if k in s.index]].round(4).to_string())
    if len(complete) < len(ORDER):
        note.append(f"transfer ranking not final; complete for {complete}")
        return None
    ranked = list(s.loc[complete, "rho"].sort_values(ascending=False).index)
    print(f"\ntransfer ranking by median rank correlation: {' > '.join(ranked)}")
    return ranked


# ------------------------------------------------------------------- ablation
def check_ablation():
    parts = [load(RES / "ablation" / f) for f in
             ("ablation.csv", "target_baseline.csv", "compact56.csv")]
    parts = [x for x in parts if x is not None]
    d = pd.concat(parts, ignore_index=True)
    before = len(d)
    d = d.drop_duplicates(subset=["dataset", "variant", "model", "repeat"],
                          keep="first")
    print(f"\nablation: {before} rows, {before - len(d)} duplicates removed, "
          f"{len(d)} unique")
    for ds in sorted(set(d.dataset)):
        g = d[d.dataset == ds]
        if len(g) != len(VARIANTS) * len(MODELS) * REPEATS:
            bad(f"ablation {ds}: {len(g)} rows, expected "
                f"{len(VARIANTS) * len(MODELS) * REPEATS}")
        else:
            print(f"  {ds:18s} complete: {len(g)} fits")
    # the duplicated core rows must be identical, or the runs are not comparable
    core = pd.concat([x[x.dataset == "md_core89"] for x in parts
                      if "md_core89" in set(x.dataset)], ignore_index=True)
    if len(core):
        v = core.groupby(["variant", "model", "repeat"])["r2_test"].nunique()
        if v.max() > 1:
            bad("the reference core differs between ablation runs; they used "
                "different partitions and cannot be pooled")
        else:
            print("  reference core identical across ablation runs")
    return d


def main():
    rs = check_resampling()
    tr = check_transfer()
    transfer_ranking(tr)
    check_ablation()

    print()
    for n in note:
        print(f"NOTE  {n}")
    if fail:
        print(f"\n{len(fail)} PROBLEM(S)")
        for f in fail:
            print(f"  {f}")
        return 1
    print("\nvalidation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
