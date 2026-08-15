"""The block ablation, redone with a fingerprint block that describes the molecules.

WHY IT HAD TO BE REDONE. The manuscript concluded that the 89 simulation-derived
descriptors beat the full 272-descriptor set, and read that as evidence that the
fingerprint and protein-confidence blocks carry nothing and the signal sits
unambiguously in the simulation. The 167 fingerprint columns of the 272-descriptor
matrix are not the MACCS keys of these molecules: rows holding the same molecule
disagree in up to 52 of the 167 bits, the block agrees with a correct recomputation
in 2 rows of 122, and it is not a permutation of the correct block either. An
ablation against 167 columns of noise cannot support a conclusion about
fingerprints, because adding noise to a small sample degrades any model whatever the
noise is. See maccs_fix.py for the diagnosis.

WHAT IS RUN. The same repeated nested resampling used everywhere else, with
structure-disjoint inner folds, over five representations built from verified blocks:

  md_core89              the 89 simulation-derived descriptors, unchanged
  maccs_correct          the 167 recomputed MACCS keys alone
  alphafold16            the 16 AlphaFold confidence summaries alone
  md_core89_maccs        89 + 167, isolating the fingerprint contribution
  md272_correct          89 + 16 + 167, the 272-descriptor set as it should have been

The comparison that matters is md272_correct against md_core89. If the corrected
272-descriptor set still loses, the original conclusion survives on honest data. If
it does not, the conclusion was an artefact of the corrupt block and has to go.

    python ablation.py --n-jobs 20
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "benchmark_work" / "src"))
sys.path.insert(0, str(ROOT))
import datasets  # noqa: E402
from maccs_fix import maccs_frame  # noqa: E402

RESULTS = ROOT / "results"
BLOCKS = ("md272_correct", "md_core89", "md_compact56", "md_core89_maccs",
          "maccs_correct", "alphafold16", "target_indicator")
LABEL = {"md_core89": "Simulation derived (89)",
         "md_compact56": "Compact trajectory subset (56)",
         "maccs_correct": "MACCS keys, recomputed (167)",
         "alphafold16": "AlphaFold confidence (16)",
         "md_core89_maccs": "Simulation derived plus recomputed MACCS (256)",
         "md272_correct": "Full 272-descriptor set with correct MACCS (272)",
         "target_indicator": "Protein identity alone, four indicator columns"}

# Columns removed to form the compact subset, each for a stated reason rather than by
# a selection procedure fitted to the data, so no held-out information is used to
# choose them and the subset needs no correction for selection.
ARTEFACT = ("PCA-components-shape1", "PCA-components-shape2",
            "Average-structure-shape")
NOT_FROM_TRAJECTORY = ("Docking score",)

_ORIGINAL_LOAD = datasets.load
_CACHE = {}


def _corrected(key):
    """Build a block combination from verified parts.

    The MACCS frame is recomputed from the mol2 files by maccs_fix, which refuses to
    return unless the keys are identical within every recurring structure. The other
    two blocks come from the study matrix unchanged, so any difference from the
    published ablation is attributable to the fingerprints alone.
    """
    if key in _CACHE:
        return _CACHE[key]
    core, y, meta = _ORIGINAL_LOAD("md_core89")
    af = _ORIGINAL_LOAD("alphafold16")[0]
    mac = maccs_frame()[0]
    assert not mac.isna().any().any(), "recomputed MACCS has gaps; refusing to run"
    if key == "md_compact56":
        # Three removals, none of them fitted to the outcome:
        #   columns that take the same value in all 122 systems, which cannot inform
        #   any prediction and are dropped by the variance gate inside every pipeline
        #   in any case;
        #   two columns recording array dimensions rather than conformational
        #   quantities, which correlate with potency more strongly than any genuine
        #   descriptor and are therefore an artefact of how the extraction was
        #   written;
        #   the docking score, which belongs to the pose that seeded the simulation
        #   and is the one column of the core not derived from the trajectory.
        # What remains is purely trajectory-derived and carries variance throughout.
        const = [c for c in core.columns if core[c].nunique(dropna=False) <= 1]
        drop = set(const) | set(ARTEFACT) | set(NOT_FROM_TRAJECTORY)
        X = core[[c for c in core.columns if c not in drop]]
        meta = dict(meta, key=key, label=LABEL[key], n_features=X.shape[1])
        assert X.shape == (122, 56), f"compact subset is {X.shape}, expected 122 by 56"
        _CACHE[key] = (X, y, meta)
        return _CACHE[key]
    if key == "target_indicator":
        # The reference point the paper has never stated. Every descriptor set is
        # asked to beat a model that knows only which of the four proteins a row
        # belongs to and nothing whatever about the molecule. Any representation
        # scoring below this is contributing nothing that a one-line lookup would
        # not, and the reader cannot judge the other numbers without it.
        ann = datasets.annotations()
        X = pd.get_dummies(ann["target"].iloc[meta["row_ids"]].reset_index(drop=True),
                           prefix="target").astype(float)
    else:
        parts = {"maccs_correct": [mac],
                 "alphafold16": [af],
                 "md_core89_maccs": [core, mac],
                 "md272_correct": [core, af, mac]}[key]
        X = pd.concat(parts, axis=1)
    expected = {"maccs_correct": 167, "alphafold16": 16,
                "md_core89_maccs": 256, "md272_correct": 272,
                "target_indicator": 4}[key]
    assert X.shape == (122, expected), f"{key}: got {X.shape}, expected 122 by {expected}"
    meta = dict(meta, key=key, label=LABEL[key], n_features=X.shape[1])
    _CACHE[key] = (X, y, meta)
    return _CACHE[key]


def _patched_load(key):
    if key in ("maccs_correct", "md_core89_maccs", "md272_correct",
               "target_indicator", "md_compact56"):
        return _corrected(key)
    return _ORIGINAL_LOAD(key)


datasets.load = _patched_load          # must precede the robust import chain
from robust import run  # noqa: E402
from sweep import MODELS, VARIANTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(BLOCKS))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=20)
    ap.add_argument("--out", default="ablation/ablation.csv")
    a = ap.parse_args()

    warnings.filterwarnings("ignore")
    path = RESULTS / a.out
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.perf_counter()
    for ds in a.datasets:
        for variant in a.variants:
            for model in a.models:
                rows.extend(run(ds, variant, model, a.repeats, False, a.n_jobs,
                                "group"))
                pd.DataFrame(rows).to_csv(path, index=False)
    d = pd.DataFrame(rows)
    s = (d.groupby(["dataset", "variant", "model"])["r2_test"]
         .median().reset_index().sort_values("r2_test", ascending=False))
    best = s.groupby("dataset").head(1).set_index("dataset")
    best.to_csv(path.parent / "ablation_best.csv")

    ref = float(best.loc["md_core89", "r2_test"])
    lines = []
    for ds in a.datasets:
        if ds == "md_core89":
            continue
        b = best.loc[ds]
        w = d[(d.dataset == ds) & (d.variant == b["variant"]) &
              (d.model == b["model"])].set_index("repeat")["r2_test"].sort_index()
        r = d[(d.dataset == "md_core89") &
              (d.variant == best.loc["md_core89", "variant"]) &
              (d.model == best.loc["md_core89", "model"])
              ].set_index("repeat")["r2_test"].sort_index()
        from scipy import stats
        lines.append({"Representation": LABEL[ds],
                      "Best learner": f"{b['model']} ({b['variant']})",
                      "Median held-out R2": round(float(b["r2_test"]), 4),
                      "Simulation-derived core": round(ref, 4),
                      "Median paired difference": round(float((w - r).median()), 4),
                      "Partitions where the core wins": f"{int((r > w).sum())} of 10",
                      "Wilcoxon p": round(float(stats.wilcoxon(r, w).pvalue), 4)})
    frame = pd.DataFrame(lines)
    frame.to_csv(path.parent / "ablation_tests.csv", index=False)
    print()
    print(best.round(4).to_string())
    print()
    print(frame.to_string(index=False))
    print(f"\nTOTAL {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
