"""Emit the exact variable list behind every representation in the benchmark.

The benchmark compares representations of different sizes, and the reduction from
272 columns to 89 and then to 56 is the central claim. None of that is checkable
unless the membership of each set is published, so this writes the lists out and
records, for every column of the 89-descriptor core, which block it belongs to and
why it survives or is removed at each stage.

Removal criteria for the 56-column subset, all fixed from feature properties and
provenance rather than from any observed predictive outcome:

  constant     the column takes one value across all 122 systems, so it cannot
               inform any prediction. Thirty columns, comprising eight residue
               descriptors that no interaction type ever reached and twenty-one
               interaction energy terms for water and metal coordinates these
               systems do not contain, plus one array-shape constant.
  artefact     the column records an array dimension rather than a conformational
               quantity. Two such columns vary across systems and correlate with
               potency more strongly than any genuine descriptor.
  not from     the docking score belongs to the pose that seeded the simulation and
  trajectory   is the only column of the core not derived from the trajectory.

    python descriptor_lists.py --out ../benchmark_work/github_repo/data/descriptors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "benchmark_work" / "src"))
sys.path.insert(0, str(ROOT))
import datasets  # noqa: E402
from maccs_fix import maccs_frame  # noqa: E402

ARTEFACT = ("PCA-components-shape1", "PCA-components-shape2",
            "Average-structure-shape")
NOT_FROM_TRAJECTORY = ("Docking score",)
AMINO = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
         "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]


def block_of(col):
    """Which of the five blocks of the simulation-derived core a column belongs to."""
    c = col.strip()
    if c.upper() in AMINO:
        return "per-residue interaction"
    if c in NOT_FROM_TRAJECTORY:
        return "docking score"
    if any(k in c for k in ("stretch", "angle", "dihedral")):
        return "interaction energy decomposition"
    if "PCA" in c or "Average-structure" in c or "Explained" in c:
        return "ligand conformational motion"
    return "structural stability"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="descriptor_lists")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    core, y, meta = datasets.load("md_core89")
    af = datasets.load("alphafold16")[0]
    mac = maccs_frame()[0]

    const = [c for c in core.columns if core[c].nunique(dropna=False) <= 1]
    rows = []
    for i, c in enumerate(core.columns, start=1):
        reasons = []
        if c in const:
            reasons.append("constant across all 122 systems")
        if c in ARTEFACT:
            reasons.append("records an array dimension, not a conformational quantity")
        if c in NOT_FROM_TRAJECTORY:
            reasons.append("from the docking pose, not the trajectory")
        rows.append({
            "Index in the 89-column core": i,
            "Variable": c,
            "Block": block_of(c),
            "Constant across all systems": c in const,
            "In the 56-column compact subset": not reasons,
            "Reason for removal": "; ".join(reasons) if reasons else "",
            "Minimum": round(float(core[c].min()), 6),
            "Maximum": round(float(core[c].max()), 6),
            "Mean": round(float(core[c].mean()), 6)})
    core_tab = pd.DataFrame(rows)
    compact = core_tab[core_tab["In the 56-column compact subset"]]["Variable"].tolist()
    assert len(compact) == 56, f"compact subset is {len(compact)} columns, expected 56"

    core_tab.to_csv(out / "core_89_variables.csv", index=False)
    pd.DataFrame({"Index": range(1, 57), "Variable": compact}).to_csv(
        out / "compact_56_variables.csv", index=False)

    full = ([(c, "simulation derived core") for c in core.columns]
            + [(c, "AlphaFold model confidence") for c in af.columns]
            + [(c, "MACCS structural key, recomputed") for c in mac.columns])
    assert len(full) == 272, f"full set is {len(full)} columns, expected 272"
    pd.DataFrame({"Index": range(1, 273),
                  "Variable": [c for c, _ in full],
                  "Block": [b for _, b in full]}).to_csv(
        out / "full_272_variables.csv", index=False)

    summary = pd.DataFrame([
        {"Representation": "Full corrected set", "Columns": 272,
         "Composition": "89 simulation derived, 16 AlphaFold confidence, "
                        "167 recomputed MACCS keys"},
        {"Representation": "Core plus recomputed MACCS", "Columns": 256,
         "Composition": "89 simulation derived, 167 recomputed MACCS keys"},
        {"Representation": "Simulation derived core", "Columns": 89,
         "Composition": "five blocks, see core 89 variables"},
        {"Representation": "Compact trajectory subset", "Columns": 56,
         "Composition": "the core less 30 constant columns, 2 array-shape "
                        "artefacts and the docking score"}])
    summary.to_csv(out / "representation_summary.csv", index=False)

    lines = ["# Descriptor variable lists", "",
             "Exact membership of every representation used in the benchmark, so that "
             "the reduction from 272 columns to 89 and then to 56 can be checked "
             "rather than taken on trust.", "",
             "| File | Contents |", "| --- | --- |",
             "| `full_272_variables.csv` | all 272 columns with their block |",
             "| `core_89_variables.csv` | the 89 simulation-derived columns, with "
             "block, range, and whether each survives into the compact subset |",
             "| `compact_56_variables.csv` | the 56 purely trajectory-derived columns |",
             "| `representation_summary.csv` | one row per representation |", "",
             "## How the 56-column subset is defined", "",
             "89 minus 30 constant columns, minus 2 array-shape artefacts, minus the "
             "docking score. Every criterion is a property of the feature or its "
             "provenance. None uses the outcome, so the subset needs no correction "
             "for selection.", "",
             "## Blocks of the simulation-derived core", ""]
    counts = core_tab.groupby("Block").agg(
        columns=("Variable", "size"),
        constant=("Constant across all systems", "sum"),
        in_compact=("In the 56-column compact subset", "sum")).reset_index()
    lines += ["| Block | Columns | Constant | In the 56 |", "| --- | --- | --- | --- |"]
    for r in counts.itertuples():
        lines.append(f"| {r.Block} | {r.columns} | {int(r.constant)} | "
                     f"{int(r.in_compact)} |")
    lines += ["", "The eight per-residue descriptors that are constant are ASN, CYS, "
                  "GLN, GLY, HIS, PRO, SER and THR. They are exactly the residues "
                  "reachable only by the three interaction types that the extraction "
                  "code never evaluated; see `docs/RESIDUE_BLOCK_AUDIT.md`.", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(counts.to_string(index=False))
    print(f"\nwrote 4 CSV files and a README to {out}")
    print(f"  272 columns, 89 core, {len(compact)} compact, {len(const)} constant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
