"""Which interaction types actually reached the 20 residue descriptors.

The residue block is the novelty of the representation, and Table 1 of the manuscript
describes five contact types with geometric criteria and adopted energies: hydrogen
bonds, cation-pi contacts, hydrophobic contacts, ionic interactions and water bridges.
Three of the five never fired. The extraction script contains, and documents, three
defects:

  the two angle calls passed the same atom twice, so the cosine was 0/0 and both
  angles evaluated to not-a-number. Every comparison against a cutoff was then false,
  which silently removed hydrogen bonds and water bridges;

  cation-pi had a cutoff defined but no detection code at all.

The consequence is checkable without rerunning anything, because the two surviving
types are selective about residues. Hydrophobic contacts are counted only for ALA,
VAL, ILE, LEU, MET, PHE, TRP and TYR, ionic contacts only for ARG, LYS, ASP and GLU,
while hydrogen bonds and water bridges are counted for any residue with a nitrogen,
oxygen or sulphur, which is all twenty. If all five types had contributed, every
residue descriptor would be non-zero somewhere in 122 systems. This script tests
that.

The trajectories needed to recompute the block are not on this machine: the residue
descriptors cover 122 systems and 56 trajectory files exist, most of them cocrystal
comparisons. The finding is therefore reported rather than repaired, and the paper
must describe the block as what it is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "benchmark_work" / "src"))
import datasets  # noqa: E402

RESULTS = ROOT / "results"
AMINO = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
         "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR"}
CHARGED = {"ARG", "LYS", "ASP", "GLU"}
ENERGY = {"hbond": 5.0, "cation_pi": 3.5, "hydrophobic": 1.5, "ionic": 4.0,
          "water_bridge": 2.0}


def main():
    X, y, meta = datasets.load("md_core89")
    cols = {c.strip().upper(): c for c in X.columns if c.strip().upper() in AMINO}
    assert len(cols) == 20, f"expected 20 residue columns, found {len(cols)}"

    rows = []
    for a in AMINO:
        v = X[cols[a]].to_numpy(dtype=float)
        rows.append({
            "Residue": a,
            "Reachable by hydrophobic contacts": a in HYDROPHOBIC,
            "Reachable by ionic contacts": a in CHARGED,
            "Reachable only by the three suppressed types":
                a not in HYDROPHOBIC and a not in CHARGED,
            "Systems with a non-zero value": int((v != 0).sum()),
            "Maximum": round(float(v.max()), 3),
            "Mean": round(float(v.mean()), 3)})
    frame = pd.DataFrame(rows)

    dead = frame[frame["Systems with a non-zero value"] == 0]["Residue"].tolist()
    live = set(AMINO) - set(dead)
    verdict = live == (HYDROPHOBIC | CHARGED)

    # the descriptor is a sum over contact types before aggregation by residue type,
    # so a value is a weighted count and the weights cannot be separated afterwards
    summary = pd.DataFrame([{
        "Residue descriptors identically zero in all 122 systems": len(dead),
        "Which residues": ", ".join(dead),
        "Non-zero residues equal the hydrophobic and charged sets exactly": verdict,
        "Interaction types that contributed": "hydrophobic, ionic",
        "Interaction types suppressed": "hydrogen bond, cation-pi, water bridge",
        "Energy weights that were applied":
            f"hydrophobic {ENERGY['hydrophobic']}, ionic {ENERGY['ionic']} kcal per mole",
        "Energy weights never applied":
            f"hydrogen bond {ENERGY['hbond']}, cation-pi {ENERGY['cation_pi']}, "
            f"water bridge {ENERGY['water_bridge']} kcal per mole"}])

    # the constant columns matter beyond the residue block: they are carried into
    # every model as inputs that cannot inform any prediction
    const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    dims = pd.DataFrame([{
        "Columns in the simulation-derived set": X.shape[1],
        "Columns constant across all 122 systems": len(const),
        "Of those, residue descriptors": len([c for c in const
                                              if c.strip().upper() in AMINO]),
        "Columns carrying variance": X.shape[1] - len(const)}])

    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / "residue_audit.csv", index=False)
    summary.T.to_csv(RESULTS / "residue_audit_summary.csv", header=False)
    dims.T.to_csv(RESULTS / "residue_audit_dimensions.csv", header=False)

    print(frame.to_string(index=False))
    print()
    print(summary.T.to_string(header=False))
    print()
    print(dims.T.to_string(header=False))
    assert verdict, (
        "the non-zero residues do not match the hydrophobic and charged sets, so the "
        "explanation offered here for the zeros is wrong and must not be published")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
