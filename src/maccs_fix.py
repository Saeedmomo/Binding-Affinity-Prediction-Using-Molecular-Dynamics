"""Compute and verify the 167-bit MACCS structural key block for the 122 systems.

A MACCS key is a function of the molecular graph alone, so two rows holding the same
molecule must hold bit-for-bit identical keys. That property is the natural check on
the computation, and this module treats it as a gate rather than an expectation: the
frame is not returned unless the keys agree for every pair of rows sharing a
canonical structure. Nineteen structures occur more than once across the four
targets, giving 38 such pairs.

Molecules are read from the same mol2 files every other descriptor set is computed
from, so the block is aligned with the rest of the study by construction. Twelve of
the 122 files carry aromaticity or kekulisation errors that stop full sanitisation.
Rather than dropping those rows or silently accepting an unsanitised molecule, a
fallback chain is used and the route taken is recorded per row:

    mol2 sanitized  -> mol2 partial -> smiles sanitized -> smiles partial

The partial route sanitises everything except the step that failed and then perceives
rings explicitly. Kekulisation and aromaticity failures do not affect the
substructure matches MACCS keys are built from, but an unsanitised molecule carries
no ring information at all and roughly a third of the keys ask about rings, so
recovering rings is what makes the partial route usable.

    python maccs_fix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys

ROOT = Path(__file__).resolve().parent
for candidate in (ROOT.parent / "benchmark_work" / "src", ROOT):
    if (candidate / "datasets.py").exists():
        sys.path.insert(0, str(candidate))
        break
import datasets  # noqa: E402

RDLogger.DisableLog("rdApp.*")
N_BITS = 167


def _partial(mol):
    """Sanitise everything except the step that failed, then perceive rings."""
    if mol is None:
        return None
    ops = (Chem.SanitizeFlags.SANITIZE_ALL
           ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
           ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    try:
        Chem.SanitizeMol(mol, sanitizeOps=ops)
        Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_RDKIT)
    except Exception:
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
        except Exception:
            return None
    return mol


def _molecule(path, smiles):
    for route, get in (
            ("mol2 sanitized", lambda: Chem.MolFromMol2File(path, sanitize=True)),
            ("mol2 partial", lambda: _partial(Chem.MolFromMol2File(path,
                                                                   sanitize=False))),
            ("smiles sanitized", lambda: Chem.MolFromSmiles(smiles)),
            ("smiles partial", lambda: _partial(Chem.MolFromSmiles(smiles,
                                                                   sanitize=False)))):
        try:
            mol = get()
        except Exception:
            mol = None
        if mol is not None and mol.GetNumAtoms() > 0:
            return mol, route
    return None, "failed"


def maccs_frame():
    """A 122 by 167 frame of MACCS keys, with the parse route per row.

    Raises rather than returning if the keys are not identical within every recurring
    structure, which is the defining property of a graph-based fingerprint.
    """
    ann = datasets.annotations()
    bits, routes = [], []
    for _, r in ann.iterrows():
        mol, route = _molecule(r["mol2_path"], r["smiles"])
        routes.append(route)
        if mol is None:
            bits.append([np.nan] * N_BITS)
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        bits.append([int(fp.GetBit(k)) for k in range(N_BITS)])
    X = pd.DataFrame(bits, columns=[f"MACCS_{k + 1}" for k in range(N_BITS)])

    groups = ann["smiles"].to_numpy()
    checked = failed = 0
    for s in pd.unique(groups):
        idx = [i for i in np.flatnonzero(groups == s) if not X.iloc[i].isna().any()]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                checked += 1
                failed += not np.array_equal(X.iloc[idx[a]].to_numpy(),
                                             X.iloc[idx[b]].to_numpy())
    assert failed == 0, (
        f"{failed} of {checked} same-structure pairs disagree; the computed block "
        f"fails the graph-invariance check and must not be used")
    return X, pd.Series(routes, name="parse_route"), checked


def main():
    X, routes, checked = maccs_frame()
    good = ~X.isna().any(axis=1)
    print(routes.value_counts().to_string())
    print(f"\ncomputed {int(good.sum())} of 122 rows")
    print(f"same-structure pairs verified identical: {checked}")
    print(f"mean bits set: {X[good].to_numpy().sum(axis=1).mean():.1f}")
    out = ROOT.parent / "results" / "benchmark" / "maccs_recomputed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    X.assign(parse_route=routes).to_csv(out, index=False)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
