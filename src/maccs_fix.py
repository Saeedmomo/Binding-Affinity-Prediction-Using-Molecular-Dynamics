"""A verified MACCS block, because the one carried in the study matrix is not one.

WHAT IS WRONG. The 272-descriptor matrix `MDS_analysis/data_model/2.csv` carries 167
columns named MACCS_1 to MACCS_167. A MACCS key is a function of the molecular graph
alone, so two rows holding the same molecule must hold bit-for-bit identical keys.
They do not. Nineteen canonical structures occur more than once in the 122 rows,
giving 38 same-structure row pairs, and the stored block is identical in none of
them; one pair of rows holding the same 444.488 dalton anthraquinone differs in 52
of the 167 bits and in the number of bits set (24 against 56). Recomputing the keys
from the molecules reproduces the required identity in every parseable pair. The
stored block agrees with the correct one in 2 rows of 110 and is not a permutation
of it, so it is neither aligned nor merely reordered: it does not describe these
molecules.

WHAT IT AFFECTS. The simulation-derived core of 89 descriptors excludes this block
and is untouched. The 272-descriptor set contains it, so any comparison involving
that set is a comparison against 167 columns of noise, and an ablation showing that
the 89 descriptors beat the 272 is not evidence about fingerprints. This module
supplies a correct block so that comparison can be made honestly.

HOW THE KEYS ARE RECOVERED. The molecule is read from its mol2 file, which is the
same file every other descriptor set was computed from. Twelve of the 122 files
carry aromaticity or kekulisation errors that stop full sanitisation, so a fallback
chain is used and the route taken is recorded per row rather than hidden. The
function refuses to return unless the keys are identical within every recurring
structure, which is the property whose failure identified the problem.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "benchmark_work" / "src"))
import datasets  # noqa: E402

RDLogger.DisableLog("rdApp.*")
N_BITS = 167


def _partial(mol):
    """Sanitise everything except the step that failed, then perceive rings.

    Kekulisation and aromaticity failures do not affect the substructure matches
    MACCS keys are built from, but an unsanitised molecule has no ring information
    at all, and roughly a third of the keys ask about rings.
    """
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
    """A 122 by 167 frame of MACCS keys, with the parse route per row."""
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
        f"{failed} of {checked} same-structure pairs disagree; the recomputed block "
        f"has the same defect as the stored one and must not be used")
    return X, pd.Series(routes, name="parse_route"), checked


def main():
    X, routes, checked = maccs_frame()
    stored, _, _ = datasets.load("maccs167")
    A, B = stored.to_numpy(float), X.to_numpy(float)
    good = ~np.isnan(B).any(axis=1)
    agree = int(sum(np.array_equal(A[i], B[i]) for i in np.flatnonzero(good)))
    print(routes.value_counts().to_string())
    print(f"\nrecovered {int(good.sum())} of 122 rows")
    print(f"same-structure pairs verified identical: {checked}")
    print(f"stored block agrees with the recomputed block in {agree} of "
          f"{int(good.sum())} rows")
    print(f"mean bits set, stored {A[good].mean(axis=0).sum():.1f}, "
          f"recomputed {B[good].mean(axis=0).sum():.1f}")
    out = ROOT / "results" / "maccs_recomputed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    X.assign(parse_route=routes).to_csv(out, index=False)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
