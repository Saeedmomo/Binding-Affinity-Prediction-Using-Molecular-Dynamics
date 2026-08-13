"""Principal component analysis of ligand-only trajectories.

Produces the ligand conformational-dynamics descriptors: the explained variance ratios
of the first three components, and scalar measures of the motion each component
describes.

CORRECTION relative to the original script. The original wrote `pca.components_.shape`
and `average_structure.shape` into the descriptor columns. Those are array dimensions,
not values: `PCA-components-shape1` came out as the constant 3 for every molecule, and
`Average-structure-shape` as the atom count of the trajectory file. In the released
descriptor matrices those two columns correlate with pIC50 at r = 0.581, so a model
trained on them leans on trajectory bookkeeping rather than on conformational behaviour.

This version writes real quantities in their place:

    Explained-variance-ratio-pc1/2/3   as before, these were always correct
    PC1-rmsd-amplitude                 root-mean-square atomic displacement along PC1,
                                       in angstrom, i.e. how far the ligand actually
                                       moves along its dominant mode
    PC2-rmsd-amplitude                 the same for PC2
    Average-structure-radius-gyration  radius of gyration of the mean conformer, in
                                       angstrom, a size-and-shape measure of the average
                                       structure

For continuity, --legacy additionally emits the three original shape columns so that the
published descriptor matrices can still be reproduced exactly.

Usage
    python ligand_pca_feature_extraction.py --traj LigandOnly_*.pdb --out ligand_pca.csv
    python ligand_pca_feature_extraction.py --traj-dir path/to/ligands --plots plots/
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def pca_features(path: str, n_components: int = 3, drop_frames=(),
                 legacy: bool = False) -> dict:
    from ase.io import read
    from sklearn.decomposition import PCA

    frames = read(path, index=':')
    frames = [f for i, f in enumerate(frames)
              if len(f) > 0 and i not in set(drop_frames)]
    if not frames:
        raise ValueError('no usable frames')

    n_atoms = len(frames[0])
    keep = [f for f in frames if len(f) == n_atoms]
    if len(keep) < n_components + 1:
        raise ValueError(f'only {len(keep)} frames with a consistent {n_atoms} atoms; '
                         f'need more than {n_components}')

    coords = np.array([f.get_positions().ravel() for f in keep], float)

    pca = PCA(n_components=min(n_components, len(keep) - 1, coords.shape[1]))
    scores = pca.fit_transform(coords)
    evr = pca.explained_variance_ratio_

    mean_xyz = coords.mean(0).reshape(n_atoms, 3)
    com = mean_xyz.mean(0)
    rg = float(np.sqrt(((mean_xyz - com) ** 2).sum(1).mean()))

    # Physical amplitude of each mode: the eigenvector is unit length over 3N
    # coordinates, so scaling it by the spread of the projections and converting to a
    # per-atom root mean square gives an angstrom displacement.
    amps = []
    for k in range(pca.n_components_):
        vec = pca.components_[k].reshape(n_atoms, 3)
        amps.append(float(np.sqrt((vec ** 2).sum(1).mean()) * scores[:, k].std()))

    out = {
        'n_frames': len(keep),
        'n_atoms': n_atoms,
        'Explained-variance-ratio-pc1': float(evr[0]) if len(evr) > 0 else np.nan,
        'Explained-variance-ratio-pc2': float(evr[1]) if len(evr) > 1 else np.nan,
        'Explained-variance-ratio-pc3': float(evr[2]) if len(evr) > 2 else np.nan,
        'PC1-rmsd-amplitude': amps[0] if len(amps) > 0 else np.nan,
        'PC2-rmsd-amplitude': amps[1] if len(amps) > 1 else np.nan,
        'Average-structure-radius-gyration': rg,
    }
    if legacy:
        # exactly what the original script recorded, for reproducing published matrices
        out['PCA-components-shape1'] = int(pca.components_.shape[0])
        out['PCA-components-shape2'] = int(pca.components_.shape[1])
        out['Average-structure-shape'] = int(mean_xyz.shape[0])
    out['_scores'] = scores
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--traj', nargs='+', help='ligand-only trajectory files')
    src.add_argument('--traj-dir', help='directory searched for *.pdb')
    ap.add_argument('--out', default='ligand_pca_features.csv')
    ap.add_argument('--plots', default=None,
                    help='directory for PC1 against PC2 scatter plots')
    ap.add_argument('--components', type=int, default=3)
    ap.add_argument('--drop-frames', type=int, nargs='*', default=[],
                    help='frame indices to discard (the original hard-coded 1001)')
    ap.add_argument('--legacy', action='store_true',
                    help='also emit the original array-shape columns')
    a = ap.parse_args()

    paths = (sorted(glob.glob(os.path.join(a.traj_dir, '*.pdb')))
             if a.traj_dir else a.traj)
    if not paths:
        print('no trajectory files found', file=sys.stderr)
        return 1
    if a.plots:
        os.makedirs(a.plots, exist_ok=True)

    rows, failed = [], []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            rec = pca_features(p, a.components, a.drop_frames, a.legacy)
            scores = rec.pop('_scores')
            if a.plots:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.figure(figsize=(5, 4))
                plt.scatter(scores[:, 0], scores[:, 1], s=12, alpha=0.7)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.title(stem, fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(a.plots, f'PCA_{stem}.png'), dpi=200)
                plt.close()
            rows.append({'ligand': stem, **rec})
            print(f'  [ok] {stem:44s} {rec["n_frames"]:5d} frames  '
                  f'{rec["n_atoms"]:3d} atoms  '
                  f'EVR1 {rec["Explained-variance-ratio-pc1"]:.3f}  '
                  f'PC1 amp {rec["PC1-rmsd-amplitude"]:.3f} A')
        except Exception as e:
            failed.append((stem, str(e)))
            print(f'  [FAIL] {stem}: {e}', file=sys.stderr)

    if not rows:
        return 1
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f'\nwrote {a.out}: {len(rows)} ligands')
    if failed:
        print(f'{len(failed)} failed', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
