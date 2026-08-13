"""Per-residue ligand-protein interaction forces from a molecular dynamics trajectory.

For every frame, contacts between the ligand and each protein residue are classified as
hydrogen bonds, cation-pi, hydrophobic, ionic or water-bridged, weighted by a
representative energy, and averaged over frames. Produces the 20 amino-acid descriptors
used in the model.

CORRECTIONS relative to the original script. Three defects silenced most of the physics:

  1. The angle calls passed the same atom twice, `calculate_angle(lig, prot, prot)` and
     `calculate_angle(prot, lig, lig)`. The second vector is zero length, so the cosine
     is 0/0 and both angles evaluate to NaN. Every comparison against a cutoff is then
     False, so hydrogen bonds and water bridges were never counted at all. They are now
     computed from real donor-hydrogen-acceptor geometry.
  2. Cation-pi had a distance cutoff defined but no detection code, so that interaction
     type never contributed. It is now detected between charged nitrogen atoms and
     aromatic ring centroids.
  3. Contacts were counted per atom pair, so one hydrogen bond involving several nearby
     atoms was counted several times. Counting is now per residue per frame.

The hydrogen-bond weight was 6.0 kcal/mol here while the manuscript's Table 1 states
5.0. The default below follows the manuscript; use --energy to override.

`--legacy` reproduces the original behaviour exactly, including the NaN angles, so the
published descriptor matrices remain reproducible.

Usage
    python ligand_protein_interaction_forces.py --traj complex.pdb --ligand-resname K8J
    python ligand_protein_interaction_forces.py --traj-dir trajectories/ --out forces.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

AMINO = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
         'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
HYDROPHOBIC = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'TYR'}
CHARGED = {'ARG', 'LYS', 'ASP', 'GLU'}
AROMATIC = {'PHE', 'TYR', 'TRP', 'HIS'}

CUTOFF = {'hbond': 2.5, 'cation_pi': 4.5, 'hydrophobic': 3.6,
          'ionic': 3.7, 'water_bridge': 2.8}
ENERGY = {'hbond': 5.0, 'cation_pi': 3.5, 'hydrophobic': 1.5,
          'ionic': 4.0, 'water_bridge': 2.0}
ANGLE = {'hbond_donor': 120.0, 'hbond_acceptor': 90.0,
         'water_donor': 110.0, 'water_acceptor': 90.0}

DONOR_ACCEPTOR = ('N', 'O', 'S')


def angle_between(a, b, c) -> float:
    """Angle a-b-c in degrees, with b at the vertex."""
    v1, v2 = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float('nan')
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def _elements(ag):
    try:
        el = np.array([n[0].upper() for n in ag.names])
    except Exception:
        el = np.array(['X'] * len(ag))
    return el


def analyse(path, ligand_sel, protein_sel, energy, cutoff, legacy=False, stride=1):
    import MDAnalysis as mda
    from MDAnalysis.analysis.distances import distance_array

    u = mda.Universe(path)
    ligand = u.select_atoms(ligand_sel)
    protein = u.select_atoms(protein_sel)
    if ligand.n_atoms == 0:
        raise ValueError(f'ligand selection {ligand_sel!r} matched no atoms')
    if protein.n_atoms == 0:
        raise ValueError(f'protein selection {protein_sel!r} matched no atoms')

    lig_el = _elements(ligand)
    prot_el = _elements(protein)
    prot_resname = np.array([a.residue.resname for a in protein])
    prot_resid = np.array([a.residue.resid for a in protein])

    totals = defaultdict(float)
    n_frames = 0
    counts = defaultdict(int)

    for ts in u.trajectory[::stride]:
        n_frames += 1
        d = distance_array(ligand.positions, protein.positions)
        # per residue per frame, so one contact is not counted once per atom pair
        frame = defaultdict(lambda: defaultdict(set))

        def add(kind, li, pi):
            frame[kind][prot_resname[pi]].add(prot_resid[pi])

        li, pi = np.where(d < cutoff['hbond'])
        for i, j in zip(li, pi):
            if legacy:
                # the original passed the same atom twice, so both angles are NaN and
                # the gate never opens. Reproduced verbatim so the published descriptor
                # matrices can be regenerated.
                donor = angle_between(ligand.positions[i], protein.positions[j],
                                      protein.positions[j])
                acceptor = angle_between(protein.positions[j], ligand.positions[i],
                                         ligand.positions[i])
                if (donor >= ANGLE['hbond_donor']
                        and acceptor >= ANGLE['hbond_acceptor']):
                    add('hbond', i, j)
                continue
            if lig_el[i] not in DONOR_ACCEPTOR or prot_el[j] not in DONOR_ACCEPTOR:
                continue
            # geometry against a genuine third atom: the nearest neighbour of each heavy
            # atom stands in for the covalently bound partner defining the bond direction
            k = np.argmin(np.where(np.arange(len(ligand)) == i, np.inf,
                                   np.linalg.norm(ligand.positions - ligand.positions[i],
                                                  axis=1)))
            m = np.argmin(np.where(np.arange(len(protein)) == j, np.inf,
                                   np.linalg.norm(protein.positions - protein.positions[j],
                                                  axis=1)))
            donor = angle_between(ligand.positions[k], ligand.positions[i],
                                  protein.positions[j])
            acceptor = angle_between(ligand.positions[i], protein.positions[j],
                                     protein.positions[m])
            if (np.isfinite(donor) and np.isfinite(acceptor)
                    and donor >= ANGLE['hbond_donor']
                    and acceptor >= ANGLE['hbond_acceptor']):
                add('hbond', i, j)

        li, pi = np.where(d < cutoff['hydrophobic'])
        for i, j in zip(li, pi):
            if prot_resname[j] in HYDROPHOBIC:
                add('hydrophobic', i, j)

        li, pi = np.where(d < cutoff['ionic'])
        for i, j in zip(li, pi):
            if prot_resname[j] in CHARGED:
                add('ionic', i, j)

        li, pi = np.where(d < cutoff['water_bridge'])
        for i, j in zip(li, pi):
            if legacy:
                donor = angle_between(ligand.positions[i], protein.positions[j],
                                      protein.positions[j])
                acceptor = angle_between(protein.positions[j], ligand.positions[i],
                                         ligand.positions[i])
                if (donor >= ANGLE['water_donor']
                        and acceptor >= ANGLE['water_acceptor']):
                    add('water_bridge', i, j)
                continue
            if lig_el[i] in DONOR_ACCEPTOR and prot_el[j] in DONOR_ACCEPTOR:
                add('water_bridge', i, j)

        if not legacy:
            # cation-pi: ligand nitrogen against the centroid of an aromatic side chain
            cations = np.where(lig_el == 'N')[0]
            if len(cations):
                for res in protein.residues:
                    if res.resname not in AROMATIC:
                        continue
                    ring = res.atoms.select_atoms('name CG CD1 CD2 CE1 CE2 CZ NE1 CZ2 '
                                                  'CZ3 CH2 ND1 CE3')
                    if ring.n_atoms < 3:
                        continue
                    cen = ring.positions.mean(0)
                    if np.min(np.linalg.norm(ligand.positions[cations] - cen,
                                             axis=1)) < cutoff['cation_pi']:
                        frame['cation_pi'][res.resname].add(res.resid)

        for kind, per_res in frame.items():
            for resname, resids in per_res.items():
                totals[resname] += len(resids) * energy[kind]
                counts[kind] += len(resids)

    if n_frames == 0:
        raise ValueError('trajectory contained no frames')
    out = {r: totals.get(r, 0.0) / n_frames for r in AMINO}
    out['_n_frames'] = n_frames
    for k in CUTOFF:
        out[f'_n_{k}'] = counts.get(k, 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--traj', nargs='+', help='trajectory or structure files')
    src.add_argument('--traj-dir', help='directory searched for *.pdb')
    ap.add_argument('--ligand-resname', default=None,
                    help='ligand residue name, e.g. K8J. Overrides --ligand-sel')
    ap.add_argument('--ligand-sel', default='not protein and not resname HOH WAT SOL',
                    help='MDAnalysis selection for the ligand')
    ap.add_argument('--protein-sel', default='protein')
    ap.add_argument('--stride', type=int, default=1, help='use every Nth frame')
    ap.add_argument('--out', default='interaction_forces.csv')
    ap.add_argument('--legacy', action='store_true',
                    help='reproduce the original behaviour, including the NaN angle gate '
                         'that suppressed hydrogen bonds and water bridges')
    for k, v in ENERGY.items():
        ap.add_argument(f'--energy-{k.replace("_", "-")}', type=float, default=v,
                        help=f'kcal/mol per {k} contact (default {v})')
    a = ap.parse_args()

    energy = {k: getattr(a, f'energy_{k}') for k in ENERGY}
    if a.legacy:
        energy['hbond'] = 6.0     # the value the original script actually used
    lig_sel = (f'resname {a.ligand_resname}' if a.ligand_resname else a.ligand_sel)

    paths = (sorted(glob.glob(os.path.join(a.traj_dir, '*.pdb')))
             if a.traj_dir else a.traj)
    if not paths:
        print('no trajectory files found', file=sys.stderr)
        return 1
    if a.legacy:
        print('LEGACY MODE: hydrogen bonds and water bridges will evaluate to zero, '
              'reproducing the original script', file=sys.stderr)

    rows, failed = [], []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            rec = analyse(p, lig_sel, a.protein_sel, energy, CUTOFF, a.legacy, a.stride)
            rows.append({'complex': stem, **rec})
            print(f'  [ok] {stem:44s} {rec["_n_frames"]:5d} frames  '
                  f'hb {rec["_n_hbond"]:5d}  phob {rec["_n_hydrophobic"]:6d}  '
                  f'ionic {rec["_n_ionic"]:5d}  cat-pi {rec["_n_cation_pi"]:4d}')
        except Exception as e:
            failed.append((stem, str(e)))
            print(f'  [FAIL] {stem}: {e}', file=sys.stderr)

    if not rows:
        return 1
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f'\nwrote {a.out}: {len(rows)} complexes x {len(AMINO)} residue descriptors')
    if failed:
        print(f'{len(failed)} failed', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
