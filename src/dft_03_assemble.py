"""Step 3 of the DFT benchmark: turn the per-molecule QC JSON into feature matrices.

Emits three files in data/dft/:
  quantum_descriptors.csv   one row per ligand, all numeric QC descriptors + PIC50
  quantum_summary.csv       the headline chemistry, for the manuscript's table
  qc_provenance.json        convergence, timings, basis sizes, failures

Also runs physical sanity checks that must hold if the calculations are sound, and
prints them, because a silently wrong quantum descriptor set is worse than none.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QCD = os.path.join(ROOT, 'data', 'dft')

# columns that are provenance, not chemistry
DROP = {'status', 'error', 'traceback', 'xtb_error', '_opt_xyz', '_dipole_next',
        'mol_id', 'mol_name', 'target', 'xyz'}
PROV = {'dft_seconds', 'xtb_opt_seconds', 'wall_seconds', 'dft_converged',
        'xtb_opt_ok', 'n_basis', 'n_electrons', 'row_index', 'PIC50'}

HEADLINE = ['E_HOMO_eV', 'E_LUMO_eV', 'HOMO_LUMO_gap_eV', 'dipole_total_D',
            'chemical_potential_eV', 'hardness_eV', 'electrophilicity_eV',
            'electronegativity_eV', 'softness_inv_eV',
            'q_mulliken_min', 'q_mulliken_max', 'q_mulliken_range',
            'quadrupole_aniso_au', 'xtb_Gsolv_water_Eh', 'xtb_gap_eV']


def main():
    files = sorted(glob.glob(os.path.join(QCD, 'qc_raw', '*.json')))
    if not files:
        raise SystemExit('no qc_raw/*.json found - run dft_02_run_qc.py first')
    recs = [json.load(open(f)) for f in files]
    df = pd.DataFrame(recs).sort_values('row_index').reset_index(drop=True)

    ok = df['status'] == 'ok'
    print(f'{len(df)} JSON records | ok={int(ok.sum())} failed={int((~ok).sum())}')
    if (~ok).any():
        print('\nFAILURES:')
        for r in df[~ok].itertuples():
            print(f'  {r.mol_id:9s} {r.mol_name:24s} {getattr(r, "error", "")[:140]}')

    prov = dict(
        n_records=int(len(df)), n_ok=int(ok.sum()), n_failed=int((~ok).sum()),
        failures=df.loc[~ok, ['mol_id', 'mol_name']].to_dict('records'),
        method='GFN2-xTB(opt, tight) then B3LYP/def2-SVP//RI-J single point (PySCF)',
        ecp='def2-svp ECP (iodine)',
        dft_converged=int(df.get('dft_converged', pd.Series(dtype=float)).fillna(0).sum()),
        n_basis_min=(int(df['n_basis'].min()) if 'n_basis' in df else None),
        n_basis_max=(int(df['n_basis'].max()) if 'n_basis' in df else None),
        total_cpu_minutes=round(float(df.get('wall_seconds', pd.Series([0])).sum()) / 60, 1),
        median_seconds_per_molecule=round(float(df.get('wall_seconds', pd.Series([0])).median()), 1),
    )

    dfo = df[ok].copy()
    num = [c for c in dfo.columns
           if c not in DROP and pd.api.types.is_numeric_dtype(dfo[c])]
    feat = [c for c in num if c not in PROV]

    # drop all-constant and all-NaN descriptor columns (e.g. q_*_Br_* when no Br present)
    const = [c for c in feat if dfo[c].nunique(dropna=True) <= 1]
    feat = [c for c in feat if c not in const]
    prov['dropped_constant_columns'] = const
    prov['n_features'] = len(feat)

    out = dfo[['row_index', 'mol_id', 'mol_name', 'target'] + feat + ['PIC50']]
    out.to_csv(os.path.join(QCD, 'quantum_descriptors.csv'), index=False)
    dfo[['row_index', 'mol_id', 'mol_name', 'target'] +
        [c for c in HEADLINE if c in dfo.columns] + ['PIC50']] \
        .to_csv(os.path.join(QCD, 'quantum_summary.csv'), index=False)
    json.dump(prov, open(os.path.join(QCD, 'qc_provenance.json'), 'w'), indent=1)

    print(f'\nfeature matrix: {len(out)} rows x {len(feat)} descriptors')
    print(f'dropped {len(const)} constant columns')
    print(f'basis functions {prov["n_basis_min"]}-{prov["n_basis_max"]}, '
          f'{prov["total_cpu_minutes"]:.0f} CPU-min total, '
          f'median {prov["median_seconds_per_molecule"]:.0f} s/molecule')

    print('\n--- headline chemistry ---')
    print(dfo[[c for c in HEADLINE if c in dfo.columns]].describe()
          .T[['mean', 'std', 'min', '50%', 'max']].round(3).to_string())

    # ------------------------------------------------------------ sanity checks
    print('\n--- physical sanity checks ---')
    checks = []

    def chk(name, cond, detail=''):
        checks.append((name, bool(cond), detail))
        print(f'  [{"PASS" if cond else "FAIL"}] {name}  {detail}')

    chk('all SCF converged', int(dfo['dft_converged'].sum()) == len(dfo),
        f'{int(dfo["dft_converged"].sum())}/{len(dfo)}')
    chk('HOMO below LUMO for every molecule',
        (dfo['E_LUMO_eV'] > dfo['E_HOMO_eV']).all())
    chk('gap in a chemically sane 0.5-9 eV window',
        dfo['HOMO_LUMO_gap_eV'].between(0.5, 9).all(),
        f'range {dfo["HOMO_LUMO_gap_eV"].min():.2f}-{dfo["HOMO_LUMO_gap_eV"].max():.2f} eV')

    # Frontier-orbital energies must be judged per charge state. In gas-phase DFT an
    # anion's extra electron is only weakly bound, so its HOMO rises and its LUMO can go
    # positive; that is a known limitation of the method, not a failed calculation.
    neut = dfo[dfo['charge'] == 0]
    chk('HOMO in -12..-3 eV for neutral ligands',
        neut['E_HOMO_eV'].between(-12, -3).all(),
        f'n={len(neut)}, range {neut["E_HOMO_eV"].min():.2f}..{neut["E_HOMO_eV"].max():.2f}')
    cat = dfo[dfo['charge'] > 0]
    if len(cat):
        chk('HOMO in -12..-3 eV for cations',
            cat['E_HOMO_eV'].between(-12, -3).all(),
            f'n={len(cat)}, range {cat["E_HOMO_eV"].min():.2f}..{cat["E_HOMO_eV"].max():.2f}')
    ani = dfo[dfo['charge'] < 0]
    if len(ani):
        names = ', '.join(ani['mol_id'])
        prov['anion_caveat'] = (
            f'{len(ani)} anionic ligand(s) ({names}) have destabilised frontier orbitals '
            f'(HOMO {ani["E_HOMO_eV"].max():.2f} eV, LUMO {ani["E_LUMO_eV"].max():+.2f} eV). '
            f'A positive LUMO means the extra electron is unbound in a gas-phase '
            f'def2-SVP treatment, so the Koopmans reactivity indices for these rows '
            f'(electrophilicity, chemical potential) are not comparable with the neutral '
            f'and cationic ligands and should be treated as flagged values.')
        print(f'  [NOTE] {prov["anion_caveat"]}')
    chk('hardness positive', (dfo['hardness_eV'] > 0).all())
    chk('dipole non-negative and < 40 D', dfo['dipole_total_D'].between(0, 40).all(),
        f'max {dfo["dipole_total_D"].max():.2f} D')
    chk('Mulliken charges sum to the formal charge',
        np.allclose(dfo['q_mulliken_sum_pos'] + dfo['q_mulliken_sum_neg'],
                    dfo['charge'], atol=1e-6)
        if 'charge' in dfo else False)

    # the decisive anchor: Mol_4 is unsubstituted 9,10-anthraquinone, D2h centrosymmetric
    a = dfo[dfo['mol_id'] == 'Mol_4']
    if len(a):
        d = float(a['dipole_total_D'].iloc[0])
        g = float(a['HOMO_LUMO_gap_eV'].iloc[0])
        chk('anthraquinone (Mol_4) dipole ~ 0 D by symmetry', d < 0.05, f'{d:.4f} D')
        chk('anthraquinone (Mol_4) B3LYP gap 3.5-4.8 eV', 3.5 < g < 4.8, f'{g:.3f} eV')

    # duplicate 2D structures must still differ, or the descriptors carry no pose info
    ann = pd.read_csv(os.path.join(ROOT, 'data', 'ligands_122_annotated.csv'))
    m = dfo.merge(ann[['row_index', 'smiles']], on='row_index')
    dup = m[m.duplicated('smiles', keep=False)]
    if len(dup):
        spreads = dup.groupby('smiles')['HOMO_LUMO_gap_eV'].agg(lambda s: s.max() - s.min())
        chk('duplicate 2D structures give distinguishable gaps',
            float(spreads.max()) > 1e-3,
            f'{len(spreads)} duplicate groups, max gap spread {spreads.max():.4f} eV, '
            f'median {spreads.median():.4f} eV')

    prov['sanity_checks'] = [dict(name=n, passed=p, detail=d) for n, p, d in checks]
    json.dump(prov, open(os.path.join(QCD, 'qc_provenance.json'), 'w'), indent=1)

    failed = [n for n, p, _ in checks if not p]
    print(f'\n{len(checks) - len(failed)}/{len(checks)} sanity checks passed')
    if failed:
        print('FAILED:', failed)
    print(f'\nwrote {QCD}\\quantum_descriptors.csv, quantum_summary.csv, qc_provenance.json')


if __name__ == '__main__':
    main()
