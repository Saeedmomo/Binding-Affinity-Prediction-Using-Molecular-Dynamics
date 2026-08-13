"""Step 1 of the DFT benchmark: export the 122 study ligands as XYZ for xtb/PySCF.

Two decisions here matter scientifically.

1. GEOMETRY SOURCE. Coordinates are taken from the docked/MD pose stored in each .mol2
   file, not from a fresh 2D->3D embedding. 28 of the 122 rows are duplicate 2D
   structures (94 unique SMILES over 122 rows, docs/GROUND_TRUTH.md section 6) that
   differ only in the conformation adopted against their respective target. Starting
   from the pose is the only way quantum descriptors can carry any per-complex
   information at all.

2. TOTAL CHARGE. RDKit's mol2 reader ignores Sybyl `N.4` (protonated amine) typing and
   returned charge 0 for 14 protonated ligands and mis-assigned two others, which left
   17 molecules with an ODD electron count -- i.e. they would have been treated as
   open-shell doublets. Drug-like ligands here are closed-shell, so that would have
   produced meaningless orbital energies.
   The authoritative source is the mol2 partial-charge column, which the docking/MD
   preparation wrote and which sums to a clean integer for every molecule
   (106 neutral, 15 at +1, 1 carboxylate at -1; |residual| < 5e-4 in all cases).
   We therefore take charge = round(sum of mol2 partial charges) and assert that every
   resulting electron count is even.

   One molecule needs an explicit override. CHEMBL448651 has composition C12H13ClN4
   (sum Z = 130) and a mol2 charge sum of +0.990, but +1 would make it a 129-electron
   radical cation. Its guanidine group is Sybyl-typed `C.cat` as if protonated while one
   N-H is absent from the file; neutral C12H13ClN4 has an integral degree of unsaturation
   (8) and is closed-shell, so the composition is decisive and the charge is 0. The rule
   below is therefore: prefer the rounded mol2 charge sum, but if it violates electron
   parity, fall back to the nearest charge that does not -- and log the override.
"""
import os
import numpy as np
import pandas as pd

ROOT = r'D:\Chua files\fourth paper\benchmark_work'
OUT = os.path.join(ROOT, 'data', 'dft', 'xyz')
os.makedirs(OUT, exist_ok=True)

Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16,
     'Cl': 17, 'Br': 35, 'I': 53}


def parse_mol2(path):
    """Return (atoms, charge_sum). atoms = [(element, x, y, z, sybyl, q), ...]."""
    txt = open(path, errors='replace').read()
    if '@<TRIPOS>ATOM' not in txt:
        raise ValueError(f'no ATOM record in {path}')
    sec = txt.split('@<TRIPOS>ATOM')[1].split('@<TRIPOS>')[0]
    atoms, qsum = [], 0.0
    for ln in sec.strip().splitlines():
        f = ln.split()
        if len(f) < 6:
            continue
        sybyl = f[5]
        elem = sybyl.split('.')[0]
        # Sybyl writes some elements in caps (CL, BR); normalise to title case
        if elem.upper() in ('CL', 'BR'):
            elem = elem.capitalize()
        q = float(f[8]) if len(f) > 8 else 0.0
        qsum += q
        atoms.append((elem, float(f[2]), float(f[3]), float(f[4]), sybyl, q))
    return atoms, qsum


lig = pd.read_csv(os.path.join(ROOT, 'data', 'ligands_122_annotated.csv'))
assert len(lig) == 122

rows, problems, overrides = [], [], []
for r in lig.itertuples():
    atoms, qsum = parse_mol2(r.mol2_path)

    unknown = sorted({a[0] for a in atoms} - set(Z))
    if unknown:
        problems.append(f'{r.mol_name}: unknown elements {unknown}')
        continue

    zsum = sum(Z[a[0]] for a in atoms)
    charge = int(round(qsum))
    residual = abs(qsum - charge)

    if (zsum - charge) % 2 != 0:
        # parity violation: take the nearest charge that yields a closed shell
        cands = sorted((c for c in range(-2, 3) if (zsum - c) % 2 == 0),
                       key=lambda c: (abs(c - qsum), abs(c)))
        overrides.append(f'{r.mol_name}: mol2 charge sum {qsum:+.3f} -> rounded '
                         f'{charge:+d} gives {zsum - charge} electrons (odd); '
                         f'overriding to {cands[0]:+d}')
        charge = cands[0]

    nelec = zsum - charge
    if nelec % 2 != 0:
        problems.append(f'{r.mol_name}: ODD electron count {nelec} at charge {charge}')
        continue

    lines = [str(len(atoms)),
             f'{r.mol_name} charge={charge} mult=1 PIC50={r.PIC50}']
    for elem, x, y, z_, _s, _q in atoms:
        lines.append(f'{elem:<3s} {x:14.8f} {y:14.8f} {z_:14.8f}')

    fn = f'{r.row_index:03d}_{r.mol_id}.xyz'
    with open(os.path.join(OUT, fn), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    rows.append(dict(row_index=r.row_index, mol_id=r.mol_id, mol_name=r.mol_name,
                     PIC50=r.PIC50, target=r.target, smiles=r.smiles, xyz=fn,
                     n_atoms=len(atoms), n_heavy=sum(1 for a in atoms if a[0] != 'H'),
                     charge=charge, mult=1, nelec=nelec,
                     mol2_charge_sum=round(qsum, 5), charge_residual=round(residual, 6),
                     elements=','.join(sorted({a[0] for a in atoms}))))

man = pd.DataFrame(rows)
man.to_csv(os.path.join(ROOT, 'data', 'dft', 'manifest.csv'), index=False)

print(f'exported {len(man)}/122 geometries to {OUT}')
if overrides:
    print('\n*** CHARGE OVERRIDES (parity-driven) ***')
    for o in overrides:
        print('   ', o)
if problems:
    print('\n*** PROBLEMS ***')
    for p in problems:
        print('   ', p)
else:
    print('all 122 molecules are closed-shell (even electron count) at the assigned charge')

print('\natom counts : min=%d median=%.0f max=%d  total=%d'
      % (man.n_atoms.min(), man.n_atoms.median(), man.n_atoms.max(), man.n_atoms.sum()))
print('charges     :', man.charge.value_counts().to_dict())
print('max |residual| between summed partial charges and the integer: %.2e'
      % man.charge_residual.max())
print('elements    :', sorted({e for s in man.elements for e in s.split(',')}))
print('\ncharged ligands:')
print(man[man.charge != 0][['mol_id', 'mol_name', 'charge', 'mol2_charge_sum']]
      .to_string(index=False))
assert len(man) == 122, 'not all molecules exported'
