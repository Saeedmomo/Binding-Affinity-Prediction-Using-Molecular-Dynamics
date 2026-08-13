"""Step 2 of the DFT benchmark: quantum-chemistry descriptor generation (runs in WSL).

Two-tier protocol, chosen so that 122 drug-sized ligands are tractable without
sacrificing the quality of the electronic descriptors:

  Tier 1  GFN2-xTB geometry optimisation from the docked/MD pose (seconds per molecule).
          Also yields an ALPB(water) solvation free energy, used as its own descriptor.
  Tier 2  B3LYP/def2-SVP single-point on the xTB geometry with PySCF, density-fitted.
          Source of HOMO, LUMO, gap, dipole, traceless quadrupole and the
          Mulliken / meta-Lowdin partial charges.

Geometry at the semi-empirical level, electronic structure at DFT: the standard,
defensible compromise for descriptor generation on a set this size. Gas phase is used
for the orbital energies (the convention for conceptual-DFT reactivity indices); the
solvation term is carried separately as its own feature. Iodine is present in one
ligand, so def2-SVP is paired with its matching ECP.

Two performance findings, both measured rather than assumed:

  * PySCF must be limited to a few threads per process. Left to grab all 24 cores it
    took 295 s for a 264-basis-function molecule, versus 48 s on 4 threads
    (oversubscription). Throughput comes from running molecules in parallel, not from
    threading one molecule wide, and OMP_NUM_THREADS must be set before pyscf is
    imported because OpenBLAS binds its pool at load time.
  * Density fitting is NOT used. With 6 concurrent workers on a 15 GB machine, RI-J
    spilled its 3-centre integral cache to disk -- 8 GB per worker, 49 GB of /tmp -- and
    thrashed. Conventional direct SCF was also simply faster on this molecule size when
    measured head to head (39.1 s vs 47.4 s at 264 basis functions), so it is used
    throughout. Integrals are recomputed on the fly and memory stays flat.
  * GPU offload was evaluated and rejected on this hardware. GPU4PySCF 1.8.1 + CuPy
    14.1.1 run correctly on the available RTX 5060 Laptop GPU (sm_120, CUDA 13.1) and
    reproduce the CPU result exactly (E = -2233.93132 Eh, gap 3.357 eV on the 742-basis-
    function ligand), but take 595 s against 548 s for the same molecule on CPU. Consumer
    Blackwell parts throttle FP64 to about 1/64 of FP32 -- measured here at ~53 GFLOPS
    double precision -- and closed-shell SCF is FP64-bound, so 24 CPU cores are faster.
    On a datacentre card (A100/H100) the conclusion would reverse; keep the CPU path
    unless one is available.

Validation anchor: molecule Mol_4 is unsubstituted 9,10-anthraquinone (D2h,
centrosymmetric). Its dipole must come out at ~0 D and its B3LYP gap near 4.1 eV.

Usage (from Windows PowerShell):
  wsl -d Ubuntu-24.04 -- bash -lc "$HOME/.local/bin/micromamba run -r $HOME/micromamba \
      -n qc python3 '/mnt/d/.../src/dft_02_run_qc.py' --workers 6 --threads 4"
"""
import os

# Thread limits MUST be set before numpy/pyscf are imported: OpenBLAS and OpenMP read
# them once at library load time, and on Linux the multiprocessing children inherit the
# already-initialised pools by fork. Setting them inside the worker was ineffective --
# each worker still grabbed ~9 threads. Pass THREADS_PER_WORKER from the launcher.
_T = os.environ.get('THREADS_PER_WORKER', '4')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = _T

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

HARTREE_EV = 27.211386245988
ROOT = '/mnt/d/Chua files/fourth paper/benchmark_work'
XYZ = os.path.join(ROOT, 'data', 'dft', 'xyz')
OUTD = os.path.join(ROOT, 'data', 'dft', 'qc_raw')

# atomic numbers, for the nuclear contribution to the multipoles
Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16,
     'Cl': 17, 'Br': 35, 'I': 53}
# def2 ECPs replace core electrons, so the *effective* nuclear charge is what the
# electronic structure sees. Only iodine carries an ECP in def2-SVP here (28 core e-).
ECP_CORE = {'I': 28}


# --------------------------------------------------------------------------- xtb
def run_xtb(xyz_path, charge, workdir, threads):
    props = {}
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    t0 = time.time()
    r = subprocess.run(['xtb', os.path.abspath(xyz_path), '--opt', 'tight',
                        '--chrg', str(charge), '--uhf', '0', '--gfn', '2'],
                       cwd=workdir, capture_output=True, text=True,
                       timeout=7200, env=env)
    props['xtb_opt_seconds'] = round(time.time() - t0, 2)
    props['xtb_opt_ok'] = int(r.returncode == 0)

    opt = os.path.join(workdir, 'xtbopt.xyz')
    if not os.path.exists(opt):
        props['xtb_error'] = (r.stderr or r.stdout)[-600:]
        return None, props

    for line in r.stdout.splitlines():
        s = line.strip()
        if 'HOMO-LUMO GAP' in s:
            for tok in s.split():
                try:
                    props['xtb_gap_eV'] = float(tok)
                    break
                except ValueError:
                    continue
        elif s.startswith('| TOTAL ENERGY'):
            try:
                props['xtb_total_energy_Eh'] = float(s.split()[3])
            except (IndexError, ValueError):
                pass
        elif 'Mol. a(0)' in s or 'Mol. α(0)' in s:
            try:
                props['xtb_polarizability_au'] = float(s.split()[-1])
            except (IndexError, ValueError):
                pass

    # ALPB(water) single point on the optimised geometry -> solvation free energy
    r2 = subprocess.run(['xtb', 'xtbopt.xyz', '--chrg', str(charge), '--uhf', '0',
                         '--gfn', '2', '--alpb', 'water'],
                        cwd=workdir, capture_output=True, text=True,
                        timeout=7200, env=env)
    for line in r2.stdout.splitlines():
        s = line.strip()
        for key, tag in (('xtb_Gsolv_water_Eh', '-> Gsolv'),
                         ('xtb_Gsasa_Eh', '-> Gsasa'),
                         ('xtb_Ghb_Eh', '-> Ghb')):
            if s.startswith(tag):
                for tok in reversed(s.split()):
                    try:
                        props[key] = float(tok)
                        break
                    except ValueError:
                        continue
    if 'xtb_total_energy_Eh' in props and 'xtb_Gsolv_water_Eh' in props:
        props['xtb_logP_proxy'] = -props['xtb_Gsolv_water_Eh']
    return opt, props


# -------------------------------------------------------------------------- pyscf
def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0].split()[0])
    return [(p[0], (float(p[1]), float(p[2]), float(p[3])))
            for p in (l.split() for l in lines[2:2 + n])]


def run_dft(opt_xyz, charge, conv_tol=1e-8, grid_level=3, max_memory=2000):
    import numpy as np
    from pyscf import gto, dft

    atoms = read_xyz(opt_xyz)
    mol = gto.M(atom=atoms, basis='def2-svp', ecp='def2-svp', charge=charge, spin=0,
                unit='Angstrom', verbose=0, max_memory=max_memory)

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.direct_scf = True           # no DF: see the performance note in the module docstring
    mf.grids.level = grid_level
    mf.conv_tol = conv_tol
    mf.max_cycle = 200
    t0 = time.time()
    e_tot = mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        e_tot = mf.kernel()
    secs = time.time() - t0

    occ = mf.mo_occ > 0
    mo = mf.mo_energy
    e_homo, e_lumo = mo[occ].max(), mo[~occ].min()
    occ_sorted = np.sort(mo[occ])
    vir_sorted = np.sort(mo[~occ])

    # ---- multipoles about the centre of *effective* nuclear charge (origin-independent)
    coords = mol.atom_coords()                       # bohr
    zeff = np.array([Z[mol.atom_symbol(i)] - ECP_CORE.get(mol.atom_symbol(i), 0)
                     for i in range(mol.natm)], float)
    origin = (zeff[:, None] * coords).sum(0) / zeff.sum()

    with mol.with_common_orig(origin):
        dm = mf.make_rdm1()
        dip_ao = mol.intor('int1e_r', comp=3)
        dip_el = -np.einsum('xij,ji->x', dip_ao, dm).real
        rr_ao = mol.intor('int1e_rr').reshape(3, 3, mol.nao, mol.nao)
        m_el = -np.einsum('xyij,ji->xy', rr_ao, dm).real

    rel = coords - origin
    dip_nuc = (zeff[:, None] * rel).sum(0)
    m_nuc = np.einsum('a,ax,ay->xy', zeff, rel, rel)

    dip_au = dip_nuc + dip_el                        # e*bohr
    dip_D = dip_au * 2.541746473                     # -> Debye
    m_tot = m_nuc + m_el                             # second moment, a.u.
    quad = 3.0 * m_tot - np.eye(3) * np.trace(m_tot)  # traceless quadrupole, a.u.
    qeig = np.sort(np.linalg.eigvalsh(quad))

    _p, chg_mul = mf.mulliken_pop(verbose=0)
    try:
        _p, chg_low = mf.mulliken_pop_meta_lowdin_ao(verbose=0)
    except Exception:
        chg_low = None

    return dict(
        dft_converged=int(bool(mf.converged)),
        dft_seconds=round(secs, 2),
        n_basis=int(mol.nao),
        n_electrons=int(mol.nelectron),
        E_total_Eh=float(e_tot),
        E_HOMO_eV=float(e_homo * HARTREE_EV),
        E_LUMO_eV=float(e_lumo * HARTREE_EV),
        E_HOMO_minus1_eV=float(occ_sorted[-2] * HARTREE_EV) if occ_sorted.size > 1 else float('nan'),
        E_LUMO_plus1_eV=float(vir_sorted[1] * HARTREE_EV) if vir_sorted.size > 1 else float('nan'),
        dipole_x_D=float(dip_D[0]), dipole_y_D=float(dip_D[1]), dipole_z_D=float(dip_D[2]),
        dipole_total_D=float(np.linalg.norm(dip_D)),
        quadrupole_aniso_au=float(np.sqrt(1.5 * (quad ** 2).sum())),
        quadrupole_eig_min_au=float(qeig[0]),
        quadrupole_eig_max_au=float(qeig[-1]),
        quadrupole_asymmetry=float((qeig[1] - qeig[0]) / (qeig[-1] - qeig[0]))
        if abs(qeig[-1] - qeig[0]) > 1e-12 else float('nan'),
        _charges_mulliken=np.asarray(chg_mul).tolist(),
        _charges_lowdin=(np.asarray(chg_low).tolist() if chg_low is not None else None),
        _symbols=[mol.atom_symbol(i) for i in range(mol.natm)],
    )


# --------------------------------------------------------------------- descriptors
def charge_descriptors(chg, syms, prefix):
    import numpy as np
    chg = np.asarray(chg, float)
    syms = np.asarray(syms)
    d = {f'{prefix}_max': chg.max(), f'{prefix}_min': chg.min(),
         f'{prefix}_range': chg.max() - chg.min(),
         f'{prefix}_absmax': np.abs(chg).max(),
         f'{prefix}_mean_abs': np.abs(chg).mean(),
         f'{prefix}_std': chg.std(),
         f'{prefix}_sum_pos': float(chg[chg > 0].sum()) if (chg > 0).any() else 0.0,
         f'{prefix}_sum_neg': float(chg[chg < 0].sum()) if (chg < 0).any() else 0.0}
    for el in ('C', 'N', 'O', 'S', 'H', 'F', 'Cl', 'Br', 'I', 'P'):
        m = syms == el
        d[f'{prefix}_{el}_mean'] = float(chg[m].mean()) if m.any() else 0.0
        d[f'{prefix}_{el}_min'] = float(chg[m].min()) if m.any() else 0.0
        d[f'{prefix}_{el}_max'] = float(chg[m].max()) if m.any() else 0.0
    return {k: float(v) for k, v in d.items()}


def reactivity_indices(homo_eV, lumo_eV):
    """Conceptual-DFT indices in the Koopmans approximation (eV)."""
    ip, ea = -homo_eV, -lumo_eV
    mu = -(ip + ea) / 2.0
    eta = (ip - ea) / 2.0
    out = dict(HOMO_LUMO_gap_eV=lumo_eV - homo_eV,
               ionization_potential_eV=ip, electron_affinity_eV=ea,
               chemical_potential_eV=mu, electronegativity_eV=-mu, hardness_eV=eta)
    ok = abs(eta) > 1e-9
    out['softness_inv_eV'] = 1.0 / eta if ok else float('nan')
    out['electrophilicity_eV'] = mu ** 2 / (2 * eta) if ok else float('nan')
    out['max_charge_transfer'] = -mu / eta if ok else float('nan')
    return out


# ------------------------------------------------------------------------- worker
def process_one(job):
    rec_in, threads, conv_tol, grid = job
    dst = os.path.join(OUTD, f"{rec_in['row_index']:03d}_{rec_in['mol_id']}.json")
    if os.path.exists(dst):
        return f"  [skip  ] {rec_in['mol_id']}"

    t0 = time.time()
    rec = dict(rec_in)
    try:
        with tempfile.TemporaryDirectory() as wd:
            opt, xp = run_xtb(os.path.join(XYZ, rec_in['xyz']), rec_in['charge'], wd, threads)
            rec.update(xp)
            if opt is None:
                raise RuntimeError('xtb optimisation produced no geometry')
            rec['_opt_xyz'] = open(opt).read()
            d = run_dft(opt, rec_in['charge'], conv_tol=conv_tol, grid_level=grid)
        syms = d.pop('_symbols')
        cm = d.pop('_charges_mulliken')
        cl = d.pop('_charges_lowdin')
        rec.update(d)
        rec.update(reactivity_indices(d['E_HOMO_eV'], d['E_LUMO_eV']))
        rec.update(charge_descriptors(cm, syms, 'q_mulliken'))
        if cl is not None:
            rec.update(charge_descriptors(cl, syms, 'q_lowdin'))
        rec['status'] = 'ok'
    except Exception as e:
        rec['status'] = 'failed'
        rec['error'] = f'{type(e).__name__}: {e}'
        rec['traceback'] = traceback.format_exc()[-1500:]

    rec['wall_seconds'] = round(time.time() - t0, 1)
    tmp = dst + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, dst)
    return (f"  [{rec['status']:6s}] {rec_in['mol_id']:9s} {rec_in['mol_name']:24s} "
            f"n={rec_in['n_atoms']:3d} {rec['wall_seconds']:7.1f}s "
            f"gap={rec.get('HOMO_LUMO_gap_eV', float('nan')):7.3f} eV "
            f"dip={rec.get('dipole_total_D', float('nan')):6.2f} D")


def main():
    import pandas as pd
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--conv-tol', type=float, default=1e-8)
    ap.add_argument('--grid', type=int, default=3)
    ap.add_argument('--only', type=str, default=None, help='comma-separated mol_ids')
    ap.add_argument('--redo', action='store_true', help='delete existing json first')
    a = ap.parse_args()

    os.makedirs(OUTD, exist_ok=True)
    man = pd.read_csv(os.path.join(ROOT, 'data', 'dft', 'manifest.csv'))
    if a.only:
        man = man[man.mol_id.isin(set(a.only.split(',')))]

    jobs = []
    for r in man.itertuples():
        dst = os.path.join(OUTD, f'{r.row_index:03d}_{r.mol_id}.json')
        if a.redo and os.path.exists(dst):
            os.remove(dst)
        jobs.append((dict(row_index=int(r.row_index), mol_id=r.mol_id,
                          mol_name=r.mol_name, PIC50=float(r.PIC50),
                          target=r.target, charge=int(r.charge),
                          n_atoms=int(r.n_atoms), xyz=r.xyz),
                     a.threads, a.conv_tol, a.grid))
    # biggest first: better load balance across workers
    jobs.sort(key=lambda j: -j[0]['n_atoms'])

    print(f'{len(jobs)} molecules | {a.workers} workers x {_T} threads '
          f'(THREADS_PER_WORKER) | B3LYP/def2-SVP conv={a.conv_tol:g} grid={a.grid}',
          flush=True)
    if str(a.threads) != str(_T):
        print(f'  note: --threads {a.threads} ignored; thread count is fixed at import '
              f'time from THREADS_PER_WORKER={_T}', flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(process_one, jobs), 1):
            print(f'{i:3d}/{len(jobs)} {msg}', flush=True)
    print(f'DONE in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
