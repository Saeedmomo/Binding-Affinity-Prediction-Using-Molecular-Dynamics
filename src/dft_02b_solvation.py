"""Recover the GFN2-xTB ALPB(water) solvation descriptors (runs in WSL).

The solvation terms were not captured during the main run: the output parser in
dft_02_run_qc.py matched lines beginning "-> Gsolv", but xtb indents that block. Rather
than repeat four CPU-hours of DFT, this pass re-runs only the xtb single point, starting
from the optimised geometry that the main run already stored in each JSON under
'_opt_xyz'. It costs a few seconds per molecule.

The parser here is deliberately format-tolerant: it takes the last float on any line
mentioning the term, and reports which molecules yielded nothing.

Usage:
  wsl -d Ubuntu-24.04 -- bash -lc '$HOME/.local/bin/micromamba run -r $HOME/micromamba \
      -n qc python3 "/mnt/d/.../src/dft_02b_solvation.py"'
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import tempfile

ROOT = '/mnt/d/Chua files/fourth paper/benchmark_work'
QCD = os.path.join(ROOT, 'data', 'dft', 'qc_raw')

TERMS = {
    'xtb_Gsolv_water_Eh': 'gsolv',
    'xtb_Gsasa_Eh': 'gsasa',
    'xtb_Ghb_Eh': 'ghb',
    'xtb_Gelec_Eh': 'gelec',
    'xtb_Gshift_Eh': 'gshift',
}
FLOAT = re.compile(r'[-+]?\d+\.\d+(?:[eEdD][-+]?\d+)?')


def parse(out: str) -> dict:
    """xtb prints the solvation breakdown as indented '-> Gxxx  <value> Eh' rows.

    The term name must be matched on those rows only. A plain substring search also hits
    the summary row 'total w/o Gsasa and Gshift', which is a total energy, not Gsasa --
    that mistake put -116.24 Eh into the Gsasa field on the first pass.
    """
    got = {}
    for line in out.splitlines():
        low = line.lower()
        nums = FLOAT.findall(line)
        if not nums:
            continue
        if '->' in line and 'w/o' not in low:
            for key, tok in TERMS.items():
                if key in got:
                    continue
                # the term must be the token immediately after the arrow
                after = low.split('->', 1)[1].strip().split()
                if after and after[0] == tok:
                    got[key] = float(nums[-1].replace('D', 'E').replace('d', 'e'))
        elif 'total energy' in low and 'xtb_total_energy_alpb_Eh' not in got:
            got['xtb_total_energy_alpb_Eh'] = float(nums[0])
    return got


def main():
    files = sorted(glob.glob(os.path.join(QCD, '*.json')))
    print(f'{len(files)} records', flush=True)
    done = miss = skipped = 0
    for f in files:
        rec = json.load(open(f))
        if rec.get('status') != 'ok' or '_opt_xyz' not in rec:
            skipped += 1
            continue
        # a Gsasa near the total energy is the signature of the first-pass parser bug
        stale = ('xtb_Gsasa_Eh' in rec and abs(rec['xtb_Gsasa_Eh']) > 1.0)
        if 'xtb_Gsolv_water_Eh' in rec and not stale:
            skipped += 1
            continue
        with tempfile.TemporaryDirectory() as wd:
            g = os.path.join(wd, 'g.xyz')
            open(g, 'w').write(rec['_opt_xyz'])
            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            r = subprocess.run(['xtb', 'g.xyz', '--chrg', str(rec['charge']),
                                '--uhf', '0', '--gfn', '2', '--alpb', 'water'],
                               cwd=wd, capture_output=True, text=True,
                               timeout=1800, env=env)
            got = parse(r.stdout)
        if got:
            rec.update(got)
            if 'xtb_total_energy_Eh' in rec and 'xtb_Gsolv_water_Eh' in rec:
                rec['xtb_logP_proxy'] = -rec['xtb_Gsolv_water_Eh']
            tmp = f + '.tmp'
            json.dump(rec, open(tmp, 'w'), indent=1)
            os.replace(tmp, f)
            done += 1
            print(f"  [ok  ] {rec['mol_id']:9s} "
                  + ' '.join(f'{k.split("_")[1]}={v:+.5f}' for k, v in got.items()),
                  flush=True)
        else:
            miss += 1
            print(f"  [MISS] {rec['mol_id']:9s} no solvation terms parsed", flush=True)
    print(f'\nupdated {done}, no-parse {miss}, skipped {skipped}', flush=True)


if __name__ == '__main__':
    main()
