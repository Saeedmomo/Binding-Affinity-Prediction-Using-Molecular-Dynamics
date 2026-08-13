"""Assemble the public repository contents. Stages files and commits; never pushes.

The push is deliberately left to the author so that the published history carries only
their authorship.
"""
from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, 'github_repo')
CHUA = r'D:\Chua files'

# (source, destination inside the repo). Directories are copied wholesale.
TREES = [
    (os.path.join(ROOT, 'src'), 'src'),
    (os.path.join(ROOT, 'docs'), 'docs'),
    (os.path.join(ROOT, 'results'), 'results'),
    (os.path.join(ROOT, 'tables'), 'tables'),
    (os.path.join(ROOT, 'data', 'dft', 'xyz'), 'data/dft/xyz'),
    (os.path.join(ROOT, 'data', 'dft', 'qc_raw'), 'data/dft/qc_raw'),
]
FILES = [
    (os.path.join(ROOT, 'data', 'ligands_122_smiles.csv'), 'data/ligands_122_smiles.csv'),
    (os.path.join(ROOT, 'data', 'ligands_122_annotated.csv'), 'data/ligands_122_annotated.csv'),
    (os.path.join(ROOT, 'data', 'ceiling.json'), 'data/ceiling.json'),
    (os.path.join(ROOT, 'data', 'dft', 'manifest.csv'), 'data/dft/manifest.csv'),
    (os.path.join(ROOT, 'data', 'dft', 'quantum_descriptors.csv'), 'data/dft/quantum_descriptors.csv'),
    (os.path.join(ROOT, 'data', 'dft', 'quantum_summary.csv'), 'data/dft/quantum_summary.csv'),
    (os.path.join(ROOT, 'data', 'dft', 'qc_provenance.json'), 'data/dft/qc_provenance.json'),
    # the study's input descriptor matrices
    (os.path.join(CHUA, 'MDS_analysis', 'data_model', '2.csv'), 'data/inputs/md272.csv'),
    (os.path.join(CHUA, 'MDS_analysis', 'data_model', '2_cleaned.csv'), 'data/inputs/md272_outliers_removed.csv'),
    (os.path.join(CHUA, 'fourth paper', 'JDes_output3.csv'), 'data/inputs/padel_1444.csv'),
    (os.path.join(CHUA, 'MDS_analysis', 'data_model', '5_imputed.csv'), 'data/inputs/md_padel_fused_1549.csv'),
]
# too large to ship raw; compresses well because 44 % of its columns are all zero
GZIP = [(os.path.join(CHUA, 'Said_mol2_files', 'merged_descriptors_with_pic50.csv'),
         'data/inputs/pose3d_112194.csv.gz')]

FIG_EXT = ('.png', '.pdf')          # TIFF is for journal submission, not the repository
SKIP_DIRS = {'__pycache__', '_pipeline_cache', '.ipynb_checkpoints'}


def clean_repo():
    for name in os.listdir(REPO):
        if name == '.git':
            continue
        p = os.path.join(REPO, name)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)


def copy_tree(src, dst_rel):
    dst = os.path.join(REPO, dst_rel.replace('/', os.sep))
    if not os.path.isdir(src):
        print(f'  [skip] {dst_rel} (source missing)')
        return 0
    n = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel = os.path.relpath(dirpath, src)
        out = dst if rel == '.' else os.path.join(dst, rel)
        os.makedirs(out, exist_ok=True)
        for fn in filenames:
            if fn.endswith(('.pyc', '.tmp')):
                continue
            shutil.copy2(os.path.join(dirpath, fn), os.path.join(out, fn))
            n += 1
    print(f'  {dst_rel:24s} {n:4d} files')
    return n


def main():
    if not os.path.isdir(os.path.join(REPO, '.git')):
        sys.exit(f'{REPO} is not a git clone; clone the repository there first')

    print('clearing tracked content (preserving .git)')
    clean_repo()

    print('copying trees')
    total = sum(copy_tree(s, d) for s, d in TREES)

    print('copying figures (png, pdf)')
    os.makedirs(os.path.join(REPO, 'figures'), exist_ok=True)
    nf = 0
    for fn in sorted(os.listdir(os.path.join(ROOT, 'figures'))):
        if fn.lower().endswith(FIG_EXT):
            shutil.copy2(os.path.join(ROOT, 'figures', fn),
                         os.path.join(REPO, 'figures', fn))
            nf += 1
    print(f'  figures                  {nf:4d} files')

    print('copying individual files')
    nfl = 0
    for src, rel in FILES:
        if not os.path.exists(src):
            print(f'  [skip] {rel} (missing)')
            continue
        dst = os.path.join(REPO, rel.replace('/', os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        nfl += 1
    print(f'  inputs and metadata      {nfl:4d} files')

    for src, rel in GZIP:
        if not os.path.exists(src):
            continue
        dst = os.path.join(REPO, rel.replace('/', os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, 'rb') as fi, gzip.open(dst, 'wb', compresslevel=9) as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
        print(f'  {rel:24s} {os.path.getsize(src)/1e6:6.1f} MB -> '
              f'{os.path.getsize(dst)/1e6:.1f} MB')

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, dn, fs in os.walk(REPO) if '.git' not in dp
               for f in fs)
    print(f'\nstaged payload: {size/1e6:.1f} MB')


if __name__ == '__main__':
    main()
