"""SHAP attribution for the MD core descriptor block (manuscript correction A2).

Why this replaces the original analysis. The published SHAP figure was computed on the
principal components of the full 272-descriptor matrix, and PC1 (mean |SHAP| 0.807)
dominated it. PC1 loads on the AlphaFold pLDDT quantiles, which are constant within a
protein, so that component largely encodes which target a row belongs to rather than any
property of the ligand or the complex. Attributing it to "protein structure quality" is
not supportable: indicator variables for target identity alone reach a cross-validated
R2 of 0.34.

This analysis instead attributes the 89-descriptor MD core, the block that the ablation
identifies as carrying the predictive signal (test R2 0.795 structure-disjoint, against
0.627 for the full 272). Attribution is computed on the raw descriptors rather than on
principal components, so each value belongs to a named physical quantity and can be read
directly. Components are a modelling convenience; they are not what a reader wants
attributed.

Output: results/shap_md_core.csv and figures/FigB8_shap_md_core.*
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

import datasets
import vizstyle as vs
from hybrid_pipeline import fit_nusvr, make_splits

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')

# Feature families, assigned from the exact column names of the 89-descriptor core so
# that nothing lands in a catch-all. The five families and their sizes match the blocks
# described in the manuscript's feature-engineering section.
AMINO = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
         'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
STABILITY = {'PL RMSD', 'PL RMSF', 'L RMSD', 'L RMSF', 'Docking score', 'SASA'}
ENERGETICS = {'Total energy', 'Potential energy', 'Kinetic energy', 'Coulombic energy',
              'Excluded volume interactions', 'Energy with forces', 'SS charges_atoms'}
LIGAND_PCA = {'Explained-variance-ratio-pc1', 'Explained-variance-ratio-pc2',
              'Explained-variance-ratio-pc3', 'PCA-components-shape1',
              'PCA-components-shape2', 'Average-structure-shape'}
INTERACTION_SUFFIX = ('_angle', '_dihedral', '_elec', '_stretch', '_vdw')

FAMILIES = ['Residue interaction forces', 'Interaction geometry and energy',
            'Ligand conformational dynamics', 'System energetics', 'MD stability']


def family(col: str) -> str:
    c = col.strip()
    if c.upper() in AMINO:
        return 'Residue interaction forces'
    if c in STABILITY:
        return 'MD stability'
    if c in ENERGETICS:
        return 'System energetics'
    if c in LIGAND_PCA:
        return 'Ligand conformational dynamics'
    if c.lower().endswith(INTERACTION_SUFFIX):
        return 'Interaction geometry and energy'
    raise ValueError(f'unclassified descriptor {col!r}; update the family map')


def main():
    import shap

    X, y, meta = datasets.load('md_core89')
    groups = datasets.groups_for(meta)
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    names = list(X.columns)
    sp = make_splits(len(yv), groups=groups, regime='structure_disjoint',
                     random_state=1)
    print(f'{meta["label"]}: {Xv.shape[0]} molecules x {Xv.shape[1]} descriptors')

    est, pick, _ = fit_nusvr(Xv, yv, sp, selection='cv_r2', cv=5, verbose=False)
    print(f'Nu-SVR {pick["params"]}, {pick["n_components"]} PCs, '
          f'CV R2 {pick["r2_cv"]:.4f}, test R2 {pick["r2_test"]:.4f}')

    # Attribute on the training molecules, with a compact k-means background so the
    # kernel explainer stays tractable on 91 descriptors.
    bg = shap.kmeans(Xv[sp['train']], 25)
    expl = shap.KernelExplainer(est.predict, bg)
    idx = np.concatenate([sp['test'], sp['holdout']])
    print(f'explaining {len(idx)} held-out molecules ...', flush=True)
    sv = expl.shap_values(Xv[idx], nsamples=400, silent=True)
    sv = np.asarray(sv)

    mean_abs = np.abs(sv).mean(0)
    df = pd.DataFrame({'descriptor': names,
                       'family': [family(n) for n in names],
                       'mean_abs_shap': mean_abs,
                       'mean_shap': sv.mean(0)})
    df = df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'shap_md_core.csv'), index=False)

    fam = (df.groupby('family')['mean_abs_shap'].agg(['sum', 'size'])
             .sort_values('sum', ascending=False))
    fam['share_%'] = (100 * fam['sum'] / fam['sum'].sum()).round(1)
    fam.to_csv(os.path.join(RES, 'shap_md_core_families.csv'))

    print('\n--- top 15 descriptors by mean |SHAP| ---')
    print(df.head(15).to_string(index=False))
    print('\n--- attribution by feature family ---')
    print(fam.round(4).to_string())

    # ---------------------------------------------------------------- figure
    import matplotlib.pyplot as plt
    vs.apply_style()
    top = df.head(18).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0),
                             gridspec_kw={'width_ratios': [1.55, 1]})

    ax = axes[0]
    ax.barh(np.arange(len(top)), top['mean_abs_shap'], 0.72,
            color=vs.BLUE, edgecolor=vs.SURFACE, linewidth=0.8, zorder=3)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top['descriptor'], fontsize=6.5)
    ax.set_xlabel('Mean |SHAP| (log units of p$IC_{50}$)')
    ax.set_title('Individual descriptors', loc='left')
    ax.grid(True, axis='x')
    ax.grid(False, axis='y')

    ax = axes[1]
    f = fam.iloc[::-1]
    ax.barh(np.arange(len(f)), f['sum'], 0.66, color=vs.AQUA,
            edgecolor=vs.SURFACE, linewidth=0.8, zorder=3)
    for i, (v, s) in enumerate(zip(f['sum'], f['share_%'])):
        ax.annotate(f'{s:.0f}%', (v, i), xytext=(3, 0), textcoords='offset points',
                    va='center', fontsize=6.5, color=vs.INK2)
    ax.set_yticks(np.arange(len(f)))
    ax.set_yticklabels([t.replace(' and ', ' &\n') for t in f.index], fontsize=6.5)
    ax.set_xlabel('Summed mean |SHAP|')
    ax.set_title('Feature family', loc='left')
    ax.grid(True, axis='x')
    ax.grid(False, axis='y')
    ax.set_xlim(0, f['sum'].max() * 1.22)

    vs.panel_tag(axes[0], 'A', dx=-0.42)
    vs.panel_tag(axes[1], 'B', dx=-0.55)
    fig.suptitle('SHAP attribution of the 89-descriptor molecular dynamics core',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    vs.save(fig, 'FigB8_shap_md_core')


if __name__ == '__main__':
    main()
