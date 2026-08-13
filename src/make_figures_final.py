"""Consolidated monochrome figures for journal submission.

Six main-text figures, each multi-panel, all black and white at 600 dpi. Captions are
written in the manuscript builder, not here. Everything that does not earn a main-text
slot goes to the supplementary set at the bottom of this file.

Figure 1  the feature-extraction pipeline, the methodological contribution
Figure 2  benchmark performance across descriptor sets, both partitioning regimes
Figure 3  feature-block ablation and the ceiling imposed by repeated structures
Figure 4  predicted against observed potency for held-out molecules
Figure 5  SHAP attribution of the molecular dynamics core
Figure 6  quantum-chemical descriptors and statistical validation
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy import stats

import vizstyle_bw as vs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
CEILING = 0.8875

SHORT = {
    'md272': 'All blocks\n(272)', 'md_core89': 'MD core\n(89)',
    'padel': 'PaDEL\n(1444)', 'fused': 'MD and PaDEL\n(1549)',
    'mol2desc': 'Pose 3D\n(62571)', 'dft': 'Quantum\n(109)',
    'dft_plus_md': 'Quantum and MD\n(381)', 'maccs167': 'MACCS\n(167)',
    'alphafold16': 'AlphaFold\n(16)', 'md272_clean': 'All blocks,\noutliers removed',
}
MAIN = ['md_core89', 'md272', 'fused', 'padel', 'mol2desc', 'dft_plus_md', 'dft']
ABL = ['md272', 'md_core89', 'maccs167', 'alphafold16']
REG = {'random': 'Random partition', 'structure_disjoint': 'Structure-disjoint partition'}
FLOOR = -1.0


def metrics():
    return pd.read_csv(os.path.join(RES, 'metrics_all.csv'))


# ------------------------------------------------------------------ Figure 1
def fig1_pipeline():
    """Schematic of the descriptor-generation pipeline, strictly left to right."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axis('off')
    ax.set_xlim(0, 10.4)
    ax.set_ylim(-0.9, 6.6)

    def box(x, y, w, h, text, fill='#ffffff', fs=6.9, weight='normal', lw=0.9):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=vs.INK, lw=lw,
                               zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
                color='#ffffff' if fill == '#3d3d3d' else vs.INK,
                fontweight=weight, linespacing=1.35, zorder=4)

    def arrow(x1, y1, x2, y2, lw=0.9):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=8, lw=lw, color=vs.INK,
                                     shrinkA=0, shrinkB=0, zorder=5))

    # column 1: inputs
    box(0.05, 4.55, 1.85, 1.05, 'Protein-ligand\ncomplex\n(4 targets)', lw=1.6)
    box(0.05, 2.95, 1.85, 1.05, '50 ns molecular\ndynamics\n(Desmond, NPT)', lw=1.6)
    arrow(0.98, 4.55, 0.98, 4.00)
    box(0.05, 0.55, 1.85, 0.95, 'Ligand\n2D structure')

    # column 2: descriptor blocks, simulation-derived above, conventional below
    blocks = [('Structural stability (6)', 5.30),
              ('Interaction geometry\nand energy (50)', 4.15),
              ('Residue interaction\nforces (20)', 3.00),
              ('Ligand conformational\ndynamics (6)', 1.85),
              ('AlphaFold confidence (16)', 0.85)]
    for txt, y in blocks:
        box(2.55, y, 2.5, 0.85, txt, lw=1.6)
        arrow(1.90, 3.48, 2.55, y + 0.42, lw=0.7)
    box(2.55, -0.35, 2.5, 0.75, 'MACCS fingerprints (167)', '#9e9e9e')
    arrow(1.90, 1.02, 2.55, 0.02, lw=0.7)

    ax.text(3.80, 6.35, 'Descriptors generated from simulation', ha='center',
            fontsize=7.2, fontweight='bold', color=vs.INK)

    # column 3: preprocessing, fed by every block
    box(5.75, 2.55, 1.75, 1.05, 'Standardise,\nPCA to 95%\nvariance')
    for _, y in blocks:
        ax.plot([5.05, 5.75], [y + 0.42, 3.08], lw=0.5, color=vs.MUTED, zorder=1)
    ax.plot([5.05, 5.75], [0.02, 3.08], lw=0.5, color=vs.MUTED, zorder=1)

    # column 4: models and output
    box(8.15, 3.55, 2.1, 1.05, 'Nu-SVR, DNN,\nRidge stack')
    box(8.15, 1.55, 2.1, 1.05, 'Predicted\np$IC_{50}$', '#3d3d3d', 7.6, 'bold')
    arrow(7.50, 3.15, 8.15, 4.05)
    arrow(9.20, 3.55, 9.20, 2.60)

    ax.text(5.2, -0.85,
            'Heavy outline, descriptor blocks generated in this work. '
            'Grey, conventional ligand fingerprints included for comparison.',
            ha='center', fontsize=6.2, color=vs.INK2)
    return vs.save(fig, 'Figure_1_pipeline')


# ------------------------------------------------------------------ Figure 2
def fig2_benchmark(df):
    regimes = ['random', 'structure_disjoint']
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True)
    models = ['nusvr', 'dnn', 'hybrid']

    for ax, reg in zip(axes, regimes):
        keys = [k for k in MAIN if ((df.dataset == k) & (df.split_regime == reg)).any()]
        x = np.arange(len(keys))
        w = 0.27
        for i, m in enumerate(models):
            s = df[(df.split_regime == reg) & (df.subset == 'test') & (df.model == m)]
            v = np.array([float(s[s.dataset == k]['r2'].iloc[0]) if (s.dataset == k).any()
                          else np.nan for k in keys])
            lo = np.array([float(s[s.dataset == k]['r2_ci_lo'].iloc[0])
                           if (s.dataset == k).any() else np.nan for k in keys])
            hi = np.array([float(s[s.dataset == k]['r2_ci_hi'].iloc[0])
                           if (s.dataset == k).any() else np.nan for k in keys])
            vc = np.clip(v, FLOOR, None)
            err = np.vstack([np.clip(vc - np.clip(lo, FLOOR, None), 0, None),
                             np.clip(np.clip(hi, FLOOR, None) - vc, 0, None)])
            pos = x + (i - 1) * w
            b = vs.bar(ax, pos, vc, w * 0.9, fill=vs.MODEL_FILL[m],
                       hatch=vs.MODEL_HATCH[m], label=vs.MODEL_LABELS[m])
            ax.errorbar(pos, vc, yerr=err, fmt='none', ecolor=vs.INK,
                        elinewidth=0.6, capsize=1.6, zorder=4)
            for p_, raw in zip(pos, v):
                if np.isnan(raw):
                    continue
                if raw < FLOOR:
                    ax.plot([p_], [FLOOR + 0.05], marker='v', ms=3.4,
                            mfc='#ffffff', mec=vs.INK, mew=0.8, zorder=5)
                    ax.annotate(f'{raw:.1f}', (p_, FLOOR + 0.12), ha='center',
                                va='bottom', fontsize=5.4, color=vs.INK, zorder=5)
                else:
                    ax.annotate(f'{raw:.2f}', (p_, raw + (0.02 if raw >= 0 else -0.02)),
                                ha='center', va='bottom' if raw >= 0 else 'top',
                                fontsize=5.4, color=vs.INK, zorder=5)

        ax.axhline(CEILING, color=vs.INK, ls=':', lw=1.0, zorder=1)
        ax.annotate(f'ligand-only ceiling, {CEILING:.3f}', (0.004, CEILING),
                    xycoords=('axes fraction', 'data'), ha='left', va='bottom',
                    fontsize=6.2, color=vs.INK)
        ax.axhline(0, color=vs.INK, lw=0.7, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(k, k) for k in keys])
        ax.set_ylabel('Test-set $R^2$')
        ax.set_title(REG[reg], loc='left')
        ax.set_ylim(FLOOR, 1.22)
        ax.set_xlim(-1.25, len(keys) - 0.45)
    vs.panel(axes[0], 'A', dx=-0.075)
    vs.panel(axes[1], 'B', dx=-0.075)
    axes[0].legend(loc='upper right', ncol=3, fontsize=7)
    fig.tight_layout()
    return vs.save(fig, 'Figure_2_benchmark')


# ------------------------------------------------------------------ Figure 3
def fig3_ablation_ceiling(df):
    fig = plt.figure(figsize=(7.0, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.55, wspace=0.3)

    for col, reg in enumerate(['random', 'structure_disjoint']):
        ax = fig.add_subplot(gs[0, col])
        keys = [k for k in ABL if ((df.dataset == k) & (df.split_regime == reg)).any()]
        x = np.arange(len(keys))
        w = 0.38
        for i, sub in enumerate(['test', 'holdout']):
            d = df[(df.split_regime == reg) & (df.subset == sub) & (df.model == 'nusvr')]
            v = np.array([float(d[d.dataset == k]['r2'].iloc[0]) for k in keys])
            b = vs.bar(ax, x + (i - 0.5) * w, v, w * 0.88,
                       fill=vs.G[0] if sub == 'test' else vs.G[2],
                       hatch='' if sub == 'test' else '///',
                       label=sub.capitalize())
            vs.label_bars(ax, b, fontsize=5.8)
        ax.axhline(CEILING, color=vs.INK, ls=':', lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(k, k) for k in keys], fontsize=6.5)
        ax.set_ylim(0, 1.08)
        ax.set_title(REG[reg], loc='left', fontsize=8)
        if col == 0:
            ax.set_ylabel('Nu-SVR $R^2$')
            ax.legend(loc='lower left', fontsize=6.5, ncol=2)
    vs.panel(fig.axes[0], 'A', dx=-0.16)
    vs.panel(fig.axes[1], 'B', dx=-0.12)

    ann = pd.read_csv(os.path.join(ROOT, 'data', 'ligands_122_annotated.csv'))
    g = ann.groupby('smiles')['PIC50'].agg(['size', 'min', 'max'])
    multi = g[g['size'] > 1].sort_values('max', ascending=False)

    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(multi))
    ax.hlines(y, multi['min'], multi['max'], color=vs.G[1], lw=2.6, zorder=2)
    for i, smi in enumerate(multi.index):
        sub = ann[ann.smiles == smi]
        for t, gg in sub.groupby('target'):
            ax.scatter(gg['PIC50'], [i] * len(gg), s=20,
                       marker=vs.TARGET_MARKER.get(t, 'o'),
                       facecolor=vs.TARGET_FILL.get(t, '#ffffff'),
                       edgecolor=vs.INK, linewidths=0.6, zorder=4, label=t)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel('Observed p$IC_{50}$')
    ax.set_ylabel(f'Repeated structures (n = {len(multi)})')
    ax.grid(True, axis='x')
    ax.grid(False, axis='y')
    h, lab = ax.get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for a_, b_ in zip(h, lab):
        if b_ not in seen:
            seen.add(b_)
            hh.append(a_)
            ll.append(b_)
    ax.legend(hh, ll, loc='lower right', fontsize=6, ncol=2, handletextpad=0.3)
    vs.panel(ax, 'C', dx=-0.16)

    ax = fig.add_subplot(gs[1, 1])
    yv = ann['PIC50'].to_numpy()
    pred = ann.groupby('smiles')['PIC50'].transform('mean').to_numpy()
    ax.scatter(yv, pred, s=16, marker='o', facecolor='#ffffff', edgecolor=vs.INK,
               linewidths=0.6, zorder=3)
    lim = [yv.min() - 0.3, yv.max() + 0.3]
    ax.plot(lim, lim, ls='--', lw=0.8, color=vs.INK, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal', 'box')
    ax.grid(True, axis='both')
    ax.set_xlabel('Observed p$IC_{50}$')
    ax.set_ylabel('Within-structure mean')
    ax.annotate(f'$R^2$ = {CEILING:.4f}\nRMSE = 0.450', (0.04, 0.96),
                xycoords='axes fraction', ha='left', va='top', fontsize=6.8,
                color=vs.INK)
    vs.panel(ax, 'D', dx=-0.16)
    return vs.save(fig, 'Figure_3_ablation_ceiling')


# ------------------------------------------------------------------ Figure 4
def fig4_pred_obs(df):
    keys = ['md_core89', 'md272', 'padel', 'dft']
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.2))
    for ax, k, tag in zip(axes.ravel(), keys, 'ABCD'):
        f = os.path.join(RES, 'predictions', f'{k}__structure_disjoint.csv')
        if not os.path.exists(f):
            ax.axis('off')
            continue
        d = pd.read_csv(f)
        d = d[d.subset.isin(['test', 'holdout'])]
        for t, g in d.groupby('target'):
            ax.scatter(g.y_true, g.pred_nusvr, s=22, marker=vs.TARGET_MARKER.get(t, 'o'),
                       facecolor=vs.TARGET_FILL.get(t, '#ffffff'), edgecolor=vs.INK,
                       linewidths=0.6, label=t, zorder=3)
        lim = [3.6, 10.0]
        ax.plot(lim, lim, ls='--', lw=0.8, color=vs.INK, zorder=2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal', 'box')
        ax.grid(True, axis='both')
        m = df[(df.dataset == k) & (df.split_regime == 'structure_disjoint') &
               (df.model == 'nusvr') & (df.subset == 'test')]
        if len(m):
            ax.annotate(f"$R^2$ = {m['r2'].iloc[0]:.3f}\nRMSE = {m['rmse'].iloc[0]:.3f}",
                        (0.04, 0.96), xycoords='axes fraction', ha='left', va='top',
                        fontsize=6.6, color=vs.INK)
        ax.set_title(SHORT.get(k, k).replace('\n', ' '), fontsize=8)
        ax.set_xlabel('Observed p$IC_{50}$')
        ax.set_ylabel('Predicted p$IC_{50}$')
        vs.panel(ax, tag, dx=-0.2)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02),
               fontsize=7)
    fig.tight_layout()
    return vs.save(fig, 'Figure_4_pred_vs_obs')


# ------------------------------------------------------------------ Figure 5
def fig5_shap():
    p = os.path.join(RES, 'shap_md_core.csv')
    if not os.path.exists(p):
        return []
    d = pd.read_csv(p)
    fam = pd.read_csv(os.path.join(RES, 'shap_md_core_families.csv'), index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0),
                             gridspec_kw={'width_ratios': [1.5, 1]})

    top = d.head(16).iloc[::-1]
    ax = axes[0]
    ax.barh(np.arange(len(top)), top['mean_abs_shap'], 0.72, facecolor=vs.G[1],
            edgecolor=vs.INK, linewidth=0.6, zorder=3)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([t.replace('_', ' ').replace('-', ' ')
                        for t in top['descriptor']], fontsize=6.3)
    ax.set_xlabel('Mean absolute SHAP value (log units)')
    ax.set_title('Individual descriptors', loc='left')
    ax.grid(True, axis='x')
    ax.grid(False, axis='y')
    vs.panel(ax, 'A', dx=-0.42)

    ax = axes[1]
    f = fam.sort_values('sum').iloc[:]
    hatches = ['', '///', '', 'xxx', '...']
    fills = [vs.G[0], vs.G[1], vs.G[2], vs.G[1], vs.G[0]]
    for i, (name, row) in enumerate(f.iterrows()):
        ax.barh(i, row['sum'], 0.66, facecolor=fills[i % len(fills)],
                hatch=hatches[i % len(hatches)], edgecolor=vs.INK, linewidth=0.6,
                zorder=3)
        ax.annotate(f"{row['share_%']:.0f}%", (row['sum'], i), xytext=(3, 0),
                    textcoords='offset points', va='center', fontsize=6.5,
                    color=vs.INK)
    ax.set_yticks(np.arange(len(f)))
    ax.set_yticklabels([t.replace(' and ', ' and\n') for t in f.index], fontsize=6.3)
    ax.set_xlabel('Summed mean absolute SHAP value')
    ax.set_title('Descriptor family', loc='left')
    ax.grid(True, axis='x')
    ax.grid(False, axis='y')
    ax.set_xlim(0, f['sum'].max() * 1.25)
    vs.panel(ax, 'B', dx=-0.6)
    fig.tight_layout()
    return vs.save(fig, 'Figure_5_shap')


# ------------------------------------------------------------------ Figure 6
def fig6_quantum_stats(df):
    p = os.path.join(ROOT, 'data', 'dft', 'quantum_descriptors.csv')
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4))

    if os.path.exists(p):
        d = pd.read_csv(p)
        for ax, col, xlab, tag in (
                (axes[0, 0], 'HOMO_LUMO_gap_eV', 'HOMO-LUMO gap (eV)', 'A'),
                (axes[0, 1], 'electrophilicity_eV', 'Electrophilicity index (eV)', 'B'),
                (axes[1, 0], 'dipole_total_D', 'Dipole moment (D)', 'C')):
            for t, g in d.groupby('target'):
                ax.scatter(g[col], g['PIC50'], s=18, marker=vs.TARGET_MARKER.get(t, 'o'),
                           facecolor=vs.TARGET_FILL.get(t, '#ffffff'),
                           edgecolor=vs.INK, linewidths=0.55, label=t, zorder=3)
            r, pv = stats.pearsonr(d[col], d['PIC50'])
            ax.annotate(f'$r$ = {r:.3f}\n$p$ = {pv:.3g}', (0.04, 0.97),
                        xycoords='axes fraction', ha='left', va='top', fontsize=6.4,
                        color=vs.INK)
            ax.set_xlabel(xlab)
            ax.set_ylabel('p$IC_{50}$')
            ax.grid(True, axis='both')
            vs.panel(ax, tag, dx=-0.2)

    ax = axes[1, 1]
    keys = ['md_core89', 'md272', 'padel', 'dft']
    keys = [k for k in keys if ((df.dataset == k) &
                                (df.split_regime == 'structure_disjoint')).any()]
    q2 = []
    nested = json.load(open(os.path.join(RES, 'metrics_all.json')))
    for k in keys:
        v = nested.get(f'{k}__structure_disjoint', {})
        q2.append(v.get('q2_loo_nusvr', {}).get('q2', np.nan))
    b = vs.bar(ax, np.arange(len(keys)), q2, 0.6, fill=vs.G[1])
    vs.label_bars(ax, b, fmt='{:.2f}', fontsize=6.2)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([SHORT.get(k, k) for k in keys], fontsize=6.2)
    ax.set_ylabel('Leave-one-out $Q^2$')
    ax.axhline(0, color=vs.INK, lw=0.7)
    ax.set_title('Internal validation', loc='left', fontsize=8)
    vs.panel(ax, 'D', dx=-0.2)

    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.03),
               fontsize=7)
    fig.tight_layout()
    return vs.save(fig, 'Figure_6_quantum_validation')


# ------------------------------------------------------- supplementary figures
def figS_dnn():
    p = os.path.join(RES, 'dnn_variants.csv')
    if not os.path.exists(p):
        return []
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    order = sorted(d['model'].unique())
    data = [d[d.model == m]['r2_test'].to_numpy() for m in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color=vs.INK, lw=1.2),
                    boxprops=dict(facecolor=vs.G[1], edgecolor=vs.INK, lw=0.7),
                    whiskerprops=dict(color=vs.INK, lw=0.7),
                    capprops=dict(color=vs.INK, lw=0.7),
                    flierprops=dict(marker='o', mfc='#ffffff', mec=vs.INK, ms=3))
    for i, v in enumerate(data, 1):
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).normal(0, 0.05, len(v)),
                   v, s=12, marker='o', facecolor='#ffffff', edgecolor=vs.INK,
                   linewidths=0.5, zorder=4)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([m.replace(' ', '\n') for m in order], fontsize=6.5)
    ax.axhline(0, color=vs.INK, lw=0.7)
    ax.set_ylabel('Test-set $R^2$')
    fig.tight_layout()
    return vs.save(fig, 'Figure_S1_dnn_variants')


def figS_permutation():
    nested = json.load(open(os.path.join(RES, 'metrics_all.json')))
    items = [(k, v) for k, v in nested.items()
             if v.get('permutation', {}).get('n', 0) > 0
             and k.endswith('__structure_disjoint')
             and k.split('__')[0] in MAIN]
    if not items:
        return []
    items.sort(key=lambda kv: MAIN.index(kv[0].split('__')[0]))
    n = len(items)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 1.9 * nrow), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (k, v) in zip(axes.ravel(), items):
        p = v['permutation']
        null = rng.normal(p['null_mean'], max(p['null_sd'], 1e-6), 2000)
        ax.hist(null, bins=26, facecolor=vs.G[1], edgecolor=vs.INK, linewidth=0.35,
                zorder=2)
        ax.axvline(p['observed_test_r2'], color=vs.INK, lw=1.6, zorder=4)
        ax.set_title(SHORT.get(k.split('__')[0], k).replace('\n', ' '), fontsize=7)
        ax.annotate(f"$p$ = {p['p_value']:.3f}", (0.03, 0.95),
                    xycoords='axes fraction', ha='left', va='top', fontsize=6.2)
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes.ravel()[len(items):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('Test $R^2$ under permutation', fontsize=6.8)
    fig.tight_layout()
    return vs.save(fig, 'Figure_S2_y_scrambling')


def main():
    vs.apply_style()
    df = metrics()
    print('main figures')
    fig1_pipeline()
    fig2_benchmark(df)
    fig3_ablation_ceiling(df)
    fig4_pred_obs(df)
    fig5_shap()
    fig6_quantum_stats(df)
    print('supplementary figures')
    figS_dnn()
    figS_permutation()


if __name__ == '__main__':
    main()
