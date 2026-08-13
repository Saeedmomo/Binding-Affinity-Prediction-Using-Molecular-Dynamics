"""Publication figures for the benchmark study.

Every figure has a companion table in tables/ (the relief rule for the aqua slot,
see vizstyle) and every bar carries a direct value label. No dual axes anywhere.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import vizstyle as vs
from vizstyle import (AQUA, BLUE, GRID, INK, INK2, MODEL_COLORS, MODEL_LABELS, MUTED,
                      ORANGE, REF, TARGET_COLORS, TARGET_MARKERS, VIOLET)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
TAB = os.path.join(ROOT, 'tables')

# published reference points (docs/GROUND_TRUTH.md section 4)
PUB_HYBRID_TEST, PUB_HYBRID_HOLD = 0.6560, 0.6680
PUB_SVR_TEST = 0.6532
CEILING_R2 = 0.8875

SHORT = {
    'md272': 'MD+AF+\nMACCS\n(272)',
    'md272_clean': 'MD+AF+\nMACCS, no\noutliers (272)',
    'padel': 'PaDEL\n2D/3D\n(1444)',
    'fused': 'MD $\\oplus$\nPaDEL\n(1549)',
    'mol2desc': 'Pose 3D\n(112194)',
    'dft': 'DFT\nquantum\n(109)',
    'dft_plus_md': 'DFT+MD\n(381)',
    'md_core89': 'MD core\n(89)',
    'maccs167': 'MACCS\nonly (167)',
    'alphafold16': 'AlphaFold\nonly (16)',
}
REGIME_LABEL = {'random': 'Random split (as published)',
                'structure_disjoint': 'Structure-disjoint split'}
ORDER = ['md272', 'md272_clean', 'md_core89', 'padel', 'fused', 'mol2desc',
         'dft', 'dft_plus_md', 'maccs167', 'alphafold16']
# Fig B1 compares whole descriptor sets. The three single-block variants carved out of
# md272 belong to the ablation figure instead; showing all ten here collides the tick
# labels and buries the comparison the figure exists to make.
ORDER_MAIN = ['md272', 'md_core89', 'padel', 'fused', 'mol2desc', 'dft', 'dft_plus_md']


def load_metrics():
    df = pd.read_csv(os.path.join(RES, 'metrics_all.csv'))
    df['order'] = df['dataset'].map({k: i for i, k in enumerate(ORDER)}).fillna(99)
    return df.sort_values(['order', 'split_regime', 'model'])


def load_nested():
    p = os.path.join(RES, 'metrics_all.json')
    return json.load(open(p)) if os.path.exists(p) else {}


def _present(df, regime, subset='test', order=None):
    d = df[(df.split_regime == regime) & (df.subset == subset)]
    return [k for k in (order or ORDER) if k in set(d.dataset)]


# --------------------------------------------------------------- Fig 1: benchmarks
FLOOR = -1.0   # axis floor; anything below is drawn clipped and labelled with its value


def fig_benchmark(df):
    regimes = [r for r in ('random', 'structure_disjoint') if r in set(df.split_regime)]
    fig, axes = plt.subplots(len(regimes), 1, figsize=(7.2, 3.6 * len(regimes)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    models = ['nusvr', 'dnn', 'hybrid']

    for ax, regime in zip(axes, regimes):
        keys = _present(df, regime, order=ORDER_MAIN)
        x = np.arange(len(keys))
        w = 0.26
        for i, m in enumerate(models):
            sub = df[(df.split_regime == regime) & (df.subset == 'test') & (df.model == m)]

            def col(c):
                return np.array([float(sub[sub.dataset == k][c].iloc[0])
                                 if (sub.dataset == k).any() else np.nan for k in keys],
                                float)
            v, lo, hi = col('r2'), col('r2_ci_lo'), col('r2_ci_hi')
            vclip = np.clip(v, FLOOR, None)
            loc_, hic = np.clip(lo, FLOOR, None), np.clip(hi, FLOOR, None)
            err = np.vstack([np.clip(vclip - loc_, 0, None),
                             np.clip(hic - vclip, 0, None)])
            pos = x + (i - 1) * w
            bars = ax.bar(pos, vclip, w * 0.92, label=MODEL_LABELS[m],
                          color=MODEL_COLORS[m], edgecolor=vs.SURFACE, linewidth=1.0,
                          zorder=3)
            ax.errorbar(pos, vclip, yerr=err, fmt='none', ecolor=INK2,
                        elinewidth=0.7, capsize=1.8, zorder=4)

            for j, (p_, raw) in enumerate(zip(pos, v)):
                if np.isnan(raw):
                    continue
                if raw < FLOOR:
                    # off-scale: mark the break and print the true value inside the bar
                    ax.plot([p_], [FLOOR + 0.045], marker='v', ms=4,
                            color=vs.SURFACE, mec=MODEL_COLORS[m], mew=0.9, zorder=5)
                    ax.annotate(f'{raw:.1f}', (p_, FLOOR + 0.11), ha='center',
                                va='bottom', fontsize=5.6, color=INK2, zorder=5)
                else:
                    ax.annotate(f'{raw:.2f}',
                                (p_, raw + (0.02 if raw >= 0 else -0.02)),
                                ha='center', va='bottom' if raw >= 0 else 'top',
                                fontsize=5.6, color=INK2, zorder=5)

        # reference lines: labels parked at the left margin, clear of the bars
        for yv, lab, ls in ((CEILING_R2, f'ceiling {CEILING_R2:.3f}', ':'),
                            (PUB_HYBRID_TEST, f'published {PUB_HYBRID_TEST:.3f}', '--')):
            ax.axhline(yv, color=INK2, ls=ls, lw=1.0, zorder=1)
            ax.annotate(lab, (0.004, yv), xycoords=('axes fraction', 'data'),
                        ha='left', va='bottom', fontsize=6.2, color=INK2, zorder=6)

        ax.axhline(0, color=vs.BASELINE, lw=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(k, k) for k in keys])
        ax.set_ylabel('Test-set $R^2$')
        ax.set_title(REGIME_LABEL[regime], loc='left')
        ax.set_ylim(FLOOR, 1.30)
        # left gutter wide enough that the reference-line labels never sit on a bar
        ax.set_xlim(-1.35, len(keys) - 0.45)

    axes[0].legend(loc='upper right', ncol=1, fontsize=6.8)
    fig.suptitle('Predictive performance by descriptor set, with bootstrap 95% CI',
                 fontsize=10, fontweight='bold', y=0.998)
    fig.text(0.5, -0.005,
             f'Dotted line, ligand-only ceiling ($R^2$ = {CEILING_R2:.3f}); dashed line, '
             f'the hybrid test $R^2$ reported in the original analysis '
             f'({PUB_HYBRID_TEST:.3f}). Bars below the axis floor are clipped and the '
             f'marker gives the true value. Test sets hold 23–25 molecules, so the '
             f'intervals are wide.',
             ha='center', fontsize=6.2, color=INK2)
    fig.tight_layout()
    return vs.save(fig, 'FigB1_benchmark_test_r2')


# ---------------------------------------------------- Fig 2: predicted vs observed
def fig_pred_obs(regime='random'):
    files = sorted(glob.glob(os.path.join(RES, 'predictions', f'*__{regime}.csv')))
    keys = [k for k in ORDER
            if any(os.path.basename(f).startswith(k + '__') for f in files)]
    if not keys:
        return []
    n = len(keys)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.65 * nrow),
                             squeeze=False)
    metr = load_metrics()

    for ax, k in zip(axes.ravel(), keys):
        d = pd.read_csv(os.path.join(RES, 'predictions', f'{k}__{regime}.csv'))
        d = d[d.subset.isin(['test', 'holdout'])]
        for tgt, g in d.groupby('target'):
            ax.scatter(g.y_true, g.pred_hybrid, s=26,
                       c=TARGET_COLORS.get(tgt, MUTED),
                       marker=TARGET_MARKERS.get(tgt, 'o'),
                       edgecolors=vs.SURFACE, linewidths=0.7, alpha=0.95,
                       label=tgt, zorder=3)
        lim = [min(d.y_true.min(), d.pred_hybrid.min()) - 0.3,
               max(d.y_true.max(), d.pred_hybrid.max()) + 0.3]
        ax.plot(lim, lim, ls='--', lw=0.9, color=INK2, zorder=2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal', 'box')
        ax.grid(True, axis='both')
        m = metr[(metr.dataset == k) & (metr.split_regime == regime) &
                 (metr.model == 'hybrid') & (metr.subset == 'test')]
        txt = ''
        if len(m):
            txt = (f"test $R^2$ = {m['r2'].iloc[0]:.3f}\n"
                   f"RMSE = {m['rmse'].iloc[0]:.3f}")
        ax.annotate(txt, (0.04, 0.96), xycoords='axes fraction', ha='left', va='top',
                    fontsize=6.8, color=INK2)
        ax.set_title(SHORT.get(k, k).replace('\n', ' '), fontsize=8)

    for ax in axes.ravel()[len(keys):]:
        ax.axis('off')
    # the grid is ragged whenever the panel count is not a multiple of ncol, so label
    # every panel rather than only the bottom row and left column
    for ax in axes.ravel()[:len(keys)]:
        ax.set_xlabel('Observed p$IC_{50}$')
        ax.set_ylabel('Predicted p$IC_{50}$')

    h, lab = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.015),
               title='Target protein', title_fontsize=7.5)
    fig.suptitle(f'Hybrid-model predictions, held-out molecules '
                 f'({REGIME_LABEL[regime].lower()})',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    return vs.save(fig, f'FigB2_pred_vs_obs_{regime}')


# ------------------------------------------------------------- Fig 3: ablation
def fig_ablation(df):
    keys = [k for k in ('md272', 'md_core89', 'maccs167', 'alphafold16')
            if k in set(df.dataset)]
    if len(keys) < 2:
        return []
    regimes = [r for r in ('random', 'structure_disjoint') if r in set(df.split_regime)]
    fig, axes = plt.subplots(1, len(regimes), figsize=(3.7 * len(regimes), 3.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    x = np.arange(len(keys))
    w = 0.38
    for ax, regime in zip(axes, regimes):
        for i, sub in enumerate(('test', 'holdout')):
            d = df[(df.split_regime == regime) & (df.subset == sub) &
                   (df.model == 'nusvr')]
            v = np.array([float(d[d.dataset == k]['r2'].iloc[0])
                          if (d.dataset == k).any() else np.nan for k in keys], float)
            bars = ax.bar(x + (i - 0.5) * w, v, w * 0.9,
                          color=(BLUE if sub == 'test' else AQUA),
                          label=sub.capitalize(), edgecolor=vs.SURFACE,
                          linewidth=1.0, zorder=3)
            vs.bar_labels(ax, bars, fmt='{:.2f}', dy=0.012, fontsize=6.2)
        ax.axhline(CEILING_R2, color=INK2, ls=':', lw=1.0, zorder=1)
        ax.annotate(f'ceiling {CEILING_R2:.3f}', (0.02, CEILING_R2), va='bottom',
                    xycoords=('axes fraction', 'data'), fontsize=6.2, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(k, k) for k in keys])
        ax.set_title(REGIME_LABEL[regime], loc='left', fontsize=8.5)
        ax.axhline(0, color=vs.BASELINE, lw=0.8)
        ax.set_ylim(0, 1.06)
    axes[0].set_ylabel('Nu-SVR $R^2$')
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h[:2], lab[:2], loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.035), fontsize=7.5)
    fig.suptitle('Feature-block ablation: the 89-descriptor MD core carries the signal',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    return vs.save(fig, 'FigB3_feature_block_ablation')


# --------------------------------------------------- Fig 4: ligand-only ceiling
def fig_ceiling():
    ann = pd.read_csv(os.path.join(ROOT, 'data', 'ligands_122_annotated.csv'))
    g = ann.groupby('smiles')['PIC50'].agg(['size', 'min', 'max', 'mean'])
    multi = g[g['size'] > 1].sort_values('max', ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2),
                             gridspec_kw={'width_ratios': [1.35, 1]})

    ax = axes[0]
    y = np.arange(len(multi))
    ax.hlines(y, multi['min'], multi['max'], color=GRID, lw=3.2, zorder=2)
    for i, (smi, row) in enumerate(multi.iterrows()):
        sub = ann[ann.smiles == smi]
        for t, gg in sub.groupby('target'):
            ax.scatter(gg['PIC50'], [i] * len(gg), s=30,
                       c=TARGET_COLORS.get(t, MUTED),
                       marker=TARGET_MARKERS.get(t, 'o'),
                       edgecolors=vs.SURFACE, linewidths=0.7, zorder=4, label=t)
    ax.set_yticks(y)
    ax.set_yticklabels([f'S{i + 1}' for i in y], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel('Observed p$IC_{50}$')
    ax.set_ylabel(f'Repeated structure ({len(multi)} groups)')
    ax.grid(True, axis='x')
    ax.grid(False, axis='y')
    ax.set_title('Identical ligands, different targets', loc='left')
    h, lab = ax.get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for a, b in zip(h, lab):
        if b not in seen:
            seen.add(b)
            hh.append(a)
            ll.append(b)
    ax.legend(hh, ll, loc='lower right', ncol=2, fontsize=6.5, title='Target',
              title_fontsize=6.5)

    ax = axes[1]
    yv = ann['PIC50'].to_numpy()
    pred = ann.groupby('smiles')['PIC50'].transform('mean').to_numpy()
    ax.scatter(yv, pred, s=24, c=BLUE, edgecolors=vs.SURFACE, linewidths=0.7, zorder=3)
    lim = [yv.min() - 0.3, yv.max() + 0.3]
    ax.plot(lim, lim, ls='--', lw=0.9, color=INK2, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal', 'box')
    ax.grid(True, axis='both')
    ax.set_xlabel('Observed p$IC_{50}$')
    ax.set_ylabel('Best ligand-only prediction\n(within-structure mean)')
    ax.set_title('Theoretical best', loc='left')
    ax.annotate(f'$R^2$ = {CEILING_R2:.4f}\nRMSE = 0.450\n'
                f'{ann["smiles"].nunique()} structures / {len(ann)} rows',
                (0.04, 0.96), xycoords='axes fraction', ha='left', va='top',
                fontsize=7, color=INK2)
    vs.panel_tag(axes[0], 'A')
    vs.panel_tag(axes[1], 'B')
    fig.suptitle('Upper bound on any purely ligand-based descriptor set',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    return vs.save(fig, 'FigB4_ligand_only_ceiling')


# ------------------------------------------------------- Fig 5: permutation test
def fig_permutation(nested):
    items = [(k, v) for k, v in nested.items()
             if v.get('permutation', {}).get('n', 0) > 0 and k.endswith('__random')]
    if not items:
        return []
    items.sort(key=lambda kv: ORDER.index(kv[0].split('__')[0])
               if kv[0].split('__')[0] in ORDER else 99)
    n = len(items)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.1 * nrow), squeeze=False)
    for ax, (k, v) in zip(axes.ravel(), items):
        p = v['permutation']
        null = np.random.default_rng(0).normal(p['null_mean'], max(p['null_sd'], 1e-6), 2000)
        ax.hist(null, bins=28, color=GRID, edgecolor=vs.SURFACE, linewidth=0.4, zorder=2)
        ax.axvline(p['observed_test_r2'], color=ORANGE, lw=2.0, zorder=4)
        ax.axvline(p['null_q95'], color=INK2, lw=1.0, ls=':', zorder=3)
        ax.set_title(SHORT.get(k.split('__')[0], k).replace('\n', ' '), fontsize=8)
        ax.annotate(f"observed {p['observed_test_r2']:.3f}\n"
                    f"null {p['null_mean']:.3f}$\\pm${p['null_sd']:.3f}\n"
                    f"$p$ = {p['p_value']:.3f}",
                    (0.03, 0.96), xycoords='axes fraction', ha='left', va='top',
                    fontsize=6.5, color=INK2)
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes.ravel()[len(items):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('Test $R^2$ under label permutation')
    fig.suptitle('Y-scrambling: observed performance against the permutation null '
                 '(Nu-SVR, 100 shuffles)', fontsize=9.5, fontweight='bold')
    fig.tight_layout()
    return vs.save(fig, 'FigB5_y_scrambling')


# ------------------------------------------------------------ Fig 6: DFT overview
def fig_dft():
    p = os.path.join(ROOT, 'data', 'dft', 'quantum_descriptors.csv')
    if not os.path.exists(p):
        print('  (skipping FigB6: quantum descriptors not generated yet)')
        return []
    d = pd.read_csv(p)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))

    ax = axes[0, 0]
    for col, c, lab in ((('E_HOMO_eV'), BLUE, 'HOMO'), ('E_LUMO_eV', ORANGE, 'LUMO')):
        ax.hist(d[col], bins=22, color=c, alpha=0.85, edgecolor=vs.SURFACE,
                linewidth=0.4, label=lab, zorder=3)
    ax.set_xlabel('Orbital energy (eV)')
    ax.set_ylabel('Ligands')
    ax.legend(loc='upper left', ncol=1, fontsize=7)
    ax.set_title('Frontier orbitals', loc='left')

    panels = [
        (axes[0, 1], 'HOMO_LUMO_gap_eV', 'HOMO–LUMO gap (eV)'),
        (axes[0, 2], 'dipole_total_D', 'Dipole moment (D)'),
        (axes[1, 0], 'electrophilicity_eV', 'Electrophilicity $\\omega$ (eV)'),
        (axes[1, 1], 'hardness_eV', 'Chemical hardness $\\eta$ (eV)'),
        (axes[1, 2], 'q_mulliken_range', 'Mulliken charge range (e)'),
    ]
    for ax, col, xlab in panels:
        if col not in d:
            ax.axis('off')
            continue
        for t, g in d.groupby('target'):
            ax.scatter(g[col], g['PIC50'], s=22, c=TARGET_COLORS.get(t, MUTED),
                       marker=TARGET_MARKERS.get(t, 'o'), edgecolors=vs.SURFACE,
                       linewidths=0.6, alpha=0.95, label=t, zorder=3)
        r, pv = stats.pearsonr(d[col], d['PIC50'])
        rs = stats.spearmanr(d[col], d['PIC50'])[0]
        ax.annotate(f'$r$ = {r:.3f} ($p$ = {pv:.3g})\n$\\rho$ = {rs:.3f}',
                    (0.03, 0.97), xycoords='axes fraction', ha='left', va='top',
                    fontsize=6.3, color=INK2)
        ax.set_xlabel(xlab)
        ax.set_ylabel('p$IC_{50}$')
        ax.grid(True, axis='both')

    h, lab = axes[0, 1].get_legend_handles_labels()
    fig.suptitle('DFT/xTB quantum descriptors (B3LYP/def2-SVP // GFN2-xTB) '
                 'against measured potency', fontsize=9.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.legend(h, lab, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.018),
               title='Target protein', title_fontsize=7.5)
    fig.text(0.5, -0.005,
             'No descriptor separates potency within a target; the vertical separation of '
             'the TDP1 series reflects target identity rather than ligand electronics.',
             ha='center', fontsize=6.4, color=INK2)
    return vs.save(fig, 'FigB6_dft_descriptors')


# --------------------------------------------- Fig 7: corrected model comparison
def fig_corrected_models(df):
    """Reproduces the four-panel figure of Hybrid model.docx with corrected numbers."""
    d = df[(df.split_regime == 'random') & (df.dataset == 'md272_clean')]
    if not len(d):
        return []
    models = ['nusvr', 'dnn', 'hybrid']
    subsets = ['train', 'cv', 'test', 'holdout']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))

    ax = axes[0]
    x = np.arange(len(subsets))
    w = 0.26
    for i, m in enumerate(models):
        v = [float(d[(d.model == m) & (d.subset == s)]['r2'].iloc[0])
             if len(d[(d.model == m) & (d.subset == s)]) else np.nan for s in subsets]
        bars = ax.bar(x + (i - 1) * w, v, w * 0.92, color=MODEL_COLORS[m],
                      label=MODEL_LABELS[m].split(' (')[0], edgecolor=vs.SURFACE,
                      linewidth=1.0, zorder=3)
        vs.bar_labels(ax, bars, fmt='{:.2f}', fontsize=5.6)
    ax.set_xticks(x)
    ax.set_xticklabels(['Train', 'CV', 'Test', 'Holdout'])
    ax.set_ylabel('$R^2$')
    ax.axhline(0, color=vs.BASELINE, lw=0.8)
    ax.set_title('Accuracy', loc='left')
    ax.legend(loc='lower left', fontsize=6.5)

    ax = axes[1]
    for i, m in enumerate(models):
        v = [float(d[(d.model == m) & (d.subset == s)]['rmse'].iloc[0])
             if len(d[(d.model == m) & (d.subset == s)]) else np.nan for s in subsets]
        bars = ax.bar(x + (i - 1) * w, v, w * 0.92, color=MODEL_COLORS[m],
                      edgecolor=vs.SURFACE, linewidth=1.0, zorder=3)
        vs.bar_labels(ax, bars, fmt='{:.2f}', fontsize=5.6)
    ax.set_xticks(x)
    ax.set_xticklabels(['Train', 'CV', 'Test', 'Holdout'])
    ax.set_ylabel('RMSE (log units)')
    ax.set_title('Error', loc='left')

    ax = axes[2]
    keys = _present(df, 'random')
    dd = df[(df.split_regime == 'random') & (df.subset == 'test') &
            (df.model == 'hybrid') & (df.dataset.isin(keys))]
    ax.scatter(dd['n_features'], dd['r2'], s=44, c=BLUE, edgecolors=vs.SURFACE,
               linewidths=0.8, zorder=3)
    for _, r in dd.iterrows():
        ax.annotate(SHORT.get(r['dataset'], r['dataset']).split('\n')[0],
                    (r['n_features'], r['r2']), textcoords='offset points',
                    xytext=(4, 4), fontsize=6, color=INK2)
    ax.set_xscale('log')
    ax.set_xlabel('Number of input descriptors (log)')
    ax.set_ylabel('Hybrid test $R^2$')
    ax.set_title('Dimensionality', loc='left')
    ax.grid(True, axis='both')

    for a, t in zip(axes, 'ABC'):
        vs.panel_tag(a, t)
    fig.suptitle('Corrected model comparison on the manuscript dataset '
                 '(leak-free out-of-fold stacking)', fontsize=9.5, fontweight='bold')
    fig.tight_layout()
    return vs.save(fig, 'FigB7_corrected_model_comparison')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    vs.apply_style()
    os.makedirs(TAB, exist_ok=True)
    df = load_metrics()
    nested = load_nested()

    todo = {
        'benchmark': lambda: fig_benchmark(df),
        'predobs': lambda: [fig_pred_obs('random'), fig_pred_obs('structure_disjoint')],
        'ablation': lambda: fig_ablation(df),
        'ceiling': fig_ceiling,
        'perm': lambda: fig_permutation(nested),
        'dft': fig_dft,
        'corrected': lambda: fig_corrected_models(df),
    }
    keys = [a.only] if a.only else list(todo)
    for k in keys:
        print(f'[{k}]')
        try:
            todo[k]()
        except Exception as e:
            import traceback
            print(f'  FAILED: {type(e).__name__}: {e}')
            traceback.print_exc(limit=3)


if __name__ == '__main__':
    main()
