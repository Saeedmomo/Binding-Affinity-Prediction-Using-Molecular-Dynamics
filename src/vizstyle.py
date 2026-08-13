"""Shared publication figure style.

Palette taken from the dataviz reference instance and validated with its own
validator against a white print surface on the ALL-PAIRS list (scatter, small
multiples), not just adjacent pairs:

  models  #2a78d6 blue, #eb6834 orange, #1baf7a aqua
          worst all-pairs CVD dE 9.2 (deutan), normal-vision dE 24.0  -> PASS
  targets the same three plus #4a3aa7 violet
          worst all-pairs CVD dE 9.2, normal-vision dE 16.3           -> PASS

Aqua sits at 2.82:1 against white, below the 3:1 chrome floor, so the relief rule
applies: every bar carries a visible direct label and every figure has a companion
table in tables/. Target identity is additionally encoded by marker shape, so colour
is never the only channel.
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')

# categorical slots
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
MODEL_COLORS = {'nusvr': BLUE, 'dnn': ORANGE, 'hybrid': AQUA}
MODEL_LABELS = {'nusvr': 'Nu-SVR', 'dnn': 'DNN', 'hybrid': 'Hybrid (Nu-SVR+DNN+Ridge)'}
TARGET_COLORS = {'ESR1': BLUE, 'MAPK1': ORANGE, 'TDP1': AQUA, 'TP53': VIOLET}
TARGET_MARKERS = {'ESR1': 'o', 'MAPK1': 's', 'TDP1': '^', 'TP53': 'D'}

# chrome & ink
SURFACE = '#ffffff'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
REF = '#52514e'

DPI = 600


def apply_style():
    mpl.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'savefig.bbox': 'tight',
        'savefig.dpi': DPI, 'figure.dpi': 110,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.titleweight': 'bold', 'axes.labelcolor': INK, 'text.color': INK,
        'axes.edgecolor': BASELINE, 'axes.linewidth': 0.7,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.alpha': 1.0,
        'axes.grid': True, 'axes.grid.axis': 'y', 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'lines.linewidth': 2.0,
        'lines.markersize': 4.5, 'errorbar.capsize': 2,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def save(fig, name, formats=('png', 'pdf', 'tiff')):
    """Write one figure in every format a journal is likely to ask for."""
    os.makedirs(FIGDIR, exist_ok=True)
    out = []
    for ext in formats:
        p = os.path.join(FIGDIR, f'{name}.{ext}')
        kw = {}
        if ext == 'tiff':
            kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(p, dpi=DPI, **kw)
        out.append(p)
    plt.close(fig)
    print('  saved ' + ', '.join(os.path.basename(p) for p in out))
    return out


def bar_labels(ax, bars, fmt='{:.3f}', dy=0.012, fontsize=6.0, skip_below=None):
    """Direct labels on bars - required here, not decorative (see module docstring)."""
    for b in bars:
        h = b.get_height()
        if skip_below is not None and abs(h) < skip_below:
            continue
        va, off = ('bottom', dy) if h >= 0 else ('top', -dy)
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h + off),
                    ha='center', va=va, fontsize=fontsize, color=INK2)


def hline(ax, y, label, color=REF, ls='--', lw=1.0, x=0.995, va='bottom', fontsize=6.5):
    ax.axhline(y, color=color, ls=ls, lw=lw, zorder=1)
    ax.annotate(label, (x, y), xycoords=('axes fraction', 'data'),
                ha='right', va=va, fontsize=fontsize, color=color)


def panel_tag(ax, letter, dx=-0.085, dy=1.045):
    ax.annotate(letter, (dx, dy), xycoords='axes fraction', ha='left', va='top',
                fontsize=10, fontweight='bold', color=INK)
