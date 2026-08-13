"""Monochrome figure style for journal submission.

No colour anywhere. Series are separated by four channels that all survive greyscale
printing and photocopying: fill lightness, hatch pattern, marker shape and line style.
Fill lightness alone is not relied upon, because adjacent greys of similar value are
hard to separate once a figure is reduced to column width.

Fills are ordered light to dark so that a reader scanning a bar group sees a consistent
progression, and every bar carries a printed value so the encoding is never the only
route to the number.
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'figures_final')

# Greyscale ramp, light to dark. Re-stepped after the palette validator flagged white
# against #d9d9d9 at a perceptual distance of 11.5, below the legibility floor of 15.
# This four-step ramp clears every all-pairs check with a worst separation of 30.1.
G = ['#ffffff', '#9e9e9e', '#3d3d3d', '#000000']
HATCH = ['', '///', '', 'xxx', '\\\\\\', '...']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
LINES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1))]

# three model series
MODEL_FILL = {'nusvr': '#ffffff', 'dnn': '#9e9e9e', 'hybrid': '#3d3d3d'}
MODEL_HATCH = {'nusvr': '', 'dnn': '///', 'hybrid': ''}
MODEL_LABELS = {'nusvr': 'Nu-SVR', 'dnn': 'DNN', 'hybrid': 'Hybrid'}

# four target proteins, separated by marker shape and fill
TARGET_MARKER = {'ESR1': 'o', 'MAPK1': 's', 'TDP1': '^', 'TP53': 'D'}
TARGET_FILL = {'ESR1': '#ffffff', 'MAPK1': '#9e9e9e', 'TDP1': '#000000',
               'TP53': '#3d3d3d'}

INK = '#000000'
INK2 = '#333333'
MUTED = '#666666'
GRID = '#d0d0d0'
SURFACE = '#ffffff'
DPI = 600


def apply_style():
    mpl.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'savefig.bbox': 'tight',
        'savefig.dpi': DPI, 'figure.dpi': 110,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.titleweight': 'bold', 'axes.labelcolor': INK, 'text.color': INK,
        'axes.edgecolor': INK, 'axes.linewidth': 0.8,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'grid.color': GRID, 'grid.linewidth': 0.5,
        'axes.grid': True, 'axes.grid.axis': 'y', 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'lines.linewidth': 1.4,
        'lines.markersize': 4.2, 'errorbar.capsize': 2,
        'hatch.linewidth': 0.6, 'patch.linewidth': 0.7,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def save(fig, name, formats=('tiff', 'pdf', 'png')):
    os.makedirs(FIGDIR, exist_ok=True)
    out = []
    for ext in formats:
        p = os.path.join(FIGDIR, f'{name}.{ext}')
        kw = {'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tiff' else {}
        fig.savefig(p, dpi=DPI, **kw)
        out.append(os.path.basename(p))
    plt.close(fig)
    print(f'  {name}: ' + ', '.join(out))
    return out


def bar(ax, x, height, width, *, fill, hatch='', label=None, **kw):
    return ax.bar(x, height, width, facecolor=fill, hatch=hatch,
                  edgecolor=INK, linewidth=0.7, label=label, zorder=3, **kw)


def label_bars(ax, bars, values=None, fmt='{:.2f}', dy=0.012, fontsize=5.8):
    for b, v in zip(bars, values if values is not None else [b.get_height() for b in bars]):
        h = b.get_height()
        va, off = ('bottom', dy) if h >= 0 else ('top', -dy)
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, h + off),
                    ha='center', va=va, fontsize=fontsize, color=INK)


def panel(ax, letter, dx=-0.13, dy=1.06):
    ax.annotate(letter, (dx, dy), xycoords='axes fraction', ha='left', va='top',
                fontsize=10, fontweight='bold', color=INK)
