"""Static scientific figures; heatmaps are labelled RGB-node slices, not projections."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import numpy as np

from utils.normal_mapping import NormalCalibration
from .metric import bin_samples


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _slice_image(ax, values, arrays, lut, blue, *, counts=False):
    data = np.full((lut.size, lut.size), np.nan)
    xyz = arrays['coordinates']
    selected = xyz[:, 2] == blue
    data[xyz[selected, 0], xyz[selected, 1]] = values[selected]
    options = dict(norm=LogNorm(vmin=1, vmax=max(2, float(values.max())))) if counts else dict(vmin=0, vmax=1)
    im = ax.imshow(data.T, origin='lower', extent=(-.5, lut.size-.5, -.5, lut.size-.5),
                   cmap='viridis', interpolation='nearest', **options)
    residual_b = lut.color_min[2] + blue / (lut.size - 1) * (lut.color_max[2] - lut.color_min[2])
    ax.set(title=f'B node {blue}, dB={residual_b:.4g}', xlabel='R node', ylabel='G node')
    ax.set_facecolor('#f3e9d7')
    return im


def save_plots(output: Path, report: dict, arrays: dict, examples: dict,
               colors: np.ndarray, slopes: np.ndarray, lut: NormalCalibration):
    rows = report['reference']['curves']
    taus = [row['tau'] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in (('incompatibility', 'A: best fixed-node incompatibility'),
                       ('node_error', 'E_node: stored nearest-node output'),
                       ('runtime_error', 'E_runtime: actual trilinear lookup')):
        ax.plot(taus, [row[key] for row in rows], 'o-', label=label)
    if report['holdout']:
        held = report['holdout']['curves']
        ax.plot(taus, [row['runtime_error'] for row in held], 's--', label='Held-out E_runtime')
        ax.plot(taus, [row['transferred_disc_failure_or_unseen'] for row in held], 's--',
                label='Held-out fitted disc failure or unseen bin')
    ax.set(xlabel='Slope tolerance tau: Euclidean distance in (p, q)',
           ylabel='Failure fraction (lower is better)', ylim=(-.02, 1.02),
           title='LUT quality: empirical fixed-node bound vs actual lookup')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    _save(fig, output / 'curves.png')

    # Three actual slices through occupied B levels; no hidden averaging/max projection.
    levels = np.unique(arrays['coordinates'][:, 2])
    blue_levels = np.unique(levels[np.rint(np.linspace(0, len(levels)-1, min(3, len(levels)))).astype(int)])
    fig, axes = plt.subplots(len(taus), len(blue_levels), squeeze=False,
                             figsize=(4 * len(blue_levels), 3.3 * len(taus)))
    for t, tau in enumerate(taus):
        for col, blue in enumerate(blue_levels):
            im = _slice_image(axes[t, col], arrays['incompatibility'][t], arrays, lut, int(blue))
            axes[t, col].set_ylabel(f'tau={tau:g}\nG node')
            fig.colorbar(im, ax=axes[t, col], label='A_bin')
    fig.suptitle('Per-bin incompatibility: RGB node slices (beige = no samples)')
    fig.tight_layout()
    _save(fig, output / 'bin_heatmaps.png')

    fig, axes = plt.subplots(1, len(blue_levels), squeeze=False,
                             figsize=(4 * len(blue_levels), 3.6))
    for col, blue in enumerate(blue_levels):
        im = _slice_image(axes[0, col], arrays['sample_count'], arrays, lut, int(blue), counts=True)
        fig.colorbar(im, ax=axes[0, col], label='Raw sample count (log scale)')
    fig.suptitle('Sampling support; singleton bins are not evidence of uniqueness')
    fig.tight_layout()
    _save(fig, output / 'sample_counts.png')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.pcolormesh(arrays['phi_edges_degrees'], arrays['theta_edges_degrees'],
                       arrays['normal_coverage_count'], shading='flat', cmap='viridis')
    ax.set(xlabel='Normal azimuth phi (degrees)', ylabel='Normal polar angle theta (degrees)',
           title='Raw reference-sample normal coverage (not pose-balanced)')
    fig.colorbar(im, ax=ax, label='Raw sample count')
    _save(fig, output / 'normal_coverage.png')

    shown = examples['bins'][:4]
    fig, axes = plt.subplots(1, len(shown), squeeze=False, figsize=(4.3 * len(shown), 4))
    ids = bin_samples(colors, lut)
    for ax, row in zip(axes[0], shown, strict=True):
        points = slopes[ids == row['bin_id']]
        ax.scatter(points[:, 0], points[:, 1], s=9, alpha=.35, color='#3f708f', label='All raw rows in bin')
        center, node = np.asarray(row['optimal_center']), np.asarray(row['node_output'])
        ax.add_patch(Circle(center, examples['tau'], fill=False, color='#dd8a16', linewidth=2))
        ax.scatter(*center, marker='x', color='#dd8a16', s=50, label='Optimal disc centre')
        ax.scatter(*node, marker='+', color='#ba2f46', s=65, label='Stored node output')
        ax.set(xlabel='p', ylabel='q', title=f'RGB node {row["rgb_node"]}\nN={row["sample_count"]}, A={row["incompatibility"]:.3f}')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle(f'High-incompatibility bins: all raw slopes, tau={examples["tau"]:g}')
    fig.tight_layout()
    _save(fig, output / 'top_bins.png')
