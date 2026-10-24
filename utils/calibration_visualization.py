"""Diagnostic plots adapted from recon_stable's normal calibration views."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .calibration_ambiguity import _ambiguities, run_ambiguity_analysis
from .normal_mapping import lookup_slopes_raw, slopes_to_normals


def _matplotlib():
    try:
        cache = Path(tempfile.gettempdir()) / 'tactile_sensor_sim_matplotlib'
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('MPLCONFIGDIR', str(cache))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            'Visualization requires the calibration extra: install matplotlib') from exc
    return plt


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=_json_value), encoding='utf-8')
    temporary.replace(path)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Not JSON serializable: {type(value)}')


def _angle_errors(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    first = slopes_to_normals(predicted)
    second = slopes_to_normals(truth)
    return np.degrees(np.arccos(np.clip(np.sum(first * second, axis=-1), -1, 1)))


def _lookup_chunked(colors: np.ndarray, calibration, chunk_size=1_000_000):
    predicted, variance, confidence = [], [], []
    for start in range(0, len(colors), chunk_size):
        values = lookup_slopes_raw(colors[start:start + chunk_size], calibration)
        predicted.append(values[0])
        variance.append(values[1])
        confidence.append(values[2])
    return (np.concatenate(predicted), np.concatenate(variance),
            np.concatenate(confidence))


def _plot_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _save_figure(figure, path: Path) -> None:
    figure.savefig(path, dpi=140, bbox_inches='tight')
    _matplotlib().close(figure)


def _plot_calibration_summary(path: Path, truth: np.ndarray,
                              fitted: np.ndarray, fit_angles: np.ndarray,
                              held_truth: np.ndarray, held_predicted: np.ndarray,
                              held_angles: np.ndarray, validation: dict) -> None:
    plt = _matplotlib()
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    selected = _plot_indices(len(truth), 8000)
    held_selected = _plot_indices(len(held_truth), 8000)
    labels = ('p', 'q')
    for axis, component in zip(axes[0, :2], range(2), strict=True):
        axis.scatter(truth[selected, component], fitted[selected, component],
                     s=2, alpha=.18, rasterized=True)
        limits = [min(truth[selected, component].min(), fitted[selected, component].min()),
                  max(truth[selected, component].max(), fitted[selected, component].max())]
        axis.plot(limits, limits, 'k--', linewidth=1)
        axis.set(title=f'Calibration fit: {labels[component]}',
                 xlabel='geometry truth', ylabel='raw LUT')
        axis.set_aspect('equal', adjustable='box')
    axes[0, 2].hist(fit_angles, bins=50, color='#4477aa')
    axes[0, 2].set(title='Calibration fit normal error', xlabel='degrees', ylabel='samples')
    for axis, component in zip(axes[1, :2], range(2), strict=True):
        axis.scatter(held_truth[held_selected, component],
                     held_predicted[held_selected, component], s=2, alpha=.18,
                     color='#cc6677', rasterized=True)
        limits = [min(held_truth[held_selected, component].min(),
                      held_predicted[held_selected, component].min()),
                  max(held_truth[held_selected, component].max(),
                      held_predicted[held_selected, component].max())]
        axis.plot(limits, limits, 'k--', linewidth=1)
        axis.set(title=f'Held-out images: {labels[component]}',
                 xlabel='geometry truth', ylabel='training LUT')
        axis.set_aspect('equal', adjustable='box')
    image_rows = validation.get('images', [])
    labels = [row['image_key'] for row in image_rows]
    axes[1, 2].bar(labels, [row['rms_degrees'] for row in image_rows],
                   color='#cc6677', label='held-out RMS')
    threshold = validation.get('thresholds', {}).get('rms_degrees')
    if threshold is not None:
        axes[1, 2].axhline(threshold, color='red', linestyle='--',
                           label=f'limit {threshold:g} deg')
    axes[1, 2].set(title=(f'Held-out images ({validation["status"]})\n'
                          f'global RMS={validation["rms_degrees"]:.3f} deg'),
                   xlabel='image key', ylabel='RMS degrees')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].legend(fontsize=8)
    figure.suptitle('Normal LUT calibration and whole-image validation')
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_color_mapping(path: Path, colors: np.ndarray, predicted: np.ndarray) -> None:
    plt = _matplotlib()
    selected = _plot_indices(len(colors), 6000)
    figure = plt.figure(figsize=(13, 6))
    slope_bound = max(float(np.max(np.abs(predicted))), 1e-6)
    for number, component in enumerate((0, 1), 1):
        axis = figure.add_subplot(1, 2, number, projection='3d')
        plotted = axis.scatter(colors[selected, 0], colors[selected, 1],
                               colors[selected, 2], c=predicted[selected, component],
                               s=5, alpha=.65, cmap='coolwarm', rasterized=True,
                               vmin=-slope_bound, vmax=slope_bound)
        axis.set(xlabel='delta R', ylabel='delta G', zlabel='delta B',
                 title=f'Raw LUT {"p" if component == 0 else "q"}')
        figure.colorbar(plotted, ax=axis, shrink=.65)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_lut_slices(path: Path, calibration) -> None:
    plt = _matplotlib()
    names = ('R', 'G', 'B')
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    bound = float(np.max(np.abs(calibration.slopes)))
    for channel in range(3):
        remaining = [item for item in range(3) if item != channel]
        coordinates = [np.linspace(calibration.color_min[index],
                                   calibration.color_max[index], calibration.size)
                       for index in remaining]
        first, second = np.meshgrid(*coordinates)
        query = np.zeros((*first.shape, 3), np.float32)
        query[..., remaining[0]] = first
        query[..., remaining[1]] = second
        values = lookup_slopes_raw(query, calibration)[0]
        for component in range(2):
            shown = axes[component, channel].imshow(
                values[..., component].T, origin='lower', cmap='coolwarm',
                vmin=-bound, vmax=bound, aspect='auto')
            axes[component, channel].set(
                title=(f'{"p" if component == 0 else "q"}: '
                       f'delta {names[channel]}=0'),
                xlabel=f'delta {names[remaining[0]]} bin',
                ylabel=f'delta {names[remaining[1]]} bin')
            figure.colorbar(shown, ax=axes[component, channel], shrink=.75)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_ambiguities(path: Path, slopes: np.ndarray, ambiguity: dict) -> None:
    plt = _matplotlib()
    mask = ambiguity['mask']
    regular = np.flatnonzero(~mask)
    marked = np.flatnonzero(mask)
    regular = regular[_plot_indices(len(regular), 6000)]
    marked = marked[_plot_indices(len(marked), 6000)]
    full_extent = max(.5, float(np.ceil(np.max(np.abs(slopes)) * 1.05 * 2) / 2))
    views = [(full_extent, 'All geometric slopes')]
    if full_extent > 2:
        views.append((2., 'Central [-2, 2] view'))
    figure, axes = plt.subplots(1, len(views), figsize=(6 * len(views), 5), squeeze=False)
    for axis, (extent, title) in zip(axes.ravel(), views, strict=True):
        axis.scatter(slopes[regular, 0], slopes[regular, 1], s=3, alpha=.15,
                     color='#4477aa', rasterized=True, label='other samples')
        axis.scatter(slopes[marked, 0], slopes[marked, 1], s=12, alpha=.65,
                     marker='*', color='red', rasterized=True, label='ambiguous endpoints')
        axis.set(xlabel='p', ylabel='q', title=title,
                 xlim=(-extent, extent), ylim=(extent, -extent))
        axis.set_aspect('equal', adjustable='box')
    axes.ravel()[0].legend(loc='best')
    sampling = ambiguity.get('sampling', {})
    scope = (f'sampled {sampling["analyzed_sample_count"]}/{sampling["original_sample_count"]}'
             if sampling.get('is_sampled') else f'all {len(slopes)} samples')
    figure.suptitle(f'Color ambiguity: {ambiguity["pair_count"]} pairs within {scope}')
    figure.tight_layout()
    _save_figure(figure, path)


def generate_calibration_visualizations(samples, final_lut, training_lut,
                                        settings, output_dir: Path) -> Path:
    """Generate global and branch plots exclusively from first-pass samples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    colors = np.asarray(samples['color_residual_samples'], np.float32)
    truth = np.asarray(samples['slope_samples'], np.float32)
    image_index = np.asarray(samples['image_index'], np.int32)
    keys = np.asarray(samples['image_paths']).astype(str)
    fitted, _, _ = _lookup_chunked(colors, final_lut)
    fit_angles = _angle_errors(fitted, truth)
    validation_keys = set(final_lut.metadata.get('validation', {}).get(
        'validation_images', training_lut.metadata.get('validation_images', [])))
    held_indices = np.asarray([index for index, key in enumerate(keys)
                               if key in validation_keys], np.int32)
    held_mask = np.isin(image_index, held_indices)
    held_predicted, _, _ = _lookup_chunked(colors[held_mask], training_lut)
    held_truth = truth[held_mask]
    held_angles = _angle_errors(held_predicted, held_truth)
    validation = final_lut.metadata.get('validation', {})
    if not validation:
        validation = {'status': final_lut.metadata.get('validation_status', 'unknown'),
                      'rms_degrees': float(np.sqrt(np.mean(held_angles ** 2)))}

    _plot_calibration_summary(output_dir / 'calibration_summary.png', truth, fitted,
                              fit_angles, held_truth, held_predicted, held_angles,
                              validation)
    _plot_color_mapping(output_dir / 'color_pq_mapping.png', colors, fitted)
    _plot_lut_slices(output_dir / 'lut_slices.png', final_lut)
    ambiguity = run_ambiguity_analysis(
        settings.output_dir / 'ambiguity_inputs.npz', settings.output_dir,
        stride=settings.ambiguity_stride,
        color_threshold=settings.color_distance_threshold,
        angle_threshold=settings.normal_angle_threshold_degrees,
        export_pairs=settings.export_ambiguity_pairs,
        dominance_fraction=getattr(settings, 'branch_dominance_fraction', .8))

    summary = {
        'format_version': 1,
        'fit_kind': ('diagnostic_full_sample_fit' if
                     final_lut.metadata.get('validation_status') == 'failed'
                     else 'published_full_sample_fit'),
        'sample_count': len(colors),
        'accepted_pose_count': len(keys),
        'supported_lut_node_count': int(np.count_nonzero(final_lut.original_valid)),
        'retained_lut_sample_count': int(np.sum(final_lut.sample_counts)),
        'calibration_fit': {
            'rms_degrees': float(np.sqrt(np.mean(fit_angles ** 2))),
            'median_degrees': float(np.median(fit_angles)),
            'p95_degrees': float(np.percentile(fit_angles, 95)),
        },
        'held_out_validation': validation,
        'ambiguity': {key: value for key, value in ambiguity.items() if key != 'mask'},
        'artifacts': {
            'calibration_summary': 'calibration_summary.png',
            'color_pq_mapping': 'color_pq_mapping.png',
            'lut_slices': 'lut_slices.png',
            'pq_ambiguities': 'pq_ambiguities.png',
            'branch_classification': 'ambiguity_patterns/branch_classification.png',
            'branch_continuation': 'ambiguity_patterns/branch_continuation.png',
            'branch_analysis': 'ambiguity_patterns/analysis.json',
        },
        'diagnostics_source': 'first_pass_samples',
        'visualization_render_count': 0,
        'visualization_render_seconds': 0.,
        'total_visualization_seconds': time.perf_counter() - started,
    }
    _write_json(output_dir / 'summary.json', summary)
    return output_dir
