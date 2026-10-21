"""Diagnostic plots adapted from recon_stable's normal calibration views."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .calibration_samples import (compact_frame_samples,
                                  image_space_diagnostics, sample_pose)
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


def _ambiguities(colors: np.ndarray, slopes: np.ndarray, image_index: np.ndarray,
                 color_threshold: float, angle_threshold: float,
                 export_pairs: bool, output_dir: Path) -> dict:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError('Ambiguity analysis requires scipy') from exc
    tree = cKDTree(np.asarray(colors, np.float64))
    normals = slopes_to_normals(slopes)
    endpoints = np.zeros(len(colors), bool)
    pair_count = within_count = cross_count = 0
    saved_pairs, saved_distances, saved_angles = [], [], []
    chunk_size = 8192
    for start in range(0, len(colors), chunk_size):
        stop = min(len(colors), start + chunk_size)
        neighborhoods = tree.query_ball_point(colors[start:stop], color_threshold,
                                              eps=0, workers=-1)
        for local, neighbors in enumerate(neighborhoods):
            first = start + local
            second = np.asarray(neighbors, np.int64)
            second = second[second > first]
            if not len(second):
                continue
            dot = np.sum(normals[first] * normals[second], axis=-1)
            cross = np.linalg.norm(np.cross(normals[first], normals[second]), axis=-1)
            angle = np.degrees(np.arctan2(cross, dot))
            selected = angle > angle_threshold
            if not np.any(selected):
                continue
            second = second[selected]
            angle = angle[selected]
            endpoints[first] = True
            endpoints[second] = True
            pair_count += len(second)
            same = image_index[first] == image_index[second]
            within_count += int(np.count_nonzero(same))
            cross_count += int(np.count_nonzero(~same))
            if export_pairs:
                pair = np.column_stack((np.full(len(second), first, np.int64), second))
                saved_pairs.append(pair)
                saved_distances.append(np.linalg.norm(colors[first] - colors[second], axis=-1))
                saved_angles.append(angle)
    artifact = None
    if export_pairs:
        pairs = (np.concatenate(saved_pairs) if saved_pairs
                 else np.empty((0, 2), np.int64))
        distances = (np.concatenate(saved_distances) if saved_distances
                     else np.empty(0, np.float64))
        angles = (np.concatenate(saved_angles) if saved_angles
                  else np.empty(0, np.float64))
        artifact = output_dir / 'ambiguities.npz'
        temporary = artifact.with_suffix(artifact.suffix + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, pairs=pairs, distances=distances,
                                angles_degrees=angles,
                                ambiguous_sample_mask=endpoints)
        temporary.replace(artifact)
    return {
        'mask': endpoints,
        'pair_count': pair_count,
        'ambiguous_sample_count': int(endpoints.sum()),
        'within_image_pair_count': within_count,
        'cross_image_pair_count': cross_count,
        'pairs_exported': export_pairs,
        'pair_artifact': artifact.name if artifact else None,
        'color_distance_threshold': color_threshold,
        'normal_angle_threshold_degrees': angle_threshold,
    }


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
    figure.suptitle(f'Color ambiguity: {ambiguity["pair_count"]} exact pairs')
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_frame_color(path: Path, residual: np.ndarray, valid: np.ndarray,
                      difference_gain: float, color_bounds: np.ndarray, key: str) -> None:
    plt = _matplotlib()
    masked = np.where(valid[..., None], residual, np.nan)
    composite = np.clip(.5 + difference_gain * residual, 0, 1)
    composite[~valid] = 0
    color_map = plt.get_cmap('coolwarm').copy()
    color_map.set_bad('black')
    figure, axes = plt.subplots(1, 4, figsize=(17, 4))
    axes[0].imshow(composite)
    axes[0].set_title('Signed delta RGB composite')
    for component, name in enumerate('RGB'):
        shown = axes[component + 1].imshow(masked[..., component], cmap=color_map,
                                           vmin=-color_bounds[component],
                                           vmax=color_bounds[component])
        axes[component + 1].set_title(f'delta {name}')
        figure.colorbar(shown, ax=axes[component + 1], shrink=.75)
    for axis in axes:
        axis.axis('off')
    figure.suptitle(f'{key}: material-aligned color difference')
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_frame_slopes(path: Path, residual: np.ndarray, truth: np.ndarray,
                       photometric_valid: np.ndarray, paired_valid: np.ndarray,
                       calibration, slope_bound: float, angle_bound: float,
                       key: str) -> None:
    plt = _matplotlib()
    predicted = np.zeros_like(truth, np.float32)
    if np.any(photometric_valid):
        predicted[photometric_valid] = lookup_slopes_raw(
            residual[photometric_valid], calibration)[0]
    error = np.full(paired_valid.shape, np.nan, np.float32)
    error[paired_valid] = _angle_errors(predicted[paired_valid], truth[paired_valid])
    pred_mag = np.linalg.norm(predicted, axis=-1)
    pred_mag[~photometric_valid] = np.nan
    diverging = plt.get_cmap('coolwarm').copy()
    diverging.set_bad('black')
    sequential = plt.get_cmap('viridis').copy()
    sequential.set_bad('black')
    angle_map = plt.get_cmap('magma').copy()
    angle_map.set_bad('black')
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    values = (predicted[..., 0], predicted[..., 1], pred_mag,
              truth[..., 0], truth[..., 1], error)
    titles = ('raw LUT p', 'raw LUT q', 'raw LUT magnitude',
              'geometry p', 'geometry q', 'normal angle error (deg)')
    for index, (axis, value, title) in enumerate(zip(axes.flat, values, titles, strict=True)):
        mask = photometric_valid if index < 3 else paired_valid
        data = np.where(mask, value, np.nan)
        if index in (0, 1, 3, 4):
            shown = axis.imshow(data, cmap=diverging,
                                vmin=-slope_bound, vmax=slope_bound)
        elif index == 5:
            shown = axis.imshow(data, cmap=angle_map, vmin=0, vmax=angle_bound)
        else:
            shown = axis.imshow(data, cmap=sequential, vmin=0,
                                vmax=math.sqrt(2) * slope_bound)
        axis.set_title(title)
        axis.axis('off')
        figure.colorbar(shown, ax=axis, shrink=.72)
    figure.suptitle(f'{key}: LUT prediction and analytic sphere truth')
    figure.tight_layout()
    _save_figure(figure, path)


def generate_calibration_visualizations(context, reference, grid, poses, rows,
                                        samples, final_lut, training_lut,
                                        settings, output_dir: Path) -> Path:
    """Generate all global and per-pose diagnostics without frame caches."""
    output_dir = Path(output_dir)
    color_dir = output_dir / 'color_difference'
    slope_dir = output_dir / 'pq_mapping'
    color_dir.mkdir(parents=True, exist_ok=False)
    slope_dir.mkdir(parents=True, exist_ok=False)
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
    ambiguity = _ambiguities(colors, truth, image_index,
                             settings.color_distance_threshold,
                             settings.normal_angle_threshold_degrees,
                             settings.export_ambiguity_pairs, output_dir)
    _plot_ambiguities(output_dir / 'pq_ambiguities.png', truth, ambiguity)

    accepted = {row['image_key']: pose for row, pose in zip(rows, poses, strict=True)
                if row['status'] == 'accepted'}
    color_bounds = np.maximum(np.maximum(np.abs(final_lut.color_min),
                                         np.abs(final_lut.color_max)), 1e-6)
    slope_bound = max(float(np.max(np.abs(truth))), 1e-6)
    angle_bound = max(5., float(validation.get('thresholds', {}).get(
        'rms_degrees', 5.)))
    per_image = []
    visualization_render_seconds = 0.
    for index, key in enumerate(keys):
        pose = accepted[key]
        context.torch.cuda.synchronize(context.device)
        frame_started = time.perf_counter()
        frame = sample_pose(context, reference, grid, pose)
        context.torch.cuda.synchronize(context.device)
        visualization_render_seconds += time.perf_counter() - frame_started
        compact = compact_frame_samples(frame, grid)
        expected = image_index == index
        if not np.array_equal(compact['material_grid_index'],
                              samples['material_grid_index'][expected]) \
                or not np.allclose(compact['color_residual_samples'],
                                   colors[expected], rtol=0, atol=2e-6) \
                or not np.allclose(compact['slope_samples'], truth[expected],
                                   rtol=0, atol=2e-6):
            raise RuntimeError(f'Second-pass samples do not match saved samples for {key}')
        residual, true_image, photo_valid, paired_valid = image_space_diagnostics(
            context, reference, frame)
        residual = residual.detach().cpu().numpy()
        true_image = true_image.detach().cpu().numpy()
        photo_valid = photo_valid.detach().cpu().numpy()
        paired_valid = paired_valid.detach().cpu().numpy()
        color_path = color_dir / f'{key}.png'
        slope_path = slope_dir / f'{key}.png'
        _plot_frame_color(color_path, residual, photo_valid,
                          settings.difference_gain, color_bounds, key)
        _plot_frame_slopes(slope_path, residual, true_image, photo_valid,
                           paired_valid, final_lut, slope_bound, angle_bound, key)
        predicted_image = lookup_slopes_raw(colors[expected], final_lut)[0]
        angles = _angle_errors(predicted_image, truth[expected])
        per_image.append({
            'image_key': key,
            'sample_count': int(expected.sum()),
            'calibration_fit_rms_degrees': float(np.sqrt(np.mean(angles ** 2))),
            'color_difference': str(color_path.relative_to(output_dir)),
            'pq_mapping': str(slope_path.relative_to(output_dir)),
        })

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
        'per_image': per_image,
        'artifacts': {
            'calibration_summary': 'calibration_summary.png',
            'color_pq_mapping': 'color_pq_mapping.png',
            'lut_slices': 'lut_slices.png',
            'pq_ambiguities': 'pq_ambiguities.png',
        },
        'visualization_render_seconds': visualization_render_seconds,
        'total_visualization_seconds': time.perf_counter() - started,
    }
    _write_json(output_dir / 'summary.json', summary)
    return output_dir
