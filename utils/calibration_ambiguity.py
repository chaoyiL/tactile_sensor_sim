"""Persist and reanalyze color/normal ambiguities without a renderer or CUDA."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from .normal_mapping import slopes_to_normals


def _write_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def save_ambiguity_inputs(path: Path, samples: dict, source_samples: Path) -> None:
    """Keep the full, pre-MAD samples; sampling is applied only during analysis."""
    fields = ('color_residual_samples', 'slope_samples', 'image_index', 'image_paths')
    values = {key: np.asarray(samples[key]) for key in fields}
    metadata = json.loads(str(np.asarray(samples.get('metadata', '{}')).item()))
    values.update(format_version=np.asarray(1, np.int32),
                  source_samples=np.asarray(str(source_samples.resolve())),
                  run_signature=np.asarray(metadata.get('run_signature', '')),
                  metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    _write_npz(Path(path), **values)


def select_ambiguity_samples(image_index: np.ndarray, stride: int) -> np.ndarray:
    """Select every stride-th sample within each image without a global limit."""
    if type(stride) is not int or stride < 1:
        raise ValueError('ambiguity_stride must be an integer >= 1')
    image_index = np.asarray(image_index)
    if image_index.ndim != 1 or not np.issubdtype(image_index.dtype, np.integer):
        raise ValueError('image_index must be a one-dimensional integer array')
    if not len(image_index):
        raise ValueError('No samples available for ambiguity analysis')
    # Stable grouping also handles imported archives whose images are interleaved.
    order = np.argsort(image_index, kind='stable')
    grouped = image_index[order]
    starts = np.r_[0, np.flatnonzero(grouped[1:] != grouped[:-1]) + 1]
    ends = np.r_[starts[1:], len(grouped)]
    selected = np.concatenate([order[start:stop:stride]
                               for start, stop in zip(starts, ends, strict=True)])
    return np.sort(selected)


def _ambiguities(colors: np.ndarray, slopes: np.ndarray, image_index: np.ndarray,
                 color_threshold: float, angle_threshold: float,
                 export_pairs: bool, output_dir: Path, *, pair_sink=None) -> dict:
    """Exact pair statistics on the supplied subset, with bounded query batches."""
    from scipy.spatial import cKDTree

    if not math.isfinite(color_threshold) or color_threshold < 0:
        raise ValueError('color_distance_threshold must be finite and >= 0')
    if not math.isfinite(angle_threshold) or not 0 <= angle_threshold <= 180:
        raise ValueError('normal_angle_threshold_degrees must be in [0,180]')
    colors = np.asarray(colors, np.float64)
    slopes = np.asarray(slopes, np.float64)
    image_index = np.asarray(image_index)
    if colors.shape != (len(image_index), 3) or slopes.shape != (len(image_index), 2) \
            or not np.isfinite(colors).all() or not np.isfinite(slopes).all():
        raise ValueError('Ambiguity inputs must be aligned finite RGB, slopes and image indices')
    tree = cKDTree(colors)
    normals = slopes_to_normals(slopes)
    endpoints = np.zeros(len(colors), bool)
    pair_count = within_count = cross_count = 0
    saved_pairs, saved_distances, saved_angles = [], [], []
    # Unlike an 8192-row query, even completely coincident colors cannot create
    # millions of Python neighbor lists at once. Full mode still needs O(N) per row.
    chunk_size = max(1, min(128, 262144 // max(1, len(colors))))
    last_progress = time.perf_counter()
    cosine_threshold = math.cos(math.radians(angle_threshold))
    for start in range(0, len(colors), chunk_size):
        stop = min(len(colors), start + chunk_size)
        neighborhoods = tree.query_ball_point(colors[start:stop], color_threshold,
                                              eps=0, workers=1, return_sorted=False)
        for local, neighbors in enumerate(neighborhoods):
            first = start + local
            second = np.asarray(neighbors, np.int64)
            second = second[second > first]
            if not len(second):
                continue
            dot = np.sum(normals[first] * normals[second], axis=-1)
            # Reject clearly small angles cheaply; atan2 below retains the original
            # strict angle comparison, including threshold-boundary cases.
            possible = dot <= cosine_threshold + 1e-12
            second, dot = second[possible], dot[possible]
            if not len(second):
                continue
            cross = np.linalg.norm(np.cross(normals[first], normals[second]), axis=-1)
            angle = np.degrees(np.arctan2(cross, dot))
            selected = angle > angle_threshold
            second, angle = second[selected], angle[selected]
            if not len(second):
                continue
            endpoints[first] = True
            endpoints[second] = True
            pair_count += len(second)
            same = image_index[first] == image_index[second]
            within_count += int(np.count_nonzero(same))
            cross_count += int(np.count_nonzero(~same))
            if pair_sink is not None:
                pair = np.column_stack((np.full(len(second), first, np.int64), second))
                pair_sink.write(pair.tobytes())
            if export_pairs:
                saved_pairs.append(np.column_stack((np.full(len(second), first, np.int64), second)))
                saved_distances.append(np.linalg.norm(colors[first] - colors[second], axis=-1))
                saved_angles.append(angle)
        if time.perf_counter() - last_progress >= 5 or stop == len(colors):
            print(f'[ambiguity] {stop}/{len(colors)} samples; {pair_count} conflicting pairs',
                  flush=True)
            last_progress = time.perf_counter()
    artifact = None
    if export_pairs:
        artifact = Path(output_dir) / 'ambiguities.npz'
        _write_npz(artifact,
                   pairs=np.concatenate(saved_pairs) if saved_pairs else np.empty((0, 2), np.int64),
                   distances=np.concatenate(saved_distances) if saved_distances else np.empty(0),
                   angles_degrees=np.concatenate(saved_angles) if saved_angles else np.empty(0),
                   ambiguous_sample_mask=endpoints)
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


def run_ambiguity_analysis(source: str | Path, output_dir: str | Path, *,
                            stride: int = 100,
                            color_threshold: float = .01, angle_threshold: float = 20.,
                            export_pairs: bool = False,
                            dominance_fraction: float = .8) -> dict:
    """Read first-pass samples, persist reusable inputs, and write only ambiguity outputs."""
    source = Path(source).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f'Ambiguity input archive does not exist: {source}')
    started = time.perf_counter()
    with np.load(source, allow_pickle=False) as archive:
        names = ('color_residual_samples', 'slope_samples', 'image_index', 'image_paths')
        samples = {name: archive[name] for name in names}
        samples['metadata'] = archive['metadata'] if 'metadata' in archive else np.asarray('{}')
    colors, truth, groups = (samples[key] for key in names[:3])
    if colors.shape != (len(groups), 3) or truth.shape != (len(groups), 2) \
            or not np.issubdtype(groups.dtype, np.integer) or groups.ndim != 1 \
            or np.any(groups < 0) or np.any(groups >= len(samples['image_paths'])) \
            or not np.isfinite(colors).all() or not np.isfinite(truth).all():
        raise ValueError('Invalid or mismatched ambiguity sample arrays')
    indices = select_ambiguity_samples(groups, stride)
    inputs = output_dir / 'ambiguity_inputs.npz'
    if source != inputs:
        save_ambiguity_inputs(inputs, samples, source)
    visualization = output_dir / 'visualization'
    visualization.mkdir(parents=True, exist_ok=True)
    print(f'[ambiguity] {len(colors)} original samples -> {len(indices)} analyzed '
          f'(stride={stride})', flush=True)
    metadata = json.loads(str(np.asarray(samples['metadata']).item()))
    sampling = {
        'method': 'per_image_stride', 'stride': stride,
        'original_sample_count': len(colors), 'analyzed_sample_count': len(indices),
        'retained_fraction': len(indices) / len(colors),
        'original_image_count': len(np.unique(groups)),
        'analyzed_image_count': len(np.unique(groups[indices])),
        'is_sampled': len(indices) != len(colors),
    }
    selected = dict(color_residual_samples=colors[indices], slope_samples=truth[indices],
                    image_index=groups[indices], image_paths=samples['image_paths'])
    from .ambiguity_patterns import analyze_branch_patterns, classify_branch_support
    classify_branch_support(np.empty((0, 2)), dominance_fraction=dominance_fraction)
    patterns_dir = visualization / 'ambiguity_patterns'
    patterns_dir.mkdir(exist_ok=True)
    pair_file = patterns_dir / '.conflicting_pairs.tmp.bin'
    pairs = None
    try:
        with pair_file.open('wb') as stream:
            result = _ambiguities(selected['color_residual_samples'], selected['slope_samples'],
                                   selected['image_index'], color_threshold, angle_threshold,
                                   export_pairs, visualization, pair_sink=stream)
        pairs = (np.memmap(pair_file, dtype=np.int64, mode='r', shape=(result['pair_count'], 2))
                 if result['pair_count'] else np.empty((0, 2), np.int64))
        patterns = analyze_branch_patterns(
            selected, pairs, patterns_dir, color_threshold=color_threshold,
            angle_threshold=angle_threshold, dominance_fraction=dominance_fraction,
            source_sample_index=indices, run_signature=metadata.get('run_signature', ''),
            sampling=sampling)
    finally:
        if isinstance(pairs, np.memmap):
            pairs._mmap.close()
        pair_file.unlink(missing_ok=True)
    mask = result.pop('mask')
    report = {
        'format_version': 1,
        'status': 'completed',
        'source_archive': str(source),
        'inputs_archive': str(inputs),
        'run_signature': metadata.get('run_signature'),
        'sampling': sampling,
        'count_scope': ('exact_within_selected_subset' if len(indices) != len(colors)
                        else 'exact_full_dataset'),
        'full_dataset_pair_count_estimated': False,
        **result,
        'ambiguous_sample_fraction': float(mask.mean()),
        'render_count': 0,
        'artifacts': {
            'inputs': 'ambiguity_inputs.npz',
            'samples_and_mask': 'visualization/ambiguity_analysis.npz',
            'plot': 'visualization/pq_ambiguities.png',
            'branch_classification': 'visualization/ambiguity_patterns/branch_classification.png',
            'branch_continuation': 'visualization/ambiguity_patterns/branch_continuation.png',
            'branch_analysis': 'visualization/ambiguity_patterns/analysis.json',
        },
        'branch_classification': patterns['branch_classification'],
    }
    _write_npz(visualization / 'ambiguity_analysis.npz',
               source_sample_index=indices, color_residual_samples=colors[indices],
               slope_samples=truth[indices], image_index=groups[indices],
               image_paths=samples['image_paths'], ambiguous_sample_mask=mask,
               run_signature=np.asarray(metadata.get('run_signature', '')))
    from .calibration_visualization import _plot_ambiguities, _write_json
    _plot_ambiguities(visualization / 'pq_ambiguities.png', truth[indices],
                      {**report, 'mask': mask})
    report['total_seconds'] = time.perf_counter() - started
    _write_json(visualization / 'ambiguity_summary.json', report)
    print(f'[ambiguity] completed in {report["total_seconds"]:.2f}s; '
          f'{report["ambiguous_sample_count"]}/{len(indices)} ambiguous samples', flush=True)
    return report
