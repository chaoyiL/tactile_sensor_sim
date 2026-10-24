"""Raw-sample, fixed-radius disc coverage; no angular clustering or subsampling.

The fixed-node model and the actual trilinear runtime are deliberately separate.
Only the former has A(tau) as its empirical lower bound.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from utils.normal_mapping import NormalCalibration, lookup_slopes_raw, slopes_to_normals

try:
    from numba import njit
    COVERAGE_BACKEND = 'numba_float64_angular_sweep'
except ImportError:
    COVERAGE_BACKEND = 'python_float64_angular_sweep'

    def njit(**kwargs):
        return lambda function: function


METRIC_VERSION = 'lut_single_value_incompatibility_v1'
BOUNDARY_EPS = 1e-12


@njit(cache=True)
def _disc_sweep(points, weights, radius):
    """Exact weighted angular sweep over all boundary anchors, O(U^2 log U).

An optimal disc can be translated until a data point is on its boundary.
Each other point then defines a closed interval of admissible centre angles.
Unique locations carry the sum of ALL original sample weights.
"""
    size = len(points)
    best_weight = -1.
    best_x, best_y = points[0, 0], points[0, 1]
    full = 2 * math.pi
    for i in range(size):
        angles = np.empty(2 * size, np.float64)
        changes = np.empty(2 * size, np.float64)
        count = 0
        current = weights[i]
        for j in range(size):
            if i == j:
                continue
            dx, dy = points[j, 0] - points[i, 0], points[j, 1] - points[i, 1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > 2 * radius:
                continue
            half = math.acos(min(1., distance / (2 * radius)))
            direction = math.atan2(dy, dx)
            start, end = (direction - half) % full, (direction + half) % full
            if start > end:
                current += weights[j]
            angles[count], changes[count] = start, weights[j]
            angles[count + 1], changes[count + 1] = end, -weights[j]
            count += 2
        if current > best_weight:
            best_weight = current
            best_x, best_y = points[i, 0] + radius, points[i, 1]
        order = np.argsort(angles[:count])
        k = 0
        while k < count:
            angle = angles[order[k]]
            end = k + 1
            # Closed intervals: coincident entries precede exits, including tangencies.
            while end < count and angles[order[end]] - angle <= 1e-13:
                end += 1
            for index in range(k, end):
                if changes[order[index]] > 0:
                    current += changes[order[index]]
            if current > best_weight + 1e-14:
                best_weight = current
                best_x = points[i, 0] + radius * math.cos(angle)
                best_y = points[i, 1] + radius * math.sin(angle)
            for index in range(k, end):
                if changes[order[index]] < 0:
                    current += changes[order[index]]
            k = end
    return best_x, best_y


def inside_disc(points: np.ndarray, center: np.ndarray, tau: float) -> np.ndarray:
    """Closed Euclidean slope disc; tau=0 means exactly identical slopes."""
    radius = tau + BOUNDARY_EPS * max(1., tau) if tau > 0 else 0.
    return np.linalg.norm(points - center, axis=-1) <= radius


def maximum_disc(points: np.ndarray, weights: np.ndarray, tau: float):
    """Return the optimal centre and covered weight (not restricted to sample centres)."""
    points = np.asarray(points, np.float64)
    weights = np.asarray(weights, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) \
            or weights.shape != (len(points),) or not np.isfinite(points).all() \
            or not np.isfinite(weights).all() or np.any(weights <= 0) \
            or not math.isfinite(tau) or tau < 0:
        raise ValueError('Disc coverage requires finite Nx2 points, positive weights and tau >= 0')
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    if tau == 0 or len(unique) == 1:
        index = int(np.argmax(mass))
        return unique[index].copy(), float(mass[index])
    # Exact success certificate; this only avoids work, it never drops samples.
    middle = (unique.min(axis=0) + unique.max(axis=0)) / 2
    if inside_disc(unique, middle, tau).all():
        return middle, float(mass.sum())
    center = np.asarray(_disc_sweep(unique, mass, float(tau)))
    covered = float(mass[inside_disc(unique, center, tau)].sum())
    return center, covered


def bin_samples(colors: np.ndarray, lut: NormalCalibration) -> np.ndarray:
    """Same nearest-node assignment as build_normal_lut (np.rint, clip, RGB order)."""
    colors = np.asarray(colors, np.float64)
    if colors.ndim != 2 or colors.shape[1] != 3 or not np.isfinite(colors).all():
        raise ValueError('Expected finite Nx3 signed linear-RGB residuals')
    lower, upper = lut.color_min.astype(np.float64), lut.color_max.astype(np.float64)
    normalized = np.clip((colors - lower) / (upper - lower), 0, 1)
    nodes = np.rint(normalized * (lut.size - 1)).astype(np.int32)
    return np.ravel_multi_index(nodes.T, (lut.size,) * 3)


def sample_weights(pose_index: np.ndarray, weighting: str) -> np.ndarray:
    """Weights are normalized within each evaluated split, not within each bin."""
    if weighting == 'sample':
        return np.ones(len(pose_index), np.float64)
    if weighting != 'pose_equal':
        raise ValueError(f'Unknown weighting: {weighting}')
    _, inverse, counts = np.unique(pose_index, return_inverse=True, return_counts=True)
    return 1. / counts[inverse]


def split_samples(image_index: np.ndarray, image_keys: np.ndarray,
                  lut: NormalCalibration, split: str):
    """Only training_only LUTs have a saved, truly held-out internal pose split."""
    if split not in ('auto', 'all'):
        raise ValueError('split must be auto or all')
    all_samples = np.ones(len(image_index), bool)
    if split == 'all' or lut.metadata.get('validation_status') != 'training_only':
        return all_samples, ~all_samples, 'all_samples_in_sample_no_independent_validation'
    training = lut.metadata.get('training_images', [])
    validation = lut.metadata.get('validation_images', [])
    if not training or not validation or set(training) & set(validation):
        raise ValueError('training_only LUT requires nonempty disjoint training/validation_images')
    keys = image_keys[image_index]
    train_mask, validation_mask = np.isin(keys, training), np.isin(keys, validation)
    if not np.all(train_mask | validation_mask) or not train_mask.any() or not validation_mask.any():
        raise ValueError('Saved pose split does not cover the sample archive')
    return train_mask, validation_mask, 'saved_whole_pose_holdout'


def evaluate_quality(colors: np.ndarray, slopes: np.ndarray, image_index: np.ndarray,
                     lut: NormalCalibration, taus: list[float], *,
                     weighting: str = 'sample', progress: Callable | None = None):
    """Evaluate all supplied raw samples. Return JSON summary and sparse per-bin arrays."""
    colors, slopes = np.asarray(colors, np.float64), np.asarray(slopes, np.float64)
    image_index = np.asarray(image_index)
    if slopes.shape != (len(colors), 2) or image_index.shape != (len(colors),) \
            or not len(colors) or not np.isfinite(slopes).all():
        raise ValueError('Expected nonempty paired finite colors, slopes and pose indices')
    if not taus or any(not math.isfinite(tau) or tau < 0 for tau in taus):
        raise ValueError('taus must be finite nonnegative slope tolerances')
    flat = bin_samples(colors, lut)
    order = np.argsort(flat, kind='stable')
    ids, starts, counts = np.unique(flat[order], return_index=True, return_counts=True)
    weights = sample_weights(image_index, weighting)
    total = float(weights.sum())
    nodes = lut.slopes.reshape(-1, 2)
    # Match the existing runtime exactly, but bound its temporary memory usage.
    runtime_errors = np.empty(len(colors), np.float64)
    for start in range(0, len(colors), 65536):
        end = start + 65536
        prediction = lookup_slopes_raw(colors[start:end], lut)[0]
        runtime_errors[start:end] = np.linalg.norm(prediction - slopes[start:end], axis=1)
    arrays = dict(bin_ids=ids, coordinates=np.array(np.unravel_index(ids, (lut.size,) * 3)).T,
                  sample_count=counts, weight=np.zeros(len(ids)),
                  pose_count=np.zeros(len(ids), np.int32),
                  unique_slope_count=np.zeros(len(ids), np.int32),
                  slope_min=np.zeros((len(ids), 2)), slope_max=np.zeros((len(ids), 2)),
                  theta_min_degrees=np.zeros(len(ids)), theta_max_degrees=np.zeros(len(ids)),
                  normal_rms_degrees=np.zeros(len(ids)),
                  optimal_center=np.zeros((len(taus), len(ids), 2)),
                  incompatibility=np.zeros((len(taus), len(ids))),
                  node_error=np.zeros((len(taus), len(ids))),
                  runtime_error=np.zeros((len(taus), len(ids))))
    arrays['original_valid'] = (lut.original_valid.ravel()[ids] if lut.original_valid is not None
                                else np.full(len(ids), False))
    for b, (start, count) in enumerate(zip(starts, counts, strict=True)):
        indices = order[start:start + count]
        values, mass = slopes[indices], weights[indices]
        bin_weight = mass.sum()
        arrays['weight'][b] = bin_weight
        arrays['pose_count'][b] = len(np.unique(image_index[indices]))
        arrays['unique_slope_count'][b] = len(np.unique(values, axis=0))
        arrays['slope_min'][b], arrays['slope_max'][b] = values.min(axis=0), values.max(axis=0)
        normals = slopes_to_normals(values)
        theta = np.degrees(np.arctan(np.linalg.norm(values, axis=1)))
        arrays['theta_min_degrees'][b], arrays['theta_max_degrees'][b] = theta.min(), theta.max()
        mean_normal = (normals * mass[:, None]).sum(axis=0)
        mean_normal /= np.linalg.norm(mean_normal)
        angles = np.degrees(np.arccos(np.clip(normals @ mean_normal, -1, 1)))
        arrays['normal_rms_degrees'][b] = np.sqrt(np.sum(mass * angles**2) / bin_weight)
        for t, tau in enumerate(taus):
            center, covered = maximum_disc(values, mass, tau)
            # A node is another feasible centre. Also guards numerical sweep ties.
            node_covered = float(mass[inside_disc(values, nodes[ids[b]], tau)].sum())
            if node_covered > covered:
                center, covered = nodes[ids[b]], node_covered
            arrays['optimal_center'][t, b] = center
            arrays['incompatibility'][t, b] = np.clip(1 - covered / bin_weight, 0, 1)
            arrays['node_error'][t, b] = np.clip(1 - node_covered / bin_weight, 0, 1)
            radius = tau + BOUNDARY_EPS * max(1., tau) if tau > 0 else 0.
            arrays['runtime_error'][t, b] = np.sum(mass * (runtime_errors[indices] > radius)) / bin_weight
        if progress and (b == 0 or (b + 1) % 500 == 0 or b + 1 == len(ids)):
            progress(b + 1, len(ids))
    bin_weight = arrays['weight'] / total
    curves = []
    for t, tau in enumerate(taus):
        ambiguity = float(arrays['incompatibility'][t] @ bin_weight)
        node_error = float(arrays['node_error'][t] @ bin_weight)
        curves.append(dict(tau=float(tau), incompatibility=ambiguity,
                           node_error=node_error, node_excess_error=node_error - ambiguity,
                           runtime_error=float(arrays['runtime_error'][t] @ bin_weight)))
    theta = np.degrees(np.arctan(np.linalg.norm(slopes, axis=1)))
    phi = np.degrees(np.arctan2(-slopes[:, 1], -slopes[:, 0]))
    coverage, theta_edges, phi_edges = np.histogram2d(
        theta, phi, bins=(np.linspace(0, 90, 19), np.linspace(-180, 180, 37)))
    arrays.update(normal_coverage_count=coverage, theta_edges_degrees=theta_edges,
                  phi_edges_degrees=phi_edges, taus=np.asarray(taus))
    summary = dict(sample_count=len(colors), pose_count=len(np.unique(image_index)),
                   occupied_bin_count=len(ids), singleton_bin_count=int(np.sum(counts == 1)),
                   singleton_sample_fraction=float(weights[order][np.repeat(counts == 1, counts)].sum() / total),
                   maximum_bin_count=int(counts.max()),
                   out_of_range_fraction=float(weights[((colors < lut.color_min)
                                                        | (colors > lut.color_max)).any(axis=1)].sum() / total),
                   theta_range_degrees=[float(theta.min()), float(theta.max())],
                   slope_min=slopes.min(axis=0).tolist(), slope_max=slopes.max(axis=0).tolist(),
                   curves=curves)
    return summary, arrays


def evaluate_holdout(colors, slopes, image_index, lut, taus, fitted, *, weighting='sample'):
    """Transfer fitted centres; do NOT choose centres using validation labels."""
    ids = bin_samples(colors, lut)
    weights = sample_weights(image_index, weighting)
    total = weights.sum()
    position = np.searchsorted(fitted['bin_ids'], ids)
    safe = np.minimum(position, len(fitted['bin_ids']) - 1)
    known = (position < len(fitted['bin_ids'])) & (fitted['bin_ids'][safe] == ids)
    nodes = lut.slopes.reshape(-1, 2)[ids]
    runtime = np.empty_like(slopes, dtype=np.float64)
    for start in range(0, len(colors), 65536):
        runtime[start:start + 65536] = lookup_slopes_raw(colors[start:start + 65536], lut)[0]
    rows = []
    for t, tau in enumerate(taus):
        covered = known & inside_disc(slopes, fitted['optimal_center'][t, safe], tau)
        rows.append(dict(tau=float(tau),
                         transferred_disc_failure_or_unseen=float(weights[~covered].sum() / total),
                         transferred_disc_known_bin_error=(float(weights[known & ~covered].sum() / weights[known].sum())
                                                           if known.any() else None),
                         node_error=float(weights[~inside_disc(slopes, nodes, tau)].sum() / total),
                         runtime_error=float(weights[~inside_disc(slopes, runtime, tau)].sum() / total)))
    return dict(sample_count=len(colors), pose_count=len(np.unique(image_index)),
                unseen_bin_fraction=float(weights[~known].sum() / total),
                out_of_range_fraction=float(weights[((colors < lut.color_min)
                                                     | (colors > lut.color_max)).any(axis=1)].sum() / total),
                curves=rows)
