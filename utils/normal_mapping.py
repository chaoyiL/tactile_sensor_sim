"""Signed linear-RGB residual LUT construction and lookup.

The numerical rules mirror chaoyiL/recon_stable commit
e02f95f119aa42d5aaf1ebae1f2304d60f045e8c.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


UPSTREAM_COMMIT = 'e02f95f119aa42d5aaf1ebae1f2304d60f045e8c'


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Not JSON serializable: {type(value)}')


def slopes_to_normals(slopes: np.ndarray) -> np.ndarray:
    values = np.asarray(slopes, np.float64)
    normals = np.concatenate((-values, np.ones((*values.shape[:-1], 1))), axis=-1)
    return normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-15)


@dataclass(frozen=True)
class NormalCalibration:
    slopes: np.ndarray
    variances: np.ndarray
    color_min: np.ndarray
    color_max: np.ndarray
    sigma_ref2: float
    original_valid: np.ndarray | None = None
    sample_counts: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        slopes = np.asarray(self.slopes, np.float32)
        variances = np.asarray(self.variances, np.float32)
        color_min = np.asarray(self.color_min, np.float32)
        color_max = np.asarray(self.color_max, np.float32)
        if slopes.ndim != 4 or slopes.shape[0] < 2 or slopes.shape[-1] != 2 \
                or slopes.shape[:3] != (slopes.shape[0],) * 3:
            raise ValueError('slopes must be a cubic NxNxNx2 lookup table')
        if variances.shape != slopes.shape[:3]:
            raise ValueError('variances must match the first three LUT dimensions')
        if color_min.shape != (3,) or color_max.shape != (3,) \
                or not np.isfinite(color_min).all() or not np.isfinite(color_max).all() \
                or np.any(color_max <= color_min) or np.any(color_min > 0) \
                or np.any(color_max < 0):
            raise ValueError('color_min/color_max must be finite ranges containing zero')
        if not np.isfinite(slopes).all() or not np.isfinite(variances).all() \
                or np.any(variances < 0) or not np.isfinite(self.sigma_ref2) \
                or self.sigma_ref2 <= 0:
            raise ValueError('Calibration contains invalid numeric values')
        if self.original_valid is not None \
                and np.asarray(self.original_valid).shape != variances.shape:
            raise ValueError('original_valid has an invalid shape')
        if self.sample_counts is not None \
                and np.asarray(self.sample_counts).shape != variances.shape:
            raise ValueError('sample_counts has an invalid shape')
        if not isinstance(self.metadata, dict):
            raise ValueError('LUT metadata must be a mapping')
        object.__setattr__(self, 'slopes', slopes)
        object.__setattr__(self, 'variances', variances)
        object.__setattr__(self, 'color_min', color_min)
        object.__setattr__(self, 'color_max', color_max)
        object.__setattr__(self, 'sigma_ref2', float(self.sigma_ref2))
        object.__setattr__(self, 'metadata', dict(self.metadata))

    @property
    def size(self) -> int:
        return int(self.slopes.shape[0])

    def save(self, path: str | Path, **metadata: Any) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        values: dict[str, Any] = {
            'format_version': np.asarray(5, np.int32),
            'color_residual_mode': np.asarray('signed'),
            'slopes': self.slopes,
            'variances': self.variances,
            'color_min': self.color_min,
            'color_max': self.color_max,
            'sigma_ref2': np.asarray(self.sigma_ref2, np.float32),
        }
        if self.original_valid is not None:
            values['original_valid'] = np.asarray(self.original_valid, np.bool_)
        if self.sample_counts is not None:
            values['sample_counts'] = np.asarray(self.sample_counts, np.int32)
        combined = dict(self.metadata)
        combined.update(metadata)
        for key, value in metadata.items():
            if key in values or key == 'metadata':
                raise ValueError(f'Metadata must not overwrite LUT field {key}')
            array = np.asarray(value) if not isinstance(value, dict) else None
            values[key] = (array if array is not None and not array.dtype.hasobject
                           else np.asarray(json.dumps(value, ensure_ascii=False,
                                                      default=_json_value)))
        values['metadata'] = np.asarray(json.dumps(combined, ensure_ascii=False,
                                                   default=_json_value))
        temporary = output.with_suffix(output.suffix + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **values)
        temporary.replace(output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> 'NormalCalibration':
        with np.load(Path(path).expanduser(), allow_pickle=False) as data:
            version = int(data['format_version']) if 'format_version' in data else 0
            if version not in (4, 5):
                raise ValueError(f'Unsupported normal calibration version: {version}')
            if 'color_residual_mode' not in data \
                    or str(data['color_residual_mode']) != 'signed':
                raise ValueError('Calibration does not declare signed color residuals')
            metadata = json.loads(str(data['metadata'])) if 'metadata' in data else {}
            return cls(data['slopes'], data['variances'], data['color_min'],
                       data['color_max'], float(data['sigma_ref2']),
                       data['original_valid'] if 'original_valid' in data else None,
                       data['sample_counts'] if 'sample_counts' in data else None,
                       metadata)


def build_normal_lut(colors: np.ndarray, slopes: np.ndarray, *, size: int = 64,
                     maximum_rms_angle_degrees: float = 5.) -> NormalCalibration:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError('LUT fitting requires the calibration extra: install scipy') from exc
    color = np.asarray(colors, np.float64).reshape(-1, 3)
    slope = np.asarray(slopes, np.float64).reshape(-1, 2)
    if color.shape[0] != slope.shape[0]:
        raise ValueError('Color and slope sample counts differ')
    finite = np.all(np.isfinite(color), axis=1) & np.all(np.isfinite(slope), axis=1)
    color, slope = color[finite], slope[finite]
    if len(color) < 1:
        raise ValueError('Color and slope samples must be nonempty and finite')
    if type(size) is not int or size < 2 or not np.isfinite(maximum_rms_angle_degrees) \
            or maximum_rms_angle_degrees <= 0:
        raise ValueError('LUT size and angular threshold must be positive')

    color_min = np.minimum(np.min(color, axis=0), 0)
    color_max = np.maximum(np.max(color, axis=0), 0)
    span = color_max - color_min
    if np.any(span <= 0):
        raise ValueError('Every color residual channel must span a nonzero range')
    coordinates = np.rint(np.clip((color - color_min) / span, 0, 1)
                            * (size - 1)).astype(np.int32)
    flat_nodes = np.ravel_multi_index(coordinates.T, (size, size, size))
    order = np.argsort(flat_nodes)
    flat_nodes, slope = flat_nodes[order], slope[order]

    lut = np.zeros((size, size, size, 2), np.float64)
    variance = np.zeros((size, size, size), np.float64)
    sample_counts = np.zeros((size, size, size), np.int32)
    original_valid = np.zeros((size, size, size), np.bool_)
    maximum_rms = np.deg2rad(maximum_rms_angle_degrees)
    starts = np.r_[0, np.flatnonzero(np.diff(flat_nodes)) + 1]
    ends = np.r_[starts[1:], flat_nodes.size]
    for start, end in zip(starts, ends, strict=True):
        values = slope[start:end]
        normals = slopes_to_normals(values)
        mean = np.sum(normals, axis=0)
        mean /= np.maximum(np.linalg.norm(mean), 1e-12)
        angles = np.arccos(np.clip(normals @ mean, -1, 1))
        median = np.median(angles)
        mad = 1.4826 * np.median(np.abs(angles - median))
        kept = angles <= median + 3 * mad + 1e-12
        values, normals = values[kept], normals[kept]
        mean_normal = np.sum(normals, axis=0)
        mean_normal /= np.maximum(np.linalg.norm(mean_normal), 1e-12)
        rms = np.sqrt(np.mean(np.arccos(np.clip(normals @ mean_normal, -1, 1)) ** 2))
        if rms > maximum_rms:
            continue
        node = np.unravel_index(int(flat_nodes[start]), (size, size, size))
        node_mean = np.mean(values, axis=0)
        lut[node] = node_mean
        variance[node] = np.mean(np.sum((values - node_mean) ** 2, axis=1))
        sample_counts[node] = len(values)
        original_valid[node] = True

    valid_coordinates = np.argwhere(original_valid)
    if not valid_coordinates.size:
        raise ValueError('No LUT node satisfies the angular consistency threshold')
    sigma_ref2 = max(float(np.median(variance[original_valid])), 1e-8)
    missing = np.argwhere(~original_valid)
    tree = cKDTree(valid_coordinates / (size - 1))
    neighbor_count = min(8, len(valid_coordinates))
    for start in range(0, len(missing), 65536):
        target = missing[start:start + 65536]
        distances, neighbors = tree.query(target / (size - 1), k=neighbor_count)
        if neighbor_count == 1:
            distances, neighbors = distances[:, None], neighbors[:, None]
        weights = 1 / np.maximum(distances, 1e-12) ** 2
        weights /= np.sum(weights, axis=1, keepdims=True)
        source = valid_coordinates[neighbors]
        source_slopes = lut[source[..., 0], source[..., 1], source[..., 2]]
        filled = np.sum(weights[..., None] * source_slopes, axis=1)
        source_variance = variance[source[..., 0], source[..., 1], source[..., 2]]
        filled_variance = np.sum(weights * (source_variance + np.sum(
            (source_slopes - filled[:, None]) ** 2, axis=-1)), axis=1)
        lut[target[:, 0], target[:, 1], target[:, 2]] = filled
        variance[target[:, 0], target[:, 1], target[:, 2]] = filled_variance

    zero = (size - 1) * np.clip((0 - color_min) / span, 0, 1)
    lower = np.floor(zero).astype(int)
    upper = np.minimum(lower + 1, size - 1)
    for red in {lower[0], upper[0]}:
        for green in {lower[1], upper[1]}:
            for blue in {lower[2], upper[2]}:
                lut[red, green, blue] = 0
                variance[red, green, blue] = sigma_ref2
    return NormalCalibration(lut.astype(np.float32), variance.astype(np.float32),
                             color_min.astype(np.float32), color_max.astype(np.float32),
                             sigma_ref2, original_valid, sample_counts)


def lookup_slopes_raw(color_residual: np.ndarray, calibration: NormalCalibration,
                      valid_mask: np.ndarray | None = None):
    colors = np.asarray(color_residual, np.float32)
    if colors.shape[-1:] != (3,):
        raise ValueError('Color residual must end in RGB')
    valid = np.isfinite(colors).all(axis=-1)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, bool)
        if mask.shape != valid.shape:
            raise ValueError('Color valid mask shape mismatch')
        valid &= mask
    scale = calibration.color_max - calibration.color_min
    colors = np.where(valid[..., None], colors, 0)
    normalized = np.clip((colors - calibration.color_min) / scale, 0, 1)
    coordinate = normalized * (calibration.size - 1)
    lower = np.floor(coordinate).astype(np.int32)
    upper = np.minimum(lower + 1, calibration.size - 1)
    fraction = coordinate - lower
    slopes = np.zeros((*colors.shape[:-1], 2), np.float32)
    variance = np.zeros(colors.shape[:-1], np.float32)
    corner_values, corner_variances, corner_weights = [], [], []
    for ar in (0, 1):
        ir = np.where(ar, upper[..., 0], lower[..., 0])
        wr = np.where(ar, fraction[..., 0], 1 - fraction[..., 0])
        for ag in (0, 1):
            ig = np.where(ag, upper[..., 1], lower[..., 1])
            wg = np.where(ag, fraction[..., 1], 1 - fraction[..., 1])
            for ab in (0, 1):
                ib = np.where(ab, upper[..., 2], lower[..., 2])
                wb = np.where(ab, fraction[..., 2], 1 - fraction[..., 2])
                weight = (wr * wg * wb).astype(np.float32)
                value = calibration.slopes[ir, ig, ib]
                corner_values.append(value)
                corner_variances.append(calibration.variances[ir, ig, ib])
                corner_weights.append(weight)
                slopes += weight[..., None] * value
    for weight, value, node_variance in zip(
            corner_weights, corner_values, corner_variances, strict=True):
        variance += weight * (node_variance + np.sum((value - slopes) ** 2, axis=-1))
    confidence = 1 / (1 + variance / calibration.sigma_ref2)
    return (np.where(valid[..., None], slopes, 0).astype(np.float32),
            np.where(valid, variance, 0).astype(np.float32),
            np.where(valid, confidence, 0).astype(np.float32))


def find_ambiguous_color_pairs(colors: np.ndarray, slopes: np.ndarray, *,
                               color_distance_threshold: float = .01,
                               normal_angle_threshold_degrees: float = 20.):
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError('Ambiguity analysis requires scipy') from exc
    colors = np.asarray(colors, np.float64)
    slopes = np.asarray(slopes, np.float64)
    if colors.ndim != 2 or colors.shape[1] != 3 or slopes.shape != (len(colors), 2):
        raise ValueError('Expected paired Nx3 colors and Nx2 slopes')
    if not np.isfinite(colors).all() or not np.isfinite(slopes).all():
        raise ValueError('Ambiguity samples must be finite')
    if (not np.isfinite([color_distance_threshold,
                         normal_angle_threshold_degrees]).all()
            or color_distance_threshold < 0
            or not 0 <= normal_angle_threshold_degrees <= 180):
        raise ValueError('Color threshold must be nonnegative; normal angle must be in [0, 180]')
    pairs = cKDTree(colors).query_pairs(color_distance_threshold, output_type='ndarray')
    normals = slopes_to_normals(slopes)
    first, second = normals[pairs[:, 0]], normals[pairs[:, 1]]
    angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(first, second), axis=-1),
                                   np.sum(first * second, axis=-1)))
    selected = angles > normal_angle_threshold_degrees
    pairs = pairs[selected]
    distances = np.linalg.norm(colors[pairs[:, 0]] - colors[pairs[:, 1]], axis=-1)
    return pairs, distances, angles[selected]

