"""NormDiff evaluation for simulated RGB/normal calibration samples.

The paper describes the hierarchy but leaves the normal-neighbour policy,
noise-region shape, and angular weights open.  This module makes those choices
explicit and reports them with the score.
"""
from __future__ import annotations

import time

import numpy as np

from utils.normal_mapping import slopes_to_normals


DEFAULT_NOISE_FRACTION = .03


def _validate_noise(noise_fraction, noise_floor):
    for name, value in (('noise_fraction', noise_fraction), ('noise_floor', noise_floor)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) \
                or not np.isrealobj(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be a finite nonnegative number')
    if noise_fraction > 1:
        raise ValueError('noise_fraction must be in [0, 1]; use 0.03 for +/-3%')


def resolve_noise_settings(config: dict, *, noise_fraction=None, noise_floor=None) -> dict:
    """Resolve experimental noise settings; accept the legacy nested API as well."""
    raw = config.get('norm_diff', config.get('calibration', {}).get('norm_diff', {}))
    if not isinstance(raw, dict):
        raise ValueError('norm_diff must be an object')
    fraction = (raw.get('noise_fraction', DEFAULT_NOISE_FRACTION)
                if noise_fraction is None else noise_fraction)
    floor = raw.get('noise_floor', 0.) if noise_floor is None else noise_floor
    _validate_noise(fraction, floor)
    return dict(noise_fraction=float(fraction), noise_floor=float(floor))


def slopes_to_spherical_degrees(slopes: np.ndarray):
    """Return normals, polar theta [0, 180], and azimuth phi [-180, 180)."""
    normals = slopes_to_normals(slopes)
    theta = np.degrees(np.arccos(np.clip(normals[:, 2], -1, 1)))
    phi = np.degrees(np.arctan2(normals[:, 1], normals[:, 0]))
    phi = np.where(theta <= 1e-12, 0., phi)
    return normals, theta, phi


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64)
    return dict(min=float(values.min()), median=float(np.median(values)),
                mean=float(values.mean()), p95=float(np.percentile(values, 95)),
                max=float(values.max()))


def _normal_grid_anchors(normals, theta, phi, *, theta_bins, phi_bins,
                         theta_ceiling, anchors_per_target):
    """Choose the closest calibration samples to each occupied spherical bin centre."""
    theta_index = np.minimum((theta / theta_ceiling * theta_bins).astype(np.int64),
                             theta_bins - 1)
    phi_index = np.minimum(((phi + 180.) / 360. * phi_bins).astype(np.int64),
                           phi_bins - 1)
    target = theta_index * phi_bins + phi_index
    target_theta = (theta_index + .5) * theta_ceiling / theta_bins
    target_phi = -180. + (phi_index + .5) * 360. / phi_bins
    target_theta_rad, target_phi_rad = np.radians(target_theta), np.radians(target_phi)
    target_normals = np.column_stack((np.sin(target_theta_rad) * np.cos(target_phi_rad),
                                      np.sin(target_theta_rad) * np.sin(target_phi_rad),
                                      np.cos(target_theta_rad)))
    alignment = np.sum(normals * target_normals, axis=1)
    order = np.lexsort((-alignment, target))
    ordered_target = target[order]
    starts = np.r_[0, np.flatnonzero(ordered_target[1:] != ordered_target[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    selected = np.concatenate([order[start:min(end, start + anchors_per_target)]
                               for start, end in zip(starts, ends, strict=True)])
    anchor_targets = target[selected]
    return selected, anchor_targets


def _pose_norm_diff(colors, normals, theta, phi, *, noise_fraction, noise_floor,
                    theta_weight, phi_weight, theta_bins, phi_bins,
                    theta_ceiling, anchors_per_target):
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError('NormDiff requires the calibration extra: install scipy') from exc

    anchors, anchor_targets = _normal_grid_anchors(
        normals, theta, phi, theta_bins=theta_bins, phi_bins=phi_bins,
        theta_ceiling=theta_ceiling, anchors_per_target=anchors_per_target)
    anchor_colors = colors[anchors]
    half_width = noise_fraction * np.maximum(np.abs(anchor_colors), noise_floor)
    radius = np.nextafter(np.max(half_width, axis=1), np.inf)
    tree = cKDTree(colors)
    neighborhoods = tree.query_ball_point(anchor_colors, radius, p=np.inf, eps=0,
                                           workers=1, return_sorted=False)
    candidate_counts = np.empty(len(anchors), np.int64)
    theta_ranges = np.empty(len(anchors), np.float64)
    phi_ranges = np.empty(len(anchors), np.float64)
    for row, (anchor_color, width, candidates) in enumerate(
            zip(anchor_colors, half_width, neighborhoods, strict=True)):
        candidates = np.asarray(candidates, np.int64)
        inside = np.all(np.abs(colors[candidates] - anchor_color) <= width, axis=1)
        candidates = candidates[inside]
        if not len(candidates):
            raise RuntimeError('A color-noise region unexpectedly excluded its own anchor')
        candidate_counts[row] = len(candidates)
        theta_ranges[row] = np.ptp(theta[candidates])
        # Paper-faithful coordinate range. This intentionally does not repair the
        # discontinuity between -180 and +180 degrees.
        phi_ranges[row] = np.ptp(phi[candidates])
    confusion = theta_weight * theta_ranges + phi_weight * phi_ranges
    order = np.argsort(anchor_targets, kind='stable')
    ordered_target = anchor_targets[order]
    starts = np.r_[0, np.flatnonzero(ordered_target[1:] != ordered_target[:-1]) + 1]
    anchors_per_selected_target = np.diff(np.r_[starts, len(order)])
    target_theta_ranges = (np.add.reduceat(theta_ranges[order], starts)
                           / anchors_per_selected_target)
    target_phi_ranges = (np.add.reduceat(phi_ranges[order], starts)
                         / anchors_per_selected_target)
    target_confusion = theta_weight * target_theta_ranges + phi_weight * target_phi_ranges
    return dict(norm_diff=-float(target_confusion.mean()),
                mean_confusion_degrees=float(target_confusion.mean()),
                sample_count=len(colors), target_count=len(starts), anchor_count=len(anchors),
                mean_theta_range_degrees=float(target_theta_ranges.mean()),
                mean_phi_range_degrees=float(target_phi_ranges.mean()),
                candidate_counts=candidate_counts,
                anchor_confusion=confusion)


def compute_norm_diff(colors: np.ndarray, slopes: np.ndarray, image_index: np.ndarray, *,
                      image_keys=None, noise_fraction: float = DEFAULT_NOISE_FRACTION,
                      noise_floor: float = 0.,
                      theta_weight: float = 1., phi_weight: float = 1.,
                      theta_bins: int = 12, phi_bins: int = 24,
                      anchors_per_target: int = 1,
                      theta_ceiling_degrees: float | None = None,
                      progress: bool = False) -> dict:
    """Compute pose-balanced NormDiff using every sample in each pose's candidate pool."""
    colors = np.asarray(colors, np.float64)
    slopes = np.asarray(slopes, np.float64)
    image_index = np.asarray(image_index)
    if colors.ndim != 2 or colors.shape[1] != 3 or slopes.shape != (len(colors), 2) \
            or image_index.shape != (len(colors),):
        raise ValueError('Expected aligned Nx3 colors, Nx2 slopes, and N image indices')
    if not len(colors) or not np.isfinite(colors).all() or not np.isfinite(slopes).all() \
            or not np.issubdtype(image_index.dtype, np.integer):
        raise ValueError('NormDiff inputs must be nonempty, finite, and use integer image indices')
    _validate_noise(noise_fraction, noise_floor)
    if not np.isfinite((theta_weight, phi_weight)).all() \
            or theta_weight < 0 or phi_weight < 0 or theta_weight + phi_weight == 0:
        raise ValueError('Angular weights must be finite and nonnegative with a positive sum')
    if type(theta_bins) is not int or type(phi_bins) is not int \
            or type(anchors_per_target) is not int \
            or min(theta_bins, phi_bins, anchors_per_target) < 1:
        raise ValueError('Grid sizes and anchors_per_target must be positive integers')

    normals, theta, phi = slopes_to_spherical_degrees(slopes)
    observed_max = float(theta.max())
    theta_ceiling = observed_max if theta_ceiling_degrees is None else float(theta_ceiling_degrees)
    if not np.isfinite(theta_ceiling) or theta_ceiling <= 0 or theta_ceiling + 1e-12 < observed_max:
        raise ValueError('theta_ceiling_degrees must cover every observed normal')

    if np.any(image_index < 0):
        raise ValueError('image_index values must be nonnegative')
    if image_keys is not None:
        image_keys = np.asarray(image_keys).astype(str)
        if np.any(image_index >= len(image_keys)):
            raise ValueError('image_index refers beyond image_keys')
    if len(image_index) > 1 and np.any(image_index[1:] < image_index[:-1]):
        order = np.argsort(image_index, kind='stable')
        colors, normals, theta, phi, image_index = (
            values[order] for values in (colors, normals, theta, phi, image_index))

    starts = np.r_[0, np.flatnonzero(image_index[1:] != image_index[:-1]) + 1]
    ends = np.r_[starts[1:], len(image_index)]
    pose_rows, all_candidates, all_confusions = [], [], []
    last_progress = time.perf_counter()
    for number, (start, end) in enumerate(zip(starts, ends, strict=True), 1):
        result = _pose_norm_diff(
            colors[start:end], normals[start:end], theta[start:end], phi[start:end],
            noise_fraction=noise_fraction, noise_floor=noise_floor,
            theta_weight=theta_weight, phi_weight=phi_weight,
            theta_bins=theta_bins, phi_bins=phi_bins, theta_ceiling=theta_ceiling,
            anchors_per_target=anchors_per_target)
        index = int(image_index[start])
        pose_rows.append(dict(image_index=index,
                              image_key=(str(image_keys[index]) if image_keys is not None else None),
                              **{key: value for key, value in result.items()
                                 if key not in ('candidate_counts', 'anchor_confusion')}))
        all_candidates.append(result['candidate_counts'])
        all_confusions.append(result['anchor_confusion'])
        if progress and (time.perf_counter() - last_progress >= 5 or number == len(starts)):
            print(f'[norm-diff] {number}/{len(starts)} poses', flush=True)
            last_progress = time.perf_counter()

    pose_scores = np.asarray([row['norm_diff'] for row in pose_rows])
    pose_theta_ranges = np.asarray([row['mean_theta_range_degrees'] for row in pose_rows])
    pose_phi_ranges = np.asarray([row['mean_phi_range_degrees'] for row in pose_rows])
    candidate_counts = np.concatenate(all_candidates)
    anchor_confusion = np.concatenate(all_confusions)
    return {
        'format_version': 1,
        'metric': 'NormDiff',
        'norm_diff': float(pose_scores.mean()),
        'mean_confusion_degrees': float(-pose_scores.mean()),
        'mean_theta_range_degrees': float(pose_theta_ranges.mean()),
        'mean_phi_range_degrees': float(pose_phi_ranges.mean()),
        'pose_count': len(pose_rows),
        'sample_count': len(colors),
        'target_count': int(sum(row['target_count'] for row in pose_rows)),
        'anchor_count': int(sum(row['anchor_count'] for row in pose_rows)),
        'settings': {
            'aggregation': 'equal anchors within target, equal targets within pose, equal poses',
            'candidate_scope': 'all calibration samples from the same pose',
            'normal_target_policy': 'occupied regular theta-phi bins',
            'anchor_policy': 'nearest samples to each occupied bin centre',
            'theta_bins': theta_bins,
            'phi_bins': phi_bins,
            'theta_ceiling_degrees': theta_ceiling,
            'anchors_per_target': anchors_per_target,
            'noise_region': 'per-channel relative axis-aligned box',
            'noise_fraction': noise_fraction,
            'noise_floor': noise_floor,
            'noise_fraction_definition': 'relative half-width, e.g. 0.03 means +/-3%, total width 6%',
            'noise_half_width_formula': 'noise_fraction * max(abs(anchor_channel), noise_floor)',
            'theta_weight': theta_weight,
            'phi_weight': phi_weight,
            'theta_definition': 'acos(normal_z), degrees',
            'phi_definition': ('atan2(normal_y, normal_x), degrees in [-180, 180); '
                               'undefined pole azimuth is set to zero'),
            'phi_range': 'linear max-minus-min; periodic seam is intentionally not corrected',
        },
        'candidate_count_statistics': _summary(candidate_counts),
        'anchor_confusion_statistics_degrees': _summary(anchor_confusion),
        'pose_norm_diff_statistics': _summary(pose_scores),
        'poses': pose_rows,
    }


def evaluate_lut_noise(colors: np.ndarray, residuals: np.ndarray, slopes: np.ndarray,
                       calibration, *, noise_fraction: float = DEFAULT_NOISE_FRACTION,
                       noise_floor: float = 0., repetitions: int = 3, seed: int = 1274) -> dict:
    """Evaluate a LUT on held-out samples with the same RGB noise box as NormDiff.

    Callers select held-out poses before passing data. The reference remains fixed,
    so a perturbation to observed RGB adds the same delta to the signed residual.
    """
    from utils.normal_mapping import lookup_slopes_raw

    _validate_noise(noise_fraction, noise_floor)
    colors, residuals, slopes = (np.asarray(value, np.float64)
                                 for value in (colors, residuals, slopes))
    count = len(colors)
    if not count or colors.shape != (count, 3) or residuals.shape != (count, 3) \
            or slopes.shape != (count, 2) \
            or not all(np.isfinite(value).all() for value in (colors, residuals, slopes)):
        raise ValueError('Expected nonempty finite held-out RGB, residual, and slope arrays')
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('repetitions must be a positive integer')
    truth = slopes_to_normals(slopes)

    def angular_rms(values):
        squared_error = 0.
        for start in range(0, count, 65536):
            stop = min(start + 65536, count)
            prediction, _, _ = lookup_slopes_raw(values[start:stop], calibration)
            dot = np.sum(slopes_to_normals(prediction) * truth[start:stop], axis=1)
            angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
            squared_error += float(angle @ angle)
        return float(np.sqrt(squared_error / count))

    width = noise_fraction * np.maximum(np.abs(colors), noise_floor)
    baseline_rms = angular_rms(residuals)
    trial_rms = []
    for trial in range(repetitions):
        rng = np.random.default_rng(seed + trial)
        perturbed_residual = residuals + rng.uniform(-1., 1., colors.shape) * width
        trial_rms.append(angular_rms(perturbed_residual))
    return dict(sample_count=count, noise_fraction=float(noise_fraction),
                noise_floor=float(noise_floor), repetitions=repetitions, seed=seed,
                noise_distribution='independent uniform per sample/channel within the NormDiff box',
                reference_noise='reference is fixed; only observed RGB is perturbed',
                baseline_rms_degrees=baseline_rms,
                perturbed_rms_degrees=float(np.sqrt(np.mean(np.square(trial_rms)))),
                replicate_rms_degrees=trial_rms)
