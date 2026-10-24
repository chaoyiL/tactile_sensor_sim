"""Pose-equal, reference-area metrics. Color clamping never removes test errors."""
from __future__ import annotations
import itertools
import numpy as np
from utils.normal_mapping import lookup_slopes_raw, slopes_to_normals, NormalCalibration


def support(colors: np.ndarray, lut: NormalCalibration) -> tuple[np.ndarray, np.ndarray]:
    """Range membership and all positive trilinear corners originally calibrated."""
    c = np.asarray(colors, np.float32)
    inside = np.isfinite(c).all(-1) & ((c >= lut.color_min) & (c <= lut.color_max)).all(-1)
    xyz = np.clip((c - lut.color_min) / (lut.color_max - lut.color_min), 0, 1) * (lut.size - 1)
    lo = np.floor(xyz).astype(int)
    hi = np.minimum(lo + 1, lut.size - 1)
    f = xyz - lo
    valid = inside.copy()
    if lut.original_valid is None:
        return inside, np.zeros_like(inside)
    for corner in itertools.product((0, 1), repeat=3):
        index = [hi[:, k] if bit else lo[:, k] for k, bit in enumerate(corner)]
        weight = np.prod([f[:, k] if bit else 1 - f[:, k] for k, bit in enumerate(corner)], axis=0)
        valid &= (weight <= 0) | lut.original_valid[tuple(index)]
    return inside, valid


def weighted_quantile(values, weights, quantile: float) -> float | None:
    if not len(values) or np.sum(weights) <= 0:
        return None
    order = np.argsort(values)
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(values)[order[min(np.searchsorted(cumulative, quantile * cumulative[-1]), len(order)-1)]])


def evaluate_pose(sample: dict, lut: NormalCalibration, indices: np.ndarray | None = None,
                  prediction: np.ndarray | None = None) -> tuple[dict, dict]:
    take = np.arange(len(sample['grid_index'])) if indices is None else np.searchsorted(sample['grid_index'], indices)
    if indices is not None and (np.any(take >= len(sample['grid_index']))
                               or not np.array_equal(sample['grid_index'][take], indices)):
        raise ValueError('Evaluation indices must belong to the sampled material domain')
    c, truth = sample['color_residual_samples'][take], sample['slope_samples'][take]
    area = sample['area_mm2'][take].astype(float)
    if not len(take) or area.sum() <= 0:
        return {'status': 'empty', 'sample_count': 0, 'mse': None}, {}
    pred = lookup_slopes_raw(c, lut)[0] if prediction is None else prediction[take]
    error = pred.astype(float) - truth
    norm = np.linalg.norm(error, axis=-1)
    npred, ngt = slopes_to_normals(pred), slopes_to_normals(truth)
    angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(npred, ngt), axis=-1), np.sum(npred*ngt, axis=-1)))
    inside, original = support(c, lut)
    w = area / area.sum()
    row = dict(status='ok', sample_count=len(take), area_mm2=float(area.sum()),
               mse=float(w @ norm**2), rmse=float(np.sqrt(w @ norm**2)),
               angular_mse=float(w @ angles**2), angular_rmse_degrees=float(np.sqrt(w @ angles**2)),
               bias_p=float(w @ error[:, 0]), bias_q=float(w @ error[:, 1]),
               out_of_range_fraction=float(w @ ~inside), original_support_fraction=float(w @ original),
               p95=weighted_quantile(norm, w, .95))
    return row, dict(error=norm, angle=angles, weight=w, prediction=pred, grid_index=sample['grid_index'][take])


def aggregate(rows: list[dict], details: list[dict], tolerances: list[float]) -> dict:
    failures = [r['pose_id'] for r in rows if r['status'] != 'ok']
    if failures:
        return dict(status='incomplete', failed_poses=failures, pose_count=len(rows), rmse=None)
    if not rows:
        return dict(status='empty', failed_poses=[], pose_count=0, rmse=None)
    error = np.concatenate([d['error'] for d in details])
    angle = np.concatenate([d['angle'] for d in details])
    weight = np.concatenate([d['weight'] / len(rows) for d in details])
    result = dict(status='ok', failed_poses=[], pose_count=len(rows),
                  rmse=float(np.sqrt(np.mean([r['mse'] for r in rows]))),
                  angular_rmse_degrees=float(np.sqrt(np.mean([r['angular_mse'] for r in rows]))),
                  p95=weighted_quantile(error, weight, .95),
                  angular_p95_degrees=weighted_quantile(angle, weight, .95),
                  worst_pose=max(rows, key=lambda r: r['rmse'])['pose_id'],
                  worst_pose_rmse=max(r['rmse'] for r in rows),
                  exceedance={str(t): float(weight @ (error > t)) for t in tolerances})
    for k in ('bias_p', 'bias_q', 'out_of_range_fraction', 'original_support_fraction'):
        result[k] = float(np.mean([r[k] for r in rows]))
    return result


def paired_bootstrap(rows_a: list[dict], rows_b: list[dict], *, repeats: int, seed: int) -> dict:
    """Resample anchor clusters with all depths together; return RMSE(A)-RMSE(B)."""
    if len(rows_a) != len(rows_b) or any(a['pose_id'] != b['pose_id'] for a, b in zip(rows_a, rows_b)):
        raise ValueError('Paired bootstrap requires exactly matching ordered poses')
    if any(r['status'] != 'ok' for r in rows_a + rows_b):
        return {'status': 'incomplete', 'interval_95': None}
    anchors = sorted({r['anchor_id'] for r in rows_a})
    groups = [np.array([i for i, r in enumerate(rows_a) if r['anchor_id'] == anchor]) for anchor in anchors]
    a, b = np.array([r['mse'] for r in rows_a]), np.array([r['mse'] for r in rows_b])
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats)
    for j in range(repeats):
        idx = np.concatenate([groups[k] for k in rng.integers(len(groups), size=len(groups))])
        differences[j] = np.sqrt(a[idx].mean()) - np.sqrt(b[idx].mean())
    return dict(status='ok', difference=float(np.sqrt(a.mean())-np.sqrt(b.mean())),
                interval_95=np.quantile(differences, [.025, .975]).tolist(),
                anchor_count=len(anchors), repeats=repeats, seed=seed)


def mapping_difference(flat_lut: NormalCalibration, curved_lut: NormalCalibration,
                       samples_a: list[dict], samples_b: list[dict]) -> tuple[dict, dict]:
    """Identical physical RGB queries, separately report common-box and strict support."""
    colors, weights = [], []
    if any(not len(s['grid_index']) for s in samples_a + samples_b):
        return {'status': 'empty_pose', 'rms': None}, {}
    for group in (samples_a, samples_b):
        for sample in group:
            colors.append(sample['color_residual_samples'])
            a = sample['area_mm2']
            weights.append(.5 / len(group) * a / a.sum())
    c, w = np.concatenate(colors), np.concatenate(weights)
    ia, sa = support(c, flat_lut)
    ib, sb = support(c, curved_lut)
    box, strict = ia & ib, sa & sb
    delta = lookup_slopes_raw(c, curved_lut)[0] - lookup_slopes_raw(c, flat_lut)[0]
    squared = np.sum(delta.astype(float)**2, -1)
    report = {'status': 'ok', 'meaning': 'function_difference_not_accuracy'}
    for name, mask in (('common_box', box), ('strict_support', strict)):
        mass = float(w[mask].sum())
        report[name] = dict(coverage=mass, rms=float(np.sqrt(w[mask] @ squared[mask] / mass)) if mass else None,
                            sample_count=int(mask.sum()))
    return report, dict(colors=c, weight=w, delta=delta, common_box=box, strict_support=strict)
