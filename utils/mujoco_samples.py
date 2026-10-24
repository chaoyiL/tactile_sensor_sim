"""NumPy material-grid sampling for the fixed-gel Xensim SimpleSensor mode."""
from __future__ import annotations

import numpy as np

from .mujoco_backend import inverse_srgb_rgb8


def bilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray,
                    source_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = field.shape[:2]
    valid = ((x >= 0) & (y >= 0) & (x < width - 1) & (y < height - 1)
             & np.isfinite(x) & np.isfinite(y))
    ix = np.clip(np.floor(np.nan_to_num(x)).astype(np.int64), 0, width - 2)
    iy = np.clip(np.floor(np.nan_to_num(y)).astype(np.int64), 0, height - 2)
    fx = (np.clip(x, 0, width - 1) - ix)[..., None]
    fy = (np.clip(y, 0, height - 1) - iy)[..., None]
    valid &= (source_valid[iy, ix] & source_valid[iy, ix + 1]
              & source_valid[iy + 1, ix] & source_valid[iy + 1, ix + 1])
    result = ((1 - fy) * ((1 - fx) * field[iy, ix] + fx * field[iy, ix + 1])
              + fy * ((1 - fx) * field[iy + 1, ix] + fx * field[iy + 1, ix + 1]))
    return np.where(valid[..., None], result, 0), valid


def grid_image_coordinates(grid, config: dict) -> tuple[np.ndarray, np.ndarray]:
    width_mm, height_mm = config['surface']['size_mm']
    width_px, height_px = config['mujoco']['render_size_wh']
    x = (grid.xy_mm[..., 0] / width_mm + .5) * width_px - .5
    y = (grid.xy_mm[..., 1] / height_mm + .5) * height_px - .5
    return x, y


def reference_grid(frame: dict, grid, config: dict):
    rgb8 = frame['rgb8']
    image_valid = np.all(rgb8 < 255, axis=-1)
    x, y = grid_image_coordinates(grid, config)
    color, valid = bilinear_sample(inverse_srgb_rgb8(rgb8), x, y, image_valid)
    return color, valid


def pose_samples(frame: dict, reference_color: np.ndarray, reference_valid: np.ndarray,
                 grid, config: dict, pose) -> tuple[dict, dict]:
    """Use ideal sphere labels only where MuJoCo's actual depth agrees."""
    rgb8 = frame['rgb8']
    width_mm, height_mm = config['surface']['size_mm']
    width_px, height_px = config['mujoco']['render_size_wh']
    x_mm, y_mm = grid.xy_mm[..., 0], grid.xy_mm[..., 1]
    dx = x_mm - pose.center_xy_mm[0]
    dy = y_mm - pose.center_xy_mm[1]
    rr = dx * dx + dy * dy
    radius = float(config['indentation']['radius_mm'])
    contact_radius = pose.contact_radius_mm
    margin = float(config['calibration']['sampling'].get('contact_margin_mm', 0))
    contact = rr < (contact_radius - margin) ** 2
    root = np.sqrt(np.maximum(radius * radius - rr, 1e-16))
    slope = np.stack((dx / root, dy / root), axis=-1)
    facing = 1 / np.sqrt(1 + np.sum(slope * slope, axis=-1)) > .1
    px = (x_mm / width_mm + .5) * width_px - .5
    py = (y_mm / height_mm + .5) * height_px - .5
    observed, observed_valid = bilinear_sample(
        inverse_srgb_rgb8(rgb8), px, py, np.all(rgb8 < 255, axis=-1))
    valid = contact & facing & observed_valid & reference_valid
    residual = observed - reference_color
    valid &= np.isfinite(residual).all(axis=-1)
    rows, cols = np.nonzero(valid)
    compact = {
        'color_residual_samples': residual[rows, cols].astype(np.float32),
        'slope_samples': slope[rows, cols].astype(np.float32),
        'material_coordinates': grid.xy_mm[rows, cols].astype(np.float64),
        'st_samples': grid.st_mm[rows, cols].astype(np.float32),
        'material_grid_index': np.stack((rows, cols), axis=-1).astype(np.int32),
    }
    diagnostics = {'sample_count': len(rows), 'contact_grid_count': int(contact.sum()),
                   'reference_valid_grid_count': int(reference_valid.sum()),
                   'saturated_pixel_fraction': float(np.mean(np.any(rgb8 == 255, axis=-1)))}
    return compact, diagnostics


def depth_geometry_error(frame: dict, config: dict, pose) -> dict:
    """Independent raster-depth check in the contact interior (millimetres)."""
    actual = np.asarray(frame['depth_input_mm'], np.float64)
    height, width = actual.shape
    width_mm, height_mm = config['surface']['size_mm']
    xs = (np.arange(width) + .5) / width * width_mm - width_mm / 2
    ys = (np.arange(height) + .5) / height * height_mm - height_mm / 2
    x, y = np.meshgrid(xs, ys)
    rr = (x - pose.center_xy_mm[0]) ** 2 + (y - pose.center_xy_mm[1]) ** 2
    radius = config['indentation']['radius_mm']
    expected = pose.depth_mm - radius + np.sqrt(np.maximum(radius * radius - rr, 0))
    inside = rr < (pose.contact_radius_mm - 2 * max(width_mm / width,
                                                     height_mm / height)) ** 2
    if not inside.any():
        raise ValueError('Depth raster too coarse for interior geometry validation')
    error = np.abs(actual[inside] + expected[inside])
    report = {'pixel_count': int(inside.sum()), 'max_abs_mm': float(error.max()),
              'p95_abs_mm': float(np.percentile(error, 95))}
    if report['max_abs_mm'] > .05:
        raise ValueError(f'MuJoCo sphere depth disagrees with analytic label: {report}')
    return report
