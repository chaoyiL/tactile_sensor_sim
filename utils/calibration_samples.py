"""Material-grid sampling and analytic sphere labels for LUT calibration."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time

import numpy as np

from .calibration_scan import ScanPose, closed_axis
from .geometry_gpu import height_normal_torch, intersect_surface_torch
from .surface_coordinates import require_flat


COORDINATE_KIND = 'reference_arclength'
FRAME_CONVENTION = 'st_grid_columns_rows'


@dataclass(frozen=True)
class MaterialGrid:
    x_mm: np.ndarray
    y_mm: np.ndarray
    xy_mm: np.ndarray
    st_mm: np.ndarray
    chart_id: str

    @property
    def shape(self):
        return self.xy_mm.shape[:2]


@dataclass
class CalibrationReference:
    config: dict
    sensor: object
    rgb8: object
    linear_rgb: object
    image_valid: object
    gt: dict
    grid_color: object
    grid_valid: object
    grid_uv: object


@dataclass
class CalibrationFrame:
    pose: ScanPose
    config: dict
    sensor: object
    rgb8: object
    linear_rgb: object
    image_valid: object
    gt: dict
    uv_current: object
    color_residual_grid: object
    photometric_valid_mask: object
    slope_grid: object
    sample_valid_mask: object
    contact_mask: object
    material_grid_index: object
    render_seconds: float
    sampling_seconds: float

    @property
    def sample_count(self) -> int:
        return int(self.sample_valid_mask.sum().item())


def build_material_grid(config: dict) -> MaterialGrid:
    require_flat(config, 'LUT material-grid sampling')
    width, height = map(float, config['surface']['size_mm'])
    step_x, step_y = map(float, config['calibration']['material_grid']['step_xy_mm'])
    if not all(math.isfinite(value) and value > 0 for value in (step_x, step_y)):
        raise ValueError('Material-grid steps must be positive and finite')
    xs = np.asarray(closed_axis(-width / 2, width / 2, step_x), np.float64)
    ys = np.asarray(closed_axis(-height / 2, height / 2, step_y), np.float64)
    x, y = np.meshgrid(xs, ys)
    xy = np.stack((x, y), axis=-1)
    st = xy + np.asarray([width / 2, height / 2])
    digest = hashlib.sha256()
    digest.update(np.asarray([width, height, step_x, step_y], '<f8').tobytes())
    digest.update(xs.astype('<f8').tobytes())
    digest.update(ys.astype('<f8').tobytes())
    return MaterialGrid(xs, ys, xy, st, 'flat-' + digest.hexdigest()[:20])


def bilinear_sample_torch(field, xy, source_valid, *, torch):
    """Upstream-compatible four-neighbour bilinear sampling on CUDA tensors."""
    height, width = field.shape[:2]
    finite = torch.isfinite(xy).all(dim=-1)
    safe = torch.where(finite[..., None], xy, torch.zeros_like(xy))
    valid = (finite & (safe[..., 0] >= 0) & (safe[..., 1] >= 0)
             & (safe[..., 0] < width - 1) & (safe[..., 1] < height - 1))
    safe_x = torch.clamp(safe[..., 0], 0, width - 1)
    safe_y = torch.clamp(safe[..., 1], 0, height - 1)
    x = torch.clamp(torch.floor(safe_x).to(torch.long), 0, width - 2)
    y = torch.clamp(torch.floor(safe_y).to(torch.long), 0, height - 2)
    fx = (safe_x - x).to(field.dtype)[..., None]
    fy = (safe_y - y).to(field.dtype)[..., None]
    a, b = field[y, x], field[y, x + 1]
    c, d = field[y + 1, x], field[y + 1, x + 1]
    valid &= (torch.isfinite(a).all(dim=-1) & torch.isfinite(b).all(dim=-1)
              & torch.isfinite(c).all(dim=-1) & torch.isfinite(d).all(dim=-1))
    if source_valid is not None:
        valid &= (source_valid[y, x] & source_valid[y, x + 1]
                  & source_valid[y + 1, x] & source_valid[y + 1, x + 1])
    values = (1 - fy) * ((1 - fx) * a + fx * b) \
        + fy * ((1 - fx) * c + fx * d)
    return torch.where(valid[..., None], values, torch.zeros_like(values)), valid


def _grid_tensors(context, grid: MaterialGrid):
    cached = context.material_grid_cache.get(grid.chart_id)
    if cached is not None:
        return cached
    torch = context.torch
    xy = torch.as_tensor(grid.xy_mm, device=context.device, dtype=context.geometry_dtype)
    rows, columns = grid.shape
    yy = torch.arange(rows, device=context.device, dtype=torch.int32)[:, None].expand(rows, columns)
    xx = torch.arange(columns, device=context.device, dtype=torch.int32)[None, :].expand(rows, columns)
    indices = torch.stack((yy, xx), dim=-1)
    context.material_grid_cache[grid.chart_id] = (xy, indices)
    return xy, indices


def render_reference(context, grid: MaterialGrid) -> CalibrationReference:
    torch = context.torch
    config = context.config_for_pose()
    config['indentation']['depth_mm'] = 0.
    sensor, _, gt, config = context.render_tensors(config=config)
    rgb8 = context.encode_display(sensor, config)
    linear = context.inverse_srgb(rgb8)
    image_valid = gt['mask'] & torch.all(rgb8 < 255, dim=-1)
    xy, _ = _grid_tensors(context, grid)
    points = torch.cat((xy, torch.zeros((*grid.shape, 1), device=context.device,
                                       dtype=context.geometry_dtype)), dim=-1)
    uv, in_front = context.project_world(points, config)
    color, sampled = bilinear_sample_torch(linear, uv, image_valid, torch=torch)
    return CalibrationReference(config, sensor, rgb8, linear, image_valid, gt,
                                color, sampled & in_front, uv)


def sample_pose(context, reference: CalibrationReference, grid: MaterialGrid,
                pose: ScanPose) -> CalibrationFrame:
    torch = context.torch
    torch.cuda.synchronize(context.device)
    render_started = time.perf_counter()
    sensor, _, gt, config = context.render_tensors(pose=pose)
    torch.cuda.synchronize(context.device)
    render_seconds = time.perf_counter() - render_started
    sampling_started = time.perf_counter()
    rgb8 = context.encode_display(sensor, config)
    linear = context.inverse_srgb(rgb8)
    image_valid = gt['mask'] & torch.all(rgb8 < 255, dim=-1)
    xy, grid_index = _grid_tensors(context, grid)
    z, optical_normal, hx, hy, inside = height_normal_torch(
        xy[..., 0], xy[..., 1], config, optical=True, torch=torch)
    points = torch.cat((xy, z[..., None]), dim=-1)
    uv, in_front = context.project_world(points, config)
    observed, observed_valid = bilinear_sample_torch(linear, uv, image_valid, torch=torch)

    vector = points - context.camera_origin
    distance = torch.linalg.vector_norm(vector, dim=-1)
    direction = vector / torch.clamp(distance[..., None],
                                     min=torch.finfo(distance.dtype).tiny)
    first_hit = intersect_surface_torch(context.camera_origin, direction, config, torch=torch)
    visible = torch.isclose(first_hit, distance, rtol=1e-7, atol=1e-6)
    facing = torch.sum(optical_normal * -direction, dim=-1) > 0
    residual = observed - reference.grid_color
    photo = (reference.grid_valid & observed_valid & in_front & visible & facing
             & torch.isfinite(residual).all(dim=-1))

    margin = float(config['calibration']['sampling'].get('contact_margin_mm', 0.))
    inner_radius = pose.contact_radius_mm - margin
    dx = xy[..., 0] - pose.center_xy_mm[0]
    dy = xy[..., 1] - pose.center_xy_mm[1]
    contact = inside & (dx * dx + dy * dy < inner_radius * inner_radius)
    slopes = torch.stack((hx, hy), dim=-1)
    reference_dot = 1 / torch.sqrt(1 + hx * hx + hy * hy)
    sample_valid = (contact & photo & (reference_dot > .1)
                    & torch.isfinite(slopes).all(dim=-1))
    residual = torch.where(photo[..., None], residual, torch.zeros_like(residual))
    torch.cuda.synchronize(context.device)
    sampling_seconds = time.perf_counter() - sampling_started
    return CalibrationFrame(pose, config, sensor, rgb8, linear, image_valid, gt,
                            uv, residual, photo, slopes, sample_valid, contact,
                            grid_index, render_seconds, sampling_seconds)


def compact_frame_samples(frame: CalibrationFrame, grid: MaterialGrid) -> dict[str, np.ndarray]:
    valid = frame.sample_valid_mask
    count = int(valid.sum().item())
    if not count:
        return {
            'color_residual_samples': np.empty((0, 3), np.float32),
            'slope_samples': np.empty((0, 2), np.float32),
            'material_coordinates': np.empty((0, 2), np.float64),
            'st_samples': np.empty((0, 2), np.float32),
            'material_grid_index': np.empty((0, 2), np.int32),
        }
    selected_index = frame.material_grid_index[valid].detach().cpu().numpy()
    rows, columns = selected_index[:, 0], selected_index[:, 1]
    return {
        'color_residual_samples': frame.color_residual_grid[valid].detach().cpu().numpy().astype(np.float32),
        'slope_samples': frame.slope_grid[valid].detach().cpu().numpy().astype(np.float32),
        'material_coordinates': grid.xy_mm[rows, columns].astype(np.float64),
        'st_samples': grid.st_mm[rows, columns].astype(np.float32),
        'material_grid_index': selected_index.astype(np.int32),
    }


def image_space_diagnostics(context, reference: CalibrationReference,
                            frame: CalibrationFrame):
    """Return residual, truth slopes, photometric validity, and paired validity."""
    torch = context.torch
    positions = frame.gt['position_mm']
    current_valid = frame.image_valid & torch.isfinite(positions).all(dim=-1)
    safe_positions = torch.where(current_valid[..., None], positions,
                                 torch.zeros_like(positions))
    plane_points = safe_positions.clone()
    plane_points[..., 2] = 0
    uv_reference, in_front = context.project_world(plane_points, reference.config)
    background, background_valid = bilinear_sample_torch(
        reference.linear_rgb, uv_reference, reference.image_valid, torch=torch)
    residual = frame.linear_rgb - background
    x, y = safe_positions[..., 0], safe_positions[..., 1]
    _, _, hx, hy, inside = height_normal_torch(
        x, y, frame.config, optical=False, torch=torch)
    dx = x - frame.pose.center_xy_mm[0]
    dy = y - frame.pose.center_xy_mm[1]
    margin = float(frame.config['calibration']['sampling'].get('contact_margin_mm', 0.))
    inner = frame.pose.contact_radius_mm - margin
    contact = inside & (dx * dx + dy * dy < inner * inner)
    photometric_valid = (current_valid & in_front & background_valid
                         & torch.isfinite(residual).all(dim=-1))
    slopes = torch.stack((hx, hy), dim=-1)
    paired_valid = (photometric_valid & contact & torch.isfinite(slopes).all(dim=-1)
                    & (1 / torch.sqrt(1 + hx * hx + hy * hy) > .1))
    residual = torch.where(photometric_valid[..., None], residual,
                           torch.zeros_like(residual))
    slopes = torch.where(paired_valid[..., None], slopes, torch.zeros_like(slopes))
    return residual, slopes, photometric_valid, paired_valid


def calibration_declarations(grid: MaterialGrid) -> dict:
    return {
        'coordinate_kind': COORDINATE_KIND,
        'coordinate_chart_id': grid.chart_id,
        'coordinate_chart_storage': 'embedded_flat_axes',
        'frame_convention': FRAME_CONVENTION,
        'residual_domain': 'material_grid',
        'color_space': 'linear_rgb',
        'image_encoding': 'srgb8_roundtrip',
    }

