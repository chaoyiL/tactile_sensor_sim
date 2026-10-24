"""Reference-material observations and tangent slope labels for analytic surfaces.

The observation encoding and bilinear sampling are shared with planar calibration.
Torch is imported only when a sampler is created, never during configuration checks.
"""
from __future__ import annotations

import copy
import numpy as np

from .calibration_samples import MaterialGrid, build_material_grid, bilinear_sample_torch
from .config import validate_config
from .curve_geometry import sphere_pose
from .surface_coordinates import bending_radius, center_uv, normal_direction

SLOPE_FRAME_ID = 'reference_local_tangent_b_v1'
SAMPLING_VERSION = 'material_paired_srgb8_curve_v1'


def canonical_snapshot(config: dict) -> dict:
    """Convert legacy planar input fields in memory without changing physics."""
    from .surface_coordinates import analytic_parameter
    result = copy.deepcopy(config)
    uv, sign = center_uv(config), normal_direction(config)
    result['indentation'].pop('center_xy_mm', None)
    result['indentation'].pop('direction_z', None)
    result['indentation'].update(center_uv=uv.tolist(), direction_normal=sign)
    result['camera']['pinhole_half_width_mm'] = analytic_parameter(config, 'camera', 'pinhole_half_width_mm')
    result['render']['mesh_step_mm'] = analytic_parameter(config, 'render', 'mesh_step_mm')
    result['surface']['bending'] = {'axis': 'x', 'radius_mm': None, 'direction_z': -1}
    return result


def reference_grid(config: dict) -> MaterialGrid:
    """Use identical reference arclength axes at every curvature."""
    flat = copy.deepcopy(config)
    flat['surface']['bending'] = {'axis': 'x', 'radius_mm': None, 'direction_z': -1}
    return build_material_grid(flat)


def reference_area_weights(grid: MaterialGrid) -> np.ndarray:
    def axis_weights(axis):
        if len(axis) < 2:
            raise ValueError('Material grid requires at least two points on each axis')
        edges = np.r_[axis[0], (axis[1:] + axis[:-1]) / 2, axis[-1]]
        return np.diff(edges)
    return np.outer(axis_weights(grid.y_mm), axis_weights(grid.x_mm))


def tangent_slopes(normal: np.ndarray, reference: np.ndarray,
                   frame: np.ndarray) -> np.ndarray:
    """Return -E.T n / (n0.n); callers must reject nonpositive/small denominators."""
    dot = np.sum(normal * reference, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return -np.einsum('...ij,...i->...j', frame, normal) / dot[..., None]


class MaterialSampler:
    """Reusable renderer plus exact reference chart for one fixed curvature."""

    def __init__(self, config: dict):
        from .renderer_gpu import AnalyticGpuContext, require_torch
        from .curve_renderer_gpu import CurveGpuContext
        validate_config(config, allow_curve=True, allow_contact_tangent=True)
        if normal_direction(config) != -1 or config['surface']['view_side_z'] != -1:
            raise ValueError('Curve LUT sampling requires geometric +Z / optical -Z conventions')
        if config['calibration']['sampling'].get('contact_margin_mm', 0.) != 0:
            raise ValueError('Curve LUT sampling currently requires contact_margin_mm=0; no silent erosion')
        self.config = copy.deepcopy(config)
        self.grid = reference_grid(config)
        self.area = reference_area_weights(self.grid)
        self.torch = torch = require_torch()
        self.device = torch.device(config['render'].get('device', 'cuda:0'))
        self.geometry_dtype = getattr(torch, config['render'].get('geometry_dtype', 'float64'))
        self.radiometry_dtype = getattr(torch, config['render'].get('radiometry_dtype', 'float32'))
        self.context = None
        if self.device.type == 'cuda':
            self.context = (AnalyticGpuContext(config) if bending_radius(config) is None else
                            CurveGpuContext(config, allow_contact_tangent=True))
        elif self.device.type != 'cpu':
            raise ValueError('MaterialSampler supports cpu or cuda only')
        self.response = torch.as_tensor(config['camera']['response_rgb_3x3'],
                                        device=self.device, dtype=self.radiometry_dtype)
        self.camera_origin = torch.as_tensor(config['camera']['position_mm'],
                                             device=self.device, dtype=self.geometry_dtype)
        self.xy = torch.as_tensor(self.grid.xy_mm, device=self.device, dtype=self.geometry_dtype)
        self.reference = None

    # Use precisely the existing observation and projection operations on CPU/CUDA.
    def encode_display(self, sensor, config):
        from .renderer_gpu import AnalyticGpuContext
        return AnalyticGpuContext.encode_display(self, sensor, config)

    def inverse_srgb(self, image):
        from .renderer_gpu import AnalyticGpuContext
        return AnalyticGpuContext.inverse_srgb(self, image)

    def project_world(self, points, config):
        from .geometry_gpu import project_world_torch
        return project_world_torch(points, config, torch=self.torch)

    def geometry(self, config: dict):
        torch = self.torch
        s, t = self.xy.unbind(-1)
        R = bending_radius(config)
        if R is None:
            from .geometry_gpu import height_normal_torch
            z, nd, p, q, contact = height_normal_torch(s, t, config, optical=False, torch=torch)
            base = torch.stack((s, t, torch.zeros_like(s)), -1)
            point = torch.stack((s, t, z), -1)
            n0 = torch.zeros_like(point)
            n0[..., 2] = 1
            e1, e2 = torch.zeros_like(point), torch.zeros_like(point)
            e1[..., 0], e2[..., 1] = 1, 1
            return base, point, n0, nd, torch.stack((e1, e2), -1), torch.stack((p, q), -1), contact, -z
        angle = s / R
        n0 = torch.stack((torch.sin(angle), torch.zeros_like(s), torch.cos(angle)), -1)
        base = torch.stack((R * torch.sin(angle), t, -2 * R * torch.sin(angle / 2) ** 2), -1)
        e1 = torch.stack((torch.cos(angle), torch.zeros_like(s), -torch.sin(angle)), -1)
        e2 = torch.zeros_like(base)
        e2[..., 1] = 1
        E = torch.stack((e1, e2), -1)
        center = torch.as_tensor(sphere_pose(config)[2], device=self.device, dtype=self.geometry_dtype)
        radius, depth = config['indentation']['radius_mm'], config['indentation']['depth_mm']
        offset = center - base
        a = torch.sum(offset * -n0, -1)
        b = torch.sum(offset * offset, -1) - radius * radius
        contact = (b < 0) & (depth > 0)
        root = torch.sqrt(torch.clamp(a * a - b, min=0))
        delta = torch.where(contact, -b / torch.clamp(root - a, min=torch.finfo(s.dtype).tiny), 0)
        point = base - delta[..., None] * n0
        nd = torch.where(contact[..., None], -(point - center) / radius, n0)
        dot = torch.sum(n0 * nd, -1)
        slopes = -torch.sum(E * nd[..., :, None], dim=-2) / dot[..., None]
        return base, point, n0, nd, E, slopes, contact, delta

    def _visible(self, points, normals, config):
        torch = self.torch
        vector = points - self.camera_origin
        distance = torch.linalg.vector_norm(vector, dim=-1)
        direction = vector / torch.clamp(distance[..., None], min=torch.finfo(distance.dtype).tiny)
        R = bending_radius(config)
        if R is None:
            from .geometry_gpu import intersect_surface_torch
            first = intersect_surface_torch(self.camera_origin, direction, config, torch=torch)
        else:
            from .curve_geometry_gpu import intersect_curve_torch, pack_curve_pose
            pose = torch.as_tensor(pack_curve_pose(config), device=self.device, dtype=self.geometry_dtype)
            first, _ = intersect_curve_torch(self.camera_origin, direction, pose,
                width=config['surface']['size_mm'][0], height=config['surface']['size_mm'][1],
                radius=config['indentation']['radius_mm'], curve_radius=R, torch=torch)
        return (torch.isclose(first, distance, rtol=1e-7, atol=1e-6),
                torch.sum(-normals * -direction, -1) > 0)

    def draw(self, config: dict) -> dict:
        torch = self.torch
        if self.context is not None:
            sensor, radiance, gt, _ = self.context.render_tensors(config=config)
        else:
            from .renderer_cpu import render_cpu
            sensor, radiance, gt = render_cpu(config)
            sensor = torch.as_tensor(sensor, dtype=self.radiometry_dtype)
            radiance = torch.as_tensor(radiance, dtype=self.radiometry_dtype)
            gt = {k: torch.as_tensor(v) for k, v in gt.items()}
        rgb8 = self.encode_display(sensor, config)
        return dict(sensor=sensor, radiance=radiance, gt=gt, rgb8=rgb8,
                    linear=self.inverse_srgb(rgb8), config=config,
                    image_valid=gt['mask'] & torch.all(rgb8 < 255, dim=-1))

    def set_reference(self, frame: dict) -> None:
        torch = self.torch
        cfg = frame['config']
        if cfg['indentation']['depth_mm'] != 0:
            raise ValueError('Reference must have zero local indentation')
        base, _, normal, *_ = self.geometry(cfg)
        uv, front = self.project_world(base, cfg)
        color, valid = bilinear_sample_torch(frame['linear'], uv, frame['image_valid'], torch=torch)
        visible, facing = self._visible(base, normal, cfg)
        self.reference = dict(color=color, valid=valid & front & visible & facing, frame=frame)

    def make_reference(self) -> dict:
        cfg = copy.deepcopy(self.config)
        cfg['indentation']['depth_mm'] = 0.
        with self.torch.inference_mode():
            frame = self.draw(cfg)
            self.set_reference(frame)
        return frame

    def restore_reference(self, rgb8: np.ndarray, mask: np.ndarray) -> None:
        cfg = copy.deepcopy(self.config)
        cfg['indentation']['depth_mm'] = 0.
        image = self.torch.as_tensor(rgb8, device=self.device)
        valid = self.torch.as_tensor(mask, device=self.device)
        self.set_reference(dict(config=cfg, linear=self.inverse_srgb(image),
                                image_valid=valid & self.torch.all(image < 255, dim=-1)))

    def sample(self, config: dict) -> tuple[dict[str, np.ndarray], dict]:
        if self.reference is None:
            raise ValueError('Create/restore the same-curvature reference before sampling')
        validate_config(config, allow_curve=True, allow_contact_tangent=True)
        torch = self.torch
        with torch.inference_mode():
            frame = self.draw(config)
            base, point, n0, nd, E, slopes, contact, delta = self.geometry(config)
            uv, front = self.project_world(point, config)
            color, sampled = bilinear_sample_torch(frame['linear'], uv, frame['image_valid'], torch=torch)
            visible, facing = self._visible(point, nd, config)
            residual = color - self.reference['color']
            normal_valid = (torch.sum(n0 * nd, -1) > .1) & torch.isfinite(slopes).all(-1)
            photo = self.reference['valid'] & sampled & front & visible & facing & torch.isfinite(residual).all(-1)
            valid = contact & photo & normal_valid
            bits = (contact.to(torch.uint8) + self.reference['valid'].to(torch.uint8) * 2
                    + sampled.to(torch.uint8) * 4 + front.to(torch.uint8) * 8
                    + visible.to(torch.uint8) * 16 + facing.to(torch.uint8) * 32
                    + normal_valid.to(torch.uint8) * 64)
            indices = torch.nonzero(valid.reshape(-1), as_tuple=False)[:, 0]
            def host(value):
                return value.detach().cpu().numpy()
            selected = host(indices)
            result = dict(grid_index=selected, color_residual_samples=host(residual[valid]).astype(np.float32),
                slope_samples=host(slopes[valid]).astype(np.float32),
                material_coordinates=self.grid.xy_mm.reshape(-1, 2)[selected],
                area_mm2=self.area.reshape(-1)[selected],
                reference_normal_samples=host(n0[valid]), tangent_frame_samples=host(E[valid]),
                normal_geometric_world=host(nd[valid]), normal_displacement_mm=host(delta[valid]),
                reference_color_samples=host(self.reference['color'][valid]),
                sample_mask=host(valid), contact_mask=host(contact), validity_bits=host(bits))
        return result, frame

    def execution(self) -> dict:
        if self.context is None:
            return dict(device='cpu', renderer='numpy_float64', observation_dtype=str(self.radiometry_dtype))
        c = self.context
        return dict(device=str(c.device), compile=c.compile_metrics(),
                    cuda_graph_replay_count=c.cuda_graph_replay_count,
                    cuda_graph_fallback_reason=c.cuda_graph_fallback_reason, oom_retry_count=c.oom_retry_count)
