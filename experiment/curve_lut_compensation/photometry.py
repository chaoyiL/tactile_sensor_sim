"""Continuous material-point photometry using the analytic renderer kernels.

This module deliberately evaluates batches of requested material points only. It
does not construct a camera image or synthesize pixel samples. Positions use
millimetres; input normals use the project's geometric-normal convention.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Any

import numpy as np

from utils.geometry import camera_frame
from utils.geometry_gpu import pack_pose_parameters
from utils.light_distribution import analytic_light_samples
from utils.renderer_gpu import require_torch, _dtype
from utils.surface_coordinates import analytic_parameter, normal_direction


@dataclass(frozen=True)
class PhotometryResponse:
    """Pointwise renderer outputs; RGB values are linear-light, before sRGB encoding."""

    display_rgb: np.ndarray
    camera_linear_rgb: np.ndarray
    radiance_w_m2_sr: np.ndarray
    camera_factor: np.ndarray
    valid: np.ndarray
    active_light_mask: np.ndarray


def encode_srgb8(display_rgb: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    """Encode Reinhard/display-linear RGB to the renderer's rounded sRGB8 values."""
    # Preserve the originating renderer precision. The archived source uses
    # float32, while supported numerical fixtures may use float64.
    raw = np.asarray(display_rgb)
    if dtype is None:
        dtype = raw.dtype if raw.dtype in (np.dtype('float32'), np.dtype('float64')) else np.dtype('float32')
    else:
        dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('sRGB8 encoding supports float32 or float64 display values')
    rgb = np.asarray(raw, dtype=dtype)
    if rgb.shape[-1:] != (3,):
        raise ValueError('display_rgb must have a final RGB axis of length 3')
    zero, one = dtype.type(0.), dtype.type(1.)
    clipped = np.clip(rgb, zero, one).astype(dtype, copy=False)
    encoded = np.where(clipped <= dtype.type(.0031308), dtype.type(12.92) * clipped,
                       dtype.type(1.055) * np.power(clipped, dtype.type(1. / 2.4))
                       - dtype.type(.055)).astype(dtype, copy=False)
    return np.rint(dtype.type(255.) * encoded).astype(np.uint8)


def _same_except_pose(base: dict[str, Any], state: dict[str, Any]) -> bool:
    left, right = copy.deepcopy(base), copy.deepcopy(state)
    for value in (left, right):
        indentation = value.get('indentation', {})
        indentation['depth_mm'] = 0.
        if 'center_uv' in indentation:
            indentation['center_uv'] = [0., 0.]
        if 'center_xy_mm' in indentation:
            indentation['center_xy_mm'] = [0., 0.]
    return left == right


class PhotometryModel:
    """GPU-first analytic point photometry with curve-aware moving lights/shadows.

    ``config`` is the source physical snapshot for one fixed surface curvature.
    ``state_config`` may change only the indentation centre and depth, allowing
    contact and no-contact states to share the same captured optics and lights.
    CUDA is mandatory unless ``device='cpu'`` is explicitly requested for a
    numerical fixture; there is no automatic device fallback.
    """

    def __init__(self, config: dict[str, Any], *, device: str = 'cuda') -> None:
        if not isinstance(config, dict):
            raise TypeError('config must be a physical configuration dictionary')
        if not isinstance(device, str) or not device:
            raise ValueError('device must be a nonempty torch device string')
        self.torch = require_torch()
        torch = self.torch
        try:
            self.device = torch.device(device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f'Invalid photometry device {device!r}') from exc
        if self.device.type not in ('cuda', 'cpu'):
            raise ValueError("PhotometryModel supports only CUDA or explicit CPU fixtures")
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA was requested for continuous photometry, but PyTorch CUDA is unavailable')
            try:
                torch.empty(1, device=self.device).add_(1)
                torch.cuda.synchronize(self.device)
            except Exception as exc:
                raise RuntimeError(f'CUDA device {self.device} cannot execute photometry tensors') from exc

        self.config = copy.deepcopy(config)
        self.radius_mm = self.config['surface'].get('bending', {}).get('radius_mm')
        self.width_mm, self.height_mm = map(float, self.config['surface']['size_mm'])
        self.geometry_dtype = _dtype(torch, self.config.get('render', {}).get('geometry_dtype', 'float64'))
        self.radiometry_dtype = _dtype(torch, self.config.get('render', {}).get('radiometry_dtype', 'float32'))
        self.radiometry_numpy_dtype = np.float64 if self.radiometry_dtype == torch.float64 else np.float32
        self.light_chunk_size = int(self.config.get('render', {}).get('light_chunk_size', 16))
        if self.light_chunk_size < 1:
            raise ValueError('render.light_chunk_size must be positive')
        self.shadows = self.config['render']['shadows']
        if self.config['render']['tone_map'] != 'reinhard':
            raise ValueError('curve_lut_compensation requires the source Reinhard tone map')
        if self.config['surface']['view_side_z'] != -1 or normal_direction(self.config) != -1:
            raise ValueError('Continuous compensation photometry requires view_side_z=-1 and direction_normal=-1')

        point_parts, normal_parts, power_parts, exponent_parts = [], [], [], []
        for strip in self.config['lights']:
            points, normal, powers = analytic_light_samples(strip, self.config)
            point_parts.append(points)
            normal_parts.append(np.broadcast_to(normal, points.shape))
            power_parts.append(powers)
            exponent_parts.append(np.full(len(points), strip['cosine_exponent'], dtype=np.float64))
        if not point_parts:
            raise ValueError('config.lights must contain at least one light sample')
        points = np.concatenate(point_parts).astype(np.float64, copy=False)
        normals = np.concatenate(normal_parts).astype(np.float64, copy=False)
        powers = np.concatenate(power_parts).astype(np.float64, copy=False)
        exponents = np.concatenate(exponent_parts).astype(np.float64, copy=False)
        self.light_points = torch.as_tensor(points, device=self.device, dtype=self.geometry_dtype)
        self.light_normals = torch.as_tensor(normals, device=self.device, dtype=self.geometry_dtype)
        self.light_powers = torch.as_tensor(powers, device=self.device, dtype=self.radiometry_dtype)
        self.light_exponents = torch.as_tensor(exponents, device=self.device, dtype=self.geometry_dtype)
        self.light_normalization = (self.light_exponents + 1.) / (2. * math.pi)
        self.unit_light_exponents = bool(np.all(exponents == 1.))
        self.reflectance = torch.as_tensor(self.config['surface']['reflectance_rgb'],
                                           device=self.device, dtype=self.radiometry_dtype)
        camera = self.config['camera']
        self.camera_position = torch.as_tensor(camera['position_mm'], device=self.device,
                                               dtype=self.geometry_dtype)
        self.camera_forward = torch.as_tensor(camera_frame(camera)[:, 2], device=self.device,
                                              dtype=self.geometry_dtype)
        self.camera_response = torch.as_tensor(camera['response_rgb_3x3'], device=self.device,
                                               dtype=self.radiometry_dtype)
        self.exposure = float(self.config['render']['exposure_scale'])
        self.pinhole_half_width_mm = analytic_parameter(self.config, 'camera', 'pinhole_half_width_mm')
        self.focal_length_mm = analytic_parameter(self.config, 'camera', 'focal_length_mm')
        self.execution = dict(device=str(self.device), geometry_dtype=str(self.geometry_dtype).split('.')[-1],
                              radiometry_dtype=str(self.radiometry_dtype).split('.')[-1],
                              light_sample_count=len(points), light_chunk_size=self.light_chunk_size,
                              shadows=bool(self.shadows), tone_map='reinhard',
                              model='continuous_material_point_H_pt')

    @classmethod
    def from_snapshot(cls, config: dict[str, Any], *, device: str = 'cuda') -> 'PhotometryModel':
        """Build a model from a saved physical configuration snapshot."""
        return cls(config, device=device)

    def _validate_state(self, state_config: dict[str, Any] | None) -> dict[str, Any]:
        state = self.config if state_config is None else state_config
        if not isinstance(state, dict):
            raise TypeError('state_config must be a physical configuration dictionary')
        if not _same_except_pose(self.config, state):
            raise ValueError('state_config changes physical parameters beyond indentation centre/depth')
        return state

    def _pose(self, state: dict[str, Any]):
        torch = self.torch
        if self.radius_mm is None:
            return torch.as_tensor(pack_pose_parameters(state), device=self.device,
                                   dtype=self.geometry_dtype)
        from utils.curve_geometry_gpu import pack_curve_pose
        return torch.as_tensor(pack_curve_pose(state), device=self.device,
                               dtype=self.geometry_dtype)

    def _shade(self, points, normal_optical, state, *, need_linear_matrix: bool = False):
        torch = self.torch
        from utils.renderer_kernels import light_block_factor
        from utils.curve_renderer_kernels import curve_light_block_factor

        pose = self._pose(state)
        shadow_origin = (points[:, None, :] + normal_optical[:, None, :] * 1e-6) \
            if self.shadows else None
        irradiance = torch.zeros((len(points), 3), device=self.device,
                                 dtype=self.radiometry_dtype)
        active_blocks = []
        coefficient = (torch.zeros((len(points), 3, 3), device=self.device,
                                   dtype=self.radiometry_dtype) if need_linear_matrix else None)
        if self.radius_mm is None:
            factor_function = light_block_factor
            meta = dict(width=self.width_mm, height=self.height_mm,
                        radius=float(state['indentation']['radius_mm']),
                        sign=float(normal_direction(state)), shadows=self.shadows,
                        unit_exponents=self.unit_light_exponents)
        else:
            factor_function = curve_light_block_factor
            meta = dict(width=self.width_mm, height=self.height_mm,
                        radius=float(state['indentation']['radius_mm']),
                        curve_radius=float(self.radius_mm), shadows=self.shadows,
                        unit_exponents=self.unit_light_exponents)

        for start in range(0, len(self.light_points), self.light_chunk_size):
            stop = min(start + self.light_chunk_size, len(self.light_points))
            light = self.light_points[start:stop]
            emission = self.light_normals[start:stop]
            powers = self.light_powers[start:stop]
            exponent = self.light_exponents[start:stop]
            normalization = self.light_normalization[start:stop]
            args = (points, normal_optical, shadow_origin, light, emission, powers,
                    exponent, normalization, pose)
            factor, visible = factor_function(*args, **meta)
            contribution = torch.sum(factor[..., None] * powers[None, :, :], dim=1)
            irradiance += contribution

            if need_linear_matrix:
                vector = light[None, :, :] - points[:, None, :]
                distance2 = torch.sum(vector * vector, dim=-1)
                direction = vector / torch.sqrt(torch.clamp(
                    distance2, min=torch.finfo(distance2.dtype).tiny))[..., None]
                cosine_in = torch.sum(normal_optical[:, None, :] * direction, dim=-1)
                cosine_emit = torch.clamp(torch.sum(-direction * emission[None, :, :], dim=-1), min=0.)
                active = visible & (cosine_in > 0.) & (cosine_emit > 0.)
                active_blocks.append(active)
                emission_power = cosine_emit if self.unit_light_exponents \
                    else cosine_emit ** exponent[None, :]
                base_weight = normalization[None, :] * emission_power
                base_weight = base_weight * visible / torch.clamp(
                    distance2, min=torch.finfo(distance2.dtype).tiny)
                base_weight = torch.where(active, base_weight, torch.zeros_like(base_weight))
                base_weight = base_weight.to(self.radiometry_dtype)
                direction_rad = direction.to(self.radiometry_dtype)
                block = torch.sum(base_weight[..., None, None]
                                  * powers[None, :, :, None]
                                  * direction_rad[..., None, :], dim=1)
                coefficient += block
            else:
                # Activity is kept for diagnostics even when only H_pt is requested.
                active_blocks.append(visible & (factor > 0.))

        reflectance = self.reflectance
        radiance = irradiance * reflectance[None, :] / math.pi * 1e6
        camera_ray = points - self.camera_position
        camera_ray = camera_ray / torch.clamp(torch.linalg.vector_norm(
            camera_ray, dim=-1, keepdim=True), min=torch.finfo(camera_ray.dtype).tiny)
        cos4 = torch.clamp(camera_ray @ self.camera_forward, min=0.) ** 4
        area_over_focal2 = (2. * self.pinhole_half_width_mm / self.focal_length_mm) ** 2
        camera_factor = (cos4 * area_over_focal2).to(self.radiometry_dtype)
        sensor = radiance * camera_factor[:, None]
        camera_linear = sensor @ self.camera_response.T * self.exposure
        camera_linear = torch.clamp(camera_linear, min=0.)
        display = camera_linear / (1. + camera_linear)
        display = torch.clamp(display, 0., 1.)

        camera_matrix = None
        if need_linear_matrix:
            # B = eta*g*M*(1e6/pi)*diag(rho)*sum_j(b_j P_j omega_j^T).
            row_scale = reflectance * (1e6 / math.pi)
            radiance_matrix = coefficient * row_scale[None, :, None]
            camera_matrix = torch.matmul(self.camera_response[None, :, :], radiance_matrix)
            camera_matrix = camera_matrix * (camera_factor * self.exposure)[:, None, None]

        return (radiance, camera_linear, display, camera_factor,
                torch.cat(active_blocks, dim=1) if active_blocks else
                torch.zeros((len(points), 0), device=self.device, dtype=torch.bool),
                camera_matrix)

    def _evaluate_tensors(self, points_mm, geometric_normals, *, state_config=None,
                          need_linear_matrix: bool = False):
        torch = self.torch
        state = self._validate_state(state_config)
        points_np = np.asarray(points_mm, dtype=np.float64)
        normals_np = np.asarray(geometric_normals, dtype=np.float64)
        if points_np.ndim != 2 or points_np.shape[1] != 3 or normals_np.shape != points_np.shape:
            raise ValueError('points_mm and geometric_normals must both have shape (N,3)')
        if not np.isfinite(points_np).all() or not np.isfinite(normals_np).all():
            raise ValueError('points_mm and geometric_normals must be finite')
        norms = np.linalg.norm(normals_np, axis=-1)
        if len(norms) and not np.allclose(norms, 1., rtol=0., atol=1e-6):
            raise ValueError('geometric_normals must be unit length')
        points = torch.as_tensor(points_np, device=self.device, dtype=self.geometry_dtype)
        normals = torch.as_tensor(normals_np, device=self.device, dtype=self.geometry_dtype)
        # The renderer maps geometric surface normals to its optical normal using this sign.
        normal_optical = normals * float(state['surface']['view_side_z'])
        return self._shade(points, normal_optical, state, need_linear_matrix=need_linear_matrix)

    @staticmethod
    def _host(value):
        return value.detach().cpu().numpy()

    def evaluate(self, points_mm: np.ndarray, geometric_normals: np.ndarray, *,
                 state_config: dict[str, Any] | None = None) -> PhotometryResponse:
        """Evaluate H_pt ingredients at N requested points, never a full image."""
        points = np.asarray(points_mm)
        normals = np.asarray(geometric_normals)
        if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
            raise ValueError('points_mm and geometric_normals must both have shape (N,3)')
        if not len(points):
            empty3 = np.empty((0, 3), dtype=np.float32)
            return PhotometryResponse(empty3, empty3.copy(), empty3.copy(),
                                      np.empty(0, np.float32), np.empty(0, bool),
                                      np.empty((0, len(self.light_points)), bool))
        radiance, camera_linear, display, camera_factor, active, _ = self._evaluate_tensors(
            points, normals, state_config=state_config)
        display_host = self._host(display)
        radiance_host = self._host(radiance)
        camera_host = self._host(camera_factor)
        valid = np.isfinite(points).all(axis=-1) & np.isfinite(normals).all(axis=-1)
        valid &= np.isfinite(display_host).all(axis=-1) & np.isfinite(radiance_host).all(axis=-1)
        valid &= np.isfinite(camera_host)
        camera_forward = camera_frame(self.config['camera'])[:, 2]
        camera_position = np.asarray(self.config['camera']['position_mm'], dtype=np.float64)
        valid &= ((points - camera_position) @ camera_forward) > 0.
        return PhotometryResponse(display_rgb=display_host,
                                  camera_linear_rgb=self._host(camera_linear),
                                  radiance_w_m2_sr=radiance_host,
                                  camera_factor=camera_host,
                                  valid=valid,
                                  active_light_mask=self._host(active))

