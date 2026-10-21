"""Full-device analytic renderer for PyTorch CUDA tensors."""
from __future__ import annotations

import copy
import math

import numpy as np

from .geometry import camera_frame, strip_samples
from .geometry_gpu import (camera_rays_torch, height_normal_torch,
                           intersect_surface_torch, project_world_torch)


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('GPU rendering requires the gpu extra (PyTorch CUDA)') from exc
    return torch


def _dtype(torch, name: str):
    return {'float32': torch.float32, 'float64': torch.float64}[name]


class AnalyticGpuContext:
    """Reusable CUDA constants for one camera, surface and lighting setup."""

    def __init__(self, config: dict, device: str | None = None):
        torch = require_torch()
        self.torch = torch
        self.base_config = copy.deepcopy(config)
        self.device = torch.device(device or config.get('render', {}).get('device', 'cuda:0'))
        if self.device.type != 'cuda':
            raise ValueError('AnalyticGpuContext requires a CUDA device')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available to PyTorch')
        try:
            torch.empty(1, device=self.device).add_(1)
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            raise RuntimeError(f'CUDA device {self.device} cannot execute tensors') from exc
        render = config.get('render', {})
        self.geometry_dtype = _dtype(torch, render.get('geometry_dtype', 'float64'))
        self.radiometry_dtype = _dtype(torch, render.get('radiometry_dtype', 'float32'))
        self.pixel_chunk_size = int(render.get('pixel_chunk_size', 8192))
        self.light_chunk_size = int(render.get('light_chunk_size', 16))
        self.oom_retry_count = 0
        self.rays = camera_rays_torch(config, device=self.device,
                                      dtype=self.geometry_dtype, torch=torch)
        camera = config['camera']
        self.camera_origin = torch.as_tensor(camera['position_mm'], device=self.device,
                                             dtype=self.geometry_dtype)
        rotation = torch.as_tensor(camera_frame(camera), device=self.device,
                                   dtype=self.geometry_dtype)
        self.camera_forward = rotation[:, 2]
        self.cos4 = (self.rays @ self.camera_forward) ** 4
        self.response = torch.as_tensor(camera['response_rgb_3x3'], device=self.device,
                                        dtype=self.radiometry_dtype)
        points, normals, powers, exponents = [], [], [], []
        for strip in config['lights']:
            samples, normal = strip_samples(strip, config)
            count = len(samples)
            points.append(samples)
            normals.append(np.broadcast_to(normal, (count, 3)))
            powers.append(np.broadcast_to(np.asarray(strip['power_rgb_w'], float) / count,
                                          (count, 3)))
            exponents.append(np.full(count, strip['cosine_exponent'], float))
        self.light_points = torch.as_tensor(np.concatenate(points), device=self.device,
                                            dtype=self.geometry_dtype)
        self.light_normals = torch.as_tensor(np.concatenate(normals), device=self.device,
                                             dtype=self.geometry_dtype)
        self.light_powers = torch.as_tensor(np.concatenate(powers), device=self.device,
                                            dtype=self.radiometry_dtype)
        self.light_exponents = torch.as_tensor(np.concatenate(exponents), device=self.device,
                                               dtype=self.geometry_dtype)
        self.material_grid_cache = {}

    def config_for_pose(self, pose=None, config=None):
        result = copy.deepcopy(config or self.base_config)
        if pose is not None:
            result['indentation']['radius_mm'] = float(
                result['calibration']['sphere_radius_mm'])
            result['indentation']['center_xy_mm'] = list(pose.center_xy_mm)
            result['indentation']['depth_mm'] = float(pose.depth_mm)
        return result

    def camera_ground_truth(self, config: dict):
        torch = self.torch
        t = intersect_surface_torch(self.camera_origin, self.rays, config, torch=torch)
        hit = torch.isfinite(t)
        safe_t = torch.where(hit, t, torch.zeros_like(t))
        positions = self.camera_origin + safe_t[..., None] * self.rays
        _, normals, _, _, _ = height_normal_torch(
            positions[..., 0], positions[..., 1], config, optical=True, torch=torch)
        hit &= torch.sum(normals * -self.rays, dim=-1) > 0
        depth = torch.sum((positions - self.camera_origin) * self.camera_forward, dim=-1)
        nan = torch.full_like(positions, float('nan'))
        return {'mask': hit,
                'position_mm': torch.where(hit[..., None], positions, nan),
                'normal_world': torch.where(hit[..., None], normals, nan),
                'depth_mm': torch.where(hit, depth, torch.full_like(depth, float('nan')))}

    def render_tensors(self, *, pose=None, config=None):
        """Render one frame, reducing chunks and retrying after a CUDA OOM."""
        torch = self.torch
        while True:
            try:
                return self._render_tensors_once(pose=pose, config=config)
            except torch.cuda.OutOfMemoryError:
                if self.pixel_chunk_size <= 1 and self.light_chunk_size <= 1:
                    raise
                self.oom_retry_count += 1
                if self.pixel_chunk_size >= self.light_chunk_size \
                        and self.pixel_chunk_size > 1:
                    self.pixel_chunk_size = max(1, self.pixel_chunk_size // 2)
                else:
                    self.light_chunk_size = max(1, self.light_chunk_size // 2)
                torch.cuda.empty_cache()

    def _render_tensors_once(self, *, pose=None, config=None):
        torch = self.torch
        cfg = self.config_for_pose(pose, config)
        gt = self.camera_ground_truth(cfg)
        mask = gt['mask']
        positions = gt['position_mm'][mask]
        normals = gt['normal_world'][mask]
        irradiance = torch.zeros((len(positions), 3), device=self.device,
                                 dtype=self.radiometry_dtype)
        pixel_step = self.pixel_chunk_size
        light_step = self.light_chunk_size
        for pixel_start in range(0, len(positions), pixel_step):
            pixel_stop = min(pixel_start + pixel_step, len(positions))
            point = positions[pixel_start:pixel_stop]
            normal = normals[pixel_start:pixel_stop]
            accumulated = torch.zeros((len(point), 3), device=self.device,
                                      dtype=self.radiometry_dtype)
            for light_start in range(0, len(self.light_points), light_step):
                light_stop = min(light_start + light_step, len(self.light_points))
                light = self.light_points[light_start:light_stop]
                emission = self.light_normals[light_start:light_stop]
                exponent = self.light_exponents[light_start:light_stop]
                vector = light[None, :, :] - point[:, None, :]
                distance2 = torch.sum(vector * vector, dim=-1)
                distance = torch.sqrt(torch.clamp(distance2,
                                                   min=torch.finfo(distance2.dtype).tiny))
                direction = vector / distance[..., None]
                cosine_in = torch.clamp(torch.sum(normal[:, None, :] * direction, dim=-1), min=0)
                cosine_emit = torch.clamp(torch.sum(-direction * emission[None, :, :], dim=-1), min=0)
                visible = torch.ones_like(cosine_in, dtype=torch.bool)
                if cfg['render']['shadows']:
                    origin = point[:, None, :] + normal[:, None, :] * 1e-6
                    hit_t = intersect_surface_torch(origin, direction, cfg, torch=torch)
                    visible = hit_t >= distance - 1e-5
                factor = ((exponent[None, :] + 1) / (2 * math.pi)
                          * cosine_emit ** exponent[None, :])
                factor *= ((cosine_emit > 0) * cosine_in * visible
                           / torch.clamp(distance2, min=torch.finfo(distance2.dtype).tiny))
                accumulated += torch.sum(
                    factor.to(self.radiometry_dtype)[..., None]
                    * self.light_powers[None, light_start:light_stop, :], dim=1)
            irradiance[pixel_start:pixel_stop] = accumulated

        reflectance = torch.as_tensor(cfg['surface']['reflectance_rgb'], device=self.device,
                                      dtype=self.radiometry_dtype)
        radiance = irradiance * reflectance / math.pi
        shape = (*mask.shape, 3)
        radiance_image = torch.zeros(shape, device=self.device, dtype=self.radiometry_dtype)
        radiance_image[mask] = radiance * 1e6
        camera, zemax = cfg['camera'], cfg['zemax']
        area = (2 * float(zemax['pinhole_half_width_mm'])) ** 2
        sensor = (radiance_image * (area / float(camera['focal_length_mm']) ** 2)
                  * self.cos4.to(self.radiometry_dtype)[..., None])
        return sensor, radiance_image, gt, cfg

    def encode_display(self, sensor, config):
        torch = self.torch
        rgb = sensor @ self.response.T * float(config['render']['exposure_scale'])
        rgb = torch.clamp(rgb, min=0)
        if config['render']['tone_map'] == 'reinhard':
            rgb = rgb / (1 + rgb)
        rgb = torch.clamp(rgb, 0, 1)
        srgb = torch.where(rgb <= .0031308, 12.92 * rgb,
                           1.055 * rgb ** (1 / 2.4) - .055)
        return torch.round(255 * srgb).to(torch.uint8)

    def inverse_srgb(self, image):
        torch = self.torch
        srgb = image.to(self.radiometry_dtype) / 255
        return torch.where(srgb <= .04045, srgb / 12.92,
                           ((srgb + .055) / 1.055) ** 2.4)

    def project_world(self, points, config):
        return project_world_torch(points, config, torch=self.torch)


def render_host(config: dict, device: str | None = None):
    torch = require_torch()
    with torch.inference_mode():
        context = AnalyticGpuContext(config, device)
        sensor, radiance, gt, _ = context.render_tensors(config=config)
        host_gt = {key: value.detach().cpu().numpy() for key, value in gt.items()}
        return (sensor.detach().cpu().numpy(), radiance.detach().cpu().numpy(), host_gt)

