"""Torch geometry kernels used by the analytic CUDA renderer."""
from __future__ import annotations

import math

import numpy as np

from .geometry import camera_frame


def unit_torch(value, torch):
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    # Kernels in the render hot path must not synchronize with the host.  The
    # corresponding configuration vectors are validated before reaching here.
    return value / torch.clamp(norm, min=torch.finfo(value.dtype).tiny)


def camera_rays_torch(config, *, device, dtype, torch):
    camera = config['camera']
    x = torch.arange(camera['width_px'], device=device, dtype=dtype)
    y = torch.arange(camera['height_px'], device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    local = torch.stack(((xx - camera['cx_px']) / camera['fx_px'],
                         (yy - camera['cy_px']) / camera['fy_px'],
                         torch.ones_like(xx)), dim=-1)
    rotation = torch.as_tensor(camera_frame(camera), device=device, dtype=dtype)
    return unit_torch(local @ rotation.T, torch)


def contact_radius_torch(config, *, device, dtype, torch):
    indentation = config['indentation']
    radius = torch.as_tensor(indentation['radius_mm'], device=device, dtype=dtype)
    depth = torch.as_tensor(indentation['depth_mm'], device=device, dtype=dtype)
    return torch.sqrt(torch.clamp(2 * radius * depth - depth * depth, min=0))


def height_normal_torch(x, y, config, *, optical=True, torch):
    surface, indentation = config['surface'], config['indentation']
    x, y = torch.broadcast_tensors(x, y)
    center = torch.as_tensor(indentation['center_xy_mm'], device=x.device, dtype=x.dtype)
    radius = torch.as_tensor(indentation['radius_mm'], device=x.device, dtype=x.dtype)
    depth = torch.as_tensor(indentation['depth_mm'], device=x.device, dtype=x.dtype)
    dx, dy = x - center[0], y - center[1]
    radial2 = dx * dx + dy * dy
    contact2 = torch.clamp(2 * radius * depth - depth * depth, min=0)
    inside = (radial2 < contact2) & (depth > 0)
    root = torch.sqrt(torch.clamp(radius * radius - radial2,
                                  min=torch.finfo(x.dtype).eps))
    sign = float(indentation['direction_z'])
    z = torch.where(inside, sign * (depth - radius + root), torch.zeros_like(x))
    hx = torch.where(inside, -sign * dx / root, torch.zeros_like(x))
    hy = torch.where(inside, -sign * dy / root, torch.zeros_like(x))
    normal = unit_torch(torch.stack((-hx, -hy, torch.ones_like(hx)), dim=-1), torch)
    if optical:
        normal = normal * float(surface['view_side_z'])
    return z, normal, hx, hy, inside


def intersect_surface_torch(origin, direction, config, *, torch, t_min=1e-7):
    """Nearest positive intersection with the finite plane or spherical cap."""
    origin, direction = torch.broadcast_tensors(origin, direction)
    surface, indentation = config['surface'], config['indentation']
    width, height = map(float, surface['size_mm'])
    center_xy = torch.as_tensor(indentation['center_xy_mm'],
                                device=origin.device, dtype=origin.dtype)
    radius = float(indentation['radius_mm'])
    depth = float(indentation['depth_mm'])
    sign = float(indentation['direction_z'])
    contact2 = max(0., 2 * radius * depth - depth * depth)

    dz = direction[..., 2]
    finite_denominator = torch.abs(dz) > torch.finfo(direction.dtype).tiny
    safe_dz = torch.where(finite_denominator, dz, torch.ones_like(dz))
    plane_t = -origin[..., 2] / safe_dz
    plane_t = torch.where(finite_denominator & torch.isfinite(plane_t),
                          plane_t, torch.full_like(plane_t, -1))
    plane_point = origin + plane_t[..., None] * direction
    radial = ((plane_point[..., 0] - center_xy[0]) ** 2
              + (plane_point[..., 1] - center_xy[1]) ** 2)
    valid = (plane_t > t_min) & (radial >= contact2 - 1e-10)
    valid &= ((torch.abs(plane_point[..., 0]) <= width / 2)
              & (torch.abs(plane_point[..., 1]) <= height / 2))
    result = torch.where(valid, plane_t, torch.full_like(plane_t, float('inf')))

    if depth > 0:
        sphere_center = torch.stack((center_xy[0], center_xy[1],
                                     center_xy.new_tensor(sign * (depth - radius))))
        oc = origin - sphere_center
        qa = torch.sum(direction * direction, dim=-1)
        qb = torch.sum(oc * direction, dim=-1)
        qc = torch.sum(oc * oc, dim=-1) - radius * radius
        discriminant = qb * qb - qa * qc
        root = torch.sqrt(torch.clamp(discriminant, min=0))
        safe_qa = torch.clamp(qa, min=torch.finfo(qa.dtype).tiny)
        for candidate in ((-qb - root) / safe_qa, (-qb + root) / safe_qa):
            point = origin + candidate[..., None] * direction
            sphere_radial = ((point[..., 0] - sphere_center[0]) ** 2
                             + (point[..., 1] - sphere_center[1]) ** 2)
            good = ((discriminant >= 0) & (candidate > t_min)
                    & (sphere_radial <= contact2 + 1e-10)
                    & (sign * (point[..., 2] - sphere_center[2]) >= 0)
                    & (torch.abs(point[..., 0]) <= width / 2)
                    & (torch.abs(point[..., 1]) <= height / 2))
            result = torch.minimum(result,
                                   torch.where(good, candidate,
                                               torch.full_like(candidate, float('inf'))))
    return result


def project_world_torch(points, config, *, torch):
    camera = config['camera']
    rotation = torch.as_tensor(camera_frame(camera), device=points.device,
                               dtype=points.dtype)
    position = torch.as_tensor(camera['position_mm'], device=points.device,
                               dtype=points.dtype)
    camera_xyz = (points - position) @ rotation
    z = camera_xyz[..., 2]
    safe_z = torch.where(torch.abs(z) > torch.finfo(z.dtype).tiny,
                         z, torch.ones_like(z))
    u = camera['fx_px'] * camera_xyz[..., 0] / safe_z + camera['cx_px']
    v = camera['fy_px'] * camera_xyz[..., 1] / safe_z + camera['cy_px']
    return torch.stack((u, v), dim=-1), z > 0

