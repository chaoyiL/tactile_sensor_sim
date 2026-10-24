"""Curve-aware lighting factor compiled and captured independently of the plane."""
from __future__ import annotations

import torch
from .curve_geometry_gpu import intersect_curve_torch


def curve_light_block_factor(point, normal, shadow_origin, light, emission, powers,
                             exponent, normalization, pose_parameters, *,
                             width, height, radius, curve_radius, shadows, unit_exponents):
    vector = light[None, :, :] - point[:, None, :]
    distance2 = torch.sum(vector * vector, -1)
    distance = torch.sqrt(torch.clamp(distance2, min=torch.finfo(distance2.dtype).tiny))
    direction = vector / distance[..., None]
    cosine_in = torch.clamp(torch.sum(normal[:, None, :] * direction, -1), min=0)
    cosine_emit = torch.clamp(torch.sum(-direction * emission[None, :, :], -1), min=0)
    visible = torch.ones_like(cosine_in, dtype=torch.bool)
    if shadows:
        hit, _ = intersect_curve_torch(shadow_origin, direction, pose_parameters,
                                       width=width, height=height, radius=radius,
                                       curve_radius=curve_radius, torch=torch)
        visible = hit >= distance - 1e-5
    emission_power = cosine_emit if unit_exponents else cosine_emit ** exponent[None, :]
    factor = normalization[None, :] * emission_power
    factor *= ((cosine_emit > 0) * cosine_in * visible
               / torch.clamp(distance2, min=torch.finfo(distance2.dtype).tiny))
    return factor.to(powers.dtype), visible


def shade_curve_light_block(*args, **meta):
    factor, visible = curve_light_block_factor(*args, **meta)
    return torch.sum(factor[..., None] * args[5][None, :, :], dim=1), visible
