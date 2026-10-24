"""Pure light-block tensor kernel, imported only by the CUDA renderer."""
from __future__ import annotations

import torch

from .geometry_gpu import intersect_surface_parameters_torch


def light_block_factor(point, normal, shadow_origin, light, emission, powers,
                       exponent, normalization, pose_parameters, *,
                       width, height, radius, sign, shadows, unit_exponents):
    vector = light[None, :, :] - point[:, None, :]
    distance2 = torch.sum(vector * vector, dim=-1)
    distance = torch.sqrt(torch.clamp(distance2, min=torch.finfo(distance2.dtype).tiny))
    direction = vector / distance[..., None]
    cosine_in = torch.clamp(torch.sum(normal[:, None, :] * direction, dim=-1), min=0)
    cosine_emit = torch.clamp(torch.sum(-direction * emission[None, :, :], dim=-1), min=0)
    visible = torch.ones_like(cosine_in, dtype=torch.bool)
    if shadows:
        hit_t = intersect_surface_parameters_torch(
            shadow_origin, direction, pose_parameters, width=width, height=height,
            radius=radius, sign=sign, torch=torch)
        visible = hit_t >= distance - 1e-5
    emission_power = cosine_emit if unit_exponents else cosine_emit ** exponent[None, :]
    factor = normalization[None, :] * emission_power
    factor *= ((cosine_emit > 0) * cosine_in * visible
               / torch.clamp(distance2, min=torch.finfo(distance2.dtype).tiny))
    return factor.to(powers.dtype), visible


def shade_light_block(*args, **meta):
    factor, visible = light_block_factor(*args, **meta)
    powers = args[5]
    contribution = torch.sum(factor[..., None] * powers[None, :, :], dim=1)
    return contribution, visible
