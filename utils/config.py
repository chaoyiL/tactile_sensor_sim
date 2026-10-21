"""Shared JSON configuration loading and validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import camera_frame, contact_radius, strip_geometry


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    _require(result.shape == (size,) and np.all(np.isfinite(result)),
             f'{name}: expected a finite vector of length {size}')
    return result


def _positive(value, name: str, *, zero: bool = False) -> float:
    _require(not isinstance(value, bool) and np.isscalar(value) and np.isfinite(value)
             and (value >= 0 if zero else value > 0), f'{name}: invalid number')
    return float(value)


def _positive_int(value, name: str, *, minimum: int = 1) -> int:
    _require(type(value) is int and value >= minimum, f'{name}: expected integer >= {minimum}')
    return value


def validate_config(config: dict, *, allow_contact_tangent: bool = False) -> dict:
    """Validate a simulation config and return the same mapping.

    ``allow_contact_tangent`` is used by the calibration scanner, whose first and
    last poses deliberately place the analytic contact circle on the rectangle edge.
    """
    _require(isinstance(config, dict), 'Configuration root must be an object')
    try:
        surface, indentation, camera, zemax = (
            config[key] for key in ('surface', 'indentation', 'camera', 'zemax'))
    except KeyError as exc:
        raise ValueError(f'Missing configuration section: {exc.args[0]}') from exc
    render = config.get('render')
    _require(isinstance(render, dict), 'Missing configuration section: render')

    size = _vector(surface['size_mm'], 2, 'surface.size_mm')
    _require(np.all(size > 0), 'surface.size_mm must be positive')
    reflectance = _vector(surface['reflectance_rgb'], 3, 'surface.reflectance_rgb')
    _require(np.all((reflectance >= 0) & (reflectance <= 1)),
             'surface.reflectance_rgb must be in [0,1]')
    _require(surface['view_side_z'] in (-1, 1), 'surface.view_side_z must be +/-1')
    _require(indentation['direction_z'] in (-1, 1), 'indentation.direction_z must be +/-1')
    _positive(indentation['radius_mm'], 'indentation.radius_mm')
    _positive(indentation['depth_mm'], 'indentation.depth_mm', zero=True)
    _require(indentation['depth_mm'] < indentation['radius_mm'],
             'Require 0 <= depth_mm < radius_mm (single-valued cap)')
    center = _vector(indentation['center_xy_mm'], 2, 'indentation.center_xy_mm')
    radius = float(contact_radius(config))
    tolerance = 1e-10 * max(1., *size.tolist(), radius)
    if allow_contact_tangent:
        _require(np.all(np.abs(center) + radius <= size / 2 + tolerance),
                 'Contact disc must lie inside or tangent to the surface')
    else:
        _require(np.all(np.abs(center) + radius < size / 2),
                 'Contact disc must lie strictly inside the surface')

    for key in ('width_px', 'height_px'):
        _positive_int(camera[key], f'camera.{key}')
    for key in ('fx_px', 'fy_px', 'focal_length_mm'):
        _positive(camera[key], f'camera.{key}')
    for key in ('cx_px', 'cy_px'):
        _require(np.isfinite(camera[key]), f'camera.{key} must be finite')
    position = _vector(camera['position_mm'], 3, 'camera.position_mm')
    _require(position[2] * surface['view_side_z']
             > max(0, indentation['direction_z'] * surface['view_side_z']
                   * indentation['depth_mm']) + .1,
             'Camera must be outside the entire surface, on view_side_z')
    if 'rotation_camera_to_world' not in camera:
        _vector(camera['look_at_mm'], 3, 'camera.look_at_mm')
        _vector(camera['up_world'], 3, 'camera.up_world')
    rotation = camera_frame(camera)
    _require(rotation.shape == (3, 3)
             and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
             and np.isclose(np.linalg.det(rotation), 1),
             'camera rotation must be a proper orthonormal matrix')
    response = np.asarray(camera['response_rgb_3x3'])
    _require(response.shape == (3, 3) and np.all(np.isfinite(response)),
             'Invalid camera response matrix')
    if 'distortion_opencv' in camera:
        _require(np.all(np.asarray(camera['distortion_opencv']) == 0),
                 'Only undistorted pinhole intrinsics are supported')

    _require(isinstance(config.get('lights'), list) and len(config['lights']) > 0,
             'At least one LED strip is required')
    for index, light in enumerate(config['lights']):
        prefix = f'lights[{index}]'
        _require(light['side'] in ('left', 'right', 'top', 'bottom'),
                 f'{prefix}.side is unknown')
        for key in ('offset_mm', 'length_mm', 'width_mm'):
            _positive(light[key], f'{prefix}.{key}', zero=key == 'offset_mm')
        _require(np.isfinite(light['z_mm'])
                 and light['z_mm'] * surface['view_side_z'] > 0,
                 f'{prefix}.z_mm must lie on the viewing side')
        power = _vector(light['power_rgb_w'], 3, f'{prefix}.power_rgb_w')
        _require(np.all(power >= 0), f'{prefix}.power_rgb_w cannot be negative')
        exponent = _positive(light['cosine_exponent'], f'{prefix}.cosine_exponent')
        _require(exponent <= 100, f'{prefix}.cosine_exponent must be <= 100')
        for key in ('samples_length', 'samples_width'):
            _positive_int(light[key], f'{prefix}.{key}')
        if 'normal_world' in light:
            _vector(light['normal_world'], 3, f'{prefix}.normal_world')
        _require(np.isfinite(light.get('along_offset_mm', 0)),
                 f'{prefix}.along_offset_mm must be finite')
        strip_geometry(light, config)
    _require(sum(sum(light['power_rgb_w']) for light in config['lights']) > 0,
             'At least one light must have positive power')

    for key in ('mesh_step_mm', 'pinhole_half_width_mm'):
        _positive(zemax[key], f'zemax.{key}')
    rays = _positive_int(zemax['rays_per_strip'], 'zemax.rays_per_strip')
    _require(rays < 2 ** 31, 'zemax.rays_per_strip is too large')
    seed = zemax['random_seed']
    _require(type(seed) is int and 0 <= seed < 2 ** 31 - 3,
             'zemax.random_seed is invalid')
    waves = _vector(zemax['wavelengths_rgb_um'], 3, 'zemax.wavelengths_rgb_um')
    _require(np.all(waves > 0) and len(set(waves.tolist())) == 3,
             'Need three distinct positive wavelengths')

    _positive(render['exposure_scale'], 'render.exposure_scale')
    _require(render['tone_map'] in ('clip', 'reinhard'),
             'render.tone_map must be clip or reinhard')
    _require(type(render['shadows']) is bool, 'render.shadows must be boolean')
    device = render.get('device', 'cuda:0')
    _require(isinstance(device, str) and device, 'render.device must be a nonempty string')
    _require(render.get('geometry_dtype', 'float64') in ('float32', 'float64'),
             'render.geometry_dtype must be float32 or float64')
    _require(render.get('radiometry_dtype', 'float32') in ('float32', 'float64'),
             'render.radiometry_dtype must be float32 or float64')
    _positive_int(render.get('pixel_chunk_size', 8192), 'render.pixel_chunk_size')
    _positive_int(render.get('light_chunk_size', 16), 'render.light_chunk_size')
    return config


def load_config(path: str | Path, *, allow_contact_tangent: bool = False) -> dict:
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    return validate_config(config, allow_contact_tangent=allow_contact_tangent)

