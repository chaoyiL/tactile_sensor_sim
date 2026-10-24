"""Small local-optics diagnostic for an existing length experiment manifest.

This is NOT NormDiff and does not fit a LUT. At the flat surface centre, vary
only the optical normal n(p,q)=view_side_z*normalize(-p,-q,1), holding position fixed.
Compute the RGB relative slope Jacobian before sRGB8 quantization. Occlusion
changes and the geometry of a finite indentation are not represented here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .runner import length_config, write_json
from utils.geometry import LIGHT_GEOMETRY_MODEL, camera_frame, strip_samples
from utils.surface_coordinates import analytic_parameter


def evaluate(config: dict) -> dict:
    # E = matrix @ normal for the currently front-facing flat-surface lights.
    matrix = np.zeros((3, 3))
    flat_normal = np.array([0., 0., config['surface']['view_side_z']])
    for light in config['lights']:
        points, emission_normal = strip_samples(light, config)
        distance2 = np.sum(points * points, axis=1)
        direction = points / np.sqrt(distance2[:, None])
        front = direction @ flat_normal > 0
        cosine = np.maximum((-direction) @ emission_normal, 0.)
        exponent = light['cosine_exponent']
        coefficients = ((exponent + 1) / (2 * np.pi) * cosine ** exponent
                        / distance2 / len(points) * 1e6 * front)
        matrix += np.einsum('i,c,ij->cj', coefficients,
                            np.asarray(light['power_rgb_w']), direction)
    irradiance = matrix @ flat_normal
    irradiance_jacobian = -config['surface']['view_side_z'] * matrix[:, :2]
    camera = config['camera']
    ray = -np.asarray(camera['position_mm'], dtype=float)
    ray /= np.linalg.norm(ray)
    cos4 = float(ray @ camera_frame(camera)[:, 2]) ** 4
    factor = (np.asarray(config['surface']['reflectance_rgb']) / np.pi
              * (2 * analytic_parameter(config, 'camera', 'pinhole_half_width_mm') / camera['focal_length_mm']) ** 2
              * cos4 * config['render']['exposure_scale'])
    response = np.asarray(camera['response_rgb_3x3'])
    exposed_rgb = response @ (irradiance * factor)
    exposed_jacobian = response @ (irradiance_jacobian * factor[:, None])
    if np.any(exposed_rgb <= 0):
        raise ValueError('Relative Jacobian requires positive RGB at the flat centre')
    relative_jacobian = exposed_jacobian / exposed_rgb[:, None]
    if config['render']['tone_map'] == 'reinhard':
        observed_rgb = exposed_rgb / (1 + exposed_rgb)
        relative_jacobian /= (1 + exposed_rgb[:, None])
    else:
        observed_rgb = np.clip(exposed_rgb, 0, 1)
        relative_jacobian[exposed_rgb >= 1] = 0
    singular_values = np.linalg.svd(relative_jacobian, compute_uv=False)
    return dict(flat_center_irradiance_rgb_w_m2=irradiance.tolist(),
                prequantization_observed_rgb=observed_rgb.tolist(),
                relative_rgb_slope_jacobian=relative_jacobian.tolist(),
                singular_values=singular_values.tolist(),
                minimum_singular_value=float(singular_values[-1]))


def run_diagnostics(manifest_path: str | Path, *, output_dir: str | Path | None = None):
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('light_geometry_model') != LIGHT_GEOMETRY_MODEL:
        raise ValueError('Experiment manifest uses a different or legacy light geometry model; '
                         'run a new length experiment in a new output directory first')
    rows = [dict(length_mm=length,
                 **evaluate(length_config(manifest['base_config'], length)))
            for length in manifest['lengths_mm']]
    report = dict(experiment_signature=manifest['signature'], points=rows,
                  definition='d log(observed_RGB) / d(p,q) at p=q=0; '
                             'n=view_side_z*normalize(-p,-q,1), fixed flat centre position',
                  limitations='Local diagnostic, not NormDiff or LUT error. '
                              'Before quantization; no finite indentation or changing occlusion.')
    directory = Path(output_dir).expanduser().resolve() if output_dir else manifest_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'center_light_diagnostics.json'
    write_json(target, report)
    for row in rows:
        print(f'length={row["length_mm"]:g}: relative slope sensitivity '
              f'sigma_min={row["minimum_singular_value"]:.6f}')
    print(target)
    return report
