"""Zemax sphere scan using the shared LUT fitting and validation pipeline."""
from __future__ import annotations

from collections import Counter
import copy
import csv
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .calibration_ambiguity import save_ambiguity_inputs
from .calibration_pipeline import (_atomic_json, _atomic_npz, _config_signature,
                                   _ensure_new_output, _save_samples,
                                   fit_and_validate, CalibrationValidationFailed)
from .calibration_samples import calibration_declarations
from .geometry import LIGHT_GEOMETRY_MODEL, camera_frame, height_normal, intersect_surface
from .mujoco_backend import inverse_srgb_rgb8
from .mujoco_samples import bilinear_sample
from .normal_mapping import UPSTREAM_COMMIT
from .renderer import encode_display
from .surface_coordinates import normal_direction, set_center_material_mm
from .renderer_cpu import camera_ground_truth
from .zemax_backend import render as zemax_render


OBSERVATION_MODEL_ID = 'zemax_nsc_detector_srgb8_v1'


def _sample_grid(rgb8: np.ndarray, config: dict, points: np.ndarray,
                 source_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = config['camera']
    origin = np.asarray(camera['position_mm'], np.float64)
    local = (points - origin) @ camera_frame(camera)
    front = local[..., 2] > 0
    denominator = np.where(front, local[..., 2], 1.)
    px = local[..., 0] / denominator * camera['fx_px'] + camera['cx_px']
    py = local[..., 1] / denominator * camera['fy_px'] + camera['cy_px']
    color, valid = bilinear_sample(inverse_srgb_rgb8(rgb8), px, py, source_valid)
    return color, valid & front


def _pose_samples(rgb8: np.ndarray, config: dict, pose, grid,
                  reference_color: np.ndarray, reference_valid: np.ndarray,
                  image_valid: np.ndarray) -> tuple[dict, dict]:
    xy = grid.xy_mm
    z, optical_normal = height_normal(xy[..., 0], xy[..., 1], config)
    points = np.concatenate((xy, z[..., None]), axis=-1)
    observed, observed_valid = _sample_grid(rgb8, config, points, image_valid)
    origin = np.asarray(config['camera']['position_mm'], np.float64)
    vector = points - origin
    distance = np.linalg.norm(vector, axis=-1)
    direction = vector / np.maximum(distance[..., None], np.finfo(float).tiny)
    nearest = intersect_surface(origin, direction, config)
    visible = np.isclose(nearest, distance, rtol=1e-7, atol=1e-6)
    facing = np.sum(optical_normal * -direction, axis=-1) > 0
    dx = xy[..., 0] - pose.center_xy_mm[0]
    dy = xy[..., 1] - pose.center_xy_mm[1]
    rr = dx * dx + dy * dy
    sphere = config['indentation']
    root = np.sqrt(np.maximum(sphere['radius_mm'] ** 2 - rr, 1e-16))
    slope = np.stack((-normal_direction(config) * dx / root,
                      -normal_direction(config) * dy / root), axis=-1)
    margin = config['calibration']['sampling'].get('contact_margin_mm', 0.)
    contact = rr < (pose.contact_radius_mm - margin) ** 2
    reference_dot = 1. / np.sqrt(1. + np.sum(slope * slope, axis=-1))
    residual = observed - reference_color
    valid = (contact & reference_valid & observed_valid & visible & facing
             & (reference_dot > .1) & np.isfinite(residual).all(axis=-1))
    rows, columns = np.nonzero(valid)
    compact = {
        'color_residual_samples': residual[rows, columns].astype(np.float32),
        'slope_samples': slope[rows, columns].astype(np.float32),
        'material_coordinates': xy[rows, columns].astype(np.float64),
        'st_samples': grid.st_mm[rows, columns].astype(np.float32),
        'material_grid_index': np.stack((rows, columns), axis=-1).astype(np.int32),
    }
    diagnostics = {'sample_count': len(rows), 'contact_grid_count': int(contact.sum()),
                   'reference_valid_grid_count': int(reference_valid.sum()),
                   'saturated_pixel_fraction': float(np.mean(np.any(rgb8 == 255, axis=-1)))}
    return compact, diagnostics


def run_zemax_calibration(config: dict, settings, poses, grid) -> dict:
    """Collect Zemax observations, then use the common LUT fit and pose holdout."""
    if not config['render']['shadows']:
        raise ValueError('Zemax calibration requires render.shadows=true')
    output = settings.output_dir
    _ensure_new_output(output)
    started = time.perf_counter()
    signature = _config_signature(config)
    metadata = dict(backend='zemax', calibration_kind='simulated_sphere_scan',
                    light_geometry_model=LIGHT_GEOMETRY_MODEL,
                    observation_model_id=OBSERVATION_MODEL_ID,
                    upstream_commit=UPSTREAM_COMMIT, run_signature=signature,
                    simulation_config_signature=signature,
                    sphere_radius_mm=float(config['indentation']['radius_mm']),
                    sphere_direction_z=int(normal_direction(config)),
                    sphere_side=('positive_normal' if normal_direction(config) == -1
                                 else 'negative_normal'),
                    boundary_model='zero_boundary_displacement',
                    scan=copy.deepcopy(config['calibration']['scan']),
                    reference_color_source='zemax_zero_depth_srgb8_roundtrip',
                    fixed_reference_normal=[0., 0., 1.],
                    fixed_tangent_frame=[[1., 0.], [0., 1.], [0., 0.]],
                    mesh_label_note='Analytic cap labels; Zemax traces its STL approximation',
                    **calibration_declarations(grid))
    manifest = dict(format_version=2, status='running', run_status='running',
                    stage='reference', backend='zemax', run_signature=signature,
                    config=config, reference={'storage': 'archived_material_grid',
                                              'depth_mm': 0.},
                    poses=[], artifacts={}, mesh_label_note=metadata['mesh_label_note'])
    _atomic_json(output / 'manifest.json', manifest)
    flat = copy.deepcopy(config)
    flat['indentation']['depth_mm'] = 0.
    reference_dir = output / 'reference_scene'
    reference_dir.mkdir()
    sensor, _ = zemax_render(flat, reference_dir)
    flat_gt = camera_ground_truth(flat)
    reference_rgb8 = encode_display(sensor, flat)
    Image.fromarray(reference_rgb8).save(output / 'reference.png')
    flat_points = np.concatenate((grid.xy_mm, np.zeros((*grid.shape, 1))), axis=-1)
    flat_valid = flat_gt['mask'] & np.all(reference_rgb8 < 255, axis=-1)
    reference_color, reference_valid = _sample_grid(
        reference_rgb8, flat, flat_points, flat_valid)
    _atomic_npz(output / 'reference_grid.npz', color=reference_color,
                valid=reference_valid, raw_rgb8=reference_rgb8)
    manifest['stage'] = 'sampling'
    _atomic_json(output / 'manifest.json', manifest)
    pieces, keys, pose_ids, rows = [], [], [], []
    for sequence, pose in enumerate(poses, 1):
        pose_config = copy.deepcopy(config)
        set_center_material_mm(pose_config, pose.center_xy_mm)
        pose_config['indentation']['depth_mm'] = pose.depth_mm
        scene_dir = output / 'pose_scenes' / pose.image_key
        scene_dir.mkdir(parents=True)
        pose_started = time.perf_counter()
        sensor, _ = zemax_render(pose_config, scene_dir)
        gt = camera_ground_truth(pose_config)
        rgb8 = encode_display(sensor, pose_config)
        image_valid = gt['mask'] & np.all(rgb8 < 255, axis=-1)
        compact, diagnostics = _pose_samples(
            rgb8, pose_config, pose, grid, reference_color, reference_valid,
            image_valid)
        count = diagnostics['sample_count']
        accepted = count >= settings.minimum_samples_per_pose
        status = 'accepted' if accepted else 'rejected_insufficient_samples'
        preview = output / 'verification' / f'{pose.image_key}.png'
        Image.fromarray(rgb8).save(preview)
        if accepted:
            pieces.append(compact)
            keys.append(pose.image_key)
            pose_ids.append(pose.pose_id)
        row = dict(**pose.to_dict(), status=status, sample_count=count,
                   diagnostics=diagnostics, render_and_sample_seconds=time.perf_counter()-pose_started,
                   preview_file=str(preview.relative_to(output)),
                   scene_dir=str(scene_dir.relative_to(output)))
        rows.append(row)
        manifest['poses'].append(row)
        _atomic_json(output / 'manifest.json', manifest)
        print(f'[{sequence}/{len(poses)}] {pose.image_key}: {status}, {count} samples', flush=True)
    sample_metadata = dict(metadata, accepted_pose_count=len(pieces),
                           rejected_pose_count=len(poses)-len(pieces),
                           rejected_pose_reasons=dict(Counter(
                               row['status'] for row in rows if row['status'] != 'accepted')),
                           pose_key_by_image_index=keys, pose_id_by_image_index=pose_ids,
                           material_grid_shape=list(grid.shape))
    samples = _save_samples(output / 'normal_samples.npz', pieces, keys, pose_ids,
                            grid, sample_metadata, reference_grid_color=reference_color,
                            reference_grid_valid=reference_valid)
    save_ambiguity_inputs(output / 'ambiguity_inputs.npz', samples,
                          output / 'normal_samples.npz')
    with (output / 'verification' / 'report.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['pose_id', 'image_key', 'status',
                                                    'sample_count', 'seconds'])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(pose_id=row['pose_id'], image_key=row['image_key'],
                                 status=row['status'], sample_count=row['sample_count'],
                                 seconds=row['render_and_sample_seconds']))
    manifest['stage'] = 'fit_and_validate'
    _atomic_json(output / 'manifest.json', manifest)
    report, training_lut, formal_lut, diagnostic = fit_and_validate(
        samples, settings, output, sample_metadata)
    visualization = None
    if settings.visualization_enabled and formal_lut is not None:
        from .calibration_visualization import generate_calibration_visualizations
        visualization = generate_calibration_visualizations(
            samples, formal_lut, training_lut, settings, output / 'visualization')
    manifest.update(status=report['status'], run_status='completed', stage='completed',
                    validation=report, performance={'total_seconds': time.perf_counter()-started,
                                                     'render_count': len(poses)+1},
                    artifacts={'samples': 'normal_samples.npz',
                               'reference': 'reference_grid.npz',
                               'training_lut': 'normal_lut_train.npz',
                               'formal_lut': 'normal_lut.npz' if formal_lut is not None else None,
                               'diagnostic_lut': diagnostic.name if diagnostic else None,
                               'validation': 'verification/validation.json',
                               'visualization': str(visualization.relative_to(output))
                               if visualization else None})
    _atomic_json(output / 'manifest.json', manifest)
    if report['status'] != 'passed':
        raise CalibrationValidationFailed(report)
    return manifest
