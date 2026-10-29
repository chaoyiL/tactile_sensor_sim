"""Continuous-response endpoint migration and shared LUT fitting helpers."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable

import numpy as np

from utils.calibration_pipeline import fit_and_validate, parse_calibration_settings
from utils.calibration_samples import MaterialGrid
from utils.curve_calibration import reference_area_weights, tangent_slopes
from utils.curve_geometry import (base_surface, deformed_surface, intersect_curve,
                                  sphere_pose)
from utils.geometry import camera_frame, height_normal, intersect_surface
from utils.normal_mapping import NormalCalibration, build_normal_lut
from experiment.curve_lut_transfer.storage import json_write, npz_write

MIGRATION_SAMPLING_VERSION = 'material_paired_endpoint_H_pt_curve_v1'
MIGRATION_IMAGE_ENCODING = 'srgb8_roundtrip_plus_continuous_H_pt_delta'


def condition_id(radius_mm: float | None) -> str:
    """Return a stable directory-safe condition label."""
    return 'flat' if radius_mm is None else 'radius_' + format(radius_mm, '.12g').replace('.', 'p')


def pose_config(base: dict, radius_mm: float | None, pose: dict | None = None) -> dict:
    """Copy the source physical snapshot and set one reference-surface pose."""
    config = copy.deepcopy(base)
    config['surface']['bending']['radius_mm'] = radius_mm
    if pose is None:
        config['indentation'].update(center_uv=[0., 0.], depth_mm=0.)
    else:
        config['indentation'].update(center_uv=pose['center_uv'], depth_mm=pose['depth_mm'])
    return config


def reference_geometry(material_coordinates: np.ndarray, radius_mm: float | None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X0, geometric reference normal, and the oriented tangent frame."""
    coordinates = np.asarray(material_coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError('material_coordinates must be a finite (N,2) array')
    s, t = coordinates.T
    if radius_mm is None:
        position = np.stack((s, t, np.zeros_like(s)), axis=-1)
        normal = np.zeros_like(position)
        normal[..., 2] = 1.
        e1 = np.zeros_like(position)
        e1[..., 0] = 1.
    else:
        position, normal = base_surface(s, t, radius_mm)
        angle = s / radius_mm
        e1 = np.stack((np.cos(angle), np.zeros_like(s), -np.sin(angle)), axis=-1)
    e2 = np.zeros_like(position)
    e2[..., 1] = 1.
    frame = np.stack((e1, e2), axis=-1)
    return position, normal, frame


def contact_geometry(material_coordinates: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    """Evaluate exact reference-normal sphere contact and target tangent slopes."""
    coordinates = np.asarray(material_coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError('material_coordinates must be a finite (N,2) array')
    s, t = coordinates.T
    radius = config['surface']['bending']['radius_mm']
    if radius is None:
        base = np.stack((s, t, np.zeros_like(s)), axis=-1)
        n0 = np.zeros_like(base)
        n0[..., 2] = 1.
        z, optical_normal = height_normal(s, t, config)
        point = np.stack((s, t, z), axis=-1)
        normal_d = optical_normal * config['surface']['view_side_z']
        delta = -z
        ball = config['indentation']
        center_uv = np.asarray(ball['center_uv'], dtype=np.float64)
        center_xy = center_uv * np.asarray(config['surface']['size_mm'], dtype=np.float64) / 2
        radial_squared = (s - center_xy[0]) ** 2 + (t - center_xy[1]) ** 2
        contact = (radial_squared < max(2 * ball['radius_mm'] * ball['depth_mm']
                                        - ball['depth_mm'] ** 2, 0.)) & (ball['depth_mm'] > 0)
    else:
        base, n0 = base_surface(s, t, radius)
        point, normal_d, delta = deformed_surface(s, t, config)
        _, _, center = sphere_pose(config)
        contact = (np.sum((base - center) ** 2, axis=-1) < config['indentation']['radius_mm'] ** 2) \
            & (config['indentation']['depth_mm'] > 0)
    if radius is None:
        e1 = np.zeros_like(base)
        e1[..., 0] = 1.
    else:
        angle = s / radius
        e1 = np.stack((np.cos(angle), np.zeros_like(s), -np.sin(angle)), axis=-1)
    e2 = np.zeros_like(base)
    e2[..., 1] = 1.
    frame = np.stack((e1, e2), axis=-1)
    denominator = np.sum(n0 * normal_d, axis=-1)
    slopes = tangent_slopes(normal_d, n0, frame)
    valid = contact & (delta > 0.) & (denominator > .1)
    valid &= np.isfinite(slopes).all(axis=-1) & np.isfinite(normal_d).all(axis=-1)
    return dict(base_position_mm=base, point_mm=point, reference_normal=n0,
                geometric_normal=normal_d, tangent_frame=frame, slope=slopes,
                displacement_mm=delta, contact=contact, valid=valid,
                reference_dot=denominator)


def _camera_project(points: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    camera = config['camera']
    rotation = camera_frame(camera)
    camera_xyz = (np.asarray(points, dtype=np.float64) - np.asarray(camera['position_mm'])) @ rotation
    z = camera_xyz[..., 2]
    safe_z = np.where(np.abs(z) > 1e-300, z, 1.)
    uv = np.stack((camera['fx_px'] * camera_xyz[..., 0] / safe_z + camera['cx_px'],
                   camera['fy_px'] * camera_xyz[..., 1] / safe_z + camera['cy_px']), axis=-1)
    return uv, z > 0


def _ray_hits(origin: np.ndarray, rays: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    if config['surface']['bending']['radius_mm'] is None:
        distance = intersect_surface(origin, rays, config)
        return distance, np.where(np.isfinite(distance), 1, 0).astype(np.int8)
    return intersect_curve(origin, rays, config)


def _strict_contact_neighborhood(points: np.ndarray, uv: np.ndarray, config: dict,
                                 target_model=None, chunk_size: int = 8192) -> np.ndarray:
    """Check all four bilinear neighbours by ray geometry and predicted saturation."""
    camera = config['camera']
    width, height = int(camera['width_px']), int(camera['height_px'])
    uv = np.asarray(uv, dtype=np.float64)
    finite = np.isfinite(uv).all(axis=-1)
    bounded = finite & (uv[:, 0] >= 0.) & (uv[:, 0] < width - 1) \
        & (uv[:, 1] >= 0.) & (uv[:, 1] < height - 1)
    result = np.zeros(len(uv), dtype=bool)
    selected = np.flatnonzero(bounded)
    if not len(selected):
        return result
    corner_x = np.floor(uv[selected, 0]).astype(np.int64)
    corner_y = np.floor(uv[selected, 1]).astype(np.int64)
    pixels = np.stack((np.stack((corner_x, corner_y), -1),
                       np.stack((corner_x + 1, corner_y), -1),
                       np.stack((corner_x, corner_y + 1), -1),
                       np.stack((corner_x + 1, corner_y + 1), -1)), axis=1).reshape(-1, 2)
    rotation = camera_frame(camera)
    local = np.stack(((pixels[:, 0] - camera['cx_px']) / camera['fx_px'],
                      (pixels[:, 1] - camera['cy_px']) / camera['fy_px'],
                      np.ones(len(pixels))), axis=-1)
    rays = local @ rotation.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    origin = np.asarray(camera['position_mm'], dtype=np.float64)
    ray_distance, part = _ray_hits(origin, rays, config)
    hit = np.isfinite(ray_distance) & (part > 0)
    hit_points = np.full((len(rays), 3), np.nan, dtype=np.float64)
    geometric_normal = np.full((len(rays), 3), np.nan, dtype=np.float64)
    # camera_ground_truth_curve keeps only surface normals facing the sensor.
    if np.any(hit):
        hit_indices = np.flatnonzero(hit)
        hit_points[hit_indices] = origin + ray_distance[hit_indices, None] * rays[hit_indices]
        radius = config['surface']['bending']['radius_mm']
        if radius is None:
            _, optical_normal = height_normal(hit_points[hit_indices, 0], hit_points[hit_indices, 1], config)
            geometric_normal[hit_indices] = optical_normal * config['surface']['view_side_z']
        else:
            from utils.curve_geometry import project_to_base, sphere_pose
            _, _, _, reference_normal, _, _ = project_to_base(hit_points[hit_indices], radius)
            _, _, ball_center = sphere_pose(config)
            cap = part[hit_indices] == 2
            normal_values = reference_normal.copy()
            if np.any(cap):
                cap_normal = -(hit_points[hit_indices[cap]] - ball_center) / config['indentation']['radius_mm']
                normal_values[cap] = cap_normal
            geometric_normal[hit_indices] = normal_values
        hit[hit_indices] &= np.sum(geometric_normal[hit_indices] * rays[hit_indices], axis=-1) > 0.
    if target_model is not None and np.any(hit):
        display, display_valid = _evaluate_display(target_model, hit_points[hit],
                                                    geometric_normal[hit], config, chunk_size)
        from .photometry import encode_srgb8
        rgb8 = _to_numpy(encode_srgb8(
            display, dtype=getattr(target_model, 'radiometry_numpy_dtype', None))).astype(np.uint8, copy=False)
        unsaturated = display_valid & np.all(rgb8 < 255, axis=-1)
        hit_indices = np.flatnonzero(hit)
        hit[hit_indices] &= unsaturated
    result[selected] = hit.reshape(-1, 4).all(axis=1)
    return result


def target_valid_grid_indices(material_coordinates: np.ndarray, material_grid_index: np.ndarray,
                              config: dict, reference_valid: np.ndarray,
                              *, minimum_reference_dot: float = .1,
                              target_model=None, chunk_size: int = 8192) -> np.ndarray:
    """Get target-valid flat material indices using exact contact/camera geometry.

    The no-contact reference-valid grid is reused from the selected historical transfer run.
    Contact-frame bilinear neighbors are checked with ray intersections; this does not read
    target training colors or labels and does not render an image.
    """
    coords = np.asarray(material_coordinates, dtype=np.float64)
    ij = np.asarray(material_grid_index)
    ref_valid = np.asarray(reference_valid, dtype=bool)
    if ij.shape != (len(coords), 2) or not np.issubdtype(ij.dtype, np.integer):
        raise ValueError('material_grid_index must be an integer (N,2) row/column array')
    height, width = ref_valid.shape
    if np.any(ij[:, 0] < 0) or np.any(ij[:, 0] >= height) \
            or np.any(ij[:, 1] < 0) or np.any(ij[:, 1] >= width):
        raise ValueError('material_grid_index falls outside the saved reference-valid grid')
    flat = ij[:, 0].astype(np.int64) * width + ij[:, 1]
    state = contact_geometry(coords, config)
    uv, front = _camera_project(state['point_mm'], config)
    camera = config['camera']
    vector = state['point_mm'] - np.asarray(camera['position_mm'], dtype=np.float64)
    distance = np.linalg.norm(vector, axis=-1)
    ray = vector / np.maximum(distance[:, None], 1e-300)
    first, _ = _ray_hits(np.asarray(camera['position_mm'], dtype=np.float64), ray, config)
    visible = np.isclose(first, distance, rtol=1e-7, atol=1e-6)
    facing = np.sum(state['geometric_normal'] * ray, axis=-1) > 0.
    strict_pixels = _strict_contact_neighborhood(state['point_mm'], uv, config,
                                                 target_model=target_model, chunk_size=chunk_size)
    valid = state['valid'] & (state['reference_dot'] > minimum_reference_dot) \
        & ref_valid[ij[:, 0], ij[:, 1]] & front & visible & facing & strict_pixels
    return np.unique(flat[valid])


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _evaluate_display(model, points: np.ndarray, normals: np.ndarray, state_config: dict,
                      chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    pieces, valid_pieces = [], []
    for begin in range(0, len(points), chunk_size):
        end = min(begin + chunk_size, len(points))
        response = model.evaluate(points[begin:end], normals[begin:end], state_config=state_config)
        # display_rgb is linear-light RGB after exposure and Reinhard, matching
        # inverse_srgb(saved_srgb8) LUT residual samples.
        color = _to_numpy(response.display_rgb).astype(np.float64, copy=False)
        if color.shape != (end - begin, 3):
            raise ValueError(f'PhotometryModel.display_rgb returned an invalid shape: {color.shape}')
        valid = _to_numpy(response.valid).astype(bool, copy=False)
        if valid.shape != (end - begin,):
            raise ValueError(f'PhotometryModel.valid returned an invalid shape: {valid.shape}')
        pieces.append(color)
        valid_pieces.append(valid)
    return ((np.concatenate(pieces, axis=0), np.concatenate(valid_pieces, axis=0))
            if pieces else (np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=bool)))


def _display_rgb(model, points: np.ndarray, normals: np.ndarray, state_config: dict,
                 chunk_size: int) -> np.ndarray:
    return _evaluate_display(model, points, normals, state_config, chunk_size)[0]


def migrate_pose_samples(source_arrays: dict, pose_index: int, pose: dict,
                         radius_mm: float | None, base_config: dict,
                         target_indices: np.ndarray, flat_model, target_model,
                         chunk_size: int) -> dict[str, np.ndarray]:
    """Apply Eq. (5) to one source pose and rebuild the target geometric labels.

    Target frame colors and slopes are never read here. `target_indices` carries only
    independently computed geometric/camera support for the material grid.
    """
    if type(pose_index) is not int or pose_index < 0:
        raise ValueError('pose_index must be a nonnegative integer')
    image_index = np.asarray(source_arrays['image_index'])
    selected = np.flatnonzero(image_index == pose_index)
    if not len(selected):
        raise ValueError(f'Source pose index {pose_index} has no samples')
    grid_index_2d = np.asarray(source_arrays['material_grid_index'])
    if grid_index_2d.ndim != 2 or grid_index_2d.shape[1] != 2:
        raise ValueError('Source material_grid_index must have shape (N,2)')
    grid_x = np.asarray(source_arrays['material_grid_x_mm'], dtype=np.float64)
    grid_y = np.asarray(source_arrays['material_grid_y_mm'], dtype=np.float64)
    flat_width = len(grid_x)
    flat_indices = grid_index_2d[selected, 0].astype(np.int64) * flat_width \
        + grid_index_2d[selected, 1].astype(np.int64)
    if len(np.unique(flat_indices)) != len(flat_indices):
        raise ValueError(f'Source pose {pose_index} contains duplicate material grid indices')
    keep = np.isin(flat_indices, np.asarray(target_indices, dtype=np.int64), assume_unique=False)
    selected = selected[keep]
    flat_indices = flat_indices[keep]
    if not len(selected):
        return dict(grid_index=np.empty(0, np.int64), color_residual_samples=np.empty((0, 3), np.float32),
                    slope_samples=np.empty((0, 2), np.float32), material_coordinates=np.empty((0, 2), np.float64),
                    material_grid_index=np.empty((0, 2), np.int32), reference_normal_samples=np.empty((0, 3), np.float32),
                    tangent_frame_samples=np.empty((0, 3, 2), np.float32), normal_displacement_mm=np.empty(0, np.float64),
                    area_mm2=np.empty(0, np.float64), source_candidate_count=int(len(np.flatnonzero(image_index == pose_index))),
                    target_geometry_retained_count=0, model_invalid_count=0)
    order = np.argsort(flat_indices, kind='stable')
    selected = selected[order]
    flat_indices = flat_indices[order]
    coordinates = np.asarray(source_arrays['material_coordinates'][selected], dtype=np.float64)
    flat_state_config = pose_config(base_config, None, pose)
    flat_reference_config = pose_config(base_config, None, None)
    target_state_config = pose_config(base_config, radius_mm, pose)
    target_reference_config = pose_config(base_config, radius_mm, None)
    flat_contact = contact_geometry(coordinates, flat_state_config)
    target_contact = contact_geometry(coordinates, target_state_config)
    flat_labels = flat_contact['valid'] & np.isfinite(flat_contact['slope']).all(axis=-1)
    target_labels = target_contact['valid'] & np.isfinite(target_contact['slope']).all(axis=-1)
    if not np.all(flat_labels):
        raise ValueError(f'Source samples do not satisfy the saved planar sphere geometry: pose {pose_index}')
    model_colors = np.empty((len(selected), 3), dtype=np.float64)
    model_valid = np.ones(len(selected), dtype=bool)
    for begin in range(0, len(selected), chunk_size):
        end = min(begin + chunk_size, len(selected))
        h_flat_contact, v_flat_contact = _evaluate_display(
            flat_model, flat_contact['point_mm'][begin:end], flat_contact['geometric_normal'][begin:end],
            flat_state_config, chunk_size)
        h_flat_reference, v_flat_reference = _evaluate_display(
            flat_model, flat_contact['base_position_mm'][begin:end], flat_contact['reference_normal'][begin:end],
            flat_reference_config, chunk_size)
        h_target_contact, v_target_contact = _evaluate_display(
            target_model, target_contact['point_mm'][begin:end], target_contact['geometric_normal'][begin:end],
            target_state_config, chunk_size)
        h_target_reference, v_target_reference = _evaluate_display(
            target_model, target_contact['base_position_mm'][begin:end], target_contact['reference_normal'][begin:end],
            target_reference_config, chunk_size)
        h_flat = h_flat_contact - h_flat_reference
        h_target = h_target_contact - h_target_reference
        model_colors[begin:end] = h_target - h_flat
        model_valid[begin:end] = v_flat_contact & v_flat_reference & v_target_contact & v_target_reference
    model_valid &= np.isfinite(model_colors).all(axis=-1)
    keep = target_labels & model_valid
    retained = selected[keep]
    ij = np.asarray(source_arrays['material_grid_index'][retained], dtype=np.int32)
    xy_grid = np.stack(np.meshgrid(grid_x, grid_y, indexing='xy'), axis=-1)
    st_grid = xy_grid - np.asarray([grid_x[0], grid_y[0]])
    area_grid = reference_area_weights(MaterialGrid(grid_x, grid_y, xy_grid, st_grid,
                                                    'archived-source-grid'))
    colors = np.asarray(source_arrays['color_residual_samples'][selected], dtype=np.float64) \
        + model_colors
    colors = colors[keep].astype(np.float32)
    return dict(grid_index=flat_indices[keep].astype(np.int64),
                color_residual_samples=colors,
                slope_samples=target_contact['slope'][keep].astype(np.float32),
                material_coordinates=coordinates[keep], material_grid_index=ij,
                reference_normal_samples=target_contact['reference_normal'][keep].astype(np.float32),
                tangent_frame_samples=target_contact['tangent_frame'][keep].astype(np.float32),
                normal_displacement_mm=target_contact['displacement_mm'][keep].astype(np.float64),
                area_mm2=area_grid[ij[:, 0], ij[:, 1]].astype(np.float64),
                source_candidate_count=int(len(np.flatnonzero(image_index == pose_index))),
                target_geometry_retained_count=int(np.count_nonzero(target_labels)),
                model_invalid_count=int(np.count_nonzero(~model_valid)))


def fit_migrated_condition(samples_by_pose: list[dict], base_config: dict, output_dir: Path,
                           metadata: dict, pose_split: dict[str, list[str]],
                           required_pose_keys: list[str], *, fit_fn: Callable | None = None
                           ) -> tuple[dict, NormalCalibration | None]:
    """Fit and validate one migrated LUT using the source workflow and fixed pose split."""
    if not samples_by_pose or not required_pose_keys:
        raise ValueError('A migrated condition requires pose samples and required pose keys')
    if len(samples_by_pose) != len(required_pose_keys):
        raise ValueError('samples_by_pose and required_pose_keys must have equal lengths')
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(base_config)
    calibration = parse_calibration_settings(config, output_dir / 'config.json', output_dir)
    counts = [len(sample['grid_index']) for sample in samples_by_pose]
    accepted = [count >= calibration.minimum_samples_per_pose for count in counts]
    fields = ('color_residual_samples', 'slope_samples', 'material_coordinates',
              'reference_normal_samples', 'tangent_frame_samples', 'normal_displacement_mm',
              'material_grid_index', 'area_mm2')
    used = [sample for sample, keep in zip(samples_by_pose, accepted, strict=True) if keep]
    if used:
        samples = {key: np.concatenate([sample[key] for sample in used], axis=0) for key in fields}
        samples['image_index'] = np.concatenate([
            np.full(len(sample['grid_index']), index, np.int32)
            for index, (sample, keep) in enumerate(zip(samples_by_pose, accepted, strict=True)) if keep])
    else:
        sample0 = samples_by_pose[0]
        samples = {key: np.asarray(sample0[key])[:0] for key in fields}
        samples['image_index'] = np.empty(0, np.int32)
    samples['image_paths'] = np.asarray(required_pose_keys)
    from utils.curve_calibration import reference_grid
    grid = reference_grid(config)
    grid_x = np.asarray(grid.x_mm, dtype=np.float64)
    grid_y = np.asarray(grid.y_mm, dtype=np.float64)
    json_write(output_dir / 'config.json', config)
    array_metadata = dict(metadata, source_pose_sample_counts=dict(zip(required_pose_keys, counts)),
                          minimum_samples_per_pose=calibration.minimum_samples_per_pose)
    npz_write(output_dir / 'normal_samples.npz', **samples,
              format_version=np.asarray(5, np.int32), metadata=json.dumps(array_metadata, ensure_ascii=False),
              material_grid_x_mm=grid_x, material_grid_y_mm=grid_y,
              coordinate_kind=np.asarray('reference_arclength'),
              coordinate_chart_id=np.asarray(metadata.get('coordinate_chart_id', 'cylinder_reference_arclength')),
              color_space=np.asarray('linear_rgb'),
              image_encoding=np.asarray(metadata.get('image_encoding', MIGRATION_IMAGE_ENCODING)),
              sampling_version=np.asarray(metadata.get('sampling_version', MIGRATION_SAMPLING_VERSION)),
              residual_domain=np.asarray('material_grid'))
    fit_function = fit_fn or fit_and_validate
    training_accepted = any(ok and key in pose_split['training_images']
                            for key, ok in zip(required_pose_keys, accepted, strict=True))
    validation_accepted = any(ok and key in pose_split['validation_images']
                              for key, ok in zip(required_pose_keys, accepted, strict=True))
    if training_accepted and validation_accepted:
        report, _, final, diagnostic = fit_function(
            samples, calibration, output_dir, metadata,
            pose_split=pose_split, required_pose_keys=required_pose_keys)
    else:
        report = dict(status='failed', reason='Insufficient accepted poses in fixed source split',
                      missing_required_poses=[key for key, ok in zip(required_pose_keys, accepted, strict=True) if not ok],
                      training_accepted_pose_count=sum(ok and key in pose_split['training_images']
                                                       for key, ok in zip(required_pose_keys, accepted, strict=True)),
                      validation_accepted_pose_count=sum(ok and key in pose_split['validation_images']
                                                         for key, ok in zip(required_pose_keys, accepted, strict=True)))
        final, diagnostic = None, None
        if calibration.visualization_enabled and len(samples['color_residual_samples']):
            final = build_normal_lut(samples['color_residual_samples'], samples['slope_samples'],
                                     size=calibration.lut_size,
                                     maximum_rms_angle_degrees=calibration.maximum_rms_angle_degrees)
            final = type(final)(final.slopes, final.variances, final.color_min, final.color_max,
                                final.sigma_ref2, final.original_valid, final.sample_counts,
                                dict(metadata, validation_status='failed', validation=report))
            diagnostic = output_dir / 'normal_lut_diagnostic.npz'
            final.save(diagnostic, total_sample_count=len(samples['color_residual_samples']))
    report['pose_sample_counts'] = dict(zip(required_pose_keys, counts))
    report['accepted_pose_count'] = int(sum(accepted))
    report['required_pose_count'] = len(required_pose_keys)
    json_write(output_dir / 'validation.json', report)
    status = report.get('status', 'failed')
    if final is not None:
        expected_path = output_dir / ('normal_lut.npz' if status == 'passed' else 'normal_lut_diagnostic.npz')
        if not expected_path.is_file():
            final.save(expected_path, total_sample_count=len(samples['color_residual_samples']))
    lut_path = ('normal_lut.npz' if status == 'passed' and final is not None else
                'normal_lut_diagnostic.npz' if final is not None else None)
    info = dict(status=status, lut=lut_path, required_pose_count=len(required_pose_keys),
                accepted_pose_count=int(sum(accepted)), missing_required_poses=report.get('missing_required_poses', []))
    json_write(output_dir / 'manifest.json', dict(info, config=config,
             run_signature=metadata.get('run_signature'), stage='completed' if status == 'passed' else 'diagnostic'))
    return report, final
