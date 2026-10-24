"""End-to-end simulated normal-LUT calibration pipeline."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from collections import Counter
import copy
import csv
import hashlib
import json
import math
import time

import numpy as np
from PIL import Image, ImageDraw

from .calibration_ambiguity import save_ambiguity_inputs
from .calibration_samples import (COORDINATE_KIND, FRAME_CONVENTION,
                                  build_material_grid, calibration_declarations,
                                  compact_frame_samples, image_space_diagnostics,
                                  render_reference, sample_pose)
from .calibration_scan import ScanPose, build_scan_poses
from .backend_layout import CONFIG_DIR, lut_result_dir
from .config import load_config
from .surface_coordinates import normal_direction, require_flat
from .geometry import LIGHT_GEOMETRY_MODEL
from .normal_mapping import (NormalCalibration, UPSTREAM_COMMIT,
                             build_normal_lut, lookup_slopes_raw,
                             slopes_to_normals)


@dataclass(frozen=True)
class CalibrationSettings:
    output_dir: Path
    lut_size: int
    maximum_rms_angle_degrees: float
    validation_fraction: float
    validation_rms_degrees: float
    maximum_out_of_range_fraction: float
    minimum_samples_per_pose: int
    difference_gain: float
    visualization_enabled: bool
    color_distance_threshold: float
    normal_angle_threshold_degrees: float
    export_ambiguity_pairs: bool
    ambiguity_stride: int = 100
    branch_dominance_fraction: float = .8


class CalibrationValidationFailed(ValueError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__(f'Calibration validation failed; see {report.get("report_file", "validation.json")}')


def _finite_positive(value, name: str, *, zero=False) -> float:
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) \
            or (value < 0 if zero else value <= 0):
        raise ValueError(f'{name} must be finite and {">= 0" if zero else "> 0"}')
    return float(value)


def parse_calibration_settings(config: dict, config_path: Path,
                               output_override: Path | None = None) -> CalibrationSettings:
    raw = config.get('calibration')
    if not isinstance(raw, dict):
        raise ValueError('Configuration requires a calibration object')
    required = ('scan', 'material_grid', 'sampling', 'lut', 'validation',
                'visualization')
    if any(not isinstance(raw.get(key), dict) for key in required):
        raise ValueError('Calibration configuration is incomplete')
    radius = _finite_positive(config['indentation']['radius_mm'],
                              'indentation.radius_mm')
    scan = raw['scan']
    for key in ('depth_min_mm', 'depth_max_mm', 'depth_step_mm'):
        _finite_positive(scan.get(key), f'calibration.scan.{key}')
    if not 0 < scan['depth_min_mm'] <= scan['depth_max_mm'] < radius:
        raise ValueError('Calibration depths must satisfy 0 < min <= max < sphere radius')
    steps = np.asarray(scan.get('xy_step_mm'), float)
    if steps.shape != (2,) or not np.isfinite(steps).all() or np.any(steps <= 0):
        raise ValueError('calibration.scan.xy_step_mm must contain two positive values')
    if type(scan.get('include_endpoints', True)) is not bool:
        raise ValueError('calibration.scan.include_endpoints must be boolean')
    grid_steps = np.asarray(raw['material_grid'].get('step_xy_mm'), float)
    if grid_steps.shape != (2,) or not np.isfinite(grid_steps).all() or np.any(grid_steps <= 0):
        raise ValueError('calibration.material_grid.step_xy_mm must contain two positive values')
    sampling = raw['sampling']
    margin = _finite_positive(sampling.get('contact_margin_mm', 0),
                              'calibration.sampling.contact_margin_mm', zero=True)
    minimum_radius = math.sqrt(2 * radius * scan['depth_min_mm'] - scan['depth_min_mm'] ** 2)
    if margin >= minimum_radius:
        raise ValueError('contact_margin_mm must be smaller than the shallowest contact radius')
    if float(sampling.get('minimum_reference_dot', .1)) != .1:
        raise ValueError('calibration.sampling.minimum_reference_dot must be 0.1')
    minimum_samples = sampling.get('min_samples_per_pose', 16)
    if type(minimum_samples) is not int or minimum_samples < 1:
        raise ValueError('min_samples_per_pose must be a positive integer')
    lut = raw['lut']
    lut_size = lut.get('size', 64)
    if type(lut_size) is not int or lut_size < 2:
        raise ValueError('calibration.lut.size must be an integer >= 2')
    maximum_rms = _finite_positive(lut.get('maximum_rms_angle_degrees', 5),
                                   'calibration.lut.maximum_rms_angle_degrees')
    validation = raw['validation']
    fraction = float(validation.get('fraction', .2))
    maximum_out = float(validation.get('maximum_out_of_range_fraction', .05))
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('calibration.validation.fraction must be in (0,1)')
    if not np.isfinite(maximum_out) or not 0 <= maximum_out <= 1:
        raise ValueError('maximum_out_of_range_fraction must be in [0,1]')
    validation_rms = _finite_positive(validation.get('rms_degrees', 20),
                                      'calibration.validation.rms_degrees')
    visualization = raw['visualization']
    enabled = visualization.get('enabled', True)
    export_pairs = visualization.get('export_ambiguity_pairs', False)
    if type(enabled) is not bool or type(export_pairs) is not bool:
        raise ValueError('Visualization enabled/export_ambiguity_pairs must be booleans')
    difference_gain = _finite_positive(visualization.get('difference_gain', 2),
                                       'calibration.visualization.difference_gain')
    color_distance = _finite_positive(visualization.get('color_distance_threshold', .01),
                                      'color_distance_threshold', zero=True)
    normal_angle = float(visualization.get('normal_angle_threshold_degrees', 20))
    if not np.isfinite(normal_angle) or not 0 <= normal_angle <= 180:
        raise ValueError('normal_angle_threshold_degrees must be in [0,180]')
    ambiguity_stride = visualization.get('ambiguity_stride', 100)
    if type(ambiguity_stride) is not int or ambiguity_stride < 1:
        raise ValueError('ambiguity_stride must be an integer >= 1')
    dominance = visualization.get('branch_dominance_fraction', .8)
    if isinstance(dominance, bool) or not np.isscalar(dominance) \
            or not np.isfinite(dominance) or not .5 < dominance <= 1:
        raise ValueError('branch_dominance_fraction must be finite and in (0.5,1]')
    if output_override is not None:
        output = output_override.expanduser().resolve()
    else:
        if 'output_dir' in raw:
            configured = Path(raw['output_dir']).expanduser()
            output = (config_path.parent / configured).resolve() if not configured.is_absolute() \
                else configured.resolve()
        else:
            output = lut_result_dir(config.get('backend', 'analytic'))
    return CalibrationSettings(output, lut_size, maximum_rms, fraction,
                               validation_rms, maximum_out, minimum_samples,
                               difference_gain, enabled, color_distance,
                               normal_angle, export_pairs, ambiguity_stride, float(dominance))


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=_json_value), encoding='utf-8')
    temporary.replace(path)


def _atomic_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ScanPose):
        return value.to_dict()
    raise TypeError(f'Not JSON serializable: {type(value)}')


def _config_signature(config: dict) -> str:
    if config.get('backend') in ('xense', 'mujoco'):
        from .mujoco_backend import simulation_signature
        return simulation_signature(config)
    value = copy.deepcopy(config)
    value.get('calibration', {}).pop('output_dir', None)
    # Post-render evaluation settings must not invalidate reusable render samples.
    value.get('calibration', {}).pop('norm_diff', None)
    # Identical JSON has different optics after the emitting-plane convention
    # changes. Never reconstruct old residuals with a new-geometry reference.
    value = {'light_geometry_model': LIGHT_GEOMETRY_MODEL, 'config': value}
    canonical = json.dumps(value, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def _ensure_new_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Calibration output directory is not empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    (path / 'verification').mkdir(exist_ok=True)


def _circle_pixels(context, frame, radius: float, *, use_surface_height: bool):
    torch = context.torch
    theta = torch.linspace(0, 2 * math.pi, 361, device=context.device,
                           dtype=context.geometry_dtype)
    x = frame.pose.center_xy_mm[0] + radius * torch.cos(theta)
    y = frame.pose.center_xy_mm[1] + radius * torch.sin(theta)
    if use_surface_height:
        from .geometry_gpu import height_normal_torch
        z, _, _, _, _ = height_normal_torch(x, y, frame.config, optical=False, torch=torch)
    else:
        z = torch.zeros_like(x)
    uv, front = context.project_world(torch.stack((x, y, z), dim=-1), frame.config)
    array = uv[front & torch.isfinite(uv).all(dim=-1)].detach().cpu().numpy()
    return [tuple(map(float, point)) for point in array]


def save_verification_preview(context, reference, frame, path: Path,
                              settings: CalibrationSettings, status: str,
                              diagnostics=None) -> None:
    residual, _, valid, _ = (diagnostics if diagnostics is not None else
                            image_space_diagnostics(context, reference, frame))
    difference = residual.detach().cpu().numpy()
    validity = valid.detach().cpu().numpy()
    rgb = np.rint(255 * np.clip(.5 + settings.difference_gain * difference, 0, 1)).astype(np.uint8)
    rgb[~validity] = 0
    image = Image.fromarray(rgb, 'RGB')
    draw = ImageDraw.Draw(image)
    outer = _circle_pixels(context, frame, frame.pose.contact_radius_mm,
                           use_surface_height=False)
    if len(outer) >= 2:
        draw.line(outer, fill=(255, 235, 32), width=2, joint='curve')
    margin = float(frame.config['calibration']['sampling'].get('contact_margin_mm', 0))
    if margin > 0:
        inner = _circle_pixels(context, frame, frame.pose.contact_radius_mm - margin,
                               use_surface_height=True)
        if len(inner) >= 2:
            draw.line(inner, fill=(0, 255, 100), width=2, joint='curve')
    text = (f'{frame.pose.image_key}  center=({frame.pose.center_xy_mm[0]:.3f},'
            f'{frame.pose.center_xy_mm[1]:.3f}) mm  d={frame.pose.depth_mm:.3f} mm  '
            f'a={frame.pose.contact_radius_mm:.3f} mm  samples={frame.sample_count}  {status}')
    box = draw.textbbox((5, 5), text)
    draw.rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_report(path: Path, rows: list[dict]) -> None:
    columns = ('pose_id', 'image_key', 'depth_index', 'row_index', 'column_index',
               'cx_mm', 'cy_mm', 'depth_mm', 'contact_radius_mm', 'status',
               'sample_count', 'rejected_reason', 'gpu_render_ms', 'sampling_ms',
               'preview_ms', 'preview_file')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_samples(path: Path, pieces: list[dict], keys: list[str], pose_ids: list[int],
                  grid, metadata: dict, *, reference_grid_color: np.ndarray | None = None,
                  reference_grid_valid: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if not pieces:
        raise ValueError('No calibration pose supplied valid samples')
    fields = ('color_residual_samples', 'slope_samples', 'material_coordinates',
              'st_samples', 'material_grid_index')
    values = {key: np.concatenate([piece[key] for piece in pieces]) for key in fields}
    counts = [len(piece['slope_samples']) for piece in pieces]
    values['image_index'] = np.concatenate([
        np.full(count, index, np.int32) for index, count in enumerate(counts)])
    values['image_paths'] = np.asarray(keys)
    values['pose_ids'] = np.asarray(pose_ids, np.int32)
    count = len(values['slope_samples'])
    values['reference_normal_samples'] = np.broadcast_to(
        np.asarray([0, 0, 1], np.float32), (count, 3)).copy()
    tangent = np.asarray([[1, 0], [0, 1], [0, 0]], np.float32)
    values['tangent_frame_samples'] = np.broadcast_to(tangent, (count, 3, 2)).copy()
    values.update(format_version=np.asarray(1, np.int32),
                  material_grid_x_mm=grid.x_mm,
                  material_grid_y_mm=grid.y_mm,
                  metadata=np.asarray(json.dumps(metadata, ensure_ascii=False,
                                                  default=_json_value)))
    for key, value in calibration_declarations(grid).items():
        values[key] = np.asarray(value)
    values['reference_image'] = np.asarray(
        'analytic_reference:' + metadata['run_signature'])
    values['reference_storage'] = np.asarray('rendered_once_in_memory')
    if reference_grid_color is not None:
        if reference_grid_valid is None or reference_grid_color.shape != (*grid.shape, 3) \
                or reference_grid_valid.shape != grid.shape:
            raise ValueError('Invalid archived reference grid dimensions')
        values['format_version'] = np.asarray(2, np.int32)
        values['reference_grid_color'] = np.asarray(reference_grid_color, np.float64)
        values['reference_grid_valid'] = np.asarray(reference_grid_valid, bool)
        values['reference_storage'] = np.asarray('archived_material_grid')
        source = ('xensim_zero_depth' if metadata['backend'] in ('xense', 'mujoco')
                  else 'zemax_zero_depth')
        values['reference_image'] = np.asarray(source + ':' + metadata['run_signature'])
        values['observation_model_id'] = np.asarray(metadata['observation_model_id'])
        values['backend'] = np.asarray(metadata['backend'])
        values['image_encoding'] = np.asarray(metadata['image_encoding'])
    values['sphere_radius_mm'] = np.asarray(metadata['sphere_radius_mm'], np.float64)
    values['sphere_side'] = np.asarray(metadata['sphere_side'])
    values['boundary_model'] = np.asarray('zero_boundary_displacement')
    _atomic_npz(path, **values)
    return values


def _angle_errors(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    first, second = slopes_to_normals(predicted), slopes_to_normals(truth)
    return np.degrees(np.arccos(np.clip(np.sum(first * second, axis=-1), -1, 1)))


def fit_and_validate(samples: dict[str, np.ndarray], settings: CalibrationSettings,
                     output_dir: Path, metadata: dict, *,
                     pose_split: dict[str, list[str]] | None = None,
                     required_pose_keys: list[str] | None = None):
    colors = np.asarray(samples['color_residual_samples'], np.float32)
    slopes = np.asarray(samples['slope_samples'], np.float32)
    image_index = np.asarray(samples['image_index'])
    keys = np.asarray(samples['image_paths']).astype(str)
    active = sorted(set(image_index.tolist()), key=lambda index: keys[index])
    if len(active) < 2:
        raise ValueError('Whole-image validation requires at least two accepted poses')
    if pose_split is None:
        shuffled = np.asarray(active)
        np.random.default_rng(42).shuffle(shuffled)
        count = min(len(active) - 1,
                    max(1, math.ceil(settings.validation_fraction * len(active))))
        held_out = shuffled[:count]
    else:
        training = list(pose_split['training_images'])
        validation = list(pose_split['validation_images'])
        expected = set(required_pose_keys if required_pose_keys is not None else keys[active])
        if (not training or not validation or len(training) != len(set(training))
                or len(validation) != len(set(validation)) or set(training) & set(validation)
                or (set(training) | set(validation)) != expected):
            raise ValueError('Fixed pose split must be disjoint and cover the required poses exactly')
        held_out = np.asarray([i for i in active if keys[i] in set(validation)], dtype=int)
        train_indices = np.asarray([i for i in active if keys[i] in set(training)], dtype=int)
        if not len(held_out) or not len(train_indices):
            raise ValueError('Fixed pose split has no accepted training or validation poses')
        shuffled = np.r_[held_out, train_indices]
        count = len(held_out)
    validation_mask = np.isin(image_index, held_out)
    training_mask = ~validation_mask
    options = dict(size=settings.lut_size,
                   maximum_rms_angle_degrees=settings.maximum_rms_angle_degrees)
    training_lut = build_normal_lut(colors[training_mask], slopes[training_mask], **options)
    common = dict(metadata,
                  validation_status='training_only',
                  training_images=keys[shuffled[count:]].tolist(),
                  validation_images=keys[held_out].tolist(),
                  total_sample_count=int(training_mask.sum()),
                  original_supported_node_count=int(np.count_nonzero(
                      training_lut.original_valid)),
                  filled_node_fraction=float(1 - np.mean(
                      training_lut.original_valid)))
    training_lut = replace(training_lut, metadata=common)
    training_path = output_dir / 'normal_lut_train.npz'
    training_lut.save(training_path, total_sample_count=int(training_mask.sum()))

    predicted, _, _ = lookup_slopes_raw(colors[validation_mask], training_lut)
    angles = _angle_errors(predicted, slopes[validation_mask])
    out_of_range = ((colors[validation_mask] < training_lut.color_min)
                    | (colors[validation_mask] > training_lut.color_max)).any(axis=-1)
    validation_index = image_index[validation_mask]
    image_rows = []
    for index in held_out:
        selected = validation_index == index
        rms = float(np.sqrt(np.mean(angles[selected] ** 2)))
        fraction = float(np.mean(out_of_range[selected]))
        image_rows.append({
            'image_key': keys[index],
            'sample_count': int(selected.sum()),
            'rms_degrees': rms,
            'median_degrees': float(np.median(angles[selected])),
            'p95_degrees': float(np.percentile(angles[selected], 95)),
            'out_of_range_fraction': fraction,
            'passed': (rms <= settings.validation_rms_degrees
                       and fraction <= settings.maximum_out_of_range_fraction),
        })
    missing = sorted(set(required_pose_keys or []) - set(keys[active]))
    passed = not missing and all(row['passed'] for row in image_rows)
    report = {
        'status': 'passed' if passed else 'failed',
        'missing_required_poses': missing,
        'training_images': common['training_images'],
        'validation_images': common['validation_images'],
        'images': image_rows,
        'rms_degrees': float(np.sqrt(np.mean(angles ** 2))),
        'median_degrees': float(np.median(angles)),
        'p95_degrees': float(np.percentile(angles, 95)),
        'out_of_range_fraction': float(out_of_range.mean()),
        'thresholds': {
            'rms_degrees': settings.validation_rms_degrees,
            'maximum_out_of_range_fraction': settings.maximum_out_of_range_fraction,
        },
        'report_file': str(output_dir / 'verification' / 'validation.json'),
    }
    _atomic_json(output_dir / 'verification' / 'validation.json', report)
    diagnostic = None
    if passed:
        final = build_normal_lut(colors, slopes, **options)
        final_metadata = dict(metadata, validation_status='passed', validation=report,
                              training_images=common['training_images'],
                              validation_images=common['validation_images'],
                              total_sample_count=len(colors),
                              original_supported_node_count=int(np.count_nonzero(
                                  final.original_valid)),
                              filled_node_fraction=float(1 - np.mean(
                                  final.original_valid)))
        final = replace(final, metadata=final_metadata)
        final.save(output_dir / 'normal_lut.npz', total_sample_count=len(colors))
    elif settings.visualization_enabled:
        final = build_normal_lut(colors, slopes, **options)
        final = replace(final, metadata=dict(metadata, validation_status='failed',
                                             validation=report,
                                             total_sample_count=len(colors),
                                             original_supported_node_count=int(
                                                 np.count_nonzero(final.original_valid)),
                                             filled_node_fraction=float(
                                                 1 - np.mean(final.original_valid))))
        diagnostic = output_dir / 'normal_lut_diagnostic.npz'
        final.save(diagnostic, total_sample_count=len(colors))
    else:
        final = None
    return report, training_lut, final, diagnostic


def _dry_run_summary(config: dict, poses: list[ScanPose], grid, settings) -> dict:
    depth_counts = {}
    for pose in poses:
        row = depth_counts.setdefault(str(pose.depth_mm), {
            'depth_mm': pose.depth_mm, 'contact_radius_mm': pose.contact_radius_mm,
            'pose_count': 0})
        row['pose_count'] += 1
    pixels = int(config['camera']['width_px'] * config['camera']['height_px'])
    return {
        'status': 'dry-run',
        'backend': config.get('backend', 'analytic'),
        'pose_count': len(poses),
        'render_count_sampling_pass': len(poses) + 1,
        'render_count_with_visualization': len(poses) + 1,
        'visualization_render_count': 0,
        'material_grid_shape': list(grid.shape),
        'material_grid_points': int(np.prod(grid.shape)),
        'image_pixels': pixels,
        'depths': list(depth_counts.values()),
        'first_pose': poses[0].to_dict(),
        'last_pose': poses[-1].to_dict(),
        'output_dir': str(settings.output_dir),
    }


def load_calibration_config(config_path: str | Path,
                            calibration_config_path: str | Path | None = None) -> dict:
    """Combine one simulator's physics with independently configured LUT workflow."""
    simulation = load_config(config_path)
    if 'calibration' in simulation and calibration_config_path is None:
        return simulation  # Historical full configuration compatibility.
    source = Path(calibration_config_path or CONFIG_DIR / 'calibration.json').expanduser().resolve()
    document = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(document, dict) or set(document) != {'format_version', 'calibration'} \
            or document['format_version'] != 1 or not isinstance(document['calibration'], dict):
        raise ValueError(f'Invalid LUT calibration configuration: {source}')
    if 'calibration' in simulation:
        raise ValueError(f'Simulation config must not embed calibration when using {source}')
    return dict(simulation, calibration=copy.deepcopy(document['calibration']))


def _run_calibration(config_path: str | Path, output_override: str | Path | None = None,
                     *, dry_run: bool = False,
                     calibration_config_path: str | Path | None = None) -> dict:
    config_path = Path(config_path).expanduser().resolve()
    config = load_calibration_config(config_path, calibration_config_path)
    settings = parse_calibration_settings(
        config, config_path, Path(output_override) if output_override is not None else None)
    if config.get('backend') in ('xense', 'mujoco'):
        from .mujoco_calibration import run_mujoco_calibration
        return run_mujoco_calibration(config, config_path, settings, dry_run=dry_run)
    poses = build_scan_poses(config)
    grid = build_material_grid(config)
    summary = _dry_run_summary(config, poses, grid, settings)
    if dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary
    if config.get('backend') == 'zemax':
        from .zemax_calibration import run_zemax_calibration

        return run_zemax_calibration(config, settings, poses, grid)
    if not str(config['render'].get('device', 'cuda:0')).startswith('cuda'):
        raise ValueError('Full calibration requires render.device to be a CUDA device')
    _ensure_new_output(settings.output_dir)

    from .renderer_gpu import AnalyticGpuContext
    import torch
    run_signature = _config_signature(config)
    metadata = dict(calibration_kind='simulated_sphere_scan',
                    light_geometry_model=LIGHT_GEOMETRY_MODEL,
                    upstream_commit=UPSTREAM_COMMIT,
                    run_signature=run_signature,
                    simulation_config_signature=run_signature,
                    sphere_radius_mm=float(config['indentation']['radius_mm']),
                    sphere_direction_z=int(normal_direction(config)),
                    sphere_side=('positive_normal'
                                 if normal_direction(config) == -1
                                 else 'negative_normal'),
                    boundary_model='zero_boundary_displacement',
                    scan=copy.deepcopy(config['calibration']['scan']),
                    reference_color_source='analytic_zero_depth_srgb8_roundtrip',
                    fixed_reference_normal=[0., 0., 1.],
                    fixed_tangent_frame=[[1., 0.], [0., 1.], [0., 0.]],
                    render=dict(device=config['render']['device'],
                                geometry_dtype=config['render']['geometry_dtype'],
                                radiometry_dtype=config['render']['radiometry_dtype'],
                                torch_version=torch.__version__,
                                cuda_version=torch.version.cuda),
                    **calibration_declarations(grid))
    manifest = {
        'format_version': 1,
        'status': 'running',
        'run_status': 'running',
        'stage': 'sampling',
        'run_signature': run_signature,
        'light_geometry_model': LIGHT_GEOMETRY_MODEL,
        'config_path': str(config_path),
        'config': config,
        'reference': {'storage': 'rendered_once_in_memory', 'depth_mm': 0.0},
        'preview_semantics': {
            'kind': 'diagnostic_signed_color_difference',
            'used_as_calibration_input': False,
            'source_image_and_verification_share_one_file': True,
        },
        'poses': [],
        'artifacts': {},
    }
    _atomic_json(settings.output_dir / 'manifest.json', manifest)
    rows, pieces, accepted_keys, accepted_pose_ids = [], [], [], []
    start_total = time.perf_counter()
    with torch.inference_mode():
        context = AnalyticGpuContext(config)
        torch.cuda.reset_peak_memory_stats(context.device)
        torch.cuda.synchronize(context.device)
        start_reference = time.perf_counter()
        reference = render_reference(context, grid)
        torch.cuda.synchronize(context.device)
        reference_seconds = time.perf_counter() - start_reference
        for sequence, pose in enumerate(poses, 1):
            frame = sample_pose(context, reference, grid, pose)
            render_seconds = frame.render_seconds
            sampling_seconds = frame.sampling_seconds
            start_preview = time.perf_counter()
            compact = compact_frame_samples(frame, grid)
            status = ('accepted' if frame.sample_count >= settings.minimum_samples_per_pose
                      else 'rejected_insufficient_samples')
            rejected_reason = '' if status == 'accepted' \
                else f'{frame.sample_count} < {settings.minimum_samples_per_pose}'
            preview = settings.output_dir / 'verification' / f'{pose.image_key}.png'
            diagnostics = image_space_diagnostics(context, reference, frame)
            save_verification_preview(context, reference, frame, preview, settings, status,
                                      diagnostics=diagnostics)
            preview_seconds = time.perf_counter() - start_preview
            if status == 'accepted':
                pieces.append(compact)
                accepted_keys.append(pose.image_key)
                accepted_pose_ids.append(pose.pose_id)
            row = dict(pose_id=pose.pose_id, image_key=pose.image_key,
                       depth_index=pose.depth_index, row_index=pose.row_index,
                       column_index=pose.column_index, cx_mm=pose.center_xy_mm[0],
                       cy_mm=pose.center_xy_mm[1], depth_mm=pose.depth_mm,
                       contact_radius_mm=pose.contact_radius_mm, status=status,
                       sample_count=frame.sample_count, rejected_reason=rejected_reason,
                       gpu_render_ms=1000 * render_seconds,
                       sampling_ms=1000 * sampling_seconds,
                       preview_ms=1000 * preview_seconds,
                       preview_file=str(preview.relative_to(settings.output_dir)))
            rows.append(row)
            manifest['poses'].append({**pose.to_dict(), 'status': status,
                                      'image_index': (len(pieces) - 1
                                                      if status == 'accepted' else None),
                                      'sample_count': frame.sample_count,
                                      'rejected_reason': rejected_reason,
                                      'source_image': row['preview_file'],
                                      'verification_image': row['preview_file']})
            print(f'[{sequence}/{len(poses)}] {pose.image_key}: {status}, '
                  f'{frame.sample_count} samples, {render_seconds:.3f}s', flush=True)

        sample_metadata = dict(metadata,
                               accepted_pose_count=len(pieces),
                               rejected_pose_count=len(poses) - len(pieces),
                               rejected_pose_reasons=dict(Counter(
                                   row['rejected_reason'] for row in rows
                                   if row['rejected_reason'])),
                               pose_key_by_image_index=accepted_keys,
                               pose_id_by_image_index=accepted_pose_ids,
                               material_grid_shape=list(grid.shape))
        samples = _save_samples(settings.output_dir / 'normal_samples.npz', pieces,
                                accepted_keys, accepted_pose_ids, grid, sample_metadata)
        save_ambiguity_inputs(settings.output_dir / 'ambiguity_inputs.npz', samples,
                              settings.output_dir / 'normal_samples.npz')
        _write_report(settings.output_dir / 'verification' / 'report.csv', rows)
        # Persist sampling progress before fitting or potentially lengthy diagnostics.
        manifest['stage'] = 'fit_and_validate'
        manifest['artifacts'].update(samples='normal_samples.npz',
                                      ambiguity_inputs='ambiguity_inputs.npz')
        _atomic_json(settings.output_dir / 'manifest.json', manifest)
        report, training_lut, final_lut, diagnostic = fit_and_validate(
            samples, settings, settings.output_dir, sample_metadata)

        visualization_path = None
        visualization_error = None
        if settings.visualization_enabled and final_lut is not None:
            manifest['stage'] = 'visualization'
            manifest['validation'] = report
            _atomic_json(settings.output_dir / 'manifest.json', manifest)
            try:
                from .calibration_visualization import generate_calibration_visualizations
                visualization_path = generate_calibration_visualizations(
                    samples, final_lut, training_lut, settings,
                    settings.output_dir / 'visualization')
            except Exception as exc:
                visualization_error = f'{type(exc).__name__}: {exc}'

    total_seconds = time.perf_counter() - start_total
    manifest['status'] = report['status']
    manifest['run_status'] = 'completed'
    manifest['stage'] = 'completed'
    manifest['validation'] = report
    manifest['artifacts'] = {
        'samples': 'normal_samples.npz',
        'ambiguity_inputs': 'ambiguity_inputs.npz',
        'training_lut': 'normal_lut_train.npz',
        'formal_lut': 'normal_lut.npz' if report['status'] == 'passed' else None,
        'diagnostic_lut': diagnostic.name if diagnostic else None,
        'verification_report': 'verification/report.csv',
        'validation': 'verification/validation.json',
        'visualization': (str(visualization_path.relative_to(settings.output_dir))
                          if visualization_path else None),
    }
    compilation = context.compile_metrics()
    manifest['performance'] = {
        'reference_seconds': reference_seconds,
        'sampling_pose_median_seconds': float(np.median(
            [row['gpu_render_ms'] for row in rows])) / 1000,
        'sampling_pose_p95_seconds': float(np.percentile(
            [row['gpu_render_ms'] for row in rows], 95)) / 1000,
        'material_sampling_median_seconds': float(np.median(
            [row['sampling_ms'] for row in rows])) / 1000,
        'preview_and_host_copy_median_seconds': float(np.median(
            [row['preview_ms'] for row in rows])) / 1000,
        'total_seconds': total_seconds,
        'cuda_max_memory_bytes': int(torch.cuda.max_memory_allocated(context.device)),
        'device': str(context.device),
        'effective_pixel_chunk_size': context.pixel_chunk_size,
        'effective_light_chunk_size': context.light_chunk_size,
        'cuda_oom_retries': context.oom_retry_count,
        'cuda_graphs_requested': context.cuda_graphs,
        'cuda_graph_capture_count': context.cuda_graph_capture_count,
        'cuda_graph_replay_count': context.cuda_graph_replay_count,
        'cuda_graph_fallback_reason': context.cuda_graph_fallback_reason,
        'compile_lighting_requested': compilation['requested'],
        'compile_lighting_effective': compilation['effective'],
        'compile_backend': compilation['backend'],
        'compile_region': compilation['region'],
        'compile_options': compilation['options'],
        'compile_versions': compilation['versions'],
        'compile_variant_count': compilation['variant_count'],
        'compile_first_call_seconds': compilation['first_call_seconds'],
        'compile_prepare_seconds': compilation['prepare_seconds'],
        'compile_warmup_seconds': compilation['warmup_seconds'],
        'compile_graph_capture_seconds': compilation['graph_capture_seconds'],
        'compile_fallback_reason': compilation['fallback_reason'],
        'compile_failure_stage': compilation['failure_stage'],
        'compile_validation_status': compilation['validation_status'],
        'sampling_render_count': len(poses) + 1,
        'visualization_render_count': 0,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
    }
    manifest['visualization_status'] = ('disabled' if not settings.visualization_enabled
                                        else 'failed' if visualization_error else 'passed')
    if visualization_error:
        manifest['visualization_error'] = visualization_error
    _atomic_json(settings.output_dir / 'manifest.json', manifest)
    if visualization_error:
        raise RuntimeError('LUT processing completed, but visualization failed: '
                           + visualization_error)
    if report['status'] != 'passed':
        raise CalibrationValidationFailed(report)
    return manifest


def run_calibration(config_path: str | Path, output_override: str | Path | None = None,
                    *, dry_run: bool = False,
                    calibration_config_path: str | Path | None = None) -> dict:
    """Run calibration and make any post-creation failure explicit in manifest."""
    try:
        return _run_calibration(config_path, output_override, dry_run=dry_run,
                                calibration_config_path=calibration_config_path)
    except Exception as exc:
        if not dry_run:
            try:
                resolved_config = Path(config_path).expanduser().resolve()
                config = load_calibration_config(resolved_config, calibration_config_path)
                settings = parse_calibration_settings(
                    config, resolved_config,
                    Path(output_override) if output_override is not None else None)
                manifest_path = settings.output_dir / 'manifest.json'
                # MuJoCo owns resumable checkpoints. In particular, a rejected
                # incompatible resume must not rewrite the previous manifest.
                if manifest_path.exists() and config.get('backend') not in ('xense', 'mujoco'):
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                    if manifest.get('status') == 'running':
                        manifest['status'] = 'failed'
                    manifest['run_status'] = 'failed'
                    manifest['error'] = {
                        'type': type(exc).__name__,
                        'message': str(exc),
                    }
                    _atomic_json(manifest_path, manifest)
            except Exception:
                # Preserve the original failure if even the diagnostic update fails.
                pass
        raise

