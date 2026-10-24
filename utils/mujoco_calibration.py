"""Sphere scan and LUT fitting using MuJoCo depth and Xensim RGB."""
from __future__ import annotations

from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
from .surface_coordinates import center_material_mm
import time

import numpy as np
from PIL import Image

from .calibration_ambiguity import save_ambiguity_inputs
from .calibration_pipeline import (_atomic_json, _atomic_npz, _save_samples,
                                   fit_and_validate, CalibrationValidationFailed)
from .calibration_samples import build_material_grid, calibration_declarations
from .calibration_scan import build_scan_poses
from .mujoco_backend import (GEOMETRY_MODEL_ID, OBSERVATION_MODEL_ID,
                             MujocoSession, check_mujoco_runtime_paths,
                             simulation_signature)
from .mujoco_samples import depth_geometry_error, pose_samples, reference_grid
from .normal_mapping import UPSTREAM_COMMIT


def _archive_piece(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def run_mujoco_calibration(config: dict, config_path: Path, settings, *, dry_run: bool) -> dict:
    poses = build_scan_poses(config)
    grid = build_material_grid(config)
    output = settings.output_dir
    summary = {'status': 'dry-run', 'backend': 'xense', 'pose_count': len(poses),
               'render_count_sampling_pass': len(poses) + 1,
               'material_grid_shape': list(grid.shape),
               'material_grid_points': int(np.prod(grid.shape)),
               'image_pixels': int(np.prod(config['mujoco']['render_size_wh'])),
               'first_pose': poses[0].to_dict(), 'last_pose': poses[-1].to_dict(),
               'output_dir': str(output), 'runtime_required': False,
               'unsupported_parameter_sweeps': ['LED strip length/power/position',
                                                'arbitrary gel size without matched optical table']}
    if dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary
    check_mujoco_runtime_paths(config, config_path)

    existing = output.exists() and any(output.iterdir())
    saved_manifest = None
    if existing:
        manifest_path = output / 'manifest.json'
        if not manifest_path.is_file():
            raise ValueError(f'Nonempty MuJoCo output has no resumable manifest: {output}')
        saved_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if saved_manifest.get('stage') == 'completed':
            raise ValueError(f'MuJoCo calibration already completed: {output}')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'verification').mkdir(exist_ok=True)
    (output / 'pose_checkpoints').mkdir(exist_ok=True)
    start = time.perf_counter()
    signature = simulation_signature(config)
    with MujocoSession(config, config_path, output / 'worker') as session:
        worker_script = Path(__file__).resolve().parents[1] / 'integrations' / 'xense_mujoco_runtime' / 'worker.py'
        identity = {'simulation_config_signature': signature,
                    'calibration_table_sha256': session.handshake['calibration_table_sha256'],
                    'worker_sha256': hashlib.sha256(worker_script.read_bytes()).hexdigest(),
                    'sampler_sha256': hashlib.sha256(
                        (Path(__file__).with_name('mujoco_samples.py')).read_bytes()).hexdigest(),
                    'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'fitter_sha256': hashlib.sha256(
                        (Path(__file__).with_name('normal_mapping.py')).read_bytes()).hexdigest(),
                    'versions': session.handshake['versions'],
                    'geometry_model_id': GEOMETRY_MODEL_ID,
                    'observation_model_id': OBSERVATION_MODEL_ID}
        run_signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if saved_manifest is not None and (saved_manifest.get('format_version') != 2 or
                                           saved_manifest.get('run_signature') != run_signature or
                                           saved_manifest.get('identity') != identity):
            raise ValueError('MuJoCo resume manifest is incompatible with configuration, '
                             'runtime versions, optical table, or worker implementation')
        if saved_manifest is not None and len(saved_manifest.get('poses', [])) > len(poses):
            raise ValueError('MuJoCo resume manifest contains more poses than the current scan')
        metadata = dict(backend='xense', sensor_mode='simple',
                        calibration_kind='simulated_sphere_scan',
                        light_geometry_model=GEOMETRY_MODEL_ID,
                        geometry_model_id=GEOMETRY_MODEL_ID,
                        observation_model_id=OBSERVATION_MODEL_ID,
                        upstream_commit=UPSTREAM_COMMIT,
                        run_signature=run_signature,
                        simulation_config_signature=signature,
                        sphere_radius_mm=float(config['indentation']['radius_mm']),
                        sphere_direction_z=-1, sphere_side='positive_normal',
                        boundary_model='zero_boundary_displacement',
                        scan=copy.deepcopy(config['calibration']['scan']),
                        reference_color_source='xensim_simple_zero_depth_rgb8',
                        fixed_reference_normal=[0., 0., 1.],
                        fixed_tangent_frame=[[1., 0.], [0., 1.], [0., 0.]],
                        marker_policy='require_absent',
                        marker_mode='SimpleSensor_without_marker_overlay',
                        asset_sha256=session.handshake['calibration_table_sha256'],
                        runtime_versions=session.handshake['versions'],
                        **{**calibration_declarations(grid),
                           'image_encoding': 'xensim_rgb8_inverse_srgb_analysis'})
        manifest = saved_manifest if saved_manifest is not None else {
                    'format_version': 2, 'status': 'running', 'run_status': 'running',
                    'stage': 'reference', 'backend': 'xense', 'run_signature': run_signature,
                    'identity': identity, 'config_path': str(config_path), 'config': config,
                    'reference': {'storage': 'archived_material_grid', 'depth_mm': 0},
                    'poses': [], 'artifacts': {},
                    'marker_policy': 'require_absent'}
        if saved_manifest is None:
            _atomic_json(output / 'manifest.json', manifest)
            flat = session.render(center_material_mm(config), 0)
            Image.fromarray(flat['rgb8']).save(output / 'reference.png')
            ref_color, ref_valid = reference_grid(flat, grid, config)
            _atomic_npz(output / 'reference_grid.npz', color=ref_color,
                        valid=ref_valid, raw_rgb8=flat['rgb8'])
            manifest['reference']['rgb8_sha256'] = hashlib.sha256(flat['rgb8'].tobytes()).hexdigest()
            manifest['reference']['valid_grid_points'] = int(ref_valid.sum())
        else:
            reference = _archive_piece(output / 'reference_grid.npz')
            ref_color, ref_valid = reference['color'], reference['valid']
            if (ref_color.shape != (*grid.shape, 3) or ref_valid.shape != grid.shape or
                    hashlib.sha256(reference['raw_rgb8'].tobytes()).hexdigest() !=
                    manifest['reference'].get('rgb8_sha256') or
                    not (output / 'reference.png').is_file()):
                raise ValueError('MuJoCo resume reference archive is incomplete or incompatible')
        manifest['stage'] = 'sampling'
        manifest['status'] = 'running'
        manifest['run_status'] = 'running'
        _atomic_json(output / 'manifest.json', manifest)

        rows, pieces, keys, pose_ids = [], [], [], []
        for sequence, pose in enumerate(poses, 1):
            if sequence <= len(manifest['poses']):
                row = manifest['poses'][sequence - 1]
                if row.get('image_key') != pose.image_key or row.get('pose_id') != pose.pose_id \
                        or row.get('center_xy_mm') != list(pose.center_xy_mm) \
                        or row.get('depth_mm') != pose.depth_mm \
                        or not (output / row.get('preview_file', '')).is_file():
                    raise ValueError(f'MuJoCo resume pose checkpoint mismatch: {pose.image_key}')
                if row['status'] == 'accepted':
                    checkpoint = output / row['checkpoint']
                    compact = _archive_piece(checkpoint)
                    if len(compact['slope_samples']) != row['sample_count']:
                        raise ValueError(f'MuJoCo resume sample count mismatch: {checkpoint}')
                    pieces.append(compact)
                    keys.append(pose.image_key)
                    pose_ids.append(pose.pose_id)
                elif row['status'] != 'rejected_insufficient_samples':
                    raise ValueError(f'MuJoCo resume has unknown pose status: {row["status"]}')
                rows.append(row)
                continue
            pose_start = time.perf_counter()
            frame = session.render(pose.center_xy_mm, pose.depth_mm)
            geometry = depth_geometry_error(frame, config, pose)
            compact, diagnostic = pose_samples(frame, ref_color, ref_valid, grid, config, pose)
            count = diagnostic['sample_count']
            accepted = count >= settings.minimum_samples_per_pose
            status = 'accepted' if accepted else 'rejected_insufficient_samples'
            preview = output / 'verification' / f'{pose.image_key}.png'
            Image.fromarray(frame['rgb8']).save(preview)
            if accepted:
                checkpoint = output / 'pose_checkpoints' / f'{pose.image_key}.npz'
                _atomic_npz(checkpoint, **compact)
                pieces.append(compact)
                keys.append(pose.image_key)
                pose_ids.append(pose.pose_id)
            row = dict(**pose.to_dict(), status=status, sample_count=count,
                       geometry=geometry, diagnostics=diagnostic,
                       render_and_sample_seconds=time.perf_counter() - pose_start,
                       preview_file=str(preview.relative_to(output)),
                       checkpoint=(f'pose_checkpoints/{pose.image_key}.npz' if accepted else None))
            rows.append(row)
            manifest['poses'].append(row)
            _atomic_json(output / 'manifest.json', manifest)
            print(f'[{sequence}/{len(poses)}] {pose.image_key}: {status}, '
                  f'{count} samples, {row["render_and_sample_seconds"]:.3f}s', flush=True)

        sample_metadata = dict(metadata, accepted_pose_count=len(pieces),
                               rejected_pose_count=len(poses) - len(pieces),
                               rejected_pose_reasons=dict(Counter(
                                   row['status'] for row in rows if row['status'] != 'accepted')),
                               pose_key_by_image_index=keys,
                               pose_id_by_image_index=pose_ids,
                               material_grid_shape=list(grid.shape))
        samples = _save_samples(output / 'normal_samples.npz', pieces, keys, pose_ids,
                                grid, sample_metadata,
                                reference_grid_color=ref_color,
                                reference_grid_valid=ref_valid)
        save_ambiguity_inputs(output / 'ambiguity_inputs.npz', samples,
                              output / 'normal_samples.npz')
        with (output / 'verification' / 'report.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['pose_id', 'image_key', 'status',
                                                        'sample_count', 'geometry_max_abs_mm',
                                                        'geometry_p95_abs_mm', 'seconds'])
            writer.writeheader()
            for row in rows:
                writer.writerow({'pose_id': row['pose_id'], 'image_key': row['image_key'],
                                 'status': row['status'], 'sample_count': row['sample_count'],
                                 'geometry_max_abs_mm': row['geometry']['max_abs_mm'],
                                 'geometry_p95_abs_mm': row['geometry']['p95_abs_mm'],
                                 'seconds': row['render_and_sample_seconds']})
        _atomic_json(output / 'verification' / 'marker_check.json', {
            'policy': 'require_absent', 'native_export': 'SimpleSensor.get_image',
            'mode': 'simple', 'pixel_modification': 'none',
            'automated_marker_detection': 'not_available_from_public_SDK',
            'visual_inspection': 'pending',
            'raw_evidence': ['reference.png',
                             rows[0]['preview_file'], rows[-1]['preview_file']],
            'runtime_versions': session.handshake['versions'],
            'optical_table_sha256': session.handshake['calibration_table_sha256'],
        })
        manifest['stage'] = 'fit_and_validate'
        _atomic_json(output / 'manifest.json', manifest)
        report, training_lut, final_lut, diagnostic = fit_and_validate(
            samples, settings, output, sample_metadata)
        visualization = None
        if settings.visualization_enabled and final_lut is not None:
            from .calibration_visualization import generate_calibration_visualizations
            visualization = generate_calibration_visualizations(
                samples, final_lut, training_lut, settings, output / 'visualization')
        manifest['status'] = report['status']
        manifest['run_status'] = 'completed'
        manifest['stage'] = 'completed'
        manifest['validation'] = report
        manifest['performance'] = {'total_seconds': time.perf_counter() - start,
                                   'render_count': len(poses) + 1}
        manifest['artifacts'] = {'samples': 'normal_samples.npz',
                                 'reference': 'reference_grid.npz',
                                 'training_lut': 'normal_lut_train.npz',
                                 'formal_lut': 'normal_lut.npz' if report['status'] == 'passed' else None,
                                 'diagnostic_lut': diagnostic.name if diagnostic else None,
                                 'verification_report': 'verification/report.csv',
                                 'marker_report': 'verification/marker_check.json',
                                 'validation': 'verification/validation.json',
                                 'visualization': str(visualization.relative_to(output))
                                 if visualization else None}
        _atomic_json(output / 'manifest.json', manifest)
        if report['status'] != 'passed':
            raise CalibrationValidationFailed(report)
        return manifest
