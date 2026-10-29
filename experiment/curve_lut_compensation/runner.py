"""Validate, migrate, fit, and compare curved-surface normal LUT compensation."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import uuid

import numpy as np

from experiment.config import load_experiment_config, resolve_curve_lut_compensation_settings
from utils.backend_layout import CONFIG_DIR
from utils.calibration_pipeline import parse_calibration_settings
from utils.curve_calibration import (SLOPE_FRAME_ID, SAMPLING_VERSION,
                                     canonical_snapshot, reference_area_weights,
                                     reference_grid)
from utils.lut_results import load_lut_result
from utils.normal_mapping import NormalCalibration, lookup_slopes_raw
from experiment.curve_lut_transfer.storage import (Checkpoints, digest, json_write,
                                                    load_arrays, npz_write, signature)
from experiment.curve_lut_transfer.metrics import (aggregate, evaluate_pose,
                                                    paired_bootstrap, support,
                                                    weighted_quantile)
from .transfer import (MIGRATION_IMAGE_ENCODING, MIGRATION_SAMPLING_VERSION,
                       condition_id, fit_migrated_condition, migrate_pose_samples,
                       pose_config, reference_geometry, target_valid_grid_indices)


VERSION = 'curve_lut_compensation_v1'
DEFAULT_EXP_CONFIG = CONFIG_DIR / 'curve_lut_compensation.json'
RESUME_INPUT_FIELDS = ('source', 'source_files', 'history', 'history_files', 'settings',
                       'source_config', 'split', 'training_poses', 'test_poses', 'conditions')


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def _normal_path(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def _checkpoint_hash_index(manifest: dict) -> dict[str, str]:
    index = manifest.get('_codex_path_hash_index')
    if index is None:
        index = {}
        for entry in manifest.get('completed', {}).values():
            for saved, value in entry.get('files', {}).items():
                key = _normal_path(saved)
                if key in index and index[key] != value:
                    raise ValueError(f'Historical transfer contains conflicting hashes for {saved}')
                index[key] = value
        manifest['_codex_path_hash_index'] = index
    return index


def _checkpoint_file_hash(manifest: dict, path: Path) -> str | None:
    return _checkpoint_hash_index(manifest).get(_normal_path(path))


def _verify_checkpoint_file(manifest: dict, path: Path, *, hash_now: bool = True) -> str:
    if not path.is_file():
        raise FileNotFoundError(f'Historical transfer artifact is missing: {path}')
    expected = _checkpoint_file_hash(manifest, path)
    if expected is None:
        raise ValueError(f'Historical transfer manifest does not checkpoint the consumed artifact: {path}')
    if hash_now:
        actual = digest(path)
        if actual != expected:
            raise ValueError(f'Historical transfer artifact checksum differs from its manifest: {path}')
    return expected


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != 'metadata'}
        if 'metadata' not in archive.files:
            metadata = {}
        else:
            raw = archive['metadata']
            metadata = json.loads(str(raw.item() if raw.shape == () else raw))
    return arrays, metadata


def _pose_entries(manifest: dict, base_config: dict) -> list[dict]:
    width = np.asarray(base_config['surface']['size_mm'], dtype=np.float64)
    poses = []
    for entry in manifest.get('poses', []):
        key = entry.get('image_key')
        if entry.get('status') != 'accepted':
            raise ValueError(f'Source LUT has an unaccepted pose: {key}')
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', key):
            raise ValueError(f'Unsafe or unsupported source pose key: {key!r}')
        uv = entry.get('center_uv')
        if uv is None:
            center = np.asarray(entry.get('center_xy_mm'), dtype=np.float64)
            if center.shape != (2,) or not np.isfinite(center).all():
                raise ValueError(f'Invalid source anchor for pose {key}')
            uv = (center * 2 / width).tolist()
        uv = np.asarray(uv, dtype=np.float64)
        depth = entry.get('depth_mm')
        if uv.shape != (2,) or not np.isfinite(uv).all() or isinstance(depth, bool) \
                or not isinstance(depth, (int, float)) or not np.isfinite(depth) or depth <= 0:
            raise ValueError(f'Invalid source pose geometry for {key}')
        poses.append(dict(pose_id=key, center_uv=uv.tolist(), depth_mm=float(depth), anchor_id=str(uv.tolist())))
    if not poses or len({pose['pose_id'] for pose in poses}) != len(poses):
        raise ValueError('Source manifest must contain unique accepted calibration poses')
    return poses


def _assert_same_pose_geometry(a: list[dict], b: list[dict], description: str) -> None:
    if len(a) != len(b):
        raise ValueError(f'{description}: pose count differs ({len(a)} vs {len(b)})')
    for index, (first, second) in enumerate(zip(a, b, strict=True)):
        if first.get('pose_id') != second.get('pose_id') \
                or not np.allclose(first.get('center_uv'), second.get('center_uv'), rtol=0, atol=1e-12) \
                or not np.isclose(first.get('depth_mm'), second.get('depth_mm'), rtol=0, atol=1e-12):
            raise ValueError(f'{description}: pose geometry differs at index {index}')


def _load_history_reference(history_root: Path, manifest: dict, condition: str,
                            grid_shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    path = history_root / 'samples' / condition / 'reference.npz'
    expected = _verify_checkpoint_file(manifest, path)
    with np.load(path, allow_pickle=False) as archive:
        if 'valid' not in archive.files:
            raise ValueError(f'Historical reference is missing its material validity grid: {path}')
        valid = np.asarray(archive['valid'], dtype=bool)
    if valid.shape != grid_shape:
        raise ValueError(f'Historical reference grid shape {valid.shape} does not match {grid_shape}: {path}')
    return valid, expected


def _history_calibration(history_root: Path, manifest: dict, condition: str) -> tuple[NormalCalibration | None, dict]:
    info = manifest.get('calibration', {}).get(condition)
    if not isinstance(info, dict):
        raise ValueError(f'Historical transfer has no calibration status for {condition}')
    lut = None
    relative = info.get('lut')
    if relative:
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'Unsafe historical LUT path for {condition}: {relative}')
        path = (history_root / rel).resolve()
        if not path.is_relative_to(history_root.resolve()):
            raise ValueError(f'Historical LUT path escapes transfer results: {relative}')
        key = f'{condition}/calibration'
        saved = manifest.get('completed', {}).get(key, {}).get('files', {})
        expected = next((value for name, value in saved.items()
                         if _normal_path(name) == _normal_path(path)), None)
        if expected is None:
            raise ValueError(f'Historical transfer does not checkpoint the LUT used for {condition}: {path}')
        if digest(path) != expected:
            raise ValueError(f'Historical LUT checksum differs from its manifest: {path}')
        lut = NormalCalibration.load(path)
    elif info.get('status') not in ('failed', 'unavailable', 'incomplete'):
        raise ValueError(f'Historical calibration {condition} has no LUT but status is {info.get("status")!r}')
    return lut, copy.deepcopy(info)


def prepare(lut_results: Path, transfer_results: Path, settings: dict) -> dict:
    """Read and validate all immutable source/history inputs without rendering or CUDA."""
    root, source_manifest, backend = load_lut_result(lut_results)
    if backend != 'analytic':
        raise ValueError(f'curve_lut_compensation currently supports analytic LUTs only; got {backend}')
    source_lut = NormalCalibration.load(root / 'normal_lut.npz')
    source_metadata = source_lut.metadata
    expected_meta = {'color_space': 'linear_rgb', 'image_encoding': 'srgb8_roundtrip',
                     'coordinate_kind': 'reference_arclength', 'residual_domain': 'material_grid'}
    for key, expected in expected_meta.items():
        if source_metadata.get(key) != expected:
            raise ValueError(f'{root}: unsupported source LUT {key}={source_metadata.get(key)!r}; expected {expected!r}')
    for key, expected in (('fixed_reference_normal', [0., 0., 1.]),
                          ('fixed_tangent_frame', [[1., 0.], [0., 1.], [0., 0.]])):
        observed = np.asarray(source_metadata.get(key), dtype=np.float64)
        if observed.shape != np.shape(expected) or not np.allclose(observed, expected, rtol=0, atol=1e-12):
            raise ValueError(f'{root}: incompatible source tangent-frame metadata {key}')
    if source_lut.original_valid is None:
        raise ValueError(f'{root}: source LUT must store original_valid for support comparisons')
    base = canonical_snapshot(source_manifest['config'])
    if base['calibration']['sampling'].get('contact_margin_mm', 0.) != 0.:
        raise ValueError('Source calibration requires contact_margin_mm=0 for this transfer protocol')
    parse_calibration_settings(base, root / 'config.json', root)
    source_arrays, data_metadata = _load_npz(root / 'normal_samples.npz')
    required_arrays = ('color_residual_samples', 'slope_samples', 'material_coordinates',
                       'material_grid_index', 'image_index', 'image_paths',
                       'material_grid_x_mm', 'material_grid_y_mm')
    missing = [key for key in required_arrays if key not in source_arrays]
    if missing:
        raise ValueError(f'{root / "normal_samples.npz"}: missing arrays {missing}')
    sample_count = len(source_arrays['color_residual_samples'])
    if source_arrays['color_residual_samples'].shape != (sample_count, 3) \
            or source_arrays['slope_samples'].shape != (sample_count, 2) \
            or source_arrays['material_coordinates'].shape != (sample_count, 2) \
            or source_arrays['material_grid_index'].shape != (sample_count, 2) \
            or source_arrays['image_index'].shape != (sample_count,):
        raise ValueError('Source normal_samples arrays have inconsistent shapes')
    if not np.issubdtype(source_arrays['material_grid_index'].dtype, np.integer):
        raise ValueError('Source material_grid_index must be an integer row/column array')
    if source_arrays['material_grid_x_mm'].ndim != 1 or source_arrays['material_grid_y_mm'].ndim != 1:
        raise ValueError('Source material grid axes must be one-dimensional')
    grid_shape = (len(source_arrays['material_grid_y_mm']), len(source_arrays['material_grid_x_mm']))
    ij = source_arrays['material_grid_index']
    if np.any(ij[:, 0] < 0) or np.any(ij[:, 0] >= grid_shape[0]) \
            or np.any(ij[:, 1] < 0) or np.any(ij[:, 1] >= grid_shape[1]):
        raise ValueError('Source material_grid_index has out-of-range row/column coordinates')
    source_pose_ids = np.asarray(source_arrays['image_paths']).astype(str).tolist()
    poses = _pose_entries(source_manifest, base)
    if source_pose_ids != [pose['pose_id'] for pose in poses]:
        raise ValueError('Source normal_samples.image_paths does not match manifest pose order')
    image_index = np.asarray(source_arrays['image_index'], dtype=np.int64)
    if np.any(image_index < 0) or np.any(image_index >= len(poses)):
        raise ValueError('Source normal_samples.image_index references an unknown pose')
    if any(not np.isfinite(source_arrays[key]).all() for key in
           ('color_residual_samples', 'slope_samples', 'material_coordinates')):
        raise ValueError('Source calibration samples contain non-finite colors, slopes, or coordinates')
    for key, expected in expected_meta.items():
        value = data_metadata.get(key, source_arrays.get(key))
        if key == 'color_space' and value is None:
            value = source_arrays.get('color_space')
            value = str(value) if value is not None else None
        if key == 'image_encoding' and value is None:
            value = source_arrays.get('image_encoding')
            value = str(value) if value is not None else None
        if key == 'coordinate_kind' and value is None:
            value = source_arrays.get('coordinate_kind')
            value = str(value) if value is not None else None
        if key == 'residual_domain' and value is None:
            value = source_arrays.get('residual_domain')
            value = str(value) if value is not None else None
        if value is not None and value != expected:
            raise ValueError(f'Source normal_samples {key}={value!r} conflicts with LUT metadata')

    history_root = Path(transfer_results).expanduser().resolve()
    history_manifest_path = history_root / 'manifest.json'
    history_snapshot_path = history_root / 'config_snapshot.json'
    history_split_path = history_root / 'split.json'
    history_metrics_path = history_root / 'metrics.json'
    if not history_manifest_path.is_file() or not history_snapshot_path.is_file() or not history_split_path.is_file():
        raise FileNotFoundError(f'Historical transfer requires manifest.json, config_snapshot.json, and split.json: {history_root}')
    history_manifest = _load_json(history_manifest_path)
    if history_manifest.get('status') != 'completed':
        raise ValueError(f'Historical transfer must be completed; status={history_manifest.get("status")!r}')
    if not history_manifest.get('run_signature') or not isinstance(history_manifest.get('identity'), dict):
        raise ValueError(f'Historical transfer manifest is missing its identity: {history_manifest_path}')
    history_identity = history_manifest['identity']
    history_source = Path(history_identity.get('source', '')).expanduser().resolve()
    if history_source != root:
        raise ValueError(f'Historical transfer source differs from --lut-results: {history_source} != {root}')
    source_file_names = ('manifest.json', 'normal_lut.npz', 'normal_lut_train.npz', 'normal_samples.npz')
    source_files = {name: digest(root / name) for name in source_file_names}
    if history_identity.get('source_files') != source_files:
        changed = [name for name in source_file_names
                   if history_identity.get('source_files', {}).get(name) != source_files[name]]
        raise ValueError(f'Historical transfer was generated from different source files: {changed}')
    history_settings = history_identity.get('settings', {})
    history_radii = history_settings.get('radii_mm')
    if history_radii != settings['radii_mm']:
        raise ValueError(f'Configured radii {settings["radii_mm"]} do not match historical conditions {history_radii}')
    conditions = {condition_id(radius): radius for radius in settings['radii_mm']}
    if len(conditions) != len(settings['radii_mm']):
        raise ValueError('Configured curvature condition IDs collide')
    snapshot = _load_json(history_snapshot_path)
    if signature(snapshot.get('simulation', {})) != signature(history_identity.get('config', {})):
        raise ValueError('Historical config_snapshot simulation differs from its manifest identity')
    if signature(snapshot.get('simulation', {})) != signature(base):
        raise ValueError('Historical physical snapshot differs from the selected source LUT snapshot')
    history_split = _load_json(history_split_path)
    split = history_identity.get('split', {})
    if any(history_split.get(key) != split.get(key) for key in ('training_images', 'validation_images')):
        raise ValueError('Historical split.json differs from the saved source pose split')
    lut_split = {key: source_metadata.get(key) for key in ('training_images', 'validation_images')}
    if lut_split != {key: split.get(key) for key in ('training_images', 'validation_images')}:
        raise ValueError('Source LUT fixed training/validation split differs from the transfer archive')
    history_train = history_identity.get('training_poses', [])
    _assert_same_pose_geometry(poses, history_train, 'Historical transfer training poses')
    test_poses = history_identity.get('test_poses', [])
    archived_test = history_split.get('independent_test', [])
    _assert_same_pose_geometry(test_poses, archived_test, 'Historical transfer independent test poses')
    if not test_poses:
        raise ValueError('Historical transfer has no independent test poses')
    source_signature = source_metadata.get('run_signature')
    if source_signature != source_manifest.get('run_signature'):
        raise ValueError('Source normal LUT metadata does not match its manifest run_signature')

    history_hashes = {
        'manifest.json': digest(history_manifest_path),
        'config_snapshot.json': digest(history_snapshot_path),
        'split.json': digest(history_split_path),
    }
    _verify_checkpoint_file(history_manifest, history_metrics_path)
    if history_metrics_path.is_file():
        history_hashes['metrics.json'] = digest(history_metrics_path)
    grid = reference_grid(base)
    references = {}
    history_luts = {}
    calibration_status = {}
    test_sample_paths = {}
    for name, radius in conditions.items():
        validate = pose_config(base, radius)
        from utils.config import validate_config
        validate_config(validate, allow_curve=True, allow_contact_tangent=True)
        reference_valid, reference_hash = _load_history_reference(history_root, history_manifest,
                                                                  name, grid.shape)
        references[name] = reference_valid
        history_hashes[f'samples/{name}/reference.npz'] = reference_hash
        if radius is None:
            lut = source_lut
            status = dict(status=source_manifest.get('status', 'unknown'),
                          lut=str(root / 'normal_lut.npz'), source='selected source LUT')
        else:
            lut, status = _history_calibration(history_root, history_manifest, name)
        history_luts[name], calibration_status[name] = lut, status
        for pose in test_poses:
            path = history_root / 'samples' / name / 'test' / (pose['pose_id'] + '.npz')
            _verify_checkpoint_file(history_manifest, path, hash_now=False)
            test_sample_paths[f'{name}/{pose["pose_id"]}'] = path

    if not history_metrics_path.is_file():
        raise FileNotFoundError(f'Historical metric report is missing: {history_metrics_path}')
    history_metrics = _load_json(history_metrics_path)
    expected_test_count = len(test_poses)
    split_partition = split.get('training_images', []) + split.get('validation_images', [])
    if len(poses) != len(split_partition) or set(split_partition) != set(source_pose_ids):
        raise ValueError('Saved fixed source split does not cover all source poses')
    if len(image_index) != sample_count:
        raise ValueError('Source sample image index is inconsistent')
    grid_size = grid.shape
    if tuple(grid_size) != tuple(grid_shape):
        raise ValueError('Source material grid axes disagree with the saved simulation configuration')
    source_axes = (np.asarray(source_arrays['material_grid_x_mm'], dtype=np.float64),
                   np.asarray(source_arrays['material_grid_y_mm'], dtype=np.float64))
    if not (np.array_equal(source_axes[0], grid.x_mm) and np.array_equal(source_axes[1], grid.y_mm)):
        raise ValueError('Source normal_samples material grid axes differ from the saved physical snapshot')

    identity = dict(version=VERSION, source=str(root), source_files=source_files,
                    history=str(history_root),
                    history_files=history_hashes,
                    settings=settings, source_config=base, split=split,
                    training_poses=poses, test_poses=test_poses, conditions=conditions,
                    history_run_signature=history_manifest['run_signature'],
                    history_calibration=calibration_status)
    run_signature = signature({key: identity[key] for key in RESUME_INPUT_FIELDS})
    return dict(root=root, source_manifest=source_manifest, source_lut=source_lut,
                source_metadata=source_metadata, source_arrays=source_arrays,
                history_root=history_root, history_manifest=history_manifest,
                history_snapshot=snapshot, history_metrics=history_metrics,
                history_luts=history_luts, calibration_status=calibration_status,
                references=references, test_sample_paths=test_sample_paths,
                base=base, poses=poses, test_poses=test_poses, split=split,
                conditions=conditions, settings=settings, identity=identity,
                run_signature=run_signature,
                workload=dict(source_pose_count=len(poses), source_sample_count=sample_count,
                              source_sample_count_by_pose={pose['pose_id']: int(np.count_nonzero(image_index == i))
                                                           for i, pose in enumerate(poses)},
                              material_grid_shape=list(grid_shape), material_grid_points=int(np.prod(grid_shape)),
                              target_condition_count=len(conditions), test_pose_count=expected_test_count,
                              historical_test_sample_count=len(conditions) * expected_test_count,
                              endpoint_model_evaluations_per_source_point=4 * len(conditions),
                              full_frame_renders=0, target_test_observations_reused=True,
                              source_sample_semantics='all saved source material samples; target labels recomputed',
                              target_test_semantics='historical actual material-paired sRGB8 observations'))


def start_run(prepared: dict, settings: dict, out: Path | None, resume: bool) -> Checkpoints:
    parent = prepared['root'] / 'evaluation' / 'curve_lut_compensation'
    if out is None:
        out = parent / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                        + '_' + uuid.uuid4().hex[:8])
    out = Path(out).expanduser().resolve()
    if not out.is_relative_to(parent.resolve()) or out == parent.resolve():
        raise ValueError(f'--out must be a unique run subdirectory of {parent}')
    if resume:
        path = out / 'manifest.json'
        if not path.is_file():
            raise ValueError(f'--resume requires a saved manifest at {path}')
        manifest = _load_json(path)
        previous = manifest.get('identity', {})
        changed = [key for key in RESUME_INPUT_FIELDS
                   if previous.get(key) != prepared['identity'].get(key)]
        if changed:
            raise ValueError(f'Resume inputs differ ({", ".join(changed)}): {path}')
        checkpoint = Checkpoints(out, manifest)
        checkpoint.verify()
        prepared['run_signature'] = manifest['run_signature']
        return checkpoint
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Nonempty output requires compatible --resume: {out}')
    manifest = dict(version=VERSION, status='running', run_signature=prepared['run_signature'],
                    identity=prepared['identity'], completed={}, calibration={}, evaluation={},
                    training_diagnostics={}, execution={},
                    methods=['frozen_flat_lut', 'endpoint_migrated_lut',
                             'historical_same_curvature_lut', 'matrix_project_lut',
                             'matrix_keep_lut', 'matrix_direct', 'matrix_ridge_project_lut',
                             'matrix_ridge_keep_lut', 'matrix_ridge_direct'],
                    model_semantics='H_pt=display_linear(contact)-display_linear(reference); inverse_sRGB(srgb8) domain')
    checkpoint = Checkpoints(out, manifest)
    checkpoint.save()
    json_write(out / 'config_snapshot.json', dict(simulation=prepared['base'],
             experiment=settings, source=str(prepared['root']), transfer_results=str(prepared['history_root']),
             source_run_signature=prepared['source_manifest'].get('run_signature'),
             transfer_run_signature=prepared['history_manifest'].get('run_signature')))
    json_write(out / 'split.json', dict(prepared['split'], independent_test=prepared['test_poses']))
    json_write(out / 'method_semantics.json', dict(
        endpoint='c_target=c_flat_observed + H_target_pt - H_flat_pt; beta recomputed from target sphere geometry',
        matrix_project='F0(D0 Dq+ c)', matrix_keep='F0((I+(D0-Dq)Dq+)c)',
        matrix_direct='Dq+ c', matrix_ridge='fixed ridge_lambda from experiment config',
        observation='sRGB8 saved observations decoded by inverse_sRGB; H_pt is continuous material-point model',
        note='Target training measurements are used only for post-fit error/coverage diagnostics.'))
    checkpoint.complete('inputs', [out / 'config_snapshot.json', out / 'split.json',
                                   out / 'method_semantics.json'])
    return checkpoint


def _historical_sample(prepared: dict, path: Path, checkpoint_key: str,
                       *, verify_hash: bool = True) -> tuple[dict[str, np.ndarray], dict]:
    expected = _verify_checkpoint_file(prepared['history_manifest'], path, hash_now=verify_hash)
    arrays, metadata = _load_npz(path)
    required = ('grid_index', 'color_residual_samples', 'slope_samples',
                'material_coordinates', 'area_mm2')
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f'{path}: missing required sample arrays {missing}')
    count = len(arrays['grid_index'])
    if arrays['grid_index'].shape != (count,) or arrays['color_residual_samples'].shape != (count, 3) \
            or arrays['slope_samples'].shape != (count, 2) \
            or arrays['material_coordinates'].shape != (count, 2) \
            or arrays['area_mm2'].shape != (count,):
        raise ValueError(f'{path}: sample arrays have incompatible shapes')
    if not np.issubdtype(arrays['grid_index'].dtype, np.integer) \
            or (count > 1 and np.any(np.diff(arrays['grid_index']) <= 0)):
        raise ValueError(f'{path}: grid_index must be sorted, unique integer values')
    if count and (not np.isfinite(arrays['color_residual_samples']).all()
                  or not np.isfinite(arrays['slope_samples']).all()
                  or not np.isfinite(arrays['material_coordinates']).all()
                  or not np.isfinite(arrays['area_mm2']).all()
                  or np.any(arrays['area_mm2'] <= 0)):
        raise ValueError(f'{path}: sample data contains non-finite or nonpositive values')
    if metadata.get('slope_frame_id') not in (None, SLOPE_FRAME_ID):
        raise ValueError(f'{path}: incompatible slope frame {metadata.get("slope_frame_id")!r}')
    if metadata.get('sampling_version') not in (None, SAMPLING_VERSION):
        raise ValueError(f'{path}: incompatible sample semantics {metadata.get("sampling_version")!r}')
    arrays['_source_hash'] = np.asarray(expected)
    arrays['_checkpoint_key'] = np.asarray(checkpoint_key)
    return arrays, metadata


def _build_photometry_models(prepared: dict) -> tuple[dict[str, object], dict]:
    from .photometry import PhotometryModel

    models = {}
    execution = {}
    device = prepared['base'].get('render', {}).get('device', 'cuda:0')
    if str(device).startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f'The saved source configuration requires {device}, but CUDA is unavailable')
    for name, radius in prepared['conditions'].items():
        config = pose_config(prepared['base'], radius)
        model = PhotometryModel(config, device=device)
        models[name] = model
        info = getattr(model, 'execution', None)
        if callable(info):
            execution[name] = info()
        elif isinstance(info, dict):
            execution[name] = info
        else:
            execution[name] = dict(device='cuda')
    return models, execution


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _compute_matrix_cache(prepared: dict, checkpoint: Checkpoints, models: dict[str, object],
                          settings: dict) -> dict[str, dict[str, np.ndarray]]:
    from .matrix import compute_local_response_matrix, build_compensation_matrices

    grid = reference_grid(prepared['base'])
    coordinates = grid.xy_mm.reshape(-1, 2)
    weights = reference_area_weights(grid).reshape(-1)
    x0, n0, e0 = reference_geometry(coordinates, None)
    flat_config = pose_config(prepared['base'], None)
    flat_response = compute_local_response_matrix(
        models['flat'], x0, n0, e0, state_config=flat_config,
        slope_step=settings['matrix']['slope_step'], rcond=settings['matrix']['rcond'])
    D0 = _as_numpy(flat_response.D)
    cache = {}
    for name, radius in prepared['conditions'].items():
        key = f'matrix/{name}'
        path = checkpoint.root / 'matrix' / f'{name}.npz'
        if checkpoint.has(key):
            cache[name] = load_arrays(path)
            continue
        xq, nq, eq = reference_geometry(coordinates, radius)
        target_config = pose_config(prepared['base'], radius)
        response = compute_local_response_matrix(
            models[name], xq, nq, eq, state_config=target_config,
            slope_step=settings['matrix']['slope_step'], rcond=settings['matrix']['rcond'])
        matrices = build_compensation_matrices(
            D0, _as_numpy(response.D), settings['matrix']['ridge_lambda'],
            rcond=settings['matrix']['rcond'], device=str(models[name].device))
        data = {field: _as_numpy(getattr(flat_response, field)) for field in
                ('B', 'D', 'valid', 'singular_values', 'rank', 'condition_number',
                 'active_light_mask', 'finite_difference_D', 'finite_difference_error')}
        target = {field: _as_numpy(getattr(response, field)) for field in
                  ('B', 'D', 'valid', 'singular_values', 'rank', 'condition_number',
                   'active_light_mask', 'finite_difference_D', 'finite_difference_error')}
        result = {f'D0_{field}': data[field] for field in data}
        result.update({f'Dq_{field}': target[field] for field in target})
        result.update({field: _as_numpy(value) for field, value in matrices.items()})
        matrix_valid = np.asarray(result['valid'], dtype=bool) if 'valid' in result else np.asarray(matrices['valid'], dtype=bool)
        matrix_valid &= np.asarray(data['valid'], dtype=bool) & np.asarray(target['valid'], dtype=bool)
        matrix_valid &= np.asarray(target['rank']) == 2
        matrix_valid &= np.asarray(target['singular_values'])[:, -1] >= settings['matrix']['minimum_singular_value']
        matrix_valid &= np.isfinite(np.asarray(target['condition_number']))
        result['matrix_valid'] = matrix_valid
        result['reference_area_mm2'] = weights
        result['grid_index'] = np.arange(len(coordinates), dtype=np.int64)
        result['reference_coordinates_mm'] = coordinates
        npz_write(path, **result, metadata=json.dumps(dict(condition=name, radius_mm=radius,
            ridge_lambda=settings['matrix']['ridge_lambda'], rcond=settings['matrix']['rcond'],
            slope_step=settings['matrix']['slope_step'],
            minimum_singular_value=settings['matrix']['minimum_singular_value'],
            method='local_response_matrix_equations_15_to_22'), ensure_ascii=False))
        checkpoint.complete(key, [path], valid_count=int(matrix_valid.sum()),
                            invalid_count=int(len(matrix_valid) - matrix_valid.sum()))
        cache[name] = result
    return cache


def _pose_source_selection(source_arrays: dict, pose_index: int) -> np.ndarray:
    return np.flatnonzero(np.asarray(source_arrays['image_index']) == pose_index)


def run_migrations(prepared: dict, checkpoint: Checkpoints, models: dict[str, object],
                   settings: dict) -> dict[str, list[dict]]:
    """Create per-pose endpoint samples; every completed pose is atomically checkpointed."""
    by_condition = {}
    axes_x = np.asarray(prepared['source_arrays']['material_grid_x_mm'], dtype=np.float64)
    axes_y = np.asarray(prepared['source_arrays']['material_grid_y_mm'], dtype=np.float64)
    chunk_size = settings['chunk_size']
    for name, radius in prepared['conditions'].items():
        out_dir = checkpoint.root / 'samples' / name / 'migrated'
        out_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        for pose_index, pose in enumerate(prepared['poses']):
            key = f'migration/{name}/{pose["pose_id"]}'
            path = out_dir / f'{pose["pose_id"]}.npz'
            if checkpoint.has(key):
                sample = load_arrays(path)
            else:
                source_indices = _pose_source_selection(prepared['source_arrays'], pose_index)
                coordinates = prepared['source_arrays']['material_coordinates'][source_indices]
                grid_indices = prepared['source_arrays']['material_grid_index'][source_indices]
                state_config = pose_config(prepared['base'], radius, pose)
                target_indices = target_valid_grid_indices(
                    coordinates, grid_indices, state_config, prepared['references'][name],
                    minimum_reference_dot=.1, target_model=models[name], chunk_size=chunk_size)
                sample = migrate_pose_samples(prepared['source_arrays'], pose_index, pose,
                                              radius, prepared['base'], target_indices,
                                              models['flat'], models[name], chunk_size)
                sample['material_grid_x_mm'] = axes_x
                sample['material_grid_y_mm'] = axes_y
                npz_write(path, **sample, metadata=json.dumps(dict(
                    pose=pose, condition=name, radius_mm=radius,
                    run_signature=prepared['run_signature'],
                    method='endpoint_H_pt_color_delta_and_recomputed_target_geometry',
                    source_sample_count=int(len(source_indices)),
                    target_grid_index_count=int(len(target_indices))), ensure_ascii=False))
                checkpoint.complete(key, [path], sample_count=len(sample['grid_index']),
                                    source_candidate_count=sample['source_candidate_count'],
                                    target_geometry_retained_count=sample['target_geometry_retained_count'],
                                    model_invalid_count=sample['model_invalid_count'])
                print(f'{name} migrated: {pose_index + 1}/{len(prepared["poses"])}; '
                      f'{len(sample["grid_index"])} samples', flush=True)
            samples.append(sample)
        by_condition[name] = samples
    return by_condition


def fit_conditions(prepared: dict, checkpoint: Checkpoints, by_condition: dict) -> dict[str, NormalCalibration | None]:
    """Build and validate every endpoint LUT using the saved fixed source split."""
    endpoint_luts = {}
    for name, radius in prepared['conditions'].items():
        key = f'fit/{name}'
        out_dir = checkpoint.root / 'calibration' / name
        if checkpoint.has(key):
            info = checkpoint.manifest['calibration'][name]
            lut_name = info.get('lut')
            endpoint_luts[name] = NormalCalibration.load(out_dir / lut_name) if lut_name else None
            continue
        metadata = dict(prepared['source_metadata'],
                        run_signature=signature(dict(parent=prepared['run_signature'], condition=name,
                                                      method='endpoint_H_pt')),
                        parent_run_signature=prepared['run_signature'], backend='analytic',
                        slope_frame_id=SLOPE_FRAME_ID,
                        coordinate_chart_id='cylinder_reference_arclength_' + name,
                        coordinate_chart_storage='embedded_reference_arclength_axes',
                        calibration_kind='plane_to_curve_endpoint_H_pt_transfer',
                        sampling_version=MIGRATION_SAMPLING_VERSION,
                        image_encoding=MIGRATION_IMAGE_ENCODING,
                        source_observation_encoding='inverse_srgb(srgb8)',
                        query_observation_encoding='inverse_srgb(srgb8)',
                        training_color_generation='source_inverse_srgb8_plus_continuous_H_pt_target_minus_flat',
                        frame_convention='per_sample_reference_local_tangent_frame_b_v1',
                        fixed_reference_normal=([0., 0., 1.] if radius is None else None),
                        fixed_tangent_frame=([[1., 0.], [0., 1.], [0., 0.]] if radius is None else None),
                        color_semantics='signed_linear_rgb_inverse_srgb8_plus_continuous_H_pt_delta',
                        simulation_config_signature=signature(pose_config(prepared['base'], radius)))
        for field in ('validation', 'validation_status', 'training_images', 'validation_images'):
            metadata.pop(field, None)
        report, lut = fit_migrated_condition(by_condition[name], pose_config(prepared['base'], radius),
                                             out_dir, metadata, prepared['split'],
                                             [pose['pose_id'] for pose in prepared['poses']])
        summary = dict(status=report.get('status', 'failed'),
                       lut=('normal_lut.npz' if report.get('status') == 'passed' and lut is not None else
                            'normal_lut_diagnostic.npz' if lut is not None else None),
                       required_pose_count=len(prepared['poses']),
                       accepted_pose_count=report.get('accepted_pose_count'),
                       missing_required_poses=report.get('missing_required_poses', []))
        checkpoint.manifest['calibration'][name] = summary
        files = [path for path in out_dir.rglob('*') if path.is_file()]
        checkpoint.complete(key, files, status=summary['status'])
        endpoint_luts[name] = lut
        print(f'{name} endpoint LUT status: {summary["status"]}', flush=True)
    return endpoint_luts


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('', encoding='utf-8')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def diagnose_training_transfer(prepared: dict, checkpoint: Checkpoints,
                               by_condition: dict[str, list[dict]]) -> dict:
    """Compare migrated samples with archived target scans after every LUT fit is done."""
    summaries = {}
    source_arrays = prepared['source_arrays']
    grid_width = len(source_arrays['material_grid_x_mm'])
    source_grid_area = reference_area_weights(reference_grid(prepared['base']))
    for condition in prepared['conditions']:
        key = f'training_diagnostics/{condition}'
        out = checkpoint.root / 'training_diagnostics' / condition
        csv_path, json_path = out / 'per_pose.csv', out / 'summary.json'
        if checkpoint.has(key):
            summaries[condition] = _load_json(json_path)['summary']
            continue
        rows, color_squared, slope_squared, coverage = [], [], [], []
        for pose_index, (pose, migrated) in enumerate(zip(prepared['poses'], by_condition[condition], strict=True)):
            source_rows = _pose_source_selection(source_arrays, pose_index)
            source_ij = np.asarray(source_arrays['material_grid_index'][source_rows], dtype=np.int64)
            source_grid = np.unique(source_ij[:, 0] * grid_width + source_ij[:, 1])
            if prepared['conditions'][condition] is None:
                order = np.argsort(source_grid, kind='stable')
                source_grid_ordered = source_grid[order]
                row_index, col_index = np.divmod(source_grid_ordered, grid_width)
                source_lookup = {int(row * grid_width + col): index
                                 for index, (row, col) in enumerate(source_ij)}
                ordered_source = np.asarray([source_lookup[int(value)] for value in source_grid_ordered], dtype=np.int64)
                target = dict(grid_index=source_grid_ordered,
                              color_residual_samples=source_arrays['color_residual_samples'][source_rows][ordered_source],
                              slope_samples=source_arrays['slope_samples'][source_rows][ordered_source],
                              material_coordinates=source_arrays['material_coordinates'][source_rows][ordered_source],
                              area_mm2=source_grid_area[row_index, col_index])
            else:
                target_path = prepared['history_root'] / 'samples' / condition / 'training' / (pose['pose_id'] + '.npz')
                target, _ = _historical_sample(prepared, target_path,
                                               f'{condition}/training/{pose["pose_id"]}', verify_hash=True)
            target_grid = np.asarray(target['grid_index'], dtype=np.int64)
            migrated_grid = np.asarray(migrated['grid_index'], dtype=np.int64)
            common, im, it = np.intersect1d(migrated_grid, target_grid, assume_unique=True,
                                            return_indices=True)
            target_only_source = np.setdiff1d(target_grid, source_grid, assume_unique=True)
            target_only_migrated = np.setdiff1d(target_grid, migrated_grid, assume_unique=True)
            full_area = float(np.sum(target['area_mm2']))
            common_area = target['area_mm2'][it]
            if len(common) and float(common_area.sum()) > 0:
                weights = common_area / common_area.sum()
                color_error = migrated['color_residual_samples'][im].astype(np.float64) \
                    - target['color_residual_samples'][it].astype(np.float64)
                slope_error = migrated['slope_samples'][im].astype(np.float64) \
                    - target['slope_samples'][it].astype(np.float64)
                color_norm = np.linalg.norm(color_error, axis=-1)
                slope_norm = np.linalg.norm(slope_error, axis=-1)
                color_rmse = float(np.sqrt(weights @ (color_norm ** 2)))
                slope_rmse = float(np.sqrt(weights @ (slope_norm ** 2)))
                color_bias = (weights[:, None] * color_error).sum(axis=0).tolist()
                slope_bias = (weights[:, None] * slope_error).sum(axis=0).tolist()
                color_p95 = weighted_quantile(color_norm, weights, .95)
                slope_p95 = weighted_quantile(slope_norm, weights, .95)
            else:
                color_rmse = slope_rmse = color_p95 = slope_p95 = None
                color_bias = slope_bias = None
            gap_area = 0.
            if len(target_only_source) and full_area > 0:
                target_only_offsets = np.searchsorted(target_grid, target_only_source)
                gap_area = float(target['area_mm2'][target_only_offsets].sum() / full_area)
            row = dict(condition=condition, pose_id=pose['pose_id'], anchor_id=pose['anchor_id'],
                       source_count=len(source_grid), migrated_count=len(migrated_grid),
                       target_observed_count=len(target_grid), common_count=len(common),
                       source_retention_fraction=(len(migrated_grid) / len(source_grid) if len(source_grid) else None),
                       target_observed_coverage=(len(common) / len(target_grid) if len(target_grid) else None),
                       target_domain_gap_count=len(target_only_source),
                       target_not_migrated_count=len(target_only_migrated),
                       target_domain_gap_area_fraction=gap_area,
                       common_area_fraction=(float(common_area.sum() / full_area) if full_area else None),
                       color_rmse=color_rmse, color_p95=color_p95, color_bias=color_bias,
                       slope_rmse=slope_rmse, slope_p95=slope_p95, slope_bias=slope_bias,
                       status='ok' if color_rmse is not None else 'empty_common_domain')
            rows.append(row)
            if color_rmse is not None:
                color_squared.append(color_rmse ** 2)
                slope_squared.append(slope_rmse ** 2)
            coverage.append(row['target_observed_coverage'])
        result = dict(status='post_fit_diagnostic_only', condition=condition, pose_count=len(rows),
                      pose_equal_color_rmse=float(np.sqrt(np.mean(color_squared))) if color_squared else None,
                      pose_equal_slope_rmse=float(np.sqrt(np.mean(slope_squared))) if slope_squared else None,
                      mean_target_observed_coverage=(float(np.mean([value for value in coverage if value is not None]))
                                                     if any(value is not None for value in coverage) else None),
                      total_source_samples=int(sum(row['source_count'] for row in rows)),
                      total_migrated_samples=int(sum(row['migrated_count'] for row in rows)),
                      total_target_observed_samples=int(sum(row['target_observed_count'] for row in rows)),
                      total_target_domain_gap_samples=int(sum(row['target_domain_gap_count'] for row in rows)),
                      note='Target training colors and labels are read only after endpoint LUT fitting; they never generate migrated samples or choose parameters.')
        _write_csv(csv_path, rows)
        json_write(json_path, dict(summary=result, per_pose_rows=rows))
        checkpoint.manifest.setdefault('training_diagnostics', {})[condition] = result
        checkpoint.complete(key, [csv_path, json_path], pose_count=len(rows), status=result['status'])
        summaries[condition] = result
    return summaries


def _predict_method(method: str, sample: dict, condition: str, prepared: dict,
                    endpoint_luts: dict, matrix_cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return predicted slopes, prediction-valid mask, LUT query colors, and LUT use flag."""
    colors = np.asarray(sample['color_residual_samples'], dtype=np.float64)
    grid_index = np.asarray(sample['grid_index'], dtype=np.int64)
    n = len(grid_index)
    flat_lut = prepared['source_lut']
    lut = None
    if method == 'frozen_flat_lut':
        lut = flat_lut
    elif method == 'endpoint_migrated_lut':
        lut = endpoint_luts.get(condition)
    elif method == 'historical_same_curvature_lut':
        lut = prepared['history_luts'].get(condition)
    if method in ('frozen_flat_lut', 'endpoint_migrated_lut', 'historical_same_curvature_lut'):
        if lut is None:
            return np.full((n, 2), np.nan), np.zeros(n, dtype=bool), colors, True
        prediction, _, _ = lookup_slopes_raw(colors, lut)
        return prediction, np.isfinite(prediction).all(axis=-1), colors, True

    matrices = matrix_cache[condition]
    valid = np.asarray(matrices['matrix_valid'][grid_index], dtype=bool)
    if method in ('matrix_project_lut', 'matrix_keep_lut',
                  'matrix_ridge_project_lut', 'matrix_ridge_keep_lut'):
        field = {'matrix_project_lut': 'C_project', 'matrix_keep_lut': 'C_keep',
                 'matrix_ridge_project_lut': 'C_project_ridge',
                 'matrix_ridge_keep_lut': 'C_keep_ridge'}[method]
        corrected = np.full((n, 3), np.nan, dtype=np.float64)
        corrected[valid] = np.einsum('nij,nj->ni', matrices[field][grid_index[valid]], colors[valid])
        prediction = np.full((n, 2), np.nan, dtype=np.float64)
        finite = valid & np.isfinite(corrected).all(axis=-1)
        if np.any(finite):
            prediction[finite] = lookup_slopes_raw(corrected[finite], flat_lut)[0]
        return prediction, finite & np.isfinite(prediction).all(axis=-1), corrected, True
    if method in ('matrix_direct', 'matrix_ridge_direct'):
        field = 'R_direct' if method == 'matrix_direct' else 'R_direct_ridge'
        prediction = np.full((n, 2), np.nan, dtype=np.float64)
        prediction[valid] = np.einsum('nij,nj->ni', matrices[field][grid_index[valid]], colors[valid])
        return prediction, valid & np.isfinite(prediction).all(axis=-1), colors, False
    raise ValueError(f'Unknown compensation method: {method}')


def _evaluate_prediction(sample: dict, prediction: np.ndarray, valid: np.ndarray,
                         query_colors: np.ndarray, support_lut: NormalCalibration,
                         indices: np.ndarray | None, *, method: str,
                         metadata: dict,
                         raw_flat_support_lut: NormalCalibration) -> tuple[dict, dict]:
    grid = np.asarray(sample['grid_index'], dtype=np.int64)
    if indices is None:
        take = np.arange(len(grid))
    else:
        take = np.searchsorted(grid, indices)
        if np.any(take >= len(grid)) or not np.array_equal(grid[take], indices):
            raise ValueError('Evaluation domain indices are not present in the historical sample')
    invalid_count = int(np.count_nonzero(~valid[take]))
    row = dict(status='incomplete' if invalid_count else 'ok',
               sample_count=int(len(take)), valid_prediction_count=int(len(take) - invalid_count),
               invalid_prediction_count=invalid_count, **metadata)
    if invalid_count:
        row.update(rmse=None, mse=None, angular_rmse_degrees=None,
                   reason='matrix rank/conditioning failure or non-finite prediction')
        return row, {}
    sample_view = dict(sample, color_residual_samples=np.asarray(query_colors, dtype=np.float32))
    metric, detail = evaluate_pose(sample_view, support_lut, indices, prediction=prediction)
    metric.update(metadata)
    metric['valid_prediction_count'] = len(take)
    if method in ('matrix_direct', 'matrix_ridge_direct'):
        metric['lut_support_applicable'] = False
        metric['support_note'] = 'flat LUT support describes raw RGB context only; direct slope method does not query a LUT'
    else:
        metric['lut_support_applicable'] = True
    # Keep the original target RGB support visible even when a matrix changes the query color.
    raw_inside, raw_supported = support(np.asarray(sample['color_residual_samples'], dtype=np.float32)[take],
                                        raw_flat_support_lut)
    area = np.asarray(sample['area_mm2'], dtype=np.float64)[take]
    normalized_area = area / area.sum() if area.sum() else np.zeros_like(area)
    metric['raw_flat_lut_out_of_range_fraction'] = float(normalized_area @ ~raw_inside) if len(take) else None
    metric['raw_flat_lut_original_support_fraction'] = float(normalized_area @ raw_supported) if len(take) else None
    return metric, detail


def _finite_stat(values: np.ndarray, statistic: str, quantile: float | None = None) -> float | None:
    """Return a scalar over finite values only, or None when none are available."""
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    if statistic == 'min':
        return float(np.min(finite))
    if statistic == 'max':
        return float(np.max(finite))
    if statistic == 'percentile' and quantile is not None:
        return float(np.percentile(finite, quantile))
    raise ValueError(f'Unknown finite statistic: {statistic}')


def evaluate_all(prepared: dict, checkpoint: Checkpoints, endpoint_luts: dict,
                 matrix_cache: dict) -> dict:
    """Compare all methods on full, shared, and matrix-supported test domains."""
    condition_names = list(prepared['conditions'])
    datasets = {}
    for name in condition_names:
        rows = []
        for pose in prepared['test_poses']:
            path = prepared['test_sample_paths'][f'{name}/{pose["pose_id"]}']
            sample, meta = _historical_sample(prepared, path,
                                              f'{name}/test/{pose["pose_id"]}', verify_hash=True)
            saved_pose = meta.get('pose')
            if saved_pose is not None and (saved_pose.get('pose_id') != pose['pose_id']
                    or not np.allclose(saved_pose.get('center_uv'), pose['center_uv'], rtol=0, atol=1e-12)
                    or not np.isclose(saved_pose.get('depth_mm'), pose['depth_mm'], rtol=0, atol=1e-12)):
                raise ValueError(f'Historical test sample pose metadata differs from split: {path}')
            sample['pose'] = pose
            rows.append(sample)
        datasets[name] = rows
    common_by_pose = []
    for pose_index in range(len(prepared['test_poses'])):
        common = set(datasets[condition_names[0]][pose_index]['grid_index'].tolist())
        for name in condition_names[1:]:
            common.intersection_update(datasets[name][pose_index]['grid_index'].tolist())
        common_by_pose.append(np.asarray(sorted(common), dtype=np.int64))

    methods = ('frozen_flat_lut', 'endpoint_migrated_lut', 'historical_same_curvature_lut',
               'matrix_project_lut', 'matrix_keep_lut', 'matrix_direct',
               'matrix_ridge_project_lut', 'matrix_ridge_keep_lut', 'matrix_ridge_direct')
    domains = ('full', 'common', 'matrix_matched')
    metric_summaries, row_sets, per_pose_rows, coverage_rows = {}, {}, [], []
    tolerances = prepared['settings']['metrics']['slope_tolerances']
    for name in condition_names:
        matrix = matrix_cache[name]
        for pose_index, (pose, sample) in enumerate(zip(prepared['test_poses'], datasets[name], strict=True)):
            idx = sample['grid_index']
            valid = matrix['matrix_valid'][idx]
            common = common_by_pose[pose_index]
            common_take = np.searchsorted(idx, common)
            matched = common[valid[common_take]] if len(common) else np.empty(0, np.int64)
            area = np.asarray(sample['area_mm2'], dtype=np.float64)
            selected_matrix_area = area[valid]
            sigma1 = np.asarray(matrix['Dq_singular_values'][idx, 0], dtype=np.float64)
            sigma2 = np.asarray(matrix['Dq_singular_values'][idx, 1], dtype=np.float64)
            condition_numbers = np.asarray(matrix['Dq_condition_number'][idx], dtype=np.float64)
            finite_difference_error = np.asarray(matrix['Dq_finite_difference_error'][idx], dtype=np.float64)
            coverage_rows.append(dict(condition=name, pose_id=pose['pose_id'],
                valid_test_samples=len(idx), common_samples=len(common), matrix_rank_valid_samples=int(valid.sum()),
                common_area_fraction=float(area[common_take].sum() / area.sum()) if len(common) and area.sum() else 0.,
                matrix_valid_area_fraction=float(selected_matrix_area.sum() / area.sum()) if area.sum() else None,
                matrix_matched_area_fraction=float(area[np.searchsorted(idx, matched)].sum() / area.sum())
                    if len(matched) and area.sum() else 0.,
                matrix_invalid_count=int(np.count_nonzero(~valid)),
                rank_failure_count=int(np.count_nonzero(~valid)),
                Dq_singular_values_nonfinite_count=int(np.count_nonzero(~(np.isfinite(sigma1) & np.isfinite(sigma2)))),
                sigma1_min=_finite_stat(sigma1, 'min'),
                sigma2_min=_finite_stat(sigma2, 'min'),
                sigma2_p05=_finite_stat(sigma2, 'percentile', 5),
                singular_value_statistics_population='finite_values_only',
                condition_number_p95=_finite_stat(condition_numbers, 'percentile', 95),
                condition_number_statistic_population='finite_values_only',
                condition_number_finite_count=int(np.count_nonzero(np.isfinite(condition_numbers))),
                condition_number_nonfinite_count=int(np.count_nonzero(~np.isfinite(condition_numbers))),
                finite_difference_D_error_p95=_finite_stat(finite_difference_error, 'percentile', 95),
                finite_difference_D_error_max=_finite_stat(finite_difference_error, 'max'),
                finite_difference_D_error_statistic_population='finite_values_only',
                finite_difference_D_error_finite_count=int(np.count_nonzero(np.isfinite(finite_difference_error))),
                finite_difference_D_error_nonfinite_count=int(np.count_nonzero(~np.isfinite(finite_difference_error))),
                active_light_set_changed_fraction=float(np.mean(np.any(
                    matrix['D0_active_light_mask'][idx] != matrix['Dq_active_light_mask'][idx], axis=-1)))
                    if len(idx) else None,
                status='ok' if np.all(valid) else 'incomplete'))

    for method in methods:
        for name in condition_names:
            grouped_rows = {domain: [] for domain in domains}
            grouped_details = {domain: [] for domain in domains}
            for pose_index, (pose, sample) in enumerate(zip(prepared['test_poses'], datasets[name], strict=True)):
                pred, valid, query_colors, uses_lut = _predict_method(method, sample, name,
                                                                       prepared, endpoint_luts, matrix_cache)
                if uses_lut and method in ('endpoint_migrated_lut', 'historical_same_curvature_lut'):
                    method_lut = (endpoint_luts.get(name) if method == 'endpoint_migrated_lut'
                                  else prepared['history_luts'].get(name))
                else:
                    method_lut = prepared['source_lut']
                lut_status = ('unavailable' if method_lut is None else
                              prepared['calibration_status'].get(name, {}).get('status', 'published')
                              if method == 'historical_same_curvature_lut' else
                              checkpoint.manifest.get('calibration', {}).get(name, {}).get('status', 'published')
                              if method == 'endpoint_migrated_lut' else 'published')
                pose_meta = dict(pose_id=pose['pose_id'], anchor_id=pose['anchor_id'],
                                 u=pose['center_uv'][0], v=pose['center_uv'][1], depth_mm=pose['depth_mm'],
                                 method=method, condition=name, lut_status=lut_status)
                method_support_lut = method_lut if method_lut is not None else prepared['source_lut']
                for domain in domains:
                    if domain == 'full':
                        indices = None
                    elif domain == 'common':
                        indices = common_by_pose[pose_index]
                    else:
                        sample_grid = sample['grid_index']
                        current_valid = matrix_cache[name]['matrix_valid'][sample_grid]
                        common = common_by_pose[pose_index]
                        common_take = np.searchsorted(sample_grid, common)
                        indices = common[current_valid[common_take]] if len(common) else np.empty(0, np.int64)
                    metric, detail = _evaluate_prediction(sample, pred, valid, query_colors,
                                                           method_support_lut, indices,
                                                           method=method, metadata=dict(pose_meta, domain=domain),
                                                           raw_flat_support_lut=prepared['source_lut'])
                    grouped_rows[domain].append(metric)
                    if detail:
                        grouped_details[domain].append(detail)
                    per_pose_rows.append(metric)
            for domain in domains:
                key = f'{method}/{name}/{domain}'
                metric_summaries[key] = aggregate(grouped_rows[domain], grouped_details[domain], tolerances)
                metric_summaries[key]['method'] = method
                metric_summaries[key]['condition'] = name
                metric_summaries[key]['domain'] = domain
                if method in ('matrix_direct', 'matrix_ridge_direct'):
                    metric_summaries[key]['out_of_range_fraction'] = None
                    metric_summaries[key]['original_support_fraction'] = None
                    metric_summaries[key]['lut_support_applicable'] = False
                    for row in grouped_rows[domain]:
                        row['out_of_range_fraction'] = None
                        row['original_support_fraction'] = None
                if method == 'historical_same_curvature_lut':
                    metric_summaries[key]['lut_status'] = prepared['calibration_status'].get(name, {}).get('status', 'published')
                elif method == 'endpoint_migrated_lut':
                    metric_summaries[key]['lut_status'] = checkpoint.manifest.get('calibration', {}).get(name, {}).get('status')
                row_sets[key] = grouped_rows[domain]

    comparisons = {}
    repeats = prepared['settings']['metrics']['bootstrap_repeats']
    seed = prepared['settings']['metrics']['seed']
    for name in condition_names:
        comparisons[name] = {}
        for domain in ('common', 'matrix_matched'):
            baseline = row_sets[f'frozen_flat_lut/{name}/{domain}']
            for method in methods:
                if method == 'frozen_flat_lut':
                    continue
                rows = row_sets[f'{method}/{name}/{domain}']
                comparisons[name][f'{method}/{domain}'] = paired_bootstrap(
                    rows, baseline, repeats=repeats, seed=seed)

    historical_reproduction = {}
    for name in condition_names:
        key = f'flat/{name}/common'
        archived = prepared['history_metrics'].get('metrics', {}).get(key, {}).get('rmse')
        reproduced = metric_summaries[f'frozen_flat_lut/{name}/common'].get('rmse')
        matches = (archived is not None and reproduced is not None
                   and np.isclose(archived, reproduced, rtol=0, atol=1e-12))
        historical_reproduction[name] = dict(archived_rmse=archived, reproduced_rmse=reproduced,
                                             absolute_difference=(abs(archived - reproduced)
                                                                  if archived is not None and reproduced is not None else None),
                                             status='matched' if matches else 'mismatch')
    result = dict(version=VERSION, status='completed', conditions=prepared['conditions'],
                  methods=list(methods), metrics=metric_summaries,
                  comparisons=comparisons, coverage=coverage_rows,
                  historical_flat_baseline_reproduction=historical_reproduction,
                  calibration=checkpoint.manifest['calibration'],
                  training_transfer_diagnostics=checkpoint.manifest.get('training_diagnostics', {}),
                  matrix_settings=prepared['settings']['matrix'],
                  metric_definition='sqrt(mean_pose(sum_reference_area_weighted squared vector slope error)); local tangent beta',
                  historical_observation='actual inverse_srgb(saved sRGB8), paired by material grid index',
                  endpoint_method='source inverse_srgb(sRGB8) + H_pt(target contact-reference) - H_pt(flat contact-reference); target beta recomputed')
    _write_csv(checkpoint.root / 'per_pose.csv', per_pose_rows)
    _write_csv(checkpoint.root / 'cross_method.csv',
               [dict(metric=key, **value) for key, value in metric_summaries.items()])
    _write_csv(checkpoint.root / 'coverage.csv', coverage_rows)
    json_write(checkpoint.root / 'metrics.json', result)
    failed_baselines = {name: check for name, check in historical_reproduction.items()
                        if check['status'] != 'matched'}
    if failed_baselines:
        result['status'] = 'historical_baseline_mismatch'
        result['historical_baseline_failures'] = failed_baselines
        json_write(checkpoint.root / 'metrics.json', result)
        raise ValueError(f'Frozen flat LUT baseline does not reproduce historical metrics: {failed_baselines}')
    return result


def _plot_evaluation(out: Path, methods: tuple[str, ...], conditions: list[str],
                     metrics: dict, coverage: list[dict], status_labels: dict | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Plot output requires matplotlib') from exc
    figure_dir = out / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = [name.replace('radius_', 'R=').replace('p', '.') + (' mm' if name != 'flat' else '')
              for name in conditions]
    values = np.full((len(methods), len(conditions)), np.nan, dtype=np.float64)
    for i, method in enumerate(methods):
        for j, condition in enumerate(conditions):
            summary = metrics.get(f'{method}/{condition}/common', {})
            value = summary.get('rmse')
            if value is not None:
                values[i, j] = value

    lut_methods = ('frozen_flat_lut', 'endpoint_migrated_lut', 'historical_same_curvature_lut')
    matrix_methods = ('matrix_project_lut', 'matrix_keep_lut', 'matrix_direct',
                      'matrix_ridge_project_lut', 'matrix_ridge_keep_lut', 'matrix_ridge_direct')
    method_index = {method: index for index, method in enumerate(methods)}
    method_labels = {
        'frozen_flat_lut': 'Frozen flat LUT',
        'endpoint_migrated_lut': 'Endpoint H_pt LUT',
        'historical_same_curvature_lut': 'Historical same-curvature LUT',
        'matrix_project_lut': 'Project → flat LUT',
        'matrix_keep_lut': 'Keep → flat LUT',
        'matrix_direct': 'Direct slopes',
        'matrix_ridge_project_lut': 'Ridge project → flat LUT',
        'matrix_ridge_keep_lut': 'Ridge keep → flat LUT',
        'matrix_ridge_direct': 'Ridge direct slopes',
    }

    def draw_comparison(selected_methods: tuple[str, ...], title: str, path: Path,
                        with_status: bool = False) -> None:
        present = [method for method in selected_methods if method in method_index]
        if with_status:
            fig = plt.figure(figsize=(11, 7.4), layout='constrained')
            grid_spec = fig.add_gridspec(2, 1, height_ratios=(4.2, 1.35), hspace=.12)
            ax = fig.add_subplot(grid_spec[0])
            status_ax = fig.add_subplot(grid_spec[1])
        else:
            fig, ax = plt.subplots(figsize=(11, 6.2))
        for method in present:
            ax.plot(np.arange(len(conditions)), values[method_index[method]], marker='o', linewidth=1.6,
                    label=method_labels.get(method, method))
        ax.set_xticks(np.arange(len(conditions)), labels)
        ax.set_ylabel('Slope-vector RMSE (unitless; lower is better)')
        ax.set_xlabel('Reference-surface condition')
        ax.set_title(title)
        ax.grid(True, alpha=.25)
        ax.legend(fontsize=8.5, ncol=2 if len(present) > 3 else 1)
        if with_status:
            status_map = status_labels or {}
            status_values = [[method_labels.get(method, method)]
                             + [status_map.get(method, {}).get(condition, 'n/a')
                                for condition in conditions] for method in present]
            status_ax.axis('off')
            table = status_ax.table(
                cellText=status_values,
                colLabels=['Method', *labels],
                colWidths=[.34] + [.66 / len(conditions)] * len(conditions),
                cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.35)
            status_ax.set_title('LUT validation status by condition', fontsize=9, pad=2)
        else:
            fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    draw_comparison(lut_methods, 'LUT comparisons on the common test domain',
                    figure_dir / 'rmse_lut_comparison.png', with_status=True)
    draw_comparison(('frozen_flat_lut', *matrix_methods), 'Local matrix compensation on the common test domain',
                    figure_dir / 'rmse_matrix_comparison.png')

    valid_coverage = []
    for condition in conditions:
        rows = [row['matrix_valid_area_fraction'] for row in coverage
                if row['condition'] == condition and row['matrix_valid_area_fraction'] is not None]
        valid_coverage.append(float(np.mean(rows)) if rows else np.nan)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(conditions)), valid_coverage, color='#4c78a8')
    ax.set_xticks(np.arange(len(conditions)), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Area fraction with full-rank local response')
    ax.set_xlabel('Reference-surface condition')
    ax.set_title('Local response matrix support; invalid positions remain reported')
    ax.grid(True, axis='y', alpha=.25)
    fig.tight_layout()
    fig.savefig(figure_dir / 'matrix_coverage.png', dpi=160)
    plt.close(fig)
    npz_write(figure_dir / 'figure_data.npz', method_names=np.asarray(methods),
              condition_names=np.asarray(conditions), slope_rmse=values,
              matrix_valid_area_fraction=np.asarray(valid_coverage, dtype=np.float64),
              lut_validation_status=np.asarray([
                  [(status_labels or {}).get(method, {}).get(condition, 'n/a') for condition in conditions]
                  for method in lut_methods], dtype=str),
              lut_validation_methods=np.asarray(lut_methods),
              matrix_comparison_methods=np.asarray(('frozen_flat_lut', *matrix_methods)))


def _dry_run_report(prepared: dict, output: Path | None) -> dict:
    return dict(status='dry-run', source=str(prepared['root']),
                source_run_signature=prepared['source_manifest'].get('run_signature'),
                transfer_results=str(prepared['history_root']),
                transfer_run_signature=prepared['history_manifest'].get('run_signature'),
                conditions=prepared['conditions'], workload=prepared['workload'],
                fixed_training_poses=dict(training=len(prepared['split']['training_images']),
                                          validation=len(prepared['split']['validation_images'])),
                independent_test_poses=len(prepared['test_poses']),
                output=str(output or prepared['root'] / 'evaluation' / 'curve_lut_compensation' / '<unique_run>'),
                render_count=0, cuda_initialized=False,
                methods=['frozen_flat_lut', 'endpoint_migrated_lut',
                         'historical_same_curvature_lut', 'matrix_project_lut',
                         'matrix_keep_lut', 'matrix_direct', 'matrix_ridge_project_lut',
                         'matrix_ridge_keep_lut', 'matrix_ridge_direct'])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lut-results', required=True, type=Path,
                        help='Published planar LUT result directory')
    parser.add_argument('--transfer-results', required=True, type=Path,
                        help='Completed historical curve_lut_transfer run with saved observations')
    parser.add_argument('--exp-config', type=Path, default=DEFAULT_EXP_CONFIG,
                        help='Experiment configuration; physical parameters come from the source snapshot')
    parser.add_argument('--out', type=Path,
                        help='New run directory under source/evaluation/curve_lut_compensation')
    parser.add_argument('--resume', action='store_true',
                        help='Verify and resume the compatible run selected by --out')
    parser.add_argument('--dry-run', action='store_true',
                        help='Read-only input/config/workload check; no CUDA, rendering, or output writes')
    args = parser.parse_args(argv)
    if args.resume and args.out is None:
        parser.error('--resume requires --out')
    config = load_experiment_config(args.exp_config)
    settings = resolve_curve_lut_compensation_settings(config)
    prepared = prepare(args.lut_results, args.transfer_results, settings)
    if args.out is not None:
        parent = (prepared['root'] / 'evaluation' / 'curve_lut_compensation').resolve()
        resolved_out = args.out.expanduser().resolve()
        if not resolved_out.is_relative_to(parent) or resolved_out == parent:
            raise ValueError(f'--out must be a unique run subdirectory of {parent}')
        if not args.resume and resolved_out.exists() and any(resolved_out.iterdir()):
            raise ValueError(f'Nonempty output requires a compatible --resume: {resolved_out}')
    else:
        resolved_out = None
    if args.dry_run:
        if args.resume:
            start_run(prepared, settings, resolved_out, True)
        print(json.dumps(_dry_run_report(prepared, resolved_out), ensure_ascii=False, indent=2))
        return

    checkpoint = start_run(prepared, settings, resolved_out, args.resume)
    print(f'Experiment output: {checkpoint.root}', flush=True)
    if checkpoint.manifest.get('status') == 'completed':
        print('Completed run verified; no artifacts rewritten.')
        return
    start_time = time.perf_counter()
    models, execution = _build_photometry_models(prepared)
    checkpoint.manifest.setdefault('execution', {}).update(
        device=prepared['base'].get('render', {}).get('device', 'cuda:0'),
        photometry=execution)
    checkpoint.save()
    migrated = run_migrations(prepared, checkpoint, models, settings)
    endpoint_luts = fit_conditions(prepared, checkpoint, migrated)
    training_diagnostics = diagnose_training_transfer(prepared, checkpoint, migrated)
    checkpoint.manifest.setdefault('training_diagnostics', {}).update(training_diagnostics)
    checkpoint.save()
    matrix_cache = _compute_matrix_cache(prepared, checkpoint, models, settings)
    metrics = evaluate_all(prepared, checkpoint, endpoint_luts, matrix_cache)
    status_labels = {
        'endpoint_migrated_lut': {
            name: checkpoint.manifest['calibration'][name]['status']
            + (' diagnostic' if checkpoint.manifest['calibration'][name]['status'] != 'passed' else '')
            for name in prepared['conditions']},
        'historical_same_curvature_lut': {
            name: prepared['calibration_status'].get(name, {}).get('status', 'published')
            + (' diagnostic' if prepared['calibration_status'].get(name, {}).get('status') != 'passed' else '')
            for name in prepared['conditions']},
        'frozen_flat_lut': {name: 'published' for name in prepared['conditions']},
    }
    _plot_evaluation(checkpoint.root, tuple(metrics['methods']), list(prepared['conditions']),
                     metrics['metrics'], metrics['coverage'], status_labels=status_labels)
    elapsed = time.perf_counter() - start_time
    json_write(checkpoint.root / 'runtime.json',
               dict(elapsed_seconds=elapsed, execution=checkpoint.manifest['execution']))
    checkpoint.manifest['evaluation'] = dict(
        status=metrics['status'], metric_count=len(metrics['metrics']),
        baseline_reproduction=metrics['historical_flat_baseline_reproduction'])
    files = [checkpoint.root / name for name in
             ('metrics.json', 'per_pose.csv', 'cross_method.csv', 'coverage.csv', 'runtime.json')]
    files += [path for folder in ('figures', 'training_diagnostics', 'calibration', 'matrix', 'samples')
              for path in (checkpoint.root / folder).rglob('*') if path.is_file()]
    checkpoint.complete('evaluation', files, status=metrics['status'], elapsed_seconds=elapsed)
    checkpoint.manifest['status'] = 'completed'
    checkpoint.manifest['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    checkpoint.save()
    print(f'Completed: {checkpoint.root / "metrics.json"}', flush=True)
