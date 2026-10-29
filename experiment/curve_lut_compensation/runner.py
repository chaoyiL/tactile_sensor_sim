"""Migrate, fit, and compare curved normal LUTs using endpoint compensation."""
from __future__ import annotations

import argparse
import copy
import csv
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
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
                       pose_config, target_valid_grid_indices)
from .timing import RunTimings


VERSION = 'curve_lut_compensation_v2'
METHODS = ('frozen_flat_lut', 'endpoint_migrated_lut', 'historical_same_curvature_lut')
DOMAINS = ('full', 'common')
DEFAULT_EXP_CONFIG = CONFIG_DIR / 'curve_lut_compensation.json'
RESUME_INPUT_FIELDS = ('source', 'source_files', 'history', 'history_files', 'settings',
                       'source_config', 'split', 'training_poses', 'test_poses', 'conditions')


def _timed_phase(timings: RunTimings | None, stage: str,
                 **metadata) -> AbstractContextManager[dict]:
    return timings.phase(stage, **metadata) if timings is not None else nullcontext(dict(metadata))


def _endpoint_resume_settings(settings: dict) -> dict:
    """Ignore retired settings that never affected endpoint colors or labels."""
    return {key: value for key, value in settings.items() if key not in ('matrix', 'nonlinear')}


def _method_semantics() -> dict[str, str]:
    return dict(
        endpoint='c_target=c_flat_observed + H_target_pt - H_flat_pt; beta recomputed from target sphere geometry',
        observation='sRGB8 saved observations decoded by inverse_sRGB; H_pt is continuous material-point model',
        note='Target training measurements are used only for post-fit error/coverage diagnostics.')


def _continue_endpoint_scope(checkpoint: Checkpoints, prepared: dict) -> None:
    """Preserve earlier reports before an unfinished run drops retired methods."""
    manifest = checkpoint.manifest
    previous_methods = list(manifest.get('methods', []))
    archive = checkpoint.root / 'legacy_reports' / uuid.uuid4().hex[:8]
    json_write(archive / 'manifest.json', manifest)
    preserved = [archive / 'manifest.json']
    names = ('metrics.json', 'per_pose.csv', 'cross_method.csv', 'coverage.csv',
             'runtime.json', 'config_snapshot.json', 'method_semantics.json')
    old_reports = [checkpoint.root / name for name in names]
    old_reports += [path for path in (checkpoint.root / 'figures').rglob('*') if path.is_file()]
    for source in old_reports:
        if source.is_file():
            target = archive / source.relative_to(checkpoint.root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            preserved.append(target)
    retired = {key: value for key, value in manifest['completed'].items()
               if key == 'evaluation' or key.startswith('matrix/')}
    manifest['completed'] = {key: value for key, value in manifest['completed'].items()
                             if key not in retired}
    transition = dict(previous_methods=previous_methods, methods=list(METHODS),
                      report_domains=list(DOMAINS), archive=str(archive.relative_to(checkpoint.root)),
                      retired_checkpoints=retired,
                      archived_manifest_semantics='Original snapshot; historical paths are retained, not relocated',
                      retired_artifacts_note='Original retired files remain at their historical paths; they are not active endpoint outputs',
                      reason='Retired comparison methods; endpoint sample and label semantics unchanged',
                      resumed_at_utc=datetime.now(timezone.utc).isoformat())
    manifest.setdefault('scope_transitions', []).append(transition)
    manifest.update(methods=list(METHODS), report_domains=list(DOMAINS),
                    evaluation_version=VERSION, effective_settings=copy.deepcopy(prepared['settings']),
                    evaluation=dict(status='pending_endpoint_evaluation', archive=transition['archive']))
    snapshot_path = checkpoint.root / 'config_snapshot.json'
    snapshot = _load_json(snapshot_path)
    snapshot.update(experiment=prepared['settings'], original_snapshot=transition['archive'] + '/config_snapshot.json')
    json_write(snapshot_path, snapshot)
    semantics_path = checkpoint.root / 'method_semantics.json'
    json_write(semantics_path, _method_semantics())
    json_write(checkpoint.root / 'metrics.json',
               dict(status='pending_endpoint_evaluation', methods=list(METHODS), domains=list(DOMAINS),
                    original_report=transition['archive'] + '/metrics.json'))
    checkpoint.complete('inputs', [snapshot_path, checkpoint.root / 'split.json', semantics_path])
    scope_path = checkpoint.root / 'endpoint_scope.json'
    json_write(scope_path, dict(transition, effective_settings=prepared['settings']))
    checkpoint.complete('endpoint_scope', [scope_path, *preserved])


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


def start_run(prepared: dict, settings: dict, out: Path | None, resume: bool, *,
              update_scope: bool = True) -> Checkpoints:
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
        changed = []
        for key in RESUME_INPUT_FIELDS:
            before, after = previous.get(key), prepared['identity'].get(key)
            if key == 'settings' and isinstance(before, dict) and isinstance(after, dict):
                before, after = _endpoint_resume_settings(before), _endpoint_resume_settings(after)
            if before != after:
                changed.append(key)
        if changed:
            raise ValueError(f'Resume inputs differ ({", ".join(changed)}): {path}')
        checkpoint = Checkpoints(out, manifest)
        scope_change = (manifest.get('status') != 'completed'
                        and any(method not in METHODS for method in manifest.get('methods', [])))
        if scope_change:
            required = {key: value for key, value in manifest['completed'].items()
                        if key != 'evaluation' and not key.startswith('matrix/')}
            Checkpoints(out, dict(manifest, completed=required)).verify()
        else:
            checkpoint.verify()
        prepared['run_signature'] = manifest['run_signature']
        if scope_change and update_scope:
            _continue_endpoint_scope(checkpoint, prepared)
        return checkpoint
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Nonempty output requires compatible --resume: {out}')
    manifest = dict(version=VERSION, status='running', run_signature=prepared['run_signature'],
                    identity=prepared['identity'], completed={}, calibration={}, evaluation={},
                    training_diagnostics={}, execution={},
                    methods=list(METHODS), report_domains=list(DOMAINS),
                    model_semantics='H_pt=display_linear(contact)-display_linear(reference); inverse_sRGB(srgb8) domain')
    checkpoint = Checkpoints(out, manifest)
    checkpoint.save()
    json_write(out / 'config_snapshot.json', dict(simulation=prepared['base'],
             experiment=settings, source=str(prepared['root']), transfer_results=str(prepared['history_root']),
             source_run_signature=prepared['source_manifest'].get('run_signature'),
             transfer_run_signature=prepared['history_manifest'].get('run_signature')))
    json_write(out / 'split.json', dict(prepared['split'], independent_test=prepared['test_poses']))
    json_write(out / 'method_semantics.json', _method_semantics())
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


def _pose_source_selection(source_arrays: dict, pose_index: int) -> np.ndarray:
    return np.flatnonzero(np.asarray(source_arrays['image_index']) == pose_index)


def run_migrations(prepared: dict, checkpoint: Checkpoints, models: dict[str, object],
                   settings: dict, *, timings: RunTimings | None = None
                   ) -> dict[str, list[dict]]:
    """Create per-pose endpoint samples; every completed pose is atomically checkpointed."""
    by_condition = {}
    axes_x = np.asarray(prepared['source_arrays']['material_grid_x_mm'], dtype=np.float64)
    axes_y = np.asarray(prepared['source_arrays']['material_grid_y_mm'], dtype=np.float64)
    chunk_size = settings['chunk_size']
    for name, radius in prepared['conditions'].items():
        with _timed_phase(timings, 'endpoint_migration', condition=name,
                          new_pose_count=0, reused_pose_count=0) as timing:
            out_dir = checkpoint.root / 'samples' / name / 'migrated'
            out_dir.mkdir(parents=True, exist_ok=True)
            samples = []
            for pose_index, pose in enumerate(prepared['poses']):
                key = f'migration/{name}/{pose["pose_id"]}'
                path = out_dir / f'{pose["pose_id"]}.npz'
                if checkpoint.has(key):
                    sample = load_arrays(path)
                    timing['reused_pose_count'] += 1
                else:
                    timing['new_pose_count'] += 1
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
            timing['sample_count'] = sum(len(sample['grid_index']) for sample in samples)
            by_condition[name] = samples
    return by_condition


def fit_conditions(prepared: dict, checkpoint: Checkpoints, by_condition: dict, *,
                   timings: RunTimings | None = None) -> dict[str, NormalCalibration | None]:
    """Build and validate every endpoint LUT using the saved fixed source split."""
    endpoint_luts = {}
    for name, radius in prepared['conditions'].items():
        with _timed_phase(timings, 'endpoint_fit', condition=name,
                          action='reuse' if checkpoint.has(f'fit/{name}') else 'compute',
                          sample_count=sum(len(sample['grid_index']) for sample in by_condition[name])):
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
                    endpoint_luts: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Query one of the retained LUTs using the actual target observation."""
    colors = np.asarray(sample['color_residual_samples'], dtype=np.float64)
    if method == 'frozen_flat_lut':
        lut = prepared['source_lut']
    elif method == 'endpoint_migrated_lut':
        lut = endpoint_luts.get(condition)
    elif method == 'historical_same_curvature_lut':
        lut = prepared['history_luts'].get(condition)
    else:
        raise ValueError(f'Unknown compensation method: {method}')
    if lut is None:
        return np.full((len(colors), 2), np.nan), np.zeros(len(colors), dtype=bool), colors, True
    prediction, _, _ = lookup_slopes_raw(colors, lut)
    return prediction, np.isfinite(prediction).all(axis=-1), colors, True

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
                   reason='non-finite LUT prediction')
        return row, {}
    sample_view = dict(sample, color_residual_samples=np.asarray(query_colors, dtype=np.float32))
    metric, detail = evaluate_pose(sample_view, support_lut, indices, prediction=prediction)
    metric.update(metadata)
    metric['valid_prediction_count'] = len(take)
    metric['lut_support_applicable'] = True
    # Keep support in the original planar color domain visible for each LUT.
    raw_inside, raw_supported = support(np.asarray(sample['color_residual_samples'], dtype=np.float32)[take],
                                        raw_flat_support_lut)
    area = np.asarray(sample['area_mm2'], dtype=np.float64)[take]
    normalized_area = area / area.sum() if area.sum() else np.zeros_like(area)
    metric['raw_flat_lut_out_of_range_fraction'] = float(normalized_area @ ~raw_inside) if len(take) else None
    metric['raw_flat_lut_original_support_fraction'] = float(normalized_area @ raw_supported) if len(take) else None
    return metric, detail


def evaluate_all(prepared: dict, checkpoint: Checkpoints, endpoint_luts: dict, *,
                 timings: RunTimings | None = None) -> dict:
    """Compare endpoint and reference LUTs on full and shared material domains."""
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

    metric_summaries, row_sets, per_pose_rows, coverage_rows = {}, {}, [], []
    tolerances = prepared['settings']['metrics']['slope_tolerances']
    for name in condition_names:
        for pose_index, (pose, sample) in enumerate(zip(prepared['test_poses'], datasets[name], strict=True)):
            idx = sample['grid_index']
            common = common_by_pose[pose_index]
            common_take = np.searchsorted(idx, common)
            area = np.asarray(sample['area_mm2'], dtype=np.float64)
            full_area, common_area = float(area.sum()), float(area[common_take].sum())
            coverage_rows.append(dict(condition=name, pose_id=pose['pose_id'],
                valid_test_samples=len(idx), common_samples=len(common),
                full_area_mm2=full_area, common_area_mm2=common_area,
                common_area_fraction=common_area / full_area if full_area > 0. else 0.,
                status='ok' if len(idx) and len(common) else 'empty'))

    for method in METHODS:
        for name in condition_names:
            grouped_rows = {domain: [] for domain in DOMAINS}
            grouped_details = {domain: [] for domain in DOMAINS}
            for pose_index, (pose, sample) in enumerate(zip(prepared['test_poses'], datasets[name], strict=True)):
                query_start = time.perf_counter()
                pred, valid, query_colors, _ = _predict_method(method, sample, name, prepared, endpoint_luts)
                if timings is not None:
                    timings.record_query(method, name, time.perf_counter() - query_start,
                                         len(sample['grid_index']))
                if method == 'endpoint_migrated_lut':
                    method_lut = endpoint_luts.get(name)
                    lut_status = checkpoint.manifest.get('calibration', {}).get(name, {}).get('status', 'unknown')
                elif method == 'historical_same_curvature_lut':
                    method_lut = prepared['history_luts'].get(name)
                    lut_status = prepared['calibration_status'].get(name, {}).get('status', 'published')
                else:
                    method_lut = prepared['source_lut']
                    lut_status = 'published'
                if method_lut is None:
                    lut_status = 'unavailable'
                pose_meta = dict(pose_id=pose['pose_id'], anchor_id=pose['anchor_id'],
                                 u=pose['center_uv'][0], v=pose['center_uv'][1], depth_mm=pose['depth_mm'],
                                 method=method, condition=name, lut_status=lut_status)
                method_support_lut = method_lut if method_lut is not None else prepared['source_lut']
                for domain in DOMAINS:
                    indices = None if domain == 'full' else common_by_pose[pose_index]
                    metric, detail = _evaluate_prediction(sample, pred, valid, query_colors,
                                                           method_support_lut, indices,
                                                           method=method, metadata=dict(pose_meta, domain=domain),
                                                           raw_flat_support_lut=prepared['source_lut'])
                    grouped_rows[domain].append(metric)
                    if detail:
                        grouped_details[domain].append(detail)
                    per_pose_rows.append(metric)
            for domain in DOMAINS:
                key = f'{method}/{name}/{domain}'
                summary = aggregate(grouped_rows[domain], grouped_details[domain], tolerances)
                summary.update(method=method, condition=name, domain=domain)
                if method == 'historical_same_curvature_lut':
                    summary['lut_status'] = prepared['calibration_status'].get(name, {}).get('status', 'published')
                elif method == 'endpoint_migrated_lut':
                    summary['lut_status'] = checkpoint.manifest.get('calibration', {}).get(name, {}).get('status')
                metric_summaries[key] = summary
                row_sets[key] = grouped_rows[domain]

    comparisons = {}
    repeats = prepared['settings']['metrics']['bootstrap_repeats']
    seed = prepared['settings']['metrics']['seed']
    for name in condition_names:
        baseline = row_sets[f'frozen_flat_lut/{name}/common']
        comparisons[name] = {
            f'{method}/common': paired_bootstrap(row_sets[f'{method}/{name}/common'], baseline,
                                                 repeats=repeats, seed=seed)
            for method in METHODS if method != 'frozen_flat_lut'}

    historical_reproduction = {}
    for name in condition_names:
        archived = prepared['history_metrics'].get('metrics', {}).get(f'flat/{name}/common', {}).get('rmse')
        reproduced = metric_summaries[f'frozen_flat_lut/{name}/common'].get('rmse')
        matches = (archived is not None and reproduced is not None
                   and np.isclose(archived, reproduced, rtol=0, atol=1e-12))
        historical_reproduction[name] = dict(archived_rmse=archived, reproduced_rmse=reproduced,
                                             absolute_difference=(abs(archived - reproduced)
                                                                  if archived is not None and reproduced is not None else None),
                                             status='matched' if matches else 'mismatch')
    result = dict(version=VERSION, status='completed', conditions=prepared['conditions'],
                  methods=list(METHODS), domains=list(DOMAINS), metrics=metric_summaries,
                  evaluation_settings=prepared['settings'],
                  scope_transitions=copy.deepcopy(checkpoint.manifest.get('scope_transitions', [])),
                  comparisons=comparisons, coverage=coverage_rows,
                  historical_flat_baseline_reproduction=historical_reproduction,
                  calibration=checkpoint.manifest['calibration'],
                  training_transfer_diagnostics=checkpoint.manifest.get('training_diagnostics', {}),
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
                     metrics: dict, status_labels: dict | None = None) -> None:
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
            value = metrics.get(f'{method}/{condition}/common', {}).get('rmse')
            if value is not None:
                values[i, j] = value
    method_labels = {'frozen_flat_lut': 'Frozen flat LUT',
                     'endpoint_migrated_lut': 'Endpoint H_pt LUT',
                     'historical_same_curvature_lut': 'Historical same-curvature LUT'}
    fig = plt.figure(figsize=(11, 7.4), layout='constrained')
    grid_spec = fig.add_gridspec(2, 1, height_ratios=(4.2, 1.35), hspace=.12)
    ax = fig.add_subplot(grid_spec[0])
    status_ax = fig.add_subplot(grid_spec[1])
    for index, method in enumerate(methods):
        ax.plot(np.arange(len(conditions)), values[index], marker='o', linewidth=1.6,
                label=method_labels.get(method, method))
    ax.set_xticks(np.arange(len(conditions)), labels)
    ax.set_ylabel('Slope-vector RMSE (unitless; lower is better)')
    ax.set_xlabel('Reference-surface condition')
    ax.set_title('LUT comparisons on the common test domain')
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8.5)
    status_map = status_labels or {}
    status_values = [[method_labels.get(method, method)]
                     + [status_map.get(method, {}).get(condition, 'n/a') for condition in conditions]
                     for method in methods]
    status_ax.axis('off')
    table = status_ax.table(cellText=status_values, colLabels=['Method', *labels],
                            colWidths=[.34] + [.66 / len(conditions)] * len(conditions),
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    status_ax.set_title('LUT validation status by condition', fontsize=9, pad=2)
    fig.savefig(figure_dir / 'rmse_lut_comparison.png', dpi=160)
    plt.close(fig)
    npz_write(figure_dir / 'figure_data.npz', method_names=np.asarray(methods),
              condition_names=np.asarray(conditions), slope_rmse=values,
              lut_validation_status=np.asarray([
                  [status_map.get(method, {}).get(condition, 'n/a') for condition in conditions]
                  for method in methods], dtype=str),
              lut_validation_methods=np.asarray(methods))

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
                methods=list(METHODS), domains=list(DOMAINS))


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
    parser.add_argument('--compare-nonlinear', action='store_true',
                        help='Compare the v2 finite-slope inverse with existing endpoint LUTs')
    parser.add_argument('--endpoint-results', type=Path,
                        help='Completed compensation run reused by --compare-nonlinear')
    args = parser.parse_args(argv)
    timings = RunTimings(device='cpu', run_mode='resume' if args.resume else 'fresh')
    if args.resume and args.out is None:
        parser.error('--resume requires --out')
    if args.compare_nonlinear != (args.endpoint_results is not None):
        parser.error('--compare-nonlinear and --endpoint-results must be supplied together')
    with timings.phase('input_preparation'):
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
    if args.compare_nonlinear:
        from .comparison import run_comparison
        run_comparison(prepared, args.endpoint_results, resolved_out,
                       resume=args.resume, dry_run=args.dry_run)
        return
    if args.dry_run:
        if args.resume:
            start_run(prepared, settings, resolved_out, True, update_scope=False)
        print(json.dumps(_dry_run_report(prepared, resolved_out), ensure_ascii=False, indent=2))
        return

    timings.set_device(str(prepared['base'].get('render', {}).get('device', 'cuda:0')))
    with timings.phase('output_setup'):
        checkpoint = start_run(prepared, settings, resolved_out, args.resume)
    print(f'Experiment output: {checkpoint.root}', flush=True)
    if checkpoint.manifest.get('status') == 'completed':
        print('Completed run verified; no artifacts rewritten.')
        return
    resumed_checkpoint_count = len(checkpoint.manifest.get('completed', {})) if args.resume else 0
    timings.attach(checkpoint.root)
    start_time = time.perf_counter()
    try:
        with timings.phase('model_initialization'):
            models, execution = _build_photometry_models(prepared)
            checkpoint.manifest.setdefault('execution', {}).update(
                device=prepared['base'].get('render', {}).get('device', 'cuda:0'),
                photometry=execution)
            checkpoint.save()
        migrated = run_migrations(prepared, checkpoint, models, settings, timings=timings)
        endpoint_luts = fit_conditions(prepared, checkpoint, migrated, timings=timings)
        with timings.phase('training_diagnostics'):
            training_diagnostics = diagnose_training_transfer(prepared, checkpoint, migrated)
            checkpoint.manifest.setdefault('training_diagnostics', {}).update(training_diagnostics)
            checkpoint.save()
        with timings.phase('evaluation'):
            metrics = evaluate_all(prepared, checkpoint, endpoint_luts, timings=timings)
        with timings.phase('plotting'):
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
                             metrics['metrics'], status_labels=status_labels)
    except BaseException as error:
        try:
            timings.finish('failed')
        except Exception as reporting_error:
            add_note = getattr(error, 'add_note', None)
            if add_note is not None:
                add_note(f'Unable to save timing report: {reporting_error}')
            else:
                print(f'Unable to save timing report: {reporting_error}', flush=True)
        raise
    elapsed = time.perf_counter() - start_time
    timings.finish('completed')
    json_write(checkpoint.root / 'runtime.json',
               dict(elapsed_seconds=elapsed,
                    elapsed_scope='model_initialization_through_plotting_current_invocation',
                    execution=checkpoint.manifest['execution'],
                    resumed_checkpoint_count=resumed_checkpoint_count,
                    workload={key: value for key, value in prepared['workload'].items()
                              if key != 'source_sample_count_by_pose'},
                    timing=timings.report(status='completed')))
    checkpoint.manifest['evaluation'] = dict(
        status=metrics['status'], metric_count=len(metrics['metrics']),
        baseline_reproduction=metrics['historical_flat_baseline_reproduction'])
    files = [checkpoint.root / name for name in
             ('metrics.json', 'per_pose.csv', 'cross_method.csv', 'coverage.csv', 'runtime.json')]
    files += [checkpoint.root / 'figures' / name for name in ('rmse_lut_comparison.png', 'figure_data.npz')]
    files += [path for folder in ('training_diagnostics', 'calibration', 'samples',
                                 'runtime_sessions')
              for path in (checkpoint.root / folder).rglob('*') if path.is_file()]
    checkpoint.complete('evaluation', files, status=metrics['status'], elapsed_seconds=elapsed)
    checkpoint.manifest['status'] = 'completed'
    checkpoint.manifest['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    checkpoint.save()
    print(f'Completed: {checkpoint.root / "metrics.json"}', flush=True)
