"""Sweep surface/strip length, calibrate each geometry and evaluate global LUT quality."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiment.config import (PROJECT_ROOT, load_experiment_config,
                               resolve_config_path, resolve_length_lutqual_settings,
                               resolve_lut_quality_settings)
from utils.backend_layout import CONFIG_DIR
from experiment.length_normdiff.runner import center_poses, length_config
from experiment.lut_quality.metric import METRIC_VERSION, evaluate_holdout
from experiment.lut_quality.runner import load_inputs, main as quality_main, sha256, write_json
from utils.calibration_pipeline import (_config_signature, fit_and_validate,
                                         parse_calibration_settings)
from utils.calibration_samples import build_material_grid
from utils.calibration_scan import build_scan_poses
from utils.geometry import LIGHT_GEOMETRY_MODEL
from utils.normal_mapping import NormalCalibration
from .scan import render_samples


EXPERIMENT_VERSION = 'length_lutqual_v1'


def checkpoint_stage(point: Path, state: dict, stage: str, files: list[Path], data: dict):
    state['stages'][stage] = dict(data=data, artifacts={
        str(path.relative_to(point)): sha256(path) for path in files})
    state['status'] = 'running'
    state.pop('error', None)
    write_json(point / 'checkpoint.json', state)


def verify_stages(point: Path, state: dict):
    """Reject missing/changed completed artifacts rather than silently relabelling them."""
    for stage, entry in state['stages'].items():
        for name, expected in entry['artifacts'].items():
            path = (point / name).resolve()
            if not path.is_relative_to(point.resolve()) or not path.is_file() or sha256(path) != expected:
                raise ValueError(f'Invalid {stage} checkpoint artifact: {path}; choose a new --out')


def fit_lut(config: dict, config_path: Path, point: Path):
    """Use the existing full-pose split, fitting, thresholds and publication rules."""
    with np.load(point / 'normal_samples.npz', allow_pickle=False) as archive:
        samples = {key: archive[key] for key in ('color_residual_samples', 'slope_samples',
                                                'image_index', 'image_paths')}
        metadata = json.loads(str(archive['metadata']))
    settings = replace(parse_calibration_settings(config, config_path, point), visualization_enabled=False)
    report, training, final, diagnostic = fit_and_validate(samples, settings, point, metadata)
    # A failed validation is an experimental outcome, not a reason to omit this length.
    files = [point / 'normal_lut_train.npz', point / 'verification' / 'validation.json']
    if final is not None:
        files.append(point / 'normal_lut.npz')
    return report, files


def quality_attempt_dir(point: Path) -> Path:
    """Preserve interrupted report output; reuse expensive render/fit stages on retry."""
    target = point / 'quality'
    number = 1
    while target.exists() and (not target.is_dir() or any(target.iterdir())):
        target = point / f'quality_retry_{number:03d}'
        number += 1
    return target


def evaluate_point(config_path: Path, point: Path, metric: dict):
    target = quality_attempt_dir(point)
    report = quality_main([
        '--lut-results', str(point),
        '--exp-config', str(config_path),
        '--samples', str(point / 'normal_samples.npz'),
        '--lut', str(point / 'normal_lut_train.npz'), '--out', str(target), '--split', 'auto',
        '--taus', *map(str, metric['taus']), '--detail-tau', str(metric['detail_tau']),
        '--weighting', metric['weighting'], '--top-bins', str(metric['top_bins']),
        '--examples-per-bin', str(metric['examples_per_bin'])])
    return report, target


def center_quality(point: Path, quality_dir: Path, metric: dict):
    colors, slopes, indices, keys, metadata, lut, warnings = load_inputs(
        point / 'center_samples.npz', point / 'normal_lut_train.npz')
    with np.load(quality_dir / 'per_bin.npz', allow_pickle=False) as archive:
        fitted = {key: archive[key] for key in ('bin_ids', 'optimal_center')}
    report = evaluate_holdout(colors, slopes, indices, lut, metric['taus'], fitted,
                              weighting=metric['weighting'])
    report.update(independent_validation=False,
                  meaning='fixed-center control, evaluated by the full-scan training LUT; '
                          'may overlap calibration poses; never used to fit another LUT',
                  image_keys=keys.tolist(), warnings=warnings)
    write_json(point / 'center_quality.json', report)
    return report


def run_point(config: dict, poses: list, point: Path, config_path: Path,
              signature: str, metric: dict):
    checkpoint = point / 'checkpoint.json'
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding='utf-8'))
        if state.get('signature') != signature or state.get('config_signature') != _config_signature(config):
            raise ValueError(f'Point checkpoint signature mismatch: {point}')
        verify_stages(point, state)
        if 'completed' in state['stages']:
            print(f'[length={config["surface"]["size_mm"][0]:g}] resuming completed point', flush=True)
            return json.loads((point / 'point.json').read_text(encoding='utf-8'))
    else:
        if point.exists() and (not point.is_dir() or any(point.iterdir())):
            raise ValueError(f'Point output has no valid checkpoint: {point}')
        point.mkdir(parents=True, exist_ok=True)
        state = dict(signature=signature, config_signature=_config_signature(config),
                     status='running', stages={})
        write_json(point / 'config.json', config)
        write_json(checkpoint, state)
    started = time.perf_counter()
    try:
        if 'samples' not in state['stages']:
            minimum = parse_calibration_settings(config, config_path, point).minimum_samples_per_pose
            scan = render_samples(config, poses, point, minimum)
            write_json(point / 'scan.json', scan)
            checkpoint_stage(point, state, 'samples', [point / name for name in (
                'normal_samples.npz', 'center_samples.npz', 'scan.json', 'config.json')],
                dict(sample_count=scan['full_scan']['sample_count']))
        if 'fit' not in state['stages']:
            validation, files = fit_lut(config, config_path, point)
            checkpoint_stage(point, state, 'fit', files, dict(validation_status=validation['status']))
        if 'quality' not in state['stages']:
            quality, quality_dir = evaluate_point(config_path, point, metric)
            checkpoint_stage(point, state, 'quality', sorted(path for path in quality_dir.iterdir() if path.is_file()),
                             dict(directory=quality_dir.name))
        quality_dir = point / state['stages']['quality']['data']['directory']
        quality = json.loads((quality_dir / 'quality.json').read_text(encoding='utf-8'))
        if 'center' not in state['stages']:
            center_quality(point, quality_dir, metric)
            checkpoint_stage(point, state, 'center', [point / 'center_quality.json'], {})
        center = json.loads((point / 'center_quality.json').read_text(encoding='utf-8'))
        validation = json.loads((point / 'verification' / 'validation.json').read_text(encoding='utf-8'))
        scan = json.loads((point / 'scan.json').read_text(encoding='utf-8'))
        training = NormalCalibration.load(point / 'normal_lut_train.npz')
        strips = {light['side']: light['length_mm'] for light in config['lights']}
        result = dict(signature=signature, length_mm=config['surface']['size_mm'][0],
                      top_length_mm=strips['top'], bottom_length_mm=strips['bottom'],
                      config_signature=state['config_signature'], quality_directory=quality_dir.name,
                      lut_size=training.size, lut_color_min=training.color_min.tolist(),
                      lut_color_max=training.color_max.tolist(),
                      scan=scan, validation=validation, full_scan=quality, fixed_center=center,
                      elapsed_seconds_this_attempt=time.perf_counter() - started)
        write_json(point / 'point.json', result)
        checkpoint_stage(point, state, 'completed', [point / 'point.json'], {})
        state['status'] = 'completed'
        write_json(checkpoint, state)
        return result
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(checkpoint, state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-config', '--config', dest='exp_config', type=Path,
                        default=CONFIG_DIR / 'length_lutqual.json')
    parser.add_argument('--lengths', nargs='+', type=float)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--backend', choices=('analytic', 'zemax', 'xense'),
                        help='Check the backend selected by the experiment snapshot')
    parser.add_argument('--taus', nargs='+', type=float, help='Shared slope tolerance curve, not RGB noise or degrees')
    parser.add_argument('--detail-tau', type=float)
    parser.add_argument('--weighting', choices=('sample', 'pose_equal'))
    parser.add_argument('--dry-run', action='store_true', help='Validate all lengths and show scan counts without rendering/writing')
    args = parser.parse_args(argv)
    config_path = args.exp_config.expanduser().resolve()
    experiment = load_experiment_config(config_path)
    selected_backend = experiment['simulation'].get('backend', 'analytic')
    if args.backend is not None and args.backend != selected_backend:
        parser.error(f'--backend {args.backend} disagrees with snapshot {selected_backend}')
    if selected_backend in ('mujoco', 'xense', 'zemax'):
        parser.error(f'Length LUT-quality sweep supports only analytic backend; '
                     f'snapshot selects {selected_backend}')
    sweep = resolve_length_lutqual_settings(experiment, lengths_mm=args.lengths)
    metric = resolve_lut_quality_settings(experiment, taus=args.taus, detail_tau=args.detail_tau,
                                         weighting=args.weighting, split='auto')
    # Existing-sample paths are intentionally not part of this freshly rendered experiment.
    metric = {key: metric[key] for key in ('taus', 'detail_tau', 'weighting', 'split', 'top_bins', 'examples_per_bin')}
    output = args.out.expanduser().resolve() if args.out else resolve_config_path(sweep['output_dir'], config_path)
    base = experiment['simulation']
    plan = []
    for length in sweep['lengths_mm']:
        config = length_config(base, length)
        settings = parse_calibration_settings(config, config_path, output)
        poses = build_scan_poses(config)
        if len(poses) < 2:
            raise ValueError(f'length={length:g}: LUT holdout requires at least two full scan poses')
        grid = build_material_grid(config)
        plan.append((length, config, poses))
        print(f'length={length:g} mm; {len(poses)} full scan + {len(center_poses(poses))} centre poses '
              f'+ 1 reference; material grid={grid.shape}; LUT={settings.lut_size}^3', flush=True)
    print(f'common taus={metric["taus"]}, weighting={metric["weighting"]}; split seed=42', flush=True)
    print(f'experiment config: {config_path}\noutput: {output}', flush=True)
    if args.dry_run:
        return dict(lengths_mm=sweep['lengths_mm'], metric_settings=metric,
                    pose_counts=[len(poses) for _, _, poses in plan])
    if not str(base['render'].get('device', '')).startswith('cuda'):
        raise ValueError('length_lutqual requires a CUDA render.device, just like length_normdiff')
    # Fail before rendering if scientific plotting dependencies are unavailable.
    from .plots import summarize
    paths = [PROJECT_ROOT / name for name in (
        'experiment/config.py', 'experiment/length_lutqual/runner.py', 'experiment/length_lutqual/scan.py',
        'experiment/length_lutqual/plots.py', 'experiment/length_normdiff/runner.py',
        'experiment/lut_quality/metric.py', 'experiment/lut_quality/runner.py', 'experiment/lut_quality/plots.py')]
    paths += sorted((PROJECT_ROOT / 'utils').glob('*.py'))
    identity = dict(experiment_version=EXPERIMENT_VERSION, metric_version=METRIC_VERSION,
                    base_config=base, lengths_mm=sweep['lengths_mm'], metric_settings=metric,
                    source_exp_config=str(config_path), light_geometry_model=LIGHT_GEOMETRY_MODEL,
                    sampling='all configured poses and every valid contact sample; insufficient poses abort',
                    fitting='existing fit_and_validate; whole-pose split seed 42; quality always uses training LUT',
                    binning='per-length trained LUT range and common LUT size; no shared RGB bounds imposed',
                    power_policy='unchanged strip total power_rgb_w',
                    implementation_sha256={str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in paths})
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()
    manifest_path = output / 'manifest.json'
    if output.exists() and not output.is_dir():
        raise ValueError(f'Output is not a directory: {output}')
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('signature') != signature:
            raise ValueError('Output contains a different experiment/configuration/implementation; choose a new --out')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Output is not empty and has no experiment manifest; choose a new --out')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(format_version=1, signature=signature, status='running', experiment=identity,
                    completed_lengths_mm=[])
    write_json(manifest_path, manifest)
    rows = []
    try:
        for length, config, poses in plan:
            # Round-trippable names avoid collisions between nearby floating-point lengths.
            point = output / ('length_' + repr(length))
            result = run_point(config, poses, point, config_path, signature, metric)
            rows.append(result)
            summarize(rows, output, identity)
            manifest['completed_lengths_mm'] = [row['length_mm'] for row in rows]
            write_json(manifest_path, manifest)
            detail = next(row for row in result['full_scan']['holdout']['curves'] if row['tau'] == metric['detail_tau'])
            print(f'[length={length:g}] held-out runtime error={detail["runtime_error"]:.6f}; '
                  f'calibration validation={result["validation"]["status"]}', flush=True)
        manifest['status'] = 'completed'
        write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(manifest_path, manifest)
        raise
    print(f'Figure: {output / "length_lutqual.png"}', flush=True)
    return rows
