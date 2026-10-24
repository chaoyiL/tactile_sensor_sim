"""Run unchanged full calibration and quality CLIs for power-only candidates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiment.config import (load_experiment_config, resolve_light_distribution_settings,
                               PROJECT_ROOT, resolve_config_path)
from utils.config import validate_config
from utils.calibration_scan import build_scan_poses
from utils.lut_results import load_lut_result
from utils.light_distribution import DISCRETE_EMISSION_MODEL, point_offsets_mm


VERSION = 'power_only_full_calibration_v2'


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def fixed_conditions(config: dict) -> dict:
    """Every field except the explicitly allowed absolute RGB powers is fixed."""
    result = copy.deepcopy(config)
    for light in result['lights']:
        light.pop('point_power_rgb_w')
    return result


def candidate_config(source: dict, candidate: dict) -> dict:
    """Build candidate watts, including unrestricted per-emitter RGB arrays."""
    result = copy.deepcopy(source)
    if sorted(light['side'] for light in result['lights']) != ['bottom', 'left', 'right', 'top']:
        raise ValueError('Power families require exactly one strip per edge')
    for light in result['lights']:
        if light.get('emission_model') != DISCRETE_EMISSION_MODEL or light['point_spacing_mm'] != 6.25:
            raise ValueError('Power optimization requires the saved 6.25 mm discrete RGB model')
        side = light['side']
        if candidate['family'] == 'explicit':
            light['point_power_rgb_w'] = copy.deepcopy(candidate['point_power_rgb_w'][side])
            continue
        offsets = point_offsets_mm(light)
        count = len(offsets)
        total = candidate['short_power_w'] if side in ('left', 'right') else candidate['long_power_w']
        if candidate['family'] == 'complement':
            colors = {'left': [0., 0., 1.], 'right': [.5, .5, 0.],
                      'top': [1., 0., 0.], 'bottom': [0., 1., 0.]}
            powers = np.tile(colors[side], (count, 1)) * total / count
        else:
            # Smooth G/B spectral gradient gives one consistent X colour direction;
            # top-only R gives the independent Y colour direction.
            t = offsets / (light['length_mm'] / 2)
            blend = .5 * (1 + candidate['strength'] * t)
            if side == 'top':
                colors = np.column_stack((np.full(count, .5), .5 * blend, .5 * (1 - blend)))
            elif side == 'bottom':
                colors = np.column_stack((np.zeros(count), blend, 1 - blend))
            else:
                colors = np.tile([0., 0., 1.] if side == 'left' else [0., 1., 0.], (count, 1))
            powers = colors * total / count
        light['point_power_rgb_w'] = powers.tolist()
    if fixed_conditions(result) != fixed_conditions(source):
        raise ValueError('Candidate changed a non-power condition')
    return validate_config(result)


def curve_summary(root: Path) -> dict:
    quality = json.loads((root / 'evaluation/lut_quality/quality.json').read_text(encoding='utf-8'))
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    ref = quality['reference']
    holdout = quality['holdout']
    return dict(calibration_status=quality.get('calibration_status', manifest['status']),
                metric_version=quality['metric_version'],
                tolerance_domain=quality['tolerance_domain'],
                sample_count=ref['sample_count'] + holdout['sample_count'],
                reference=ref['curves'], holdout=holdout['curves'],
                weighting=quality['weighting'], split=quality['split'])


def implementation_hashes() -> dict[str, str]:
    paths = [Path(__file__), PROJECT_ROOT / 'experiment/config.py',
             *sorted((PROJECT_ROOT / 'utils').glob('*.py')),
             *sorted((PROJECT_ROOT / 'experiment/lut_quality').glob('*.py'))]
    return {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lut-results', type=Path, required=True,
                        help='Saved analytic baseline; its snapshot fixes all non-power conditions')
    parser.add_argument('--target-results', type=Path, required=True,
                        help='Saved target LUT with an existing evaluation/lut_quality report')
    parser.add_argument('--exp-config', type=Path, default=PROJECT_ROOT / 'configs/light_distribution.json')
    parser.add_argument('--out', type=Path, help='New experiment folder under baseline/evaluation')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume completed candidates, never overwrite partial ones')
    args = parser.parse_args(argv)
    source_root, source_manifest, backend = load_lut_result(args.lut_results, require_published=False)
    target_root, _, _ = load_lut_result(args.target_results, require_published=False)
    if backend != 'analytic':
        raise ValueError('Light distribution optimization only supports analytic')
    settings_path = args.exp_config.expanduser().resolve()
    settings = resolve_light_distribution_settings(load_experiment_config(settings_path))
    source = source_manifest['config']
    candidates = [(row['name'], candidate_config(source, row)) for row in settings['candidates']]
    source_quality, target_quality = curve_summary(source_root), curve_summary(target_root)
    quality_settings = json.loads((source_root / 'evaluation/lut_quality/manifest.json')
                                  .read_text(encoding='utf-8'))['settings']
    implementation = implementation_hashes()
    for key in ('metric_version', 'tolerance_domain', 'weighting', 'split'):
        if source_quality[key] != target_quality[key]:
            raise ValueError(f'Source and target quality {key} differ')
    if [row['tau'] for row in source_quality['reference']] != [
            row['tau'] for row in target_quality['reference']]:
        raise ValueError('Source and target quality tolerances differ')
    inputs = {f'{role}/{name}': hashlib.sha256((root / name).read_bytes()).hexdigest()
              for role, root in (('source', source_root), ('target', target_root))
              for name in ('manifest.json', 'evaluation/lut_quality/quality.json',
                           'evaluation/lut_quality/manifest.json')}
    output = (args.out.expanduser().resolve() if args.out else
              source_root / 'evaluation/light_distribution')
    if not output.is_relative_to(source_root / 'evaluation'):
        raise ValueError(f'Output must be under {source_root / "evaluation"}: {output}')
    signature_data = dict(version=VERSION, source=source, settings=settings,
                          implementation=implementation, quality_settings=quality_settings,
                          inputs=inputs,
                          source_quality=source_quality, target_quality=target_quality)
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    config_folder = resolve_config_path(f'light_distribution_runs/{signature[:16]}',
                                        PROJECT_ROOT / 'configs/light_distribution.json')
    summary = dict(signature=signature, source=str(source_root), target=str(target_root),
                   output=str(output), config_folder=str(config_folder),
                   candidates=len(candidates), poses_per_candidate=len(build_scan_poses(source)),
                   source_quality=source_quality, target_quality=target_quality,
                   implementation=implementation, quality_settings=quality_settings,
                   input_sha256=inputs,
                   settings=settings, mode='full fresh rendering, unchanged calibration and quality')
    if args.dry_run:
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    manifest_path = output / 'manifest.json'
    if output.exists() and any(output.iterdir()):
        if not args.resume or not manifest_path.exists():
            raise ValueError(f'Nonempty output requires compatible --resume: {output}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['signature'] != signature:
            raise ValueError(f'Incompatible experiment signature: {manifest_path}')
    else:
        output.mkdir(parents=True, exist_ok=True)
        manifest = dict(summary, status='running', results={})
    config_folder.mkdir(parents=True, exist_ok=True)
    calibration_path = config_folder / 'calibration.json'
    quality_path = config_folder / 'lut_quality.json'
    for path, document in ((calibration_path, dict(format_version=1, calibration=source['calibration'])),
                           (quality_path, dict(format_version=1, lut_quality=quality_settings))):
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != document:
                raise ValueError(f'Configuration snapshot changed: {path}')
        else:
            write_json(path, document)
    (output / 'logs').mkdir(exist_ok=True)
    write_json(manifest_path, manifest)
    try:
        for name, config in candidates:
            if implementation_hashes() != implementation:
                raise RuntimeError('Implementation changed during experiment; refusing mixed-code results')
            if manifest['results'].get(name, {}).get('status') == 'completed':
                continue
            trial = output / name
            if trial.exists() and any(trial.iterdir()):
                raise ValueError(f'Partial candidate preserved; choose a new output folder: {trial}')
            config_path = config_folder / f'{name}.json'
            simulation = {key: value for key, value in config.items() if key != 'calibration'}
            if config_path.exists():
                if json.loads(config_path.read_text(encoding='utf-8')) != simulation:
                    raise ValueError(f'Candidate config already exists with different content: {config_path}')
            else:
                write_json(config_path, simulation)
            print(f'[{name}] full calibration: {trial}', flush=True)
            command = [sys.executable, str(PROJECT_ROOT / 'calibrate_lut.py'), '--backend', 'analytic',
                       '--config', str(config_path), '--calibration-config', str(calibration_path),
                       '--out', str(trial)]
            with (output / 'logs' / f'{name}_dry_run.log').open('w', encoding='utf-8') as log:
                subprocess.run([*command, '--dry-run'], cwd=PROJECT_ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            with (output / 'logs' / f'{name}_calibration.log').open('w', encoding='utf-8') as log:
                calibrated = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
            trial_manifest_path = trial / 'manifest.json'
            if not trial_manifest_path.exists():
                raise RuntimeError(f'Calibration did not produce a manifest: {trial}')
            trial_manifest = json.loads(trial_manifest_path.read_text(encoding='utf-8'))
            if trial_manifest['stage'] != 'completed' or trial_manifest['status'] not in ('passed', 'failed'):
                raise RuntimeError(f'Calibration incomplete, preserved for diagnosis: {trial}')
            if fixed_conditions(trial_manifest['config']) != fixed_conditions(source):
                raise RuntimeError(f'Non-power conditions changed: {trial}')
            print(f'[{name}] full quality (calibration {trial_manifest["status"]})', flush=True)
            command = [sys.executable, '-m', 'experiment.lut_quality', '--lut-results', str(trial),
                       '--exp-config', str(quality_path)]
            with (output / 'logs' / f'{name}_quality.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            if implementation_hashes() != implementation:
                raise RuntimeError('Implementation changed during candidate execution; result not adopted')
            result = curve_summary(trial)
            result.update(status='completed', path=str(trial), config=str(config_path),
                          calibration_exit_code=calibrated.returncode,
                          same_sample_count=result['sample_count'] == source_quality['sample_count'])
            manifest['results'][name] = result
            write_json(manifest_path, manifest)
            a = next(row['incompatibility'] for row in result['reference'] if row['tau'] == .1)
            print(f'[{name}] A(0.1)={a:.6f}, samples={result["sample_count"]}', flush=True)
    except BaseException as exc:
        manifest['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        manifest['error'] = dict(type=type(exc).__name__, message=str(exc))
        write_json(manifest_path, manifest)
        raise
    manifest['status'] = 'completed'
    write_json(manifest_path, manifest)
    return manifest
