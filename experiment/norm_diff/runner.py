"""Calculate the NormDiff score for the current simulated lighting setup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiment.config import (load_experiment_config,
                               resolve_config_path, resolve_metric_settings)
from utils.backend_layout import CONFIG_DIR, canonical_backend
from utils.calibration_pipeline import _config_signature
from utils.calibration_samples import build_material_grid, render_reference
from utils.config import load_config, validate_config
from utils.lut_results import load_lut_result
from utils.geometry import LIGHT_GEOMETRY_MODEL
from .metric import compute_norm_diff, evaluate_lut_noise


def _observed_linear_rgb(config, residual, material_index):
    try:
        import torch
        from utils.renderer_gpu import AnalyticGpuContext
    except ImportError as exc:
        raise RuntimeError('Reconstructing observed RGB requires the gpu extra') from exc
    grid = build_material_grid(config)
    if material_index.ndim != 2 or material_index.shape[1] != 2 \
            or np.any(material_index < 0) \
            or np.any(material_index[:, 0] >= grid.shape[0]) \
            or np.any(material_index[:, 1] >= grid.shape[1]):
        raise ValueError('material_grid_index is invalid for the configured material grid')
    with torch.inference_mode():
        context = AnalyticGpuContext(config)
        reference = render_reference(context, grid)
        reference_color = reference.grid_color.detach().cpu().numpy()
    return residual + reference_color[material_index[:, 0], material_index[:, 1]]


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Compute pose-balanced NormDiff under the current simulated lighting')
    parser.add_argument('--lut-results', type=Path, required=True,
                        help='Published calibration directory containing manifest and LUT')
    parser.add_argument('--samples', type=Path, help='Override saved calibration samples')
    parser.add_argument('--config', type=Path, help='Rendering config matching the saved samples')
    parser.add_argument('--exp-config', type=Path, default=CONFIG_DIR / 'norm_diff.json',
                        help='Independent NormDiff evaluation settings (not the rendering config)')
    parser.add_argument('--out', type=Path,
                        help='Override exp_config.norm_diff_evaluation.output_file')
    parser.add_argument('--color-input', choices=('observed', 'residual'),
                        help='Observed reconstructs post-tonemap linear RGB; residual is an explicit extension')
    parser.add_argument('--noise-fraction', type=float,
                        help='Override exp_config.norm_diff.noise_fraction; 0.03 means +/-3%% '
                             'per channel (fallback: 0.03)')
    parser.add_argument('--noise-floor', type=float,
                        help='Override exp_config.norm_diff.noise_floor (fallback: 0)')
    parser.add_argument('--validate-lut-noise', action=argparse.BooleanOptionalAction, default=None,
                        help='Also evaluate the saved training LUT on held-out poses using '
                             'the same configured RGB perturbation')
    parser.add_argument('--theta-weight', type=float)
    parser.add_argument('--phi-weight', type=float)
    parser.add_argument('--theta-bins', type=int)
    parser.add_argument('--phi-bins', type=int)
    parser.add_argument('--anchors-per-target', type=int)
    parser.add_argument('--theta-ceiling-degrees', type=float,
                        help='Fixed polar-angle grid ceiling; default uses the observed maximum')
    args = parser.parse_args(argv)
    result_dir, manifest, backend = load_lut_result(args.lut_results)
    exp_config_path = args.exp_config.expanduser().resolve()
    exp_config = load_experiment_config(exp_config_path)
    if 'norm_diff' not in exp_config:
        raise ValueError(f'NormDiff settings are missing from {exp_config_path}')
    evaluation = exp_config.get('norm_diff_evaluation', {})
    if args.color_input is None:
        args.color_input = evaluation.get('color_input', 'observed')
    if args.validate_lut_noise is None:
        args.validate_lut_noise = evaluation.get('validate_lut_noise', False)
    if args.validate_lut_noise and args.color_input != 'observed':
        parser.error('--validate-lut-noise requires --color-input observed')

    samples_path = args.samples.expanduser().resolve() if args.samples else result_dir / 'normal_samples.npz'
    config_path = args.config.expanduser().resolve() if args.config else result_dir / 'manifest.json'
    output = args.out.expanduser().resolve() if args.out else result_dir / 'evaluation' / 'norm_diff' / 'report.json'
    if not output.is_relative_to(result_dir / 'evaluation'):
        raise ValueError(f'NormDiff output must be under {result_dir / "evaluation"}: {output}')
    if output.exists():
        raise ValueError(f'Refusing to overwrite existing NormDiff report: {output}')
    config = (load_config(config_path) if args.config else
              validate_config(manifest['config'], config_path=config_path))
    if canonical_backend(config.get('backend', 'analytic')) != backend:
        raise ValueError(f'{config_path} does not match {result_dir} backend {backend}')
    metric = resolve_metric_settings(exp_config, **{
        key: getattr(args, key) for key in ('noise_fraction', 'noise_floor', 'theta_weight',
                                           'phi_weight', 'theta_bins', 'phi_bins',
                                           'anchors_per_target', 'theta_ceiling_degrees')})
    noise = {key: metric[key] for key in ('noise_fraction', 'noise_floor')}
    with np.load(samples_path, allow_pickle=False) as archive:
        required = {'color_residual_samples', 'slope_samples', 'image_index'}
        if not required.issubset(archive.files):
            raise ValueError(f'{samples_path} is missing {sorted(required - set(archive.files))}')
        residual = np.asarray(archive['color_residual_samples'], np.float64)
        slopes = np.asarray(archive['slope_samples'], np.float64)
        image_index = np.asarray(archive['image_index'])
        image_keys = archive['image_paths'] if 'image_paths' in archive else None
        metadata = json.loads(str(archive['metadata'])) if 'metadata' in archive else {}
        material_index = (np.asarray(archive['material_grid_index'])
                          if 'material_grid_index' in archive else None)
        archived_reference = (np.asarray(archive['reference_grid_color'], np.float64)
                              if 'reference_grid_color' in archive else None)
        archived_valid = (np.asarray(archive['reference_grid_valid'], bool)
                          if 'reference_grid_valid' in archive else None)

    signature = _config_signature(config)
    if metadata.get('run_signature') != manifest.get('run_signature'):
        raise ValueError(f'{samples_path} does not belong to {result_dir} run_signature')
    source_signature = metadata.get('simulation_config_signature', metadata.get('run_signature'))
    if source_signature and source_signature != signature:
        raise ValueError('The sample archive was generated by a different simulation configuration, '
                         'light geometry model, or observation model')
    if canonical_backend(metadata.get('backend', 'analytic')) != backend:
        raise ValueError('Sample backend and simulation config backend disagree')

    training_lut = None
    held_out = None
    if args.validate_lut_noise:
        from utils.normal_mapping import NormalCalibration
        training_lut = NormalCalibration.load(samples_path.with_name('normal_lut_train.npz'))
        if metadata.get('run_signature') and training_lut.metadata.get('run_signature') \
                != metadata['run_signature']:
            raise ValueError('The training LUT does not match the sample archive')
        validation_keys = training_lut.metadata.get('validation_images', [])
        if image_keys is None or not validation_keys:
            raise ValueError('LUT noise validation requires saved held-out image keys')
        if set(validation_keys) & set(training_lut.metadata.get('training_images', [])):
            raise ValueError('Training and validation poses must be disjoint')
        held_out = np.isin(image_keys[image_index], validation_keys)
        if not held_out.any():
            raise ValueError('No held-out validation samples were found')

    print(f'[norm-diff] per-channel RGB half-width: +/-{100 * noise["noise_fraction"]:g}% '
          f'(noise_floor={noise["noise_floor"]:g})', flush=True)
    if args.color_input == 'observed':
        if material_index is None:
            raise ValueError('--color-input observed requires material_grid_index in the sample archive')
        if archived_reference is not None:
            grid = build_material_grid(config)
            if archived_reference.shape != (*grid.shape, 3) or archived_valid is None \
                    or archived_valid.shape != grid.shape or material_index.shape != (len(residual), 2) \
                    or np.any(material_index < 0) \
                    or np.any(material_index[:, 0] >= grid.shape[0]) \
                    or np.any(material_index[:, 1] >= grid.shape[1]) \
                    or not np.all(archived_valid[material_index[:, 0], material_index[:, 1]]):
                raise ValueError('Archived reference grid does not match sample material indices')
            colors = residual + archived_reference[material_index[:, 0], material_index[:, 1]]
            print('[norm-diff] restoring observed RGB from archived zero-depth reference', flush=True)
        else:
            if config.get('backend', 'analytic') != 'analytic':
                raise ValueError('MuJoCo samples require an archived reference grid')
            print('[norm-diff] rendering zero-depth reference to reconstruct observed linear RGB',
                  flush=True)
            colors = _observed_linear_rgb(config, residual, material_index)
        color_definition = 'observed RGB = signed residual + matched zero-depth reference'
    else:
        colors = residual
        color_definition = 'signed linear-RGB residual (explicit extension, not paper raw RGB)'

    report = compute_norm_diff(
        colors, slopes, image_index, image_keys=image_keys,
        **metric, progress=True)
    report.update(source_samples=str(samples_path), source_config=str(config_path),
                  source_exp_config=str(exp_config_path),
                  backend=backend,
                  observation_model_id=metadata.get('observation_model_id'),
                  light_geometry_model=metadata.get('light_geometry_model', LIGHT_GEOMETRY_MODEL),
                  simulation_config_signature=signature, color_input=args.color_input,
                  color_definition=color_definition, lighting=config.get('lights'))
    if args.validate_lut_noise:
        print('[norm-diff] validating the training LUT on held-out poses with configured RGB noise',
              flush=True)
        validation = evaluate_lut_noise(colors[held_out], residual[held_out], slopes[held_out],
                                        training_lut, **noise)
        validation.update(pose_count=int(len(np.unique(image_index[held_out]))),
                          training_lut=str(samples_path.with_name('normal_lut_train.npz')))
        report['lut_noise_validation'] = validation
        print(f'LUT RMS: {validation["baseline_rms_degrees"]:.6f} deg baseline -> '
              f'{validation["perturbed_rms_degrees"]:.6f} deg with RGB noise')
    _write_json(output, report)
    print(f'NormDiff: {report["norm_diff"]:.6f} deg '
          f'(mean confusion {report["mean_confusion_degrees"]:.6f} deg)')
    print(f'poses={report["pose_count"]}, samples={report["sample_count"]}, '
          f'targets={report["target_count"]}, anchors={report["anchor_count"]}')
    print(f'report: {output}')


if __name__ == '__main__':
    main()
