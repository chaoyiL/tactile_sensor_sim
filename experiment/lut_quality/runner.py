"""Evaluate existing calibration archives without importing or running a renderer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from experiment.config import load_experiment_config, resolve_lut_quality_settings
from utils.backend_layout import CONFIG_DIR, canonical_backend
from utils.lut_results import load_lut_result
from utils.normal_mapping import NormalCalibration
from .metric import (BOUNDARY_EPS, COVERAGE_BACKEND, METRIC_VERSION, bin_samples,
                     evaluate_holdout, evaluate_quality, inside_disc, split_samples)


def load_inputs(samples_path: Path, lut_path: Path):
    """Read raw pairs, preserve every row, and reject incompatible declared provenance."""
    lut = NormalCalibration.load(lut_path)
    with np.load(samples_path, allow_pickle=False) as data:
        required = {'color_residual_samples', 'slope_samples', 'image_index', 'image_paths'}
        if not required.issubset(data.files):
            raise ValueError(f'{samples_path} missing {sorted(required - set(data.files))}')
        colors = np.asarray(data['color_residual_samples'], np.float64)
        slopes = np.asarray(data['slope_samples'], np.float64)
        index = np.asarray(data['image_index'])
        keys = np.asarray(data['image_paths']).astype(str)
        metadata = json.loads(str(data['metadata'])) if 'metadata' in data else {}
        declarations = {key: str(data[key]) for key in (
            'color_space', 'residual_domain', 'image_encoding', 'coordinate_chart_id',
            'observation_model_id', 'backend') if key in data}
    if colors.ndim != 2 or colors.shape[1] != 3 or not len(colors) \
            or slopes.shape != (len(colors), 2) or index.shape != (len(colors),) \
            or index.dtype.kind not in 'iu' or keys.ndim != 1 or len(np.unique(keys)) != len(keys) \
            or np.any(index < 0) or np.any(index >= len(keys)) \
            or not np.isfinite(colors).all() or not np.isfinite(slopes).all():
        raise ValueError(f'{samples_path}: expected finite raw Nx3 residuals, Nx2 slopes and valid pose indices')
    if not isinstance(metadata, dict):
        raise ValueError(f'{samples_path}: metadata must be a JSON object')
    if declarations.get('color_space', 'linear_rgb') != 'linear_rgb':
        raise ValueError(f'{samples_path}: expected signed linear-RGB residuals, not display RGB')
    warnings = []
    for key in ('run_signature', 'simulation_config_signature', 'light_geometry_model',
                'coordinate_chart_id', 'color_space', 'residual_domain', 'image_encoding',
                'observation_model_id', 'backend'):
        source = metadata.get(key, declarations.get(key))
        target = lut.metadata.get(key)
        if source is not None and target is not None and source != target:
            raise ValueError(f'{samples_path} and {lut_path} disagree on {key}')
    if not metadata.get('run_signature') or not lut.metadata.get('run_signature'):
        warnings.append('Missing run_signature: common source provenance could not be verified.')
    return colors, slopes, index.astype(np.int64), keys, metadata, lut, warnings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def top_bin_examples(colors, slopes, image_index, keys, lut, arrays, settings, source_rows):
    """Keep diverse actual raw rows, including incompatible/minority outcomes."""
    t = settings['taus'].index(settings['detail_tau'])
    ambiguity = arrays['incompatibility'][t]
    ranking = np.lexsort((-arrays['sample_count'], -ambiguity))[:settings['top_bins']]
    ids = bin_samples(colors, lut)
    order = np.argsort(ids, kind='stable')
    rows = []
    for b in ranking:
        target = arrays['bin_ids'][b]
        left = np.searchsorted(ids[order], target, 'left')
        right = np.searchsorted(ids[order], target, 'right')
        selected = order[left:right]
        center = arrays['optimal_center'][t, b]
        distances = np.linalg.norm(slopes[selected] - center, axis=1)
        near = selected[np.argsort(distances, kind='stable')]
        # Alternate near/far examples; the plot and metric retain ALL rows.
        examples = []
        for i in range(min(settings['examples_per_bin'], len(near))):
            row = int(near[i // 2] if i % 2 == 0 else near[-1 - i // 2])
            examples.append(dict(archive_row=int(source_rows[row]), image_key=str(keys[image_index[row]]),
                                 residual=colors[row].tolist(), slopes=slopes[row].tolist(),
                                 covered_by_optimal_disc=bool(inside_disc(slopes[row], center, settings['detail_tau']))))
        rows.append(dict(bin_id=int(target), rgb_node=arrays['coordinates'][b].tolist(),
                         sample_count=int(arrays['sample_count'][b]),
                         pose_count=int(arrays['pose_count'][b]),
                         incompatibility=float(ambiguity[b]),
                         node_error=float(arrays['node_error'][t, b]),
                         optimal_center=center.tolist(), node_output=lut.slopes.reshape(-1, 2)[target].tolist(),
                         examples=examples))
    return dict(tau=settings['detail_tau'], row_index_domain='source normal_samples.npz', bins=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Raw-sample LUT single-value incompatibility and runtime quality')
    parser.add_argument('--lut-results', type=Path, required=True,
                        help='Calibration directory containing samples and a training LUT')
    parser.add_argument('--exp-config', type=Path, default=CONFIG_DIR / 'lut_quality.json')
    parser.add_argument('--samples', type=Path, help='Existing normal_samples.npz (no rendering)')
    parser.add_argument('--lut', type=Path, help='Saved normal_lut*.npz; default: training LUT with pose holdout')
    parser.add_argument('--out', type=Path, help='New or empty report directory; existing results are never overwritten')
    parser.add_argument('--taus', nargs='+', type=float, help='At least two Euclidean (p,q) tolerances, NOT degrees')
    parser.add_argument('--detail-tau', type=float, help='One of taus, for bin examples and scatter plots')
    parser.add_argument('--weighting', choices=('sample', 'pose_equal'))
    parser.add_argument('--split', choices=('auto', 'all'), help='auto preserves saved training-only LUT holdout')
    parser.add_argument('--top-bins', type=int)
    parser.add_argument('--examples-per-bin', type=int)
    parser.add_argument('--dry-run', action='store_true', help='Check inputs/split and show workload; no evaluation or output writes')
    args = parser.parse_args(argv)
    result_dir, calibration_manifest, backend = load_lut_result(
        args.lut_results, require_published=False)
    config_path = args.exp_config.expanduser().resolve()
    config = load_experiment_config(config_path)
    if 'lut_quality' not in config:
        raise ValueError(f'LUT quality settings are missing from {config_path}')
    settings = resolve_lut_quality_settings(config, **{
        key: getattr(args, key) for key in ('taus', 'detail_tau', 'weighting', 'split',
                                           'top_bins', 'examples_per_bin')})
    paths = {
        'samples': args.samples.expanduser().resolve() if args.samples else result_dir / 'normal_samples.npz',
        'lut': args.lut.expanduser().resolve() if args.lut else result_dir / 'normal_lut_train.npz',
        'output_dir': args.out.expanduser().resolve() if args.out else result_dir / 'evaluation' / 'lut_quality',
    }
    output = paths['output_dir']
    if (result_dir / 'manifest.json').is_file() and not output.is_relative_to(result_dir / 'evaluation'):
        raise ValueError(f'LUT quality output must be under {result_dir / "evaluation"}: {output}')
    if not args.dry_run and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Refusing to overwrite nonempty output: {output}; choose --out with a new directory')
    colors, slopes, index, keys, metadata, lut, warnings = load_inputs(paths['samples'], paths['lut'])
    run_signature = calibration_manifest.get('run_signature')
    if run_signature and (metadata.get('run_signature') != run_signature
                          or lut.metadata.get('run_signature') != run_signature):
        raise ValueError(f'{paths["samples"]} does not belong to {result_dir} run_signature')
    if canonical_backend(metadata.get('backend', backend)) != backend:
        raise ValueError(f'{paths["samples"]} does not match {result_dir} backend {backend}')
    reference, held_out, split_description = split_samples(index, keys, lut, settings['split'])
    work_ids, work_counts = np.unique(bin_samples(colors[reference], lut), return_counts=True)
    workload = dict(source_sample_count=len(colors), reference_sample_count=int(reference.sum()),
                    held_out_sample_count=int(held_out.sum()), reference_bin_count=len(work_ids),
                    largest_bin=int(work_counts.max()),
                    squared_bin_count_sum=int(np.sum(work_counts.astype(np.int64)**2)),
                    lut_size=lut.size, split=split_description, backend=COVERAGE_BACKEND,
                    settings=settings, paths={key: str(value) for key, value in paths.items()})
    print(json.dumps(workload, indent=2), flush=True)
    if args.dry_run:
        return workload
    # Check plotting dependency before the expensive exact coverage calculation.
    from .plots import save_plots
    output.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), Path(__file__).with_name('metric.py'),
                    Path(__file__).with_name('plots.py'), Path(__file__).parents[1] / 'config.py',
                    Path(__file__).parents[2] / 'utils' / 'normal_mapping.py']
    manifest = dict(format_version=1, status='running', metric_version=METRIC_VERSION,
                    calibration_status=calibration_manifest.get('status', 'unknown'),
                    source_exp_config=str(config_path), settings=settings, workload=workload,
                    inputs={key: dict(path=str(paths[key]), sha256=sha256(paths[key])) for key in ('samples', 'lut')},
                    implementation_sha256={str(path.relative_to(Path(__file__).parents[2])): sha256(path)
                                           for path in source_files},
                    sample_metadata=metadata, lut_metadata=lut.metadata,
                    environment=dict(python=platform.python_version(), numpy=np.__version__,
                                     coverage_backend=COVERAGE_BACKEND,
                                     coverage_dtype='float64', runtime_dtype='float32'))
    write_json(output / 'manifest.json', manifest)
    started = time.perf_counter()
    try:
        print('[lut-quality] evaluating all reference samples; no rerendering or subsampling', flush=True)
        summary, arrays = evaluate_quality(
            colors[reference], slopes[reference], index[reference], lut, settings['taus'],
            weighting=settings['weighting'], progress=lambda done, total: print(
                f'[lut-quality] bins {done}/{total}', flush=True))
        holdout = (evaluate_holdout(colors[held_out], slopes[held_out], index[held_out], lut,
                                   settings['taus'], arrays, weighting=settings['weighting'])
                   if held_out.any() else None)
        if not held_out.any():
            warnings.append('No independent holdout: these empirical rates are in-sample, not generalization guarantees.')
        warnings.extend([
            'A and node_error use a fixed nearest-node output model; runtime_error uses actual trilinear lookup.',
            'runtime_error can be below A; do not interpret their difference as excess error.',
            'Singleton/sparsely sampled bins do not establish absence of ambiguity.',
            'Configured taus are illustrative slope tolerances, not a calibrated pass/fail threshold.',
        ])
        report = dict(metric_version=METRIC_VERSION, better='lower',
                      calibration_status=calibration_manifest.get('status', 'unknown'),
                      tolerance_domain='Euclidean distance in dimensionless slopes (p,q)',
                      boundary_tolerance=f'tau>0: {BOUNDARY_EPS:g} * max(1,tau); tau=0: exact equality',
                      binning='nearest node: rint(clip((signed residual - color_min)/span)* (size-1)), RGB order',
                      weighting=settings['weighting'], split=split_description,
                      lut_status=lut.metadata.get('validation_status', 'unknown'),
                      reference=summary, holdout=holdout, warnings=warnings,
                      elapsed_seconds=time.perf_counter() - started)
        examples = top_bin_examples(colors[reference], slopes[reference], index[reference], keys,
                                    lut, arrays, settings, np.flatnonzero(reference))
        temporary = output / 'per_bin.npz.tmp'
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(output / 'per_bin.npz')
        write_json(output / 'top_bins.json', examples)
        write_json(output / 'quality.json', report)
        with (output / 'curves.csv').open('w', encoding='utf-8', newline='') as stream:
            rows = []
            for t, row in enumerate(summary['curves']):
                combined = dict(row)
                if holdout:
                    combined.update({f'holdout_{key}': value for key, value in holdout['curves'][t].items() if key != 'tau'})
                rows.append(combined)
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_plots(output, report, arrays, examples, colors[reference], slopes[reference], lut)
        manifest.update(status='completed', elapsed_seconds=time.perf_counter() - started,
                        artifacts=sorted(path.name for path in output.iterdir() if path.name != 'manifest.json'))
        write_json(output / 'manifest.json', manifest)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'manifest.json', manifest)
        raise
    for row in summary['curves']:
        print(f'tau={row["tau"]:g}: A={row["incompatibility"]:.6f}, '
              f'E_node={row["node_error"]:.6f}, E_runtime={row["runtime_error"]:.6f}')
    print(f'report: {output / "quality.json"}', flush=True)
    return report
