"""Re-render complete calibration scans while changing surface/strip length together.

Run with python -m experiment.length_normdiff.
Only surface.size_mm[0] and top/bottom length_mm are changed in each simulation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from experiment.config import (PROJECT_ROOT, experiment_output,
                               load_experiment_config, resolve_metric_settings)
from utils.backend_layout import CONFIG_DIR

ROOT = PROJECT_ROOT

from utils.calibration_samples import (build_material_grid, compact_frame_samples,
                                       render_reference, sample_pose)
from utils.calibration_scan import ScanPose, build_scan_poses
from utils.config import validate_config
from utils.geometry import LIGHT_GEOMETRY_MODEL, strip_samples
from experiment.norm_diff.metric import compute_norm_diff


def length_config(base: dict, length: float) -> dict:
    """Preserve the original strip-to-surface length differences, without mutation."""
    if not math.isfinite(length) or length <= 0:
        raise ValueError('Surface length must be finite and positive')
    config = copy.deepcopy(base)
    delta = length - float(base['surface']['size_mm'][0])
    config['surface']['size_mm'][0] = float(length)
    sides = set()
    for light in config['lights']:
        if light['side'] in ('top', 'bottom'):
            light['length_mm'] = float(light['length_mm']) + delta
            sides.add(light['side'])
    if sides != {'top', 'bottom'}:
        raise ValueError('The experiment requires both top and bottom strips')
    validate_config(config)
    return config


def default_lengths(base: dict) -> list[float]:
    """Eight points including the square and original rectangle endpoints."""
    start = float(base['surface']['size_mm'][1])
    stop = float(base['surface']['size_mm'][0])
    if start == stop:
        return [start]
    direction = 1 if stop > start else -1
    values = [round(start + direction * 5 * i, 10)
              for i in range(int(abs(stop - start) // 5) + 1)]
    if not math.isclose(values[-1], stop, abs_tol=1e-10):
        values.append(stop)
    return sorted(set(values))


def normal_ceiling(poses: list[ScanPose], radius: float) -> float:
    # A sphere cap's largest polar angle is acos((radius-depth)/radius).
    # A single physical bound gives every length the identical normal-target grid.
    return math.degrees(math.acos(1 - max(p.depth_mm for p in poses) / radius))


def center_poses(poses: list[ScanPose]) -> list[ScanPose]:
    depths = {p.depth_index: p for p in poses}
    return [ScanPose(index, f'center_{index:02d}', p.depth_index, -1, -1,
                     (0., 0.), p.depth_mm, p.contact_radius_mm, (0., 0.))
            for index, (_, p) in enumerate(sorted(depths.items()))]


def center_irradiance(config: dict) -> list[dict]:
    """Flat-centre raw surface irradiance, same quadrature as the GPU renderer.

    Diagnostic only: no camera/tone mapping; not an alternative NormDiff input.
    On the flat surface, all the light segments are unoccluded.
    """
    rows = []
    normal = np.array([0., 0., float(config['surface']['view_side_z'])])
    for light in config['lights']:
        points, emission_normal = strip_samples(light, config)
        distance2 = np.sum(points * points, axis=1)
        direction = points / np.sqrt(distance2[:, None])
        cos_emit = np.maximum((-direction) @ emission_normal, 0.)
        cos_incident = np.maximum(direction @ normal, 0.)
        exponent = light['cosine_exponent']
        factor = ((exponent + 1) / (2 * np.pi) * cos_emit ** exponent
                  * cos_incident / distance2)
        # Geometry units are mm; convert W/mm^2 to W/m^2.
        irradiance = factor.mean() * np.asarray(light['power_rgb_w']) * 1e6
        rows.append(dict(side=light['side'], rgb_w_m2=irradiance.tolist()))
    return rows


def write_json(path: Path, data: dict):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding='utf-8')
    temporary.replace(path)


def render_scores(config: dict, poses: list[ScanPose], *, metric: dict) -> tuple[dict, dict]:
    """Keep all valid contact samples; stream rendering but not sample subsampling."""
    import torch
    from utils.renderer_gpu import AnalyticGpuContext

    grid = build_material_grid(config)
    centers = center_poses(poses)
    minimum = int(config['calibration']['sampling'].get('min_samples_per_pose', 16))
    with torch.inference_mode():
        context = AnalyticGpuContext(config)
        reference = render_reference(context, grid)
        reference_color = reference.grid_color.detach().cpu().numpy()

        def evaluate(group, label):
            colors, slopes, indices = [], [], []
            last_progress = time.perf_counter()
            for number, pose in enumerate(group, 1):
                frame = sample_pose(context, reference, grid, pose)
                compact = compact_frame_samples(frame, grid)
                count = len(compact['slope_samples'])
                if count < minimum:
                    raise RuntimeError(f'{label} pose {pose.image_key}: {count} valid samples '
                                       f'< {minimum}; refusing a biased partial scan')
                rows, columns = compact['material_grid_index'].T
                # Exactly the saved-sample NormDiff evaluator's observed-RGB path,
                # including archive float32 residuals promoted to float64.
                colors.append(compact['color_residual_samples'].astype(np.float64)
                              + reference_color[rows, columns])
                slopes.append(compact['slope_samples'])
                indices.append(np.full(count, number - 1, np.int32))
                if time.perf_counter() - last_progress >= 10 or number == len(group):
                    print(f'[length={config["surface"]["size_mm"][0]:g}] '
                          f'{label} render {number}/{len(group)}', flush=True)
                    last_progress = time.perf_counter()
                del frame, compact
            report = compute_norm_diff(
                np.concatenate(colors), np.concatenate(slopes), np.concatenate(indices),
                image_keys=[p.image_key for p in group], **metric, progress=True)
            for row, pose in zip(report['poses'], group, strict=True):
                row.update(center_xy_mm=list(pose.center_xy_mm), depth_mm=pose.depth_mm)
            return report

        full = evaluate(poses, 'full scan')
        center = evaluate(centers, 'fixed centre')
        full['renderer_compile_metrics'] = context.compile_metrics()
    # Each length requires different physical light geometry. Release its graphs
    # before constructing another renderer, including any reference cycles.
    del context, reference
    gc.collect()
    torch.cuda.empty_cache()
    return full, center


def summarize(rows: list[dict], output: Path, manifest: dict):
    records = []
    for row in sorted(rows, key=lambda r: r['length_mm']):
        records.append(dict(
            length_mm=row['length_mm'], top_length_mm=row['top_length_mm'],
            bottom_length_mm=row['bottom_length_mm'],
            norm_diff=row['full_scan']['norm_diff'],
            center_norm_diff=row['fixed_center']['norm_diff'],
            mean_theta_range_degrees=row['full_scan']['mean_theta_range_degrees'],
            mean_phi_range_degrees=row['full_scan']['mean_phi_range_degrees'],
            pose_count=row['full_scan']['pose_count'], sample_count=row['full_scan']['sample_count'],
            elapsed_seconds=row['elapsed_seconds']))
    if not records:
        return
    with (output / 'length_normdiff.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(output / 'summary.json', dict(experiment=manifest, points=records))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11}):
        fig, ax = plt.subplots(figsize=(9, 5.7), layout='constrained')
        x = [r['length_mm'] for r in records]
        ax.plot(x, [r['norm_diff'] for r in records], 'o-', color='#176ca4',
                linewidth=2.2, markersize=6, label='Full-surface scan (pose-balanced)')
        ax.plot(x, [r['center_norm_diff'] for r in records], 's--', color='#df7522',
                linewidth=1.8, markersize=5, label='Fixed centre (same depth set)')
        ax.set(xlabel='Surface length (mm)', ylabel='NormDiff (deg; closer to 0 is better)',
               title='Effect of surface length on NormDiff')
        ax.set_xticks(x)
        ax.grid(alpha=.22)
        ax.set_axisbelow(True)
        ax.legend(loc='best', framealpha=.95)
        noise = manifest['noise']
        height = manifest['base_config']['surface']['size_mm'][1]
        fig.suptitle(f'Width = {height:g} mm | RGB half-width = '
                     f'+/-{noise["noise_fraction"] * 100:g}% | fixed strip total power',
                     fontsize=10, color='#555555')
        fig.savefig(output / 'length_normdiff.png', dpi=200)
        fig.savefig(output / 'length_normdiff.svg')
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-config', '--config', dest='exp_config', type=Path,
                        default=CONFIG_DIR / 'length_normdiff.json',
                        help='Independent experiment JSON; not config.json')
    parser.add_argument('--lengths', type=float, nargs='+',
                        help='Override exp_config.length_normdiff.lengths_mm')
    parser.add_argument('--out', type=Path, help='Override the experiment output directory')
    parser.add_argument('--backend', choices=('analytic', 'zemax', 'xense'),
                        help='Check the backend selected by the experiment snapshot')
    parser.add_argument('--noise-fraction', type=float, help='Override exp_config RGB relative half-width')
    parser.add_argument('--noise-floor', type=float, help='Override exp_config noise floor')
    parser.add_argument('--theta-weight', type=float)
    parser.add_argument('--phi-weight', type=float)
    parser.add_argument('--theta-bins', type=int)
    parser.add_argument('--phi-bins', type=int)
    parser.add_argument('--anchors-per-target', type=int)
    parser.add_argument('--theta-ceiling-degrees', type=float,
                        help='Fixed polar-angle ceiling; null in config uses the physical cap bound')
    parser.add_argument('--dry-run', action='store_true', help='Print the scan plan without rendering/writing')
    parser.add_argument('--diagnostics-only', action='store_true',
                        help='Analyze a completed scan manifest without rendering')
    parser.add_argument('--manifest', type=Path, help='Completed scan manifest for --diagnostics-only')
    args = parser.parse_args(argv)
    if args.manifest is not None and not args.diagnostics_only:
        parser.error('--manifest requires --diagnostics-only')
    if args.dry_run and args.diagnostics_only:
        parser.error('--dry-run previews a scan; it cannot be combined with --diagnostics-only')
    config_path = args.exp_config.expanduser().resolve()
    exp_config = load_experiment_config(config_path)
    selected_backend = exp_config['simulation'].get('backend', 'analytic')
    if args.backend is not None and args.backend != selected_backend:
        parser.error(f'--backend {args.backend} disagrees with snapshot {selected_backend}')
    if selected_backend in ('mujoco', 'xense', 'zemax'):
        parser.error(f'Length NormDiff sweep supports only analytic backend; '
                     f'snapshot selects {selected_backend}')
    output = args.out.expanduser().resolve() if args.out else experiment_output(exp_config, config_path)
    if args.diagnostics_only:
        from .diagnostics import run_diagnostics
        source = args.manifest or experiment_output(exp_config, config_path) / 'manifest.json'
        run_diagnostics(source, output_dir=(args.out if args.out else None))
        return
    base = exp_config['simulation']
    lengths = sorted(set(args.lengths if args.lengths is not None else
                         exp_config['length_normdiff']['lengths_mm']))
    metric = resolve_metric_settings(exp_config, **{
        key: getattr(args, key) for key in ('noise_fraction', 'noise_floor', 'theta_weight',
                                           'phi_weight', 'theta_bins', 'phi_bins',
                                           'anchors_per_target', 'theta_ceiling_degrees')})
    experiments = [(length, length_config(base, length)) for length in lengths]
    scans = [build_scan_poses(config) for _, config in experiments]
    physical_ceiling = normal_ceiling(scans[0], float(base['indentation']['radius_mm']))
    ceiling = metric['theta_ceiling_degrees']
    if ceiling is None:
        ceiling = physical_ceiling
    if ceiling + 1e-10 < physical_ceiling:
        raise ValueError('NormDiff theta ceiling must cover the configured sphere-cap angle range')
    metric['theta_ceiling_degrees'] = ceiling
    noise = {key: metric[key] for key in ('noise_fraction', 'noise_floor')}
    manifest = dict(format_version=1, base_config=base, lengths_mm=lengths, noise=noise,
                    source_exp_config=str(config_path), norm_diff_settings=metric,
                    light_geometry_model=LIGHT_GEOMETRY_MODEL,
                    theta_ceiling_degrees=ceiling,
                    color_input='observed: float64(archived float32 residual) + reference',
                    sampling='all configured scan poses and every valid contact sample',
                    power_policy='unchanged total power_rgb_w for every strip',
                    code_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__), ROOT / 'experiment/norm_diff/metric.py',
                                              ROOT / 'experiment/config.py',
                                              ROOT / 'utils/calibration_samples.py',
                                              ROOT / 'utils/renderer_gpu.py',
                                              ROOT / 'utils/geometry.py')})
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest['signature'] = signature
    for (length, config), poses in zip(experiments, scans, strict=True):
        strips = {p['side']: p['length_mm'] for p in config['lights']}
        print(f'length={length:g}, top={strips["top"]:g}, bottom={strips["bottom"]:g} mm; '
              f'{len(poses)} scan poses + {len(center_poses(poses))} centre poses', flush=True)
    print(f'RGB half-width=+/-{noise["noise_fraction"] * 100:g}%, '
          f'common theta ceiling={ceiling:.8f} deg', flush=True)
    print(f'experiment config: {config_path}\noutput: {output}', flush=True)
    if args.dry_run:
        return
    if output.exists() and any(output.iterdir()) and not (output / 'manifest.json').exists():
        raise ValueError('Output is not empty and has no experiment manifest; choose a new --out')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['signature'] != signature:
            raise ValueError('Output contains a different experiment; choose a new --out directory')
    else:
        write_json(manifest_path, manifest)
    rows = []
    for (length, config), poses in zip(experiments, scans, strict=True):
        point_path = output / f'length_{length:.10g}.json'
        if point_path.exists():
            result = json.loads(point_path.read_text(encoding='utf-8'))
            if result.get('signature') != signature:
                raise ValueError(f'Checkpoint signature mismatch: {point_path}')
            print(f'[length={length:g}] resuming completed point', flush=True)
        else:
            started = time.perf_counter()
            full, center = render_scores(config, poses, metric=metric)
            strips = {p['side']: p['length_mm'] for p in config['lights']}
            result = dict(length_mm=length, top_length_mm=strips['top'],
                          bottom_length_mm=strips['bottom'], signature=signature, config=config,
                          full_scan=full, fixed_center=center,
                          flat_center_irradiance_by_strip=center_irradiance(config),
                          elapsed_seconds=time.perf_counter() - started)
            write_json(point_path, result)
        rows.append(result)
        summarize(rows, output, manifest)
        print(f'[length={length:g}] NormDiff={result["full_scan"]["norm_diff"]:.6f}, '
              f'centre={result["fixed_center"]["norm_diff"]:.6f}', flush=True)
    print(f'Figure: {output / "length_normdiff.png"}', flush=True)


if __name__ == '__main__':
    main()
