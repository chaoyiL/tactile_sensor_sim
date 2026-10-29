"""Compare the v2 finite-slope inverse with archived endpoint LUTs.

Only saved test RGB/material identities enter inference. Test geometry and slope
labels are consumed by the common metric implementation after predictions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import numpy as np

from experiment.config import resolve_curve_lut_nonlinear_settings
from experiment.curve_lut_transfer.metrics import aggregate, paired_bootstrap
from experiment.curve_lut_transfer.storage import Checkpoints, digest, json_write, npz_write, signature
from utils.curve_calibration import reference_grid
from utils.normal_mapping import NormalCalibration, lookup_slopes_raw
from . import runner
from .transfer import pose_config


VERSION = 'finite_slope_comparison_v1'
BASELINES = ('frozen_flat_lut', 'endpoint_migrated_lut')


def load_endpoint_luts(prepared: dict, root: Path) -> tuple[dict, dict, dict]:
    """Read compatible fitted endpoint LUTs without changing the original run."""
    root = Path(root).expanduser().resolve()
    manifest = runner._load_json(root / 'manifest.json')
    if manifest.get('status') != 'completed':
        raise ValueError(f'Endpoint comparison requires a completed run: {root}')
    old = manifest.get('identity', {})
    for key in runner.RESUME_INPUT_FIELDS:
        before, after = old.get(key), prepared['identity'].get(key)
        if key == 'settings':
            before = runner._endpoint_resume_settings(before or {})
            after = runner._endpoint_resume_settings(after or {})
        if before != after:
            raise ValueError(f'Endpoint comparison input differs ({key}): {root}')
    luts, statuses = {}, {}
    consumed = {'manifest.json': digest(root / 'manifest.json')}
    runner._verify_checkpoint_file(manifest, root / 'metrics.json')
    consumed['metrics.json'] = digest(root / 'metrics.json')
    for name in prepared['conditions']:
        info = manifest.get('calibration', {}).get(name, {})
        filename = info.get('lut')
        if info.get('status') != 'passed' or filename != 'normal_lut.npz':
            raise ValueError(f'Comparison requires a published, passed endpoint LUT for {name}: {root}')
        path = root / 'calibration' / name / filename
        runner._verify_checkpoint_file(manifest, path)
        luts[name] = NormalCalibration.load(path)
        metadata = luts[name].metadata
        if (metadata.get('calibration_kind') != 'plane_to_curve_endpoint_H_pt_transfer'
                or metadata.get('parent_run_signature') != manifest.get('run_signature')
                or metadata.get('coordinate_chart_id') != 'cylinder_reference_arclength_' + name):
            raise ValueError(f'Endpoint LUT metadata disagrees with the run/condition: {path}')
        statuses[name] = info
        consumed[str(path.relative_to(root))] = digest(path)
    runtime = root / 'runtime.json'
    if runtime.is_file():
        runner._verify_checkpoint_file(manifest, runtime)
        consumed['runtime.json'] = digest(runtime)
    return luts, statuses, dict(root=str(root), files=consumed)


def _datasets(prepared: dict) -> tuple[dict, list[np.ndarray]]:
    data = {}
    grid = reference_grid(prepared['base']).xy_mm.reshape(-1, 2)
    for name in prepared['conditions']:
        data[name] = []
        for pose in prepared['test_poses']:
            key = f'{name}/{pose["pose_id"]}'
            sample, meta = runner._historical_sample(
                prepared, prepared['test_sample_paths'][key], f'{name}/test/{pose["pose_id"]}')
            if meta.get('pose') is not None:
                runner._assert_same_pose_geometry([pose], [meta['pose']], str(prepared['test_sample_paths'][key]))
            # A material ID must select the matching reference-field coordinate.
            indices = sample['grid_index']
            if np.any(indices < 0) or np.any(indices >= len(grid)) or not np.allclose(
                    sample['material_coordinates'], grid[indices], rtol=0, atol=1e-10):
                raise ValueError(f'Material coordinates disagree with grid_index: {key}')
            data[name].append(sample)
    common = []
    for i in range(len(prepared['test_poses'])):
        indices = None
        for rows in data.values():
            indices = rows[i]['grid_index'] if indices is None else np.intersect1d(indices, rows[i]['grid_index'])
        common.append(indices)
    return data, common


def _percentiles(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return dict(count=len(values), mean_ms=float(a.mean()),
                p50_ms=float(np.quantile(a, .5)), p95_ms=float(np.quantile(a, .95)),
                p99_ms=float(np.quantile(a, .99)), total_seconds=float(a.sum() / 1000.))


def _synchronize(torch, device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _summary_diagnostics(diagnostics: dict, torch) -> dict:
    result = {}
    for key, value in diagnostics.items():
        if not torch.is_tensor(value) or not value.numel():
            continue
        array = value.detach().cpu().numpy()
        if array.dtype == bool:
            result[key] = dict(count=int(np.count_nonzero(array)), total=int(array.size),
                               fraction=float(np.mean(array)))
        elif np.issubdtype(array.dtype, np.number):
            finite = array[np.isfinite(array)]
            result[key] = dict(finite_count=int(len(finite)), total=int(array.size),
                               mean=float(finite.mean()) if len(finite) else None,
                               max=float(finite.max()) if len(finite) else None)
    return result


def _archived_endpoint_cost(identity: dict) -> dict:
    path = Path(identity['root']) / 'runtime.json'
    if not path.is_file():
        return dict(status='not_available')
    runtime = runner._load_json(path)
    entries = runtime.get('timing', {}).get('entries', [])
    return dict(status='historical_measurement_not_rerun', source=str(path),
                source_sha256=digest(path),
                entries=[e for e in entries if e.get('stage') in ('endpoint_migration', 'endpoint_fit')],
                note='Includes the archived sample-generation, I/O, fit and validation work; not a pure GPU kernel time.')


def _plot(root: Path, methods: tuple[str, ...], conditions: list[str], metrics: dict, speed: dict) -> None:
    import matplotlib.pyplot as plt

    labels = {'frozen_flat_lut': 'Frozen LUT', 'endpoint_migrated_lut': 'Endpoint LUT',
              'nonlinear_closed_or_gn3': 'Closed / fixed GN3 fallback',
              'nonlinear_active_gn3': 'Activity-updated GN3'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    values, latency = [], []
    for method in methods:
        row = [metrics[f'{method}/{name}/common']['rmse'] for name in conditions]
        ms = [speed[f'{method}/{name}']['host_pipeline']['p50_ms'] for name in conditions]
        values.append(row)
        latency.append(ms)
        label = labels.get(method, method.replace('nonlinear_gn', 'GN '))
        axes[0].plot(conditions, row, marker='o', label=label)
        axes[1].plot(conditions, ms, marker='o', label=label)
    axes[0].set_ylabel('Slope-vector RMSE (unitless; lower is better)')
    pose_count = metrics[f'{methods[0]}/{conditions[0]}/common'].get('pose_count')
    axes[0].set_title(f'Common material domain; {pose_count} poses / condition')
    axes[1].set_ylabel('Median processing time per contact patch (ms)')
    axes[1].set_title('Host RGB to host slope; geometry update excluded')
    for ax in axes:
        ax.set_xlabel('Reference configuration')
        ax.grid(alpha=.25)
        ax.tick_params(axis='x', rotation=15)
    axes[0].legend(fontsize=8)
    path = root / 'figures'
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'accuracy_speed.png', dpi=160)
    plt.close(fig)
    npz_write(path / 'figure_data.npz', methods=np.asarray(methods), conditions=np.asarray(conditions),
              common_rmse=np.asarray(values), host_p50_ms=np.asarray(latency))


def run_comparison(prepared: dict, endpoint_results: Path, out: Path | None, *,
                   resume: bool = False, dry_run: bool = False) -> Path | None:
    """Run a fixed-parameter archive comparison, with per-pose checkpoints."""
    settings = resolve_curve_lut_nonlinear_settings(prepared['settings'].get('nonlinear', {}))
    endpoint_luts, statuses, endpoint_identity = load_endpoint_luts(prepared, endpoint_results)
    methods = (*BASELINES, *(f'nonlinear_gn{n}' for n in settings['steps']),
               *(('nonlinear_active_gn3',) if settings['active_set_update'] else ()),
               'nonlinear_closed_or_gn3')
    parent = (prepared['root'] / 'evaluation' / 'curve_lut_compensation').resolve()
    root = (out or parent / ('nonlinear_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                            + '_' + uuid.uuid4().hex[:8])).expanduser().resolve()
    if not root.is_relative_to(parent) or root == parent:
        raise ValueError(f'Comparison output must be a run subdirectory of {parent}')
    identity = dict(source={k: prepared['identity'][k] for k in runner.RESUME_INPUT_FIELDS},
                    endpoint=endpoint_identity, nonlinear=settings,
                    methods=list(methods), domains=list(runner.DOMAINS))
    if resume:
        manifest = runner._load_json(root / 'manifest.json')
        if manifest.get('identity') != identity:
            raise ValueError(f'Nonlinear comparison inputs/settings differ: {root}')
        checkpoint = Checkpoints(root, manifest)
        checkpoint.verify()
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError(f'Nonempty comparison output requires compatible --resume: {root}')
        checkpoint = Checkpoints(root, dict(version=VERSION, identity=identity, status='running', completed={},
                                            run_signature=signature(identity)))
    if dry_run:
        print(json.dumps(dict(status='dry-run', mode=VERSION, output=str(root), methods=methods,
                              nonlinear=settings, workload={k: v for k, v in prepared['workload'].items()
                                                            if k != 'source_sample_count_by_pose'},
                              endpoint_luts_reused=str(endpoint_results), cuda_initialized=False,
                              endpoint_refit_count=0, render_count=0), ensure_ascii=False, indent=2))
        return None
    if checkpoint.manifest.get('status') == 'completed':
        print(f'Completed nonlinear comparison verified; no artifacts rewritten: {root}', flush=True)
        return root
    checkpoint.save()
    json_write(root / 'config_snapshot.json', dict(simulation=prepared['base'], experiment=prepared['settings'],
                                                 nonlinear=settings, endpoint=endpoint_identity))
    json_write(root / 'split.json', dict(prepared['split'], independent_test=prepared['test_poses']))
    json_write(root / 'method_semantics.json', dict(
        nonlinear='v2 eq12-15; fixed reference position/visibility/receiving active set, normalized finite slope, Reinhard; direct beta output',
        active_variant='Each GN iteration updates receiver activity and sums all reference-visible lights at X0; reference shadow visibility remains frozen',
        initial_beta='frozen planar LUT queried with current target signed RGB; no target truth or displacement',
        closed='absolute model background + observed signed RGB; safe branch checks, otherwise GN3',
        failures='non-finite inverse output falls back to initial frozen LUT slope; counted, never removed',
        parameters='fixed before archive evaluation; no target-based tuning, no source residual training',
        timing='host pipeline includes CPU LUT initialization, field gathering, transfers and device synchronization; resident excludes these',
        scope='archived planar/cylindrical static material samples; not a general mesh, dynamic sensor or full image benchmark'))
    checkpoint.complete('inputs', [root / n for n in ('config_snapshot.json', 'split.json', 'method_semantics.json')])
    start = time.perf_counter()
    data, common = _datasets(prepared)
    from .nonlinear import (build_response_field, select_response_field,
                            predict_gauss_newton, predict_active_gauss_newton, predict_closed_form)
    from .photometry import PhotometryModel

    models, fields, updates = {}, {}, {}
    grid = reference_grid(prepared['base']).xy_mm.reshape(-1, 2)
    for name, radius in prepared['conditions'].items():
        config = pose_config(prepared['base'], radius)
        model_start = time.perf_counter()
        model = PhotometryModel(config, device=prepared['base'].get('render', {}).get('device', 'cuda:0'))
        torch = model.torch
        _synchronize(torch, model.device)
        model_seconds = time.perf_counter() - model_start
        durations = []
        # Warm CUDA and then measure full-resolution configuration updates.
        for repeat in range(settings['warmup_repeats'] + settings['benchmark_repeats']):
            _synchronize(torch, model.device)
            begin = time.perf_counter()
            field = build_response_field(model, grid, config, prepared['settings']['chunk_size'])
            _synchronize(torch, model.device)
            seconds = time.perf_counter() - begin
            if repeat >= settings['warmup_repeats']:
                durations.append(seconds * 1000.)
        models[name], fields[name] = model, field
        updates[name] = dict(point_count=len(grid), model_initialization_seconds=model_seconds,
                             field_update=_percentiles(durations), execution=model.execution,
                             field_tensor_bytes=sum(v.numel() * v.element_size() for v in field.values() if torch.is_tensor(v)))
        print(f'{name}: {len(grid)}-point nonlinear field ready', flush=True)
    json_write(root / 'field_updates.json', updates)
    checkpoint.complete('field_updates', [root / 'field_updates.json'])
    all_rows, row_sets, summaries, speed, diag_rows = [], {}, {}, {}, []
    reused_predictions = 0
    for name in prepared['conditions']:
        model, field = models[name], fields[name]
        torch = model.torch
        host_times = {m: [] for m in methods}
        resident_times = {m: [] for m in methods if m not in BASELINES}
        pose_predictions = []
        for i, (pose, sample) in enumerate(zip(prepared['test_poses'], data[name], strict=True)):
            key = f'prediction/{name}/{pose["pose_id"]}'
            path = root / 'predictions' / name / (pose['pose_id'] + '.npz')
            if checkpoint.has(key):
                reused_predictions += 1
                with np.load(path, allow_pickle=False) as saved:
                    predictions = {m: saved[m] for m in methods}
                    meta = json.loads(str(saved['metadata']))
            else:
                colors = np.asarray(sample['color_residual_samples'], dtype=np.float64)

                def prepare_device():
                    initial = lookup_slopes_raw(colors, prepared['source_lut'])[0]
                    roi = select_response_field(field, torch.as_tensor(sample['grid_index'], device=model.device))
                    c = torch.as_tensor(colors, dtype=model.radiometry_dtype, device=model.device)
                    beta = torch.as_tensor(initial, dtype=model.radiometry_dtype, device=model.device)
                    return roi, c, beta

                def inverse(method, inputs):
                    roi, c, beta = inputs
                    kwargs = dict(damping=settings['damping'], prior_weight=settings['prior_weight'],
                                  max_step=settings['max_step'])
                    if method == 'nonlinear_closed_or_gn3':
                        return predict_closed_form(roi, c, initial_beta=beta,
                            color_epsilon=settings['color_epsilon'], max_condition=settings['max_condition'],
                            minimum_normal_dot=settings['minimum_normal_dot'], **kwargs)
                    if method == 'nonlinear_active_gn3':
                        return predict_active_gauss_newton(roi, c, beta, steps=3, **kwargs)
                    return predict_gauss_newton(roi, c, beta, steps=int(method[-1]), **kwargs)

                predictions, timing, diagnostics = {}, {}, {}
                for method in methods:
                    host, resident = [], []
                    count = settings['benchmark_repeats']
                    warmup = settings['warmup_repeats'] if i == 0 else 0
                    for repeat in range(warmup + count):
                        _synchronize(torch, model.device)
                        begin = time.perf_counter()
                        if method in BASELINES:
                            lut = endpoint_luts[name] if method == 'endpoint_migrated_lut' else prepared['source_lut']
                            prediction = lookup_slopes_raw(colors, lut)[0]
                            detail = {}
                        else:
                            inputs = prepare_device()
                            output, detail = inverse(method, inputs)
                            prediction = output.detach().cpu().numpy()
                        nonfinite = ~np.isfinite(prediction).all(axis=-1)
                        if np.any(nonfinite):
                            initial = lookup_slopes_raw(colors, prepared['source_lut'])[0]
                            prediction[nonfinite] = initial[nonfinite]
                        _synchronize(torch, model.device)
                        elapsed = (time.perf_counter() - begin) * 1000.
                        if repeat >= warmup:
                            host.append(elapsed)
                    if method not in BASELINES:
                        # Same algorithm and diagnostics, inputs already on GPU.
                        for _ in range(count):
                            _synchronize(torch, model.device)
                            begin = time.perf_counter()
                            output, _ = inverse(method, inputs)
                            _synchronize(torch, model.device)
                            resident.append((time.perf_counter() - begin) * 1000.)
                        diagnostics[method] = _summary_diagnostics(detail, torch)
                    diagnostics.setdefault(method, {})['nonfinite_fallback_count'] = int(nonfinite.sum())
                    predictions[method] = prediction
                    timing[method] = dict(host_pipeline=host, resident_inverse=resident)
                meta = dict(timing=timing, diagnostics=diagnostics, sample_count=len(colors), pose_id=pose['pose_id'])
                npz_write(path, **predictions, grid_index=sample['grid_index'], metadata=json.dumps(meta))
                checkpoint.complete(key, [path], sample_count=len(colors))
            pose_predictions.append(predictions)
            for method in methods:
                host_times[method].extend(meta['timing'][method]['host_pipeline'])
                if method not in BASELINES:
                    resident_times[method].extend(meta['timing'][method]['resident_inverse'])
                diag_rows.append(dict(condition=name, pose_id=pose['pose_id'], method=method,
                                      sample_count=meta['sample_count'], diagnostics=meta['diagnostics'][method]))
            if (i + 1) % 21 == 0:
                print(f'{name}: compared {i + 1}/{len(data[name])} test poses', flush=True)
        for method in methods:
            speed[f'{method}/{name}'] = dict(host_pipeline=_percentiles(host_times[method]),
                sample_count=sum(len(s['grid_index']) for s in data[name]),
                pose_count=len(data[name]), benchmark_repeats=settings['benchmark_repeats'])
            if method not in BASELINES:
                speed[f'{method}/{name}']['resident_inverse'] = _percentiles(resident_times[method])
            for domain in runner.DOMAINS:
                rows, details = [], []
                for i, (pose, sample) in enumerate(zip(prepared['test_poses'], data[name], strict=True)):
                    prediction = pose_predictions[i][method]
                    lut = endpoint_luts[name] if method == 'endpoint_migrated_lut' else prepared['source_lut']
                    meta = dict(method=method, condition=name, domain=domain, pose_id=pose['pose_id'],
                        anchor_id=pose['anchor_id'], u=pose['center_uv'][0], v=pose['center_uv'][1], depth_mm=pose['depth_mm'])
                    row, detail = runner._evaluate_prediction(sample, prediction, np.isfinite(prediction).all(-1),
                        sample['color_residual_samples'], lut, None if domain == 'full' else common[i],
                        method=method, metadata=meta, raw_flat_support_lut=prepared['source_lut'])
                    rows.append(row)
                    if detail:
                        details.append(detail)
                summary = aggregate(rows, details, prepared['settings']['metrics']['slope_tolerances'])
                summary.update(method=method, condition=name, domain=domain)
                teacher_mse = []
                for i, sample in enumerate(data[name]):
                    take = np.arange(len(sample['grid_index'])) if domain == 'full' else np.searchsorted(sample['grid_index'], common[i])
                    area = sample['area_mm2'][take].astype(np.float64)
                    delta = pose_predictions[i][method][take].astype(np.float64) \
                        - pose_predictions[i]['endpoint_migrated_lut'][take]
                    teacher_mse.append(float((area / area.sum()) @ np.sum(delta ** 2, axis=-1)))
                summary['endpoint_prediction_rmse'] = float(np.sqrt(np.mean(teacher_mse)))
                if method not in BASELINES:
                    summary.update(lut_support_applicable=False, out_of_range_fraction=None, original_support_fraction=None)
                    for row in rows:
                        row.update(lut_support_applicable=False, out_of_range_fraction=None, original_support_fraction=None)
                key = f'{method}/{name}/{domain}'
                summaries[key], row_sets[key] = summary, rows
                all_rows.extend(rows)
    comparisons = {}
    for name in prepared['conditions']:
        comparisons[name] = {}
        for method in methods:
            if method == 'endpoint_migrated_lut':
                continue
            comparisons[name][method] = paired_bootstrap(row_sets[f'{method}/{name}/common'],
                row_sets[f'endpoint_migrated_lut/{name}/common'],
                repeats=prepared['settings']['metrics']['bootstrap_repeats'], seed=prepared['settings']['metrics']['seed'])
    archive = runner._load_json(Path(endpoint_identity['root']) / 'metrics.json')
    reproduction, reproduction_checks = {}, {}
    for method in BASELINES:
        for condition in prepared['conditions']:
            for domain in runner.DOMAINS:
                key = f'{method}/{condition}/{domain}'
                actual, expected = summaries[key]['rmse'], archive.get('metrics', {}).get(key, {}).get('rmse')
                reproduction[key] = actual == expected
                matched = actual is not None and expected is not None and bool(np.isclose(actual, expected, rtol=0, atol=1e-12))
                reproduction_checks[key] = dict(matched=matched, absolute_tolerance=1e-12,
                    archived_rmse=expected, reproduced_rmse=actual,
                    absolute_difference=abs(actual - expected) if actual is not None and expected is not None else None)
    reproduced = all(v['matched'] for v in reproduction_checks.values())
    result = dict(version=VERSION, status='completed' if reproduced else 'baseline_mismatch', methods=list(methods), domains=list(runner.DOMAINS),
                  conditions=prepared['conditions'], metrics=summaries, comparisons_against_endpoint=comparisons,
                  baseline_rmse_exact_reproduction=reproduction, nonlinear_settings=settings,
                  baseline_rmse_reproduction=reproduction_checks,
                  endpoint_calibration_status=statuses, input_identity=identity,
                  metric_definition='pose-equal reference-area-weighted slope-vector RMSE; all archived valid samples retained',
                  note='Static archived cylindrical comparison; no new target tuning, rendering or endpoint fitting.')
    json_write(root / 'metrics.json', result)
    json_write(root / 'diagnostics.json', dict(per_pose=diag_rows))
    json_write(root / 'runtime.json', dict(elapsed_seconds=time.perf_counter() - start,
        hardware=dict(torch_version=str(torch.__version__), device=str(model.device),
                      gpu_name=torch.cuda.get_device_name(model.device) if model.device.type == 'cuda' else None),
        run_mode='resume' if resume else 'fresh', benchmark=speed, configuration_update=updates,
        reused_prediction_count=reused_predictions,
        benchmark_origin='mixed_checkpoint_timings_not_current_invocation' if reused_predictions else 'current_invocation_fresh',
        elapsed_scope='after endpoint loading/output setup, through prediction/metrics computation; excludes final reports/plots and CLI preparation',
        archived_endpoint_build=_archived_endpoint_cost(endpoint_identity),
        host_scope='CPU RGB -> CPU frozen-LUT slope initialization (lookup) -> GPU field gather/transfers/inversion -> CPU slopes; synchronization included; LUT loading/disk/metrics excluded',
        resident_scope='Same GPU inverse and diagnostics, preloaded ROI/colors/initial slopes; CPU lookup/gather/H2D/D2H excluded',
        update_scope='Full saved material grid; reference geometry, shadow/receiving activity, B and diagnostic fields; no contact truth',
        sample_count_by_condition={c: sum(len(s['grid_index']) for s in data[c]) for c in data}))
    runner._write_csv(root / 'per_pose.csv', all_rows)
    runner._write_csv(root / 'cross_method.csv', [dict(key=k, **v) for k, v in summaries.items()])
    _plot(root, methods, list(prepared['conditions']), summaries, speed)
    for relative, expected in endpoint_identity['files'].items():
        if digest(Path(endpoint_identity['root']) / relative) != expected:
            raise ValueError(f'Consumed endpoint input changed during comparison: {relative}')
    if not reproduced:
        checkpoint.manifest.update(status='baseline_mismatch')
        checkpoint.save()
        raise ValueError(f'Archived baseline reproduction failed; see {root / "metrics.json"}')
    checkpoint.complete('evaluation', [root / n for n in ('metrics.json', 'runtime.json', 'diagnostics.json', 'per_pose.csv', 'cross_method.csv')]
                         + list((root / 'figures').iterdir()))
    checkpoint.manifest.update(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat())
    checkpoint.save()
    print(f'Completed nonlinear comparison: {root}', flush=True)
    return root
