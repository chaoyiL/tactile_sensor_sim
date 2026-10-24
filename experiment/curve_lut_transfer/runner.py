"""Frozen-source analytic LUT transfer experiment; no rendering during preflight."""
from __future__ import annotations
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid
import numpy as np

from experiment.config import load_experiment_config, resolve_curve_lut_transfer_settings
from utils import backend_layout
from utils.config import validate_config
from utils.curve_calibration import (canonical_snapshot, reference_grid, MaterialSampler,
                                     SAMPLING_VERSION, SLOPE_FRAME_ID)
from utils.calibration_pipeline import parse_calibration_settings, fit_and_validate
from utils.curve_geometry import GEOMETRY_MODEL, LIGHT_MODEL
from utils.lut_results import load_lut_result
from utils.normal_mapping import NormalCalibration
from .storage import digest, signature, json_write, npz_write, load_arrays, save_frame, Checkpoints
from .metrics import evaluate_pose, aggregate, paired_bootstrap, mapping_difference

VERSION = 'curve_lut_transfer_v1'
RESUME_INPUT_FIELDS = ('source', 'source_files', 'settings', 'config', 'split',
                       'training_poses', 'test_poses')


def condition_id(radius: float | None) -> str:
    return 'flat' if radius is None else 'radius_' + format(radius, '.12g').replace('.', 'p')


def pose_config(base: dict, radius: float | None, pose: dict | None = None) -> dict:
    cfg = copy.deepcopy(base)
    cfg['surface']['bending']['radius_mm'] = radius
    if pose is not None:
        cfg['indentation'].update(center_uv=pose['center_uv'], depth_mm=pose['depth_mm'])
    else:
        cfg['indentation'].update(center_uv=[0., 0.], depth_mm=0.)
    return cfg


def prepare(source: Path, settings: dict) -> dict:
    root, manifest, backend = load_lut_result(source)
    if backend != 'analytic':
        raise ValueError('curve_lut_transfer currently supports analytic LUTs only')
    lut = NormalCalibration.load(root / 'normal_lut.npz')
    meta = lut.metadata
    expected = {'color_space': 'linear_rgb', 'image_encoding': 'srgb8_roundtrip',
                'coordinate_kind': 'reference_arclength', 'residual_domain': 'material_grid'}
    for k, v in expected.items():
        if meta.get(k) != v:
            raise ValueError(f'{root}: unsupported LUT {k}={meta.get(k)!r}; expected {v}')
    for key, expected_frame in (('fixed_reference_normal', [0, 0, 1]),
                                ('fixed_tangent_frame', [[1, 0], [0, 1], [0, 0]])):
        observed = np.asarray(meta.get(key), dtype=float)
        if observed.shape != np.shape(expected_frame) or not np.allclose(observed, expected_frame, rtol=0, atol=1e-12):
            raise ValueError(f'{root}: incompatible source {key}')
    if lut.original_valid is None:
        raise ValueError(f'{root}: original_valid is required for support comparisons')
    base = canonical_snapshot(manifest['config'])
    if base['calibration']['sampling'].get('contact_margin_mm', 0) != 0:
        raise ValueError('This experiment requires source contact_margin_mm=0')
    parse_calibration_settings(base, root / 'config.json', root)
    width = np.asarray(base['surface']['size_mm'])
    train = []
    for entry in manifest['poses']:
        if entry.get('status') != 'accepted':
            raise ValueError(f'Source has an unaccepted pose: {entry.get("image_key")}')
        if not isinstance(entry.get('image_key'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', entry['image_key']):
            raise ValueError(f'Unsafe or unsupported source pose key: {entry.get("image_key")}')
        uv = entry.get('center_uv')
        if uv is None:
            uv = (np.asarray(entry['center_xy_mm']) * 2 / width).tolist()
        train.append(dict(pose_id=entry['image_key'], center_uv=uv,
                          depth_mm=entry['depth_mm'], anchor_id=str(uv)))
    keys = [p['pose_id'] for p in train]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate source pose keys')
    split = {k: meta[k] for k in ('training_images', 'validation_images')}
    if (not split['training_images'] or not split['validation_images']
            or set(split['training_images']) & set(split['validation_images'])
            or (set(split['training_images']) | set(split['validation_images'])) != set(keys)
            or sum(map(len, split.values())) != len(keys)):
        raise ValueError('Source fixed training/validation split must partition all source poses')
    source_anchors = np.asarray([p['center_uv'] for p in train])
    test = []
    grid = settings['test_grid']
    for i, u in enumerate(grid['u']):
        for j, v in enumerate(grid['v']):
            if np.any(np.all(np.isclose(source_anchors, [u, v], rtol=0, atol=1e-10), axis=1)):
                raise ValueError(f'Test anchor {(u,v)} overlaps source calibration; choose independent anchors')
            for depth in grid['depths_mm']:
                test.append(dict(pose_id=f'test_{len(test):06d}', center_uv=[u,v], depth_mm=depth,
                                 anchor_id=f'{i}_{j}'))
    conditions = {condition_id(r): r for r in settings['radii_mm']}
    if len(conditions) != len(settings['radii_mm']):
        raise ValueError('Radius condition IDs collide')
    for name, radius in conditions.items():
        validate_config(pose_config(base, radius), allow_curve=True, allow_contact_tangent=True)
        for group, tangent in ((train, True), (test, False)):
            for pose in group:
                try:
                    validate_config(pose_config(base, radius, pose), allow_curve=True, allow_contact_tangent=tangent)
                except ValueError as exc:
                    raise ValueError(f'{name}/{pose["pose_id"]}: {exc}') from exc
    if settings['adaptation']['enabled'] and max(settings['adaptation']['pose_budgets']) > len(train):
        raise ValueError('Adaptation pose budget exceeds available source poses')
    source_files = {name: digest(root / name) for name in
                    ('manifest.json', 'normal_lut.npz', 'normal_lut_train.npz', 'normal_samples.npz')}
    identity = dict(version=VERSION, sampling_version=SAMPLING_VERSION, slope_frame_id=SLOPE_FRAME_ID,
                    source=str(root), source_files=source_files, settings=settings,
                    config=base, split=split, training_poses=train, test_poses=test)
    count = len(conditions) * (len(test)+1) + (len(conditions)-1)*len(train)
    g = reference_grid(base)
    pixels = base['camera']['width_px'] * base['camera']['height_px']
    # Conservative uncompressed accounting: RGB/sensor/radiance/geometry plus full material diagnostics.
    estimate = count * (pixels * 200 + g.xy_mm.shape[0]*g.xy_mm.shape[1]*180)
    current_cal = backend_layout.CONFIG_DIR / 'calibration.json'
    live = json.loads(current_cal.read_text(encoding='utf-8')) if current_cal.is_file() else None
    return dict(root=root, source_manifest=manifest, lut=lut, base=base, train=train, test=test,
                conditions=conditions, split=split, identity=identity,
                run_signature=signature({k: identity[k] for k in RESUME_INPUT_FIELDS}),
                workload=dict(reference_frames=len(conditions), training_frames=(len(conditions)-1)*len(train),
                              test_frames=len(conditions)*len(test), total_frames=count,
                              material_grid_shape=list(g.xy_mm.shape[:2]),
                              uncompressed_storage_upper_estimate_bytes=estimate,
                              source_samples_reused='flat LUT only; curved training and all test observations newly rendered',
                              live_calibration_differs=(live.get('calibration', live) != base['calibration']) if live else None))


def start_run(prepared: dict, settings: dict, out: Path | None, resume: bool) -> Checkpoints:
    parent = prepared['root'] / 'evaluation' / 'curve_lut_transfer'
    if out is None:
        out = parent / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8])
    out = out.resolve()
    if not out.is_relative_to(parent.resolve()) or out == parent.resolve():
        raise ValueError(f'--out must be a run subdirectory of {parent}')
    if resume:
        path = out / 'manifest.json'
        if not path.is_file():
            raise ValueError(f'--resume requires a checkpoint manifest: {path}')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        previous = manifest.get('identity', {})
        changed = [key for key in RESUME_INPUT_FIELDS
                   if previous.get(key) != prepared['identity'].get(key)]
        if changed:
            raise ValueError(f'Resume inputs differ ({", ".join(changed)}): {path}')
        result = Checkpoints(out, manifest)
        result.verify()
        # Keep the existing run ID, including runs whose old ID also hashed code.
        # Code edits and runtime version changes are not data incompatibilities.
        prepared['run_signature'] = manifest['run_signature']
        return result
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Nonempty output requires compatible --resume: {out}')
    raw = {name: str(backend_layout.PROJECT_ROOT / ('results_render' if r is None else 'results_curve_render')
                      / 'analytic' / 'curve_lut_transfer' / out.name / name)
           for name, r in prepared['conditions'].items()}
    for path in raw.values():
        if Path(path).exists() and any(Path(path).iterdir()):
            raise ValueError(f'Raw render directory already exists: {path}')
    manifest = dict(version=VERSION, status='running', run_signature=prepared['run_signature'],
                    identity=prepared['identity'], raw_render_roots=raw, completed={},
                    workload=prepared['workload'], calibration={}, geometry_model_id=GEOMETRY_MODEL,
                    light_model_id=LIGHT_MODEL, reference_pairing='material',
                    contact_boundary='tangent allowed only for calibration; independent tests strictly interior')
    result = Checkpoints(out, manifest)
    result.save()
    json_write(out / 'config_snapshot.json', dict(simulation=prepared['base'], experiment=settings))
    json_write(out / 'split.json', dict(prepared['split'], independent_test=prepared['test']))
    result.complete('inputs', [out / 'config_snapshot.json', out / 'split.json'])
    return result


def sample_path(root: Path, condition: str, group: str, pose: dict) -> Path:
    return root / 'samples' / condition / group / (pose['pose_id']+'.npz')


def render_condition(p: dict, ck: Checkpoints, name: str, radius: float | None) -> None:
    raw = Path(ck.manifest['raw_render_roots'][name])
    reference_path = ck.root / 'samples' / name / 'reference.npz'
    jobs = [('test', p['test'])] + ([] if radius is None else [('training', p['train'])])
    if ck.has(name+'/reference') and all(ck.has(f'{name}/{group}/{pose["pose_id"]}') for group, poses in jobs for pose in poses):
        return

    # 负责生成标定球采样的对象
    sampler = MaterialSampler(pose_config(p['base'], radius))
    if ck.has(name+'/reference'):
        a = load_arrays(reference_path)
        sampler.restore_reference(a['rgb8'], a['image_mask'])
    else:
        frame = sampler.make_reference()
        files = save_frame(raw / 'reference', frame, sampler.execution())
        host = lambda t: t.detach().cpu().numpy()
        npz_write(reference_path, rgb8=host(frame['rgb8']), image_mask=host(frame['gt']['mask']),
                  color=host(sampler.reference['color']), valid=host(sampler.reference['valid']))
        ck.complete(name+'/reference', files+[reference_path])
    for group, poses in jobs:
        for i, pose in enumerate(poses):
            key = f'{name}/{group}/{pose["pose_id"]}'
            if ck.has(key):
                continue
            cfg = pose_config(p['base'], radius, pose)
            sample, frame = sampler.sample(cfg)
            path = sample_path(ck.root, name, group, pose)
            npz_write(path, **sample, metadata=json.dumps(dict(pose=pose, run_signature=p['run_signature'],
                       slope_frame_id=SLOPE_FRAME_ID, sampling_version=SAMPLING_VERSION)))
            files = save_frame(raw / group / pose['pose_id'], frame, sampler.execution())
            ck.complete(key, files+[path], sample_count=len(sample['grid_index']),
                        contact_count=int(sample['contact_mask'].sum()))
            print(f'{name} {group}: {i+1}/{len(poses)}; {len(sample["grid_index"])} samples', flush=True)


def calibrate_condition(p: dict, ck: Checkpoints, name: str) -> NormalCalibration | None:
    if ck.has(name+'/calibration'):
        info = ck.manifest['calibration'][name]
        return NormalCalibration.load(ck.root / info['lut']) if info['lut'] else None
    out = ck.root / 'calibration' / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = pose_config(p['base'], p['conditions'][name])
    settings = parse_calibration_settings(cfg, p['root'] / 'config.json', out)
    pieces = [load_arrays(sample_path(ck.root, name, 'training', pose)) for pose in p['train']]
    counts = [len(s['grid_index']) for s in pieces]
    accepted = [n >= settings.minimum_samples_per_pose for n in counts]
    fields = ('color_residual_samples', 'slope_samples', 'material_coordinates',
              'reference_normal_samples', 'tangent_frame_samples', 'normal_displacement_mm')
    samples = {k: np.concatenate([s[k] for s, ok in zip(pieces, accepted) if ok])
               if any(accepted) else pieces[0][k][:0] for k in fields}
    samples['image_index'] = np.concatenate([np.full(n, i, np.int32) for i, (n, ok) in enumerate(zip(counts, accepted)) if ok]) if any(accepted) else np.empty(0, np.int32)
    keys = [pose['pose_id'] for pose in p['train']]
    samples['image_paths'] = np.asarray(keys)
    metadata = dict(p['lut'].metadata, run_signature=signature(dict(parent=p['run_signature'], condition=name)),
                    parent_run_signature=p['run_signature'], backend='analytic',
                    geometry_model_id=GEOMETRY_MODEL, light_geometry_model=LIGHT_MODEL,
                    slope_frame_id=SLOPE_FRAME_ID, sampling_version=SAMPLING_VERSION,
                    coordinate_chart_id='cylinder_reference_arclength_'+name,
                    coordinate_chart_storage='embedded_reference_arclength_axes',
                    fixed_reference_normal=None, fixed_tangent_frame=None,
                    simulation_config_signature=signature(cfg))
    # Never inherit publication/validation claims from the source LUT.
    for key in ('validation', 'validation_status', 'training_images', 'validation_images'):
        metadata.pop(key, None)
    grid = reference_grid(cfg)
    npz_write(out / 'normal_samples.npz', **samples, metadata=json.dumps(metadata), format_version=5,
              material_grid_x_mm=grid.x_mm, material_grid_y_mm=grid.y_mm)
    json_write(out / 'config.json', cfg)
    if sum(accepted) < 2 or not all(any(ok and k in p['split'][part] for k, ok in zip(keys, accepted)) for part in p['split']):
        report = dict(status='failed', reason='Insufficient accepted poses in fixed split',
                      missing_required_poses=[k for k, ok in zip(keys, accepted) if not ok])
        final = None
    else:
        report, training, final, diagnostic = fit_and_validate(samples, settings, out, metadata,
                                                               pose_split=p['split'], required_pose_keys=keys)
    report['pose_sample_counts'] = dict(zip(keys, counts))
    json_write(out / 'validation.json', report)
    lut_name = ('normal_lut.npz' if report['status'] == 'passed' else 'normal_lut_diagnostic.npz') if final is not None else None
    info = dict(status=report['status'], lut=str((out / lut_name).relative_to(ck.root)) if lut_name else None,
                required_pose_count=len(keys), accepted_pose_count=sum(accepted))
    ck.manifest['calibration'][name] = info
    json_write(out / 'manifest.json', dict(info, config=cfg, run_signature=metadata['run_signature'],
                                         stage='completed' if report['status']=='passed' else 'diagnostic',
                                         parent_run_signature=p['run_signature']))
    ck.complete(name+'/calibration', sorted(path for path in out.rglob('*') if path.is_file()))
    return final


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(p: dict, ck: Checkpoints, luts: dict, settings: dict) -> dict:
    names = list(p['conditions'])
    datasets = {name: [load_arrays(sample_path(ck.root, name, 'test', pose)) for pose in p['test']] for name in names}
    common = [np.array(sorted(set.intersection(*(set(datasets[n][j]['grid_index']) for n in names))), dtype=np.int64)
              for j in range(len(p['test']))]
    rows, summaries, row_sets, detail_sets, coverage = [], {}, {}, {}, []
    for name in names:
        for pose, sample, indices in zip(p['test'], datasets[name], common):
            take = np.searchsorted(sample['grid_index'], indices)
            full_area = float(sample['area_mm2'].sum())
            coverage.append(dict(condition=name, pose_id=pose['pose_id'], valid_count=len(sample['grid_index']),
                                 contact_count=int(sample['contact_mask'].sum()), common_count=len(indices),
                                 common_area_fraction=float(sample['area_mm2'][take].sum()/full_area) if full_area else None,
                                 slope_min=sample['slope_samples'].min(0).tolist() if full_area else None,
                                 slope_max=sample['slope_samples'].max(0).tolist() if full_area else None,
                                 maximum_tilt_degrees=float(np.degrees(np.arctan(np.linalg.norm(sample['slope_samples'],axis=1).max()))) if full_area else None,
                                 validity_bit_counts={str(bit): int(np.count_nonzero(sample['validity_bits'] & bit)) for bit in (1,2,4,8,16,32,64)}))
    for lut_name, lut in luts.items():
        for data_name in names:
            for domain in ('common', 'full'):
                key = f'{lut_name}/{data_name}/{domain}'
                rr, dd = [], []
                if lut is None:
                    summaries[key] = dict(status='unavailable_lut', rmse=None)
                    continue
                for pose, sample, indices in zip(p['test'], datasets[data_name], common):
                    row, detail = evaluate_pose(sample, lut, indices if domain=='common' else None)
                    row.update(pose_id=pose['pose_id'], anchor_id=pose['anchor_id'],
                               u=pose['center_uv'][0], v=pose['center_uv'][1], depth_mm=pose['depth_mm'],
                               lut=lut_name, condition=data_name, domain=domain)
                    rr.append(row)
                    dd.append(detail)
                summaries[key] = aggregate(rr, dd, settings['metrics']['slope_tolerances'])
                summaries[key]['lut_status'] = 'published' if lut_name=='flat' else ck.manifest['calibration'][lut_name]['status']
                row_sets[key], detail_sets[key] = rr, dd
                rows.extend(rr)
    comparisons, differences = {}, {}
    for name in names:
        if name == 'flat':
            continue
        transfer_key = f'flat/{name}/common'
        baseline_key = 'flat/flat/common'
        bootstrap = dict(repeats=settings['metrics']['bootstrap_repeats'], seed=settings['metrics']['seed'])
        comparisons[name] = dict(transfer_increase=paired_bootstrap(row_sets[transfer_key], row_sets[baseline_key], **bootstrap))
        own = f'{name}/{name}/common'
        if own in row_sets:
            comparisons[name]['recalibration_improvement'] = paired_bootstrap(row_sets[transfer_key], row_sets[own], **bootstrap)
            report, arrays = mapping_difference(luts['flat'], luts[name], datasets['flat'], datasets[name])
            differences[name] = report
            npz_write(ck.root / f'mapping_difference_{name}.npz', **arrays)
        limit, increase = settings['metrics']['acceptance_slope_rmse'], settings['metrics']['maximum_transfer_increase']
        rmse = summaries[transfer_key]['rmse']
        delta = comparisons[name]['transfer_increase'].get('difference')
        comparisons[name]['acceptance'] = ('not_specified' if limit is None and increase is None else
            'unavailable' if rmse is None or delta is None else
            'passed' if (limit is None or rmse <= limit) and (increase is None or delta <= increase) else 'failed')
    result = dict(version=VERSION, conditions=p['conditions'], metrics=summaries, comparisons=comparisons,
                  mapping_difference=differences, coverage=coverage, calibration=ck.manifest['calibration'],
                  metric_definition='sqrt(mean_pose(sum_area_weighted squared vector slope error)); local b',
                  source_planar_lut='published frozen; all source poses used in final fit',
                  observation='signed material-paired inverse_srgb(srgb8), tone mapping not inverted')
    if settings['adaptation']['enabled']:
        from .adaptation import run_adaptation
        result['adaptation'] = run_adaptation(p, ck, datasets, common, settings)
    write_csv(ck.root / 'per_pose.csv', rows)
    write_csv(ck.root / 'cross_curvature.csv', [dict(comparison=k, **v) for k,v in summaries.items()])
    json_write(ck.root / 'metrics.json', result)
    from .plots import make_plots
    make_plots(ck.root, p, luts, summaries, row_sets, detail_sets, datasets)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lut-results', required=True, type=Path)
    parser.add_argument('--exp-config', type=Path, default=backend_layout.CONFIG_DIR / 'curve_lut_transfer.json')
    parser.add_argument('--out', type=Path, help='New run directory under source/evaluation/curve_lut_transfer')
    parser.add_argument('--resume', action='store_true', help='Verify and resume exactly the run selected by --out')
    parser.add_argument('--dry-run', action='store_true', help='Read-only source/geometry/split/workload validation')
    args = parser.parse_args(argv)
    if args.resume and args.out is None:
        parser.error('--resume requires --out')
    settings = resolve_curve_lut_transfer_settings(load_experiment_config(args.exp_config))
    p = prepare(args.lut_results, settings)
    if args.out is not None:
        parent = p['root'] / 'evaluation' / 'curve_lut_transfer'
        if not args.out.resolve().is_relative_to(parent) or args.out.resolve() == parent:
            raise ValueError(f'--out must be a run subdirectory of {parent}')
        if not args.resume and args.out.exists() and any(args.out.iterdir()):
            raise ValueError(f'Nonempty output: {args.out}')
    if args.dry_run:
        if args.resume:
            start_run(p, settings, args.out, True)  # Read-only integrity validation.
        print(json.dumps(dict(source=str(p['root']), signature=p['run_signature'], conditions=p['conditions'],
              workload=p['workload'], fixed_training_poses=len(p['split']['training_images']),
              fixed_validation_poses=len(p['split']['validation_images']), independent_test_poses=len(p['test']),
              output=str(args.out or p['root'] / 'evaluation/curve_lut_transfer/<unique_run>')),
              ensure_ascii=False, indent=2))
        return
    ck = start_run(p, settings, args.out, args.resume)
    print(f'Experiment output: {ck.root}', flush=True)
    if ck.manifest['status'] == 'completed':
        print('Completed run verified; no artifacts rewritten.')
        return
    
    for name, radius in p['conditions'].items():
        render_condition(p, ck, name, radius)
    luts = {'flat': p['lut']}
    for name, radius in p['conditions'].items():
        if radius is not None:
            luts[name] = calibrate_condition(p, ck, name)
    evaluate(p, ck, luts, settings)
    files = [ck.root / 'metrics.json', ck.root / 'per_pose.csv', ck.root / 'cross_curvature.csv']
    files += list(ck.root.glob('mapping_difference_*.npz')) + list((ck.root / 'figures').glob('*'))
    if settings['adaptation']['enabled']:
        files.append(ck.root / 'adaptation.json')
    ck.complete('evaluation', files)
    ck.manifest['status'] = 'completed'
    ck.save()
    print(f'Completed: {ck.root / "metrics.json"}', flush=True)
