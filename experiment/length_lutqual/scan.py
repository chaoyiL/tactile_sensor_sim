"""Complete GPU scans, stored in the same raw-sample format as normal calibration."""
from __future__ import annotations

import copy
import gc
from pathlib import Path
import time

from experiment.length_normdiff.runner import center_poses
from utils.calibration_pipeline import _config_signature, _save_samples
from utils.calibration_samples import (build_material_grid, calibration_declarations,
                                       compact_frame_samples, render_reference, sample_pose)
from utils.geometry import LIGHT_GEOMETRY_MODEL
from utils.surface_coordinates import normal_direction
from utils.normal_mapping import UPSTREAM_COMMIT


def render_samples(config: dict, poses: list, output: Path, minimum: int) -> dict:
    """Render one reference, every scan pose and each fixed-centre depth. Never thin samples."""
    import torch
    from utils.renderer_gpu import AnalyticGpuContext

    grid = build_material_grid(config)
    controls = center_poses(poses)
    signature = _config_signature(config)
    metadata = dict(calibration_kind='simulated_sphere_scan',
                    light_geometry_model=LIGHT_GEOMETRY_MODEL, upstream_commit=UPSTREAM_COMMIT,
                    run_signature=signature, simulation_config_signature=signature,
                    sphere_radius_mm=float(config['indentation']['radius_mm']),
                    sphere_direction_z=int(normal_direction(config)),
                    sphere_side=('positive_normal' if normal_direction(config) == -1
                                 else 'negative_normal'),
                    boundary_model='zero_boundary_displacement',
                    scan=copy.deepcopy(config['calibration']['scan']),
                    reference_color_source='analytic_zero_depth_srgb8_roundtrip',
                    fixed_reference_normal=[0., 0., 1.],
                    fixed_tangent_frame=[[1., 0.], [0., 1.], [0., 0.]],
                    render=dict(config['render'], torch_version=torch.__version__,
                                cuda_version=torch.version.cuda),
                    material_grid_shape=list(grid.shape), **calibration_declarations(grid))
    context, reference = None, None
    records = {}
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            context = AnalyticGpuContext(config)
            reference = render_reference(context, grid)
            for label, group, filename in (('full_scan', poses, 'normal_samples.npz'),
                                            ('fixed_center', controls, 'center_samples.npz')):
                pieces, rows = [], []
                last_progress = time.perf_counter()
                for sequence, pose in enumerate(group, 1):
                    frame = sample_pose(context, reference, grid, pose)
                    piece = compact_frame_samples(frame, grid)
                    count = len(piece['slope_samples'])
                    if count < minimum:
                        raise RuntimeError(f'{label} pose {pose.image_key}: {count} samples < {minimum}; '
                                           'refusing a biased partial scan')
                    pieces.append(piece)
                    rows.append(dict(pose.to_dict(), sample_count=count))
                    del frame, piece
                    if time.perf_counter() - last_progress >= 10 or sequence == len(group):
                        print(f'[length={config["surface"]["size_mm"][0]:g}] '
                              f'{label} render {sequence}/{len(group)}', flush=True)
                        last_progress = time.perf_counter()
                keys, pose_ids = [p.image_key for p in group], [p.pose_id for p in group]
                group_metadata = dict(metadata, sample_group=label, accepted_pose_count=len(group),
                                      rejected_pose_count=0, pose_key_by_image_index=keys,
                                      pose_id_by_image_index=pose_ids)
                saved = _save_samples(output / filename, pieces, keys, pose_ids, grid, group_metadata)
                records[label] = dict(pose_count=len(group), sample_count=len(saved['slope_samples']), poses=rows)
                del pieces, saved
            records['renderer_compile_metrics'] = context.compile_metrics()
    finally:
        # Different lengths have different light geometry; do not retain old CUDA graphs.
        context, reference = None, None
        gc.collect()
        torch.cuda.empty_cache()
    records['elapsed_seconds'] = time.perf_counter() - started
    records['reference_render_count'] = 1
    return records
