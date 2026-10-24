"""Render with a selected simulator and its own configuration."""
import argparse
import copy
import json
from pathlib import Path
import time
from utils.backend_layout import canonical_backend, simulation_config_path, new_render_output_dir
from utils.config import load_config
from utils.surface_coordinates import center_material_mm
from utils.geometry import LIGHT_GEOMETRY_MODEL, export_stl
from utils.renderer import render, save_result, camera_ground_truth


def main():
    parser = argparse.ArgumentParser(description='Render a spherical indentation with the selected tactile backend')
    parser.add_argument('--config', type=Path,
                        help='Selected simulator config (default: configs/<backend>.json)')
    parser.add_argument('--backend', choices=['analytic', 'zemax', 'xense'], required=True,
                        help='Simulation backend')
    parser.add_argument('--out', type=Path,
                        help='New/empty output directory (default: results_render/<backend>/<unique run>)')
    parser.add_argument('--device', help='Analytic device override, e.g. cuda:0 or cpu')
    parser.add_argument('--reference', action='store_true',
                        help='Also render a zero-depth reference with the same backend settings')
    parser.add_argument('--build-only', action='store_true', help='Only build the Zemax non-sequential scene')
    args = parser.parse_args()
    args.config = args.config or simulation_config_path(args.backend)
    config = load_config(args.config)
    selected = canonical_backend(config.get('backend', 'analytic'))
    if args.backend != selected:
        parser.error(f'--backend {args.backend} disagrees with config backend {selected}')
    if args.device is not None:
        if args.backend != 'analytic':
            parser.error('--device only applies to --backend analytic')
        config['render']['device'] = args.device
    if args.build_only and args.backend != 'zemax':
        parser.error('--build-only requires --backend zemax')
    if args.backend == 'zemax' and not config['render']['shadows']:
        raise ValueError('Zemax uses physical occlusion. Set render.shadows=true to match both backends.')
    args.out = (args.out or new_render_output_dir(args.backend)).expanduser().resolve()
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        raise ValueError(f'Output directory is not empty: {args.out}')
    if args.backend == 'xense':
        from PIL import Image
        import numpy as np
        from utils.mujoco_backend import (MujocoSession, GEOMETRY_MODEL_ID,
                                          OBSERVATION_MODEL_ID)
        args.out.mkdir(parents=True, exist_ok=True)
        with MujocoSession(config, args.config.resolve(), args.out / 'worker') as session:
            for name, cfg in [('indented', config)] + ([('reference', copy.deepcopy(config))]
                                                       if args.reference else []):
                if name == 'reference':
                    cfg['indentation']['depth_mm'] = 0.
                folder = args.out / name if args.reference else args.out
                folder.mkdir(parents=True, exist_ok=True)
                frame = session.render(center_material_mm(cfg),
                                       cfg['indentation']['depth_mm'])
                Image.fromarray(frame['rgb8']).save(folder / 'image.png')
                height, width = frame['depth_input_mm'].shape
                width_mm, height_mm = cfg['surface']['size_mm']
                x, y = np.meshgrid((np.arange(width) + .5) / width * width_mm - width_mm / 2,
                                   (np.arange(height) + .5) / height * height_mm - height_mm / 2)
                z = frame['depth_input_mm']
                np.savez_compressed(folder / 'geometry_gt.npz',
                                    position_mm=np.stack((x, y, z), axis=-1),
                                    depth_input_mm=z, depth_m=frame['depth_m'])
                (folder / 'metadata.json').write_text(json.dumps({
                    'backend': 'xense', 'geometry_model_id': GEOMETRY_MODEL_ID,
                    'observation_model_id': OBSERVATION_MODEL_ID,
                    'image_encoding': 'xensim_rgb8_unverified_transfer',
                    'marker_mode': 'SimpleSensor_without_marker_overlay',
                    'runtime': session.handshake,
                    'config': cfg, 'geometry_grid_wh': [width, height],
                    'rgb_size_wh': cfg['mujoco']['render_size_wh']},
                    indent=2, ensure_ascii=False), encoding='utf-8')
                print(f'{name}: xense -> {folder.resolve()}', flush=True)
        return
    for name, cfg in [('indented', config)] + ([('reference', copy.deepcopy(config))] if args.reference else []):
        if name == 'reference':
            cfg['indentation']['depth_mm'] = 0.
        folder = args.out / name if args.reference else args.out
        folder.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        if args.backend == 'analytic':
            sensor, radiance, gt = render(cfg, device=args.device)
            export_stl(folder / 'surface.stl', cfg)
            save_result(folder, sensor, cfg, 'analytic', gt, radiance)
        else:
            from utils.zemax_backend import render as zemax_render
            sensor, extra = zemax_render(cfg, folder, args.build_only)
            if sensor is not None:
                gt = camera_ground_truth(cfg)
                sensor[~gt['mask']] = 0
                extra['geometry_gt_source'] = 'analytic cap; Zemax traces its triangulated STL approximation'
                save_result(folder, sensor, cfg, 'zemax', gt=gt, extra=extra)
            else:
                (folder / 'metadata.json').write_text(json.dumps({
                    'backend': 'zemax-build-only', 'light_geometry_model': LIGHT_GEOMETRY_MODEL,
                    'config': cfg, **extra}, indent=2), encoding='utf-8')
        print(f'{name}: {args.backend} -> {folder.resolve()} ({time.perf_counter()-start:.2f}s)', flush=True)


if __name__ == '__main__':
    main()
