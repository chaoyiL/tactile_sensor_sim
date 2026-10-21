"""Run: uv run python simulate.py --config config.example.json --backend analytic --out results"""
import argparse
import copy
import json
from pathlib import Path
import time
from utils.config import load_config
from utils.geometry import export_stl
from utils.renderer import render, save_result, camera_ground_truth


def main():
    parser = argparse.ArgumentParser(description='Spherical indentation + colored LED strips + calibrated pinhole camera')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.example.json'))
    parser.add_argument('--backend', choices=['analytic', 'zemax'], default='analytic')
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--device', help='Analytic device override, e.g. cuda:0 or cpu')
    parser.add_argument('--reference', action='store_true', help='Also render a zero-depth reference with identical exposure')
    parser.add_argument('--build-only', action='store_true', help='Only build the Zemax non-sequential scene')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device is not None:
        if args.backend != 'analytic':
            parser.error('--device only applies to --backend analytic')
        config['render']['device'] = args.device
    if args.build_only and args.backend != 'zemax':
        parser.error('--build-only requires --backend zemax')
    if args.backend == 'zemax' and not config['render']['shadows']:
        raise ValueError('Zemax uses physical occlusion. Set render.shadows=true to match both backends.')
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
                (folder / 'metadata.json').write_text(json.dumps({'backend':'zemax-build-only', 'config':cfg, **extra}, indent=2), encoding='utf-8')
        print(f'{name}: {args.backend} -> {folder.resolve()} ({time.perf_counter()-start:.2f}s)', flush=True)


if __name__ == '__main__':
    main()
