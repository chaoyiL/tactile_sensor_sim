"""Calibrate a signed color-residual to local-slope LUT from simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.backend_layout import CONFIG_DIR, canonical_backend, simulation_config_path
from utils.calibration_pipeline import (load_calibration_config,
                                        parse_calibration_settings, run_calibration)


def main():
    parser = argparse.ArgumentParser(
        description='Scan a spherical indentation and calibrate a normal LUT')
    parser.add_argument('--config', type=Path,
                        help='Selected simulator config (default: configs/<backend>.json)')
    parser.add_argument('--calibration-config', type=Path,
                        default=CONFIG_DIR / 'calibration.json',
                        help='Shared scan, fit and validation settings')
    parser.add_argument('--out', type=Path, help='Override calibration.output_dir')
    parser.add_argument('--backend', choices=('analytic', 'zemax', 'xense'), required=True,
                        help='Simulator used to collect calibration samples')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true',
                        help='Validate config and print the scan without rendering')
    mode.add_argument('--ambiguity-only', action='store_true',
                      help='Analyze saved samples without rendering or refitting a LUT')
    parser.add_argument('--samples', type=Path,
                        help='Existing ambiguity_inputs.npz or normal_samples.npz (ambiguity-only)')
    parser.add_argument('--ambiguity-stride', type=int,
                        help='Take every Nth sample within each pose (ambiguity-only)')
    args = parser.parse_args()
    args.config = args.config or simulation_config_path(args.backend)
    selected = canonical_backend(load_calibration_config(
        args.config, args.calibration_config).get('backend', 'analytic'))
    if args.backend != selected:
        parser.error(f'--backend {args.backend} disagrees with config backend {selected}')
    if args.ambiguity_only:
        from utils.calibration_ambiguity import run_ambiguity_analysis
        config_path = args.config.expanduser().resolve()
        settings = parse_calibration_settings(load_calibration_config(
            config_path, args.calibration_config), config_path, args.out)
        source = args.samples or settings.output_dir / 'ambiguity_inputs.npz'
        if not source.is_file() and args.samples is None:
            source = settings.output_dir / 'normal_samples.npz'
        report = run_ambiguity_analysis(
            source, settings.output_dir,
            stride=(settings.ambiguity_stride if args.ambiguity_stride is None
                    else args.ambiguity_stride),
            color_threshold=settings.color_distance_threshold,
            angle_threshold=settings.normal_angle_threshold_degrees,
            export_pairs=settings.export_ambiguity_pairs,
            dominance_fraction=settings.branch_dominance_fraction)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        if args.samples is not None or args.ambiguity_stride is not None:
            parser.error('--samples and ambiguity overrides require --ambiguity-only')
        run_calibration(args.config, args.out, dry_run=args.dry_run,
                        calibration_config_path=args.calibration_config)


if __name__ == '__main__':
    main()
