"""Calibrate a signed color-residual to local-slope LUT from analytic simulation."""
from __future__ import annotations

import argparse
from pathlib import Path

from utils.calibration_pipeline import run_calibration


def main():
    parser = argparse.ArgumentParser(
        description='Scan an analytic spherical indentation and calibrate a normal LUT')
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).with_name('config.example.json'))
    parser.add_argument('--out', type=Path, help='Override calibration.output_dir')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate config and print the scan without rendering')
    args = parser.parse_args()
    run_calibration(args.config, args.out, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
