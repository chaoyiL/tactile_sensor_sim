"""Thin module/direct-script entry point."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiment.curve_lut_transfer.runner import main
else:
    from .runner import main

if __name__ == '__main__':
    main()
