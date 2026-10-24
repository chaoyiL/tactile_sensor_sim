"""Entry point: python -m experiment.lut_quality."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiment.lut_quality.runner import main
else:
    from .runner import main


if __name__ == '__main__':
    main()
