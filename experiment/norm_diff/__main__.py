"""Entry point: python -m experiment.norm_diff."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiment.norm_diff.runner import main
else:
    from .runner import main


if __name__ == '__main__':
    main()
