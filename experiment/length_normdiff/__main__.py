"""Entry point: python -m experiment.length_normdiff."""
if __package__ in (None, ''):
    # Also allow python experiment/length_normdiff/__main__.py from any directory.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiment.length_normdiff.runner import main
else:
    from .runner import main


if __name__ == '__main__':
    main()
