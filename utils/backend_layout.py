"""Public backend names, configuration locations, and result layouts."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / 'configs'
BACKEND_RESULT_DIRS = {'analytic': 'analytic', 'zemax': 'zmax', 'xense': 'xense'}


def canonical_backend(value: str) -> str:
    """Normalize historical MuJoCo artifacts to the public Xense backend name."""
    name = 'xense' if value == 'mujoco' else value
    if name not in BACKEND_RESULT_DIRS:
        raise ValueError(f'Unknown tactile backend: {value}')
    return name


def simulation_config_path(backend: str) -> Path:
    return CONFIG_DIR / f'{canonical_backend(backend)}.json'


def lut_result_dir(backend: str) -> Path:
    return PROJECT_ROOT / 'results_lut' / BACKEND_RESULT_DIRS[canonical_backend(backend)]


def new_render_output_dir(backend: str, *, curved: bool = False) -> Path:
    """Choose a new render run under the geometry-specific root without writing.

    Render folders use public backend names; the historical ``zmax`` spelling
    remains specific to LUT results. Explicit CLI output paths bypass this default.
    """
    root = 'results_curve_render' if curved else 'results_render'
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
    return PROJECT_ROOT / root / canonical_backend(backend) / run
