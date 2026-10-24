"""Host-side contract for the isolated MuJoCo + marker-free Xensim renderer."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import queue
import subprocess
import threading
import uuid

import numpy as np
from .surface_coordinates import legacy_flat_config


OBSERVATION_MODEL_ID = 'xensim_simple_rgb8_inverse_srgb_convention_v1'
GEOMETRY_MODEL_ID = 'mujoco_orthographic_sphere_depth_v1'
PROTOCOL_VERSION = 1


def _vector(value, count: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain {count} finite numbers') from exc
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain {count} finite numbers')
    return array


def _path(value: str, config_path: str | Path | None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and config_path is not None:
        path = Path(config_path).resolve().parent / path
    return path.resolve()


def _number(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def validate_mujoco_config(config: dict, *, config_path: str | Path | None = None,
                           allow_contact_tangent: bool = False) -> dict:
    """Reject unsupported Xensim physics instead of silently ignoring settings."""
    original = config
    config = legacy_flat_config(config)
    allowed = {'backend', 'surface', 'indentation', 'mujoco', 'calibration'}
    extra = set(config) - allowed
    if extra:
        raise ValueError(f'MuJoCo config has unsupported sections: {sorted(extra)}')
    if config.get('backend') not in ('xense', 'mujoco'):
        raise ValueError('backend must be xense')
    surface, indentation, runtime = (config.get(key) for key in
                                     ('surface', 'indentation', 'mujoco'))
    if not all(isinstance(value, dict) for value in (surface, indentation, runtime)):
        raise ValueError('MuJoCo requires surface, indentation and mujoco objects')
    for label, obj, keys in (
        ('surface', surface, {'size_mm', 'view_side_z'}),
        ('indentation', indentation, {'center_xy_mm', 'radius_mm', 'depth_mm', 'direction_z'}),
        ('mujoco', runtime, {'python_executable', 'sensor_mode', 'calibrate_file',
                            'depth_size_wh', 'mesh_size_wh', 'render_size_wh',
                            'camera_gap_mm', 'smooth_norm', 'smooth_mesh', 'marker_policy'}),
    ):
        if set(obj) != keys:
            raise ValueError(f'{label} keys differ; missing={sorted(keys-set(obj))}, '
                             f'unknown={sorted(set(obj)-keys)}')
    size = _vector(surface['size_mm'], 2, 'surface.size_mm')
    if np.any(size <= 0) or surface['view_side_z'] != -1:
        raise ValueError('MuJoCo SimpleSensor requires positive size_mm and view_side_z=-1')
    if not np.allclose(size, [17.3, 29.14], rtol=0, atol=1e-9):
        raise ValueError('The bundled Xensim optical table is validated only for '
                         'surface.size_mm=[17.3, 29.14]')
    center = _vector(indentation['center_xy_mm'], 2, 'indentation.center_xy_mm')
    radius = _number(indentation['radius_mm'], 'indentation.radius_mm')
    depth = _number(indentation['depth_mm'], 'indentation.depth_mm')
    if not 0 <= depth < radius or indentation['direction_z'] != -1:
        raise ValueError('MuJoCo requires direction_z=-1 and 0 <= depth < radius')
    contact_radius = np.sqrt(2 * radius * depth - depth * depth)
    margin = 1e-10 * max(1, *size)
    if np.any(np.abs(center) + contact_radius > size / 2 + margin) or (
            not allow_contact_tangent and np.any(np.abs(center) + contact_radius >= size / 2)):
        raise ValueError('Contact disc exceeds the gel surface')
    if runtime['sensor_mode'] != 'simple' or runtime['marker_policy'] != 'require_absent':
        raise ValueError('Only validated marker-free SimpleSensor is supported')
    for key in ('depth_size_wh', 'mesh_size_wh', 'render_size_wh'):
        wh = runtime[key]
        if (not isinstance(wh, list) or len(wh) != 2 or
                any(type(value) is not int or value < 4 for value in wh)):
            raise ValueError(f'mujoco.{key} must be [width, height] integers >=4')
    if _number(runtime['camera_gap_mm'], 'mujoco.camera_gap_mm') <= depth:
        raise ValueError('mujoco.camera_gap_mm must exceed indentation depth')
    for key in ('smooth_norm', 'smooth_mesh'):
        if type(runtime[key]) is not int or runtime[key] < 0:
            raise ValueError(f'mujoco.{key} must be a nonnegative integer')
    if not isinstance(runtime['python_executable'], str) or not runtime['python_executable']:
        raise ValueError('mujoco.python_executable must be a path')
    if runtime['calibrate_file'] != 'bundled:g1-ws_table':
        raise ValueError('Only the bundled:g1-ws_table optical table is validated')
    return original


def check_mujoco_runtime_paths(config: dict, config_path: Path) -> None:
    runtime = config['mujoco']
    python = _path(runtime['python_executable'], config_path)
    if not python.is_file():
        raise FileNotFoundError(f'MuJoCo worker Python: {python}')
    table = runtime['calibrate_file']
    if table != 'bundled:g1-ws_table' and not _path(table, config_path).is_file():
        raise FileNotFoundError(f'Xensim calibration table: {_path(table, config_path)}')


def simulation_signature(config: dict) -> str:
    """Hash physical/render conditions, excluding only post-render evaluation/output."""
    import copy
    value = copy.deepcopy(config)
    value.get('calibration', {}).pop('output_dir', None)
    value.get('calibration', {}).pop('norm_diff', None)
    # The interpreter's location is deployment plumbing. Runtime package versions
    # and optical-table bytes enter the stronger per-run signature separately.
    value['mujoco'].pop('python_executable', None)
    data = json.dumps({'geometry_model_id': GEOMETRY_MODEL_ID,
                       'observation_model_id': OBSERVATION_MODEL_ID,
                       'config': value}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def inverse_srgb_rgb8(rgb8: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb8, np.float64) / 255.0
    return np.where(value <= .04045, value / 12.92,
                    ((value + .055) / 1.055) ** 2.4)


class MujocoSession:
    """One persistent worker and one scene for a deterministic pose scan."""

    def __init__(self, config: dict, config_path: Path, work_dir: Path):
        validate_mujoco_config(config, config_path=config_path, allow_contact_tangent=True)
        check_mujoco_runtime_paths(config, config_path)
        runtime = dict(config['mujoco'])
        python = _path(runtime['python_executable'], config_path)
        if runtime['calibrate_file'] != 'bundled:g1-ws_table':
            runtime['calibrate_file'] = str(_path(runtime['calibrate_file'], config_path))
        worker_config = dict(legacy_flat_config(config), mujoco=runtime)
        script = Path(__file__).resolve().parents[1] / 'integrations' / 'xense_mujoco_runtime' / 'worker.py'
        work_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = work_dir
        self.stderr_path = work_dir / 'worker.stderr.log'
        self.stderr = self.stderr_path.open('w', encoding='utf-8')
        self.process = subprocess.Popen([str(python), '-u', str(script)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, encoding='utf-8',
                                        bufsize=1)
        self.responses: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_responses, daemon=True)
        self.reader.start()
        self.serial = 0
        try:
            self.handshake = self._request('open', config=worker_config)
        except Exception:
            self.close()
            raise

    def _read_responses(self) -> None:
        for line in self.process.stdout:
            if line.startswith('XENSE_JSON:'):
                self.responses.put(line)
        self.responses.put(None)

    def _request(self, operation: str, **kwargs) -> dict:
        self.serial += 1
        request = dict(id=self.serial, protocol_version=PROTOCOL_VERSION,
                       operation=operation, **kwargs)
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        try:
            line = self.responses.get(timeout=120)
        except queue.Empty as exc:
            self.process.kill()
            raise TimeoutError(f'MuJoCo worker timed out during {operation}; '
                               f'see {self.stderr_path}') from exc
        if line is None:
            raise RuntimeError(f'MuJoCo worker exited; see {self.stderr_path}')
        reply = json.loads(line[len('XENSE_JSON:'):])
        if reply.get('id') != self.serial:
            raise RuntimeError('MuJoCo worker request/response mismatch')
        if not reply.get('ok'):
            raise RuntimeError(f"MuJoCo worker: {reply.get('error')}\n{reply.get('traceback')}")
        return reply['payload']

    def render(self, center_xy_mm, depth_mm: float) -> dict[str, np.ndarray]:
        path = self.work_dir / f'frame_{uuid.uuid4().hex}.npz'
        try:
            self._request('render', center_xy_mm=list(center_xy_mm),
                          depth_mm=float(depth_mm), output_path=str(path))
            with np.load(path, allow_pickle=False) as data:
                return {key: np.asarray(data[key]).copy() for key in data.files}
        finally:
            path.unlink(missing_ok=True)

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._request('close')
            except (BrokenPipeError, RuntimeError, TimeoutError):
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.reader.join(timeout=2)
        self.process.stdout.close()
        self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
