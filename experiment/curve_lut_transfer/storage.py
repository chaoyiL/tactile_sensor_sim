"""Atomic artifacts and content-verified per-pose checkpoints."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image


_REPLACE_RETRY_DELAYS = (.05, .1, .2, .4, .8)
_RETRYABLE_WINDOWS_ERRORS = (5, 32, 33)


def _atomic_replace(temp: Path, target: Path) -> None:
    """Replace a completed temporary artifact, retrying brief Windows file locks."""
    for attempt in range(len(_REPLACE_RETRY_DELAYS) + 1):
        try:
            temp.replace(target)
            return
        except PermissionError as exc:
            if os.name != 'nt' or getattr(exc, 'winerror', None) not in _RETRYABLE_WINDOWS_ERRORS:
                raise
            if attempt == len(_REPLACE_RETRY_DELAYS):
                context = (f'Atomic replacement failed: {temp} -> {target} '
                           f'after {attempt + 1} attempts')
                add_note = getattr(exc, 'add_note', None)
                if add_note is not None:
                    add_note(context)
                else:  # Python 3.10 compatibility; retain the original exception object/type.
                    exc.args = (f'{exc}; {context}',)
                raise
            time.sleep(_REPLACE_RETRY_DELAYS[attempt])


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()


def json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    _atomic_replace(temp, path)


def npz_write(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.npz.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    _atomic_replace(temp, path)


def load_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files if k != 'metadata'}


def save_frame(path: Path, frame: dict, execution: dict) -> list[Path]:
    """Persist the actual observation image, not a separately re-encoded preview."""
    path.mkdir(parents=True, exist_ok=True)
    def host(x):
        return x.detach().cpu().numpy()
    image = path / 'image.png'
    tmp = path / 'image.tmp.png'
    Image.fromarray(host(frame['rgb8'])).save(tmp)
    _atomic_replace(tmp, image)
    raw = path / 'render.npz'
    npz_write(raw, sensor_irradiance_rgb_w_m2=host(frame['sensor']),
              radiance_rgb_w_m2_sr=host(frame['radiance']), **{k: host(v) for k, v in frame['gt'].items()})
    meta = path / 'metadata.json'
    json_write(meta, dict(config=frame['config'], execution=execution,
                         observation='inverse_srgb(actual_saved_srgb8); material bilinear sampling'))
    return [image, raw, meta]


class Checkpoints:
    def __init__(self, root: Path, manifest: dict):
        self.root, self.manifest = root, manifest

    def save(self) -> None:
        json_write(self.root / 'manifest.json', self.manifest)

    def complete(self, key: str, paths: list[Path], **metadata) -> None:
        self.manifest['completed'][key] = dict(metadata, files={str(p.resolve()): digest(p) for p in paths})
        self.save()

    def verify(self) -> None:
        for key, entry in self.manifest['completed'].items():
            for value, expected in entry['files'].items():
                path = Path(value)
                if not path.is_file() or digest(path) != expected:
                    raise ValueError(f'Checkpoint {key} is missing/corrupted: {path}')

    def has(self, key: str) -> bool:
        return key in self.manifest['completed']
