"""Atomic artifacts and content-verified per-pose checkpoints."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


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
    temp.replace(path)


def npz_write(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.npz.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


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
    tmp.replace(image)
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
