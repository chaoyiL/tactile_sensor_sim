"""Resolve a saved LUT run for evaluation without consulting live defaults."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .backend_layout import canonical_backend, lut_result_dir
from .surface_coordinates import require_flat


def load_lut_result(path: str | Path, *, require_published: bool = True
                    ) -> tuple[Path, dict, str]:
    """Return result directory, saved manifest, and canonical backend.

    Evaluation uses the conditions archived with this run. Callers that accept
    diagnostic runs still inspect the saved publication status separately.
    """
    root = Path(path).expanduser().resolve()
    manifest_path = root / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    elif not require_published and (root / 'config.json').is_file():
        manifest = {'status': 'unpublished', 'stage': 'diagnostic',
                    'config': json.loads((root / 'config.json').read_text(encoding='utf-8'))}
    else:
        raise FileNotFoundError(f'LUT manifest is missing: {manifest_path}')
    if require_published and (manifest.get('status') != 'passed'
                              or manifest.get('stage') != 'completed'):
        raise ValueError(f'LUT calibration is not published: {manifest_path}')
    required = (('normal_lut.npz',) if require_published else ())
    for name in (*required, 'normal_lut_train.npz', 'normal_samples.npz'):
        if not (root / name).is_file():
            raise FileNotFoundError(f'LUT artifact is missing: {root / name}')
    config = manifest.get('config')
    if not isinstance(config, dict):
        raise ValueError(f'LUT manifest has no simulation config snapshot: {manifest_path}')
    # Older diagnostic runs can contain only a backend snapshot. Their saved
    # samples remain evaluable without reconstructing or assuming live geometry.
    if 'surface' in config:
        require_flat(config, f'LUT evaluation of {manifest_path}')
    backend = canonical_backend(manifest.get('backend', config.get('backend', 'analytic')))
    if require_published and not root.is_relative_to(lut_result_dir(backend)):
        raise ValueError(f'{root} is outside the {backend} LUT results directory '
                         f'{lut_result_dir(backend)}')
    signature = manifest.get('run_signature')
    if require_published and not signature:
        raise ValueError(f'LUT manifest lacks run_signature: {manifest_path}')
    if signature:
        artifacts = ['normal_lut_train.npz', 'normal_samples.npz']
        if (root / 'normal_lut.npz').is_file():
            artifacts.append('normal_lut.npz')
        for name in artifacts:
            source = root / name
            with np.load(source, allow_pickle=False) as archive:
                metadata = json.loads(str(archive['metadata']))
            if metadata.get('run_signature') != signature:
                raise ValueError(f'{source} does not match {manifest_path} run_signature')
            if canonical_backend(metadata.get('backend', backend)) != backend:
                raise ValueError(f'{source} does not match {manifest_path} backend')
    return root, manifest, backend
