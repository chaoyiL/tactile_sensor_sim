"""阶段计时、CPU 查询计时及原子会话报告。"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import math
from numbers import Integral, Real
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator
import uuid

from experiment.curve_lut_transfer.storage import json_write


Clock = Callable[[], float]
Synchronize = Callable[[], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_cuda_device(device: str) -> bool:
    return device == 'cuda' or device.startswith('cuda:')


class RunTimings:
    """Collect invocation-local timings and optionally persist atomic snapshots.

    Construction is memory-only. The default CUDA synchronizer consults only an
    already-imported ``torch`` module and never initializes CUDA itself.
    """

    FORMAT_VERSION = 1

    def __init__(self, device: str, run_mode: str, *,
                 clock: Clock | None = None,
                 synchronize: Synchronize | None = None) -> None:
        self._clock: Clock = clock or time.perf_counter
        self._custom_synchronize = synchronize
        self.device = ''
        self.run_mode = ''
        self._synchronize: Synchronize = lambda: None
        self.set_device(device)
        self.run_mode = self._require_text(run_mode, 'run_mode')

        self._created_at = self._clock()
        self._created_at_utc = _utc_now()
        self._session_id = (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                            + '_' + uuid.uuid4().hex[:8])
        self._root: Path | None = None
        self.session_path: Path | None = None
        self._entries: list[dict[str, Any]] = []
        self._queries: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def _require_text(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a nonempty string')
        return value

    @staticmethod
    def _default_cuda_synchronizer(device: str) -> Synchronize:
        def synchronize_if_initialized() -> None:
            torch = sys.modules.get('torch')
            if torch is None:
                return
            cuda = getattr(torch, 'cuda', None)
            is_initialized = getattr(cuda, 'is_initialized', None)
            if callable(is_initialized) and is_initialized():
                synchronize = getattr(cuda, 'synchronize', None)
                if callable(synchronize):
                    synchronize(device)

        return synchronize_if_initialized

    def set_device(self, device: str) -> None:
        """Set the device whose already-initialized CUDA work phase() waits for."""
        device = self._require_text(device, 'device')
        self.device = device
        if not _is_cuda_device(device):
            self._synchronize = lambda: None
        elif self._custom_synchronize is not None:
            self._synchronize = self._custom_synchronize
        else:
            self._synchronize = self._default_cuda_synchronizer(device)

    def attach(self, root: Path) -> Path:
        """Attach to a run directory and atomically save the initial snapshot."""
        resolved = Path(root).expanduser().resolve()
        if self._root is not None and resolved != self._root:
            raise RuntimeError(f'Timing session is already attached to {self._root}')
        self._root = resolved
        if self.session_path is None:
            self.session_path = resolved / 'runtime_sessions' / f'{self._session_id}.json'
        self._save_snapshot(status='running')
        return self.session_path

    @staticmethod
    def _seconds(value: Real, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f'{name} must be a finite nonnegative number')
        result = float(value)
        if not math.isfinite(result) or result < 0.:
            raise ValueError(f'{name} must be a finite nonnegative number')
        return result

    @staticmethod
    def _check_metadata(stage: str, condition: str | None,
                        metadata: dict[str, Any]) -> None:
        RunTimings._require_text(stage, 'stage')
        if condition is not None:
            RunTimings._require_text(condition, 'condition')
        reserved = {'stage', 'condition', 'started_at_utc', 'ended_at_utc',
                    'recorded_at_utc', 'seconds', 'status', 'error_type',
                    'error_message', 'sync_error_type', 'sync_error_message'}
        collision = reserved.intersection(metadata)
        if collision:
            raise ValueError(f'Timing metadata uses reserved fields: {sorted(collision)}')

    def _new_entry(self, stage: str, condition: str | None,
                   metadata: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {
            'stage': stage,
            'condition': condition,
            'started_at_utc': None,
            'ended_at_utc': None,
            'seconds': 0.,
            'status': 'running',
        }
        entry.update(metadata)
        self._entries.append(entry)
        return entry

    @staticmethod
    def _exception_fields(entry: dict[str, Any], error: BaseException,
                          prefix: str = '') -> None:
        entry[f'{prefix}error_type'] = type(error).__name__
        entry[f'{prefix}error_message'] = str(error)

    def _save_snapshot(self, *, status: str = 'running') -> None:
        if self.session_path is not None:
            json_write(self.session_path, self.report(status=status))

    @contextmanager
    def phase(self, stage: str, *, condition: str | None = None,
              **metadata: Any) -> Iterator[dict[str, Any]]:
        """Measure a wall-clock stage, waiting for pending CUDA work at its edges.

        The initial synchronization is outside the measured interval. The final
        synchronization is inside it, so queued work launched by the phase is
        included. CPU devices use a no-op synchronizer by default.
        """
        self._check_metadata(stage, condition, metadata)
        try:
            self._synchronize()
        except BaseException as error:
            entry = self._new_entry(stage, condition, metadata)
            entry.update(started_at_utc=_utc_now(), ended_at_utc=_utc_now(),
                         seconds=0., status='failed')
            self._exception_fields(entry, error)
            try:
                self._save_snapshot()
            except Exception:
                pass
            raise

        started = self._clock()
        entry = self._new_entry(stage, condition, metadata)
        entry['started_at_utc'] = _utc_now()
        try:
            yield entry
        except BaseException as error:
            sync_error: BaseException | None = None
            try:
                self._synchronize()
            except BaseException as caught:
                sync_error = caught
            ended = self._clock()
            entry.update(ended_at_utc=_utc_now(),
                         seconds=max(0., ended - started), status='failed')
            self._exception_fields(entry, error)
            if sync_error is not None:
                self._exception_fields(entry, sync_error, prefix='sync_')
            try:
                self._save_snapshot()
            except Exception:
                # Preserve the exception raised by the phase body.
                pass
            raise
        else:
            try:
                self._synchronize()
            except BaseException as error:
                ended = self._clock()
                entry.update(ended_at_utc=_utc_now(),
                             seconds=max(0., ended - started), status='failed')
                self._exception_fields(entry, error)
                try:
                    self._save_snapshot()
                except Exception:
                    # Preserve the synchronization exception that ended the phase.
                    pass
                raise
            ended = self._clock()
            entry.update(ended_at_utc=_utc_now(),
                         seconds=max(0., ended - started), status='completed')
            self._save_snapshot()

    def record(self, stage: str, seconds: Real, **metadata: Any) -> dict[str, Any]:
        """Record an interval measured elsewhere, such as preflight or startup."""
        self._check_metadata(stage, None, metadata)
        elapsed = self._seconds(seconds, 'seconds')
        recorded_at = datetime.now(timezone.utc)
        entry: dict[str, Any] = {
            'stage': stage,
            'condition': None,
            'started_at_utc': (recorded_at - timedelta(seconds=elapsed)).isoformat(),
            'ended_at_utc': recorded_at.isoformat(),
            'recorded_at_utc': recorded_at.isoformat(),
            'seconds': elapsed,
            'status': 'completed',
        }
        entry.update(metadata)
        self._entries.append(entry)
        self._save_snapshot()
        return entry

    def record_query(self, method: str, condition: str, seconds: Real,
                     sample_count: Integral) -> None:
        """Accumulate CPU/NumPy prediction-query time without writing per call."""
        method = self._require_text(method, 'method')
        condition = self._require_text(condition, 'condition')
        elapsed = self._seconds(seconds, 'seconds')
        if isinstance(sample_count, bool) or not isinstance(sample_count, Integral) \
                or sample_count < 0:
            raise ValueError('sample_count must be a nonnegative integer')
        key = (method, condition)
        summary = self._queries.setdefault(key, {
            'method': method,
            'condition': condition,
            'seconds': 0.,
            'calls': 0,
            'sample_count': 0,
        })
        summary['seconds'] += elapsed
        summary['calls'] += 1
        summary['sample_count'] += int(sample_count)

    def report(self, status: str = 'running') -> dict[str, Any]:
        """Return a JSON-ready snapshot for the current process invocation."""
        status = self._require_text(status, 'status')
        stage_seconds: dict[str, float] = {}
        for entry in self._entries:
            stage = entry['stage']
            stage_seconds[stage] = stage_seconds.get(stage, 0.) + float(entry['seconds'])
        query_summary = {
            f'{method}/{condition}': dict(values)
            for (method, condition), values in sorted(self._queries.items())
        }
        updated_at = _utc_now()
        return {
            'format_version': self.FORMAT_VERSION,
            'session_id': self._session_id,
            'status': status,
            'status_scope': (
                'Status describes the measured phases only; final runtime/checkpoint '
                'writes are outside this scope. Run completion is recorded in manifest.json.'),
            'run_mode': self.run_mode,
            'device': self.device,
            'current_invocation_only': True,
            'time_range_description': (
                'wall_seconds runs from RunTimings construction to report generation; '
                'it excludes earlier process/import startup.'),
            'phase_timing_scope': (
                'Phase wall time includes the phase body, including its reads, writes, '
                'and checkpoints; initial CUDA synchronization is outside the interval, '
                'final synchronization is inside it.'),
            'started_at_utc': self._created_at_utc,
            'updated_at_utc': updated_at,
            'wall_seconds': max(0., self._clock() - self._created_at),
            'stage_seconds': dict(sorted(stage_seconds.items())),
            'entries': [dict(entry) for entry in self._entries],
            'query_summary': query_summary,
            'query_timing_scope': (
                'CPU/NumPy _predict_method prediction and LUT query only; excludes '
                'sample loading, metric aggregation, bootstrap, plotting, and output. '
                'Evaluation-domain subsets reuse predictions, so subset timings are not '
                'additive. Query totals overlap their enclosing phase and must not be '
                'added to stage_seconds.'),
            'full_run_note': (
                'A full from-zero runtime is valid only for a fresh output with no '
                'reused checkpoints; resumed timings describe this invocation only.'),
        }

    def finish(self, status: str = 'completed') -> dict[str, Any]:
        """Create and atomically save the final report when attached."""
        report = self.report(status=status)
        if self.session_path is not None:
            json_write(self.session_path, report)
        return report
