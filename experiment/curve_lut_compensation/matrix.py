"""Local normal-response matrices and rank-aware colour compensation maps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LocalResponse:
    """B, D and diagnostics evaluated at no-contact reference material points."""

    B: np.ndarray
    D: np.ndarray
    active_light_mask: np.ndarray
    valid: np.ndarray
    singular_values: np.ndarray
    rank: np.ndarray
    condition_number: np.ndarray
    finite_difference_D: np.ndarray
    finite_difference_error: np.ndarray


def _as_finite_array(value: Any, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < len(shape_tail) or array.shape[-len(shape_tail):] != shape_tail:
        raise ValueError(f'{name} must have trailing shape {shape_tail}')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite')
    return array


def _rank_diagnostics(singular_values: np.ndarray, rcond: float) -> tuple[np.ndarray, np.ndarray]:
    largest = singular_values[..., 0]
    threshold = rcond * largest
    rank = np.sum(singular_values > threshold[..., None], axis=-1).astype(np.int8)
    smallest = singular_values[..., -1]
    condition = np.divide(largest, smallest, out=np.full_like(largest, np.inf),
                          where=smallest > threshold)
    return rank, condition


def compute_local_response_matrix(model, positions_ref_mm: np.ndarray,
                                  reference_normals_geom: np.ndarray,
                                  tangent_basis: np.ndarray, *,
                                  state_config: dict[str, Any] | None = None,
                                  slope_step: float = 1e-4,
                                  rcond: float = 1e-10) -> LocalResponse:
    """Compute Eq. (15)-(17) at no-contact reference points.

    ``positions_ref_mm`` has shape ``(N,3)`` in world millimetres;
    ``reference_normals_geom`` is ``n0`` with shape ``(N,3)``; and
    ``tangent_basis`` is the oriented orthonormal material frame ``E`` with
    shape ``(N,3,2)``. The selected ``state_config`` must be no-contact. The
    renderer's view-side sign maps geometric normals to optical normals, so the
    supported compensation convention requires ``view_side_z == -1`` and
    therefore ``n_o(0)=-n0``.

    The method holds point position, light activity and visibility fixed while
    differentiating the normal. ``B`` is the full exposure-response matrix
    before Reinhard; ``D`` includes the Reinhard derivative.
    """
    if not np.isfinite(slope_step) or slope_step <= 0:
        raise ValueError('slope_step must be finite and positive')
    if not np.isfinite(rcond) or rcond < 0:
        raise ValueError('rcond must be finite and nonnegative')
    points_np = _as_finite_array(positions_ref_mm, (3,), 'positions_ref_mm')
    normals_np = _as_finite_array(reference_normals_geom, (3,), 'reference_normals_geom')
    basis_np = _as_finite_array(tangent_basis, (3, 2), 'tangent_basis')
    if points_np.ndim != 2 or normals_np.shape != points_np.shape or basis_np.shape != (len(points_np), 3, 2):
        raise ValueError('expected points/normals (N,3) and tangent_basis (N,3,2)')
    norm = np.linalg.norm(normals_np, axis=-1)
    if not np.allclose(norm, 1., rtol=0., atol=1e-6):
        raise ValueError('reference_normals_geom must be unit length')
    gram = np.swapaxes(basis_np, -1, -2) @ basis_np
    if not np.allclose(gram, np.eye(2), rtol=0., atol=1e-6) \
            or not np.allclose(np.einsum('ni,nij->nj', normals_np, basis_np), 0., rtol=0., atol=1e-6):
        raise ValueError('tangent_basis must be orthonormal and perpendicular to reference_normals_geom')
    state = model._validate_state(state_config)
    if float(state['indentation']['depth_mm']) != 0.:
        raise ValueError('local response matrices require a no-contact state_config (indentation.depth_mm=0)')
    if float(state['surface']['view_side_z']) != -1.:
        raise ValueError('local response matrices require view_side_z=-1 so n_optical=-n0')

    radiance, camera_linear, _, _, active, B_tensor = model._evaluate_tensors(
        points_np, normals_np, state_config=state, need_linear_matrix=True)
    if B_tensor is None:
        raise RuntimeError('photometry backend did not return the local response matrix')
    torch = model.torch
    B_tensor = B_tensor.to(torch.float64)
    n0 = torch.as_tensor(normals_np, device=model.device, dtype=torch.float64)
    E = torch.as_tensor(basis_np, device=model.device, dtype=torch.float64)
    Aref = torch.clamp((-torch.matmul(B_tensor, n0[..., None]).squeeze(-1)), min=0.)
    tone_slope = torch.where(Aref > 0., (1. + Aref) ** -2, torch.zeros_like(Aref))
    D_tensor = tone_slope[..., :, None] * torch.matmul(B_tensor, E)

    # Validate B against the same renderer evaluation at beta=0. This also
    # catches radiometry/domain drift between the matrix and point kernels.
    camera_from_B = torch.clamp(-torch.matmul(B_tensor, n0[..., None]).squeeze(-1), min=0.)
    reference_rgb = camera_linear.to(torch.float64)
    camera_consistency = torch.amax(torch.abs(camera_from_B - reference_rgb), dim=-1)
    valid = torch.isfinite(B_tensor).all(dim=(-1, -2)) & torch.isfinite(D_tensor).all(dim=(-1, -2))
    valid &= camera_consistency <= 2e-5 * (1. + torch.amax(torch.abs(reference_rgb), dim=-1))
    camera_ray = torch.as_tensor(points_np, device=model.device, dtype=torch.float64) \
        - model.camera_position.to(torch.float64)
    valid &= torch.sum(camera_ray * model.camera_forward.to(torch.float64), dim=-1) > 0.

    # Central finite differences of the fixed-activity local model are cheap:
    # they reuse B and do not trace any additional light/shadow rays.
    eye = torch.eye(2, device=model.device, dtype=torch.float64)
    fd_columns = []
    for axis in range(2):
        beta = eye[axis] * float(slope_step)
        plus_normal = (-n0 + torch.matmul(E, beta.expand(len(points_np), 2)[..., None]).squeeze(-1)) \
            / torch.sqrt(1. + torch.sum(beta * beta))
        minus_normal = (-n0 - torch.matmul(E, beta.expand(len(points_np), 2)[..., None]).squeeze(-1)) \
            / torch.sqrt(1. + torch.sum(beta * beta))
        plus_a = torch.clamp(torch.matmul(B_tensor, plus_normal[..., None]).squeeze(-1), min=0.)
        minus_a = torch.clamp(torch.matmul(B_tensor, minus_normal[..., None]).squeeze(-1), min=0.)
        plus_tone = plus_a / (1. + plus_a)
        minus_tone = minus_a / (1. + minus_a)
        fd_columns.append(((plus_tone - minus_tone) / (2. * float(slope_step))))
    fd_tensor = torch.stack(fd_columns, dim=-1)
    fd_error_tensor = torch.linalg.vector_norm(fd_tensor - D_tensor, dim=(-2, -1))
    singular_values = torch.linalg.svdvals(D_tensor)
    singular_np = singular_values.detach().cpu().numpy()
    rank, condition = _rank_diagnostics(singular_np, rcond)
    valid_np = valid.detach().cpu().numpy().astype(bool)
    valid_np &= rank == 2
    return LocalResponse(B=B_tensor.detach().cpu().numpy(), D=D_tensor.detach().cpu().numpy(),
                         active_light_mask=active.detach().cpu().numpy().astype(bool),
                         valid=valid_np,
                         singular_values=singular_np,
                         rank=rank,
                         condition_number=condition,
                         finite_difference_D=fd_tensor.detach().cpu().numpy(),
                         finite_difference_error=fd_error_tensor.detach().cpu().numpy())


def _device_tensor(value: np.ndarray, torch, device):
    if hasattr(value, 'detach'):
        tensor = value.to(device=device, dtype=torch.float64)
    else:
        tensor = torch.as_tensor(np.array(value, dtype=np.float64, copy=True), device=device,
                                 dtype=torch.float64)
    return tensor


def _host(tensor):
    return tensor.detach().cpu().numpy()


def build_compensation_matrices(D0: np.ndarray, Dq: np.ndarray,
                                ridge_lambda: float, *, rcond: float = 1e-10,
                                device: str = 'cuda') -> dict[str, np.ndarray]:
    """Build Eq. (18), (21), and (22) maps with explicit rank diagnostics.

    A rank-deficient target response is never represented by zero-filled maps:
    unregularized transforms are NaN and invalid. Ridge transforms are returned
    as the defined regularized algebraic solution, but ``valid_ridge`` remains
    false when either response lacks both observable slope directions.
    """
    if type(ridge_lambda) not in (int, float) or not np.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError('ridge_lambda must be finite and positive')
    if not np.isfinite(rcond) or rcond < 0:
        raise ValueError('rcond must be finite and nonnegative')
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('GPU matrix compensation requires the gpu extra (PyTorch CUDA)') from exc
    try:
        torch_device = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f'Invalid matrix device {device!r}') from exc
    if torch_device.type not in ('cuda', 'cpu'):
        raise ValueError('matrix device must be CUDA or explicit CPU fixture')
    if torch_device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested for compensation matrices, but PyTorch CUDA is unavailable')
        try:
            torch.empty(1, device=torch_device).add_(1)
            torch.cuda.synchronize(torch_device)
        except Exception as exc:
            raise RuntimeError(f'CUDA device {torch_device} cannot execute matrix operations') from exc

    target0_np = np.asarray(D0, dtype=np.float64)
    targetq_np = np.asarray(Dq, dtype=np.float64)
    if target0_np.shape[-2:] != (3, 2) or targetq_np.shape[-2:] != (3, 2):
        raise ValueError('D0 and Dq must have trailing shape (3,2)')
    try:
        leading = np.broadcast_shapes(target0_np.shape[:-2], targetq_np.shape[:-2])
    except ValueError as exc:
        raise ValueError('D0 and Dq leading shapes must broadcast') from exc
    D0_t = _device_tensor(np.broadcast_to(target0_np, (*leading, 3, 2)), torch, torch_device)
    Dq_t = _device_tensor(np.broadcast_to(targetq_np, (*leading, 3, 2)), torch, torch_device)
    finite = torch.isfinite(D0_t).all(dim=(-1, -2)) & torch.isfinite(Dq_t).all(dim=(-1, -2))
    # Replace invalid entries only for the batched SVD/solve; their outputs are
    # overwritten with NaN below and remain explicitly invalid.
    safe0 = torch.where(finite[..., None, None], D0_t, torch.zeros_like(D0_t))
    safeq = torch.where(finite[..., None, None], Dq_t, torch.zeros_like(Dq_t))
    Uq, sq, Vhq = torch.linalg.svd(safeq, full_matrices=False)
    U0, s0, Vh0 = torch.linalg.svd(safe0, full_matrices=False)
    rankq_t = torch.sum(sq > rcond * sq[..., :1], dim=-1)
    rank0_t = torch.sum(s0 > rcond * s0[..., :1], dim=-1)
    target_full = finite & (rankq_t == 2)
    both_full = target_full & (rank0_t == 2)
    # Dq+ = V diag(1/s) U^T, using SVD thresholding only where full rank.
    inv_sq = torch.where(sq > rcond * sq[..., :1], 1. / torch.clamp(sq, min=torch.finfo(sq.dtype).tiny),
                         torch.zeros_like(sq))
    Dq_plus = torch.matmul(Vhq.transpose(-2, -1) * inv_sq[..., None, :], Uq.transpose(-2, -1))
    identity = torch.eye(3, device=torch_device, dtype=torch.float64)
    C_project = torch.matmul(safe0, Dq_plus)
    C_keep = identity + torch.matmul(safe0 - safeq, Dq_plus)
    R_direct = Dq_plus

    eye2 = torch.eye(2, device=torch_device, dtype=torch.float64)
    ridge_system = torch.matmul(safeq.transpose(-2, -1), safeq) + float(ridge_lambda) * eye2
    # Solve the transposed 2x2 system in a batch for numerical stability.
    R_direct_ridge = torch.linalg.solve(ridge_system, safeq.transpose(-2, -1))
    C_project_ridge = torch.matmul(safe0, R_direct_ridge)
    C_keep_ridge = identity + torch.matmul(safe0 - safeq, R_direct_ridge)

    nan = float('nan')
    C_project = torch.where(target_full[..., None, None], C_project,
                            torch.full_like(C_project, nan))
    C_keep = torch.where(target_full[..., None, None], C_keep,
                         torch.full_like(C_keep, nan))
    R_direct = torch.where(target_full[..., None, None], R_direct,
                           torch.full_like(R_direct, nan))
    C_project_ridge = torch.where(finite[..., None, None], C_project_ridge,
                                  torch.full_like(C_project_ridge, nan))
    C_keep_ridge = torch.where(finite[..., None, None], C_keep_ridge,
                               torch.full_like(C_keep_ridge, nan))
    R_direct_ridge = torch.where(finite[..., None, None], R_direct_ridge,
                                 torch.full_like(R_direct_ridge, nan))
    sigma = torch.where(finite[..., None], sq, torch.full_like(sq, nan))
    rankq = torch.where(finite, rankq_t, torch.full_like(rankq_t, -1)).to(torch.int8)
    smallest = sq[..., -1]
    condition = torch.where((smallest > rcond * sq[..., 0]) & finite,
                            sq[..., 0] / torch.clamp(smallest, min=torch.finfo(sq.dtype).tiny),
                            torch.full_like(smallest, float('inf')))
    condition = torch.where(finite, condition, torch.full_like(condition, nan))
    valid_ridge = both_full
    return dict(C_project=_host(C_project), C_keep=_host(C_keep), R_direct=_host(R_direct),
                C_project_ridge=_host(C_project_ridge), C_keep_ridge=_host(C_keep_ridge),
                R_direct_ridge=_host(R_direct_ridge),
                valid=_host(target_full).astype(bool), valid_project=_host(target_full).astype(bool),
                valid_keep=_host(target_full).astype(bool), valid_direct=_host(target_full).astype(bool),
                valid_ridge=_host(valid_ridge).astype(bool),
                rank=_host(rankq), rank_D0=torch.where(finite, rank0_t,
                    torch.full_like(rank0_t, -1)).to(torch.int8).detach().cpu().numpy(),
                effective_rank=_host(rankq), singular_values=_host(sigma),
                singular_values_D0=_host(torch.where(finite[..., None], s0,
                    torch.full_like(s0, nan))), condition_number=_host(condition))

