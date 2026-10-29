"""GPU-first finite-slope response and inversion for curve LUT compensation.

The response field freezes material position, camera geometry, and the
reference light/shadow visibility.  Its 3-by-3 ``B`` matrix maps an optical
normal to exposure-scaled linear camera RGB before Reinhard tone mapping; it
is a physical light-response matrix, not the retired inter-configuration
``D`` compensation matrix.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from utils.renderer_gpu import require_torch
from utils.surface_coordinates import normal_direction
from .transfer import reference_geometry


# Keep these lists explicit. A field may have N samples and L lights with N == L.
_POINT_TENSOR_KEYS = (
    'position', 'normal', 'basis', 'B', 'background', 'camera_factor',
    'camera_post_factor',
    'active', 'light_visible', 'light_eligible', 'B_condition', 'valid',
)
def _host_array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    """Convert NumPy or torch input to a contiguous host array for geometry."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _tone_map(linear_rgb):
    """Apply the source model's nonnegative Reinhard display response."""
    torch = require_torch()
    positive = torch.clamp(linear_rgb, min=0.)
    return torch.clamp(positive / (1. + positive), min=0., max=1.)


def _camera_factor(model, points):
    torch = model.torch
    ray = points - model.camera_position
    ray = ray / torch.clamp(torch.linalg.vector_norm(ray, dim=-1, keepdim=True),
                            min=torch.finfo(ray.dtype).tiny)
    cos4 = torch.clamp(ray @ model.camera_forward, min=0.) ** 4
    aperture = (2. * model.pinhole_half_width_mm / model.focal_length_mm) ** 2
    return (cos4 * aperture).to(model.radiometry_dtype)


def _kernel_for_model(model):
    if model.radius_mm is None:
        from utils.renderer_kernels import light_block_factor
        return light_block_factor
    from utils.curve_renderer_kernels import curve_light_block_factor
    return curve_light_block_factor


def _kernel_metadata(model, state: dict[str, Any]) -> dict[str, Any]:
    common = dict(width=model.width_mm, height=model.height_mm,
                  radius=float(state['indentation']['radius_mm']),
                  shadows=bool(model.shadows),
                  unit_exponents=model.unit_light_exponents)
    if model.radius_mm is None:
        common['sign'] = float(normal_direction(state))
    else:
        common['curve_radius'] = float(model.radius_mm)
    return common


def build_response_field(model: Any, coordinates: Any, reference_config: dict[str, Any],
                         chunk_size: int = 8192) -> dict[str, Any]:
    """Build the fixed-position finite-slope response field on ``model.device``.

    ``coordinates`` are the two material coordinates consumed by
    :func:`reference_geometry`.  The returned pointwise tensors stay on the
    photometry model's device and retain its configured geometry/radiometry
    dtypes.  Visibility and source activity are evaluated with the same light
    kernels used by :class:`PhotometryModel` at the no-contact reference pose.
    """
    torch = model.torch
    if not isinstance(reference_config, dict):
        raise TypeError('reference_config must be a physical configuration dictionary')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    state = model._validate_state(reference_config)
    if float(state['indentation']['depth_mm']) != 0.:
        raise ValueError('reference_config.indentation.depth_mm must be zero')

    coordinates_np = _host_array(coordinates)
    if coordinates_np.ndim != 2 or coordinates_np.shape[1] != 2 \
            or not np.isfinite(coordinates_np).all():
        raise ValueError('coordinates must be a finite array with shape (N,2)')
    position_np, normal_np, basis_np = reference_geometry(coordinates_np, model.radius_mm)
    device = model.device
    geometry_dtype = model.geometry_dtype
    radiometry_dtype = model.radiometry_dtype
    position = torch.as_tensor(position_np, device=device, dtype=geometry_dtype)
    normal = torch.as_tensor(normal_np, device=device, dtype=geometry_dtype)
    basis = torch.as_tensor(basis_np, device=device, dtype=geometry_dtype)
    count, light_count = len(position), len(model.light_points)

    matrix = torch.zeros((count, 3, 3), device=device, dtype=radiometry_dtype)
    irradiance = torch.zeros((count, 3), device=device, dtype=radiometry_dtype)
    active = torch.zeros((count, light_count), device=device, dtype=torch.bool)
    visible_all = torch.zeros_like(active)
    eligible = torch.zeros_like(active)
    camera_factor = _camera_factor(model, position)
    kernel = _kernel_for_model(model)
    metadata = _kernel_metadata(model, state)
    pose = model._pose(state)
    optical_normal = -normal  # source renderer convention: view_side_z == -1
    camera_rgb_from_light = ((model.light_powers * model.reflectance[None, :])
                             @ model.camera_response.T)
    post_factor = camera_factor * (model.exposure * 1e6 / math.pi)
    point_chunk = int(chunk_size)
    light_chunk = model.light_chunk_size

    for point_start in range(0, count, point_chunk):
        point_stop = min(point_start + point_chunk, count)
        points = position[point_start:point_stop]
        normals_opt = optical_normal[point_start:point_stop]
        camera_gain = post_factor[point_start:point_stop]
        shadow_origin = (points[:, None, :] + normals_opt[:, None, :] * 1e-6
                         if model.shadows else None)
        for light_start in range(0, light_count, light_chunk):
            light_stop = min(light_start + light_chunk, light_count)
            light = model.light_points[light_start:light_stop]
            emission = model.light_normals[light_start:light_stop]
            powers = model.light_powers[light_start:light_stop]
            exponent = model.light_exponents[light_start:light_stop]
            normalization = model.light_normalization[light_start:light_stop]
            factor, visible = kernel(
                points, normals_opt, shadow_origin, light, emission, powers,
                exponent, normalization, pose, **metadata)
            contribution = torch.sum(factor[..., None] * powers[None, :, :], dim=1)
            irradiance[point_start:point_stop] += contribution

            vector = light[None, :, :] - points[:, None, :]
            distance2 = torch.sum(vector * vector, dim=-1)
            distance = torch.sqrt(torch.clamp(distance2, min=torch.finfo(distance2.dtype).tiny))
            direction = vector / distance[..., None]
            cosine_emit = torch.clamp(torch.sum(-direction * emission[None, :, :], dim=-1), min=0.)
            reference_active = visible & (factor > 0.)
            light_eligible = visible & (cosine_emit > 0.)
            active[point_start:point_stop, light_start:light_stop] = reference_active
            visible_all[point_start:point_stop, light_start:light_stop] = visible
            eligible[point_start:point_stop, light_start:light_stop] = light_eligible

            emission_power = (cosine_emit if model.unit_light_exponents
                              else cosine_emit ** exponent[None, :])
            coefficient = normalization[None, :] * emission_power
            coefficient *= ((cosine_emit > 0.) * visible /
                            torch.clamp(distance2, min=torch.finfo(distance2.dtype).tiny))
            coefficient = coefficient.to(radiometry_dtype)
            light_rgb = camera_rgb_from_light[light_start:light_stop]
            gain = coefficient[..., None] * light_rgb[None, :, :]
            gain = gain * camera_gain[:, None, None]
            gain = gain * reference_active[..., None].to(radiometry_dtype)
            direction_rad = direction.to(radiometry_dtype)
            matrix_piece = torch.sum(gain[..., :, None] * direction_rad[..., None, :], dim=1)
            matrix[point_start:point_stop] += matrix_piece

    radiance = irradiance * model.reflectance[None, :] / math.pi * 1e6
    sensor = radiance * camera_factor[:, None]
    camera_linear = sensor @ model.camera_response.T * model.exposure
    camera_linear = torch.clamp(camera_linear, min=0.)
    background = _tone_map(camera_linear)

    if count:
        singular = torch.linalg.svdvals(matrix)
        smallest = singular[:, -1]
        condition = singular[:, 0] / torch.clamp(smallest, min=torch.finfo(smallest.dtype).tiny)
        condition = torch.where(smallest > 0., condition,
                                torch.full_like(condition, float('inf')))
    else:
        condition = torch.empty((0,), device=device, dtype=radiometry_dtype)
    finite_geometry = torch.isfinite(position).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
    camera_front = ((position - model.camera_position) @ model.camera_forward) > 0.
    valid = finite_geometry & camera_front & torch.isfinite(matrix).all(dim=(-1, -2))
    valid &= torch.isfinite(background).all(dim=-1) & torch.isfinite(camera_factor)

    return {
        'position': position,
        'normal': normal,
        'basis': basis,
        'B': matrix,
        'background': background,
        'camera_factor': camera_factor,
        'camera_post_factor': post_factor,
        'active': active,
        'light_visible': visible_all,
        'light_eligible': eligible,
        'B_condition': condition,
        'valid': valid,
        'light_points': model.light_points,
        'light_normals': model.light_normals,
        'light_powers': model.light_powers,
        'light_exponents': model.light_exponents,
        'light_normalization': model.light_normalization,
        'light_camera_rgb': camera_rgb_from_light,
        'camera_response': model.camera_response,
        'reflectance': model.reflectance,
        'camera_position': model.camera_position,
        'camera_forward': model.camera_forward,
        '_metadata': {
            'device': str(device),
            'geometry_dtype': str(geometry_dtype).split('.')[-1],
            'radiometry_dtype': str(radiometry_dtype).split('.')[-1],
            'light_chunk_size': light_chunk,
            'shadows': bool(model.shadows),
            'surface_radius_mm': model.radius_mm,
            'active_set_semantics': 'reference_shadow_visibility_and_current_receiver_cosine',
            'unit_light_exponents': bool(model.unit_light_exponents),
            'response': 'fixed_position_finite_slope_reinhard',
        },
    }


def select_response_field(field: Mapping[str, Any], indices: Any) -> dict[str, Any]:
    """Select point rows while retaining global light and camera tensors."""
    torch = require_torch()
    if 'position' not in field or not hasattr(field['position'], 'device'):
        raise ValueError('field must be a response field returned by build_response_field')
    position = field['position']
    if hasattr(indices, 'detach'):
        index_tensor = indices.to(device=position.device)
    else:
        index_tensor = torch.as_tensor(indices, device=position.device)
    if index_tensor.ndim != 1 or index_tensor.dtype == torch.bool \
            or index_tensor.dtype.is_floating_point or index_tensor.dtype.is_complex:
        raise ValueError('indices must be a one-dimensional integer array')
    index_tensor = index_tensor.to(dtype=torch.long)
    if index_tensor.numel():
        if bool(torch.any(index_tensor < 0)) or bool(torch.any(index_tensor >= len(position))):
            raise IndexError('indices contains a row outside the response field')
    selected = dict(field)
    for key in _POINT_TENSOR_KEYS:
        value = field.get(key)
        if value is not None:
            selected[key] = value.index_select(0, index_tensor)
    return selected


def _beta_tensor(field: Mapping[str, Any], value: Any, name: str):
    torch = require_torch()
    template = field['normal']
    result = torch.as_tensor(value, device=template.device, dtype=template.dtype)
    if result.shape == (2,):
        result = result.expand((len(template), 2))
    if result.shape != (len(template), 2):
        raise ValueError(f'{name} must have shape (N,2) or (2,)')
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f'{name} must be finite')
    return result


def _color_tensor(field: Mapping[str, Any], value: Any, name: str):
    torch = require_torch()
    template = field['B']
    result = torch.as_tensor(value, device=template.device, dtype=template.dtype)
    if result.shape != (len(template), 3):
        raise ValueError(f'{name} must have shape (N,3)')
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f'{name} must be finite')
    return result


def _weight_tensor(field: Mapping[str, Any], value: Any | None, work_dtype):
    torch = require_torch()
    count = len(field['B'])
    if value is None:
        return torch.ones((count, 3), device=field['B'].device, dtype=work_dtype)
    weight = torch.as_tensor(value, device=field['B'].device, dtype=work_dtype)
    if weight.shape == (3,):
        weight = weight.expand((count, 3))
    elif weight.ndim == 0:
        weight = weight.expand((count, 3))
    if weight.shape != (count, 3):
        raise ValueError('color_weight must be a nonnegative scalar, (3,), or (N,3)')
    if not bool(torch.isfinite(weight).all()) or bool(torch.any(weight < 0.)):
        raise ValueError('color_weight must be finite and nonnegative')
    return weight


def _response(field: Mapping[str, Any], beta, matrix=None):
    """Evaluate equation (12) and its raw tone-mapped color at ``beta``."""
    torch = require_torch()
    work_dtype = torch.promote_types(field['B'].dtype, beta.dtype)
    work_dtype = torch.promote_types(work_dtype, field['normal'].dtype)
    B = field['B'] if matrix is None else matrix
    B = B.to(dtype=work_dtype)
    normal = field['normal'].to(dtype=work_dtype)
    basis = field['basis'].to(dtype=work_dtype)
    beta = beta.to(dtype=work_dtype)
    scale = torch.sqrt(1. + torch.sum(beta * beta, dim=-1))
    tangent = torch.einsum('nci,ni->nc', basis, beta)
    optical_normal = (-normal + tangent) / scale[:, None]
    linear = torch.einsum('nij,nj->ni', B, optical_normal)
    tone = _tone_map(linear)
    background = field['background'].to(dtype=work_dtype)
    return tone - background, tone, linear, optical_normal


def _jacobian(field: Mapping[str, Any], beta, matrix=None):
    """Evaluate equation (15) for the fixed reference ``B`` branch."""
    torch = require_torch()
    work_dtype = torch.promote_types(field['B'].dtype, beta.dtype)
    work_dtype = torch.promote_types(work_dtype, field['normal'].dtype)
    B = field['B'] if matrix is None else matrix
    B = B.to(dtype=work_dtype)
    normal = field['normal'].to(dtype=work_dtype)
    basis = field['basis'].to(dtype=work_dtype)
    beta = beta.to(dtype=work_dtype)
    scale = torch.sqrt(1. + torch.sum(beta * beta, dim=-1))
    optical_normal = (-normal + torch.einsum('nci,ni->nc', basis, beta)) / scale[:, None]
    derivative_normal = basis / scale[:, None, None]
    derivative_normal -= (optical_normal[:, :, None] * beta[:, None, :]) / scale[:, None, None] ** 2
    linear = torch.einsum('nij,nj->ni', B, optical_normal)
    raw_jacobian = torch.einsum('nij,njk->nik', B, derivative_normal)
    tone_derivative = 1. / (1. + torch.clamp(linear, min=0.)) ** 2
    tone_derivative = tone_derivative * (linear > 0.).to(work_dtype)
    jacobian = raw_jacobian * tone_derivative[:, :, None]
    return jacobian, linear


def _validate_solver_settings(steps: int, damping: float, prior_weight: float,
                              max_step: float) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    for name, value in (('damping', damping), ('prior_weight', prior_weight), ('max_step', max_step)):
        if not math.isfinite(float(value)):
            raise ValueError(f'{name} must be finite')
    if damping < 0.:
        raise ValueError('damping must be nonnegative')
    if prior_weight < 0.:
        raise ValueError('prior_weight must be nonnegative')
    if max_step <= 0.:
        raise ValueError('max_step must be positive')


def _active_switch_summary(field: Mapping[str, Any], beta, *, chunk_size: int = 8192):
    """Compare candidate receiver cosines with the frozen reference active set."""
    torch = require_torch()
    count = len(beta)
    changed_by_point = torch.zeros((count,), device=beta.device, dtype=torch.bool)
    switch_count = torch.zeros((count,), device=beta.device, dtype=torch.int32)
    normal = field['normal']
    basis = field['basis']
    points = field['position']
    lights = field['light_points']
    active_reference = field['active']
    eligible = field['light_eligible']
    # Use wider light blocks than the renderer's shading chunk while bounding
    # temporary point-by-light arrays to roughly 262k pairs.
    light_chunk = max(int(field['_metadata']['light_chunk_size']), 32)
    chunk_size = min(chunk_size, max(1, 262144 // light_chunk))
    for point_start in range(0, count, chunk_size):
        point_stop = min(point_start + chunk_size, count)
        beta_part = beta[point_start:point_stop].to(dtype=normal.dtype)
        scale = torch.sqrt(1. + torch.sum(beta_part * beta_part, dim=-1))
        optical_normal = (-normal[point_start:point_stop] +
                          torch.einsum('nci,ni->nc', basis[point_start:point_stop], beta_part))
        optical_normal = optical_normal / scale[:, None]
        point_part = points[point_start:point_stop]
        for light_start in range(0, len(lights), light_chunk):
            light_stop = min(light_start + light_chunk, len(lights))
            vector = lights[None, light_start:light_stop, :] - point_part[:, None, :]
            distance = torch.linalg.vector_norm(vector, dim=-1).clamp_min(torch.finfo(vector.dtype).tiny)
            direction = vector / distance[..., None]
            cosine = torch.sum(optical_normal[:, None, :] * direction, dim=-1)
            active_now = eligible[point_start:point_stop, light_start:light_stop] & (cosine > 0.)
            different = active_now != active_reference[point_start:point_stop, light_start:light_stop]
            switch_count[point_start:point_stop] += different.sum(dim=-1, dtype=torch.int32)
        changed_by_point[point_start:point_stop] = switch_count[point_start:point_stop] > 0
    return changed_by_point, switch_count


def _matrix_for_beta(field: Mapping[str, Any], beta, *, point_chunk_size: int = 8192):
    """Form B for beta's receiver-cosine active set, with reference shadows frozen.

    The point and all source/receiver directions are evaluated in geometry
    dtype. Light contributions are accumulated in the configured radiometry
    dtype, matching the source photometry path. Temporary point-by-light
    tensors are bounded by chunking both axes.
    """
    torch = require_torch()
    normal = field['normal']
    basis = field['basis']
    beta_geometry = beta.to(dtype=normal.dtype)
    scale = torch.sqrt(1. + torch.sum(beta_geometry * beta_geometry, dim=-1))
    optical = (-normal + torch.einsum('nci,ni->nc', basis, beta_geometry)) / scale[:, None]

    count = len(beta_geometry)
    matrix = torch.zeros_like(field['B'])
    switch_count = torch.zeros((count,), device=normal.device, dtype=torch.int32)
    lights = field['light_points']
    emissions = field['light_normals']
    exponents = field['light_exponents']
    normalization = field['light_normalization']
    eligible_all = field['light_eligible']
    active_reference = field['active']
    light_rgb = field['light_camera_rgb']
    camera_gain = field['camera_post_factor']
    light_chunk = max(int(field['_metadata']['light_chunk_size']), 32)
    point_chunk_size = min(point_chunk_size, max(1, 262144 // light_chunk))
    unit_exponents = bool(field['_metadata']['unit_light_exponents'])
    radiometry_dtype = field['B'].dtype

    for point_start in range(0, count, point_chunk_size):
        point_stop = min(point_start + point_chunk_size, count)
        points = field['position'][point_start:point_stop]
        optical_part = optical[point_start:point_stop]
        gain_post = camera_gain[point_start:point_stop]
        for light_start in range(0, len(lights), light_chunk):
            light_stop = min(light_start + light_chunk, len(lights))
            vector = lights[None, light_start:light_stop, :] - points[:, None, :]
            distance2 = torch.sum(vector * vector, dim=-1)
            distance = torch.sqrt(torch.clamp(distance2, min=torch.finfo(distance2.dtype).tiny))
            direction = vector / distance[..., None]
            cosine_emit = torch.clamp(torch.sum(
                -direction * emissions[None, light_start:light_stop, :], dim=-1), min=0.)
            cosine_in = torch.sum(optical_part[:, None, :] * direction, dim=-1)
            eligible = eligible_all[point_start:point_stop, light_start:light_stop]
            active_now = eligible & (cosine_in > 0.)
            different = active_now != active_reference[point_start:point_stop,
                                                        light_start:light_stop]
            switch_count[point_start:point_stop] += different.sum(dim=-1, dtype=torch.int32)

            emission_power = cosine_emit if unit_exponents else cosine_emit ** exponents[
                None, light_start:light_stop]
            coefficient = normalization[None, light_start:light_stop] * emission_power
            coefficient *= eligible / torch.clamp(
                distance2, min=torch.finfo(distance2.dtype).tiny)
            coefficient = coefficient.to(dtype=radiometry_dtype)
            gain = coefficient[..., None] * light_rgb[None, light_start:light_stop, :]
            gain = gain * gain_post[:, None, None]
            gain = gain * active_now[..., None].to(dtype=radiometry_dtype)
            direction = direction.to(dtype=radiometry_dtype)
            matrix[point_start:point_stop] += torch.sum(
                gain[..., :, None] * direction[..., None, :], dim=1)
    return matrix, switch_count


def _active_gauss_newton_impl(field: Mapping[str, Any], color_delta: Any, initial_beta: Any, *,
                              steps: int, damping: float, prior_beta: Any | None,
                              prior_weight: float, max_step: float, color_weight: Any | None,
                              diagnostics: bool) -> tuple[Any, dict[str, Any]]:
    """GN with a piecewise-current receiver active set and frozen visibility."""
    torch = require_torch()
    _validate_solver_settings(steps, damping, prior_weight, max_step)
    target = _color_tensor(field, color_delta, 'color_delta')
    beta = _beta_tensor(field, initial_beta, 'initial_beta')
    prior = beta if prior_beta is None else _beta_tensor(field, prior_beta, 'prior_beta')
    work_dtype = torch.promote_types(torch.promote_types(field['B'].dtype, field['normal'].dtype),
                                     target.dtype)
    beta_work = beta.to(dtype=work_dtype)
    prior_work = prior.to(dtype=work_dtype)
    target_work = target.to(dtype=work_dtype)
    weights = _weight_tensor(field, color_weight, work_dtype)
    eye = torch.eye(2, device=beta.device, dtype=work_dtype)[None, :, :]
    step_norm = torch.zeros((len(beta),), device=beta.device, dtype=work_dtype)
    solve_failed = torch.zeros((len(beta),), device=beta.device, dtype=torch.bool)

    for _ in range(steps):
        current_B, _ = _matrix_for_beta(field, beta_work)
        prediction, _, _, _ = _response(field, beta_work, current_B)
        residual = prediction - target_work
        jacobian, _ = _jacobian(field, beta_work, current_B)
        jacobian = jacobian.to(dtype=work_dtype)
        hessian = torch.einsum('nci,nc,ncj->nij', jacobian, weights, jacobian)
        gradient = torch.einsum('nci,nc,nc->ni', jacobian, weights, residual)
        if prior_weight:
            hessian = hessian + float(prior_weight) * eye
            gradient = gradient + float(prior_weight) * (beta_work - prior_work)
        effective_damping = max(float(damping), torch.finfo(work_dtype).eps)
        hessian = hessian + effective_damping * eye
        update, info = torch.linalg.solve_ex(hessian, -gradient[..., None])
        update = update[..., 0]
        failed = (info != 0) | ~torch.isfinite(update).all(dim=-1)
        update = torch.where(failed[:, None], torch.zeros_like(update), update)
        solve_failed |= failed
        norm = torch.linalg.vector_norm(update, dim=-1)
        factor = torch.clamp(float(max_step) /
                             torch.clamp(norm, min=torch.finfo(work_dtype).tiny), max=1.)
        update = update * factor[:, None]
        step_norm = torch.linalg.vector_norm(update, dim=-1)
        beta_work = beta_work + update

    final_B, switch_count = _matrix_for_beta(field, beta_work)
    final_prediction, tone, linear, optical_normal = _response(field, beta_work, final_B)
    residual_rgb = final_prediction - target_work
    active_changed = switch_count > 0
    if not diagnostics:
        return beta_work, {'diagnostics_enabled': False}
    return beta_work, {
        'diagnostics_enabled': True,
        'active_B': final_B,
        'optical_normal': optical_normal,
        'residual_rgb': residual_rgb,
        'residual_norm': torch.linalg.vector_norm(residual_rgb, dim=-1),
        'tone_mapped_rgb': tone,
        'camera_linear_rgb': torch.clamp(linear, min=0.),
        'active_changed': active_changed,
        'active_switch_count': switch_count,
        'active_switch_total': switch_count.sum(),
        'step_norm': step_norm,
        'solve_failed': solve_failed,
    }


def _gauss_newton_impl(field: Mapping[str, Any], color_delta: Any, initial_beta: Any, *,
                       steps: int, damping: float, prior_beta: Any | None,
                       prior_weight: float, max_step: float, color_weight: Any | None,
                       diagnostics: bool) -> tuple[Any, dict[str, Any]]:
    torch = require_torch()
    _validate_solver_settings(steps, damping, prior_weight, max_step)
    target = _color_tensor(field, color_delta, 'color_delta')
    beta = _beta_tensor(field, initial_beta, 'initial_beta')
    prior = beta if prior_beta is None else _beta_tensor(field, prior_beta, 'prior_beta')
    work_dtype = torch.promote_types(torch.promote_types(field['B'].dtype, field['normal'].dtype),
                                     target.dtype)
    beta_work = beta.to(dtype=work_dtype)
    prior_work = prior.to(dtype=work_dtype)
    target_work = target.to(dtype=work_dtype)
    weights = _weight_tensor(field, color_weight, work_dtype)
    eye = torch.eye(2, device=beta.device, dtype=work_dtype)[None, :, :]
    step_norm = torch.zeros((len(beta),), device=beta.device, dtype=work_dtype)
    solve_failed = torch.zeros((len(beta),), device=beta.device, dtype=torch.bool)

    for _ in range(steps):
        prediction, _, _, _ = _response(field, beta_work)
        residual = prediction - target_work
        jacobian, _ = _jacobian(field, beta_work)
        jacobian = jacobian.to(dtype=work_dtype)
        hessian = torch.einsum('nci,nc,ncj->nij', jacobian, weights, jacobian)
        gradient = torch.einsum('nci,nc,nc->ni', jacobian, weights, residual)
        if prior_weight:
            hessian = hessian + float(prior_weight) * eye
            gradient = gradient + float(prior_weight) * (beta_work - prior_work)
        # A zero configured damping is allowed; machine epsilon keeps a singular
        # two-by-two normal system from aborting the whole GPU batch.
        effective_damping = max(float(damping), torch.finfo(work_dtype).eps)
        hessian = hessian + effective_damping * eye
        rhs = -gradient[..., None]
        update, info = torch.linalg.solve_ex(hessian, rhs)
        update = update[..., 0]
        failed = (info != 0) | ~torch.isfinite(update).all(dim=-1)
        update = torch.where(failed[:, None], torch.zeros_like(update), update)
        solve_failed |= failed
        norm = torch.linalg.vector_norm(update, dim=-1)
        factor = torch.clamp(float(max_step) / torch.clamp(norm, min=torch.finfo(work_dtype).tiny), max=1.)
        update = update * factor[:, None]
        step_norm = torch.linalg.vector_norm(update, dim=-1)
        beta_work = beta_work + update

    if not diagnostics:
        return beta_work, {'diagnostics_enabled': False}
    prediction, tone, linear, _ = _response(field, beta_work)
    residual_rgb = prediction - target_work
    active_changed, switch_count = _active_switch_summary(field, beta_work)
    return beta_work, {
        'diagnostics_enabled': True,
        'residual_rgb': residual_rgb,
        'residual_norm': torch.linalg.vector_norm(residual_rgb, dim=-1),
        'tone_mapped_rgb': tone,
        'camera_linear_rgb': linear,
        'active_changed': active_changed,
        'active_switch_count': switch_count,
        'active_switch_total': switch_count.sum(),
        'step_norm': step_norm,
        'solve_failed': solve_failed,
    }


def predict_gauss_newton(field: Mapping[str, Any], color_delta: Any, initial_beta: Any, *,
                         steps: int = 3, damping: float = 1e-4,
                         prior_beta: Any | None = None, prior_weight: float = 0.,
                         max_step: float = 0.5, color_weight: Any | None = None,
                         diagnostics: bool = True) -> tuple[Any, dict[str, Any]]:
    """Run a fixed-position damped Gauss--Newton inversion of equation (14).

    The colors are signed, post-Reinhard linear-display RGB residuals, in the
    same domain as the transferred samples.  ``initial_beta`` should come from
    the source frozen LUT (or a previous estimate), never target truth labels.
    """
    return _gauss_newton_impl(
        field, color_delta, initial_beta, steps=steps, damping=damping,
        prior_beta=prior_beta, prior_weight=prior_weight, max_step=max_step,
        color_weight=color_weight, diagnostics=diagnostics)


def predict_active_gauss_newton(field: Mapping[str, Any], color_delta: Any,
                                initial_beta: Any, *, steps: int = 3,
                                damping: float = 1e-4,
                                prior_beta: Any | None = None,
                                prior_weight: float = 0., max_step: float = 0.5,
                                color_weight: Any | None = None,
                                diagnostics: bool = True) -> tuple[Any, dict[str, Any]]:
    """Run equation (14) GN while updating receiver activity at fixed ``X0``.

    At each GN step the current slope selects the positive receiver-cosine
    sources, and their physical light responses are re-summed into a current
    ``B`` for equations (12) and (15). Reference shadow visibility and source
    emission eligibility remain frozen because position is fixed. The prior
    center is constant across all steps; the target beta is never used to
    change it.
    """
    return _active_gauss_newton_impl(
        field, color_delta, initial_beta, steps=steps, damping=damping,
        prior_beta=prior_beta, prior_weight=prior_weight, max_step=max_step,
        color_weight=color_weight, diagnostics=diagnostics)


def predict_closed_form(field: Mapping[str, Any], color_delta: Any, initial_beta: Any, *,
                        color_epsilon: float = 1e-4, max_condition: float = 10000.,
                        minimum_normal_dot: float = 0.1, fallback_gn: bool = True,
                        gn_steps: int = 3, damping: float = 1e-4,
                        prior_beta: Any | None = None, prior_weight: float = 0.,
                        max_step: float = 0.5, color_weight: Any | None = None,
                        diagnostics: bool = True) -> tuple[Any, dict[str, Any]]:
    """Invert absolute contact colors with equation (13), optionally falling back.

    Unsafe channels and ill-conditioned/singular ``B`` rows are never silently
    discarded: with ``fallback_gn=True`` each such row is solved from its
    supplied ``initial_beta`` using the requested GN settings and the diagnostic
    records which rows fell back.  Active-set changes are reported separately;
    the diagnostic flag does not change beta or fallback decisions.
    """
    torch = require_torch()
    _validate_solver_settings(gn_steps, damping, prior_weight, max_step)
    for name, value in (('color_epsilon', color_epsilon), ('max_condition', max_condition),
                        ('minimum_normal_dot', minimum_normal_dot)):
        if not math.isfinite(float(value)):
            raise ValueError(f'{name} must be finite')
    if not 0. < color_epsilon < 0.5:
        raise ValueError('color_epsilon must be in (0,0.5)')
    if max_condition <= 1.:
        raise ValueError('max_condition must be greater than 1')
    if not 0. < minimum_normal_dot <= 1.:
        raise ValueError('minimum_normal_dot must be in (0,1]')

    delta = _color_tensor(field, color_delta, 'color_delta')
    initial = _beta_tensor(field, initial_beta, 'initial_beta')
    prior = initial if prior_beta is None else _beta_tensor(field, prior_beta, 'prior_beta')
    work_dtype = torch.promote_types(torch.promote_types(field['B'].dtype, field['normal'].dtype),
                                     delta.dtype)
    delta_work = delta.to(dtype=work_dtype)
    B = field['B'].to(dtype=work_dtype)
    normal = field['normal'].to(dtype=work_dtype)
    basis = field['basis'].to(dtype=work_dtype)
    background = field['background'].to(dtype=work_dtype)
    target_absolute = background + delta_work
    color_valid = torch.isfinite(target_absolute).all(dim=-1)
    color_valid &= torch.all((target_absolute >= color_epsilon) &
                             (target_absolute <= 1. - color_epsilon), dim=-1)

    matrix_finite = torch.isfinite(B).all(dim=(-1, -2))
    condition = field['B_condition'].to(dtype=work_dtype)
    condition_valid = matrix_finite & torch.isfinite(condition) & (condition <= max_condition)
    input_valid = field['valid'] & condition_valid
    identity = torch.eye(3, device=B.device, dtype=work_dtype)[None, :, :]
    safe_matrix = torch.where(input_valid[:, None, None], B, identity)
    safe_color = torch.clamp(torch.nan_to_num(target_absolute, nan=0.5, posinf=1., neginf=0.),
                             min=color_epsilon, max=1. - color_epsilon)
    absolute_linear = safe_color / (1. - safe_color)
    optical_recovered = torch.linalg.solve(safe_matrix, absolute_linear[..., None])[..., 0]
    denominator = torch.sum(normal * optical_recovered, dim=-1)
    recovered_norm = torch.linalg.vector_norm(optical_recovered, dim=-1)
    geometric_normal_dot = -denominator / torch.clamp(recovered_norm,
                                                       min=torch.finfo(work_dtype).tiny)
    orientation_valid = torch.isfinite(optical_recovered).all(dim=-1)
    orientation_valid &= (denominator < 0.) & (geometric_normal_dot >= minimum_normal_dot)
    safe_denominator = torch.where(denominator.abs() > torch.finfo(work_dtype).tiny,
                                   denominator, -torch.ones_like(denominator))
    tangent_component = torch.einsum('nci,nc->ni', basis, optical_recovered)
    candidate = -tangent_component / safe_denominator[:, None]
    candidate_valid = torch.isfinite(candidate).all(dim=-1)
    numeric_valid = color_valid & condition_valid & orientation_valid & candidate_valid & field['valid']
    candidate_for_activity = torch.where(numeric_valid[:, None], candidate,
                                         initial.to(dtype=work_dtype))
    # Activity agreement is a required closed-form assumption, not an optional
    # diagnostic. Compute it even when extra diagnostic output is disabled so
    # fallback behavior is identical in timed and accuracy runs.
    active_changed, switch_count = _active_switch_summary(field, candidate_for_activity)
    activity_stable = ~active_changed
    valid = numeric_valid & activity_stable
    candidate = torch.where(valid[:, None], candidate, initial.to(dtype=work_dtype))
    fallback_used = (~valid) & bool(fallback_gn)
    beta = candidate
    if fallback_gn and bool(torch.any(fallback_used)):
        fallback_indices = torch.nonzero(fallback_used, as_tuple=False).flatten()
        fallback_field = select_response_field(field, fallback_indices)
        fallback_prior = prior.index_select(0, fallback_indices)
        fallback_initial = initial.index_select(0, fallback_indices)
        fallback_delta = delta.index_select(0, fallback_indices)
        fallback_weight = None
        if color_weight is not None:
            full_weight = _weight_tensor(field, color_weight, work_dtype)
            fallback_weight = full_weight.index_select(0, fallback_indices)
        fallback_beta, _ = _gauss_newton_impl(
            fallback_field, fallback_delta, fallback_initial, steps=gn_steps,
            damping=damping, prior_beta=fallback_prior, prior_weight=prior_weight,
            max_step=max_step, color_weight=fallback_weight, diagnostics=False)
        beta = beta.clone()
        beta.index_copy_(0, fallback_indices, fallback_beta.to(dtype=work_dtype))

    if not diagnostics:
        return beta, {
            'diagnostics_enabled': False,
            'valid': valid,
            'fallback_used': fallback_used,
        }
    prediction, tone, linear, _ = _response(field, beta)
    residual_rgb = prediction - delta_work
    return beta, {
        'diagnostics_enabled': True,
        'valid': valid,
        'closed_form_valid': valid,
        'color_valid': color_valid,
        'condition_valid': condition_valid,
        'condition_number': condition,
        'orientation_valid': orientation_valid,
        'geometric_normal_dot': geometric_normal_dot,
        'activity_stable': activity_stable,
        'active_changed': active_changed,
        'active_switch_count': switch_count,
        'active_switch_total': switch_count.sum(),
        'fallback_used': fallback_used,
        'residual_rgb': residual_rgb,
        'residual_norm': torch.linalg.vector_norm(residual_rgb, dim=-1),
        'tone_mapped_rgb': tone,
        'camera_linear_rgb': linear,
    }
