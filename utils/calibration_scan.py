"""Deterministic depth and material-position scan generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ScanPose:
    pose_id: int
    image_key: str
    depth_index: int
    row_index: int
    column_index: int
    center_xy_mm: tuple[float, float]
    depth_mm: float
    contact_radius_mm: float

    def to_dict(self) -> dict:
        value = asdict(self)
        value['center_xy_mm'] = list(self.center_xy_mm)
        return value


def closed_axis(start: float, stop: float, step: float, *, include_endpoint: bool = True) -> list[float]:
    values = (float(start), float(stop), float(step))
    if not all(math.isfinite(value) for value in values) or step <= 0 or stop < start:
        raise ValueError('Axis bounds must be finite, ordered, and use a positive step')
    tolerance = 1e-10 * max(1., abs(start), abs(stop), abs(step))
    if abs(stop - start) <= tolerance:
        return [float(stop)]
    quotient = (stop - start) / step
    count = math.floor(quotient + 1e-10)
    result = [float(start + index * step) for index in range(count + 1)]
    result = [value for value in result if value <= stop + tolerance]
    if result and abs(result[-1] - stop) <= tolerance:
        result[-1] = float(stop)
    elif include_endpoint:
        result.append(float(stop))
    return result


def contact_radius_for(radius_mm: float, depth_mm: float) -> float:
    if not math.isfinite(radius_mm) or not math.isfinite(depth_mm) \
            or radius_mm <= 0 or depth_mm <= 0 or depth_mm >= radius_mm:
        raise ValueError('Require finite 0 < depth_mm < radius_mm')
    return math.sqrt(2 * radius_mm * depth_mm - depth_mm * depth_mm)


def build_scan_poses(config: dict) -> list[ScanPose]:
    try:
        calibration = config['calibration']
        scan = calibration['scan']
        radius = float(calibration['sphere_radius_mm'])
        width, height = map(float, config['surface']['size_mm'])
        step_x, step_y = map(float, scan['xy_step_mm'])
        include = scan.get('include_endpoints', True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Invalid calibration scan configuration') from exc
    if type(include) is not bool:
        raise ValueError('calibration.scan.include_endpoints must be boolean')
    if not all(math.isfinite(value) and value > 0
               for value in (radius, width, height, step_x, step_y)):
        raise ValueError('Scan radius, surface size, and XY steps must be positive and finite')
    depths = closed_axis(float(scan['depth_min_mm']), float(scan['depth_max_mm']),
                         float(scan['depth_step_mm']), include_endpoint=include)
    if any(depth <= 0 or depth >= radius for depth in depths):
        raise ValueError('All scan depths must satisfy 0 < depth < sphere radius')

    records = []
    temporary = []
    for depth_index, depth in enumerate(depths):
        contact_radius = contact_radius_for(radius, depth)
        if 2 * contact_radius > min(width, height) + 1e-10 * max(width, height, 1.):
            raise ValueError(f'Contact disc at depth {depth:g} mm does not fit on the surface')
        xs = closed_axis(-width / 2 + contact_radius, width / 2 - contact_radius,
                         step_x, include_endpoint=include)
        ys = closed_axis(-height / 2 + contact_radius, height / 2 - contact_radius,
                         step_y, include_endpoint=include)
        for row_index, y in enumerate(ys):
            for column_index, x in enumerate(xs):
                temporary.append((depth_index, row_index, column_index, x, y,
                                  depth, contact_radius))
    key_width = max(6, len(str(max(0, len(temporary) - 1))))
    for pose_id, row in enumerate(temporary):
        depth_index, row_index, column_index, x, y, depth, contact_radius = row
        records.append(ScanPose(pose_id, f'{pose_id:0{key_width}d}', depth_index,
                                row_index, column_index, (x, y), depth, contact_radius))
    if not records:
        raise ValueError('Calibration scan generated no poses')
    return records

