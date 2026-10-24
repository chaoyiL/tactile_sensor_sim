"""Exact color/normal range counts with compiled tree traversal and image bitsets."""
from __future__ import annotations

import time

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree


def _flatten_tree(colors, normals, groups, outer_radius, tolerance, *, leafsize=64):
    chord = max(float(2 * np.sin(np.radians(tolerance) / 2)), 1e-12)
    # The large production scan benefits from fewer internal-node visits than
    # SciPy's default leaf size, while 64 still keeps exact leaf checks bounded.
    tree = cKDTree(np.column_stack((colors / max(outer_radius, 1e-12), normals / chord)),
                   leafsize=leafsize)
    nodes = [tree.tree]
    children = []
    for node in nodes:
        if node.split_dim < 0:
            children.append((-1, -1))
        else:
            children.append((len(nodes), len(nodes) + 1))
            nodes.extend((node.lesser, node.greater))
    order = tree.indices
    points = np.column_stack((colors[order], normals[order]))
    ordered_groups = groups[order]
    bounds = np.empty((len(nodes), 6, 2))
    ranges = np.array([(node.start_idx, node.end_idx) for node in nodes], np.int64)
    children = np.asarray(children, np.int32)
    words = (int(groups.max()) + 64) // 64
    bits = np.zeros((len(nodes), words), np.uint64)
    for i in range(len(nodes) - 1, -1, -1):
        left, right = children[i]
        if left < 0:
            first, last = ranges[i]
            bounds[i, :, 0] = points[first:last].min(axis=0)
            bounds[i, :, 1] = points[first:last].max(axis=0)
            ids = ordered_groups[first:last]
            np.bitwise_or.at(bits[i], ids // 64, np.uint64(1) << (ids % 64).astype(np.uint64))
        else:
            bounds[i, :, 0] = np.minimum(bounds[left, :, 0], bounds[right, :, 0])
            bounds[i, :, 1] = np.maximum(bounds[left, :, 1], bounds[right, :, 1])
            bits[i] = bits[left] | bits[right]
    return points, ordered_groups, bounds, ranges, children, bits


@njit(cache=True, inline='always')
def _distance_bounds(point, bounds):
    minimum = 0.
    maximum = 0.
    for axis in range(3):
        low, high = bounds[axis, 0], bounds[axis, 1]
        if point[axis] < low:
            minimum += (point[axis] - low) ** 2
        elif point[axis] > high:
            minimum += (point[axis] - high) ** 2
        maximum += max((point[axis] - low) ** 2, (point[axis] - high) ** 2)
    return minimum, maximum


@njit(cache=True, inline='always')
def _normal_dot_bounds(normal, bounds):
    minimum = 0.
    maximum = 0.
    for axis in range(3):
        a, b = normal[axis] * bounds[axis + 3, 0], normal[axis] * bounds[axis + 3, 1]
        minimum += min(a, b)
        maximum += max(a, b)
    return minimum, maximum


@njit(cache=True, inline='always')
def _normal_distance_bounds(normal, bounds):
    minimum, maximum = 0., 0.
    for axis in range(3):
        low, high = bounds[axis + 3, 0], bounds[axis + 3, 1]
        if normal[axis] < low:
            minimum += (normal[axis] - low) ** 2
        elif normal[axis] > high:
            minimum += (normal[axis] - high) ** 2
        maximum += max((normal[axis] - low) ** 2, (normal[axis] - high) ** 2)
    return minimum, maximum


@njit(cache=True, parallel=True)
def _build_normal_node_states(unique_normals, bounds, cosine, chord2):
    """Cache exact conservative cone rejection/acceptance for repeated normals."""
    words = (len(bounds) + 63) // 64
    rejected = np.zeros((len(unique_normals), words), np.uint64)
    accepted = np.zeros_like(rejected)
    for normal_id in prange(len(unique_normals)):
        normal = unique_normals[normal_id]
        for node in range(len(bounds)):
            minimum, maximum = _normal_dot_bounds(normal, bounds[node])
            distance_minimum, distance_maximum = _normal_distance_bounds(
                normal, bounds[node])
            word = node // 64
            bit = np.uint64(1) << np.uint64(node % 64)
            if maximum < cosine - 1e-12 or distance_minimum > chord2 + 1e-12:
                rejected[normal_id, word] |= bit
            elif minimum > cosine + 1e-12 or distance_maximum < chord2 - 1e-12:
                accepted[normal_id, word] |= bit
    return rejected, accepted


@njit(cache=True, inline='always')
def _angle(normal, other):
    x = normal[1] * other[2] - normal[2] * other[1]
    y = normal[2] * other[0] - normal[0] * other[2]
    z = normal[0] * other[1] - normal[1] * other[0]
    dot = normal[0] * other[0] + normal[1] * other[1] + normal[2] * other[2]
    return np.degrees(np.arctan2(np.sqrt(x * x + y * y + z * z), dot))


@njit(cache=True, inline='always')
def _cone_contains(normal, other, cosine, tolerance):
    dot = normal[0] * other[0] + normal[1] * other[1] + normal[2] * other[2]
    if dot > cosine + 1e-12:
        return True
    if dot < cosine - 1e-12:
        return False
    return _angle(normal, other) <= tolerance


@njit(cache=True, parallel=True)
def _count_batch(colors, normals, groups, pairs, points, ordered_groups, bounds,
                  ranges, children, node_bits, normal_ids, rejected_normal_nodes,
                  accepted_normal_nodes,
                  radii, inner_radius, tolerance, support, images, others,
                  first_pair, last_pair):
    cosine = np.cos(np.radians(tolerance))
    chord2 = (2 * np.sin(np.radians(tolerance) / 2)) ** 2
    inner2 = inner_radius * inner_radius
    radii2 = radii * radii
    eps = 1e-15  # Bounds only; all leaf comparisons retain the exact thresholds.
    for pair_id in prange(first_pair, last_pair):
        anchor_a, anchor_b = pairs[pair_id, 0], pairs[pair_id, 1]
        a, b = colors[anchor_a], colors[anchor_b]
        anchor_normal_a, anchor_normal_b = normals[anchor_a], normals[anchor_b]
        anchor_group_a, anchor_group_b = groups[anchor_a], groups[anchor_b]
        cached_normal_bounds = len(rejected_normal_nodes) != 0
        normal_id_a, normal_id_b = normal_ids[anchor_a], normal_ids[anchor_b]
        midpoint = (a + b) / 2
        found = np.zeros((6, node_bits.shape[1]), np.uint64)
        # A balanced tree built by cKDTree has depth O(log N); DFS stores only
        # one sibling per level. 128 accommodates far more than addressable N.
        stack_nodes = np.empty(128, np.int32)
        stack_masks = np.empty(128, np.int32)
        stack_nodes[0], stack_masks[0] = 0, 63
        top = 1
        while top:
            top -= 1
            node, mask = stack_nodes[top], stack_masks[top]
            minimum, maximum = _distance_bounds(midpoint, bounds[node])
            if minimum > radii2[2] + eps:
                continue
            min_a, max_a = _distance_bounds(a, bounds[node])
            min_b, max_b = _distance_bounds(b, bounds[node])
            if max_a < inner2 - eps or max_b < inner2 - eps:
                continue
            outside = min_a > inner2 + eps and min_b > inner2 + eps
            if cached_normal_bounds:
                word = node // 64
                bit = np.uint64(1) << np.uint64(node % 64)
                state_a = (1 if rejected_normal_nodes[normal_id_a, word] & bit else
                           2 if accepted_normal_nodes[normal_id_a, word] & bit else 0)
                state_b = (1 if rejected_normal_nodes[normal_id_b, word] & bit else
                           2 if accepted_normal_nodes[normal_id_b, word] & bit else 0)
            else:
                dot_a = _normal_dot_bounds(anchor_normal_a, bounds[node])
                dot_b = _normal_dot_bounds(anchor_normal_b, bounds[node])
                distance_a = _normal_distance_bounds(anchor_normal_a, bounds[node])
                distance_b = _normal_distance_bounds(anchor_normal_b, bounds[node])
                state_a = (1 if (dot_a[1] < cosine - 1e-12
                                 or distance_a[0] > chord2 + 1e-12) else
                           2 if (dot_a[0] > cosine + 1e-12
                                 or distance_a[1] < chord2 - 1e-12) else 0)
                state_b = (1 if (dot_b[1] < cosine - 1e-12
                                 or distance_b[0] > chord2 + 1e-12) else
                           2 if (dot_b[0] > cosine + 1e-12
                                 or distance_b[1] < chord2 - 1e-12) else 0)
            for radius in range(3):
                for side in range(2):
                    slot = 2 * radius + side
                    flag = 1 << slot
                    if not mask & flag:
                        continue
                    state = state_a if side == 0 else state_b
                    if minimum > radii2[radius] + eps or state == 1:
                        mask &= ~flag
                    elif outside and maximum < radii2[radius] - eps and state == 2:
                        support[radius, pair_id, side] += ranges[node, 1] - ranges[node, 0]
                        for word in range(node_bits.shape[1]):
                            found[slot, word] |= node_bits[node, word]
                        mask &= ~flag
            if mask == 0:
                continue
            if children[node, 0] >= 0:
                stack_nodes[top], stack_masks[top] = children[node, 0], mask
                stack_nodes[top + 1], stack_masks[top + 1] = children[node, 1], mask
                top += 2
                continue
            for item in range(ranges[node, 0], ranges[node, 1]):
                point = points[item]
                ax, ay, az = point[0] - a[0], point[1] - a[1], point[2] - a[2]
                bx, by, bz = point[0] - b[0], point[1] - b[1], point[2] - b[2]
                distance_a = np.sqrt(ax * ax + ay * ay + az * az)
                distance_b = np.sqrt(bx * bx + by * by + bz * bz)
                if distance_a <= inner_radius or distance_b <= inner_radius:
                    continue
                mx, my, mz = point[0] - midpoint[0], point[1] - midpoint[1], point[2] - midpoint[2]
                distance_mid = np.sqrt(mx * mx + my * my + mz * mz)
                if distance_mid > radii[2]:
                    continue
                supported_a = bool(mask & 21) and _cone_contains(
                    anchor_normal_a, point[3:], cosine, tolerance)
                supported_b = bool(mask & 42) and _cone_contains(
                    anchor_normal_b, point[3:], cosine, tolerance)
                group = ordered_groups[item]
                word, bit = group // 64, np.uint64(1) << np.uint64(group % 64)
                for radius in range(3):
                    if distance_mid > radii[radius]:
                        continue
                    for side in range(2):
                        slot = 2 * radius + side
                        supported = supported_a if side == 0 else supported_b
                        if (mask & (1 << slot)) and supported:
                            support[radius, pair_id, side] += 1
                            found[slot, word] |= bit
        for radius in range(3):
            for side in range(2):
                slot = 2 * radius + side
                count = 0
                for word in range(node_bits.shape[1]):
                    value = found[slot, word]
                    while value:
                        value &= value - np.uint64(1)
                        count += 1
                group = anchor_group_a if side == 0 else anchor_group_b
                own = bool(found[slot, group // 64] & (np.uint64(1) << np.uint64(group % 64)))
                images[radius, pair_id, side] = count
                others[radius, pair_id, side] = count - int(own)


def compute_branch_support_compiled(colors, normals, groups, pairs, radii, *,
                                     inner_radius, normal_tolerance_degrees, outputs=None):
    colors, normals = np.asarray(colors, float), np.asarray(normals, float)
    groups, pairs = np.asarray(groups, np.int64), np.asarray(pairs, np.int64)
    radii = np.asarray(radii, float)
    shape = (3, len(pairs), 2)
    outputs = outputs if outputs is not None else tuple(np.zeros(shape, np.int32) for _ in range(3))
    if not len(pairs):
        return outputs
    flattened = _flatten_tree(colors, normals, groups, float(radii[-1]), normal_tolerance_degrees)
    unique_normals, normal_ids = np.unique(normals, axis=0, return_inverse=True)
    # Sphere-grid slopes repeat across poses. Two packed node-state bitsets per
    # unique normal avoid recomputing cone bounds for every conflicting pair.
    # Fall back to the exact inline bounds when reuse is low or the cache is large.
    state_bytes = 2 * len(unique_normals) * ((len(flattened[2]) + 63) // 64) * 8
    normal_reuse = len(normals) / len(unique_normals)
    if normal_reuse >= 2 and state_bytes <= 256 * 1024 ** 2:
        rejected_normal_nodes, accepted_normal_nodes = _build_normal_node_states(
            unique_normals, flattened[2], np.cos(np.radians(normal_tolerance_degrees)),
            (2 * np.sin(np.radians(normal_tolerance_degrees) / 2)) ** 2)
        print(f'[branches] cached {len(unique_normals)} unique normals x '
              f'{len(flattened[2])} tree nodes ({state_bytes / 1024 ** 2:.1f} MiB, '
              f'{normal_reuse:.1f}x reuse)', flush=True)
    else:
        rejected_normal_nodes = accepted_normal_nodes = np.empty((0, 0), np.uint64)
        reason = (f'only {normal_reuse:.1f}x normal reuse' if normal_reuse < 2 else
                  f'{state_bytes / 1024 ** 2:.1f} MiB exceeds budget')
        print(f'[branches] normal/node cache skipped ({reason})', flush=True)
    for start in range(0, len(pairs), 32768):
        stop = min(start + 32768, len(pairs))
        began = time.perf_counter()
        _count_batch(colors, normals, groups, pairs, *flattened, normal_ids,
                     rejected_normal_nodes, accepted_normal_nodes, radii, inner_radius,
                     normal_tolerance_degrees, *outputs, start, stop)
        print(f'[branches] {stop}/{len(pairs)} pairs, all radii, '
              f'compiled exact counts ({time.perf_counter() - began:.2f}s)', flush=True)
    return outputs
