"""Branch continuation/classification adapted from recon_stable.

Reference: calibrate/analyze_ambiguities.py at commit
e02f95f119aa42d5aaf1ebae1f2304d60f045e8c (SHA256
a911ded4a699a9722398d2eb395762d3b53491de2c24122a3b83a073865a6bf9).
Only the requested branch diagnostics are implemented; no samples are rejected.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from .normal_mapping import UPSTREAM_COMMIT, slopes_to_normals


BRANCH_NAMES = ('no_outer_support', 'main_support', 'multiple_branches')
ROLE_NAMES = ('not_ambiguous', 'no_outer_support', 'dominant_branch',
              'weak_alternative', 'partner_dependent_dominance', 'overlapping_branch')


def _angles(first, second):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(first, second), axis=-1),
                                np.sum(first * second, axis=-1)))


def compare_branch_context(colors, normals, image_index, pair, context_indices, *,
                            inner_radius=.01, normal_tolerance_degrees=10.):
    """Upstream rule: exclude BOTH inner balls before counting normal-cone support."""
    colors, normals = np.asarray(colors, float), np.asarray(normals, float)
    pair, ids = np.asarray(pair, int), np.asarray(context_indices, int)
    distances = np.linalg.norm(colors[ids, None] - colors[pair][None], axis=-1)
    outside = (distances > inner_radius).all(axis=-1)
    ids, distances = ids[outside], distances[outside]
    angles = _angles(normals[ids, None], normals[pair][None])
    result = []
    for branch, anchor in enumerate(pair):
        selected = angles[:, branch] <= normal_tolerance_degrees
        support = ids[selected]
        result.append(dict(sample_index=int(anchor), support_indices=support.tolist(),
            support_count=len(support), support_image_count=len(np.unique(image_index[support])),
            other_image_count=len(np.unique(image_index[support][image_index[support] != image_index[anchor]])),
            nearest_support_color_distance=float(distances[selected, branch].min()) if selected.any() else None,
            minimum_context_normal_angle_degrees=float(angles[:, branch].min()) if len(ids) else None))
    return result


def classify_branch_support(support_counts, *, dominance_fraction=.8):
    """Codes index BRANCH_NAMES; dominance is support density, not correctness."""
    counts = np.asarray(support_counts, float)
    if counts.ndim < 1 or counts.shape[-1] != 2 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Expected finite nonnegative support counts with shape (..., 2)')
    if not .5 < dominance_fraction <= 1:
        raise ValueError('Dominance fraction must be in (0.5, 1]')
    total = counts.sum(axis=-1)
    dominant = counts.max(axis=-1) >= dominance_fraction * total
    return np.where(total == 0, 0, np.where(dominant, 1, 2)).astype(np.int8)


def aggregate_branch_classification(pairs, pair_classes, sample_count):
    """Deduplicate endpoints; ANY overlapping partner takes priority over dominance."""
    result = np.full(sample_count, -1, np.int8)
    np.maximum.at(result, np.asarray(pairs, int).ravel(), np.repeat(pair_classes, 2))
    return result


def _class_counts(classes):
    return {name: int(np.sum(classes == index)) for index, name in enumerate(BRANCH_NAMES)}


def compute_branch_support(colors, normals, image_index, pairs, radii, *,
                            inner_radius=.01, normal_tolerance_degrees=10., outputs=None):
    """Exact upstream counts, reusing each anchor's possible outer/cone support.

    A pair midpoint is at most inner_radius/2 from its anchor. A six-dimensional
    box query bounds color AND normal distance, after which exact Euclidean and
    angular tests remove false positives. Partner-dependent exclusions and outer
    balls are evaluated in bounded matrices, not a Python KD query per pair/radius.
    """
    colors, normals = np.asarray(colors, float), np.asarray(normals, float)
    groups, pairs = np.asarray(image_index), np.asarray(pairs)
    radii = np.asarray(radii, float)
    shape = (len(radii), len(pairs), 2)
    support, images, others = outputs if outputs is not None else (
        np.zeros(shape, np.int32), np.zeros(shape, np.int32), np.zeros(shape, np.int32))
    if not len(pairs) or inner_radius == 0:
        return support, images, others
    if len(pairs) >= 1000:
        from .ambiguity_patterns_compiled import compute_branch_support_compiled
        return compute_branch_support_compiled(
            colors, normals, groups, pairs, radii, inner_radius=inner_radius,
            normal_tolerance_degrees=normal_tolerance_degrees,
            outputs=(support, images, others))
    # Outward roundoff padding affects candidate enumeration only; final tests
    # retain the exact thresholds. No approximate KD query or sampling is used.
    bound = float(radii.max()) + inner_radius / 2
    chord = 2 * np.sin(np.radians(normal_tolerance_degrees) / 2)
    normal_scale = max(float(chord), 1e-12)
    features = np.column_stack((colors / bound, normals / normal_scale))
    tree = cKDTree(features)
    last_progress = time.perf_counter()
    for side in (0, 1):
        order = np.argsort(pairs[:, side], kind='stable')
        anchors = pairs[order, side]
        starts = np.r_[0, np.flatnonzero(anchors[1:] != anchors[:-1]) + 1]
        ends = np.r_[starts[1:], len(anchors)]
        for position, (start, end) in enumerate(zip(starts, ends, strict=True), 1):
            anchor = int(anchors[start])
            candidate = np.asarray(tree.query_ball_point(features[anchor], 1 + 1e-12,
                                                         p=np.inf, eps=0), np.int64)
            distances = np.linalg.norm(colors[candidate] - colors[anchor], axis=-1)
            selected = ((distances > inner_radius) & (distances <= bound + 1e-12)
                        & (_angles(normals[candidate], normals[anchor]) <= normal_tolerance_degrees))
            candidate = candidate[selected]
            if len(candidate):
                # Sorting by source image lets logical_or.reduceat count distinct
                # supporting images without a Python set per conflicting pair.
                candidate = candidate[np.argsort(groups[candidate], kind='stable')]
                candidate_groups = groups[candidate]
                group_starts = np.r_[0, np.flatnonzero(candidate_groups[1:] != candidate_groups[:-1]) + 1]
                own = np.flatnonzero(candidate_groups[group_starts] == groups[anchor])
                point = colors[candidate]
                pair_ids = order[start:end]
                block_size = max(1, min(256, 262144 // len(candidate)))
                for offset in range(0, len(pair_ids), block_size):
                    ids = pair_ids[offset:offset + block_size]
                    partner = colors[pairs[ids, 1 - side]]
                    midpoint = (partner + colors[anchor]) / 2
                    other_distance2 = np.zeros((len(ids), len(candidate)))
                    midpoint_distance2 = np.zeros_like(other_distance2)
                    for channel in range(3):
                        delta = point[None, :, channel] - partner[:, None, channel]
                        other_distance2 += delta * delta
                        delta = point[None, :, channel] - midpoint[:, None, channel]
                        midpoint_distance2 += delta * delta
                    outside = np.sqrt(other_distance2) > inner_radius
                    for radius_index, radius in enumerate(radii):
                        valid = outside & (np.sqrt(midpoint_distance2) <= radius)
                        count = valid.sum(axis=-1)
                        present = np.logical_or.reduceat(valid, group_starts, axis=1)
                        image_count = present.sum(axis=-1)
                        own_present = present[:, own[0]] if len(own) else 0
                        support[radius_index, ids, side] = count
                        images[radius_index, ids, side] = image_count
                        others[radius_index, ids, side] = image_count - own_present
            if time.perf_counter() - last_progress >= 5 or position == len(starts):
                print(f'[branches] side {side + 1}/2: {position}/{len(starts)} anchors', flush=True)
                last_progress = time.perf_counter()
        del order, anchors
    return support, images, others


def _figure(size):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=size, layout='constrained')
    FigureCanvasAgg(figure)
    return figure


def _save(figure, path):
    figure.savefig(path, dpi=160, format='png')
    figure.clear()


def _principal_plane(points):
    """Fit a centered PCA plane to finite RGB points and return JSON-safe statistics."""
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite RGB points with shape (N, 3)')
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / (len(points) - 1)
    variances, vectors = np.linalg.eigh(covariance)
    order = np.argsort(variances)[::-1]
    variances = np.maximum(variances[order], 0.)
    components = vectors[:, order].T
    # Eigenvector signs are arbitrary. Fix them so reports and rendered views are
    # deterministic across BLAS implementations.
    for component in components:
        largest = int(np.argmax(np.abs(component)))
        if component[largest] < 0:
            component *= -1
    tolerance = np.finfo(np.float64).eps * max(float(variances[0]), 1.)
    total = float(variances.sum())
    if variances[1] <= tolerance or total <= tolerance:
        return None
    ratios = variances / total
    normal = components[2]
    return dict(sample_count=len(points), centroid=centroid.tolist(),
                components=components.tolist(), explained_variance=variances.tolist(),
                explained_variance_ratio=ratios.tolist(),
                first_two_explained_variance_ratio=float(ratios[:2].sum()),
                plane_normal=normal.tolist(), plane_offset=float(normal @ centroid))


def _classification_figure(samples, classes, report, output):
    """Render three near-coordinate-plane views of the RGB cloud and PCA plane."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    figure = _figure((18, 8))
    grid = figure.add_gridspec(2, 3, height_ratios=(15, 2))
    axes = [figure.add_subplot(grid[0, index], projection='3d') for index in range(3)]
    info_axis = figure.add_subplot(grid[1, :])
    info_axis.axis('off')
    colors = np.asarray(samples['color_residual_samples'], np.float64)
    handles = []
    layers = []
    for kind, name, color, size, alpha in (
            (-1, 'Other samples', '#bdbdbd', 1.5, .08),
            (0, 'No outer support', '#e8aa20', 4, .50),
            (1, 'Main support', '#2476b6', 3.5, .34),
            (2, 'Multiple branches', '#d33333', 4, .52)):
        selected = classes == kind
        if not selected.any():
            continue
        layers.append((colors[selected], color, size, alpha))
        handles.append(Line2D([], [], linestyle='none', marker='o', markersize=6,
                              markerfacecolor=color, markeredgecolor='none', alpha=max(alpha, .55),
                              label=f'{name}: {selected.sum()}'))

    plane = report.get('color_pca_plane')
    surface = boundary = None
    if plane is not None:
        ambiguous = colors[classes >= 0]
        centroid = np.asarray(plane['centroid'])
        components = np.asarray(plane['components'])
        coordinates = (ambiguous - centroid) @ components[:2].T
        percentile = (1, 99) if len(coordinates) >= 20 else (0, 100)
        lower, upper = np.percentile(coordinates, percentile, axis=0)
        width = upper - lower
        lower -= .04 * width
        upper += .04 * width
        first = np.linspace(lower[0], upper[0], 9)
        second = np.linspace(lower[1], upper[1], 9)
        u, v = np.meshgrid(first, second)
        surface = (centroid[None, None] + u[..., None] * components[0]
                   + v[..., None] * components[1])
        boundary = surface[(0, 0, -1, -1, 0), (0, -1, -1, 0, 0)]
        explained = 100 * plane['first_two_explained_variance_ratio']
        handles.append(Patch(facecolor='#756bb1', edgecolor='#5b4a9e', alpha=.3,
                             label=f'Ambiguous-point PCA plane: {explained:.2f}%'))
        normal = np.asarray(plane['plane_normal'])
        equation = (f'{normal[0]:.3f} delta R {normal[1]:+.3f} delta G '
                    f'{normal[2]:+.3f} delta B = {plane["plane_offset"]:.4f}')
        pca_text = (f'Centered PCA of {plane["sample_count"]} ambiguous points | {equation} | '
                    f'PC1 + PC2 explained variance: {explained:.2f}%')
    else:
        pca_text = 'A unique PCA plane requires at least three non-collinear ambiguous points.'

    if len(colors):
        lower = colors.min(axis=0)
        upper = colors.max(axis=0)
        span = upper - lower
        scale = max(float(span.max()), 1e-3)
        padding = np.where(span > 0, .04 * span, .04 * scale)
        limits = tuple((lower[index] - padding[index], upper[index] + padding[index])
                       for index in range(3))
        box_aspect = np.maximum(span, .15 * scale)
    else:
        limits = ((-.5, .5),) * 3
        box_aspect = (1, 1, 1)

    # Each image plane is close to one RGB coordinate plane, while the deliberate
    # angular offset preserves enough of the third dimension to retain depth cues.
    views = (('Near RG-plane view\n(camera mostly along B)', 72, -45),
             ('Near RB-plane view\n(camera mostly along G)', 12, 78),
             ('Near GB-plane view\n(camera mostly along R)', 12, 12))
    for axis, (title, elevation, azimuth) in zip(axes, views, strict=True):
        for points, color, size, alpha in layers:
            axis.scatter(*points.T, s=size, alpha=alpha, color=color,
                         edgecolors='none', depthshade=False, rasterized=True)
        if surface is not None:
            axis.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2],
                              color='#756bb1', alpha=.18, shade=False, linewidth=0,
                              antialiased=True)
            axis.plot_wireframe(surface[..., 0], surface[..., 1], surface[..., 2],
                                rstride=2, cstride=2, color='#5b4a9e', alpha=.45,
                                linewidth=.6)
            axis.plot(*boundary.T, color='#4a3b87', alpha=.85, linewidth=1.4)
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
        axis.set_zlim(*limits[2])
        axis.set_box_aspect(box_aspect)
        axis.set_proj_type('ortho')
        axis.view_init(elev=elevation, azim=azimuth)
        # A warm, near-white pane keeps the neutral gray context cloud distinct
        # without competing with the blue/red classifications or purple plane.
        pane_color = (1., .965, .88, 1.)
        axis.xaxis.set_pane_color(pane_color)
        axis.yaxis.set_pane_color(pane_color)
        axis.zaxis.set_pane_color(pane_color)
        axis.set(xlabel='delta R', ylabel='delta G', zlabel='delta B', title=title)
        axis.grid(alpha=.2)

    if handles:
        info_axis.legend(handles=handles, loc='upper center', ncol=len(handles),
                         fontsize=8, framealpha=.9)
    info_axis.text(.5, .05, pca_text, transform=info_axis.transAxes, ha='center', va='bottom',
                   fontsize=9, bbox=dict(boxstyle='round,pad=.4', facecolor='white',
                                         edgecolor='#756bb1' if plane is not None else '#888888',
                                         alpha=.9))
    figure.suptitle(f'Expanded branch classification | outer radius {report["primary_outer_radius"]:g} | '
        f'main support >= {100 * report["dominance_fraction"]:g}%\n'
        'Three near-coordinate-plane RGB views; PCA plane fitted to ambiguous endpoints; '
        'no sample deletion', fontsize=12)
    _save(figure, output / 'branch_classification.png')


def _continuation_figure(samples, report, output):
    examples = [(key, report['examples'][key]) for key in ('both', 'one') if key in report['examples']]
    if not examples:
        figure = _figure((13, 5))
        for index, title in enumerate(('Color context', 'Local gradient space'), 1):
            axis = figure.add_subplot(1, 2, index)
            axis.text(.5, .5, 'No conflicting pair has outer support', ha='center', va='center')
            axis.set(title=title, xticks=[], yticks=[])
        figure.suptitle('Branch continuation | no supported example; no sample deletion')
        _save(figure, output / 'branch_continuation.png')
        return
    colors, slopes = samples['color_residual_samples'], samples['slope_samples']
    figure = _figure((13, 5 * len(examples)))
    for row, (kind, example) in enumerate(examples):
        pair, ids = np.asarray(example['pair']), np.asarray(example['context_indices'], int)
        outside = (np.linalg.norm(colors[ids, None].astype(float) - colors[pair][None].astype(float), axis=-1)
                   > report['inner_radius']).all(axis=-1)
        ids = ids[outside]
        color_axis = figure.add_subplot(len(examples), 2, 2 * row + 1, projection='3d')
        slope_axis = figure.add_subplot(len(examples), 2, 2 * row + 2)
        color_axis.scatter(*colors[ids].T, color='#bbbbbb', s=9, alpha=.3)
        slope_axis.scatter(*slopes[ids].T, color='#bbbbbb', s=10, alpha=.3)
        for branch, color, name in zip(example['branches'], ('#2476b6', '#e87519'), ('A', 'B'), strict=True):
            support = np.asarray(branch['support_indices'], int)
            anchor = branch['sample_index']
            color_axis.scatter(*colors[support].T, color=color, s=20)
            color_axis.scatter(*colors[anchor], color=color, marker='*', edgecolors='black', s=150)
            slope_axis.scatter(*slopes[support].T, color=color, s=25,
                label=f'{name}: {len(support)} points / {branch["support_image_count"]} context images')
            slope_axis.scatter(*slopes[anchor], color=color, marker='*', edgecolors='black', s=180)
            slope_axis.annotate(name, slopes[anchor], xytext=(6, -10), textcoords='offset points')
        color_axis.set(xlabel='delta R', ylabel='delta G', zlabel='delta B',
            title=f'Pair {pair.tolist()} | original neighborhoods: '
                  f'{example["inner_counts"][0]} + {example["inner_counts"][1]}')
        slope_axis.set(xlabel='p', ylabel='q', title='Both branches have context' if kind == 'both'
                       else 'Only one branch has context at this radius')
        slope_axis.invert_yaxis()
        slope_axis.legend(fontsize=9)
        slope_axis.grid(alpha=.2)
    figure.suptitle(f'Continuation beyond the original color neighborhood | outer radius '
        f'{report["outer_radii"][1]:g}\nBoth inner {report["inner_radius"]:g} balls excluded; '
        f'normal support <= {report["normal_tolerance_degrees"]:g} deg; stars = excluded anchors\n'
        'Evidence of neighboring branches; no automatic sample deletion', fontsize=11)
    _save(figure, output / 'branch_continuation.png')


def analyze_branch_patterns(samples, pairs, output, *, color_threshold=.01,
                            angle_threshold=20., dominance_fraction=.8,
                            source_sample_index=None, run_signature='', sampling=None):
    """Analyze every conflicting pair of the selected dataset without subsampling it."""
    classify_branch_support(np.empty((0, 2)), dominance_fraction=dominance_fraction)
    from .calibration_ambiguity import _write_npz
    from .calibration_visualization import _write_json

    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    colors = np.asarray(samples['color_residual_samples'], float)
    normals = slopes_to_normals(samples['slope_samples'])
    groups = np.asarray(samples['image_index'])
    tree = cKDTree(colors)
    inner_counts = tree.query_ball_point(colors, color_threshold, return_length=True, workers=-1)
    radii = np.array([2, 3, 4]) * color_threshold
    shape = (3, len(pairs), 2)
    temporary_paths, buffers = [], []
    try:
        for name in ('support', 'images', 'other_images'):
            if len(pairs):
                path = output / f'.{name}.tmp.npy'
                temporary_paths.append(path)
                buffers.append(np.lib.format.open_memmap(path, mode='w+', dtype=np.int32, shape=shape))
            else:
                buffers.append(np.zeros(shape, np.int32))
        support, images, others = compute_branch_support(
            colors, normals, groups, pairs, radii, inner_radius=color_threshold,
            normal_tolerance_degrees=angle_threshold / 2, outputs=buffers)
        point_classes = np.full((3, len(colors)), -1, np.int8)
        pair_counts = np.zeros((3, 3), np.int64)
        stable_pair_counts = np.zeros(3, np.int64)
        dominant_degree = np.zeros(len(colors), np.int64)
        alternative_degree = np.zeros(len(colors), np.int64)
        main_degree = np.zeros(len(colors), np.int64)
        overlap_degree = np.zeros(len(colors), np.int64)
        cross_image_overlap = np.zeros(len(colors), bool)
        sensitivity_pair = np.zeros((3, 3), np.int64)
        sensitivity_points = np.full((3, len(colors)), -1, np.int8)
        disagrees = 0
        scopes = ('all', 'either_endpoint_fewer_than6', 'both_endpoints_only2')
        continuation_counts = np.zeros((3, 3, 7), np.int64)
        best = {}
        for start in range(0, len(pairs), 65536):
            stop = min(start + 65536, len(pairs))
            chunk = pairs[start:stop]
            sizes, image_counts, other_counts = [np.asarray(a[:, start:stop]) for a in buffers]
            kinds = classify_branch_support(sizes, dominance_fraction=dominance_fraction)
            stable = (kinds == kinds[1]).all(axis=0)
            for radius_index in range(3):
                np.maximum.at(point_classes[radius_index], chunk.ravel(), np.repeat(kinds[radius_index], 2))
                pair_counts[radius_index] += np.bincount(kinds[radius_index], minlength=3)
            stable_pair_counts += np.bincount(kinds[1, stable], minlength=3)
            main = kinds[1] == 1
            overlap = kinds[1] == 2
            main_pairs = chunk[main]
            winners = sizes[1, main].argmax(axis=-1)
            np.add.at(dominant_degree, main_pairs[np.arange(len(main_pairs)), winners], 1)
            np.add.at(alternative_degree, main_pairs[np.arange(len(main_pairs)), 1 - winners], 1)
            np.add.at(main_degree, main_pairs.ravel(), 1)
            np.add.at(overlap_degree, chunk[overlap].ravel(), 1)
            cross_image_overlap[chunk[overlap & (image_counts[1] >= 2).all(axis=-1)].ravel()] = True
            disagrees += int(np.sum(kinds[1] != classify_branch_support(
                image_counts[1], dominance_fraction=dominance_fraction)))
            for number, fraction in enumerate((.7, .8, .9)):
                classified = classify_branch_support(sizes[1], dominance_fraction=fraction)
                sensitivity_pair[number] += np.bincount(classified, minlength=3)
                np.maximum.at(sensitivity_points[number], chunk.ravel(), np.repeat(classified, 2))
            inner = inner_counts[chunk]
            selections = (np.ones(len(chunk), bool), (inner < 6).any(axis=-1), (inner == 2).all(axis=-1))
            for radius_index in range(3):
                s, im, ot = sizes[radius_index], image_counts[radius_index], other_counts[radius_index]
                flags = (np.ones(len(chunk), bool), (s > 0).all(axis=-1),
                         (s > 0).any(axis=-1) & (s == 0).any(axis=-1), (s == 0).all(axis=-1),
                         (s >= 3).all(axis=-1), (im >= 2).all(axis=-1), (ot > 0).all(axis=-1))
                for scope_index, selected in enumerate(selections):
                    continuation_counts[radius_index, scope_index] += [int(np.sum(selected & f)) for f in flags]
            # Prefer exactly the upstream sparse-pair examples. If none exist,
            # retain a supported dense-pair example and state its real inner counts.
            for name, selected in (('both', (sizes[1] > 0).all(axis=-1)),
                                    ('one', (sizes[1] > 0).any(axis=-1) & (sizes[1] == 0).any(axis=-1))):
                candidates = np.flatnonzero(selected)
                if not len(candidates):
                    continue
                sparse = selections[2][candidates].astype(int)
                if name == 'both':
                    score_parts = (sparse, sizes[1, candidates].min(axis=-1),
                                   image_counts[1, candidates].min(axis=-1), sizes[1, candidates].sum(axis=-1))
                else:
                    score_parts = (sparse, sizes[1, candidates].max(axis=-1))
                order = np.lexsort((-candidates, *score_parts[::-1]))
                chosen = int(candidates[order[-1]])
                score = tuple(int(part[order[-1]]) for part in score_parts)
                if name not in best or score > best[name][0]:
                    best[name] = (score, chunk[chosen].copy())

        primary = point_classes[1]
        stable = (point_classes == primary).all(axis=0)
        main_only = primary == 1
        roles = np.zeros(len(colors), np.int8)
        roles[primary >= 0] = 1
        roles[main_only & (dominant_degree > 0)] = 2
        roles[main_only & (alternative_degree > 0)] = 3
        roles[main_only & (dominant_degree > 0) & (alternative_degree > 0)] = 4
        roles[primary == 2] = 5
        classification = dict(primary_outer_radius=float(radii[1]), dominance_fraction=dominance_fraction,
            pair_counts=dict(zip(BRANCH_NAMES, pair_counts[1].tolist(), strict=True)),
            point_counts=_class_counts(primary),
            stable_all_radii_point_counts=_class_counts(primary[stable]),
            stable_larger_two_radii_point_counts=_class_counts(primary[primary == point_classes[2]]),
            stable_all_radii_pair_counts=dict(zip(BRANCH_NAMES, stable_pair_counts.tolist(), strict=True)),
            mixed_pair_classification_point_count=int(((main_degree > 0) & (primary == 2)).sum()),
            main_support_point_roles=dict(Counter(ROLE_NAMES[i] for i in roles[main_only])),
            multiple_branches_with_both_branches_in_multiple_images_point_count=int(cross_image_overlap.sum()),
            pair_classification_disagrees_when_counting_images=disagrees,
            radius_sensitivity=[dict(outer_radius=float(radius),
                pairs=dict(zip(BRANCH_NAMES, pair_counts[i].tolist(), strict=True)), points=_class_counts(point_classes[i]))
                for i, radius in enumerate(radii)],
            dominance_fraction_sensitivity=[dict(dominance_fraction=fraction,
                pairs=dict(zip(BRANCH_NAMES, sensitivity_pair[i].tolist(), strict=True)),
                points=_class_counts(sensitivity_points[i])) for i, fraction in enumerate((.7, .8, .9))],
            method='Every conflicting pair: outer color ball around midpoint, excluding BOTH original balls. '
                   'Compare support in the two anchor-normal cones. No minimum inner/outer count. '
                   'At 3x radius, zero total = no_outer_support; >= dominance fraction = main_support; '
                   'otherwise multiple_branches. ANY overlapping partner takes priority for a point.',
            limitation='Continuation density is not label correctness or independent physical sheets; '
                       'per-image sampling density can affect counts. No samples are rejected.')
        classification['color_pca_plane'] = _principal_plane(colors[primary >= 0])
        examples = {}
        for name, (score, pair) in best.items():
            ids = np.asarray(tree.query_ball_point(colors[pair].mean(axis=0), radii[1], return_sorted=True), int)
            examples[name] = dict(pair=pair.tolist(), context_indices=ids.tolist(),
                inner_counts=inner_counts[pair].tolist(), uses_sparse_pair=bool(score[0]),
                outer_radius=float(radii[1]), branches=compare_branch_context(
                    colors, normals, groups, pair, ids, inner_radius=color_threshold,
                    normal_tolerance_degrees=angle_threshold / 2))
        field_names = ('pair_count', 'both_have_context', 'one_has_context', 'neither_has_context',
                       'both_have_at_least3', 'both_have_at_least2_context_images', 'both_have_other_image_support')
        continuation = dict(inner_radius=color_threshold, outer_radii=radii.tolist(),
            normal_tolerance_degrees=angle_threshold / 2,
            method='Every conflicting pair; no inner sample minimum; both original color balls excluded.',
            interpretation='Support is continuation evidence, not a reason to reject either anchor.',
            summaries=[dict(scope=scope, outer_radius=float(radius),
                **dict(zip(field_names, continuation_counts[i, j].tolist(), strict=True)))
                for i, radius in enumerate(radii) for j, scope in enumerate(scopes)],
            examples=examples, expanded_classification=classification)
        _classification_figure(samples, primary, classification, output)
        _continuation_figure(samples, continuation, output)
        _write_npz(output / 'branch_classification.npz',
            source_sample_index=(np.arange(len(colors)) if source_sample_index is None else source_sample_index),
            classification=primary, classification_by_radius=point_classes, class_names=np.asarray(BRANCH_NAMES),
            role=roles, role_names=np.asarray(ROLE_NAMES), stable_all_radii=stable,
            outer_radii=radii, inner_sample_count=inner_counts,
            dominant_over_partners=dominant_degree, weaker_than_partners=alternative_degree,
            overlap_partners=overlap_degree, main_support_partners=main_degree,
            image_index=groups, image_paths=samples['image_paths'],
            run_signature=np.asarray(run_signature), not_ambiguous_code=np.asarray(-1, np.int8))
        report = dict(report_kind='simulated_ambiguity_branch_patterns', upstream_commit=UPSTREAM_COMMIT,
            sample_count=len(colors), conflicting_pair_count=len(pairs),
            run_signature=run_signature, sampling=sampling,
            count_scope=('exact_within_selected_subset' if sampling and sampling.get('is_sampled')
                         else 'exact_full_dataset'),
            ambiguous_sample_count=int((primary >= 0).sum()),
            color_distance_threshold=color_threshold, normal_angle_threshold_degrees=angle_threshold,
            branch_continuation=continuation, branch_classification=classification,
            artifacts=dict(classification='branch_classification.png', continuation='branch_continuation.png',
                           point_classes='branch_classification.npz'), render_count=0,
            total_seconds=time.perf_counter() - started)
        _write_json(output / 'analysis.json', report)
        return report
    finally:
        for array in buffers:
            if isinstance(array, np.memmap):
                array.flush()
                array._mmap.close()
        for path in temporary_paths:
            path.unlink(missing_ok=True)
