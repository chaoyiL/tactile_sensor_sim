"""Length-versus-quality tables and scientific figures; lower failure rates are better."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiment.lut_quality.runner import write_json


def records_for(rows: list[dict]) -> list[dict]:
    records = []
    for point in sorted(rows, key=lambda row: row['length_mm']):
        quality = point['full_scan']
        reference, heldout = quality['reference'], quality['holdout']
        for t, row in enumerate(reference['curves']):
            held, center = heldout['curves'][t], point['fixed_center']['curves'][t]
            records.append(dict(
                length_mm=point['length_mm'], tau=row['tau'],
                top_length_mm=point['top_length_mm'], bottom_length_mm=point['bottom_length_mm'],
                incompatibility=row['incompatibility'], node_error=row['node_error'],
                node_excess_error=row['node_excess_error'], runtime_error=row['runtime_error'],
                holdout_runtime_error=held['runtime_error'], holdout_node_error=held['node_error'],
                holdout_transferred_disc_failure_or_unseen=held['transferred_disc_failure_or_unseen'],
                center_runtime_error=center['runtime_error'],
                center_transferred_disc_failure_or_unseen=center['transferred_disc_failure_or_unseen'],
                validation_status=point['validation']['status'],
                validation_rms_degrees=point['validation']['rms_degrees'],
                full_scan_pose_count=point['scan']['full_scan']['pose_count'],
                full_scan_sample_count=point['scan']['full_scan']['sample_count'],
                training_pose_count=reference['pose_count'], training_sample_count=reference['sample_count'],
                holdout_pose_count=heldout['pose_count'], holdout_sample_count=heldout['sample_count'],
                center_sample_count=point['fixed_center']['sample_count'],
                occupied_bin_count=reference['occupied_bin_count'],
                singleton_sample_fraction=reference['singleton_sample_fraction'],
                holdout_unseen_bin_fraction=heldout['unseen_bin_fraction'],
                holdout_out_of_range_fraction=heldout['out_of_range_fraction'],
                theta_min_degrees=reference['theta_range_degrees'][0],
                theta_max_degrees=reference['theta_range_degrees'][1],
                lut_size=point['lut_size'],
                **{f'color_min_{channel}': point['lut_color_min'][c] for c, channel in enumerate('rgb')},
                **{f'color_max_{channel}': point['lut_color_max'][c] for c, channel in enumerate('rgb')}))
    return records


def _axes(ax, title, lengths):
    ax.set(title=title, xlabel='Surface length (mm)', ylabel='Failure fraction (lower is better)', ylim=(-.02, 1.02))
    ax.set_xticks(lengths)
    ax.grid(alpha=.22)
    ax.set_axisbelow(True)


def summarize(rows: list[dict], output: Path, identity: dict):
    records = records_for(rows)
    if not records:
        return
    with (output / 'length_lutqual.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(output / 'summary.json', dict(experiment=identity, points=records,
                                            incomplete=len(rows) < len(identity['lengths_mm'])))
    metric = identity['metric_settings']
    detail = [row for row in records if row['tau'] == metric['detail_tau']]
    lengths = [row['length_mm'] for row in detail]
    subtitle = (f'Width = {identity["base_config"]["surface"]["size_mm"][1]:g} mm | '
                f'weighting = {metric["weighting"]} | fixed strip total power | '
                f'{len(rows)}/{len(identity["lengths_mm"])} lengths completed')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), layout='constrained')
        for key, label, style, color in (
                ('runtime_error', 'Training: actual trilinear LUT', 'o-', '#2b8c63'),
                ('holdout_runtime_error', 'Held-out poses: actual trilinear LUT', 's-', '#b83542'),
                ('center_runtime_error', 'Fixed centre: same global training LUT', '^--', '#9b6a24')):
            axes[0].plot(lengths, [row[key] for row in detail], style, color=color, label=label)
        failed = [row for row in detail if row['validation_status'] != 'passed']
        if failed:
            axes[0].scatter([row['length_mm'] for row in failed], [row['holdout_runtime_error'] for row in failed],
                            marker='x', color='black', s=70, zorder=5, label='Calibration validation failed')
        for key, label, style, color in (
                ('incompatibility', 'A: empirical best fixed-node incompatibility', 'o-', '#2477ac'),
                ('node_error', 'E_node: stored nearest-node output', 's-', '#e18b24'),
                ('holdout_transferred_disc_failure_or_unseen', 'Held-out fitted disc failure or unseen', '^--', '#9060b2')):
            axes[1].plot(lengths, [row[key] for row in detail], style, color=color, label=label)
        _axes(axes[0], f'Actual LUT lookup error | tau={metric["detail_tau"]:g}', lengths)
        _axes(axes[1], 'Fixed-node model (A is NOT a bound for interpolation)', lengths)
        for ax in axes:
            ax.legend(fontsize=8, loc='best')
        fig.suptitle('Effect of surface length on LUT quality\n' + subtitle, fontsize=12)
        fig.savefig(output / 'length_lutqual.png', dpi=180)
        fig.savefig(output / 'length_lutqual.svg')
        plt.close(fig)

        columns = min(2, len(metric['taus']))
        row_count = math.ceil(len(metric['taus']) / columns)
        fig, axes = plt.subplots(row_count, columns, squeeze=False,
                                 figsize=(7 * columns, 4 * row_count), layout='constrained')
        for ax, tau in zip(axes.ravel(), metric['taus']):
            group = [row for row in records if row['tau'] == tau]
            for key, label, style in (
                    ('incompatibility', 'Training A: fixed-node model', 'o-'),
                    ('runtime_error', 'Training E_runtime', 's-'),
                    ('holdout_runtime_error', 'Held-out E_runtime', '^-'),
                    ('center_runtime_error', 'Fixed-centre E_runtime', 'x--')):
                ax.plot(lengths, [row[key] for row in group], style, label=label)
            _axes(ax, f'Slope tolerance tau={tau:g}', lengths)
            ax.legend(fontsize=8)
        for ax in axes.ravel()[len(metric['taus']):]:
            ax.set_visible(False)
        fig.suptitle('Length comparison at every slope tolerance\n' + subtitle, fontsize=12)
        fig.savefig(output / 'length_lutqual_all_tau.png', dpi=160)
        plt.close(fig)
