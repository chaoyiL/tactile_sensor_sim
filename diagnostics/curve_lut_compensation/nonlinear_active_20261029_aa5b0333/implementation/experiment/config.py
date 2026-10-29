"""Load experiment settings without reading or mutating the main simulation config."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from utils.config import validate_config
from utils.backend_layout import CONFIG_DIR
from experiment.norm_diff.metric import resolve_noise_settings


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
DEFAULT_EXP_CONFIG = CONFIG_DIR / 'length_normdiff.json'
METRIC_DEFAULTS = dict(noise_fraction=.03, noise_floor=0., theta_weight=1., phi_weight=1.,
                       theta_bins=12, phi_bins=24, anchors_per_target=1,
                       theta_ceiling_degrees=None)
LUT_QUALITY_DEFAULTS = dict(taus=[0., .025, .05, .1, .2], detail_tau=.1,
                            weighting='sample', split='auto', top_bins=8,
                            examples_per_bin=12)
LENGTH_LUTQUAL_DEFAULTS = dict(lengths_mm=[21.8, 26.8, 31.8, 36.8, 41.8, 46.8, 51.8, 55.9],
                               output_dir='length_lutqual/results')


def resolve_light_distribution_settings(config: dict) -> dict:
    """Validate power-only candidate families before any scan or output write."""
    raw = config.get('light_distribution')
    if not isinstance(raw, dict) or set(raw) != {'candidates'}:
        raise ValueError('light_distribution requires only a candidates array')
    candidates = raw['candidates']
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('light_distribution.candidates must be nonempty')
    names = set()
    for row in candidates:
        if not isinstance(row, dict) or not {'name', 'family'}.issubset(row):
            raise ValueError('Each candidate requires name and family')
        required = ({'name', 'family', 'point_power_rgb_w'} if row['family'] == 'explicit'
                    else {'name', 'family', 'strength', 'short_power_w', 'long_power_w'})
        if not required.issubset(row) or set(row) - required - {'description'}:
            raise ValueError(f'Candidate fields must be {sorted(required)} with optional description')
        if 'description' in row and not isinstance(row['description'], str):
            raise ValueError('Candidate description must be text')
        name = row['name']
        if not isinstance(name, str) or not name or any(
                c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name) or name in names:
            raise ValueError('Candidate names must be unique lowercase ASCII identifiers')
        names.add(name)
        if row['family'] not in ('complement', 'gradient', 'explicit'):
            raise ValueError(f'{name}: unknown power family')
        if row['family'] == 'explicit':
            powers = row['point_power_rgb_w']
            if not isinstance(powers, dict) or set(powers) != {'left', 'right', 'top', 'bottom'}:
                raise ValueError(f'{name}.point_power_rgb_w requires all four sides')
            for side, values in powers.items():
                if not isinstance(values, list) or not values or any(
                        not isinstance(rgb, list) or len(rgb) != 3 or any(
                            isinstance(v, bool) or not isinstance(v, (int, float))
                            or not math.isfinite(v) or v < 0 for v in rgb) for rgb in values):
                    raise ValueError(f'{name}.{side}: expected a nonempty array of nonnegative RGB watts')
            continue
        for key in ('strength', 'short_power_w', 'long_power_w'):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name}.{key}: expected finite nonnegative number')
        if row['strength'] > 1 or row['short_power_w'] + row['long_power_w'] <= 0:
            raise ValueError(f'{name}: require strength in [0,1] and positive combined power')
    return copy.deepcopy(raw)


def resolve_length_lutqual_settings(config: dict, **overrides) -> dict:
    """Length sweep settings are independent of the NormDiff sweep's output and range."""
    raw = config.get('length_lutqual', {})
    if not isinstance(raw, dict):
        raise ValueError('exp_config.length_lutqual must be an object')
    unknown = (set(raw) | set(overrides)) - set(LENGTH_LUTQUAL_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown length_lutqual settings: {sorted(unknown)}')
    settings = copy.deepcopy(dict(LENGTH_LUTQUAL_DEFAULTS, **raw))
    settings.update({key: value for key, value in overrides.items() if value is not None})
    lengths = settings['lengths_mm']
    if not isinstance(lengths, list) or not lengths or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 for value in lengths):
        raise ValueError('length_lutqual.lengths_mm must be a nonempty list of positive lengths')
    if len(set(lengths)) != len(lengths):
        raise ValueError('length_lutqual.lengths_mm must not contain duplicates')
    settings['lengths_mm'] = sorted(map(float, lengths))
    if not isinstance(settings['output_dir'], str) or not settings['output_dir'].strip():
        raise ValueError('length_lutqual.output_dir must be a nonempty path string')
    return settings


def resolve_lut_quality_settings(config: dict, **overrides) -> dict:
    """Resolve the independent LUT quality experiment, without changing its inputs."""
    raw = config.get('lut_quality', {})
    if not isinstance(raw, dict):
        raise ValueError('exp_config.lut_quality must be an object')
    unknown = (set(raw) | set(overrides)) - (set(LUT_QUALITY_DEFAULTS)
                                           | {'samples', 'lut', 'output_dir'})
    if unknown:
        raise ValueError(f'Unknown LUT quality settings: {sorted(unknown)}')
    settings = copy.deepcopy(dict(LUT_QUALITY_DEFAULTS, **raw))
    settings.update({key: value for key, value in overrides.items() if value is not None})
    for key in ('samples', 'lut', 'output_dir'):
        if key in settings and (not isinstance(settings[key], str) or not settings[key].strip()):
            raise ValueError(f'lut_quality.{key} must be a nonempty path string')
    taus = settings['taus']
    if not isinstance(taus, list) or len(taus) < 2 or any(
            isinstance(tau, bool) or not isinstance(tau, (int, float))
            or not math.isfinite(tau) or tau < 0 for tau in taus):
        raise ValueError('lut_quality.taus must contain at least two finite nonnegative tolerances')
    if len(set(taus)) != len(taus):
        raise ValueError('lut_quality.taus must not contain duplicates')
    settings['taus'] = sorted(map(float, taus))
    detail = settings['detail_tau']
    if isinstance(detail, bool) or not isinstance(detail, (int, float)) \
            or not math.isfinite(detail) or detail not in settings['taus']:
        raise ValueError('lut_quality.detail_tau must be one of taus')
    if settings['weighting'] not in ('sample', 'pose_equal'):
        raise ValueError('lut_quality.weighting must be sample or pose_equal')
    if settings['split'] not in ('auto', 'all'):
        raise ValueError('lut_quality.split must be auto or all')
    for key in ('top_bins', 'examples_per_bin'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'lut_quality.{key} must be a positive integer')
    return settings


def resolve_metric_settings(config: dict, **overrides) -> dict:
    """CLI values override exp_config.norm_diff, then the existing metric defaults."""
    raw = config.get('norm_diff', {})
    if not isinstance(raw, dict):
        raise ValueError('exp_config.norm_diff must be an object')
    unknown = (set(raw) | set(overrides)) - set(METRIC_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown NormDiff settings: {sorted(unknown)}')
    settings = dict(METRIC_DEFAULTS, **raw)
    settings.update({key: value for key, value in overrides.items() if value is not None})
    settings.update(resolve_noise_settings({'norm_diff': settings}))
    for key in ('theta_bins', 'phi_bins', 'anchors_per_target'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'norm_diff.{key} must be a positive integer')
    for key in ('theta_weight', 'phi_weight'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            raise ValueError(f'norm_diff.{key} must be finite and nonnegative')
    if settings['theta_weight'] + settings['phi_weight'] == 0:
        raise ValueError('NormDiff angular weights must have a positive sum')
    ceiling = settings['theta_ceiling_degrees']
    if ceiling is not None and (isinstance(ceiling, bool) or not isinstance(ceiling, (int, float))
                                or not math.isfinite(ceiling) or ceiling <= 0):
        raise ValueError('norm_diff.theta_ceiling_degrees must be null or finite and positive')
    return settings


def load_experiment_config(path: str | Path = DEFAULT_EXP_CONFIG) -> dict:
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or type(config.get('format_version')) is not int \
            or config['format_version'] != 1:
        raise ValueError('Expected an exp_config file with format_version=1')
    allowed = {'format_version', 'simulation', 'norm_diff', 'norm_diff_evaluation',
               'lut_quality', 'length_normdiff', 'length_lutqual', 'light_distribution',
               'curve_lut_transfer', 'curve_lut_compensation'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f'Unknown experiment configuration sections in {path}: {sorted(unknown)}')
    simulation = config.get('simulation')
    if simulation is not None:
        if not isinstance(simulation, dict):
            raise ValueError('exp_config.simulation must be an object')
        if 'norm_diff' in simulation.get('calibration', {}):
            raise ValueError('Move simulation.calibration.norm_diff to exp_config.norm_diff')
        validate_config(simulation, config_path=path)
    if 'norm_diff' in config:
        resolve_metric_settings(config)
    if 'lut_quality' in config:
        resolve_lut_quality_settings(config)
    if 'curve_lut_transfer' in config:
        resolve_curve_lut_transfer_settings(config)
    if 'curve_lut_compensation' in config:
        resolve_curve_lut_compensation_settings(config)
    if 'length_lutqual' in config:
        resolve_length_lutqual_settings(config)
    if 'light_distribution' in config:
        resolve_light_distribution_settings(config)
    sweep = config.get('length_normdiff')
    if sweep is not None:
        if not isinstance(sweep, dict):
            raise ValueError('exp_config.length_normdiff must be an object')
        lengths = sweep.get('lengths_mm')
        if not isinstance(lengths, list) or not lengths \
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value <= 0 for value in lengths):
            raise ValueError('length_normdiff.lengths_mm must be a nonempty list of positive lengths')
        if not isinstance(sweep.get('output_dir'), str) or not sweep['output_dir'].strip():
            raise ValueError('length_normdiff.output_dir must be a nonempty path string')
    evaluation = config.get('norm_diff_evaluation', {})
    if not isinstance(evaluation, dict):
        raise ValueError('exp_config.norm_diff_evaluation must be an object')
    for key in ('samples', 'simulation_config', 'output_file'):
        if key in evaluation and (not isinstance(evaluation[key], str) or not evaluation[key].strip()):
            raise ValueError(f'norm_diff_evaluation.{key} must be a nonempty path string')
    if evaluation.get('color_input', 'observed') not in ('observed', 'residual'):
        raise ValueError('norm_diff_evaluation.color_input must be observed or residual')
    if type(evaluation.get('validate_lut_noise', False)) is not bool:
        raise ValueError('norm_diff_evaluation.validate_lut_noise must be boolean')
    return copy.deepcopy(config)


def resolve_config_path(value: str | Path, config_path: str | Path = DEFAULT_EXP_CONFIG) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config_path).expanduser().resolve().parent / path
    return path.resolve()


def experiment_output(config: dict, config_path: str | Path = DEFAULT_EXP_CONFIG) -> Path:
    """Paths in the JSON are relative to that JSON, not the shell working directory."""
    return resolve_config_path(config['length_normdiff']['output_dir'], config_path)


CURVE_TRANSFER_DEFAULTS = {
    'radii_mm': [None, 200., 100., 50.],
    'training_pose_source': 'source_manifest', 'reference_pairing': 'material',
    'test_grid': {'u': [-.8, -.6, -.4, -.2, 0., .2, .4, .6, .8],
                  'v': [-.6, -.4, -.2, 0., .2, .4, .6], 'depths_mm': [.75, 1.25]},
    'metrics': {'aggregation': 'pose_equal_reference_area',
                'slope_tolerances': [.025, .05, .1, .2], 'bootstrap_repeats': 2000,
                'seed': 42, 'acceptance_slope_rmse': None, 'maximum_transfer_increase': None},
    'adaptation': {'enabled': False, 'pose_budgets': [16, 64, 256],
                   'reference_floor': .01, 'ridge': .001, 'maximum_iterations': 100},
}


def resolve_curve_lut_transfer_settings(config: dict, **overrides) -> dict:
    """Strict independent experiment settings; never import live physical defaults."""
    raw = copy.deepcopy(config.get('curve_lut_transfer', {}))
    raw.update({k: v for k, v in overrides.items() if v is not None})
    def merge(default, value, name):
        if not isinstance(value, dict) or set(value) - set(default):
            raise ValueError(f'{name}: expected object with keys {sorted(default)}')
        result = copy.deepcopy(default)
        for key, item in value.items():
            result[key] = merge(default[key], item, f'{name}.{key}') if isinstance(default[key], dict) else item
        return result
    s = merge(CURVE_TRANSFER_DEFAULTS, raw, 'curve_lut_transfer')
    def positive(value, name, zero=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
            raise ValueError(f'{name} must be finite and {">=0" if zero else ">0"}')
    def sequence(value, name):
        if not isinstance(value, list) or not value or len(set(value)) != len(value):
            raise ValueError(f'{name} must be a nonempty unique list')
    sequence(s['radii_mm'], 'radii_mm')
    if s['radii_mm'].count(None) != 1 or len(s['radii_mm']) < 2:
        raise ValueError('radii_mm requires a planar null and at least one curved radius')
    for r in s['radii_mm']:
        if r is not None:
            positive(r, 'radius_mm')
    if s['training_pose_source'] != 'source_manifest' or s['reference_pairing'] != 'material':
        raise ValueError('Only source_manifest poses and material reference pairing are supported')
    for key, axis in s['test_grid'].items():
        sequence(axis, f'test_grid.{key}')
        for value in axis:
            if key == 'depths_mm':
                positive(value, key)
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError(f'test_grid.{key} must be in [-1,1]')
    m, a = s['metrics'], s['adaptation']
    if m['aggregation'] != 'pose_equal_reference_area':
        raise ValueError('metrics.aggregation must be pose_equal_reference_area')
    sequence(m['slope_tolerances'], 'slope_tolerances')
    for v in m['slope_tolerances']:
        positive(v, 'slope_tolerances')
    for key in ('bootstrap_repeats', 'seed'):
        if type(m[key]) is not int or m[key] < (1 if key == 'bootstrap_repeats' else 0):
            raise ValueError(f'metrics.{key} must be an integer in range')
    for key in ('acceptance_slope_rmse', 'maximum_transfer_increase'):
        if m[key] is not None:
            positive(m[key], key, zero=True)
    if type(a['enabled']) is not bool:
        raise ValueError('adaptation.enabled must be boolean')
    sequence(a['pose_budgets'], 'pose_budgets')
    if any(type(v) is not int or v < 2 for v in a['pose_budgets']):
        raise ValueError('pose_budgets must contain integers >=2')
    positive(a['reference_floor'], 'reference_floor')
    positive(a['ridge'], 'ridge')
    if type(a['maximum_iterations']) is not int or a['maximum_iterations'] < 1:
        raise ValueError('maximum_iterations must be a positive integer')
    return s


CURVE_LUT_COMPENSATION_DEFAULTS = {
    'radii_mm': [None, 200., 100., 50.],
    'reference_pairing': 'material',
    'metrics': {'aggregation': 'pose_equal_reference_area',
                'slope_tolerances': [.025, .05, .1, .2],
                'bootstrap_repeats': 2000, 'seed': 42},
    'chunk_size': 8192,
}

NONLINEAR_DEFAULTS = {
    'steps': [1, 2, 3],
    'damping': 1e-4,
    'prior_weight': 0.0,
    'max_step': 0.5,
    'color_epsilon': 1e-4,
    'max_condition': 10000.,
    'minimum_normal_dot': 0.1,
    'active_set_update': True,
    'benchmark_repeats': 5,
    'warmup_repeats': 2,
}


def resolve_curve_lut_nonlinear_settings(settings: dict) -> dict:
    """Resolve and validate optional nonlinear comparison settings."""
    if not isinstance(settings, dict):
        raise ValueError('curve_lut_compensation.nonlinear must be an object')
    unknown = set(settings) - set(NONLINEAR_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown nonlinear settings: {sorted(unknown)}')
    resolved = copy.deepcopy(NONLINEAR_DEFAULTS)
    resolved.update(copy.deepcopy(settings))

    steps = resolved['steps']
    if not isinstance(steps, list) or not steps or any(
            type(step) is not int or step not in (1, 2, 3) for step in steps):
        raise ValueError('nonlinear.steps must be a nonempty list containing only integers 1, 2, and 3')
    if len(set(steps)) != len(steps):
        raise ValueError('nonlinear.steps must not contain duplicates')

    for key in ('damping', 'max_step', 'color_epsilon'):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            raise ValueError(f'nonlinear.{key} must be finite and nonnegative')
        resolved[key] = float(value)
    if resolved['max_step'] == 0:
        raise ValueError('nonlinear.max_step must be finite and positive')
    if not 0 < resolved['color_epsilon'] < 0.5:
        raise ValueError('nonlinear.color_epsilon must be in (0,0.5)')
    prior_weight = resolved['prior_weight']
    if isinstance(prior_weight, bool) or not isinstance(prior_weight, (int, float)) \
            or not math.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError('nonlinear.prior_weight must be finite and nonnegative')
    resolved['prior_weight'] = float(prior_weight)

    max_condition = resolved['max_condition']
    if isinstance(max_condition, bool) or not isinstance(max_condition, (int, float)) \
            or not math.isfinite(max_condition) or max_condition <= 1:
        raise ValueError('nonlinear.max_condition must be finite and >1')
    resolved['max_condition'] = float(max_condition)
    minimum_normal_dot = resolved['minimum_normal_dot']
    if isinstance(minimum_normal_dot, bool) or not isinstance(minimum_normal_dot, (int, float)) \
            or not math.isfinite(minimum_normal_dot) or not 0 < minimum_normal_dot <= 1:
        raise ValueError('nonlinear.minimum_normal_dot must be finite and in (0,1]')
    resolved['minimum_normal_dot'] = float(minimum_normal_dot)

    if type(resolved['active_set_update']) is not bool:
        raise ValueError('nonlinear.active_set_update must be boolean')

    for key, minimum in (('benchmark_repeats', 1), ('warmup_repeats', 0)):
        value = resolved[key]
        if type(value) is not int or value < minimum:
            raise ValueError(f'nonlinear.{key} must be an integer >= {minimum}')
    return resolved


def resolve_curve_lut_compensation_settings(config: dict, **overrides) -> dict:
    """Validate fixed transfer conditions for LUT compensation."""
    raw = copy.deepcopy(config.get('curve_lut_compensation', {}))
    if isinstance(raw, dict) and 'matrix' in raw:
        raise ValueError('已移除矩阵补偿，请移除matrix配置；非线性反演使用nonlinear配置')
    raw.update({key: value for key, value in overrides.items() if value is not None})
    if 'matrix' in raw:
        raise ValueError('已移除矩阵补偿，请移除matrix配置；非线性反演使用nonlinear配置')
    has_nonlinear = 'nonlinear' in raw
    nonlinear = raw.pop('nonlinear', None)

    def merge(default: dict, value: dict, name: str) -> dict:
        if not isinstance(value, dict) or set(value) - set(default):
            raise ValueError(f'{name}: expected object with keys {sorted(default)}')
        result = copy.deepcopy(default)
        for key, item in value.items():
            result[key] = merge(default[key], item, f'{name}.{key}') \
                if isinstance(default[key], dict) else item
        return result

    settings = merge(CURVE_LUT_COMPENSATION_DEFAULTS, raw, 'curve_lut_compensation')
    radii = settings['radii_mm']
    if not isinstance(radii, list) or len(radii) < 2 or radii.count(None) != 1:
        raise ValueError('curve_lut_compensation.radii_mm must contain exactly one null flat condition and at least one radius')
    normalized_radii = []
    for value in radii:
        if value is None:
            normalized_radii.append(None)
        elif isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value <= 0:
            raise ValueError('curve_lut_compensation.radii_mm values must be null or finite positive numbers')
        else:
            normalized_radii.append(float(value))
    if len({('flat' if value is None else value) for value in normalized_radii}) != len(normalized_radii):
        raise ValueError('curve_lut_compensation.radii_mm must contain unique conditions')
    settings['radii_mm'] = normalized_radii
    if settings['reference_pairing'] != 'material':
        raise ValueError('curve_lut_compensation.reference_pairing must be material')
    metrics = settings['metrics']
    if metrics['aggregation'] != 'pose_equal_reference_area':
        raise ValueError('metrics.aggregation must be pose_equal_reference_area')
    tolerances = metrics['slope_tolerances']
    if not isinstance(tolerances, list) or not tolerances:
        raise ValueError('metrics.slope_tolerances must be a nonempty array')
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in tolerances):
        raise ValueError('metrics.slope_tolerances must be finite and positive')
    if len(set(tolerances)) != len(tolerances):
        raise ValueError('metrics.slope_tolerances must be unique')
    for key, minimum in (('bootstrap_repeats', 1), ('seed', 0)):
        value = metrics[key]
        if type(value) is not int or value < minimum:
            raise ValueError(f'metrics.{key} must be an integer >= {minimum}')
    if type(settings['chunk_size']) is not int or settings['chunk_size'] < 1:
        raise ValueError('curve_lut_compensation.chunk_size must be a positive integer')
    if has_nonlinear:
        settings['nonlinear'] = resolve_curve_lut_nonlinear_settings(nonlinear)
    return settings
