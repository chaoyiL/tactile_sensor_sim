# Surface length / NormDiff experiment

Geometry update: new runs use `vertical_edge_strips_v1` (fixed vertical,
inward-facing emitting planes; offset is the horizontal edge-to-plane gap;
width is along Z). The completed run documented below used the previous tilted
geometry and remains historical data, not a result for the new model. Re-run
with a new output directory. The default now lives inside this experiment package.
Checkpoints include the geometry model and geometry source hash; old points
cannot be silently resumed under the new model.

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m experiment.length_normdiff
```

The experiment reads only `configs/length_normdiff.json`, not the main simulation
configuration. Its independent `simulation` snapshot defines geometry, lights,
camera, rendering and calibration sampling. `length_normdiff.lengths_mm` defines
the scan points (initially `21.8, 26.8, 31.8, 36.8, 41.8, 46.8, 51.8, 55.9` mm).
Preview exact pose counts with `--dry-run`; use `--exp-config` for a different
experiment JSON. Editing `configs/analytic.json` no longer changes this experiment.

Only three simulation fields change:

- `surface.size_mm[0] = length`;
- every top/bottom strip: `length_mm += length - original_surface_length`.

Thus the current top/bottom strips remain 0.1 mm longer than the surface. The
original JSON, rendering outputs, sample archives, and LUTs are not modified.
Left/right strips move with the surface edges through the existing geometry
code. Their lengths, all strip powers, exposure, camera, light quadrature counts,
and scan/material-grid spacings are unchanged.

## Evaluation

Each length is **newly rendered**, including its zero-depth reference and every
configured scan pose. Every valid contact sample is used. A larger surface
naturally has more scan poses; the score is an equal mean over poses, matching
`experiment.norm_diff.metric.compute_norm_diff`, not a sample-count-weighted mean. This is a
NormDiff experiment, not a new LUT fit or calibration validation.

The RGB input matches `python -m experiment.norm_diff --color-input observed`: archived
float32 residuals promoted to float64, plus their matching reference RGB. It is
the renderer's sRGB8 round-trip, post-tone-mapping linear RGB, not optical RAW.
The existing metric is reused without changing its noise box or angular-range
definition. In particular, the existing linear azimuth range and its +/-180 deg
seam remain part of this metric; the values are not LUT RMS angular errors.

All evaluation settings come from `exp_config.norm_diff`, including relative
RGB half-width, noise floor, angular weights, angular bins and anchors per bin.
CLI flags override that section. Initial values remain +/-3%, zero floor,
unit weights, 12 x 24 bins and one anchor per occupied bin. A null polar-angle
ceiling uses the common physical bound `acos(1 - max_scan_depth / sphere_radius)`.
An explicit ceiling must cover that bound and is shared by all lengths.
This common physical bound avoids changing the target grid between lengths;
it may cause a slight difference from archived reports using their observed
sample maximum as the angular ceiling.

The PNG contains a full-surface curve and a fixed-centre control curve. The
control renders the same `(0, 0)` indentation centre at each configured depth
(one pose per configured depth). It does not select a varying subset of
nearby scan poses. Per-pose theta/phi components and centre coordinates are
saved in each length's JSON.

The JSON also records each strip's irradiance at the flat surface centre, in
W/m^2, using the renderer's strip quadrature. This is a diagnostic before camera
response/tone mapping, not an extra NormDiff input. A darker centre alone does
not prove a worse metric: with purely relative noise, uniformly scaling all
candidate RGB values and their noise widths leaves candidate membership
unchanged (before quantization/nonlinear processing).

## Outputs and resuming

Default output: `experiment/length_normdiff/results/`, controlled by
`length_normdiff.output_dir`. Paths in the JSON are relative to the JSON file;
CLI paths are relative to the current working directory. Historical results
remain unchanged at their existing `experiment/results/` paths.

- `length_normdiff.png` and `.svg`: the requested figure;
- `length_normdiff.csv`: numerical scores, theta/phi components and counts;
- `length_<L>.json`: full and centre metrics, per-pose results, lighting diagnostic;
- `manifest.json`: original configuration, noise, scan range and source hashes;
- `summary.json`: combined metadata and curve values.

Completed lengths are checkpointed. Re-running the exact command resumes them.
If the configuration, lengths, evaluation settings or relevant source hashes
change, use a new output directory; incompatible checkpoints are rejected.

```powershell
.\.venv\Scripts\python.exe -m experiment.length_normdiff --lengths 21.8 30 40 50 55.9 --out experiment/length_normdiff/results/custom_lengths
.\.venv\Scripts\python.exe -m experiment.length_normdiff --noise-fraction 0.01 --out experiment/length_normdiff/results/noise_1pct
.\.venv\Scripts\python.exe -m unittest tests.test_length_normdiff tests.test_experiment_config -v
```

An optional, fast local-optics diagnostic reads the saved manifest (not a possibly
changed current config), and writes `center_light_diagnostics.json`:

```powershell
.\.venv\Scripts\python.exe -m experiment.length_normdiff --diagnostics-only
# A specific completed run can also be selected:
.\.venv\Scripts\python.exe -m experiment.length_normdiff --diagnostics-only --manifest experiment/length_normdiff/results/manifest.json
```

It computes the Jacobian of log RGB versus surface slopes at the flat centre,
including camera response and tone mapping but excluding 8-bit quantization.
Its smallest singular value measures local relative-RGB sensitivity in the
least-sensitive slope direction. This is not NormDiff or a calibration error,
and does not represent finite-indent geometry or changes in self-shadowing.

## Interpretation limits

NormDiff closer to zero means less normal-coordinate confusion under the stated
RGB noise model; a factor-of-two difference is not a factor-of-two reduction in
LUT prediction error. Compare held-out LUT errors separately before claiming
better calibration.

The current light arrangement is left-blue, right-white, top-red, bottom-green.
Lengthening the surface moves blue/white strips away from its centre. Lengthening
the red/green strips also spreads their **fixed total power** over more length.
The experiment therefore tests this combined length policy, not distance alone
or a constant-W/mm strip. The rectangle's 180-degree symmetry does not make a
90-degree swap between long and short edges equivalent.

## Completed run (2026-10-21)

The default eight lengths were fully evaluated: 7,280 scan poses, 40 additional
centre poses, and 14,770,159 valid contact samples across those two groups. Eight
additional flat-reference frames were rendered. No candidate subsampling or LUT
training was performed. The configuration and source hashes are in the manifest.

For this saved configuration the result does **not** reproduce the proposed
square-better trend: full-surface NormDiff improves monotonically from
`-40.355356` at 21.8 mm to `-17.587323` at 55.9 mm. The fixed-centre score improves
from `-16.832542` to `-0.969236`. These are coordinate-confusion scores under
the existing metric, not calibration RMS errors.

Centre illumination does decrease: the blue left strip's flat-centre irradiance
falls from 58.379 to 6.462 W/m^2. But the local, prequantization relative-RGB slope
Jacobian's smallest singular value rises from 1.087 to 3.185 after camera response
and Reinhard tone mapping. This supports a competing local mechanism: lower
brightness need not mean lower *relative* directional sensitivity. The local
calculation alone does not establish the cause of the complete-scan trend.

Consequently, this run does not support centre degradation over the tested range
under the current noise/power/lighting policy. It neither proves that rectangles
are generally better nor explains earlier runs with a different light arrangement,
noise model, angular-grid policy or exposure. Existing archived NormDiff JSONs
should be checked for their own noise setting before comparing them to this run.
