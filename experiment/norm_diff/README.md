# Saved-sample NormDiff evaluation

```powershell
.\.venv\Scripts\python.exe -m experiment.norm_diff --lut-results results_lut/xense --out results_lut/xense/evaluation/norm_diff_new.json
.\.venv\Scripts\python.exe -m experiment.norm_diff --lut-results results_lut/xense --exp-config configs/norm_diff.json --out results_lut/xense/evaluation/norm_diff_run_02.json
.\.venv\Scripts\python.exe -m experiment.norm_diff --lut-results results_lut/xense --noise-fraction 0.01 --out results_lut/xense/evaluation/norm_diff_noise_1pct.json
.\.venv\Scripts\python.exe -m experiment.norm_diff --lut-results results_lut/xense --validate-lut-noise --out results_lut/xense/evaluation/norm_diff_noise_validation.json
```

`__main__.py` is the entry point. `runner.py` loads saved samples and reconstructs
observed RGB using a newly rendered, matching zero-depth reference. It does not
re-render indentation poses. `metric.py` is also used by the length experiment.

Evaluation parameters come from `norm_diff` in `configs/norm_diff.json`;
input/output defaults come from `norm_diff_evaluation`. Relative paths in that
JSON are resolved against its directory. CLI options override the file.

`--config` is the physical simulation config matching the NPZ signature, NOT
the metric config; `--exp-config` selects metric and experiment settings. The
independent `simulation` snapshot is used by the length experiment, not to
silently reinterpret an existing sample archive. Geometry/config signature
checks still reject incompatible samples before any reference is rendered.

By default the metric uses observed post-tone-mapping, sRGB8-round-trip linear
RGB, not residual RGB or optical RAW. Each occupied theta/phi bin chooses anchor
samples; a per-channel relative RGB box searches every valid sample in the SAME
pose. Weighted theta/phi coordinate ranges are averaged within targets and poses,
then negated. A score closer to zero means less confusion under this definition.
The existing phi seam remains uncorrected. This is not a held-out LUT error.

`noise_fraction=0.03` means +/-3% half-width, not a 3% total interval or RMS noise.
No random colors are generated for NormDiff itself. `--validate-lut-noise` adds
the existing optional held-out LUT test, with independent uniform RGB perturbations
inside the same noise box. Use `--no-validate-lut-noise` to override a true file value.

A null theta ceiling uses the observed sample maximum. The length experiment's
null setting instead fixes a common sphere-cap bound across lengths. For an
exact comparison, supply the sweep's recorded `theta_ceiling_degrees` explicitly.

Default JSON output is `experiment/norm_diff/results/norm_diff.json`. Existing
calibration reports and historical experiment results are not modified unless
their exact paths are explicitly supplied with `--out`.
