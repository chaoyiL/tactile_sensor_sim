# LUT 检验与独立实验

针对已有 LUT 结果的检验位于 `experiment/`，每项都有独立入口和 `configs/` 内的实验配置。检验显式选择 `--lut-results`，默认输出到该结果的 `evaluation/<experiment_name>/`，并核对标定 manifest、样本及 LUT 元数据。`lut_quality` 可检验验证失败但样本和训练 LUT 完整的运行，并保留其失败状态；`norm_diff` 仍要求正式结果。现有通过验证的 Xense 结果在 `results_lut/xense/`。

```powershell
.\.venv\Scripts\python.exe -m experiment.norm_diff --lut-results results_lut/xense --out results_lut/xense/evaluation/norm_diff_new.json
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/xense --dry-run
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/xense --out results_lut/xense/evaluation/lut_quality_new
```

`norm_diff` 的参数在 `configs/norm_diff.json`；它衡量同一压入姿态内 RGB 候选的法向混淆，采用负值约定，**不是 LUT RMS**。可选的 `--validate-lut-noise` 使用训练 LUT 对完整留出姿态进行加噪验证。`lut_quality` 的参数在 `configs/lut_quality.json`；默认使用训练 LUT 的完整姿态留出，计算节点歧义及实际插值查表误差。已有报告不会被默认入口覆盖；重新检验时提供新的 `--out` 路径。

`length_normdiff`、`length_lutqual` 是独立的长度扫描实验，不属于对某个现有正式 LUT 的检验。它们仍分别由 `python -m experiment.length_normdiff` 和 `python -m experiment.length_lutqual` 运行，配置位于 `configs/length_normdiff.json`、`configs/length_lutqual.json`，默认输出在各自包内的 `results/`。这些配置保留独立物理快照，不随当前仿真配置自动变化。长度扫描目前只支持解析后端；Xense 的固定胶体与标定表不支持任意长度灯带扫描。

各实验细节见对应包内的 README。历史实验输出原样保留，旧合并配置归档在 `configs/legacy/`。

See [light_distribution](light_distribution/README.md) for full-scan, power-only analytic light optimization against a saved target LUT. Config: configs/light_distribution.json.

新增 [curve_lut_transfer](curve_lut_transfer/README.md)：冻结已发布的 analytic 平面 LUT，比较弯曲后的色差查表与局部切平面坡度真值，并按原姿态/划分重标定曲面 LUT。配置为 `configs/curve_lut_transfer.json`，入口为 `python -m experiment.curve_lut_transfer --lut-results results_lut/analytic --dry-run`。输出位于源结果的 `evaluation/curve_lut_transfer/<run>/`；原始平面/曲面帧分别保存到 `results_render` / `results_curve_render`。主指标为姿态等权、参考面积加权的坡度向量 RMSE；同时输出位置图、交叉查表矩阵及共同物理 RGB 上的 LUT 函数差异。
