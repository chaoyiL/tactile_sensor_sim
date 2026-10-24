# 仿真、标定和检验入口

运行以下命令前，按需安装解析 GPU、Zemax 或 Xense 的依赖。所有命令从仓库根目录执行。

## 触觉仿真

三种后端分别编辑 `configs/analytic.json`、`configs/zemax.json`、`configs/xense.json`：

```powershell
.\.venv\Scripts\python.exe simulate.py --backend analytic --out results_render/analytic/new_run
.\.venv\Scripts\python.exe simulate.py --backend zemax --out results_render/zemax/new_run
.\.venv\Scripts\python.exe simulate.py --backend xense --reference --out results_render/xense/new_run
```

可用 `--config` 显式指定相应后端的配置文件；入口会拒绝后端与配置不一致。Xense 使用不叠加 marker 的 SimpleSensor 图像。

## LUT 标定

`configs/calibration.json` 控制所有后端共用的扫描姿态、采样、拟合、完整姿态留出验证及可视化。入口必须显式指定后端；可用 `--calibration-config` 和 `--config` 分别覆盖标定与仿真配置。

```powershell
.\.venv\Scripts\python.exe calibrate_lut.py --backend analytic --dry-run
.\.venv\Scripts\python.exe calibrate_lut.py --backend zemax --dry-run
.\.venv\Scripts\python.exe calibrate_lut.py --backend xense --dry-run
.\.venv\Scripts\python.exe calibrate_lut.py --backend analytic --out results_lut/analytic/run_02
```

默认结果目录依次为 `results_lut/analytic/`、`results_lut/zmax/`、`results_lut/xense/`。目录非空时应指定新的运行子目录；Xense 仅允许兼容的未完成运行续跑。只有留出验证通过才发布 `normal_lut.npz`。Zemax 球扫描使用相同的 LUT 拟合与验证代码，逐姿态通过 OpticStudio 追迹并在材料网格上采样；它需要可用的 ZOS-API 和许可证，不能由 dry-run 代替实机验证。

## LUT 检验

检验入口必须显式指定 LUT 结果目录。`lut_quality` 可直接检验已通过或验证失败的标定，只要样本和训练 LUT 可用；报告保留原标定状态，不会发布失败的 LUT。`norm_diff` 仍要求正式结果。默认从结果目录读取 manifest、样本和训练 LUT，并写入对应的 `evaluation/` 子目录。检验参数分别在 `configs/norm_diff.json` 和 `configs/lut_quality.json`。

```powershell
.\.venv\Scripts\python.exe -m experiment.norm_diff --lut-results results_lut/xense --validate-lut-noise --out results_lut/xense/evaluation/norm_diff_run_02.json
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/xense --dry-run
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/xense --out results_lut/xense/evaluation/lut_quality_run_02
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/analytic/previous_failed --dry-run
```

`--out` 必须指向新文件或空目录。NormDiff 是同姿态 RGB 混淆度；LUT 质量检验分析多值映射及实际查表误差。两项指标各自保留原有数值定义，不把训练 LUT 或失败诊断 LUT 当作正式发布结果。

## analytic 曲面仿真

```powershell
.\.venv\Scripts\python.exe simulate_curve.py --radius-mm 100 --dry-run
.\.venv\Scripts\python.exe simulate_curve.py --radius-mm 100 --reference --out results_curve_render/analytic/new_run
```

曲面入口读取 `configs/analytic.json`；`--flat` 生成平面对照。两个仿真入口省略 `--out` 时，根据最终生效的弯曲参数选择仓库下的 `results_render/<backend>/<唯一运行标识>/`（无弯曲）或 `results_curve_render/analytic/<唯一运行标识>/`（有弯曲）。平面与曲面统一使用 `center_uv∈[-1,1]²` 指定位置，参考帧保持当前曲率并留在同次运行内。显式 `--out` 优先，非空输出目录拒绝覆盖；历史结果不搬移。目前不支持 Zemax/Xense 曲面或曲面 LUT 标定与检验。详见[使用说明](analytic_curved_surface_usage.md)和[验证记录](analytic_curved_surface_validation.md)。
