# 仿真 LUT 标定使用说明

LUT 标定需要 CUDA 版 PyTorch、SciPy 和 Matplotlib。使用锁文件同步完整环境：

```powershell
uv sync --cache-dir .uv-cache --locked --extra gpu --extra calibration
```

先检查扫描范围和资源规模；dry-run 不渲染、不拟合，也不会创建输出目录：

```powershell
uv run --cache-dir .uv-cache --extra gpu --extra calibration python calibrate_lut.py --config config.example.json --dry-run
```

执行完整标定：

```powershell
uv run --cache-dir .uv-cache --extra gpu --extra calibration python calibrate_lut.py --config config.example.json --out results_lut
```

程序只支持 analytic CUDA 标定。它先渲染一次零压入 reference，再按深度、行、列扫描标定球。每个深度的第一个接触圆与左边和上边相切。颜色样本是在固定材料网格上计算的 signed linear RGB 差值；坡度来自已知球冠几何，不需要人工圈选或 marker mask。

主要输出包括：

- `normal_samples.npz`：紧凑的颜色、坡度、材料坐标及图像分组样本。
- `normal_lut_train.npz`：整图留出验证使用的训练 LUT。
- `normal_lut.npz`：验证通过后才发布的正式 LUT；失败时不会生成。
- `normal_lut_diagnostic.npz`：验证失败且启用可视化时生成，元数据明确标为失败诊断。
- `verification/<pose>.png`：每个姿态唯一的一张色差预览，圈出几何标定范围。
- `verification/report.csv` 与 `validation.json`：逐姿态采样和整图留出指标。
- `visualization/`：全局拟合、held-out、LUT 切片、颜色歧义及逐姿态色差/坡度图。
- `manifest.json`：输入签名、所有姿态、产物、验证状态、耗时和 CUDA 峰值显存。

输出中不会创建 `frames/`、整体曲面重建 NPZ、逐帧 GT、reference PNG 或 irradiance NPY。若验证失败，命令以非零状态结束，但保留训练 LUT、诊断、预览和报告供排查。

普通 analytic 渲染默认使用配置中的 `render.device`。显式 CPU 对照可使用：

```powershell
uv run python simulate.py --config config.example.json --backend analytic --device cpu --out results_cpu
```

运行全部回归检查：

```powershell
uv run --cache-dir .uv-cache --extra gpu --extra calibration python -m unittest discover -s tests -v
```
