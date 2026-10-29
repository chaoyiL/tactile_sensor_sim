# Xense 后端扫描验证结果

本地实测：16 姿态小扫描和默认配置的 275 姿态完整扫描均完成，后者的 55 个独立验证姿态全部通过，整体 RMS 9.2939°，越界比例约 0.00058%；存档 NormDiff（含噪声 LUT 验证）和 LUT 质量评测均完成。一次人工中断后的跨进程逐姿态续跑也通过测试。原始参考图与姿态图位于各自的新输出目录；自动检查覆盖图像坐标、重复帧和球冠深度，marker 无点阵结论来自原图目视核验。

## 验证条件与复现说明

此后端使用 MuJoCo 3.3.7 正交相机输出球压入深度，再由 Xensim 0.3.0 `SimpleSensor` 输出触觉 RGB。`SimpleSensor.get_image()` 的原始图像已在零压入和中心、偏心球压入下目视核验：没有 marker 点阵。本实现没有用背景差分、遮罩或图像修补去除 marker。FEM 模式尚未接入，因为其实际表面法向与材料点对应尚未从公开接口验证。

运行环境为 Windows x64。仓库主环境保持 Python 3.13；厂商 wheel 仅支持 CPython 3.9，因此另设 `.venv-xense`。以下依赖组合已在本机实际运行；`mujoco-sim/xense-mujoco` 包声称 Python >=3.10，不能与现有 Xensim wheel 放在同一解释器，worker 直接使用 MuJoCo 与 Xensim 的公开 API。

```powershell
$env:UV_CACHE_DIR=(Resolve-Path .uv-cache).Path
$env:UV_PYTHON_INSTALL_DIR=(Join-Path (Get-Location) '.uv-python')
uv python install 3.9
uv venv --python .uv-python/cpython-3.9.25-windows-x86_64-none/python.exe .venv-xense
uv pip install --python .venv-xense/Scripts/python.exe --only-binary :all: -r integrations/xense_mujoco_runtime/requirements-win-cp39.txt mujoco-sim/xensim/xensim-0.3.0-cp39-cp39-win_amd64.whl
```

`requirements-win-cp39.txt` 记录本机成功运行时的已安装版本；最后一个 wheel 在仓库内。若 uv 重新解析依赖发生冲突，应先核对该 wheel、Python ABI 和锁定依赖，而不是升级主环境。GPU 加速不是此 SimpleSensor 扫描的前提；MuJoCo 离屏 OpenGL 和 Qt/Xensim 仍需要可用的图形驱动。

单帧和完整标定使用独立配置 [configs/xense.json](../configs/xense.json)；扫描、拟合和留出验证由 [configs/calibration.json](../configs/calibration.json) 控制。其胶体为 17.3 × 29.14 mm、球半径 6 mm。该尺寸和 Xensim bundled `g1-ws_table.npz` 配套使用；请勿把解析配置中的灯带参数直接移植到这里。

已完成的正式扫描集中在 [results_lut/xense](../results_lut/xense)；原有解析标定原样归档为 [results_lut/analytic/previous_failed](../results_lut/analytic/previous_failed)。目录索引说明两者的验证状态。再次运行完整扫描时须用新的空目录，例如：

```powershell
.\.venv\Scripts\python.exe simulate.py --backend xense --config configs/xense.json --reference --out results_render/xense/new_run
.\.venv\Scripts\python.exe calibrate_lut.py --backend xense --config configs/xense.json --dry-run
.\.venv\Scripts\python.exe calibrate_lut.py --backend xense --config configs/xense.json --out results_lut/xense/run_02
```

单帧产生原始 `image.png`、`geometry_gt.npz` 和元数据，不产生伪造的辐射量 NPY。扫描会保存 `reference.png`、高精度 `reference_grid.npz`、每姿态原始 PNG、逐姿态 checkpoint、完整 `normal_samples.npz`、训练 LUT、逐姿态验证报告、歧义输入和可视化。`normal_samples.npz` 格式版本 2 包含 `reference_grid_color`、`reference_grid_valid`、后端、观测模型和运行签名；`allow_pickle=False` 可读取。训练与验证按完整姿态隔离。仅逐姿态验证均达标才发布 `normal_lut.npz`；验证失败保留训练/诊断产物，命令以非零状态结束。

Xensim 返回 `uint8 RGB`，SDK 传递函数未有公开确证。本实现对 RGB8 作标准逆 sRGB 变换，作为**分析约定**，并在参考和每个姿态的固定材料点上计算有符号 `observed-reference`。它不是物理线性辐照度，也不能直接与解析/Zemax 辐射量解释为同一光学模型。实际球深度逐姿态在接触内部与解析球冠比较，最大绝对误差超过 0.05 mm 时拒绝该姿态，避免以错误几何标签拟合。

用独立检验配置 [norm_diff.json](../configs/norm_diff.json) 和 [lut_quality.json](../configs/lut_quality.json) 对**已保存的同批样本**计算现有指标：

```powershell
.\.venv\Scripts\python.exe -m experiment.norm_diff --lut-results results_lut/xense/run_02 --validate-lut-noise
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/xense/run_02 --dry-run
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/xense/run_02
```

NormDiff 的 observed RGB 从样本存档里的零压入参考恢复，无需启动 MuJoCo、Xensim 或 CUDA。LUT 质量直接复用原始样本与训练 LUT，默认沿用保存的完整姿态留出。两者的数值定义与解析后端一致，但只有明确对齐几何、采样、RGB 域和物理条件后才可作严格跨后端对比。

当前厂商光学表没有经验证的 LED 长度、功率、位置或任意胶体尺寸控制；`experiment.length_normdiff` 和 `experiment.length_lutqual` 会在 dry-run 之前明确拒绝 MuJoCo 快照，避免把数值缩放误报为原长度/灯带实验。需要这些参数对应的厂商光学表和能力证明后才能完成这一阶段。当前实现不提供 FEM 实际变形真值或自动 marker 检测。采集中断后可对同一输出目录重新执行同一命令：先验证配置、厂商表、worker、采样器和拟合器签名，再从已完整保存的姿态 checkpoint 续跑；已完成的运行和无有效 manifest 的非空目录会被拒绝。
