# analytic 弯曲表面实施验证

验证日期：2026-10-23。本文记录 analytic 曲面几何、渲染、坐标迁移及输出的实施验证；运行和配置见 [使用说明](analytic_curved_surface_usage.md)。

## 实施结果

- 新增 `simulate_curve.py`，支持 X 向等曲率弯曲、局部法向真实球面压入、随边缘运动的灯带、同曲率参考帧、解析遮挡和 STL 导出。
- CPU、普通 CUDA、编译照明内核与 CUDA Graph 已接入同一曲面定义。相机、阴影射线使用有限圆柱面及有效球面接触区求交，选择最近有效交点。
- `configs/analytic.json` 集中保存弯曲参数；默认 `radius_mm=null` 保持平面，通过配置或 `--radius-mm` 开启弯曲。针孔半宽及网格步长分别移至 analytic 自身的 `camera` 和 `render` 段。
- 三个后端的当前配置统一为 `center_uv`、`direction_normal`。旧平面的 XY 毫米坐标及世界 Z 方向仍可读取；有限曲率拒绝这些旧字段。平面扫描保留原姿态序列、步长、数量和数据划分，并额外记录归一化位置。
- 曲面输出具有独立模型版本、配置签名、实际源码哈希及运行状态。非空目录拒绝写入；原有结果及配置快照未迁移或覆盖。

## 自动化测试

全量测试最终结果：**134 项通过，0 失败，0 错误，0 跳过**。包含本次新增的 23 项坐标、曲面几何和渲染测试，以及既有平面、标定、光源、GPU/编译、Xense 和实验测试。

主要覆盖：

- 归一化坐标往返、旧位置迁移、非法输入与字段冲突、704 个 analytic 平面扫描姿态保持一致。
- 圆弧弧长、逆变换、极大半径平面极限、真实球面半径、法向推进深度、接触边界与连续分支校验。
- 射线已知交点、退化情形、最近有效交点、曲面遮挡、接触区外基准面、灯带朝向、连续/离散功率及间距。
- 相机投影、曲面坐标和真值、同曲率零压入参考、无 Torch 的 dry-run、输出完整性和防覆盖。
- CPU/CUDA 对照、float32 辐射路径、阴影与非整数发光指数、编译内核、执行图更新姿态，以及改变曲率时拒绝复用旧上下文。
- 不支持曲面的平面标定/检验入口明确拒绝有限曲率；旧诊断 LUT 的部分配置快照仍可用于其原有样本评估。

复现常规测试命令：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_surface_coordinates tests.test_curve_geometry tests.test_curve_renderer -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

本机受管 Windows 沙箱下，Python 临时目录的 `0700` 权限导致部分既有测试无法写入。全量验证把临时目录限制在仓库 `tests/` 下，并仅在该次测试进程中将这些目录的创建模式改为继承工作区权限的 `0777`；测试断言未改动。新曲面测试自带等效工作区临时目录 fixture。标准环境可直接使用上述命令。

首次全量执行为 133 项、1 个错误：曲面检查访问旧 LUT 的部分配置快照时缺少 `surface`。现已保持旧快照读取兼容，同时新增有限曲率 LUT 拒绝测试。最终 134 项结果保存在 [完整日志](../results_curve/validation_20261023_full_resolution/unittest_final.log) 和 [摘要](../results_curve/validation_20261023_full_resolution/unittest_final_summary.json)；首次失败日志另行保留，未覆盖。另将一项既有后端测试改用固定 fixture，避免示例配置迁移后产生新旧位置字段冲突；数值断言和 `tests/compile_reference/` 均未修改。

## 完整分辨率数值与耗时

使用实际 `configs/analytic.json` 的 **640×480 像素、52 个光源积分样本**，不减少分辨率、光源、阴影或采样范围。设置 `R=100 mm`、`center_uv=[0.3,-0.2]`，分别验证 `depth_mm=1` 及 `0`。两帧均有 66,289 个有效像素。

设备为 NVIDIA GeForce RTX 5050 Laptop GPU；PyTorch `2.12.1+cu130`、CUDA `13.0`、triton-windows `3.7.1.post27`。CPU 实际使用 float64 几何/辐射；GPU 使用 float64 几何、默认 float32 辐射。

| 模式 | 偏心压入中位耗时 | 同曲率参考中位耗时 |
|---|---:|---:|
| CPU | 2.6421 s | 1.3494 s |
| 普通 CUDA | 0.1253 s | 0.1224 s |
| 编译照明 + CUDA Graph | 0.1396 s | 0.1397 s |

每种模式测量 3 次；GPU 先执行 2 次完成编译、预热及捕获，计时前后同步 GPU。耗时只包含渲染，排除上下文构建、编译准备、GPU 回读及文件写入。编译模式没有回退，每个测试场景最终记录 20 次执行图重放。**本场景编译加执行图没有比普通 CUDA 更快**，不能据此宣称获得额外加速。

相对 CPU 的两种 GPU 路径均通过以下预设判据：

| 量 | 判据 | 两个场景/两种 GPU 模式中的最大绝对误差 |
|---|---|---:|
| 有效像素、接触掩膜 | 完全相同 | 0 |
| 世界位置 | `atol=1e-8 mm, rtol=0` | `2.56e-13 mm` |
| 光学法向 | `atol=1e-9, rtol=0` | `4.27e-14` |
| 传感器辐照度 | `atol=1e-9 W/m², rtol=3e-6` | `3.45e-9 W/m²` |
| 表面辐亮度 | `atol=1e-5 W/(m²·sr), rtol=3e-6` | `6.58e-4 W/(m²·sr)` |

辐射量按逐元素组合阈值 `abs(error) <= atol + rtol*abs(reference)` 判断，最大绝对误差不能单独与 `atol` 比较。float64 辐射路径另由固定 fixture 使用 `rtol=1e-8, atol=1e-10` 对照，不能用它代替上述默认 float32 路径验证。

[数值和耗时报告](../results_curve/validation_20261023_full_resolution/validation.json) 保存实际配置、容差、每次耗时、编译状态和源码哈希；[逐像素原始数值](../results_curve/validation_20261023_full_resolution/numeric_comparison.npz) 保存各模式的辐射量和有效像素。复现时必须使用新的输出目录：

```powershell
.\.venv\Scripts\python.exe -m tests.benchmark_curve_renderer --out results_curve/validation_new
```

## 入口及输出核对

最终示例运行：

```powershell
.\.venv\Scripts\python.exe simulate_curve.py --radius-mm 100 --reference --out results_curve/analytic/curve_r100_verified_20261023
```

输出位于 [示例运行目录](../results_curve/analytic/curve_r100_verified_20261023/)，包含压入及参考帧的显示图、辐射量、几何真值、STL、帧元数据和根目录 `run.json`。两帧保持相同弯曲、灯带和相机；参考帧深度为零，具有自己的配置签名，与压入帧共享运行签名。

analytic、Zemax、Xense 平面标定 dry-run 及配置路径检查通过；当前 analytic / Xense 配置的扫描量分别为 704 / 275 姿态。Xense 已执行既有小规模 SDK 渲染、深度/重复性及扫描续跑测试；未运行完整正式标定。Zemax 本次未执行实际光学追迹，不将入口校验算作 Zemax 渲染验证。

本次未实现 Zemax/Xense 曲面渲染、曲面 LUT 标定或曲面 LUT 检验。模型是规定几何的球面接触，不包含弹性力学、接触边缘平滑或过深压入导致的折叠；非法接触与灯带几何直接报错。历史 LUT 不自动用于新曲面。

## 输出路径规则后续调整

按用户要求，两个仿真入口现默认按最终生效的曲率写入 `results_render/<backend>/<唯一运行标识>/`（无弯曲）或 `results_curve_render/analytic/<唯一运行标识>/`（有弯曲），显式 `--out` 优先。本文前述验证产物和日志仍指向当时的实际路径，没有移动或重新标记历史结果。

本次针对输出路径执行 10 项相关测试，全部通过：覆盖配置/CLI 覆盖后的平面与曲面分类、零压入参考仍属曲面、dry-run 不写入、显式路径优先、重复运行分离、非空目录保护，以及原有曲面 CPU 入口与三个后端的配置/dry-run 回归。命令如下；此次路径调整未重新执行 GPU 性能对照或完整标定。

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_render_output_paths tests.test_curve_renderer.CurveRendererTests tests.test_backend_structure -v
```
