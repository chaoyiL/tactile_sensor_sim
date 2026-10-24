# PyTorch 编译加速验收与使用

验收日期：2026-10-21。实现使用 PyTorch Inductor 编译固定像素/灯块的几何、阴影和光照因子计算，继续复用手动 CUDA Graph。RGB 乘法、灯块内归约与跨块累加保留原 ATen 顺序。

## 环境和安装

本机已验证：Windows 11、Python 3.13.11、PyTorch 2.12.1+cu130、CUDA runtime 13.0、NVIDIA 驱动 610.62、RTX 5050 Laptop（sm120，约 8 GB）。可选编译后端锁定为 `triton-windows==3.7.1.post27`，随包 PTX assembler 实测为 12.8.93，不改变原有 torch 版本。

```powershell
uv sync --cache-dir .uv-cache --locked --extra gpu --extra calibration --extra gpu-compile
```

`gpu-compile` 在 Windows 安装已验证的 Triton 分支；Linux 遵循对应 torch wheel 的依赖。当前实测仅覆盖本机 Windows 环境。没有此可选后端仍可使用原 GPU 路径。

## 运行行为

`configs/analytic.json` 已配置：

```json
"compile_lighting": "auto"
```

该字段位于原有 `render`，和 `cuda_graphs` 独立：

- `off`：关闭编译；省略字段也按 off 执行。
- `auto`：复用同一个 CUDA 上下文时，从第二帧初始化编译器。后端缺失、编译失败或初始块验证失败会记录原因，继续执行原 GPU 路径。
- `required`：从第一帧强制编译；失败即报错，用于验收。

单次 `simulate.py` 渲染通常只使用上下文一帧，因此 auto 不承担编译初始化成本。LUT 扫描复用上下文，reference 首帧 eager，后续姿态使用编译。离线歧义分析不引入 torch/Triton 依赖。

完整标定运行时需保留新 extra：

```powershell
uv run --cache-dir .uv-cache --extra gpu --extra calibration --extra gpu-compile python calibrate_lut.py --backend analytic --config configs/analytic.json --out results_lut/analytic/run_new
```

原有“完整标定输出目录为空”规则继续生效。已有标定结果或性能报告时，使用新的 `results_lut/` 子目录作为 `--out`，或事先归档目录；不覆盖原始结果。

## 数值和回归

38 项回归全部通过，覆盖 CPU/GPU 物理对照、float64/float32 辐射、float32 几何的编译或明确回退、非整数 LED 指数、两个压入方向、阴影开关、零压入、非整块尾部、切线交点、历史帧不被覆盖、首次编译错误、后端缺失、OOM 后缩块重建，以及离线歧义/分支分析。

新增的八姿态完整标定对照中，保存的颜色残差、坡度、图像分组与材料网格索引逐值一致，完整渲染次数仍为 9（reference 加八姿态）。初始块验证仅标记 `initial_blocks_passed`，不冒充整个实验集的验收状态。

首次尝试将 RGB 归约也融合时，在 25 姿态集上出现一个像素通道的 RGB8 量化变化，三轮合计报告 3 次。最终版本将此归约留在原 ATen 路径后，传感器数组、RGB8 与命中 mask 的差异全部为零。失败实验保存在 `results_lut/performance/compile_lighting/acceptance/benchmark_full_reduction_failed.json`，不能作为当前版本结果。

## 实际性能

同一配置、640×480、192 个灯采样点、开启阴影、几何 float64/辐射 float32。25 个扫描姿态加 reference，三轮交替对照，共每种模式 78 帧：

| 路径 | 稳态中位数 | p95 |
| --- | ---: | ---: |
| 冻结的上一阶段 CUDA Graph 基线 | 96.87 ms | 124.24 ms |
| 函数提取后的编译关闭路径 | 100.61 ms | 146.24 ms |
| 仅编译，不捕获图 | 79.76 ms | 100.97 ms |
| 编译 + CUDA Graph | 76.54 ms | 91.24 ms |

相对冻结基线，中位数约加速 **1.27 倍**；三个独立轮次的中位数速度比分别为 1.30、1.25、1.31。完整传感器数组最大绝对差为 0，RGB8 和命中 mask 不同元素数也为 0。各模式同进程的峰值分配显存为约 207.63 MiB，这是进程总峰值，不是每种模式分别测得的峰值。

首次加载编译后端有成本。该轮 compile-only 路径启动合计 10.22 秒，其中首次内核调用 6.13 秒；随后同进程编译+图复用了编译缓存，不能把其 0.28 秒启动当作全新进程冷启动。用前者与冻结基线的启动差作为保守估计，约 458 帧回本。该数字取决于缓存和机器负载，不代表所有机器。

短 profiler 对照的 CUDA 设备事件数量从 6613 降到 622。带 profiler 的时间不能替代上表，图内范围事件可能重叠，不能直接把各范围时间相加当作总耗时。

这些数据只说明渲染阶段提速。本次未重新运行 1316 姿态的正式标定，也未重新进行全量分支分析，不能宣称整个任务也加速 1.27 倍。

## 复测和报告

```powershell
.\.venv\Scripts\python.exe tests/probe_torch_compile.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests/benchmark_renderer_compile.py --poses 25 --rounds 3 --out results_lut/performance/compile_lighting/acceptance_factor
.\.venv\Scripts\python.exe tests/benchmark_renderer_compile.py --profile --profile-mode compiled --out results_lut/performance/compile_lighting/profile_after
```

本轮报告：

- `results_lut/performance/compile_lighting/environment.json`：隔离环境探针通过结果。
- `results_lut/performance/compile_lighting/acceptance_factor/benchmark.json`：最终四模式数值与性能数据。
- `results_lut/performance/compile_lighting/profile_before/`、`profile_after/`：优化前后的 trace、汇总表和事件数。

基线快照存放于 `tests/compile_reference/`，记录 SHA256、Git HEAD 和当时工作区差异。正式代码仍在 `utils/`，测试与基准脚本继续放在被忽略的 `tests/`。

manifest 的 `performance` 记录 `compile_lighting_requested/effective`、后端、编译区间、版本、选项、变体数、首次调用/准备/预热/图捕获耗时、验证状态及失败阶段和原因。`effective` 为 false 时，不能把该次运行称为编译加速。
