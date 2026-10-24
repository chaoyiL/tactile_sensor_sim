# GPU 渲染优化

PyTorch 编译加速已经落地：多帧上下文默认使用 Inductor 光照因子融合，并保留原 RGB 归约及 CUDA Graph。新增阶段的验收、安装和模式设置见 [编译加速验收](pytorch_compile_acceleration_validation.md)；设计依据见 [施工计划](plan_pytorch_render_acceleration.md)。下文保留前一阶段 CUDA Graph 优化记录。

实现使用现有 PyTorch CUDA 环境，无需新增依赖。默认设置为：

```json
"render": {
  "pixel_chunk_size": 16384,
  "light_chunk_size": 16,
  "cuda_graphs": true
}
```

以上字段合并到原有 `render` 配置；曝光、阴影、设备及精度字段继续保留。`cuda_graphs` 省略时为 `true`，必须是布尔值。`pixel_chunk_size` 省略时为 `16384`。

## 实现

`utils/renderer_gpu.py` 将所有光源积分提取为 `_shade_points`，由普通 CUDA 和执行图共用。默认像素块由 8192 增至 16384，光源块维持 16。光源归一化系数在上下文初始化时缓存；同一像素块的阴影射线原点只计算一次。所有 LED 的 `cosine_exponent` 都为 1 时，直接使用出射余弦，省去通用浮点幂运算；包含其他指数时继续使用原有幂运算公式。

`utils/renderer_cuda_graph.py` 捕获固定大小的光照与阴影块。首帧用普通 CUDA 执行，第二帧在独立流上预热并捕获，以后复制新位置、法线和球参数并重放同一执行图。相机求交、可见像素选择仍在原 GPU 路径执行。计算图不捕获 PNG 写入、CPU 分析或 LUT 拟合。

捕获输入包括球心坐标、球心高度、接触边界和有无球冠标记。标量计算在写入设备张量前使用原公式完成，避免 float32 下改变舍入顺序。零压入 reference 也能通过同一个图重放。半径、表面尺寸、压入方向、阴影开关或分块大小改变时，会重新捕获。

动态有效像素数通过尾块补零处理，补零行法线为零，不参与输出；只复制有效行。执行图内部输出由上下文复用，每块结果先复制到独立辐照度数组，因此后续重放不会覆盖已经返回的帧。

只有复用 `AnalyticGpuContext` 的多帧渲染会捕获执行图，LUT 扫描已经满足这一条件。单次 `render_host` 不承担捕获开销，也能受益于分块、常量缓存及幂运算优化。设置 `render.cuda_graphs=false` 可使用优化后的普通 CUDA 路径。

OOM 时会释放缓存执行图、将像素块减半并重试，必要时再缩小光源块。不支持捕获时输出警告，将原因保存在上下文和 manifest 中，并继续使用普通 CUDA。几何精度、光源采样点数和图像分辨率保持原配置。

`manifest.json` 的 `performance` 新增：

- `cuda_graphs_requested`：是否启用。
- `cuda_graph_capture_count`：成功捕获次数。
- `cuda_graph_replay_count`：块级重放次数，不是姿态数。
- `cuda_graph_fallback_reason`：未能捕获的原因，正常为 null。

## 复测

2026-10-21 本机 RTX 5050 Laptop、PyTorch 2.12.1+cu130 实测：640×480、192 个面积灯采样点、开启阴影，几何 float64、辐射 float32；每种模式预热三帧后测九帧。

| 路径 | 像素块 | 单帧中位数 | PyTorch 峰值分配显存 |
| --- | ---: | ---: | ---: |
| 原始光照循环、普通 CUDA | 8192 | 281.94 ms | 131.78 MiB |
| 优化光照循环、普通 CUDA | 16384 | 174.47 ms | 172.64 MiB |
| 优化光照循环、普通 CUDA | 32768 | 113.59 ms | 257.89 MiB |
| 优化光照循环、CUDA Graph（默认） | 16384 | 93.77 ms | 170.59 MiB |

默认路径在此轮对照中约为原始循环的 3.01 倍，所有模式传感器数组均逐值相同。图模式的最优分块与普通 CUDA 不同，因此默认采用 16384；关闭执行图、只做单帧渲染时可另测 32768。峰值统计包含预热与捕获；默认模式三帧预热和捕获合计约 0.49 秒，不是额外独立的捕获耗时。完整数据在 `results_lut/performance/gpu_optimizations.json`。

另外将几何和辐射都设为 float64，CPU/GPU 完整分辨率对照通过 `rtol=3e-10, atol=1e-13` 检查。CPU 三帧中位数 3.032 秒，优化 GPU 为 0.09564 秒。该数据精度设置与上表不同，不能直接混算速度比；完整数据在 `results_lut/performance/cpu_gpu_float64.json`。

这些是单帧渲染测量，时钟与后台负载会影响结果；LUT 拟合、分支分析、预览写盘仍各自计时，整个标定任务的提速需另外测量。29 项回归检查通过，包含八姿态完整扫描及执行图路径；未重新运行 1316 姿态的正式标定。

对比保留的原始光照循环，以及不同分块和执行图组合：

```powershell
.\.venv\Scripts\python.exe tests/benchmark_gpu_optimizations.py --frames 9 --out results_lut/performance/gpu_optimizations.json
```

脚本每种配置先执行三帧，包含预热和捕获，再对稳态帧计时；每帧完成后同步 GPU。单独记录预热与捕获耗时，不将其混入稳态帧中位数。每种配置都与原始循环的传感器数组逐值比较，差异会直接使脚本失败。

CPU/GPU 双精度数值与速度对照：

```powershell
.\.venv\Scripts\python.exe tests/benchmark_renderer.py --config configs/analytic.json --scale 1 --frames 3 --out results_lut/performance/cpu_gpu_float64.json
```

回归检查：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖不同位置和深度、零压入、两个压入方向、阴影开关、非整数 LED 指数、float32/float64 几何、float32/float64 辐射计算、非整块像素与光源、历史帧不被覆盖、捕获失败的 GPU 路径以及 OOM 后重建执行图。另有小规模完整 LUT 扫描验证 manifest、保存数据、图表及渲染次数。
