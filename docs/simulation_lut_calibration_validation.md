# 仿真 LUT 标定验收记录

验收日期：2026-10-21。

环境：NVIDIA GeForce RTX 5050 Laptop GPU（8151 MiB）、PyTorch 2.12.1+cu130、CUDA runtime 13.0。依赖来自提交内的 `uv.lock`。

## 数值与功能检查

- `python -m unittest discover -s tests -v`：38 项通过，覆盖 stride、离线歧义分析、分支支持/分类、精确编译计数、CUDA Graph、Inductor 编译和完整 GPU 小扫描。
- stride=5 的回归用例从 300,000 个样本保留 60,000 个，确认所选样本不会再被总数量截断。
- GPU 渲染与 NumPy CPU 基线在 float64 下对照通过，包含球冠求交、面积灯积分和阴影射线；阈值为 `rtol=2e-11, atol=1e-13`。
- 移植的 LUT 构建、方差、有效节点、样本计数和 raw 三线性查询，与 `recon_stable` 固定提交 `e02f95f119aa42d5aaf1ebae1f2304d60f045e8c` 逐值一致。
- 默认配置 dry-run 得到 1316 个姿态；各深度数量为 `308, 270, 270, 234, 234`。
- 8 姿态小规模完整扫描通过：8 个姿态全部接受，共 1212 个样本；正式 LUT、训练 LUT、整图留出报告、8 张 verification 单图和全部 visualization 图表均成功生成。
- 8 姿态完整扫描的实际渲染调用次数为 9（reference 1 次、每姿态 1 次）；全局及分支图只用第一遍配对样本生成，额外渲染为 0。不创建 diagnostics、color_difference 或 pq_mapping 目录。
- 分支支持按逐对、逐半径对照上游 compare_branch_context，支持点数、来源图数和其他图支持数全部一致；80% 占优边界、零支持、多分支优先也与上游一致。
- Numba 编译树计数与 NumPy 上游对照实现一致，测试包含跨 64 位字边界的 73 个来源图，确保唯一图数不会重复计数。
- 采样后的歧义计数、端点 mask 与同一子集上全量 `query_pairs()` 一致；stride=1 时恢复全量精确模式。保存输入可在原始 normal_samples 删除且 PyTorch 不可导入时独立重算。
- 此前复用 `results_lut/normal_samples.npz` 中 2,662,622 个样本，stride=100 得到 27,421 点（原样本的约 1.03%），覆盖全部 1316 姿态。重新分析耗时 2.75 秒，额外渲染为 0，结果为 20,779 对歧义、9,423 个歧义端点；其中跨姿态 20,769 对，同姿态 10 对。此为当时采样集合的精确计数，不代表全量统计，也不是当前示例 stride=5 的分析结果。

小规模完整扫描使用 `tests/config.calibration.small.json` 测试夹具，其验证阈值特意放宽，只用于覆盖发布路径；它不代表 `configs/calibration.json` 的正式标定精度结论。

## 分支分类完整数据验收

复现命令（当前统一输出到 `results_lut/`）：

```powershell
uv run --cache-dir .uv-cache --extra calibration python calibrate_lut.py --ambiguity-only --samples results_lut/normal_samples.npz --out results_lut --ambiguity-stride 5
```

复用已有完整样本，不渲染、不重新拟合 LUT。原始 2,662,622 点按每姿态 stride=5 保留 533,176 点（约 20.02%），覆盖全部 1316 个姿态，没有总样本上限。

完整处理 7,497,301 对冲突及 `0.02/0.03/0.04` 三个外围半径。主结果为 `0.03`，占优阈值为 0.8：

| 类别 | 唯一点数 | 冲突对数 |
| --- | ---: | ---: |
| no_outer_support | 0 | 0 |
| main_support | 116,480 | 4,101,730 |
| multiple_branches | 247,221 | 3,395,571 |

363,701 个点参与冲突，其余 169,475 点没有冲突。点级采用多分支证据优先，所以不能用 pair 数比例推导 point 比例。统计只对 stride 选定集合成立，不估算完整 266 万点的结果。

独立分析总耗时 987.59 秒（约 16.46 分钟），分支阶段 944.43 秒，包含三个半径的精确支持计数、分类敏感性、图表和文件写入。这是在运行其他任务的本机上测得的整体耗时，不作为独占机器基准。

本次历史验收产物现存于 `results_lut/ambiguity_branches/visualization/ambiguity_patterns/`；按上述统一路径命令重新运行时，写入 `results_lut/visualization/ambiguity_patterns/`。产物包括两张 PNG、点级 NPZ 及 analysis.json。已检查图像布局、q 轴方向和实例支持标记，并确认源样本索引、点级 mask、分类计数和报告一致；所有临时冲突对/支持数组均已清理。

## 渲染性能

以下保留早期 CPU/GPU 对照记录。后续新增了 CUDA Graph、分块和光照常量优化，本轮原始循环对照的 GPU 单帧耗时从 281.94 ms 降到 93.77 ms，约 3.01 倍；所有对照模式数组逐值一致。当前实现、完整测量条件及 `results_lut/performance/` 报告位置见 [GPU 渲染优化](gpu_renderer_optimization.md)。

命令：

```powershell
uv run --cache-dir .uv-cache --locked --extra gpu python tests/benchmark_renderer.py --config configs/analytic.json --scale 1 --frames 3
```

条件：`640 × 480`、192 个面积灯采样点、开启阴影、float64 几何和辐射计算。CPU 与 GPU 使用相同物理配置，每侧先完成 GPU 预热；三帧结果为：

| 项目 | 结果 |
| --- | ---: |
| CPU 帧耗时 | 3.152 s, 3.115 s, 3.310 s |
| GPU 帧耗时 | 0.203 s, 0.191 s, 0.189 s |
| CPU 中位耗时 | 3.152 s |
| GPU 中位耗时 | 0.191 s |
| 中位加速比 | 16.50× |
| CUDA 峰值分配显存 | 145,393,152 bytes（约 138.7 MiB） |

基准脚本会在计时后再次比较 CPU/GPU 传感器数组；数值不一致时直接失败。当前峰值远低于 8 GB 设备的 70% 预算，且未触发自动 OOM 分块重试。

小扫描在锁定环境中的采样姿态 GPU 渲染中位耗时为 0.0167 s，材料采样中位耗时为 0.00454 s，峰值显存为 37,673,984 bytes；包含绘图的总流程为 9.00 s。
