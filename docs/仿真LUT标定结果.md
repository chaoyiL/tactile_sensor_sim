# 仿真 LUT 标定验证结果

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

以下保留早期 CPU/GPU 对照记录。后续新增了 CUDA Graph、分块和光照常量优化，本轮原始循环对照的 GPU 单帧耗时从 281.94 ms 降到 93.77 ms，约 3.01 倍；所有对照模式数组逐值一致。当前实现、完整测量条件及 `results_lut/performance/` 报告位置见 [GPU 渲染优化](GPU渲染优化验证结果.md)。

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


## 附录：标定条件与复现说明

MuJoCo + Xensim 的无 marker 第三后端安装、扫描、LUT 和评测命令见
[Xense 后端扫描验证结果](Xense后端扫描验证结果.md)。以下解析 CUDA 流程与结果说明仅适用于 `analytic` 后端。

灯带现使用 `vertical_edge_strips_v1`：发光面始终竖直且固定朝内，`offset_mm` 是接触面边缘到发光平面的水平距离，`width_mm` 沿 Z，`z_mm` 仍是中心线的有符号高度。详情见 [已验证灯带模型](逐点RGB灯光优化实验结果.md)。此变更改变光学结果；旧版自动倾斜灯带产生的样本/LUT 不能与新模型混用。请保留旧输出，重新渲染到新的空目录，例如 `--out results_lut_vertical`。

LUT 标定需要 CUDA 版 PyTorch、SciPy、Matplotlib 和用于大规模分支计数的 Numba。使用锁文件同步完整环境：

```powershell
uv sync --cache-dir .uv-cache --locked --extra gpu --extra calibration --extra gpu-compile
```

先检查扫描范围和资源规模；dry-run 不渲染、不拟合，也不会创建输出目录：

```powershell
uv run --cache-dir .uv-cache --extra gpu --extra calibration --extra gpu-compile python calibrate_lut.py --backend analytic --config configs/analytic.json --calibration-config configs/calibration.json --dry-run
```

执行完整标定：

```powershell
uv run --cache-dir .uv-cache --extra gpu --extra calibration --extra gpu-compile python calibrate_lut.py --backend analytic --config configs/analytic.json --out results_lut/analytic/run_new
```

程序只支持 analytic CUDA 标定。它先渲染一次零压入 reference，再按深度、行、列扫描标定球。每个深度的第一个接触圆与左边和上边相切。颜色样本是在固定材料网格上计算的 signed linear RGB 差值；坡度来自已知球冠几何，不需要人工圈选或 marker mask。

标定和独立歧义分析统一默认写入 `results_lut/`；省略 `--out` 即使用此路径。完整标定要求输出目录为空；再次完整扫描前需先备份并移走已有目录，独立歧义分析可直接更新现有目录。主要输出包括：

- `normal_samples.npz`：紧凑的颜色、坡度、材料坐标及图像分组样本。
- `ambiguity_inputs.npz`：第一遍扫描结束、拟合之前写入的完整歧义分析输入，包含未经过 MAD 或 stride 筛选的 signed RGB 色差、几何坡度、姿态分组、图像 key、运行签名和元数据。后续分析可只依赖此文件。
- `normal_lut_train.npz`：整图留出验证使用的训练 LUT。
- `normal_lut.npz`：验证通过后才发布的正式 LUT；失败时不会生成。
- `normal_lut_diagnostic.npz`：验证失败且启用可视化时生成，元数据明确标为失败诊断。
- `verification/<pose>.png`：每个姿态唯一的一张色差预览，圈出几何标定范围。
- `verification/report.csv` 与 `validation.json`：逐姿态采样和整图留出指标。
- `visualization/`：全局拟合、held-out、LUT 切片、颜色歧义和分支分类图表。
- `visualization/ambiguity_summary.json`：独立歧义报告，包含采样方式、样本数、姿态覆盖、阈值、同图/跨图歧义计数和耗时。
- `visualization/ambiguity_analysis.npz`：被分析的样本、在完整输入中的索引、姿态分组及歧义 bool mask。
- `visualization/ambiguity_patterns/branch_classification.png`：与上游相同的三个面板：RGB 投影、p/q 空间分类、不同外围半径下的唯一歧义点数。
- `visualization/ambiguity_patterns/branch_continuation.png`：与上游相同的 RGB 三维与 p/q 面板，展示两支都延续或仅一支有外围支持的实例。
- `visualization/ambiguity_patterns/analysis.json` 与 `branch_classification.npz`：分支支持、半径与占优比例敏感性、点级分类、角色及原始样本索引。
- `manifest.json`：输入签名、所有姿态、产物、验证状态、耗时和 CUDA 峰值显存。

不再生成 `visualization/color_difference/`、`visualization/pq_mapping/` 及其逐帧诊断缓存。verification 中每姿态一张带圈线的色差图继续生成。输出中不会创建 `frames/`、整体曲面重建 NPZ、逐帧 GT、reference PNG 或 irradiance NPY。若验证失败，命令以非零状态结束，但保留训练 LUT、诊断、预览和报告供排查。

仅重新分析已有数据，不需要 CUDA，不渲染、不重新拟合 LUT；会更新歧义图、branch classification 和 branch continuation。此模式允许在已有输出目录中更新歧义报告：

```powershell
# 已有旧版 normal_samples.npz 时，直接复用并将完整歧义输入保存到 results_lut/。
uv run --cache-dir .uv-cache --extra calibration python calibrate_lut.py --ambiguity-only --samples results_lut/normal_samples.npz --out results_lut

# 以后可直接读取保存的输入，修改 stride 后重算。
uv run --cache-dir .uv-cache --extra calibration python calibrate_lut.py --ambiguity-only --out results_lut --ambiguity-stride 5
```

`calibration.visualization.ambiguity_stride` 当前示例值为 `5`：每个姿态内按原始有效样本顺序取第 `0,5,10,...` 个样本，避免直接跳过整个姿态。程序保留所有 stride 选中的点，不设置总样本数上限，也不进行二次抽样。省略该配置字段时回退值为 `100`。CLI 覆盖只对 `--ambiguity-only` 生效；需要全量精确分析时，设 `--ambiguity-stride 1`。

stride 仅影响歧义诊断，不影响 LUT 训练或 held-out 验证。完整原始输入始终保留；分析仍使用相同的颜色距离和法线角阈值。报告中的 pair 数和歧义比例只对实际分析的集合成立，不估算全量结果。每个有有效样本的姿态都至少保留一个点，报告记录 `original_image_count/analyzed_image_count`。KD-tree 查询分小批执行并每约 5 秒打印进度。

此前只生成基础歧义图时，已有的 2,662,622 个样本在 stride=100 下分析 27,421 点，覆盖全部 1316 个姿态：20,779 对歧义（同姿态 10、跨姿态 20,769），9,423 个歧义端点，耗时约 2.75 秒。这是历史结果，不包含新加入的分支分析耗时，也不是当前 stride=5 的结果。新流程的全部图表只依赖原始配对样本，不需要逐帧显示缓存。

分支诊断复现 [recon_stable 的 analyze_ambiguities.py](https://github.com/chaoyiL/recon_stable/blob/e02f95f119aa42d5aaf1ebae1f2304d60f045e8c/calibrate/analyze_ambiguities.py) 中的外围支持规则。对 stride 选中集合中的每一个冲突对，在两点 RGB 中点周围查询 `2r/3r/4r` 的颜色球，排除两个原始 `r` 球中的全部点；仅统计落入各锚点法线 `angle_threshold/2` 锥内的外围支持。没有最小内圈或外围样本数门槛。

主结果使用 `3r`，`calibration.visualization.branch_dominance_fraction` 默认为 `0.8`。两支均无支持标为 `no_outer_support`；某一支占两支总支持至少 80% 标为 `main_support`；否则标为 `multiple_branches`。一个点参与多个冲突对时，只要其中任一对显示多分支重叠，就优先标为多分支。报告同时记录 `2r/3r/4r` 稳定性、0.7/0.8/0.9 占优阈值敏感性及按不同支持图像数计数的差异。

continuation 实例优先使用上游的“两个内圈各只有 2 点”条件；若不存在，则选择有外围支持的稠密实例，图上显示真实内圈数量。无支持实例时也输出说明图。分类描述观测支持密度，不自动删除样本或认定哪一支标签错误。所有近邻、支持和分类均在明确的 stride 集合内计算，不再二次采样。冲突对和支持数组采用磁盘临时文件及分块处理，完成后自动清除；结果 NPZ 保留点级分类和原始索引，不强制导出全部 pair 明细。

较大的冲突集合使用 Numba 编译的颜色/法线树遍历：完整满足条件的节点直接累计支持数，并按图像位集合做唯一图数统计；边界节点继续遍历，叶节点严格复用颜色距离和法线角判断，不使用近似查询或额外采样。首次调用可能包含一次编译开销，之后复用编译缓存。小集合保留 NumPy 逐锚点版本作为对照，两种路径均有上游数值测试。

普通 analytic 渲染默认使用配置中的 `render.device`。显式 CPU 对照可使用：

```powershell
uv run python simulate.py --config configs/analytic.json --backend analytic --device cpu --out results_cpu
```

运行全部回归检查：

```powershell
uv run --cache-dir .uv-cache --extra gpu --extra calibration --extra gpu-compile python -m unittest discover -s tests -v
```

GPU 渲染默认使用 `pixel_chunk_size=16384`、`light_chunk_size=16` 和 `cuda_graphs=true`。扫描上下文第一帧使用普通 CUDA，从第二帧捕获并复用光照/阴影计算图；单次渲染跳过捕获。捕获失败会明确记录原因并使用普通 CUDA，OOM 时释放计算图、缩小分块后重试。具体实现、基准结果和复测命令见 [GPU 渲染优化](GPU渲染优化验证结果.md)。

示例配置另启用 `compile_lighting="auto"`：多帧上下文从第二帧初始化 Inductor，融合光照因子计算；单帧不初始化编译器。关闭编译可设 `off`，要求编译成功可设 `required`。请在 GPU 标定命令中保留 `--extra gpu-compile`，否则 auto 在后端缺失时明确回退原 GPU 路径。当前 25 姿态对照约提升 27%，默认场景输出逐值一致；详情见 [编译加速验收](PyTorch编译加速验证结果.md)。
