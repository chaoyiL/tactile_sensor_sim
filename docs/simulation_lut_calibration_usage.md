# 仿真 LUT 标定使用说明

MuJoCo + Xensim 的无 marker 第三后端安装、扫描、LUT 和评测命令见
[MuJoCo / Xense 使用说明](mujoco_xense_usage.md)。以下解析 CUDA 流程与结果说明仅适用于 `analytic` 后端。

灯带现使用 `vertical_edge_strips_v1`：发光面始终竖直且固定朝内，`offset_mm` 是接触面边缘到发光平面的水平距离，`width_mm` 沿 Z，`z_mm` 仍是中心线的有符号高度。详情见 [灯带参数](使用说明.md)。此变更改变光学结果；旧版自动倾斜灯带产生的样本/LUT 不能与新模型混用。请保留旧输出，重新渲染到新的空目录，例如 `--out results_lut_vertical`。

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

GPU 渲染默认使用 `pixel_chunk_size=16384`、`light_chunk_size=16` 和 `cuda_graphs=true`。扫描上下文第一帧使用普通 CUDA，从第二帧捕获并复用光照/阴影计算图；单次渲染跳过捕获。捕获失败会明确记录原因并使用普通 CUDA，OOM 时释放计算图、缩小分块后重试。具体实现、基准结果和复测命令见 [GPU 渲染优化](gpu_renderer_optimization.md)。

示例配置另启用 `compile_lighting="auto"`：多帧上下文从第二帧初始化 Inductor，融合光照因子计算；单帧不初始化编译器。关闭编译可设 `off`，要求编译成功可设 `required`。请在 GPU 标定命令中保留 `--extra gpu-compile`，否则 auto 在后端缺失时明确回退原 GPU 路径。当前 25 姿态对照约提升 27%，默认场景输出逐值一致；详情见 [编译加速验收](pytorch_compile_acceleration_validation.md)。
