# LUT 标定质量：单值不可兼容率

读取已有的原始标定样本和 LUT，评估全局的有符号线性 RGB 残差 `dI → (p,q)` 映射。**不重新渲染，不重新拟合 LUT，不丢弃原始样本。**这不是 NormDiff，也不是某篇论文的同名指标。

## 运行

从项目根目录运行：

```powershell
# 检查输入、训练/验证划分、样本数与计算规模；不计算指标、不创建输出。
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/xense --dry-run

# 默认评估所选结果目录中的 normal_lut_train.npz，保留其独立验证压痕。
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/xense --out results_lut/xense/evaluation/lut_quality_run_02

# 失败标定也可评估其训练 LUT，并保留独立姿态留出集。
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/analytic/previous_failed --out results_lut/analytic/previous_failed/evaluation/lut_quality_train

# 全量样本拟合的诊断 LUT 不保留独立留出集
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/analytic/previous_failed --lut results_lut/analytic/previous_failed/normal_lut_diagnostic.npz --out results_lut/analytic/previous_failed/evaluation/lut_quality_diagnostic

# 指定容差曲线与用于详细展示的容差；这里的数值不是角度。
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/xense --taus 0 0.02 0.05 0.1 --detail-tau 0.05 --weighting pose_equal --out results_lut/xense/evaluation/lut_quality_balanced
```

也支持 `python experiment/lut_quality/__main__.py --help`。依赖 NumPy 和 Matplotlib（可通过 calibration extra 安装）；覆盖搜索会在可用时自动用 Numba 加速，没有 Numba 时使用相同算法的 Python 路径。无需 CUDA 或 Zemax。

检验参数位于 `configs/lut_quality.json` 的 `lut_quality` 段；输入样本、LUT 和默认输出目录由 `--lut-results` 指定。已通过、验证失败或尚未发布的运行都可检验，只要有可用样本与训练 LUT；报告记录原标定状态，不改变正式发布结论。

参数说明：

| 参数 | 含义 |
| --- | --- |
| `samples` | 原始 `normal_samples.npz`，不能替换成平均后的 LUT |
| `lut` | 实际保存的 `normal_lut*.npz`，默认选择保留验证集的训练 LUT |
| `output_dir` | 新的或空的输出目录；非空目录拒绝覆盖，不提供隐式续跑 |
| `taus` | 至少两个不同的非负 `(p,q)` 欧氏距离容差，计算时排序 |
| `detail_tau` | 必须属于 `taus`，用于高歧义格子详情 |
| `weighting` | `sample`：每条样本等权；`pose_equal`：每个完整压痕等权 |
| `split` | `auto`：遵守训练 LUT 保存的完整姿态划分；`all`：明确使用所有样本做经验评估 |
| `top_bins` / `examples_per_bin` | 输出多少个高歧义格子及每格的原始样例数，不影响指标计算量 |

CLI 显式参数覆盖 JSON。JSON 相对路径相对于所选实验配置文件，CLI 相对路径相对于工作目录。本实验不读取主配置、不使用 `simulation` 快照重建图像。加载公共实验配置时仍会执行其现有结构校验。

## 指标与算法

按 LUT 构建时的节点归属规则分箱：

```text
node = rint(clip((dI - color_min) / (color_max - color_min), 0, 1) * (size - 1))
```

使用 NumPy 的半偶舍入，坐标按 RGB 排列；越界值像运行时一样夹紧，但会单独报告越界比例。不会只保留 `original_valid` 节点，也不会按 LUT 拟合时的 MAD 规则再次剔除样本。原始节点保留标志与原始配对样本数分开保存。

每格取全部原始 `(p,q)`，寻找半径为 `tau` 的圆能覆盖的最大权重：

```text
A_bin(tau) = 1 - max_center(covered_weight / bin_weight)
A(tau)     = sum_bin(bin_weight * A_bin) / total_weight
```

圆心允许在整个 `(p,q)` 平面上移动，而不是限制在样本位置。实现使用精确的加权角度扫描：固定一个边界点，扫描其他点允许的圆心角区间。相同坐标合并时累加全部样本权重，**不做下采样、聚类近似或候选截断**。最坏复杂度为每格 `O(U² log U)`，`U` 为不同斜率的数量；高占用格子会比较耗时。提供逐 500 格的进度输出，首次 Numba 编译也可能需要时间。

闭圆边界使用浮点容差 `1e-12 * max(1,tau)`；`tau=0` 按完全相同的斜率判断。算法为浮点几何实现，不是符号精确运算。

同时输出：

- `node_error`（文档的 `E_LUT`）：用归属节点实际存储的 `(p,q)` 替代最优圆心，计算超出同一容差的比例。
- `node_excess_error = node_error - A`：同样分箱、样本、权重下，节点填值相比最有利单值输出多出的失败比例。
- `runtime_error`：直接调用现有 `lookup_slopes_raw` 的三线性插值输出，再计算斜率失败比例。

**三线性插值并不是每格固定输出。**因此 `A` 只约束固定节点模型，不是实际连续插值模型的不可突破下界；`runtime_error` 可以小于 `A`，二者之差不能叫做标定算法的额外错误。要研究传感器噪声下的不可辨识性，还需按实际 dI 精度/局部邻域定义观测等价类，本实验不以节点分箱代替该问题。

默认 tau 序列仅用于展示容差曲线，**不是物理上已验证的合格阈值**；必须按重建精度要求选择。斜率欧氏距离不是法向角度，同一 tau 在不同倾角处不对应固定角度。本实验不会据此宣布标定通过，也不会发布正式 LUT。

## 权重、覆盖和独立验证

`sample` 使用 `w_b=N_b/N`。`pose_equal` 在每个评估分组内令每条样本的权重为该姿态样本数的倒数，再按格内总权重聚合；既改变格间权重，也改变格内最优圆。它平衡压痕大小，但不自动平衡法向覆盖，也不能自动识别重复采集的等价姿态。

`split=auto` 的规则：

- `validation_status=training_only`：只用保存的 `training_images` 拟合最优圆；对应的 `validation_images` 是独立压痕。验证时转移已选圆心，**不使用验证标签重新挑选圆心**。
- 正式全量 LUT、失败诊断 LUT 或无训练划分的 LUT：使用全部样本，只报告经验结果；明确提示没有独立验证，不能把旧的验证名单误当作全量拟合后的留出集。
- `split=all`：用户主动选择全部样本，取消内部留出验证，报告中会标明。

验证集报告实际节点/插值错误率，以及最优圆转移后的失败比例。训练未见过的节点单独计入 `unseen_bin_fraction`；`transferred_disc_failure_or_unseen` 将其视为缺少可用预测，不把缺失当成功；`transferred_disc_known_bin_error` 则只统计已见格子。转移错误率不是验证集重新优化得到的 `A`，也不保证是 LUT 的下界。

每格保存原始样本数、姿态数、不同斜率数、p/q 范围、theta 范围、相对加权平均法向的角度 RMS。另附全局原始样本的 theta/phi 覆盖直方图。单样本格子的 `A=0` 不代表已证明唯一性。比较不同光照前应同时核对覆盖和采样权重。

## 输出

- `quality.json`：指标曲线、独立验证结果、覆盖摘要、定义及局限。
- `curves.csv` / `curves.png`：随 tau 变化的 `A`、节点错误率和实际插值错误率。
- `per_bin.npz`：按 `bin_ids` 排列的稀疏原始计数、覆盖、各 tau 的最优圆心、格内指标；数组维度为 `[tau, occupied_bin, ...]`，空格不伪装成零歧义。
- `bin_heatmaps.png`：三个实际 B 节点切片上、各 tau 的格内不可兼容率；不是投影平均。未采样位置为浅米色，完整 RGB 格数据在 NPZ 中。
- `sample_counts.png` / `normal_coverage.png`：同一组切片的原始计数和全局法向覆盖。
- `top_bins.json` / `top_bins.png`：`detail_tau` 下的高歧义格子、圆心、实际节点输出、包含兼容/不兼容情况的原始样例；散点图保留所展示格子的全部样本。
- `manifest.json`：输入与实现 SHA256、样本/LUT 元数据、实际设置、运行状态及环境；不覆盖源标定目录，不改写历史结果。

运行中断会留下 `failed`（可捕获异常）或 `running` 状态；此版本不支持续跑，请选择新的输出目录。输入签名或声明的 RGB 域不兼容时拒绝评估；签名缺失时在报告中说明无法验证来源。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_lut_quality tests.test_experiment_config -v
```

固定样例覆盖：非样本圆心、50/50 多解、少数分支、零容差、相切/跨角度零点、加权、与穷举边界圆对照、插值优于固定节点下界、姿态隔离、未见格子、完整输出及覆盖保护。
