# Length / LUT quality experiment

仿照 `experiment.length_normdiff` 改变表面长度，但每个长度都重新扫描并标定 LUT，再复用 `experiment.lut_quality` 评估质量。不是重标注已有结果，也不是用 NormDiff 代替 LUT 质量。

## 运行与配置

在项目根目录运行：

```powershell
# 查看全部长度、扫描姿态数、材料网格和 LUT 大小，不渲染、不写输出。
.\.venv\Scripts\python.exe -m experiment.length_lutqual --dry-run

# 执行完整实验；相同命令可续跑已完成的阶段。
.\.venv\Scripts\python.exe -m experiment.length_lutqual

# 显式覆盖长度与评价设置，使用新输出目录。
.\.venv\Scripts\python.exe -m experiment.length_lutqual --lengths 21.8 35 55.9 --taus 0 0.05 0.1 0.2 --detail-tau 0.1 --weighting pose_equal --out experiment/length_lutqual/results_balanced
```

也支持 `python experiment/length_lutqual/__main__.py --help`。完整扫描需要与原长度实验相同的 CUDA 渲染环境，以及 calibration extra 中的 SciPy、Matplotlib、Numba 等依赖。干运行与纯 CPU 单元测试不启动渲染。

参数全部来自独立的 `configs/length_lutqual.json`：

- `simulation`：物理模型、相机、曝光、扫描、采样、LUT 拟合和标定验证的完整快照。不自动继承主配置的修改。
- `length_lutqual.lengths_mm`：本实验自己的长度列表，默认 `[21.8, 26.8, 31.8, 36.8, 41.8, 46.8, 51.8, 55.9]` mm。不跟随 `length_normdiff` 段后续修改。
- `length_lutqual.output_dir`：默认 `length_lutqual/results`。
- `lut_quality`：复用 `taus`、`detail_tau`、`weighting`、样例展示数量。已有样本/LUT 路径在本实验中不使用；本实验始终选择新拟合的训练 LUT 并强制 `split=auto`，保留完整姿态验证集。

CLI 显式参数覆盖实验配置。JSON 相对路径以配置文件所在目录为基准，CLI 相对路径以工作目录为基准。容差 tau 是 `(p,q)` 斜率欧氏距离，不是角度或 RGB 扰动。

## 每个长度具体做什么

1. 直接复用原实验的 `length_config`：只将 `surface.size_mm[0]` 改为 length，并对每个 top/bottom 灯带的 `length_mm` 加上相同增量。固定 Y 方向尺寸，保留原来的灯带超出量。灯带朝向、offset、Z、宽度、总功率、积分采样数、曝光和相机不变。
2. 重新生成该尺寸下的全部扫描姿态与材料网格；使用原来的扫描步长、深度范围与材料网格间距。渲染一张对应的零深度参考图和每个完整扫描姿态，保留所有有效接触样本。
3. 额外渲染 `(0,0)` 的固定中心压入，每个扫描深度一张，使用同一参考图。这些样本单独保存，不混入全扫描训练集。
4. 直接调用现有 `fit_and_validate`，沿用种子 42、完整姿态训练/验证隔离、LUT 构建、MAD 筛选、角度阈值、逐姿态验证及正式 LUT 发布规则。不会放宽阈值或用全局平均掩盖失败姿态。
5. 复用 `experiment.lut_quality` 的完整评估入口：用所有原始训练样本计算 `A(tau)`、实际节点错误率和三线性查表错误率；再用独立验证压痕评估实际查表及训练最优圆心的迁移表现。不会按拟合保留节点再次删除质量评估样本。
6. 用同一份全局训练 LUT 和训练最优圆心评估中心对照。**不拟合中心专用 LUT。**中心姿态可能与扫描姿态重合，因此中心结果不是独立验证；真正的独立结果是保存的 held-out 压痕。

某个扫描或中心姿态若达不到配置规定的最小有效样本数，实验报错，不悄悄丢弃它。不同面积自然生成不同姿态数；保持的是采样密度及完整扫描规则，而不是强行让每个长度样本数相同。

本实验不运行每帧验证预览、branch/ambiguity 可视化等与质量曲线无关的输出流程，**不会因此改变渲染、原始采样或拟合规则**。每个长度仍保存原始配对样本、验证报告及 LUT 质量图表。

标定验证失败不会导致该长度被跳过：仍评估并保存训练 LUT 的质量，在曲线上标记失败，不发布正式 `normal_lut.npz`。扫描无法完成、LUT 无法构建等执行错误则保留检查点并报错，不伪造分数。

## 图表和输出

根目录输出：

- `length_lutqual.png` / `.svg`：横轴为 length。在公共 `detail_tau` 下，左图比较训练、独立验证、固定中心的真实插值查表错误率；右图展示固定节点模型的不可兼容率、节点错误率和独立验证中的圆心迁移错误。失败标定用黑色叉号标记。
- `length_lutqual_all_tau.png`：对每个 tau 展示长度关系，避免只看单一阈值。
- `length_lutqual.csv`：每个 `(length,tau)` 一行，包含质量指标、样本/姿态数、单样本格比例、验证状态/RMS、法向覆盖范围、越界比例和该长度的 LUT RGB 边界。
- `summary.json`：实验条件和汇总数值，标明是否仍是部分长度结果。
- `manifest.json`：独立配置快照、实际指标设置、实现 SHA256、模型版本、完成长度及执行状态。

每个 `length_<L>/` 内保存：

```text
config.json                    该长度的完整配置快照
normal_samples.npz             全扫描原始配对样本，兼容现有质量评估入口
center_samples.npz             固定中心对照原始样本
scan.json                      每个姿态、计数、渲染/编译信息
normal_lut_train.npz           用于质量比较的训练 LUT（所有长度都使用这一类）
normal_lut.npz                 仅标定验证通过时发布的全量 LUT
verification/validation.json   标准标定验证报告
quality/                       复用 lut_quality 的所有图、原始格内指标和报告
center_quality.json            用全局 LUT 对固定中心的评价
point.json                     该长度的完整汇总
checkpoint.json                已完成阶段、输入配置签名及产物 SHA256
```

所有失败比例都是越低越好。建议优先比较独立验证集的 `holdout_runtime_error`，再用训练集 `A` 与 `node_error` 的差排查格内冲突和填值问题。`A` 只约束每格固定输出模型，不是连续插值模型的物理下界。

## 续跑与可比性

- 相同命令会校验配置、长度列表、指标参数、物理模型和实现哈希，再校验已完成阶段的产物哈希。条件变了应使用新的 `--out`，不能静默混合历史结果。
- 检查点分为原始采样、拟合、质量评价和中心评价；例如质量计算中断后可以复用已经保存的样本/LUT，不重新渲染。
- 采样中途尚未形成完整阶段时，下次会重跑该长度的整个采样阶段；当前不支持逐姿态渲染续跑。
- 未完成的 `quality/` 不删除或覆盖，重试写入 `quality_retry_001/` 等目录。根目录汇总只包含已完成长度，每完成一个点便更新图表和 CSV。
- 各长度使用相同 LUT 大小，但 RGB 范围沿用现有标定算法、由各自训练样本决定。因此 A 的变化同时受到光照、采样分布和实际 LUT 离散分辨率影响，不是固定 RGB 噪声下的纯光学不可辨识性比较。
- 灯带总功率不变，长度增加会降低每单位长度功率。本实验评价的是与 `length_normdiff` 一致的组合长度策略，不是仅改变灯距或保持 W/mm 不变的实验。
- tau 默认值用于展示曲线，不是经过验证的重建合格阈值；改变表面尺寸也可能改变法向覆盖和训练/验证分布，应结合保存的计数、覆盖、RGB 边界和标准标定报告解释趋势。

## 验证命令

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_length_lutqual tests.test_lut_quality tests.test_experiment_config tests.test_length_normdiff -v

# 独立的小尺寸 CUDA 冒烟配置；不是默认 21.8–55.9 mm 实验的科学结果。
.\.venv\Scripts\python.exe -m experiment.length_lutqual --exp-config tests/exp_config.length_lutqual.smoke.json
```

单元测试覆盖配置隔离、几何同步、无写入干运行、真实拟合/质量评估、完整姿态隔离、失败标定保留、分阶段与整点续跑、文件篡改拒绝及汇总输出。冒烟实验使用独立的低分辨率小尺寸测试配置，不改默认实验参数，也不将执行完成表述为标定验证通过。
