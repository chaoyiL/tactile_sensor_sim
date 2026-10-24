# 项目协作约定

本文件适用于整个仓库；子目录若有 `AGENTS.md`，其更具体的约定在对应目录内优先。用户当前明确要求优先于本文件中的默认约定。

## 项目定位与目录

本项目用于光学触觉传感器仿真、球压入扫描、RGB 到表面法向的 LUT 标定，以及光照与映射歧义实验。

- `simulate.py`：触觉仿真入口，选择 analytic、zemax 或 xense 后端。
- `calibrate_lut.py`：扫描、LUT 拟合、验证和歧义分析入口。
- `configs/`：所有可编辑配置的唯一位置，包括三个仿真器各自的配置、统一 LUT 标定配置，以及每项 LUT 检验自己的实验配置。
- `utils/`：配置、几何、渲染、标定和可复用的基础能力。
- `experiment/`：独立实验包、实验配置、指标实现及实验结果。
- `tests/`：自动化测试与数值对照；不要在实验目录内散放测试脚本。
- `docs/`：方法说明、设计交接、使用文档和验证记录。
- `results*`、`render_results/`：已有输出；不是可随意清理的临时目录。

## 仓库三部分与目录结构

仓库按触觉仿真器、LUT 标定、LUT 检验三部分组织。

```text
configs/
  analytic.json                 # 解析仿真器的物理与渲染参数
  zemax.json                   # Zemax 仿真器的物理与光学参数
  xense.json                   # Xense/MuJoCo 仿真器的物理与传感器参数
  calibration.json             # 三种后端共用的扫描、拟合及验证参数
  <experiment_name>.json       # 每项 LUT 检验独立的实验参数
simulate.py                    # 统一仿真入口，显式选择后端
calibrate_lut.py               # 统一标定入口，显式选择后端
experiment/<experiment_name>/  # 每项 LUT 检验各有独立入口及实现
results_lut/
  analytic/
    evaluation/<experiment_name>/
  zmax/
    evaluation/<experiment_name>/
  xense/
    evaluation/<experiment_name>/
```

- 对外的后端名统一为 `analytic`、`zemax`、`xense`；`xense` 可在内部使用 MuJoCo/Xensim 实现。遵循用户指定的结果目录拼写：`zemax` 后端对应 `results_lut/zmax/`，其余分别对应 `analytic/`、`xense/`。在代码中集中维护这一映射，不在各入口重复拼接或猜测目录名。
- 每种仿真器只从自己的 `configs/<backend>.json` 读取其物理、光学、渲染和后端专属参数；不得依靠另一仿真器的配置隐式补值。所有可编辑 JSON 配置都放入 `configs/`，不在仓库根目录或 `experiment/` 保留新的配置副本。
- LUT 标定只有一套扫描、样本语义、拟合、留出验证与发布流程。标定入口通过参数显式选择仿真器，读取对应仿真配置和 `configs/calibration.json`；标定流程与阈值由标定配置控制，后端只负责产生统一约定的观测与几何数据。后端不支持某项能力时明确报错，不静默改变流程或验证要求。
- 每项 LUT 检验在 `experiment/<experiment_name>/` 有独立可运行入口，实验参数来自 `configs/<experiment_name>.json`。入口通过参数显式指定待检验的 LUT 结果目录或文件；从该结果的 manifest、配置快照和样本确定后端及数据语义，并检查输入兼容性，不以当前默认仿真配置替代生成该 LUT 时的条件。`lut_quality` 无须标定已发布；验证失败时只要样本和训练 LUT 完整，也可直接计算，并在报告中保留原状态。
- 正式标定输出默认写入 `results_lut/<后端目录>/`；该 LUT 的检验输出写入同一结果目录下的 `evaluation/<experiment_name>/`。多次运行必须使用明确的运行子目录或经过兼容性验证的续跑，不能覆盖历史标定、失败诊断或旧检验结果。独立的非 LUT 扫描实验若有自己的输出约定，应在该实验文档中说明。
- 迁移现有配置、入口和结果时保留原始数据、manifest 与可复现性信息。旧目录与新目录的对应关系要写清楚；不能把失败标定重命名为正式 LUT，也不能只因搬移文件就更改其通过状态。

## 通用工作方式

- 默认用中文沟通。先阅读相关入口、配置、文档及调用链，再修改代码；不要凭文件名推断行为。
- 用户要求解释、分析或诊断时，先提供证据与结论，不擅自实施修复；要求修改时再落实代码和验证。
- 工作区可能有未提交改动。保留用户现有改动，不重置仓库、不顺手整理无关文件、不覆盖历史输出。
- 修改尽量聚焦。遵循所在模块的命名、缩进和导入风格；新增公共接口提供类型标注和必要说明。
- 路径使用 `pathlib.Path`，文本/JSON 使用 UTF-8；避免硬编码本机绝对路径、GPU 型号或安装位置。
- 不为通过测试而降低验证标准、重写数值基准或隐藏异常；错误应说明对应配置、输入或输出路径。
- 性能优化必须保持处理数据量、采样范围、精度要求和物理语义；涉及近似或减少数据必须先取得用户同意。
- 修改 CLI、配置结构、指标定义或输出格式时，同步更新相应 README、文档和测试。
- 如果遇到可以并行运行、可以作为子任务的任务，务必使用subagent
- 请优先使用GPU实现计算。沙箱外有GPU

## 环境与常用命令

项目使用 Windows / PowerShell、`uv` 和仓库内 `.venv`。Python 版本以 `.python-version` 与 `pyproject.toml` 为准，依赖版本以 `uv.lock` 为准。

优先复用现有环境。仅在依赖缺失或用户要求时同步环境；不要无故升级依赖。完整 GPU 开发环境可按需安装：

```powershell
uv sync --cache-dir .uv-cache --locked --extra calibration --extra gpu --extra gpu-compile
```

以下命令均从仓库根目录运行。正式标定须选择新的空输出目录或符合续跑条件的未完成目录：

```powershell
.\.venv\Scripts\python.exe simulate.py --backend analytic --config configs/analytic.json --out results_<newname>
.\.venv\Scripts\python.exe calibrate_lut.py --backend analytic --calibration-config configs/calibration.json --dry-run
.\.venv\Scripts\python.exe calibrate_lut.py --backend xense --out results_lut/xense/<newname>
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/xense --dry-run
```

CUDA、编译后端及 Zemax 都有额外环境要求。缺少环境时明确说明未验证的后端，不以解析渲染或 CPU 测试替代 Zemax/GPU 的实际验证结论。

## 物理模型与标定边界

- 几何长度单位为 mm，未变形接触面位于 `Z=0`，`surface.size_mm` 按 `[X, Y]` 排列。
- 灯带发光面固定垂直于接触面，朝向对应边的内侧；不自动向原点倾斜。左右灯带长度沿 Y，上下灯带长度沿 X，发光带宽度沿 Z。
- `offset_mm` 是接触面边缘到发光平面的水平外侧距离；`z_mm` 是中心线的有符号 Z 坐标；`width_mm` 是完整发光带宽。不要把 offset 当成灯带宽度修正后的中心距。
- `power_rgb_w` 是各颜色通道的总功率，不是单位长度功率或 0–255 RGB 值。改变长度时不能静默调整总功率。
- CPU、CUDA、编译后端和 Zemax 应共享物理几何定义。修改模型时同步检查所有受影响后端，保留光源朝向、遮挡、功率与曝光的一致性。
- 区分物理辐射量、线性 RGB、显示/sRGB、观测 RGB 和有符号 RGB 残差；LUT 的输入语义不能被悄悄替换。
- 区分朝相机的光学法向与 LUT 使用的几何法向约定；涉及 `(p, q)` 和法向转换时核对符号及坐标系。
- 影响结果的物理/观测模型变化应更新相应版本标记、配置签名与缓存兼容判断；旧样本、参考图、LUT 和检查点不可默认复用。
- 诊断已有运行时，优先读取该运行保存的 manifest、配置快照和验证报告，不假定当前配置与运行时配置相同。
- 训练/验证按完整压入姿态隔离，避免同一姿态泄漏。保留逐姿态验证和越界判断，不能用总体平均掩盖失败姿态。
- 只有满足验证要求的结果才作为正式 LUT 发布；训练用 LUT、失败后的诊断 LUT 与正式结果应明确区分。

## experiment 构建约定

### 目录与入口

每个实验使用独立的 `snake_case` 包，相关脚本、说明及新结果放在该包内，不再向项目根目录添加一次性实验脚本。

```text
experiment/
  __init__.py
  config.py               # 共享配置加载、路径解析与校验
  README.md               # 实验索引及公共约定
  <experiment_name>/
    __init__.py
    __main__.py           # 薄入口，调用 runner.main()
    runner.py             # CLI 与实验流程
    README.md             # 目的、方法、参数、运行命令、输出和局限
    ...                   # 本实验需要的辅助模块/资源
    results/              # 仅用于不绑定某个 LUT 的独立扫描实验
configs/
  <experiment_name>.json  # 每项 LUT 检验独立的实验参数
tests/
  test_<experiment_name>.py
```

- 标准入口为 `python -m experiment.<experiment_name>`，并支持 `--help`。
- `__main__.py` 只做入口分发；不要把整个实验逻辑写入入口，也不要在导入包时启动渲染或写文件。
- 保持现有直接执行 `python experiment/<experiment_name>/__main__.py` 的用法；如需路径引导，限制在薄入口内。
- 辅助诊断若属于同一实验，放在其包内，通过子命令或参数调用；不要再散落为顶层脚本。
- 真正通用的几何、渲染、标定能力放在 `utils/`；实验特有的流程和指标放在 `experiment/`。同一指标只维护一份数值实现，其他实验导入复用。
- 新实验应接入 `experiment/README.md`；不在 `experiment/` 下创建 `test*.py`。现有历史结果保留原路径，不为了目录整齐而批量移动。

### 配置分离

- 每项 LUT 检验的参数由 `configs/<experiment_name>.json` 管理，共享加载逻辑集中于 `experiment/config.py`；新增检验应增加独立配置及校验，不把各项检验参数混在一份总配置里。
- 不把实验扫描范围、扰动量、指标分箱等写回主配置；不重新引入 `calibration.norm_diff`。
- 检验已有 LUT 时，物理配置以该 LUT 运行保存的配置快照和签名为准，不随当前 `configs/<backend>.json` 的后续修改自动变化；需要参考渲染时必须核对所选仿真配置与该快照的兼容性。需要重新扫描的独立实验若保存自己的物理快照，更新快照必须是显式行为。
- `norm_diff` 保存共享指标参数；`length_normdiff` 保存长度扫描设置；`norm_diff_evaluation` 保存已有样本的评估输入及输出设置。
- 对已有 LUT/样本做检验时，CLI 参数必须显式指定目标 LUT 结果；若 `simulation_config` / `--config` 指向物理配置，它仅用于匹配验证和必要的参考渲染，不提供指标参数。
- 参数优先级为：显式 CLI 参数 > 实验配置 > 明确的内置默认值。`None` 表示未覆盖时，不能误覆盖配置值。
- JSON 内相对路径以选中的实验配置文件所在目录为基准；CLI 相对路径以当前工作目录为基准。统一使用共享路径解析函数。
- 所有耗时操作前校验配置类型、范围、未知参数和路径；配置不合法应尽早报错，不静默回退。
- 长扫描类实验提供无渲染、无输出写入的 `--dry-run`，展示实际配置和工作量。

现有 LUT 检验入口示例：

```powershell
.\.venv\Scripts\python.exe -m experiment.norm_diff --lut-results results_lut/xense --exp-config configs/norm_diff.json --out results_lut/xense/evaluation/norm_diff_new.json
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/xense --exp-config configs/lut_quality.json --dry-run
.\.venv\Scripts\python.exe -m experiment.length_normdiff --exp-config configs/length_normdiff.json --dry-run
.\.venv\Scripts\python.exe -m experiment.length_normdiff --exp-config configs/length_normdiff.json
.\.venv\Scripts\python.exe -m experiment.length_normdiff --diagnostics-only
```

### 结果、续跑与可复现性

- 针对已有 LUT 的检验输出默认位于所选结果目录的 `evaluation/<experiment_name>/`，不得写入另一后端的结果目录。与任何 LUT 无关的独立扫描实验可继续使用其包内的 `results/`；允许通过配置/CLI 指定其他位置，但不能覆盖既有结果。
- 保存实际生效的参数，而不只是配置文件名：物理配置、指标设置、样本/姿态/候选数量、随机种子、角度范围及单位、RGB 域、聚合方式、后端和必要的模型/代码版本信息。
- 支持续跑的实验必须校验 manifest 与检查点兼容性，包括物理模型、输入、指标参数和实现版本；不兼容时明确拒绝，不能把旧结果标成新条件。
- 逐点完成后原子保存检查点，未完成点不得当作完成；无有效续跑信息的非空目录不能静默覆盖。
- 明确区分复用已有样本、重渲染参考图和重新执行完整扫描。报告运行方式时不要笼统称为“重新计算”。
- 对比实验应控制非目标变量。保持全扫描结果和固定中心诊断分开统计，记录采样/法向覆盖变化，避免把样本分布变化误当成光照改进。
- 绘图标清变量、单位、指标方向与条件；图表和原始数值一并保存。不能静默重算、改写或重新标注历史实验。

### NormDiff 的特别约定

- 修改前同时阅读 `docs/NormDiff_method_for_agents.md`、实验 README 和 `experiment/norm_diff/metric.py`；文档、论文与当前实现若有差异，应明确指出。
- 当前 NormDiff 是同一压入姿态内的 RGB 候选混淆指标，采用负值约定、越接近 0 越好；它不是 LUT 预测 RMS，也不能单凭它断言全局标定质量或倍数提升。
- `noise_fraction` 是每通道相对扰动的半宽，例如 `0.03` 表示 ±3%，不是标准差。候选噪声盒与可选的随机扰动 LUT 验证是不同流程。
- 保留并记录实际 RGB 域、角度权重、分箱、锚点、theta 上限及聚合规则。当前 phi 线性范围的环绕问题不能作为性能优化被静默改写。
- theta 上限为 null 时，已有样本评估与长度实验的自动取值规则不同；要严格对比，应显式指定一致的角度范围和指标设置。
- 全局 LUT 多值映射/歧义分析与同姿态 NormDiff 分开命名、实现和报告；可参考 `docs/LUT_ambiguity_metric_handoff.md`，但不能把设计文档当成已实现能力。

## 测试与交付

- 按改动范围验证：先配置校验、`--help` / `--dry-run` 和小规模固定样例，再运行相关测试；纯文档修改无需启动完整渲染或扫描。
- 测试使用独立、明确的配置 fixture，不依赖用户持续修改的示例配置来断言固定样本数或物理值。
- 数值/性能重构应在相同输入、采样量和物理条件下对照；记录误差容限、结果误差及性能，不只检查“能运行”。
- `tests/compile_reference/` 是数值对照实现，不要为让新实现过测而同步改写基准。
- 当前 `.gitignore` 忽略 `tests/` 和多类结果目录；搜索测试或输出时注意忽略规则，不要因默认搜索无结果就判断文件不存在。

常用测试命令（按需选择）：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_experiment_config tests.test_length_normdiff tests.test_norm_diff tests.test_light_geometry -v
.\.venv\Scripts\python.exe -m unittest tests.test_renderer_gpu -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_renderer_compile.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

交付时说明修改内容、相关文件、实际执行的验证及未验证项。已有失败与环境限制应如实报告；小规模冒烟测试不能表述为完整标定通过。涉及实验时补充运行入口、配置位置、输出位置，以及是否复用数据/需要重跑。
