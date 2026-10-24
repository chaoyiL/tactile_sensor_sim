# MuJoCo / Xense 第三仿真后端接入计划

> Historical design record: commands and config paths below predate the repository refactor. See [the current workflow](repository_workflow.md).

日期：2026-10-22。状态：已按本计划实现并完成固定胶体 SimpleSensor 后端的完整扫描与评测；FEM、长度/灯带参数实验仍受厂商接口和配套光学表限制。实施细节、命令和实测结果见 [MuJoCo / Xense 使用说明](mujoco_xense_usage.md)。下文保留原始计划与实施前调研记录，不能把其中“尚未实现”的阶段性描述当作当前状态。

## 1. 目标与建议

将 Xense 触觉仿真接入现有工程，形成 `analytic / zemax / mujoco` 三种后端，共用球扫描、材料网格采样、LUT 拟合、独立验证、歧义分析和实验评测接口。`mujoco` 明确表示 **MuJoCo 提供场景几何与深度，Xensim 提供触觉 RGB**，普通 MuJoCo 场景截图不能作为触觉图像。

首选路线为 **MuJoCo + Xensim `SimpleSensor`**：它与当前规定球冠形状、研究 RGB 到法向映射的任务更接近。`FemSensor` 作为同一后端的独立模式，待实际变形真值、材料对应和状态重置可验证后接入。两种模式分别记录，不能混合训练或合并统计。

本轮只新增此计划。后续实施首先解决运行环境与无 marker 导出，再改公共接口，最后开展完整扫描和评测。不能因演示窗口能运行，就宣布第三后端完成。

## 2. 已核查的现状

### 2.1 本地 Xense 库

| 对象 | 本地证据 | 对方案的影响 |
|---|---|---|
| `xense-mujoco` | [README](../mujoco-sim/xense-mujoco/README.md)、[pyproject.toml](../mujoco-sim/xense-mujoco/pyproject.toml)，版本 `0.0.1`，声明 Python `>=3.10`，安装依赖仅列 MuJoCo、NumPy | 是场景与可视化集成层，安装它不会自动获得完整触觉 SDK |
| Xensim | [readme](../mujoco-sim/xensim/readme.md)、两个 [Simple](../mujoco-sim/xensim/examples/example_simple_sensor.py) / [FEM](../mujoco-sim/xensim/examples/example_fem_sensor.py) 示例 | `SimpleSensor` 输入深度、输出 RGB；FEM 还计算形变、marker 和力 |
| 可用二进制 | `mujoco-sim/xensim/xensim-0.3.0-cp39-cp39-win_amd64.whl` 的 `WHEEL` / `METADATA` | 仅本地已提供的 Windows CPython 3.9 ABI；依赖 `numpy<=1.26.4`、PySide6。包含编译扩展，不能改 wheel 名称解决 ABI 冲突 |
| 标定及 FEM 资产 | wheel 文件目录包含 `xensim/assets/fem/g1-ws_table.npz`、`g1-ws.npz`，示例使用相同路径 | 前者是 Xense 正向 RGB 仿真输入，后者是 FEM 模型；与本项目最终拟合的逆向 `normal_lut*.npz` 分开管理 |
| 当前工程环境 | `.python-version=3.13`；本次只读查询实际为 Python `3.13.11`、NumPy `2.5.4`；未安装 `mujoco / xensim / xensesdk` | 现有 wheel 不能直接装入主环境；本轮无法实测图像和 marker |
| MuJoCo 示例 | [example_tac_sensor.py](../mujoco-sim/xense-mujoco/examples/example_tac_sensor.py) 使用机械臂夹爪、深度相机及 `FemSensor` | 应提取“深度→触觉 RGB”的流程，另建固定传感器加扫描球的最小场景 |
| 窗口与深度 | [base_env.py](../mujoco-sim/xense-mujoco/src/xensemujoco/mujoco_env/base_env.py) 的 `MujocoEnv` 默认创建可见 viewer；`get_cam_rgbd` 使用 near/far 线性插值，示例传感器相机为正交相机 | 不能直接照搬为任意投影的深度转换；批处理也需要独立管理 GL 上下文 |

只检查了分发包目录和元数据，没有加载或逆向编译二进制。厂商[产品页](https://www.xenserobotics.com/about/371)确认 XENSE SIM 面向 MuJoCo / Isaac Sim，但没有提供足以确认本项目所需全部接口的公开说明。实现时以当前交付包的实测为准，不能假定其他版本必然可安装或兼容。

### 2.2 当前工程的真实调用链

| 功能 | 当前实现 | 必须调整的位置 |
|---|---|---|
| 单帧与零压入参考 | [simulate.py](../simulate.py) 分支选择 analytic / Zemax | 加第三后端，并统一帧结果和保存接口 |
| 球扫描姿态 | [build_scan_poses](../utils/calibration_scan.py) 按深度、行、列确定性生成 | 原样复用姿态、端点和球半径规则 |
| 参考与材料采样 | [calibration_samples.py](../utils/calibration_samples.py) 直接调用 `AnalyticGpuContext` 张量接口、解析球冠求交和 CUDA 同步 | 拆出后端提供的观测与几何，保留解析后端现有优化路径 |
| 标定流程 | [calibration_pipeline.py](../utils/calibration_pipeline.py) 强制 CUDA，直接实例化解析上下文；参考仅在内存；存档含 `analytic_reference:` | 改为公共采集流程；移除通用流程对解析参考、CUDA 和光源模型的硬编码 |
| LUT 数值算法 | [normal_mapping.py](../utils/normal_mapping.py) 和 `fit_and_validate` | 复用分箱、MAD、填值、三线性查表、种子 42 的完整姿态留出及逐姿态发布判定 |
| NormDiff | [runner](../experiment/norm_diff/runner.py) 从残差加一张新渲染的解析参考恢复 observed RGB；[metric](../experiment/norm_diff/metric.py) 已独立 | 新数据从存档恢复参考，不再默认启动解析渲染 |
| 全局 LUT 质量 | [lut_quality](../experiment/lut_quality/README.md) 已实现，读取原始样本与 LUT，无需渲染 | 保留算法，补充后端、观测域和来源校验 |
| 长度实验 | [length_normdiff](../experiment/length_normdiff/README.md)、[length_lutqual](../experiment/length_lutqual/README.md) 直接使用解析 GPU；后者逐长度重新扫描、拟合并评测 | 共用采集接口，同时校验该后端能否真实改变目标物理参数 |
| 歧义及图表 | [calibration_ambiguity.py](../utils/calibration_ambiguity.py)、[calibration_visualization.py](../utils/calibration_visualization.py) 消费配对样本 | 继续复用，并保存新后端身份；诊断 stride 不影响拟合与完整评测 |

当前“两个渲染后端”不等于“两个后端已经都能完成 LUT 标定”：正式标定入口目前只支持解析 CUDA。公共化时同时为 Zemax 定义适配器，不能继续让公共标定依赖解析上下文。

## 3. 无 marker 图像：优先级最高的验证项

**现有证据支持尝试原生关闭 marker，但尚不足以保证导出的图像没有 marker；目前也没有证据说明无法输出无 marker 图像。**

1. 优先验证 `SimpleSensor.get_image()`。文档说它不做 marker 追踪，但“没有追踪”不证明标定背景和导出 RGB 中没有黑点。
2. 对 FEM 显式调用 `sensor.set_sim(show_marker=False, show_contact=False, show_force=False)`。README 和两个 FEM 示例都提供 `show_marker` 开关，但必须验证开关影响 `get_image()`，而不只是三维窗口覆盖层。
3. 对零压入、中心球压入、偏心球压入保存原始 RGB。固定输入和状态，对比 marker 开/关后的原始 `get_image()`；检查背景表是否已含 marker、是否有残余点阵，以及隐藏窗口是否改变结果。SDK marker 二值图若可用，只作为辅助检查。
4. 单独记录无 marker 验证报告、SDK 及资产哈希和图像证据。不能用差分背景抵消、修复黑点、遮罩删样本，冒充原生无 marker 输出。

默认策略设计为 `marker_policy=require_absent`。若 SimpleSensor 与 FEM 的原生导出都无法满足要求，停止正式扫描，向用户报告尝试的方法、具体 SDK 版本及带点原图，再由用户决定是否接受另行设计的遮罩方案。本计划不预先授权 marker 遮罩或图像修补。

`get_diff_image()` 文档返回 `uint8`，未证明是本项目需要的有符号线性 RGB 残差，不能直接拿来标定。

## 4. 运行环境方案

### 4.1 推荐：主环境与 Xense worker 分开

保留当前 Python 3.13 主环境及锁文件，用独立的 Python 3.9 worker 运行现有 Xensim wheel。主进程负责配置、扫描、采样组织、拟合、评测和报告；worker 只负责 MuJoCo 场景、Xensim 和图像/几何回读。

需要特别处理：**`xense-mujoco` 声明 Python >=3.10，而本地 Xensim wheel 只支持 3.9，这两个交付件没有可直接确认的共同解释器。** 推荐 worker 采用官方 MuJoCo Python API 加 Xensim 的公开接口，参考本地集成示例搭建最小扫描场景，不在 3.9 环境强装 `xense-mujoco`。如后续必须完整使用该 wrapper，应先取得厂商提供的兼容 Xensim 发行包。

环境阶段必须确认并锁定：

- Python 3.9 下可用且支持所需相机/深度功能的 MuJoCo 版本。
- 与 wheel 兼容的 NumPy、PySide6、`xensesdk.ezgl` 及其实际依赖；不能假定最新版 XenseSDK 仍提供示例中的接口。
- Windows OpenGL、隐藏窗口、主线程事件处理和图像回读行为；`visible=False` 不等于不需要图形上下文。
- wheel、厂商标定表、FEM 文件的来源、版本和 SHA256。二进制不随本项目重新打包分发，按本地 [License](../mujoco-sim/xensim/License) 管理厂商依赖。

如无法取得完整的 cp39 依赖组合，环境阶段的交付是明确缺失清单，需要厂商提供匹配发行包；不能以修改 ABI 标签、忽略安装限制或降级主环境代替解决。未来有兼容主环境的 SDK 时，可以替换 worker 实现，公共接口保持一致。

### 4.2 worker 通信与生命周期

采用持久子进程，控制消息为带协议版本和 request id 的 JSON，数组通过本次运行私有临时目录中的 NPZ 传输，禁止 pickle。操作包括初始化、零压入参考、指定姿态渲染、重置、关闭；错误带阶段、姿态和原始异常信息返回。

GL 初始化、渲染、事件更新、回读、释放都在 worker 主线程进行。每份配置复用同一运行会话；主进程负责超时、异常退出、日志及关闭回收。重试时从已确认完成的姿态续跑，不能把半写入文件当成完成。临时文件原子写入，只清理本次创建的私有临时目录。

## 5. 统一接口设计

### 5.1 分层

```text
simulate.py / calibrate_lut.py / experiment 各入口
                      │
        公共配置、扫描、采样与产物协议
                      │
          create_backend(analytic | zemax | mujoco)
               ┌──────┼─────────┐
           Analytic  Zemax   MuJoCo + Xensim worker
               └──────┼─────────┘
          配对样本 → fit_and_validate → 同一套实验指标
```

### 5.2 拟定的公共契约

名称为设计建议，实施时可按现有模块风格调整；职责和数据语义必须一致。

| 接口/数据 | 契约 |
|---|---|
| `describe_backend(config, task)` | 纯配置与能力检查，不启动 worker、GL、Zemax 或 CUDA；供 help / dry-run 使用 |
| `create_backend(config, run_context)` | 建立选定后端的会话；延迟导入依赖，缺少指定后端时明确失败 |
| `session.render_reference()` | 同一配置、同一观测设置下的零压入帧及几何/投影信息 |
| `session.render_pose(pose: ScanPose)` | 使用公共扫描姿态，返回实际触觉帧、可验证几何与计时 |
| `sample_frame(frame, reference, grid)` | 得到相同材料点上的 observed RGB、reference RGB、有符号残差、真值坡度、有效性和原始姿态分组 |
| `session.reset()` / `close()` | 可重复的无接触初态及资源释放；由公共流程保证异常时也关闭 |
| `RenderedFrame` | `rgb8[H,W,3]`、有效域、坐标与投影、实际表面/标签来源、后端身份；辐射量是可选字段 |
| `CalibrationObservation` | 对齐的 `dI[N,3]`、`slopes[N,2]`、材料坐标、网格索引、姿态 id；不把图像像素索引默认视为材料索引 |
| `BackendCapabilities` | 声明球扫描、原生无 marker、几何标签、材料对应、尺寸/光源可变性、物理辐射量、窗口模式等能力及其验证状态 |

公共协议不要求后端暴露 `torch`、`device` 或 `render_tensors`。解析适配器可以在内部继续用现有张量采样与编译/图捕获，避免逐帧强制 CPU 往返；主机数组和持久化数据在公共边界统一校验。

Zemax 适配器保留原光源、功率、遮挡、曝光和几何规则；每次参考/姿态渲染使用独立工作路径，不能覆盖前一姿态的 `.zos` 等文件。通用采样可消费其 RGB 与已知球冠 GT。其扫描适配与实际 ZOS-API 验证分别验收，没有 Zemax 环境时明确标为未验证。

### 5.3 CLI 兼容

- `simulate.py`：统一 `--backend analytic|zemax|mujoco`、`--config`、`--reference`、`--out`。
- `calibrate_lut.py`：新增同名 `--backend`，共用 `--dry-run`、输出规则和现有 `--ambiguity-only`。
- 需要采集的长度实验：使用同名后端选择，按能力校验目标参数。
- 纯已有样本评测：以后端存档身份为准；若提供后端断言，只能用于核对，不能改写样本来源或切换参考来源。
- 保持未指定后端的旧配置按 analytic 运行。`--device` 只对支持该选项的后端生效；Zemax 的 `--build-only` 等选项仍有明确限制。
- `--help` 和 `--dry-run` 不加载第三方运行时。dry-run 展示实际姿态、材料网格、预计渲染量、依赖静态匹配结果和能力缺口，不写输出。

## 6. MuJoCo 扫描环境与真值

### 6.1 最小场景

固定平面传感器、一个可定位的标定球、传感器深度相机和可选调试相机。不以机械臂、夹爪控制或策略训练作为 LUT 扫描的前置条件。

首期采用确定性的运动学定位：每个 `ScanPose` 精确设置球心并更新场景，不让重力、自由落体或刚体接触求解器决定压入量。深度相机只观察指定的球/几何组，不能被刚性胶体外壳、夹具、调试轴和 UI 挡住。MuJoCo 本身的刚体接触不能冒充胶体形变。

### 6.2 单位、坐标与深度

1. 公共坐标保留 mm、未变形面 `Z=0`、`surface.size_mm=[X,Y]`。MuJoCo 场景用 m，所有位姿通过显式 `T_sensor_to_world` 转换；法向只应用旋转，不能应用平移或长度缩放。
2. 球半径、中心、压入深度来自公共配置。公共球心 Z 为 `direction_z * (depth_mm - radius_mm)`；现有接触半径 `sqrt(2Rd-d²)` 和扫描端点规则保持一致。
3. 首期深度相机采用经过验证的正交映射，记录视场、像素中心、行列方向、near/far 及接触面到相机的距离。透视相机若启用，应使用对应版本的正确深度反投影，不套用示例的正交线性插值。
4. 默认朝向下，将度量深度相对未变形平面的位移转为 SDK 的 mm 负压入：`depth_sdk_mm = min(1000 * (D_m - D_plane_m), 0)`。其他朝向经显式坐标变换处理并测试，不能默默翻转坡度符号。
5. `D_plane_m` 来自实际相机/参考平面的关系，不复制示例的硬编码 `0.01`。无命中、裁剪、非有限值与真实零压入分别记录有效性。
6. 文档的尺寸参数为 `(width,height)`，数组是 `(height,width)`；输入示例中也有不一致的维度示意，因此必须用非方形网格实测转置、翻转和 RGB/BGR 顺序。

### 6.3 SimpleSensor 与 FEM 的标签边界

**SimpleSensor 主路线：** MuJoCo 生成球的实际深度输入，Xensim 生成 RGB；独立用公共球冠几何核对深度、接触圆和坡度。先验证 `smooth_mesh=0 / smooth_norm=0` 的实际含义，并使深度、渲染网格与公共材料采样分辨率满足同一精度要求。不能为了扫描速度擅自降低网格、分辨率或采样量。

若 SDK 内部重采样或平滑使实际着色法向偏离输入球冠，需要取得实际着色几何/法向并量化差异。输入解析球冠标签与实际渲染法向的对应通过前，不能把该批数据称为严格同几何对照。GT 误差容限在小样例阶段预先固定并写入报告，不在扫描失败后放宽。

LUT 法向始终采用 `n=normalize([-p,-q,1])`。朝相机的光学法向和 MuJoCo / SDK 局部法向先按坐标系转换，不能直接写入 `slope_samples`。

**FEM 扩展：** 坡度应来自实际变形表面，材料对应应来自实际网格/位移场，不能继续使用理想球冠标签或固定像素假装材料跟踪。`get_depth()` 文档给出正压入，与输入负压入相反，也必须验证。若公开 API 无法给出足够的表面与材料对应，FEM 只能作为明确标注的图像演示，不能作为正式 LUT 数据源。

逐姿态独立恢复 MuJoCo 和 SDK 状态；FEM 若没有可验证 reset，则重建传感器以保证初态。比较正序、逆序和重复姿态，确认前一姿态没有影响下一姿态。正式扫描保存实际收敛/重置设置。

## 7. RGB、参考图和产物协议

### 7.1 观测域

解析 / Zemax 可输出物理辐射量；Xensim 公开接口输出的是基于厂商标定表的 `uint8 RGB`。统一的是观测与标定接口，不能把 SDK RGB 改名为 `W/m²`，也不能再次套用解析曝光、Reinhard 和 sRGB 编码。

首选与当前标定一致的分析变换：对导出 RGB8 使用统一的逆 sRGB 解码，再在材料点插值，定义 `dI = observed - reference`，保留正负号。先核实 SDK 的颜色顺序、范围和传递函数；如其编码未获确认，必须标记该变换是分析约定，使用独立的 `observation_model_id`，不能宣称得到物理线性 RGB。未完成观测域对齐的结果可以做本后端诊断，不能标成严格同观测模型比较。

NormDiff 使用 observed RGB；LUT、LUT 质量和全局歧义默认使用有符号残差。两条输入路径分别声明，不以 SDK 的显示差分图代替。禁止逐帧自动曝光、归一化或自动增益；`rgb_gain` 等设置在参考与所有姿态中固定，并进入签名。

### 7.2 样本存档

保留 `normal_samples.npz` 的现有核心字段和 `allow_pickle=False` 加载方式：

- `color_residual_samples`、`slope_samples`、`image_index`、`image_paths`、`pose_ids`。
- `material_coordinates`、`st_samples`、`material_grid_index`、网格轴、参考法向、切向基和坐标图声明。

计划将新样本存档提升到明确的新版本，补充后端/模式、观测模型、GT 来源、实际参考颜色网格及有效域、参考哈希和配置/资产签名。参考网格保留足够精度，支持与现有规则相同的 `float64(archived_float32_dI) + reference` 恢复路径；可额外保存原始观测用于审计，但必须明确哪个数组是指标的规范输入。

这样新存档的 NormDiff 可以**只读数据计算，不启动 CUDA、MuJoCo 或 Xensim**。参考 PNG 可供人工检查，但不能代替实际参与采样的高精度参考网格。

旧存档走显式兼容路径：能确认是解析来源且配置/模型签名匹配时，允许沿用原参考重渲染，并在日志/报告中记录；来源未知或参考不匹配则拒绝恢复。绝不能用解析零压入图重建 MuJoCo 样本的 observed RGB。

### 7.3 LUT 与渲染输出

LUT 继续使用 `NormalCalibration` 的结构和 `lookup_slopes_raw`。若仅扩展元数据，现有 LUT 主体格式可保持，不为第三后端另造查表实现。加载与评测增加观测模型、来源、坐标及样本签名校验。

单帧保存改为消费公共帧对象：共同输出 `image.png`、`metadata.json` 和真实可用的几何 GT；只有后端实际提供辐照度/辐亮度时，才生成对应物理量 NPY。Xense 不输出伪造的辐射量文件，也不继承 `vertical_edge_strips_v1` 作为自身光学模型身份。

扫描至少保存：配置快照、`manifest.json`、原始样本、参考网格、逐姿态采样报告、`normal_lut_train.npz`、独立验证报告和已有诊断图。验证通过后才生成正式 `normal_lut.npz`；失败保留训练/诊断结果并标明失败，不降低原阈值。

## 8. 接入同一套评测实验

| 评测 | 接入要求 | 解释边界 |
|---|---|---|
| 标准 held-out LUT 验证 | 直接复用 `fit_and_validate`；同一完整姿态不跨训练/验证；同一姿态的重复采集也归同组 | 报告逐姿态角度 RMS、P95、越界率和判定，不用总体均值掩盖失败 |
| `experiment.norm_diff` | 新存档直接恢复 observed；复用现有 metric 和参数解析 | 同姿态候选、负值且越接近 0 越好，不是 LUT RMS |
| 可选 RGB 扰动 LUT 验证 | 复用 `evaluate_lut_noise`，只用保存的训练 LUT 及其完整姿态留出集 | 固定参考，仅扰动 observed，再相应改变残差 |
| `experiment.lut_quality` | 读取新后端原始配对样本和训练 LUT；沿用 `split=auto` | 同时报告 `A(tau)`、节点错误率、实际插值错误和训练圆心迁移；`A` 只约束固定节点模型 |
| 全局歧义/分支图 | 复用现有 ambiguity 流程、阈值和显式 stride | 与 NormDiff 分开；保留全部原始标定样本，不因诊断筛选改变训练集 |
| `experiment.length_normdiff` | 公共采集流程；各尺寸重新参考与完整扫描，固定中心单独统计 | 只有实际支持目标尺寸与光学参数变化时，才生成等价实验曲线 |
| `experiment.length_lutqual` | 每个有效尺寸重新扫描、拟合、held-out 验证，再用同一质量算法；保存阶段检查点 | 失败标定也在结果中保留，不能仅画通过的尺寸 |
| `length_normdiff --diagnostics-only` | 按后端能力分发，读取原运行 manifest | Xense 缺少物理辐射量时，原 W/m² 光照诊断明确不可用；若新增 RGB 灵敏度诊断，单独命名、标明域与有限差分设置 |

NormDiff 接入前已经对照阅读了 [方法说明](NormDiff_method_for_agents.md)、实验 README 与 `metric.py`。本计划保留现有实现约定：`noise_fraction=0.03` 是每通道 ±3% 半宽，候选覆盖同姿态全部有效样本，theta/phi 分箱及层次平均不变，phi 的线性范围环绕问题不在本次改写。严格比较时显式指定一致的 theta 上限，避免已有样本评价和长度扫描的 null 自动规则不同。

全局 LUT 质量以当前 `lut_quality` 实现为准；[交接文档](LUT_ambiguity_metric_handoff.md)是设计依据，不代替已有实现对连续插值、独立验证及未见节点的处理。

### 8.1 Xense 光学模型与长度实验的限制

Xensim 文档展示的是“深度/法向加厂商标定表→RGB”，未展示与本项目逐灯带 `offset / length / width / z / power_rgb_w` 等价的可配置光学接口。文档中的胶体宽高参数，也不能证明固定标定表和 FEM 模型可以物理有效地缩放到任意尺寸。

当前主配置及实验快照为 `[55.9,21.8] mm`，SimpleSensor 文档默认是 `[17.3,29.14] mm`，二者不能直接对齐。后续先确认资产真实感知尺寸及方向，建立独立的 Xense 配置，并为解析 / Zemax 建立显式匹配尺寸的比较配置，不修改用户当前配置来迁就 SDK。

长度/光源控制必须逐项核实，建立参数支持表：

- 真实支持：将实际物理参数、标定表及模型版本写入该点签名，并完整运行原实验。
- 需要按尺寸重新标定：必须取得每个尺寸对应的正向光学表；FEM 还需要匹配的力学模型。此项是厂商资产依赖，不能用一张表拉伸后声称满足。
- 仅支持显示或网格缩放：可以另立“数值缩放诊断”，但不是原长度/灯带实验，不能混入其科学对比。
- 不支持或尚未验证：dry-run 在耗时操作前明确拒绝该实验点，报告缺失能力，不能忽略参数或画出伪等价曲线。

完成固定传感器的 LUT 与同指标评测后，可以交付这一阶段的能力；**原长度/灯带实验的完整等价支持仍是独立验收项，未解决时必须明确报告，不能宣称所有光场实验均已覆盖。**

### 8.2 三后端比较协议

同一比较条件固定球半径、压入深度、扫描范围、材料网格密度、LUT 大小、完整姿态划分、随机种子、指标参数及可比较的观测变换。各后端分别生成自己的样本、参考和 LUT，再进入相同评测。不同后端的样本不混合训练；跨后端迁移如需开展，应单列任务和结论。

报告分开呈现两种条件：

1. 同几何和同采样协议下，比较各自实际传感器/光学模型的标定结果。
2. 只有证明光源、相机和观测条件也等价时，才称为同物理模型的渲染方法比较。

记录每个姿态有效样本数、theta/phi 覆盖、饱和/遮挡/越界比例；全扫描和固定中心保持分开。若有效域不同，保留各自完整结果，并另外报告预先定义的共同有效域结果及其覆盖损失。不得默默缩减原扫描以获得更好的分数。

如需要统一汇总图表，新增薄的 `experiment.backend_comparison` 包，只编排和汇总已有评测，不再复制指标数值实现。

## 9. 配置、文件和复现设计

### 9.1 配置

物理配置新增可选 `backend`（旧配置默认 analytic）与独立 `mujoco` 段，包含 sensor mode、运行解释器路径、SDK 标定/FEM 文件、深度/网格/输出尺寸、坐标变换、marker 策略、平滑与增益、窗口模式、状态重置及 FEM 迭代设置。胶体尺寸必须与公共表面和资产声明一致；重复声明冲突时直接报错。

配置校验分为公共几何/标定部分与所选后端部分。不能要求 MuJoCo 为通过校验填写虚假的 Zemax 设置或灯带功率，也不能让 Xense 默默忽略用户试图扫描的灯带参数。旧解析配置的校验与默认行为保留。

实验参数继续由 `experiment/exp_config.json` 和 `experiment/config.py` 管理。保留 `simulation` 独立快照；新增明确命名的后端比较配置段/各后端快照，不隐式从 `config.json` 继承。`norm_diff`、`lut_quality` 等指标参数仍只有各自一份来源，不写回物理配置。

CLI 显式值优先，`None` 表示不覆盖。JSON 相对路径以所属配置文件为基准，CLI 路径以当前目录为基准，统一 `Path` 及共享解析函数。新段拒绝未知键，运行前校验文件、ABI、尺寸、范围、观测域和目标能力。

### 9.2 拟新增/修改文件

```text
utils/tactile_backend.py               后端契约、能力与工厂
utils/tactile_observation.py           公共帧/配对样本与观测变换
utils/analytic_backend.py              包装当前解析渲染及高效采样
utils/mujoco_backend.py                主环境中的 worker 客户端
utils/zemax_backend.py                 保留现有实现，补公共适配接口
integrations/xense_mujoco_runtime/     独立依赖项目、锁文件、worker、最小球扫描 MJCF
utils/calibration_samples.py           公共材料采样与解析优化实现的衔接
utils/calibration_pipeline.py          后端无关的采集、拟合与保存
utils/config.py / utils/renderer.py    分后端校验和正确保存观测/辐射量
simulate.py / calibrate_lut.py         统一 CLI
experiment/config.py / exp_config.json 后端设置、独立快照与能力校验
experiment/{norm_diff,length_normdiff,length_lutqual}/
                                      读取新存档或调用公共采集流程
experiment/lut_quality/               来源/观测域兼容校验，数值定义不变
experiment/backend_comparison/        如需统一汇总，使用标准薄入口、runner、README
tests/test_tactile_backend.py          公共契约、旧后端回归、干运行
tests/test_mujoco_backend.py           IPC、单位/坐标、真值、marker 实测分组
tests/fixtures/                       独立小规模配置，不依赖用户示例参数
docs/mujoco_xense_usage.md             实施后再新增安装、运行与限制说明
```

同步更新涉及的现有使用文档及实验索引。新增测试全部放 `tests/`；厂商目录保持原貌，适配代码与运行环境不写入其二进制发行内容。以上文件本轮均不创建或修改。

### 9.3 身份、签名和续跑

统一签名显式包含 backend、sensor mode、物理/观测/几何标签模型版本、配置、采样规则、坐标图、SDK/wheel/MuJoCo 版本、标定表/FEM/MJCF 哈希、marker 策略及实测状态、平滑、曝光/增益、状态初始化规则和相关实现哈希。指标参数形成评测签名，避免仅更改指标便被迫重渲染。

不复用当前把 `LIGHT_GEOMETRY_MODEL` 写死的签名作为 Xense 身份。旧样本按其原版本读取，不能补写新身份后当作新模型数据。

支持逐姿态原子保存和扫描/拟合/评测阶段检查点；完成标记必须在对应数据和哈希写入成功后更新。续跑先验证条件、输入和产物。没有有效 manifest 的非空目录拒绝覆盖。新输出使用独立目录，如 `results_lut_mujoco_<run>/` 及各实验包的 `results/mujoco_<run>/`；历史扫描、LUT 和图表保持原路径。

## 10. 实施阶段与验收

| 阶段 | 工作与交付 | 进入下一阶段的条件 |
|---|---|---|
| P0：环境与原生图像验证 | 锁定可运行的独立 SDK 组合；验证 Simple/FEM 公开接口、隐藏窗口、RGB 编码、资产尺寸、无 marker 原图；输出参数能力表 | 至少一种模式能原生导出无 marker RGB；明确 ABI 与 wrapper 兼容路线；所需几何/观测信息可取得 |
| P1：最小 MuJoCo 球环境 | 固定传感器和运动学球；完整单位、投影、符号及非方形网格验证；记录 GL 生命周期和重置行为 | MuJoCo 深度、SDK 输入/输出、公共几何和实际着色法向在预先固定容限内一致 |
| P2：公共后端与产物 | 三后端适配、统一单帧/参考保存、公共观测协议、参考网格存档及配置签名 | 旧解析数值不变；Zemax 原行为保留；新后端不伪造物理辐射量；help/dry-run 无运行时依赖和副作用 |
| P3：扫描与 LUT | 统一入口完成小规模固定姿态扫描，再完整扫描；复用拟合/验证、图表与歧义分析；逐姿态/阶段续跑 | 同一输入可复现；完整姿态隔离；验证失败保留且不发布正式 LUT；通过后正式 LUT 可被原查表接口加载 |
| P4：同指标评测 | 无仿真依赖的 saved-sample NormDiff、LUT 质量、可选噪声验证；共同条件下的后端汇总 | 指标直接复用；存档来源不混淆；独立验证和覆盖完整可查；复用数据与新渲染明确区分 |
| P5：参数化实验及 FEM | 核实/取得各尺寸、光源条件资产；接入长度实验与诊断分发；满足真值条件时接入 FEM | 每个目标变量都真实生效；不支持项明确列出；FEM 使用实际变形与材料对应，不偷用球冠标签 |

P0 若无 marker 验证失败，应立即通知用户，不继续正式扫描。P5 若受厂商光学接口/资产限制，不以数值拉伸或忽略参数宣布完成；交付已完成范围和剩余条件。

### 10.1 验证层次

- **公共协议与回归：** 独立 fixture 检查旧 CLI、配置优先级和路径规则、无写入 dry-run、延迟导入、CPU/张量边界、帧保存语义；相同解析输入下样本、参考、LUT 和指标按原精度对照，不改 `tests/compile_reference/`。
- **worker：** 真实 Python 3.9 ABI 启动、版本握手、连续多帧、异常传播、超时退出、断点恢复和资源关闭；不能只用 mock 宣称 SDK 可用。
- **几何：** 平面零深度、中心/偏心/正负轴移动、多个深度、边缘相切、非方形网格、单位缩放、图像翻转、法向符号及无效像素；球面以解析解作数值对照。深度、位置、法向和重复帧误差容限在批量运行前写入测试 fixture 与验证报告。
- **图像：** marker 开/关的实际导出、背景检查、RGB/BGR、饱和像素、固定增益、参考一致性；确认 `update()` / 渲染时序，不能读到上一帧。
- **标定：** 两个以上独立压入姿态；保持拟合规则，检查训练/验证无交叉、逐姿态越界、故意失败时没有正式 LUT，以及主环境能加载和查询 worker 数据生成的 LUT。
- **评测：** 同一规范数组直接调用 metric 与存档重载后的结果一致；已有样本评价不启动任何渲染；NormDiff 与质量指标定义、角度范围、候选数量及权重不变。
- **实际后端验收：** 在匹配条件下执行完整 MuJoCo 扫描及所需评测，保存全部原始数值和报告。解析 CPU、mock 或小规模冒烟均不能替代真实 MuJoCo / SDK、GPU 或 Zemax 的对应验证。

实施后先运行相关单元测试，再执行独立小规模真实 SDK 冒烟，最后才运行完整扫描/参数实验。每次交付注明后端、配置位置、输出位置、是否复用样本、是否重渲染参考、是否重跑完整扫描，以及尚未验证的环境和功能。

## 11. 预期统一用法

以下是**实施后的接口目标，当前尚不可执行**；`config.mujoco.json` 应是新建且经验证的独立物理配置，不覆盖现有示例。

```powershell
# 单帧与相同条件下的无压入参考
.\.venv\Scripts\python.exe simulate.py --backend mujoco --config config.mujoco.json --reference --out results_mujoco_<run>

# 无渲染、无输出的扫描检查，再执行完整扫描与 LUT 标定
.\.venv\Scripts\python.exe calibrate_lut.py --backend mujoco --config config.mujoco.json --dry-run
.\.venv\Scripts\python.exe calibrate_lut.py --backend mujoco --config config.mujoco.json --out results_lut_mujoco_<run>

# 对已经保存的同一批样本做评测；新存档不需要启动参考渲染
.\.venv\Scripts\python.exe -m experiment.norm_diff --exp-config experiment/exp_config.mujoco.json --config config.mujoco.json --samples results_lut_mujoco_<run>/normal_samples.npz --out experiment/norm_diff/results/mujoco_<run>.json
.\.venv\Scripts\python.exe -m experiment.lut_quality --exp-config experiment/exp_config.mujoco.json --samples results_lut_mujoco_<run>/normal_samples.npz --lut results_lut_mujoco_<run>/normal_lut_train.npz --out experiment/lut_quality/results/mujoco_<run>

# 仅当后端通过目标尺寸/光学参数能力检查时执行长度实验
.\.venv\Scripts\python.exe -m experiment.length_normdiff --backend mujoco --exp-config experiment/exp_config.mujoco.json --dry-run
.\.venv\Scripts\python.exe -m experiment.length_lutqual --backend mujoco --exp-config experiment/exp_config.mujoco.json --dry-run
```

后续使用说明需提供独立 runtime 的确切安装命令和锁文件；在 P0 确认发行包前，本计划不编造一条保证可用的安装命令。

## 12. 依据与尚未验证项

本计划依据仓库当前工作区源码、本地厂商 README/示例、wheel 元数据及已有实验说明；未覆盖或改写工作区现有改动。公开参考查阅日期同本计划日期：

- [XENSE SIM 产品说明](https://www.xenserobotics.com/about/371)：确认平台定位，不作为具体 SDK API 或无 marker 支持的证明。
- [MuJoCo Python 官方文档](https://mujoco.readthedocs.io/en/stable/python.html)：最小场景、状态与渲染资源管理参考；worker 需按最终锁定版本核对。
- [MuJoCo 相机官方说明](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera)：投影与相机参数参考，不替代本地深度数值验证。

本轮未安装或导入 Xensim，未创建 MuJoCo 场景，未生成图像、扫描样本或 LUT，未执行任何渲染/测试。因此**无 marker 的真实导出、可运行依赖组合、有效传递函数、实际着色几何及可参数化光学能力均仍待 P0/P1 验证**。

