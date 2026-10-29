# 平面 LUT 补偿迁移实验结果

日期：2026-10-28。运行方式见 [实验说明](../experiment/curve_lut_compensation/README.md)，实现与数值验证结果一并收录于本文末尾。

## 结论

在本次已知仿真参数的圆柱实验中，**连续模型端点补偿降低了三个曲率条件下的坡度误差**，相对冻结平面 LUT 的 RMSE 降幅为 **14.59%、21.55%、22.68%**，数值接近历史同曲率重新标定。局部响应矩阵的六个版本均未改善曲面条件的误差。

这验证的是给定准确物理模型时的迁移方案；光学参数直接来自源标定快照，并非仅凭平面观测识别。历史测试档案已用于前期理论分析，本次是预先固定方案的档案对比，不是新盲测。

## 实际输入与运行

- 源 LUT：`results_lut/analytic`，源运行签名 `d54e4bea5e562c513cb3c6a1ebf10385ed3699ba367754c9ce0d84391d5f2390`。
- 历史迁移档案：`results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400`。
- 新结果：[20261028_endpoint_hpt_01](../results_lut/analytic/evaluation/curve_lut_compensation/20261028_endpoint_hpt_01/metrics.json)。
- 全部 704 个源姿态、2,279,043 条源样本；沿用 563/141 的训练/验证划分。每个条件复用 126 个测试姿态，共 504 个历史测试档案。
- 四条件为平面和 R=200/100/50 mm。正式源的分辨率、材料网格、全部光源、物理参数和原验证阈值均保留；没有重新执行旧 2,620 帧扫描。
- 连续点光度及矩阵计算使用 `cuda:0`，几何 float64、光度 float32、52 个光积分点，阴影开启；LUT 拟合复用原共享流程。

```powershell
.\.venv\Scripts\python.exe -m experiment.curve_lut_compensation --lut-results results_lut/analytic --transfer-results results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400 --exp-config configs/curve_lut_compensation.json --out results_lut/analytic/evaluation/curve_lut_compensation/20261028_endpoint_hpt_01
```

该目录现已完成。复核它时在命令后追加 `--resume`；重新做实验应选择新的目录。

## 端点补偿与主要结果

按迁移方法文档式 (5)，在相同材料点和源球姿态上计算：

```text
c_target = c_flat_observed + H_target_pt - H_flat_pt
beta_target = -E_target.T @ n_target / (n_reference dot n_target)
```

其中 $H_{\kappa,a}^{\mathrm{pt}}=C_{\kappa,a}(X_d,n_d)-C_{\kappa,0}(X_0,n_0)$；$C$ 的灯带求和、漫反射、相机因子、曝光及 Reinhard 具体公式，以及本次快照参数，见[补偿方法中的连续点 RGB 函数](平面LUT曲面迁移补偿方法.md#连续点-rgb-函数-c-的具体形式)。

`H_pt` 保留有限曲率下的完整球冠、光源几何、阴影、相机、曝光和 Reinhard 非线性。迁移颜色仍锚定原平面观测，目标标签重新由几何生成。目标训练观测只在全部迁移 LUT 拟合后用于诊断；目标测试观测不参与参数或超参数选择。没有使用冻结 LUT 的起点 Jacobian 做有限步长外推。

主要指标为坡度向量 RMSE（无量纲，越低越好）。每个姿态先按参考面积归一，姿态等权平均平方误差后开方；四条件使用同一个接触材料域，每条件 413,172 条测试点。四个冻结基线与历史报告的差均为 **0**。

| 条件 | 直接迁移平面 LUT | 端点补偿 LUT | 历史同曲率重标定 | 相对直接迁移降幅 |
|---|---:|---:|---:|---:|
| 平面 | 0.139938 | 0.139938 | 0.139938 | 0.00% |
| R=200 mm | 0.149990 | 0.128104 | 0.128401 | 14.59% |
| R=100 mm | 0.163866 | 0.128551 | 0.128940 | 21.55% |
| R=50 mm | 0.196348 | 0.151807 | 0.152471 | 22.68% |

端点补偿与同曲率重标定非常接近；本次没有对两者的小差异作专门显著性检验，不能据此认定补偿优于重标定。四个迁移 LUT 都通过原 141 个完整姿态的验证并保存正式 `normal_lut.npz`，阈值未降低。该通过状态不等于已经指定并通过了工程迁移精度要求。

![主要 LUT 对比与验证状态](../results_lut/analytic/evaluation/curve_lut_compensation/20261028_endpoint_hpt_01/figures/rmse_lut_comparison.png)

按 63 个材料锚点聚类 bootstrap，同锚点全部深度一起抽样，2,000 次，seed=42。以下差值定义为“补偿 RMSE − 冻结表 RMSE”，负值代表改善：

| 条件 | RMSE 差值 | 95% 区间 |
|---|---:|---:|
| R=200 mm | -0.021886 | [-0.024914, -0.019265] |
| R=100 mm | -0.035315 | [-0.038531, -0.032374] |
| R=50 mm | -0.044541 | [-0.047172, -0.041924] |

## 完整域、支持与误差分布

共同域保留了平面/R200/R100/R50 各自完整测试域平均面积的 95.37% / 96.54% / 97.70% / 100%。共同域由几何与原观测有效性确定，与所比较的 LUT 无关。

| 条件 | 完整域直接迁移 RMSE | 完整域端点补偿 RMSE | 共同域端点坡度 P95 | 共同域法向角 RMSE (°) | 共同域原始节点支持率 |
|---|---:|---:|---:|---:|---:|
| 平面 | 0.149781 | 0.149781 | 0.290162 | 7.058999 | 23.29% |
| R=200 mm | 0.160150 | 0.138136 | 0.264875 | 6.375335 | 26.42% |
| R=100 mm | 0.171914 | 0.137065 | 0.261983 | 6.355691 | 26.52% |
| R=50 mm | 0.196348 | 0.151807 | 0.317409 | 7.362470 | 22.55% |

端点补偿 LUT 的共同域颜色越界率均为 0，但严格原始节点支持率只有约 23%–27%，所以不能把“不越界”解释为标定支持充分。报告保留补全节点上的误差，不按支持率删点。共同域最差姿态 RMSE 在 R200/R100/R50 分别为 0.256634、0.253911、0.270619；逐姿态偏差和完整分布见 `per_pose.csv`，不能只用总体平均描述所有位置。

## 局部响应矩阵对照

按方法文档式 (15)–(22) 从无接触参考几何计算 `D0`、`Dq`；三种基本输出为 `F0(D0 Dq+ c)`、`F0((I+(D0-Dq)Dq+)c)`、`Dq+ c`。ridge 参数固定为 0.001，未用目标测试选择。下面全部是相同共同域的 RMSE：

| 方法 | 平面 | R=200 mm | R=100 mm | R=50 mm |
|---|---:|---:|---:|---:|
| 投影颜色后查平面 LUT | 0.212715 | 0.219633 | 0.253172 | 0.272682 |
| 保留残差颜色后查平面 LUT | 0.139938 | 0.181819 | 0.226444 | 0.250786 |
| 直接矩阵反演坡度 | 0.290360 | 0.275554 | 0.261460 | 0.241014 |
| Ridge 投影后查 LUT | 0.212746 | 0.219629 | 0.253135 | 0.272637 |
| Ridge 保留残差后查 LUT | 0.139938 | 0.181805 | 0.226412 | 0.250724 |
| Ridge 直接反演 | 0.290442 | 0.275670 | 0.261614 | 0.241238 |

![局部响应矩阵对比](../results_lut/analytic/evaluation/curve_lut_compensation/20261028_endpoint_hpt_01/figures/rmse_matrix_comparison.png)

四个条件的 122,640 个材料网格点均通过预定矩阵有效性检查，测试域没有因矩阵失效而删点；最小奇异值的全网格最小值约为 0.00922–0.01177。矩阵可辨识并未保证迁移准确。

保留残差版在平面精确还原冻结表；投影版在平面改变了非线性/响应子空间外的颜色，基线因此已经变差。曲面各版本均比直接迁移更差，与其小坡度、固定位置、固定活动光源的近似边界相容。本实验未分别隔离位移、非线性、阴影与 LUT 混叠的因果贡献，不能据此给出各项占比。

## 迁移样本与观测链差异

| 条件 | 源样本数 | 迁移保留数 | 历史目标训练样本数 | 源域未覆盖的目标样本 | 迁移颜色向量 RMSE |
|---|---:|---:|---:|---:|---:|
| 平面 | 2,279,043 | 2,279,043 | 2,279,043 | 0 | 0.000000 |
| R=200 mm | 2,279,043 | 2,254,308 | 2,254,561 | 253 | 0.016840 |
| R=100 mm | 2,279,043 | 2,231,521 | 2,231,690 | 169 | 0.018563 |
| R=50 mm | 2,279,043 | 2,177,783 | 2,177,783 | 0 | 0.017763 |

颜色误差在迁移与目标历史训练样本的交集上计算，姿态内面积归一、姿态等权；它是三通道有符号颜色向量误差，不是 0–255 显示 RGB 误差或坡度误差。同域保存的 float32 坡度标签差为 0。源域之外的 253/169 个点没有伪装成平面观测锚定样本。

连续模型 `H_pt` 没有模拟最终像素量化及双线性观测差分的全部影响，因此上述非零颜色误差是预期需要测量的残差，不能把 `H_pt` 当作 `H_obs`。端点迁移有效并不意味着所有观测误差都被消除。

## 补偿耗时实测（2026-10-29）

在新的空目录 [timing_20261029_d5331c8c](../results_lut/analytic/evaluation/curve_lut_compensation/timing_20261029_d5331c8c/runtime.json) 完整执行一次，未复用补偿检查点。源 LUT 和源标定样本已经存在：仍为 704 个源姿态、2,279,043 条样本、固定 563/141 划分；复用原四条件的 504 个测试档案。此次重新生成迁移样本、拟合迁移 LUT、构建局部矩阵并评价，没有重新渲染历史 2,620 帧。

环境：Windows 11，16 个逻辑 CPU，RTX 5050 Laptop GPU（8 GB），驱动 610.62，PyTorch 2.12.1+cu130。连续光度与矩阵计算使用 CUDA，几何 float64、光度 float32、52 个光积分点、阴影开启；LUT 拟合、NumPy 查询及指标主要使用 CPU。采用新的 Python 进程，未主动清空文件系统或编译缓存；以下为单次实测。

### 离线建表

单位为秒。迁移生成时间包含有效域检查、点光度计算、压缩保存和逐姿态检查点；拟合时间包含固定划分验证与最终发布表生成。

| 条件 | 迁移样本生成 | LUT 拟合验证 | 端点建表合计 | 本条件矩阵构建 |
|---|---:|---:|---:|---:|
| 平面对照 | 83.74 | 14.47 | 98.21 | 4.43 |
| R=200 mm | 124.00 | 13.95 | 137.95 | 5.10 |
| R=100 mm | 136.78 | 13.78 | 150.56 | 5.06 |
| R=50 mm | 133.19 | 13.67 | 146.86 | 4.96 |

三个目标曲率的端点建表合计 **435.37 秒（7 分 15 秒）**；加上平面对照为 **533.58 秒**。这些是本次顺序执行时各条件阶段的实测和，共享的输入读取/输出初始化 **3.82 秒**、模型/CUDA 初始化 **3.02 秒**另列，不包含源平面原始标定的成本。局部矩阵还需共享平面响应计算 **0.88 秒**，四条件矩阵构建共 **20.44 秒**。

事后训练诊断 **25.01 秒**，测试预测、指标与 bootstrap 合计 **56.01 秒**，绘图 **1.75 秒**。内部计时从参数解析后到绘图结束约 **643.67 秒**；兼容字段 elapsed_seconds 为 **639.84 秒**，从模型初始化计起。独立父进程计得从 Python 子进程启动到退出、包含导入和最终检查点归档的完整时间为 **656.43 秒（10 分 56 秒）**，见 [process_timing.json](../diagnostics/curve_lut_compensation/timing_20261029_d5331c8c/process_timing.json)。查询耗时已包含在评价阶段内，不能再次加总。

### 建表后的查询

下表为 CPU/NumPy 的每测试姿态平均查询时间（ms），由 126 个姿态的一次预测累计后除以姿态数。R200/R100/R50 分别查询 427,808 / 422,776 / 413,172 个完整有效域材料点；各评价子域复用同一预测。计时包含颜色变换、LUT 插值和预测数组操作，不含读样本、指标、bootstrap、渲染或整帧图像处理。

| 方法 | R=200 mm | R=100 mm | R=50 mm |
|---|---:|---:|---:|
| 冻结平面 LUT | 2.880 | 2.770 | 2.689 |
| 端点补偿 LUT | 2.870 | 2.766 | 2.697 |
| 矩阵投影后查 LUT | 3.124 | 3.270 | 3.050 |
| 矩阵保留残差后查 LUT | 3.261 | 3.121 | 3.133 |
| 矩阵直接反演 | 0.260 | 0.243 | 0.231 |
| Ridge 投影后查 LUT | 3.267 | 3.220 | 3.041 |
| Ridge 保留残差后查 LUT | 3.225 | 3.215 | 3.075 |
| Ridge 直接反演 | 0.249 | 0.244 | 0.251 |

本次测得的是归档接触材料点的批量查询，不能将其直接解释为 640×480 整帧重建延迟。计时明细、各阶段新算/复用计数和查询样本数保存在新运行的 runtime.json、runtime_sessions/ 中；执行日志为 [full_run.log](../diagnostics/curve_lut_compensation/timing_20261029_d5331c8c/full_run.log)。

计时改动通过 43 项相关测试。新旧运行的全部指标、bootstrap、覆盖与训练诊断逐值一致，四条件 LUT 均通过原阈值；原源标定、历史扫描与旧补偿结果的受检文件哈希和修改时间均未变。新运行的完成态续跑另耗时 23.33 秒，只校验产物，未导入 torch、未改写任何受检产物或计时会话；只读 dry-run 未创建输出。复核记录见 [verification.json](../diagnostics/curve_lut_compensation/timing_20261029_d5331c8c/verification.json)，这些复核时间不计入上述从零实测。

## 保存、续跑与最终复核

首次执行在 826 个已完成检查点后遇到 Windows 原子替换 manifest 的 `WinError 5`。增加仅针对指定 Windows 错误码的有限重试后，按完整性校验从断点续跑；已完成平面姿态没有重算。具体文件占用原因未定位，不归因于某个安全软件。

`runtime.json` 的 538.20 秒是成功续跑段的计算时间，未包含首次执行的已完成工作、人工检查或最终报告复核，不能当作从零全流程耗时。两个执行日志位于 `diagnostics/curve_lut_compensation/`。

最终检查修正了 `per_pose.csv` 中 `raw_flat_lut_*` 辅助字段的来源：所有方法均明确对冻结平面表计算原始颜色支持；各方法实际查询颜色的支持仍对其查询 LUT 计算。复核只复用保存的 LUT、矩阵和测试档案重新评价，不重新计算光照或拟合 LUT；所有主 RMSE 和 bootstrap 结果逐值一致，辅助统计最大回放差为 5.55e-17。原汇总 JSON 和 CSV 其余列保留不变。该检查记录在新运行 manifest 中，复核前 CSV 保存在诊断目录。

原平面标定、历史曲面扫描和旧报告保持原路径与原状态。尚未验证仅靠平面数据识别光学参数、真实传感器迁移、未见曲率插值、任意可展曲面、位移重建或实时端到端性能。

## 实现与数值验证结果

### 自动化验证记录

| 检查 | 执行方式 | 结果 |
|---|---|---|
| 修改前相关回归 | `tests.test_curve_calibration`、`tests.test_curve_lut_transfer`、`tests.test_experiment_config` | 主 agent 在修改前执行：22 项通过，18.569 s。修改后回归由主 agent 汇总。 |
| 包入口及直接脚本入口 | `python -m experiment.curve_lut_compensation --help`；`python experiment/curve_lut_compensation/__main__.py --help` | 均以退出码 0 返回，展示相同的来源参数、配置、输出、续跑和 dry-run 选项。 |
| 配置校验和只读 dry-run | `runner.main()` 使用独立配置与工作区合成 source/history 档案 | 通过。dry-run 报告 `cuda_initialized=false`，未调用 `torch.cuda.init()`、未创建输出目录，源/历史 manifest、LUT 和样本哈希不变。 |
| 固定 fixture 测试 | `python -m unittest tests.test_curve_lut_compensation.CompensationConfigAndGeometryTests -v` | 12 项通过，15.788 s。覆盖未知/非法配置、法向符号、端点平面恒等和曲面几何标签、目标材料有效域及饱和邻域、矩阵代数/秩/ridge、`D` 有限差分和平面极限、CPU/GPU 对照、固定 train/validation split、失败诊断 LUT 与检查点保护。 |
| 合成历史档案端到端烟测 | 上述 fixture 测试中的 `test_complete_historical_archive_fixture_and_completed_resume` | 通过。由源 fixture 通过 `MaterialSampler` 生成四个源标定姿态样本，先运行 10 帧、平面及 `R=100 mm` 两条件的历史 transfer，再运行完整补偿入口。结果状态为 `completed`，平面与 `R=100 mm` 历史基线均标记 `matched`，预期图表和矩阵文件生成。再次以 `--resume` 启动时不重建光度模型，manifest、metrics、runtime 的哈希和修改时间不变。该合成结果只验证流程连通性，不作为正式档案结果。 |
| 正式历史档案 GPU 对比 | 源 `results_lut/analytic`；历史 `20261024T034322_868406Z_8b6a0400`；新输出 `evaluation/curve_lut_compensation/20261028_endpoint_hpt_01` | 完成。704 个源姿态、2,279,043 条源样本；全部四条件和 504 个测试档案。CUDA、52 个光积分点、阴影、float64 几何及 float32 光度。四个迁移 LUT 按原阈值通过，冻结基线差全部为 0。 |
| Windows 保存容错 | `tests.test_curve_storage` | 3 项通过；只对指定 Windows PermissionError 有限重试，持续失败及其他错误均传播。正式执行曾在 manifest 替换失败后成功续跑，复用了已完成检查点。 |
| 最终相关回归 | `python -m unittest tests.test_curve_lut_compensation tests.test_curve_storage tests.test_curve_calibration tests.test_curve_lut_transfer tests.test_experiment_config -v` | **38 项全部通过，29.338 s**。包括新增的目标查询支持率与原平面支持率分离测试，以及新旧完整小样例流程。日志：`diagnostics/curve_lut_compensation/final_tests.log`。 |
| 图表与报告复核 | 三张正式 PNG、原始 figure_data、CSV/JSON | 图表已逐张查看，无文字裁切；LUT 图含发布状态，矩阵图含冻结基线。两列 raw_flat_lut 支持率已更正，主 RMSE 与 bootstrap 回放逐值一致；其他辅助统计的最大回放差为 5.55e-17，原汇总 JSON 与 CSV 其余列保持不变。 |
| 正式完成态续跑 | 正式命令追加 `--resume`，在独立 Python 进程执行 | 33.47 s 完成所有检查点校验；未导入 torch，manifest、metrics、per_pose、runtime 的哈希和修改时间均不变。记录位于 `diagnostics/curve_lut_compensation/completed_resume_verification.json`。 |

早期小样例曾出现 Matplotlib `tight_layout` 布局兼容性警告，最终绘图已改用受约束布局并将方法名称放入表格内部，正式 PNG 检查通过。最终测试日志中的 PowerShell `NativeCommandError` 来自将 unittest 正常的 stderr 进度输出重定向为 PowerShell 错误记录；unittest 最终结果为 `Ran 38 tests ... OK`，无测试失败。CPU 测试只用于小样例和接口对照。

正式源的只读 dry-run 另在全新 Python 进程核对：未导入 torch、未创建指定输出目录，恢复出 563/141 的固定划分和 126 个独立测试姿态。对历史 0/63/125 三个测试姿态、四种曲率共 12 组重新计算几何标签，最大分量差为 2.9802e-8，原有效点均仍有效。正式源第 000000 姿态的 GPU 平面迁移保留全部 1,789 条样本，颜色与坡度对源数组最大绝对差均为 0。

### 独立光度数值对照

物理模块还将连续点模型与既有曲面 renderer 独立比较。R=100 mm、球压入深度 0.7 mm、阴影开启、float64 radiometry 下，从一次完整 GPU 曲面渲染选取一个有效接触像素和一个有效背景像素，将 renderer 输出的世界坐标位置及几何法向送入 `PhotometryModel`。模型 `display_rgb` 与 renderer 按 `A = max(sensor @ response.T * exposure, 0)`、`display = A / (1 + A)` 计算的结果最大绝对差为 `1.11e-16`；`encode_srgb8` 与 renderer 的 RGB8 编码完全一致。

其他独立精度对照：float32 的 40 点 fixture 上，display RGB 最大差 `5.96e-8`、RGB8 量化不匹配数为 0；对 200,000 个随机 RGB（三通道共 600,000 个值）的 float32 量化对照也为 0 个不匹配。R=100 mm、无接触、阴影开启的两点 CPU/GPU 对照中，display RGB 最大差 `8.33e-17`，活动光源掩码完全一致；局部矩阵 `B`、`D` 的最大差分别为 `8.88e-16`、`6.66e-16`，固定 `B` 的中心有限差分误差为 `6.51e-9`、`1.93e-8`，两点秩均为 2。这些是独立小样例数值，不是全档案准确率或性能结论。

以下小脚本复现曲面 renderer 对照，并分别比较一个有效接触像素和一个有效背景像素（运行环境需有 CUDA）：

```python
import numpy as np
import torch
from experiment.curve_lut_compensation.photometry import PhotometryModel, encode_srgb8
from tests.curve_fixtures import curve_config
from utils.curve_renderer_gpu import CurveGpuContext

cfg = curve_config(100.0)
cfg['indentation'].update(center_uv=[0.0, 0.0], depth_mm=0.7)
cfg['render'].update(shadows=True, radiometry_dtype='float64')
ctx = CurveGpuContext(cfg, device='cuda')
sensor, _, gt, _ = ctx.render_tensors(config=cfg)
model = PhotometryModel(cfg, device='cuda')
for mask in (gt['contact_mask'] & gt['mask'], ~gt['contact_mask'] & gt['mask']):
    ij = tuple(torch.nonzero(mask, as_tuple=False)[0].tolist())
    response = model.evaluate(gt['position_mm'][ij][None].cpu().numpy(),
                              gt['normal_geometric_world'][ij][None].cpu().numpy(),
                              state_config=cfg)
    A = torch.clamp(sensor[ij] @ ctx.response.T * cfg['render']['exposure_scale'], min=0)
    expected_display = torch.clamp(A / (1 + A), 0, 1).cpu().numpy()
    np.testing.assert_allclose(response.display_rgb[0], expected_display,
                               rtol=0, atol=2e-16)
    np.testing.assert_array_equal(encode_srgb8(response.display_rgb)[0],
                                  ctx.encode_display(sensor[ij], cfg).cpu().numpy())
```

该模型接收材料点批次，不生成整幅图；遮挡求交复用 renderer 几何核。`H_pt` 分别计算接触状态和无接触参考状态的显示 RGB，再作差。以上对照核对的是响应实现一致性，不验证 LUT 对新曲面的泛化。
