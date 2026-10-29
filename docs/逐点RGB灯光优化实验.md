# 逐点 RGB 灯光优化实验结果

记录日期：2026-10-23。本文汇总已完成的逐点光源对照、候选扫描与最终配置验证。历史条件以各运行保存的 manifest 和配置快照为准。

## 已验证光源模型与条件

`configs/analytic.json` 现在使用 `emission_model: "discrete_rgb_points_v1"`，灯带沿长度方向按 `point_spacing_mm: 6.25` 排列。每点的 `point_power_rgb_w[i] = [R,G,B]` 为该点三个通道的绝对辐射功率（W）；三通道比例决定颜色，三通道之和决定该点总功率。允许零功率点，不自动归一化或保持总功率。所有灯带不能同时全灭。

该模式以 `point_spacing_mm`、`point_power_rgb_w` **替代** `samples_length`、`power_rgb_w`；同时填写新旧字段会报错。点数为 `floor(length_mm / point_spacing_mm) + 1`，整列居中，点距严格为给定值，端部留白均分，不拉伸间距来填满灯带。

| 侧面 | 长度 | 点数 | 各点沿灯带中心的偏移（mm） |
| --- | --- | --- | --- |
| left / right | 20 mm | 4 | -9.375、-3.125、3.125、9.375（沿世界 Y 递增） |
| top / bottom | 56 mm | 9 | -25、-18.75、-12.5、-6.25、0、6.25、12.5、18.75、25（沿世界 X 递增） |

`point_power_rgb_w` 按上述顺序填写，必须恰有对应点数行，每行三个有限非负数。例如将某一行改成 `[0.02, 0.01, 0]`，只调整该点为红光 0.02 W、绿光 0.01 W。第一次改造的等功率基线保留原先每边各 0.15 W 的总功率和颜色；后续优化允许改变总功率，当前配置采用下述平滑梯度方案。

灯带中心、长度、宽度、offset、Z、朝向不变。`samples_width` 仍对原宽度作均匀积分，每个 LED 的功率在这些积分节点间均分；这些宽度节点不是额外 LED，6.25 mm 指沿灯带长度的 LED 间距。CPU、CUDA eager、CUDA graph 和编译渲染共用同一套点位和功率。修改灯光后须新建 `AnalyticGpuContext`，旧上下文拒绝使用新灯光，避免复用旧 CUDA 常量。

逐点模型目前只支持 analytic；Zemax 收到该模式会明确报错，历史均匀灯带的 Zemax 行为不变。几何版本仍为 `vertical_edge_strips_v1`；发光模型版本和逐点功率包含在完整配置及既有标定签名中，旧样本和参考图不能复用为新分布的结果。改变功率后使用新的输出目录重新扫描和标定，标定、验证与 LUT quality 计算逻辑保持不变：

```powershell
.\.venv\Scripts\python.exe calibrate_lut.py --backend analytic --calibration-config configs/calibration.json --out results_lut/analytic/<new_run> --dry-run
.\.venv\Scripts\python.exe calibrate_lut.py --backend analytic --calibration-config configs/calibration.json --out results_lut/analytic/<new_run>
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results results_lut/analytic/<new_run>
```

2026-10-23 优化起点：既有 `analytic/previous_failed/evaluation/lut_quality/curves.csv` 的 `A(0.1)=0.4608705`，`xense/evaluation/lut_quality/curves.csv` 为 `0.3024093`（越低越好）。目标是在保持 analytic 几何与标定配置不变的条件下，仅优化逐点 RGB 功率来达到该水平。`A` 是固定最近节点模型的经验单值不可兼容比例，不等同于实际三线性 LUT 的预测误差。两次历史运行的表面尺寸分别为 55.9×21.8 mm、17.3×29.14 mm，姿态数为 704、275，观测模型也不同；曲线差异本身不足以证明原因完全是光场。比较时应同时保留训练曲线、完整姿态留出结果、样本覆盖及原标定状态。

## 首轮对照记录（2026-10-23）

首轮选中的互补色候选为：left 总蓝光 0.45 W，right 总红光/绿光各 0.225 W，top 总红光 0.15 W，bottom 总绿光 0.15 W，合计 1.2 W；各边内部均分到独立可调的发光点。后续平滑梯度方案已取代它作为当前配置，首轮结果仍完整保留。

| 条件 | A(0.1) ↓ | 训练集实际 LUT 失败率 ↓ | 留出实际 LUT 失败率 ↓ | 原始样本数 |
| --- | ---: | ---: | ---: | ---: |
| [历史 analytic](../results_lut/analytic/previous_failed/evaluation/lut_quality/curves.csv) | 0.460871 | 0.639543 | 0.680654 | 2,279,043 |
| [等功率离散基线](../results_lut/analytic/discrete_625_baseline_20261023/evaluation/lut_quality/curves.csv) | 0.470438 | 0.636299 | 0.679050 | 2,279,043 |
| [整体功率 ×0.1](../results_lut/analytic/discrete_625_power01_20261023/evaluation/lut_quality/curves.csv) | 0.560799 | 0.778496 | 0.800208 | 2,279,043 |
| [整体功率 ×10](../results_lut/analytic/discrete_625_power10_20261023/evaluation/lut_quality/curves.csv) | 0.464618 | 0.627880 | 0.652620 | 2,276,118 |
| [沿边 RGB 交错](../results_lut/analytic/discrete_625_rgbcycle_20261023/evaluation/lut_quality/curves.csv) | 0.731425 | 0.919648 | 0.933994 | 2,279,043 |
| [左右互补色、增强侧光（首轮最佳）](../results_lut/analytic/discrete_625_complement3_20261023/evaluation/lut_quality/curves.csv) | 0.436504 | 0.569368 | 0.602700 | 2,279,043 |
| [历史 xense](../results_lut/xense/evaluation/lut_quality/curves.csv) | 0.302409 | 0.425653 | 0.494477 | 870,408 |

五组新候选均重新执行完整 704 姿态扫描（含参考图共 705 次渲染），未修改几何、相机、曝光、色调映射、阴影、数值精度、标定设置或 quality 指标。逐项核对 manifest 后，五组之间只有 `point_power_rgb_w` 不同。每组 quality 复用各自完整样本和训练 LUT，保留同一组完整姿态留出划分，不做下采样。10 倍功率出现饱和筛选，少 2,925 个样本，不能据此声称同样本条件下的提升。

首轮五个新候选均未通过原逐姿态标定验证，未发布正式 LUT。互补色候选的整体留出角度 RMS 为 13.14175°，仍有 6/141 个姿态失败；历史 analytic 为 14.71538°、14/141 个姿态失败。质量曲线改善不能替代逐姿态发布标准。

0.1 倍功率候选在完成扫描、拟合和验证后，附加全局歧义可视化产生超过五千万对候选，因而手动中断该阶段；manifest 明确保留 `run_status=interrupted`、`stage=visualization_interrupted` 和原验证失败状态，其 quality 检验完整完成。其他新候选的标定流程均运行至结束并按原验证失败退出。历史结果均未改写。

相关验证共 59 项通过：逐点采样/功率/旧配置兼容性、CPU 与 CUDA eager/graph/编译路径、灯带几何、物理线性、标定采样、原编译数值基准、实验配置与 LUT quality。旧均匀光源的测试改用固定 fixture，不依赖持续编辑的示例配置；`tests/compile_reference/` 未改。新模型另与显式逐个独立均匀小光源对照，辐射量容限 2e-15（相对）/1e-13（绝对），CUDA 对 CPU 容限 3e-10/1e-13。编译对照中一个临时目录权限问题在沙箱外重跑通过。没有实际重跑 Zemax 或 xense，Zemax 对新发光模型的拒绝行为已测试。

## 第二轮方案：平滑颜色梯度

第二轮结束时采用了 `gradient_full`：每边 RGB 总功率均为 0.4 W，合计 1.6 W。left 为蓝、right 为绿；top 一半功率为红，其余 G/B 比例沿 X 平滑变化；bottom 的全部功率在 G/B 之间按同方向平滑变化。没有改变灯带点位、几何、曝光或标定设置。

完整结果位于 [gradient_full](../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_v2/gradient_full/manifest.json)，其配置快照和签名保存在该运行中。704 个姿态、2,279,043 个样本均保留；141 个完整留出姿态全部通过，整体角度 RMS 为 9.80192°，最差姿态 RMS 为 19.11423°，按原标准发布了该运行自己的 `normal_lut.npz`。

在 `tau=0.1`：A=0.344724，训练实际 LUT 失败率=0.432807，留出实际 LUT 失败率=0.472099。xense 对应值为 0.302409、0.425653、0.494477。A 曲线目标尚未达到，不能把部分指标超过 xense 说成完整目标已完成。优化期间重复查看了固定留出结果，因此它是固定条件对照，不是额外的全新独立测试集。

可复现的候选搜索入口和参数见 [light_distribution 实验](../experiment/light_distribution/README.md)；所有候选重新完整标定，仅 quality 阶段复用该候选刚生成的样本和训练 LUT。

## 最终采用方案：颜色梯度、端部加亮、下边减半

2026-10-23 用户要求停止搜索后，已终止在运行的候选，保留中断记录。当前 `configs/analytic.json` 采用完整完成的最佳方案 `bottom_half`，与[该运行的配置及签名](../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_spectral/bottom_half/manifest.json)完全一致。目标 `A(0.1)≤0.3024` 未达到，不标为优化目标完成。

26 个可独立控制的 RGB 发光点保持 6.25 mm 间距。左边 0.4 W 纯蓝、右边 0.4 W 纯绿，4 个点均分各边功率；上边 0.4 W、下边 0.2 W，各有 9 个点，总功率 1.4 W。长边点位为 `x=-25,-18.75,…,25 mm`，令 `t=x/28`、`g=(1+t)/2`，点功率权重为 `w=1+2t²`。各点总功率为 `P_side*w/sum(w)`；上边 RGB 比例为 `[0.75,0.25g,0.25(1-g)]`，下边为 `[0,g,1-g]`。宽度积分、灯带几何、相机、曝光和标定均保持原条件。

最终 `A(0.1)=0.331561`，相比原始 analytic 的 0.460871 降低 28.06%。训练实际 LUT 失败率为 0.408611，留出实际 LUT 失败率为 0.443388；xense 对应 A 为 0.302409，训练及留出实际失败率为 0.425653、0.494477。不同指标不可混称：A 尚未追平 xense，实际查表误差在这些保存结果上较低。

完整 704 姿态和 2,279,043 样本保留，563/141 个训练/留出姿态与固定基线一致。坐标、斜率、法向、姿态编号等非颜色样本数组逐项完全相等。141 个留出姿态全部通过原阈值；整体 RMS 9.092938°，最差逐姿态 RMS 17.463430°。正式 LUT 保存在该运行自己的 `normal_lut.npz` 中，原有结果未覆盖。

最终[曲线对比图](../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_spectral/final_summary/curves_comparison.png)、[灯带逐点分布图](../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_spectral/final_summary/light_distribution.png)、[全部候选数值](../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_spectral/final_summary/all_trials.csv)、[审计报告](../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_spectral/final_summary/report.json)均已保存。对比使用历史 xense 结果，没有重跑 xense；相同留出集被反复查看，不能称为全新的独立泛化验证。

四侧固定发光法向分别为：left `[1,0,0]`、right `[-1,0,0]`、top `[0,1,0]`、bottom `[0,-1,0]`。修改 `offset_mm`、`z_mm`、`along_offset_mm`、尺寸或表面大小，只移动/缩放发光矩形，不旋转发光面。例如 `z_mm=-1, width_mm=2` 对应完整发光带高度 `[-2,0] mm`；均匀模式使用带内积分采样点，不包含矩形边缘。逐点模式按上表明确安排发光点，宽度方向仍使用带内积分节点。

新模型标记为 `vertical_edge_strips_v1`，CPU、CUDA 和 Zemax 共用同一几何定义。旧版倾斜灯带的渲染、标定和 NormDiff 结果仍是历史数据，不可当作新模型的结果；修改模型后需要重新渲染与标定。标定配置签名已包含模型版本，防止旧残差样本与新模型 reference 混用。旧文件不会自动删除或重写。
