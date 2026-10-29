# 已知模型的曲面 LUT 端点补偿与非线性反演

默认运行端点颜色迁移，评价冻结平面 LUT、端点迁移 LUT、历史同曲率重标定 LUT。另提供 v2 固定位置有限坡度非线性反演，与已有端点 LUT 比较精度和速度。入口为 `python -m experiment.curve_lut_compensation`，配置为 `configs/curve_lut_compensation.json`。实验读取指定的平面 LUT 结果及 `curve_lut_transfer` 历史档案，不重新执行历史扫描，也不修改源档案。

当前方法的使用量、连续点模型、端点颜色差、目标几何标签、拟合量与输出见[平面 LUT 向曲面迁移的端点补偿方法](../../docs/平面-切平面LUT迁移补偿方法.md)。

2026-10-28 的历史 GPU 对比记录在[结果 v1](../../docs/平面-切平面LUT迁移补偿结果_v1.md)：端点补偿在 R=200/100/50 mm 下将 RMSE 降低 14.59%/21.55%/22.68%。这份历史报告也保留了当时的局部矩阵对照；矩阵补偿现已从当前实现中退役，不属于当前运行方法。

## 运行

```powershell
# 只读预检：展示来源、配置、划分与工作量，不创建输出或初始化 CUDA
.\.venv\Scripts\python.exe -m experiment.curve_lut_compensation --lut-results results_lut/analytic --transfer-results results_lut/analytic/evaluation/curve_lut_transfer/<run> --exp-config configs/curve_lut_compensation.json --dry-run

# 端点补偿、固定划分拟合与历史档案对比；默认新建唯一运行目录
.\.venv\Scripts\python.exe -m experiment.curve_lut_compensation --lut-results results_lut/analytic --transfer-results results_lut/analytic/evaluation/curve_lut_transfer/<run> --exp-config configs/curve_lut_compensation.json

# 续跑兼容的未完成运行
.\.venv\Scripts\python.exe -m experiment.curve_lut_compensation --lut-results results_lut/analytic --transfer-results results_lut/analytic/evaluation/curve_lut_transfer/<run> --exp-config configs/curve_lut_compensation.json --out results_lut/analytic/evaluation/curve_lut_compensation/<run> --resume

# 非线性方法与已有端点 LUT 比较；加 --dry-run 可只读预检
.\.venv\Scripts\python.exe -m experiment.curve_lut_compensation --lut-results results_lut/analytic --transfer-results results_lut/analytic/evaluation/curve_lut_transfer/<run> --exp-config configs/curve_lut_compensation.json --compare-nonlinear --endpoint-results results_lut/analytic/evaluation/curve_lut_compensation/<completed_endpoint_run>
```

配置接受曲率条件、材料配对、指标、计算分块和可选 `nonlinear` 设置；旧配置显式带有 `matrix` 字段会报错。端点续跑核对源输入、实际端点设置、样本划分和检查点完整性，忽略不影响端点的 `nonlinear` 设置及已退役矩阵设置，不以源码或依赖版本变化本身作为拒绝理由。旧的 completed 运行只读校验；旧的 incomplete 运行可以复用已完成的端点迁移样本和 LUT。发生方法范围变化时，原报告以原内容归档。非线性比较使用独立新运行目录，并按姿态保存预测和计时检查点；其续跑要求算法参数及消费的端点 LUT 一致。

## v2 非线性比较

完整实测、精度/速度表和边界见[端点与非线性补偿对比](../../docs/平面-切平面LUT端点与非线性补偿对比.md)。

实现范围对应 [v2 方法](../../docs/平面-切平面LUT迁移补偿方法_v2.md) 式 (11)–(15)：在参考位置构造光学法向到曝光 RGB 的物理系数 `B`，保留有限坡度的法向归一化和 Reinhard。固定活动集版本分别运行 GN1/GN2/GN3 和闭式解；闭式需满足安全色域、矩阵条件、法向方向和活动集一致性，否则回退固定活动集 GN3。另有 `nonlinear_active_gn3`：每一步按当前坡度更新接收活动集，以全部参考可见光源求和并形成当前分支的 `B`，再计算该分支 Jacobian。两者均固定参考位置和参考阴影。`B` 用于当前构型的前向光度模型，与已退役的跨构型一阶 RGB 矩阵补偿不同。

初始坡度来自当前观测查询冻结平面 LUT；闭式先恢复 `model_background + observed_signed_RGB` 的绝对颜色再反演。算法直接输出局部坡度，不使用测试球心、位移或真实法向。测试标签仅在预测后评分；参数在读取测试结果前固定。本次未实现位移积分、接触阴影更新、残差学习、教师蒸馏、一般网格或动态序列。

`nonlinear` 参数：`steps` 为无重复的 1/2/3 步列表；`active_set_update` 决定是否加入活动集更新 GN3 对照；`damping` 为正规方程阻尼，`prior_weight` 为相对初值的先验权重，`max_step` 限制每步坡度增量范数；`color_epsilon`、`max_condition`、`minimum_normal_dot` 控制闭式条件；`benchmark_repeats`、`warmup_repeats` 控制热身和重复计时。默认值及严格校验在 `experiment/config.py`，显式配置在 `configs/curve_lut_compensation.json`。

比较复用相同的完整/共同材料域、面积权重和逐姿态等权指标，报告相对几何真值的误差和相对端点输出的差异，按锚点 bootstrap。闭式回退、活动集切换和数值失败单独记录；无效点不从主指标中删去，非有限输出回退冻结表并计数。物理反演不具有 LUT 支持率，原始颜色对源 LUT 的支持只作为诊断。

非线性运行输出 `predictions/<condition>/<pose>.npz`、`metrics.json`、`diagnostics.json`、`field_updates.json`、`runtime.json`、逐姿态/汇总 CSV 和 `figures/accuracy_speed.png`。`runtime.json` 分开报告全 122,640 材料点的构型场更新、每接触斑块主机 RGB 到主机坡度的耗时（含 CPU 初值查表、设备传输及同步），以及输入已常驻 GPU 的反演耗时。两种在线计时均执行同一算法和诊断，排除磁盘、指标及绘图；P50/P95/P99 来自所有姿态的重复测量。端点迁移/拟合成本引用已校验的历史计时并注明未重跑，端点在线查询在本次重新测量。上述耗时不包含相机曝光、图像采集/解码和几何估计，不能作为整幅图实时帧率。

## 比较与输出

实验使用平面及历史档案中的 `R=200/100/50 mm` 曲率节点，不进行节点间插值。三种方法在相同测试姿态上比较：`frozen_flat_lut`、`endpoint_migrated_lut`、`historical_same_curvature_lut`。指标在每个条件的完整有效材料域（`full`）和所有曲率共同材料域（`common`）上计算；每姿态按参考面积归一并等权聚合。报告包含坡度 RMSE/P95、法向角误差、偏差、越界、LUT 支持与材料覆盖及不确定区间。目标测试观测只用于查表评价，不用于迁移补偿或拟合选择。

一次运行的主要输出：

```text
<source>/evaluation/curve_lut_compensation/<run>/
  manifest.json                 # 输入身份、检查点、拟合状态、执行信息
  config_snapshot.json
  split.json                    # 固定 source train/validation split 与历史独立测试姿态
  method_semantics.json         # 当前端点方法和观测语义
  samples/<condition>/migrated/<pose>.npz
  calibration/<condition>/      # 端点迁移 LUT、逐姿态验证与发布状态
  training_diagnostics/<condition>/{per_pose.csv,summary.json}
  metrics.json
  per_pose.csv / cross_method.csv / coverage.csv
  figures/{rmse_lut_comparison.png,figure_data.npz}
  runtime.json
  runtime_sessions/<session>.json
```

端点连续点光度及迁移样本在配置指定的 GPU 上计算；LUT 拟合与预测查表使用 CPU。GPU 计时阶段在计时体前、后显式同步；开始同步不计入阶段秒数，结束同步计入。阶段墙钟包含阶段内读取、写入和检查点。`runtime.json` 的 `timing.wall_seconds` 从计时器构造后计到计时报告生成，不含此前的进程导入/启动及最终报告/manifest 落盘。`query_summary` 只累计 CPU/NumPy `_predict_method` 预测和 LUT 查询时间，不含样本读取、指标、bootstrap 与绘图；它包含在 evaluation 阶段内，不应再加到阶段总时间。完整从零耗时只适用于 fresh 输出且没有复用检查点；续跑时间仅代表本次调用。`training_diagnostics/` 在拟合完成后读取历史目标训练样本，只作事后颜色、标签和覆盖诊断，不回流到迁移样本或拟合选择。

## 验证边界

补偿与计时测试集中在 `tests/test_curve_lut_compensation.py`；迁移与共享存储测试集中在 `tests/test_curve_lut_transfer.py`。可运行 `python -m unittest tests.test_curve_lut_compensation tests.test_curve_lut_transfer -v`。

当前端点版本的代码裁剪、正式测试样本重评与 GPU 姿态回放见[端点补偿实验裁剪验证](../../docs/平面-切平面LUT端点补偿验证.md)。保留的指标和 bootstrap 与历史结果对应项逐值一致；该复核复用已有 LUT，不是重新完成一次正式标定。

实验沿用源运行的训练/验证姿态划分、阈值和发布/失败诊断规则；目标历史测试姿态及其颜色、真值不参与补偿参数估计。2026-10-28 的测试档案已用于理论分析，因此结果是固定方案下的档案对比，不是新的盲测。结果只支持所选源快照、曲率和样本条件，不代表真实传感器参数识别、任意曲面推广、高度重建、实时部署或曲率节点间插值。
