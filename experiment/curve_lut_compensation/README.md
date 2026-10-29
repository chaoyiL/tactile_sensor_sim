# 已知模型的曲面 LUT 补偿迁移

本实验在历史平面标定 LUT 上比较两类迁移补偿，并与历史同曲率重标定 LUT 对照。入口为 `python -m experiment.curve_lut_compensation`，唯一实验配置为 `configs/curve_lut_compensation.json`。实验读取指定的平面 LUT 结果及 `curve_lut_transfer` 历史档案；不重新执行历史扫描，也不修改源档案。

端点补偿、局部响应矩阵、目标标签、样本筛选和评价口径见[平面 LUT 向曲面迁移的补偿方法](../../docs/平面LUT曲面迁移补偿方法.md)。

2026-10-28 的完整 GPU 档案对比已完成，见 [平面 LUT 补偿迁移实验结果](../../docs/平面-切平面LUT迁移补偿.md)：端点补偿在 R=200/100/50 mm 下将 RMSE 降低 14.59%/21.55%/22.68%；本次局部矩阵版本未改善误差。该结论限于下述已知模型条件。

本实验采用[迁移方法文档 §11.2](../../../recon_analysis/docs/plane_to_tangent_lut_calibration_transfer.md)第一层方法：光学参数固定为源 LUT 运行保存的快照，使用已知的仿真响应模型计算补偿。`H_pt`、局部矩阵及几何标签公式见[推导文档](../../../recon_analysis/docs/plane_to_tangent_lut_derivation.md)。固定模型为连续材料点响应 `H_pt`，并保留曲率几何、球压入、随几何移动的灯带、功率、可见性/阴影、相机因子、曝光与 Reinhard 映射。对源标定的每个姿态和材料样本，端点颜色补偿为：

```text
c_target = c_flat_observed + H_target_pt - H_flat_pt
```

其中 `c_flat_observed` 保留原始材料配对观测语义；`H_pt` 是连续仿真模型的预测响应，不能当成历史实际观测 `H_obs`。目标曲面的法向和局部坡度标签按目标参考面几何重新计算：`beta = -E.T @ n_d / (n_0 dot n_d)`。拟合沿用源运行的完整训练/验证姿态划分、阈值和发布/失败诊断规则；目标测试档案的颜色与真值不参与补偿参数估计或超参数选择。

第二类方法从无接触参考几何计算局部 RGB 响应矩阵 `D_q`，并与平面响应 `D_0` 比较投影、保留残差和直接伪逆补偿；另提供预先固定的 ridge 版本。比较中显式报告奇异值、条件数、秩失败和覆盖。矩阵失效样本不会从完整主结果中静默删除；不完整结果另附相同有效子域的配对对照。实验也复用冻结平面 LUT 基线和历史同曲率重标定结果，保留历史发布状态。

该比较使用已被理论分析使用过的历史测试档案，因此属于固定方案下的档案对比，不是新的盲测。本实验不声称完成平面参数识别、真实传感器参数识别、任意曲面推广、高度重建、实时部署或曲率节点间插值。结果可能变差；应按原始结果报告模型失配和域覆盖。

## 运行

```powershell
# 只读预检：展示实际来源、配置、划分与工作量，不创建输出或初始化 CUDA
.\.venv\Scripts\python.exe -m experiment.curve_lut_compensation --lut-results results_lut/analytic --transfer-results results_lut/analytic/evaluation/curve_lut_transfer/<run> --exp-config configs/curve_lut_compensation.json --dry-run

# 执行补偿、固定划分拟合与档案对比；默认新建唯一运行目录
.\.venv\Scripts\python.exe -m experiment.curve_lut_compensation --lut-results results_lut/analytic --transfer-results results_lut/analytic/evaluation/curve_lut_transfer/<run> --exp-config configs/curve_lut_compensation.json

# 在指定未完成运行中续跑
.\.venv\Scripts\python.exe -m experiment.curve_lut_compensation --lut-results results_lut/analytic --transfer-results results_lut/analytic/evaluation/curve_lut_transfer/<run> --exp-config configs/curve_lut_compensation.json --out results_lut/analytic/evaluation/curve_lut_compensation/<run> --resume
```

直接脚本入口也支持 `--help`。配置未知字段或非法值在昂贵计算前报错。默认结果位于所选源 LUT 的 `evaluation/curve_lut_compensation/<unique_run>/`。续跑只核对源输入、实际配置、数据语义、样本划分和已完成产物完整性；代码哈希或依赖版本变化本身不构成拒绝理由。历史源目录、历史曲面测试档案与 LUT 均只读。Windows 原子保存遇到错误码 5、32 或 33 时有限退避重试，持续失败仍抛错；已完整记录的检查点可以通过 `--resume` 复用。

## 公平比较与输出

比较条件为平面及历史档案中的 `R=200/100/50 mm` 节点，不进行节点间曲率插值。主结果使用所有条件共同材料域，另报各条件完整有效域；每姿态按参考面积归一、姿态等权，最后对均方误差开方。结果保留坡度 RMSE/P95、法向角误差、偏差、越界、原始支持、最差姿态、颜色支持、覆盖与不确定区间。颜色支持根据补偿后实际 LUT 查询颜色计算。矩阵无效点影响完整主结果的 `incomplete` 状态。

一次运行的主要输出结构如下：

```text
<source>/evaluation/curve_lut_compensation/<run>/
  manifest.json                 # 源/历史输入身份、完成检查点、拟合状态、执行信息
  config_snapshot.json
  split.json                     # 固定 source train/validation split 与历史独立测试姿态
  method_semantics.json
  samples/<condition>/migrated/<pose>.npz
  calibration/<condition>/       # 迁移 LUT、样本、配置、逐姿态验证与发布状态
  training_diagnostics/<condition>/{per_pose.csv,summary.json}
  matrix/<condition>.npz         # B/D、SVD诊断、补偿矩阵、有效域和有限差分残差
  metrics.json
  per_pose.csv / cross_method.csv / coverage.csv
  figures/{rmse_lut_comparison.png,rmse_matrix_comparison.png,matrix_coverage.png,figure_data.npz}
  runtime.json
```

`training_diagnostics/` 在 endpoint LUT 拟合完成后才读取历史目标训练样本，用于事后颜色/标签与覆盖差异诊断，不会回流到迁移颜色或参数选择。曲面 endpoint 拟合失败时只可能保存 `normal_lut_diagnostic.npz`，manifest 保留 failed 状态；只有通过原验证的拟合才保存正式 `normal_lut.npz`。`--dry-run` 打印来源、运行标识、划分、方法和工作量；正式档案路径与准确样本数以源 manifest、所选历史 transfer run 和最终 metrics 为准。

## 验证边界

本实验实现与数值验证结果收录于 [平面 LUT 补偿迁移实验结果](../../docs/平面-切平面LUT迁移补偿.md)。自动化测试使用独立固定 fixture，覆盖响应模型、法向符号、局部导数、平面极限、端点迁移、固定姿态划分、失败发布和续跑保护。CPU 测试仅用于小型 fixture；正式实验按配置要求使用 GPU。自动化测试通过不代表正式档案对比完成或补偿改善。
