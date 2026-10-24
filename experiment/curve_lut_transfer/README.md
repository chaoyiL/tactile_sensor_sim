# 平面 LUT 向曲面切平面的迁移

冻结一个已发布的 analytic 平面 LUT，对新生成的平面和圆柱曲面色差查表；同时按源标定的完整姿态及原训练/验证划分，重新拟合各曲率 LUT。所有 LUT 在同一套独立测试数据上比较。方法与范围见 [实验计划](../../docs/plan_curve_lut_transfer.md)，符号见 [局部重建原理](../../docs/局部重建原理.md)。

## 运行

从仓库根目录执行：

```powershell
# 只读预检：不初始化 CUDA、不渲染、不建立输出目录
.\.venv\Scripts\python.exe -m experiment.curve_lut_transfer --lut-results results_lut/analytic --exp-config configs/curve_lut_transfer.json --dry-run

# 完整实验，自动创建新的运行目录
.\.venv\Scripts\python.exe -m experiment.curve_lut_transfer --lut-results results_lut/analytic --exp-config configs/curve_lut_transfer.json

# 恢复指定运行；配置、源文件和实现签名必须一致
.\.venv\Scripts\python.exe -m experiment.curve_lut_transfer --lut-results results_lut/analytic --out results_lut/analytic/evaluation/curve_lut_transfer/<run> --resume
```

也支持 `python experiment/curve_lut_transfer/__main__.py --help`。`--out` 必须在所选源结果的 `evaluation/curve_lut_transfer/` 下；非空目录需要显式 `--resume`。恢复已完成运行仅核验文件，不重新渲染或改写报告。`--dry-run --resume` 可以只核验续跑条件。JSON 配置中的实验参数由 `experiment/config.py` 统一校验，CLI 路径相对当前目录。

唯一实验配置是 [configs/curve_lut_transfer.json](../../configs/curve_lut_transfer.json)。物理条件与 LUT 拟合/验证参数全部取自源 manifest 快照；当前 `configs/analytic.json` 和 `configs/calibration.json` 不覆盖源条件。预检会提示当前标定配置是否与快照不同。默认曲率半径为 `null/200/100/50 mm`，测试为 63 个锚点 × 2 个新深度。测试锚点不得与任何源标定锚点重合。

当前正式源结果对应 704 个原标定姿态、563/141 的固定训练/验证划分。默认工作量为 **2,620 帧**：4 个参考、504 个独立测试、2,112 个曲面标定姿态。正式扫描保留源分辨率、材料采样间隔、全部光源、精度与阴影。`--dry-run` 的未压缩存储估计包含图像、辐射量、几何和材料点诊断的保守开销；实际文件压缩率取决于内容。

## 数据与几何约定

- 仅支持 analytic、源平面已发布 LUT、有符号 `linear_rgb` 残差及 `srgb8_roundtrip` 观测。平面 `reference_arclength` 坐标和固定 +Z 几何法向必须匹配。CPU/CUDA 使用源配置选择，预检不静默换设备。
- 将旧 `center_xy_mm` / `direction_z` 仅在内存换成 `center_uv` / `direction_normal`，不改写源文件。各曲率只改变弯曲半径；灯带随边缘弯曲，保持原功率和局部朝内方向。
- 每个曲率单独渲染零压入参考。色差为同一材料点在当前图和参考图各自投影处的双线性采样之差。处理顺序为固定曝光/响应/tone map → sRGB8 → inverse_srgb → 材料点采样；不会反演 tone map。
- 真值为每个点的 `β=(p,q)=-Eᵀnd/(n0·nd)`，即 `b=Eβ`。接触区域使用真实球面交点，压入沿局部法向。采样保存逐点参考法向、切向基、世界法向和法向位移。
- 当前要求源 `contact_margin_mm=0`、`minimum_reference_dot=0.1`。不支持的源条件在渲染前报错，不修改采样规则。原标定允许边界相切；独立测试仍严格内接；越界姿态一律报错。不能为曲面重新缩小原姿态网格。
- 材料掩膜位：1 接触、2 参考有效、4 当前双线性邻域有效、8 相机前方、16 第一可见交点、32 正面、64 法向条件有效。`sample_mask` 保存最终有效集；非有限观测同样排除。图像饱和、遮挡等不伪装成零残差。

## 评价标准

主要指标为**坡度向量 RMSE**，在同一姿态内按参考曲面面积积分，然后对完整姿态等权平均平方误差，最后开平方。它比较整个 `(p,q)` 向量；不是 p/q 两项分别取均值，也不是各姿态 RMSE 的平均。

主图使用跨所有曲率的共同材料点接触域，且该域与 LUT 无关。完整有效接触域另行统计；报告共同面积覆盖率、法向/坡度范围和各掩膜计数。空姿态使主汇总标为 `incomplete`，不删除失败姿态再给出成功平均。

辅助指标包括局部法向角 RMSE/P95、坡度误差 P95、最差姿态、有符号 p/q 偏差、RGB 越界率及原始节点支持率。越界颜色仍按现有 LUT 截断查表并计入准确性误差；不按置信度过滤、不平滑、不把未接触背景放进主指标。

同一锚点的全部深度作为一个 bootstrap 单位，默认 2,000 次、seed=42。报告曲面直接迁移相对平面基线的 RMSE 增量，以及同曲率重标定的改善量和 95% 区间。展示容差 `[0.025,0.05,0.1,0.2]` 不代表工程验收阈值。`acceptance_slope_rmse` / `maximum_transfer_increase` 默认为 null，报告 `not_specified`；未指定时不宣布迁移合格。

LUT 变化通过**相同物理 RGB 查询的函数差异**比较。查询集来自平面与曲面测试样本，两个条件各占 1/2，内部姿态等权、面积加权。分别输出共同色盒和严格原始节点支持内的 RMS 变化及保留权重。严格支持要求所有正插值权重角点在两张 LUT 中均为 `original_valid`。不能按 LUT 数组下标相减；函数差异不是准确率。

重标定复用 `utils.calibration_pipeline.fit_and_validate`，固定源姿态划分和阈值。缺少任何必要姿态或原验证失败，均不发布 `normal_lut.npz`。源配置允许诊断可视化时保存 `normal_lut_diagnostic.npz`，报告与图中标记 failed/diagnostic；否则该 LUT 的比较单元显示不可用。不得将诊断结果解释为通过验证的 LUT。

## 可选迁移候选

`adaptation.enabled=true` 时额外评估：

1. 每材料点参考亮度比补偿：仅使用平面/曲面未压入参考；固定 `reference_floor` 剔除暗通道点，并在完全相同的剩余域上比较原 LUT。单独报告损失的覆盖率。
2. 少量完整姿态拟合 2×2 全局坡度校正 B。
3. 相同预算拟合 3×3 颜色校正 A，再交给原 LUT。

预算默认为 16/64/256，采用按深度分层的固定随机顺序，预算递增时嵌套使用完整姿态。全部标注预算均列出 pose ID；没有额外调用 141 个验证姿态选择超参数。正则系数、暗通道下限、优化迭代上限预先固定；独立测试集不参与拟合。颜色优化未收敛会明确记录，不能当成成功校正。报告不在测试集上挑选“最佳”候选后声称无偏性能。

局部响应矩阵 J、位置相关校正及未参与拟合曲率上的泛化仍属计划中的后续机理探索。本实现不把零残差节点强制为零、几何旋转协变或单一曲率校正成功解释为光学不变量。

## 输出与续跑

实验报告与派生样本写入：

```text
<source>/evaluation/curve_lut_transfer/<run>/
  manifest.json                 # 源哈希、实现签名、完整配置/姿态、逐项检查点
  config_snapshot.json / split.json
  samples/<condition>/reference.npz
  samples/<condition>/{test,training}/<pose>.npz
  calibration/<condition>/      # 共享拟合结果、样本、配置、验证和发布状态
  metrics.json / per_pose.csv / cross_curvature.csv
  mapping_difference_<condition>.npz
  adaptation.json               # 仅启用可选候选时
  figures/                      # PNG 和对应完整数值 NPZ
```

原始平面帧保存到 `results_render/analytic/curve_lut_transfer/<run>/flat/`；曲面帧保存到 `results_curve_render/analytic/curve_lut_transfer/<run>/<condition>/`。每帧有实际用于观测的 `image.png`、保存 sensor/radiance/图像几何真值的 `render.npz`、有效配置与执行信息 `metadata.json`。参考和每个姿态的所有产物完成并校验哈希后才记为完成。恢复时核对源 LUT、样本、配置、实现及已完成文件；参考图损坏或配置变化都拒绝续跑。未完成姿态可重新计算，已完成姿态不会重复渲染。

主要图为 `slope_rmse.png`、`position_heatmaps.png`（共享色标）和 `cross_curvature.png`；另输出固定左/中/右及最差姿态的向量/误差图、物理 RGB 切片。箭头密度仅影响显示，指标与 NPZ 使用完整数据。历史结果和源 LUT 从不搬移或覆盖。

## 验证边界

自动化测试位于 `tests/test_curve_calibration.py` 和 `tests/test_curve_lut_transfer.py`，覆盖局部公式/数值导数、刚体旋转、平面极限、原采样流程回放、CPU/CUDA、相切/越界、加权指标、不同色盒、固定划分、完整小样例和续跑。实际检查见 [验证记录](../../docs/curve_lut_transfer_validation.md)。小样例使用独立固定 fixture；其中源 LUT 是测试构造，不用于得出科学实验结论。完整 2,620 帧扫描需执行正式命令；代码通过测试不表示迁移成功或曲面标定已通过。
