# 灯带逐点 RGB 功率优化

目标是在保存的 analytic 几何、相机、曝光、采样、LUT 拟合和逐姿态验证条件下，仅修改 `lights[].point_power_rgb_w`，与指定 xense 结果的完整 quality 曲线对比。当前实现用于可复现的有限候选搜索，不宣称获得全局最优，也不改变标定发布标准。

当前用户确认的主要验收目标是 **reference A(τ=0.1)≤0.3024**，其余 tau 的 A、节点误差和实际运行误差曲线如实报告，不要求全部优于 xense。允许改变总功率。原标定的逐姿态验证标准不变；样本数变化必须单独核对，不能靠丢弃饱和样本达标。

## 运行

```powershell
.\.venv\Scripts\python.exe -m experiment.light_distribution --lut-results results_lut/analytic/discrete_625_complement3_20261023 --target-results results_lut/xense --dry-run
.\.venv\Scripts\python.exe -m experiment.light_distribution --lut-results results_lut/analytic/discrete_625_complement3_20261023 --target-results results_lut/xense
```

支持直接执行 `python experiment/light_distribution/__main__.py --help`。配置位于 `configs/light_distribution.json`，共享加载与校验位于 `experiment/config.py`。`--exp-config` 可显式选择其他实验配置，CLI 路径相对当前工作目录。`--out` 必须位于源结果的 `evaluation/` 下；默认是 `evaluation/light_distribution/`。

## 候选定义

每个候选包含唯一 `name`、`family`、`strength`、`short_power_w`、`long_power_w`。功率单位为 W，分别表示**每条**短边和长边灯带的 RGB 合计功率，而非四条边合计。点距固定使用源快照中的 6.25 mm，不修改点数、几何或发光方向。

- `complement`：左蓝、右黄、上红、下绿，每边内部均分。`strength` 在本族中不参与功率计算，建议置 0。
- `gradient`：左蓝、右绿；上下长边的 G/B 比例沿世界 X 平滑变化，上边额外保留一半红光功率。`strength` 在 `[0,1]` 内，0 表示沿边不变，1 表示完整线性梯度。颜色比例均非负，每点 RGB 合计功率相同。
- `explicit`：填写 `name`、`family: "explicit"` 和 `point_power_rgb_w`，后者为包含 `left/right/top/bottom` 四个数组的对象，每行是一个发光点的 `[R,G,B]` 绝对功率。点数必须匹配源快照，不归一化、不限制为某种颜色族。可选 `description` 记录候选构造方法。其他两族也支持该说明字段。

第三轮配置 `configs/light_distribution_refine.json` 使用显式数组，固定每边 0.4 W，测试更陡的 G/B 渐变、上边红光比例、每点的共同白光分量和长边端部功率权重。颜色变换只用于生成候选功率数组，渲染器、相机响应和 LUT 算法均保持原样：

```powershell
.\.venv\Scripts\python.exe -m experiment.light_distribution --lut-results results_lut/analytic/discrete_625_complement3_20261023 --target-results results_lut/xense --exp-config configs/light_distribution_refine.json --out results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_refine
```

实验版本 v2 还核对双方 quality 指标版本、容差语义、权重、划分与 tau 序列，并把源和目标 manifest、quality 报告及检验 manifest 的 SHA256 纳入续跑签名；每个候选结束后再次检查实现文件未变化。

`configs/light_distribution_power.json` 延续显式数组格式，组合顶部红光比例和长边端部功率权重，并比较 0.8–3.2 W 总功率及短边减半。运行时将上述命令中的 `--exp-config` 换为该文件，`--out` 换为新的 `evaluation/light_distribution_power`。所有候选仍执行原完整扫描和检验。

后续两组也使用相同入口、独立配置与输出目录：

- `configs/light_distribution_spectral.json` → `evaluation/light_distribution_spectral`：短边向顶部增加红光，以及下边功率减半或关闭。
- `configs/light_distribution_profiles.json` → `evaluation/light_distribution_profiles`：顶部红光占比随沿边位置呈二次变化，以及 G/B 混合比例随有符号平方、平方根变化。各候选的完整构造公式保存在 `description` 中，渲染使用保存的绝对 RGB 功率数组。

## 完整条件与结果

每个候选调用原 `calibrate_lut.py` 和 `experiment.lut_quality`，重新渲染全部姿态和参考图、重新拟合、按原完整姿态划分留出验证。quality 使用该候选完整样本和训练 LUT，不重渲染、不下采样。原附加标定可视化也照常运行；某候选过慢或异常时，终止只保留诊断，不把它当成已完成。

实际物理与标定条件来自 `--lut-results/manifest.json`，quality 设置来自其保存的 evaluation manifest，不读取后续修改的默认设置来替代它们。生成的完整配置、标定和 quality 快照均放在 `configs/light_distribution_runs/<signature>/`。搜索期间不修改 `configs/analytic.json`；选中候选后需单独核对再采用。

输出结构为：

```text
<source>/evaluation/light_distribution/
  manifest.json
  logs/<candidate>_calibration.log
  logs/<candidate>_quality.log
  <candidate>/manifest.json
  <candidate>/normal_samples.npz
  <candidate>/normal_lut_train.npz
  <candidate>/verification/validation.json
  <candidate>/evaluation/lut_quality/{quality.json,curves.csv,curves.png,...}
```

每个候选完成后原子记录曲线、样本数、原标定状态和配置位置。`--resume` 仅复用签名一致的已完成候选；签名包含完整源快照、候选参数、目标曲线、quality 设置和实现 SHA256。部分完成的目录明确拒绝覆盖。签名不匹配或运行过程中代码发生变化时拒绝混用结果。失败标定只要完整到达原流程终点，仍可计算 quality，原失败状态保留。

饱和导致的有效样本数变化单独报告；不能依靠删除困难样本宣称达到目标。跨后端结果仍有表面尺寸、观测模型和采样覆盖差异，本实验只固定 analytic 自身条件，不证明两后端差异完全来自光场。目标需综合核对原 A 曲线和完整姿态留出误差，不能用训练曲线替代正式标定通过。

## 2026-10-23 第二轮结果

完整记录：[light_distribution_v2/manifest.json](../../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_v2/manifest.json)。7 个候选全部完成完整扫描、标定和 quality 检验。源配置固定为第一轮互补色结果，之后所有候选只改变逐点 RGB 功率。

| 候选 | A(0.1) ↓ | 标定状态 | 全部样本数 |
| --- | ---: | --- | ---: |
| gradient_full | **0.344724** | passed | 2,279,043 |
| gradient_half | 0.367332 | failed | 2,279,043 |
| gradient_full_side3 | 0.368255 | failed | 2,278,524 |
| complement_side10 | 0.466608 | failed | 2,277,624 |
| complement_side1 | 0.425933 | failed | 2,279,043 |
| complement_side3_half | 0.427471 | failed | 2,279,043 |
| complement_side3_double | 0.448110 | failed | 2,278,765 |

第二轮结束时采用了 `gradient_full`，其配置及签名保存在该运行中；当前采用的后续方案见下文。141 个留出姿态全部通过，整体角度 RMS 9.801917°、最大逐姿态 RMS 19.114227°。其 A(0.1) 仍高于 xense 的 0.302409，目标未完成；留出实际 LUT 失败率为 0.472099，xense 对应 0.494477。完整 [曲线图](../../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_v2/progress_comparison.png) 与 [曲线数值](../../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_v2/progress_comparison.csv) 已保存。

搜索反复查看了同一留出集，不能称为新的独立泛化测试。后续应围绕平滑梯度微调颜色比例和功率分配，继续使用原完整扫描与发布标准。初次启动的 `light_distribution/` 因配置快照分离错误在扫描前退出，失败 manifest 和日志保留；修正后使用新的 `light_distribution_v2/`，未覆盖该失败记录。

## 2026-10-23 第三轮结果

完整记录：[light_distribution_refine/manifest.json](../../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_refine/manifest.json)。全部 7 个候选均保留 2,279,043 个样本。

| 候选 | A(0.1) ↓ | 留出 runtime_error(0.1) ↓ | 标定状态 |
| --- | ---: | ---: | --- |
| sharp15 | 0.359163 | 0.480142 | passed |
| sharp20 | 0.379537 | 0.514855 | passed |
| red075 | 0.341029 | 0.472109 | passed |
| white020 | 0.373393 | 0.498670 | failed |
| white040 | 0.424044 | 0.561172 | failed |
| edge_gain2 | **0.339429** | **0.452148** | passed |
| sharp15_red075_white020 | 0.393394 | 0.525334 | failed |

增强颜色梯度与混入共同白光分量均未改善主目标；顶部红光占比提高至 75%、长边功率按 `1+2t²` 向端部增加分别有小幅改善，`t` 是沿边坐标除以半灯带长度。后续功率轮次组合这两个因素，继续完整扫描。

## 后续功率与颜色对照，以及停止时的最终选择

| 候选 | A(0.1) ↓ | 标定状态 | 全部样本数 |
| --- | ---: | --- | ---: |
| red075_edge2 | 0.333437 | passed | 2,279,043 |
| red075_edge2_scale05 | 0.349016 | passed | 2,279,043 |
| red075_edge2_scale15 | 0.352851 | passed | 2,278,999 |
| red075_edge2_scale2 | 0.371135 | failed | 2,278,758 |
| red075_edge2_short05 | 0.343809 | passed | 2,279,043 |
| red100_edge2 | 0.362933 | passed | 2,279,043 |
| red075_edge6 | 0.358548 | passed | 2,279,043 |
| short_red030 | 0.346359 | passed | 2,279,043 |
| short_red060 | 0.376520 | passed | 2,279,043 |
| **bottom_half（最终采用）** | **0.331561** | **passed** | **2,279,043** |

用户于 2026-10-23 要求停止继续优化。正在执行的 `spectral/three_color` 和 `profiles/red_center05_edge1` 已终止，manifest 标明中断原因，未纳入最终选择；profiles 其余候选未运行。停止后仅整理已有结果、绘图、核对并采用已完成候选，没有继续搜索。共保存 29 组候选 quality 曲线，其中首轮低功率候选的标定附加可视化曾中断，原失败验证状态保留。所有记录见[全部候选 CSV](../../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_spectral/final_summary/all_trials.csv)。

最终结果位于 [bottom_half](../../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_spectral/bottom_half/manifest.json)，当前 `configs/analytic.json` 与保存快照、标定配置及签名完全一致：

- 左蓝、右绿，每边 0.4 W，4 点均分；上边 0.4 W、下边 0.2 W，各 9 点，合计 1.4 W。
- 长边 `t=x/28`、`g=(1+t)/2`、`w=1+2t²`，每点总功率 `P_side*w/sum(w)`。
- 上边 RGB 比例 `[0.75,0.25g,0.25(1-g)]`；下边 `[0,g,1-g]`。6.25 mm 点距、灯带几何和全部非功率配置不变。
- A(0.1)=0.331561，相比原始 0.460871 下降 28.06%，仍未达到 0.3024。训练/留出 runtime_error(0.1) 分别 0.408611/0.443388。
- 704 个姿态、2,279,043 个样本，非颜色数组与固定基线逐项完全相等；原 563/141 姿态划分及验证阈值不变。141 个留出姿态全部通过，整体 RMS 9.092938°，最差逐姿态 RMS 17.463430°，正式 LUT 已按原流程保存。

最终[曲线图](../../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_spectral/final_summary/curves_comparison.png)、[逐点灯带图](../../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_spectral/final_summary/light_distribution.png)、[完整曲线 CSV](../../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_spectral/final_summary/curves_comparison.csv)、[逐点功率 CSV](../../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_spectral/final_summary/point_powers.csv)与[核对记录](../../results_lut/analytic/discrete_625_complement3_20261023/evaluation/light_distribution_spectral/final_summary/report.json)均为新文件，未改写各轮原始曲线。
