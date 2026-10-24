# NormDiff 方法说明：颜色噪声下的法向混淆度

来源：Agarwal、Mirzaee、Sun、Yuan，*A Modularized Design Approach for GelSight Family of Vision-based Tactile Sensors*，2025。依据用户提供的 arXiv v1 PDF，重点为 **第 5.2 节、图 7，PDF 第 7 页**；相关背景见第 5.1 节，局限见第 8.2 节。

论文链接：[arXiv:2504.14739v1](https://arxiv.org/abs/2504.14739v1)。

> 给另一个 agent 的核心说明：NormDiff 是用于评价和优化触觉传感器光学设计的指标。对于一个 RGB 观测，先定义它在颜色噪声下的可混淆区域，再查找落入该区域的标定样本，统计这些样本对应的法向角度范围。对这些范围加权并平均，最后取负号。**法向混淆越小，NormDiff 越高，通常越接近 0。**它不要求 RGB 与法向线性，也不是预测法向与真值之间的平均角度误差。

**证据边界：**原文以文字给出七步算法，没有给出包含全部参数的闭式公式。以下公式是对该算法的数学整理；原文未明确的参数保持为符号。文末另列可执行实现的约定，不将其归为作者原始设定。

## 1. 目的与直观含义

在基于颜色光度立体法的触觉传感器中，希望不同表面法向产生足够不同的 RGB，以便由 RGB 反演法向。

NormDiff 关注的问题是：

> 当 RGB 发生给定幅度的测量扰动时，哪些法向仍可能解释这个观测？这些法向的范围有多大？

它同时能反映两类困难：

- **局部响应弱：**法向变化较大，RGB 却只变化很小，因此噪声会导致较大的法向不确定性。
- **不同法向颜色重合或接近：**即使两个法向相距较远，也可能落入同一颜色噪声区域。

论文中的 RGB2Normal 指标评价颜色与法向的线性关系；NormDiff 放宽线性要求，直接评价噪声下的法向可区分程度。非线性映射也可能有较好的 NormDiff。

## 2. 输入与符号

先考虑一个压痕的评价数据集：

$$
\mathcal D=\{(\mathbf c_k,\mathbf a_k)\}_{k=1}^{N},
\qquad
\mathbf c_k=(r_k,g_k,b_k),
\qquad
\mathbf a_k=(\theta_k,\phi_k).
$$

| 符号 | 含义 |
|---|---|
| \(\mathbf c_k\) | 第 \(k\) 个标定样本的三维 RGB 观测 |
| \(\mathbf a_k=(\theta_k,\phi_k)\) | 该样本的表面法向球坐标，而非 RGB 预测出的法向 |
| \(N\) | 有效压痕样本数 |
| \(\mathbf a_i^*\) | 第 \(i\) 个待评价的目标法向 |
| \(M\) | 待评价目标法向的数量 |
| \(\mathcal R_i\) | 在法向空间中，目标法向附近的标定样本索引集合 |
| \(U_i=|\mathcal R_i|\) | 该目标法向使用的锚点样本数 |
| \(\mathcal B_\eta(\mathbf c_j)\) | 以锚点 RGB 为基准，由噪声模型确定的可混淆颜色区域 |
| \(w_\theta,w_\phi\) | 两个法向角度范围的非负权重 |

论文首先通过压入前后的表面法向变化，识别压痕中的像素，再记录每个像素的 \((r,g,b,\theta,\phi)\)。在仿真评价中，法向来自已知几何/法向数据，不需要先训练一个 RGB 到法向的预测器。

注意：第 5.2 节将颜色记为 RGB，未充分明确其预处理过程。**不要未经说明就将其替换为背景差分 RGB、色相或归一化色度。**特别是相对噪声模型作用于原始 RGB 和作用于差分 RGB，含义并不相同。

## 3. 按原文七步算法整理的函数形式

### 3.1 为目标法向选择锚点

对目标法向 \(\mathbf a_i^*\)，在法向空间中寻找附近的标定样本，得到 \(\mathcal R_i\)。这些样本的颜色用于评估该法向的恢复不确定性。

原文使用了最近元组集合的表述，但没有明确距离定义、邻居数量或采样规则，因此这里不固定为某个特定的 kNN 参数。

### 3.2 在 RGB 空间寻找可混淆样本

对于锚点 \(j\in\mathcal R_i\)，定义：

$$
\mathcal C_j
=\{k\in\{1,\ldots,N\}:\mathbf c_k\in\mathcal B_\eta(\mathbf c_j)\}.
$$

含义：第 \(k\) 个样本的颜色落在锚点的噪声区域内，因而被视为与锚点颜色可能混淆。

**这里的候选搜索必须覆盖所选评价数据集，不能只搜索法向附近的样本。**否则，法向相距很远但颜色相近的歧义会被预先排除。

论文假设 RGB 噪声幅度与像素响应成比例，实际实现采用了 RGB 值的 **30%** 作为噪声幅度。正文没有完整明确：它是否作为标准差、区间半宽，以及三个通道如何共同界定区域。因此，忠实的函数形式应保留 \(\mathcal B_\eta\)，而非直接声称作者用了某个确定的高斯球或长方体。

### 3.3 计算候选法向的两个角度范围

$$
\Delta\theta_j
=\max_{k\in\mathcal C_j}\theta_k
-\min_{k\in\mathcal C_j}\theta_k,
$$

$$
\Delta\phi_j
=\max_{k\in\mathcal C_j}\phi_k
-\min_{k\in\mathcal C_j}\phi_k.
$$

按原文描述，锚点的混淆度是两个范围的加权和：

$$
H_j=w_\theta\Delta\theta_j+w_\phi\Delta\phi_j,
\qquad w_\theta,w_\phi\geq0.
$$

正文未给出权重数值，也没有说明它们是否经过归一化。这里不能默认其为某一组已知常数。

### 3.4 对锚点和目标法向依次平均，再取负号

目标法向 \(\mathbf a_i^*\) 的平均混淆度：

$$
\overline H_i=\frac{1}{U_i}\sum_{j\in\mathcal R_i}H_j.
$$

一个压痕的 NormDiff：

$$
\boxed{
\operatorname{NormDiff}(\mathcal D)
=-\frac{1}{M}\sum_{i=1}^{M}\frac{1}{U_i}
\sum_{j\in\mathcal R_i}
\left[
w_\theta\left(\max_{k\in\mathcal C_j}\theta_k-\min_{k\in\mathcal C_j}\theta_k\right)
+w_\phi\left(\max_{k\in\mathcal C_j}\phi_k-\min_{k\in\mathcal C_j}\phi_k\right)
\right].
}
$$

一句话表达就是：

$$
\boxed{\text{NormDiff}=-\text{平均的、颜色噪声范围内的法向角度混淆范围}}
$$

这一层次化平均对应原文第 5-7 步。**不能无条件改成对所有像素直接求平均**：当目标法向采样密度、每个目标的锚点数量或重复选取情况不同时，两者权重不同。

如果专门约定“每个数据样本各作为一次目标且只选自己为锚点”，才可简化为：

$$
\operatorname{NormDiff}_{\mathrm{sample}}
=-\frac1N\sum_{j=1}^{N}H_j.
$$

这是一种明确的样本平均实现，不代表原文已经指定了这种采样方式。

## 4. 多位置评价与设计优化

原文以每个压痕计算混淆度。若在 \(L\) 个接触位置分别评价，并采用等权汇总，可写为：

$$
S(\mathbf d)
=\frac1L\sum_{\ell=1}^{L}
\operatorname{NormDiff}(\mathcal D_\ell(\mathbf d)),
$$

其中 \(\mathbf d\) 是待优化的光学设计参数，例如灯光配置、光学件几何或膜的反射参数。设计优化目标为：

$$
\mathbf d^*=\arg\max_{\mathbf d}S(\mathbf d).
$$

跨位置等权平均是便于实施的整理；第 5.2 节没有完整指定跨位置聚合的全部细节。不要将“分别计算各位置后平均”与“先将所有位置的样本混成一个 LUT 再评分”混为一谈，两者可能评价不同的空间混淆效应。

在上述非负权重定义下，分数不大于 0，越高越好。若优化器采用最小化接口，可以最小化 \(-S\)，即平均混淆度。

## 5. 伪代码

以下伪代码保留原文未明确的近邻和噪声区域参数：

```text
inputs:
    D = [(RGB_k, theta_k, phi_k)]
    query_normals = [a_i*]
    normal_neighbor_policy
    color_noise_region
    w_theta, w_phi

query_confusions = []

for a_i* in query_normals:
    R_i = select_anchors_in_normal_space(D, a_i*, normal_neighbor_policy)
    anchor_confusions = []

    for j in R_i:
        B_j = color_noise_region(RGB_j)
        C_j = all sample indices k in D whose RGB_k belongs to B_j

        theta_range = max(theta_k for k in C_j) - min(theta_k for k in C_j)
        phi_range   = max(phi_k   for k in C_j) - min(phi_k   for k in C_j)

        H_j = w_theta * theta_range + w_phi * phi_range
        anchor_confusions.append(H_j)

    query_confusions.append(mean(anchor_confusions))

return -mean(query_confusions)
```

正常情况下，颜色区域应包含锚点本身，因而候选集不为空。若数据缺失或实现导致集合为空，应报告无效评价，不能将其默认为“零混淆”。

## 6. 一种便于编码的颜色区域约定

**本节是实现示例，不是论文明确给出的精确设定。**

若使用非负 RGB，可以把“30% 相对噪声”具体化为逐通道区间：

$$
\mathcal B_\eta^{\mathrm{box}}(\mathbf c_j)
=\left\{\mathbf c:
|c_t-c_{j,t}|\leq\eta|c_{j,t}|,
\quad t\in\{R,G,B\}\right\},
\qquad \eta=0.3.
$$

这相当于 RGB 空间中的一个长方体，可以直接实现候选样本筛选。但有以下约定需要记录：

- 这里把 30% 解释为**区间半宽**，不能同时称其为论文给定的标准差。
- 暗通道的区间可能过窄；若加入加性噪声下限，应明确这是噪声模型扩展。
- 如果使用差分 RGB，相对噪声围绕零值的行为会改变，应重新定义并验证噪声区域。
- 可以选择高斯置信区域或实测噪声模型替代长方体，但需要报告新的定义。

角度权重也需要作为配置显式给出。例如取 \(w_\theta=w_\phi=1\) 是一种实现约定，不是已核实的作者设定。

## 7. 结果如何解释

假设一个颜色区域包含三个候选法向，角度分别为：

```text
(theta, phi) = (12°, 40°), (17°, 43°), (58°, 41°)
```

则 \(\Delta\theta=46^\circ\)，\(\Delta\phi=3^\circ\)。在示例权重均为 1 时，该锚点混淆度为 49，负贡献为 -49。

这组数字仅用于说明计算，不是论文实验数据。它表达的是“相近颜色可能对应跨度很大的法向参数”，**不能解释为法向预测误差为 49°**。

还需注意：

- 原文使用的是 \(\theta\)、\(\phi\) 的坐标范围，不是候选单位法向之间的球面夹角。
- 方位角存在周期性。例如 359° 与 1° 很接近，但直接取最大值减最小值会得到 358°。原文未说明如何处理这一问题。
- 角度单位必须一致；改用弧度、修改权重或归一化方式会改变分数尺度。不同设置下的绝对分数不能直接比较。
- 该指标在有限样本上评价混淆。候选集中只有一个样本时范围为零，但这可能只是采样稀疏，不能证明连续法向域中不存在多义性。
- 指标不是神经网络训练所需的法向监督损失，也不是高度重建误差；它用于评价光学观测本身支持法向区分的程度。

## 8. 交给实现 agent 时必须明确的配置

| 配置 | 原文明确程度 | 实现时需要做什么 |
|---|---|---|
| 颜色输入 | 记为 RGB，预处理未充分展开 | 明确原始/差分、线性/伽马、数值范围 |
| 法向数据 | 记录 \((\theta,\phi)\) | 明确坐标轴、法向朝向、角度单位和周期边界 |
| 目标法向集合 | 需要遍历多个目标法向 | 明确目标域、采样密度和权重 |
| 法向近邻集合 | 有近邻选择步骤 | 明确距离、邻居数和覆盖不足时的处理 |
| RGB 噪声 | 与响应成比例；实现采用 30% | 明确噪声区域形状及 30% 的统计含义 |
| 混淆范围 | 两个角度的最大值减最小值 | 忠实实现时保留坐标范围；改球面距离须标记为扩展 |
| 角度权重 | 仅说明加权和 | 显式提供 \(w_\theta,w_\phi\) |
| 聚合方式 | 锚点平均、目标法向平均、最终取负 | 保留层级；明确多位置是否混合、如何加权 |

若目的是分析矩形传感器某条边的退化，可在总体分数之外报告各位置、各法向的混淆度分布。按固定位置或局部区域限定候选搜索，可以分析位置已知时的混淆；跨位置合并则可能同时反映全局 LUT 的空间不一致。**这是针对具体研究问题的评价域选择，应单独说明，而不是默认原文已区分这两类歧义。**

此外，第 8.2 节明确指出框架尚未精确覆盖部分阴影等传感器伪影。因此，较高 NormDiff 不能替代对实际阴影、饱和、未采样法向和制造误差的验证。

## 9. 本项目的可配置扰动量

NormDiff 位于 `experiment/norm_diff/`，入口为 `python -m experiment.norm_diff --lut-results <标定目录>`。评价参数从独立的 `configs/norm_diff.json` 读取，不再读取仿真配置的 `calibration.norm_diff`：

```json
{
  "norm_diff": {
    "noise_fraction": 0.03,
    "noise_floor": 0.0,
    "theta_weight": 1.0,
    "phi_weight": 1.0,
    "theta_bins": 12,
    "phi_bins": 24,
    "anchors_per_target": 1,
    "theta_ceiling_degrees": null
  }
}
```

`noise_fraction` 是逐通道的相对**半宽**，取值范围为 `[0, 1]`。例如 `0.03` 表示 ±3%，颜色为 0.5 时的区间为 `[0.485, 0.515]`；总区间宽度为 6%。`noise_floor=0` 时，半宽为 `noise_fraction * abs(当前通道值)`；非零下限时，半宽为 `noise_fraction * max(abs(当前通道值), noise_floor)`。3% 是本项目可修改的起始值，并非相机实测值或论文规定值。

参数优先级为命令行覆盖 > `configs/norm_diff.json` 的 `norm_diff` > 内置默认值。`--exp-config` 选择实验配置；`--lut-results` 指定已通过验证的标定结果，默认从其 manifest 读取当时的物理配置。可选的 `--config` 仅用于匹配验证和必要的参考渲染，不提供评价参数。例如：

```powershell
.\.venv\Scripts\python.exe -m experiment.norm_diff --lut-results results_lut/xense --out results_lut/xense/evaluation/norm_diff_new.json
.\.venv\Scripts\python.exe -m experiment.norm_diff --lut-results results_lut/xense --exp-config configs/norm_diff.json --noise-fraction 0.01 --out results_lut/xense/evaluation/norm_diff_noise_1pct.json
```

添加 `--validate-lut-noise` 可使用同一个扰动量验证 `normal_lut_train.npz`。该验证仅使用 LUT 记录的留出姿态，在观测 RGB 上逐通道施加区间内独立均匀扰动，参考 RGB 保持固定；重复三次，报告原始 RMS、扰动后 RMS 和每轮结果。NormDiff 的颜色盒只定义候选区间，随机扰动仅用于这个可选的 LUT 验证。两者均在本项目保存的色调映射后线性 RGB 域中计算。

独立评价参数不参与渲染数据的签名。修改它后可复用已有样本；如果光照、相机或灯带几何模型等发生变化，仍然需要匹配对应的样本。结果 JSON 会记录实际采用的评价参数和实验配置路径；不同扰动量下的分数应分别保存、比较。默认输出已改为 `experiment/norm_diff/results/norm_diff.json`，不会覆盖标定目录中的历史报告。

长度扫描入口为 `python -m experiment.length_normdiff`，只读取 `exp_config.simulation` 的独立仿真快照；中心光学诊断由同一入口的 `--diagnostics-only` 启动。`theta_ceiling_degrees=null` 时，单次样本评价沿用样本实际最大角，长度扫描沿用统一的球冠理论上限；指定同一个数值即可严格对齐两种计算。详见 [实验说明](../experiment/README.md)。
