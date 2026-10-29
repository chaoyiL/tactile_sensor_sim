# 弯曲后 LUT 迁移实验结果

本文记录 **2026-10-24 已完成的正式实验**，包括曲面弯曲后的色差变化、LUT 误差与重标定对照。数值取自保存的实验产物；第 7 节的配对漂移是交接时从完整测试样本补算的统计，没有重新渲染、重新拟合或修改原报告。

核心结果：在本次光源、相机和采样条件下，直接用平面 LUT 查询曲面色差，坡度向量 RMSE 随弯曲增强而上升；R=200、100、50 mm 时相对平面基线分别增加 **7.18%、17.10%、40.31%**。按各曲率重标定能分别降低直接迁移误差 **14.39%、21.31%、22.35%**。因此，单纯把输出解释到局部切平面，不能保持这张平面 LUT 的准确性。本实验尚未确定可标定的光学不变量，也未验证少量标定的迁移方案。

## 1. 对应运行与阅读顺序

- 后端：`analytic`，GPU 渲染；几何 float64、辐射计算 float32。
- 源平面标定：`results_lut/analytic/`，已发布的 `normal_lut.npz`；训练/验证成功。
- 本次运行：`results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400/`，下文简称 **RUN**。保存状态为 `completed`，三个曲面标定均为 `passed`。
- 实验方法与运行说明：[实验 README](../experiment/curve_lut_transfer/README.md)；几何定义见本文第 3 节；实现入口：[runner.py](../experiment/curve_lut_transfer/runner.py)。
- 事实依据：[metrics.json](../results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400/metrics.json)、[manifest.json](../results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400/manifest.json)、[config_snapshot.json](../results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400/config_snapshot.json)、[split.json](../results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400/split.json)。

建议先看第 5 节误差表和三张主图，再看第 7 节配对统计、第 8–9 节解析建模线索，最后查第 10 节数据字典。以 **RUN/config_snapshot.json 的 simulation 和 manifest.identity.config** 为物理条件依据，不用当前可编辑配置代替历史条件。

## 2. 实验问题与控制变量

记平面 LUT 为 $F_0:\Delta c\mapsto\beta$，曲率 $\kappa=1/R$ 对应的重标定 LUT 为 $F_\kappa$。实验做两件事：

1. 冻结 $F_0$，对各曲率新渲染的色差查表，比较 $F_0(\Delta c_\kappa)$ 与该点的局部切平面坡度真值 $\beta_\kappa$。平面也重新生成独立测试数据，提供基线。
2. 使用源标定的同一批完整压入姿态、同一训练/验证划分、同一拟合方法和阈值，各自得到 $F_\kappa$。四张 LUT 都在同一套独立测试姿态上交叉评价，并在相同物理 RGB 查询处比较函数变化。

“保持同一光照条件”在这里指：**同一组 LED、每颗 LED 的 RGB 总功率不变、相机和曝光不变，灯带按照指定几何随曲面弯曲**。它不代表各材料点的入射光方向、距离、余弦项、遮挡和相机观测保持不变。没有调光来抵消弯曲造成的局部照明变化。

| 项目 | 本次实际条件 |
|---|---|
| 未压入表面 | 55.9 × 21.8 mm；沿 X 等曲率弯曲，Y 保持直线 |
| 曲率半径 | 平面、200、100、50 mm |
| 弯曲方向 | 中心固定，两侧向世界 −Z，即向相机方向弯曲 |
| 相机 | 640 × 480，位置 (0,0,−60) mm，朝向中心，所有条件固定 |
| 材料 | 朗伯反射，RGB 反射率 (0.8,0.8,0.8) |
| 探测球 | 半径 6 mm，沿压入锚点的参考曲面法向压入 |
| 光源 | `discrete_rgb_points_v1`，左右各 4 颗 LED、上下各 9 颗，共 26 颗；每颗宽度方向 2 个积分点，共 52 个光样本 |
| LED 排布 | 间距 6.25 mm；左右分别蓝/绿，每颗 0.1 W；上下为保存快照中的逐 LED RGB 配置 |
| 灯带 | 左右长 20 mm，上下长 56 mm，宽 2 mm，外侧 offset=0.5 mm，中心 z=−1.1 mm，发射余弦指数 1 |
| 相机观测 | RGB 响应矩阵为单位阵，曝光 6000，Reinhard tone map，sRGB8 编码，再 inverse_sRGB 解码取样 |
| 阴影 | 开启；使用实际曲面/压入几何求可见性 |
| 材料采样 | 参考弧长网格间隔 0.1 × 0.1 mm，219 × 560 个网格点 |
| LUT | 每轴 64 格的 RGB 三维表；查询使用三线性插值，色盒外截断查询但仍计入误差 |

表面左右边缘参考法向倾角和下沉量分别为：

| R (mm) | 边缘倾角 (°) | 边缘下沉量 (mm) |
|---|---:|---:|
| 200 | 8.007 | 1.950 |
| 100 | 16.014 | 3.881 |
| 50 | 32.028 | 7.611 |

这是表面边缘的几何量，不是实验测得的法向误差。

## 3. 几何、坡度与色差语义

### 3.1 归一化弧长坐标和局部坡度

$(u,v)\in[-1,1]^2$，中心为 (0,0)。令 $s=uW/2,t=vH/2$，单位 mm。它们是未压入参考曲面的弧长坐标，不是弯曲后的世界 X/Y，也不是图像归一化坐标。

对于 $\kappa>0$，未压入参考曲面及逐点正交基为

$$
X_{0,\kappa}(s,t)=\left(\frac{\sin\kappa s}{\kappa},t,\frac{\cos\kappa s-1}{\kappa}\right),\quad
n_{0,\kappa}=(\sin\kappa s,0,\cos\kappa s),
$$

$$
e_{1,\kappa}=(\cos\kappa s,0,-\sin\kappa s),\quad e_2=(0,1,0),\quad E_\kappa=[e_{1,\kappa},e_2].
$$

平面取连续极限 $\kappa=0$。本式中的 $\kappa$ 是正的弯曲强度；按此几何法向定义，X 主曲率的形算子符号为 $-\kappa$。

压入锚点 $(s_a,t_a)$、深度 $d$、球半径 $r$ 确定球心

$$
C=X_{0,\kappa}(s_a,t_a)+(r-d)n_{0,\kappa}(s_a,t_a).
$$

压入后的材料点 $X_{d,\kappa}=X_{0,\kappa}-\delta_\kappa n_{0,\kappa}$，$\delta_\kappa\ge0$ 由参考法向射线与球的真实交点确定，非接触处为 0。接触处的正几何法向为 $n_d=-(X_d-C)/r$。LUT 目标是

$$
\boxed{\beta_\kappa=(p,q)^T=-\frac{E_\kappa^Tn_d}{n_{0,\kappa}\cdot n_d},\qquad b_\kappa=E_\kappa\beta_\kappa.}
$$

这里用的是**每一个材料点**自己的参考切平面，而不是整个压痕共用锚点切平面。因为基底弯曲，不能直接把世界高度梯度当作 $\beta$。光照计算使用朝相机的光学法向 $n_o=-n_d$，不要与上述几何法向混用。

相同锚点、球半径、压入深度在不同曲率下，逐点 $\beta_\kappa$ 仍会变化；这点在第 7 节中实测量化。因此跨曲率配对不是“完全相同坡度下只改变光照”的反事实实验。

### 3.2 色差是同一材料点两次投影后的观测差

每个曲率单独渲染未压入参考图。设 $I_{\kappa,d}$ 是相机输出的 sRGB8 图，$D$ 是 inverse_sRGB，$B$ 是双线性取样，$\pi$ 是相机投影，则实际 LUT 输入为

$$
\boxed{\Delta c_\kappa(s,t)=B[D(I_{\kappa,d})](\pi(X_{d,\kappa}))-B[D(I_{\kappa,0})](\pi(X_{0,\kappa})).}
$$

这是有符号的三通道量。当前图与参考图分别投影；不是同一图像像素相减。先对像素做 inverse_sRGB，再双线性取样，不交换顺序。

虽然字段名为 `linear_rgb`，其含义是去除了 sRGB 传递函数的观测 RGB。**没有反演 Reinhard、曝光或量化**，所以不是物理辐亮度，也不是 0–255 色差。理论拟合不能跳过这一观测链。

## 4. 执行流程与统计口径

源标定共 704 个完整压入姿态，空间扫描间隔 2 mm、深度 0.5/1.0/1.5 mm；固定训练/验证划分为 563/141。曲面重标定严格沿用此划分，通过验证后用全部 704 个姿态拟合发布表。源平面最终表同样使用全部源姿态，因此原 141 个验证姿态不能再被当作最终表的独立测试集。

本实验另取 9 × 7 = 63 个新锚点，$u=-0.8,-0.6,\ldots,0.8$，$v=-0.6,-0.4,\ldots,0.6$，深度 0.75/1.25 mm；共 **126 个独立测试姿态/条件**。锚点与全部源标定锚点不重合，测试数据不参与拟合。

执行了 **2,620 帧**：4 张参考 + 4 × 126 张测试 + 3 × 704 张曲面标定。保留源相机分辨率、全部光源、阴影和材料采样密度。曲面标定有效训练样本数分别为 2,254,561、2,231,690、2,177,783；源平面样本数为 2,279,043。样本覆盖随曲率变化，不能把它忽略为完全相同的训练分布。

主要比较域：对每个测试姿态，取四个条件的**有效接触材料点交集**，该交集与 LUT 无关。按各自完整有效接触域计算的结果另存为 `full`。共同域平均覆盖各完整域的 95.37%、96.54%、97.70%、100%；本次交集由 R=50 的有效域限制。

有效性条件包括：真实接触、参考取样有效、当前双线性邻域有效、在相机前方、第一可见交点、正面、$n_0\cdot n_d>0.1$，且观测有限、图像未饱和。未接触背景不进入误差，不使用置信度过滤或平滑。

每个姿态内部按参考材料面积加权；所有完整姿态等权。若 $w_{ai}$ 是姿态 $a$ 内归一化面积权重，则

$$
\mathrm{RMSE}=\sqrt{\frac1{126}\sum_a\sum_iw_{ai}\|\widehat\beta_{ai}-\beta_{ai}\|_2^2}.
$$

坡度无量纲，RMSE 同样无量纲；不是高度误差 mm，不是百分比，也不是两个分量平均后相消的误差。辅助角误差来自局部法向 $(-p,-q,1)/\sqrt{1+p^2+q^2}$。不能用 $\arctan(\mathrm{RMSE})$ 代替逐点计算后的角 RMSE。

置信区间按 63 个锚点作 cluster bootstrap，同一锚点的两个深度一起重采样，2,000 次、seed=42。它描述当前采样锚点分布中的统计波动，不包含物理模型不准确、其他照明或真实传感器的系统误差。

## 5. 实验一：平面 LUT 直接迁移有多少误差

以下全部使用共同材料域。

| 测试表面 | 平面 LUT 坡度 RMSE | 相对平面增幅 | 角 RMSE (°) | 坡度误差 P95 | 角误差 P95 (°) |
|---|---:|---:|---:|---:|---:|
| 平面 | 0.139938 | — | 7.0590 | 0.290162 | 14.5759 |
| R=200 mm | 0.149990 | +7.18% | 7.3946 | 0.305318 | 14.9184 |
| R=100 mm | 0.163866 | +17.10% | 8.0298 | 0.338300 | 16.2486 |
| R=50 mm | 0.196348 | +40.31% | 9.5244 | 0.400710 | 18.9119 |

| R (mm) | RMSE 增量 | 增量的配对 95% CI |
|---|---:|---:|
| 200 | +0.010052 | [0.005973, 0.014418] |
| 100 | +0.023927 | [0.018861, 0.029386] |
| 50 | +0.056410 | [0.051500, 0.061577] |

三个区间均大于 0，在当前样本范围内直接迁移退化是稳定的。平面基线本身已经有误差，不能把 0.196348 全部说成弯曲新增误差；RMSE 增量也不是新增误差向量的 RMS，详见第 7 节。

平面 LUT 对四个条件的 RGB 色盒越界率均为 **0**。因此本次退化不能简单归因于颜色超出平面 LUT 的取值范围。其严格原始节点支持率仅约 23.29%–28.85%；色盒内含有补全节点，零越界不等于每次插值都有原始标定支持。

主图：

![坡度 RMSE 与曲率](../results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400/figures/slope_rmse.png)

![位置相关误差](../results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400/figures/position_heatmaps.png)

热图用共享色标，展示各锚点处整个压痕的误差；图上的一个锚点不是一个材料像素。边缘/角落在平面基线中也较难，不能把所有位置差异都解释为弯曲。固定中心锚点、两个深度合并后，平面 LUT 的整个压痕 RMSE 为 0.07640、0.12208、0.13920、0.16098；这也不是中心单点误差。

## 6. 实验二：重标定与 LUT 函数变化

### 6.1 在独立测试上的准确性

| 曲面 | 平面 LUT RMSE | 同曲率重标定 RMSE | 相对直接迁移降低 | 重标定角 RMSE (°) |
|---|---:|---:|---:|---:|
| R=200 | 0.149990 | 0.128401 | 14.39% | 6.3908 |
| R=100 | 0.163866 | 0.128940 | 21.31% | 6.3771 |
| R=50 | 0.196348 | 0.152471 | 22.35% | 7.3842 |

重标定的 RMSE 改善量及 95% CI 分别为 0.021590 [0.018809,0.024690]、0.034925 [0.031945,0.038133]、0.043877 [0.041105,0.046785]。三个曲面 LUT 均通过源流程验证；这不等于满足用户的迁移工程容差。本次 `acceptance_slope_rmse` 和 `maximum_transfer_increase` 都未指定，验收状态为 `not_specified`。

交叉评价矩阵如下，**行是 LUT 的标定表面，列是测试表面**，单元格为坡度 RMSE，越小越好。

| LUT \ 测试 | 平面 | R=200 | R=100 | R=50 |
|---|---:|---:|---:|---:|
| 平面 | **0.139938** | 0.149990 | 0.163866 | 0.196348 |
| R=200 | 0.197278 | **0.128401** | 0.141756 | 0.180458 |
| R=100 | 0.258442 | 0.172939 | **0.128940** | 0.165246 |
| R=50 | 0.323727 | 0.265731 | 0.209619 | **0.152471** |

![交叉曲率评价矩阵](../results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400/figures/cross_curvature.png)

每列均以同条件标定的 LUT 最好。跨曲率迁移并不对称。R=200/100 重标定在自己的曲面上比平面 LUT 在平面上的基线还低，但由于光场和训练/真值分布均变了，不能据此概括“弯曲提高传感器质量”。

### 6.2 LUT 本身如何比较

不能将两张 LUT 的相同数组下标直接相减：各 LUT 色盒可能不同。这里取平面和对应曲面测试颜色组成物理查询集合，两种条件各占一半权重，各自内部姿态等权、面积加权，然后在**相同 $c$** 处计算

$$
D_\kappa=\sqrt{\langle\|F_\kappa(c)-F_0(c)\|^2\rangle}.
$$

| R (mm) | 共同色盒内函数差 RMS | 保留权重 | 双方严格原始节点支持内 RMS | 保留权重 |
|---|---:|---:|---:|---:|
| 200 | 0.118299 | 99.64% | 0.070848 | 15.40% |
| 100 | 0.172966 | 97.86% | 0.085295 | 13.26% |
| 50 | 0.229330 | 91.64% | 0.141990 | 10.66% |

函数变化随弯曲增强而增大，但 **D 不是预测误差**，也不等于真实光学映射变化的精确值；它包含拟合、离散、补全和样本支持变化。严格支持仅覆盖较小权重，应连同覆盖率解读。`mapping_slices_*.png` 中灰色部分表示缺少相应支持，不是函数差为零。

## 7. 补充：配对色差漂移与误差向量漂移

为方便理论分析，交接时对全部 126 个测试姿态、每个姿态跨四条件的共同材料点作配对统计。每个曲面与平面共有 **413,172 对材料点**，没有抽样。记

$$
\delta c=\Delta c_\kappa-\Delta c_0,\quad
\delta\beta=\beta_\kappa-\beta_0,\quad
e_\kappa=F_0(\Delta c_\kappa)-\beta_\kappa,\quad\delta e=e_\kappa-e_0.
$$

表中各 RMS 都先在姿态内按参考面积加权，再在姿态间等权；三通道色差使用向量模，不除以通道数。

| R (mm) | 色差变化 RMS $\lVert \delta c\rVert$ | 参考颜色变化 RMS | 真值坡度变化 RMS $\lVert \delta\beta\rVert$ | 查表预测变化 RMS | 误差向量变化 RMS $\lVert \delta e\rVert$ |
|---|---:|---:|---:|---:|---:|
| 200 | 0.062341 | 0.090629 | 0.008868 | 0.093444 | 0.094160 |
| 100 | 0.107589 | 0.166946 | 0.017870 | 0.120013 | 0.122727 |
| 50 | 0.162679 | 0.286924 | 0.036302 | 0.148804 | 0.165853 |

有符号色差变化的全域通道均值 $(R,G,B)$ 分别为：

- R=200：(-0.008766, -0.035435, -0.035437)。
- R=100：(-0.016349, -0.060280, -0.060302)。
- R=50：(-0.028766, -0.087215, -0.087232)。

它们是配对色差的平均变化，不是原图整体变暗量，也不意味着所有位置/坡度的变化同号。参考和当前图的变化并未在相减后完全抵消。G/B 在这项聚合中近似对称、变化幅度大于 R，与本次特定光源布局有关；不能当作一般性光学定律。

**误差向量改变可以大于整体 RMSE 增量**。精确恒等式为

$$
\mathrm{MSE}_\kappa-\mathrm{MSE}_0
=2\langle e_0\cdot\delta e\rangle+\langle\|\delta e\|^2\rangle.
$$

| R (mm) | MSE 增量 | 交叉项 $2\langle e_0\cdot\delta e\rangle$ | 漂移平方项 $\langle\lVert \delta e\rVert^2\rangle$ |
|---|---:|---:|---:|
| 200 | +0.002914448 | −0.005951567 | +0.008866015 |
| 100 | +0.007269239 | −0.007792640 | +0.015061879 |
| 50 | +0.018969757 | −0.008537578 | +0.027507335 |

交叉项为负，表明弯曲造成的误差变化与一部分原有误差反向，聚合时存在抵消。不能简单把基线和“弯曲误差”作平方相加，也不能把 RMSE 增量称为新增误差场的 RMS。这些是保存数据上的代数分解，并不是光学、投影、量化等各机制的因果分解。

原始补充数据：[paired_drift_summary.json](../results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400/handoff/20261024T045312Z/analysis/paired_drift_summary.json)、[paired_drift_per_pose.csv](../results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400/handoff/20261024T045312Z/analysis/paired_drift_per_pose.csv)。ZIP 根目录 `analysis/` 提供同内容快捷副本；生成脚本保存在交接目录 `export_analysis.py`。

## 8. 曲率如何进入颜色：与实现一致的解析起点

以下是从现有模型提取的观测方程与局部展开，供后续推导。它们说明需要分析哪些项，**不是已经验证完成的误差来源占比模型**。

### 8.1 光场与相机

在相机像素对应的首个表面交点 $x$，记第 j 个光源积分点位置为 $\ell_j$、发光面朝向为 $m_j$、RGB 功率为 $P_j$、余弦指数为 $a_j$。令

$$
r_j=\ell_j-x,\quad d_j=\|r_j\|\ \text{(mm)},\quad\omega_j=r_j/d_j,
\quad\mu_j=[n_o\cdot\omega_j]_+,\quad\nu_j=[-m_j\cdot\omega_j]_+.
$$

$V_j$ 为实际阴影/遮挡可见性。按实现的发射和朗伯模型定义

$$
K_j=\frac{a_j+1}{2\pi}\,\mathbf1_{\nu_j>0}\,\nu_j^{a_j}\,\mu_j\,V_j\,d_j^{-2},
\qquad L=\frac{10^6}{\pi}\operatorname{diag}(\rho)\sum_jP_jK_j.
$$

$L$ 单位为 W/m²/sr，$10^6$ 将 mm 距离产生的面积单位转成 m²。相机模型继续计算

$$
S=g(x)L,\quad g(x)=\left(\frac{2h}{f}\right)^2\cos^4\theta,\qquad A=\eta M S,
$$

其中针孔半宽 $h=0.004$ mm、焦距 $f=3$ mm、曝光 $\eta=6000$、响应矩阵 $M=I$。最后逐通道经过

$$
I_8=\operatorname{round}\left(255\,\operatorname{sRGB}\left(\operatorname{clip}_{[0,1]}T([A]_+)\right)\right),\qquad T(z)=\frac{z}{1+z}.
$$

与第 3.2 节的 $D,B,\pi$ 合起来，构成实际 $\Delta c_\kappa$ 的表达。连续表面点直接代入 $L(x)$ 是理解模型的近似；正式观测先在**像素射线交点**渲染，再投影取样，所以还存在像素采样与双线性插值作用。

### 8.2 光源也随曲率运动

每个光源点可写为

$$
\ell_j=X_{0,\kappa}(s_j,t_j)+o\,\mathrm{outer}_j+\zeta_j n_{0,\kappa}(s_j),\qquad m_j=-\mathrm{outer}_j.
$$

左右边：$s_j=\pm W/2$，$t_j$ 为纵向 LED 坐标，$\mathrm{outer}_j=\pm e_1$。上下边：$t_j=\pm H/2$，$\mathrm{outer}_j=\pm e_2$，纵向坐标按灯带中心线半径修正为 $s_j=\mathrm{along\_offset}+\ell_{\parallel,j}/(1+\kappa z_c)$。这里 $o$ 是外侧 offset，$\zeta_j=z_c+$ 带符号宽度积分偏移。

本次是逐 LED 离散功率，**每个积分点始终分到其所属 LED 功率的 1/2，$\partial_\kappa P_j=0$**。不要用历史“均匀灯带、曲率相关面积权重”的功率分配公式解释本次结果。全部四个条件的 52 个积分点位置、朝向和功率已由保存快照精确导出到 ZIP 中 `analysis/light_samples.npz`，可以直接代入解析计算；无需猜测上下灯的 RGB 功率。

即使所有 LED 功率不变，$\ell_j,m_j,x,n_o,\omega_j,d_j,V_j$ 都可能改变，$g$ 和投影也会改变。因此同时旋转表面局部基底不足以证明颜色响应不变。

### 8.3 小曲率的一阶项

在 $\kappa=0$ 附近

$$
X_{0,\kappa}=(s,t,-\kappa s^2/2)+O(\kappa^2),\quad
n_{0,\kappa}=(\kappa s,0,1)+O(\kappa^2),\quad
e_{1,\kappa}=(1,0,-\kappa s)+O(\kappa^2).
$$

用撇号表示对 $\kappa$ 的导数。在光样本未被遮挡（$V_j=1$）、两项余弦严格为正、远离接触及可见性切换的光滑分支上，$K_j>0$，于是

$$
\frac{K_j'}{K_j}=\frac{\mu_j'}{\mu_j}+a_j\frac{\nu_j'}{\nu_j}-2\frac{d_j'}{d_j},
$$

$$
d_j'=\omega_j\cdot(\ell_j'-x'),\quad
\omega_j'=\frac{(I-\omega_j\omega_j^T)(\ell_j'-x')}{d_j},
$$

$$
\mu_j'=n_o'\cdot\omega_j+n_o\cdot\omega_j',\qquad
\nu_j'=-m_j'\cdot\omega_j-m_j\cdot\omega_j'.
$$

始终被遮挡的稳定分支有 $K_j=K_j'=0$，直接作为零项处理，不使用上述比值式。于是 $L'=(10^6/\pi)\operatorname{diag}(\rho)\sum_jP_jK_j'$，$S'=g'L+gL'$，$A'=\eta M S'$。忽略量化、裁剪和空间取样、只看正值连续观测时，inverse_sRGB 后保留的 Reinhard 导数是

$$
c'=\operatorname{diag}((1+A)^{-2})A'.
$$

实际色差还要分别处理压入图和参考图，并加入投影移动的图像梯度项 $\nabla c\,\pi'$ 以及插值。8-bit 量化是分段常值，阴影/接触/余弦截断有分支切换，不能用全局光滑一阶导数替代。R=50 的 $\kappa W/2=0.559$ rad，是否足以用一阶近似必须检验，不能先假设高阶项很小。

## 9. LUT 误差关系与证据边界

固定平面表时存在精确关系

$$
\boxed{e_\kappa-e_0=F_0(\Delta c_\kappa)-F_0(\Delta c_0)-(\beta_\kappa-\beta_0).}
$$

在 LUT 插值单元和观测分支不切换的局部，可写

$$
e_\kappa-e_0\approx\kappa\left[J_{F_0}(\Delta c_0)\left.\partial_\kappa\Delta c\right|_0-\left.\partial_\kappa\beta\right|_0\right].
$$

这说明色差变化的方向、LUT 局部敏感度和真值变化共同决定误差；单独的颜色变化模长无法预测全部误差。

本次只评价坡度/法向，未进行高度积分，不能推出高度重建误差。也不能据此证明任意平面 LUT 都不能迁移、任何弯曲一定恶化、或者不存在某种可标定不变量。现有证据支持的是：**本模型与当前照明下，局部切平面几何表达本身不能保证平面颜色—坡度表保持准确；针对曲率重标定可以显著缓解退化。**

## 10. 结果归档与读取方式

ZIP 保留仓库相对目录结构。根目录 `README_for_analysis_AI.md` 给出入口，`PACKAGE_CONTENTS.json` 列出每个文件的字节数和 SHA256（仅供完整性/来源记录，不作为程序运行门槛）。

| 文件 | 分析用途 |
|---|---|
| 本文与实验 README | 几何、观测与指标定义 |
| 源 `normal_lut.npz`、三个 `calibration/radius_*/normal_lut.npz` | 四张最终表 |
| 各 `normal_lut_train.npz`、验证报告、manifest | 训练/验证与发布状态；训练表不等于最终表 |
| 源与三个曲面的 `normal_samples.npz` | 全部原始标定颜色—坡度配对及材料坐标、姿态归属，不是降采样数据 |
| `RUN/samples/<condition>/test/*.npz` | 全部 504 个逐姿态测试材料样本，可重算指标或分析误差场 |
| `RUN/samples/<condition>/reference.npz` | 四条件参考 sRGB8、掩膜、材料域解码颜色 |
| `metrics.json`、`per_pose.csv`、`cross_curvature.csv` | 原始汇总和逐姿态数值 |
| `mapping_difference_*.npz` | 实际物理查询色、权重、函数差、共同色盒/严格支持掩膜 |
| `figures/` | 22 张 PNG 与对应完整数值 NPZ |
| `analysis/` | 补算的配对漂移表、光源积分点表，根目录快捷副本 |
| `utils/*.py`、`experiment/curve_lut_transfer/*.py` | 渲染、取样、LUT、指标和调度的代码快照 |
| 16 帧代表性 `image.png/render.npz/metadata.json` | 每条件参考、中心两深度、平面 LUT 最差姿态的逐像素诊断 |

逐姿态测试 NPZ 的关键字段：

- `grid_index`：219×560 参考材料网格扁平索引，按序排列；跨条件求交/配对用它，不能按图像像素或 NPZ 行号直接配对。
- `material_coordinates`：$(s,t)$，mm；`area_mm2`：参考材料面积权重。
- `color_residual_samples`：有符号 $\Delta c$；`reference_color_samples`：对应参考颜色；两者相加可得当前图的材料点采样颜色。
- `slope_samples`：局部真值 $(p,q)$；`reference_normal_samples`、`tangent_frame_samples`：$n_0,E$；`normal_geometric_world`：$n_d$。
- `normal_displacement_mm`：$\delta$；`sample_mask/contact_mask/validity_bits`：完整网格掩膜和诊断标志。

读取示例（解压根目录运行，仅依赖 NumPy）：

```python
from pathlib import Path
import json
import numpy as np

run = Path('results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400')
metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
with np.load(run / 'samples/radius_50/test/test_000062.npz', allow_pickle=False) as z:
    color = z['color_residual_samples']
    beta = z['slope_samples']
    material_index = z['grid_index']
    print(color.shape, beta.shape, metrics['metrics']['flat/radius_50/common']['rmse'])
```

代表性最差姿态分别是：平面 `test_000125`，R=200 `test_000113`，R=100 `test_000001`，R=50 `test_000113`。中心姿态为 `test_000062`（0.75 mm）和 `test_000063`（1.25 mm）。实际 UV/深度见 manifest 的 `identity.test_poses`。

为便于上传，未收录约 11 GB 原始渲染中的其余帧、逐姿态训练样本的重复归档、系统排障日志、缓存和虚拟环境。**全部参与拟合的颜色—坡度配对、全部独立测试材料样本仍完整保留**；逐姿态训练档独有的法向位移/可见性细节不保证可从汇总样本恢复。包用于离线误差和理论分析，并非可以无缺失续跑/重渲染的完整项目备份。

历史 manifest 中的绝对路径保持原样；把原仓库根路径替换成解压根路径即可定位已收录文件，未收录文件仍会缺失，不要将其解释为运行失败。打包的源码是交接时版本；`runner.py` 在运行后移除了无必要的代码/依赖版本续跑门槛并调整说明，数值核心未改变。来源差异详见包清单，不重写原 manifest、不重新标注历史结果。

本次交接检查了补算基线/曲面 RMSE 与原报告一致（容差 1e−7），逐姿态满足 MSE 分解恒等式，并校验 ZIP 文件完整性。没有因打包重新启动 2,620 帧扫描，也没有宣称完成尚未执行的光学机理消融或迁移校正实验。


## 附录：实现验证结果

日期：2026-10-24。实现入口为 `experiment/curve_lut_transfer`，配置为 `configs/curve_lut_transfer.json`。本记录说明代码、数值对照与小规模流程验证，**不是完整正式迁移实验的结论**。方法与操作见 [实验 README](../experiment/curve_lut_transfer/README.md)，正式实验结论见 [弯曲后迁移 LUT 效果](弯曲后LUT迁移实验结果.md)。

### 完成范围

- 已实现冻结源平面 LUT、姿态和配置；同材料点参考配对；逐点切平面 b 真值；独立测试；按源划分重标定；共同物理 RGB 上的函数比较；指标、图表、原始数据、逐姿态检查点及续跑。
- 已提供可选参考亮度补偿、预算受控的 2×2 坡度和 3×3 颜色校正。局部 J 与未见曲率泛化尚未开展；不声称发现光学不变量。
- 通用平面标定/旧 NormDiff 的几何限制保持不变。共享拟合流程新增显式固定姿态划分和必要姿态完整性检查，不降低原验证阈值。
- `results_render` 与 `results_curve_render` 分别保存平面和曲面原始帧，报告和派生 LUT 在源运行的 `evaluation/curve_lut_transfer/<run>/` 下。

### 执行的检查

| 检查 | 实际结果 |
|---|---|
| 独立入口 `--help`，包含直接脚本用法 | 通过 |
| 现有 `results_lut/analytic` 的 `--dry-run` | 通过；704 个源姿态，固定 563/141 划分；126 个独立测试姿态；总计 2,620 帧 |
| 当前标定配置与源快照比较 | 相同；运行仍使用源快照 |
| 全量 unittest 回归 | 149 项通过，约 91 s；包括当时新增的 12 项实验测试 |
| 随后新增中断恢复测试，以及最终专项复测 | `tests.test_curve_calibration` + `tests.test_curve_lut_transfer` 共 13 项全部通过，约 13 s；累计覆盖 150 个不同测试 |
| 实际正式标定的 3 个姿态回放 | 10,171 条样本；色差、坡度最大绝对差均为 0 |
| 数值域和几何 | 位移有限差分、平面极限、刚体旋转、逐点切向基、相切/越界、零压入及 CPU/CUDA 对照通过 |
| 指标 | 姿态等权/面积加权、空姿态传播、颜色越界保留、正权重角点支持、不同网格色盒的相同函数对照、配对 bootstrap 通过 |
| 固定标定划分 | 缺少必要姿态时不发布正式 LUT，保留失败状态和诊断结果 |
| 续跑 | 已完成运行不渲染；模拟中断后只执行余下 8/10 帧；已完成文件内容和时间戳不变；配置变化、数据损坏、非空目录拒绝覆盖 |
| 可选校正 | 独立预算 ID、嵌套预算、参考有效域、2×2/3×3 校正及实际输出流程通过固定小样例测试 |
| 图表检查 | 已查看曲率曲线、共享色标位置图、交叉矩阵、向量图及 RGB 切片；矩阵使用随背景变化的文字颜色，真值/预测箭头共用比例尺 |
| 静态检查 | 新模块编译、JSON 解析、`git diff --check` 通过 |

Windows 的部分旧测试使用 `TemporaryDirectory` 的 0700 模式时会产生不可写 ACL。全量测试仅在该次 Python 进程中将 `os.mkdir(..., 0700)` 适配为继承工作区权限的 0777；未修改生产代码、测试断言或旧输出。新测试直接使用已有 `workspace_temporary()` fixture，正常命令可运行。

### 正式源数据回放

续跑检查已按用户要求简化：不再计算并比对全部程序文件哈希，也不因提交号、依赖版本或实现标签变化阻止运行。只比较实际输入、有效配置和姿态划分，并检查已有产物完整性。旧版 manifest 直接兼容，续跑保留原运行标识，不需要手动修改历史清单。新增旧代码/旧依赖版本仍可续跑的回归用例，与其余实验流程测试共 9 项通过。

使用归档的模拟配置，在内存完成旧 XY 参数到归一化弧长参数的换算，随后使用新采样器重渲染同一零压入参考和以下姿态。比较的是 `normal_samples.npz` 中对应的材料点坐标、有符号色差和坡度；没有修改源样本或 LUT。

| image key | 匹配样本数 | 色差最大绝对差 | 坡度最大绝对差 |
|---|---:|---:|---:|
| `000000` | 1,789 | 0 | 0 |
| `000351` | 3,449 | 0 | 0 |
| `000703` | 4,933 | 0 | 0 |

实际设备为 `cuda:0`，几何 float64、辐射量 float32；Inductor 编译生效，两个灯块变体初始核对通过，CUDA graph 回放 12 次，无回退和 OOM 重试。这证明本次冻结源观测链的适配一致，不代表所有历史数据或其他后端已验证。

### 小样例与正式运行记录

保留的图表检查产物位于 `tests/curve_lut_transfer_smoke_20261024T033629918297Z/`。其源 LUT 是独立测试构造，使用 4 个标定姿态、2 个测试姿态及平面/100 mm 两个条件，共 10 帧。曲面标定按原阈值失败，只输出诊断 LUT；图表明确标记 diagnostic，未创建正式曲面 `normal_lut.npz`。图表修订仅从保存的数据生成到同级 `figure_review_2/`，未覆盖原检查点或重渲染观察数据。

2026-10-24 已从原断点完成正式运行 `results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400/`，包括全部 2,620 帧及后续拟合、评价和 22 张图。200/100/50 mm 三个曲率均按原 704 个必要姿态、原验证阈值通过标定；具体迁移精度见该运行的 `metrics.json`，未在本次故障诊断中展开科学结论。预检给出的未压缩保守存储估计约为 218.8 GB；实际压缩后大小不同。实验没有通过降低分辨率、删除姿态、减少光源或稀疏材料采样来缩减正式工作量。

对同一运行的 manifest 进行了限定路径的 Process Monitor 追踪，保存于 `diagnostics/file_access_20261024T041404851Z/attempt_3/`。本轮记录 9,247 个访问事件，包含 134 次文件重命名，未记录 `ACCESS DENIED`、`SHARING VIOLATION` 等拒绝结果，程序正常完成。保留 PML、CSV 和 `capture_summary.json`；本轮未复现故障，不能据此认定具体安全软件是原先的拦截者。另有一次追踪因旧的代码哈希校验而在预检退出，该过严限制现已移除。

正式执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiment.curve_lut_transfer --lut-results results_lut/analytic --exp-config configs/curve_lut_transfer.json
```

## 几何公式核对结果

已对 R=50、100、200 mm 与三个锚点组合的 9 个球面接触内点核对几何公式：由局部切平面坡度恢复法向的最大分量误差约为 2.22e-16；使用 1e-4 mm 中心差分核对法向位移梯度，最大误差约为 3.10e-11。实现阶段的 tests/test_curve_calibration.py 验证逐点标签、位移数值导数、平面极限及旋转协变。这些几何核对只验证标签与重建公式，不构成 LUT 光学可迁移性的结论；正式迁移结果见正文。
