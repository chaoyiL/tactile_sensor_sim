# 平面 LUT 向曲面迁移的端点补偿方法

## 1. 使用的量

| 量 | 含义 |
|---|---|
| $\kappa=1/R$ | 目标曲率；平面取 $\kappa=0$ |
| $(s,t),a$ | 材料坐标（mm）、球压入姿态 |
| $c_{0,a}^{\mathrm{obs}}$ | 平面源样本的有符号 RGB 色差：sRGB8 解码后，同材料点的压入观测减参考观测 |
| $C_{\kappa,a}(X,n)$ | 已知光源、材质、相机、曝光和遮挡模型计算的连续点 RGB，经过 Reinhard；不含像素量化和图像插值 |
| $X_{0,\kappa},n_{0,\kappa},E_\kappa$ | 目标参考面的位置、几何法向、两个切向基 |
| $X_{d,\kappa},n_{d,\kappa}$ | 目标球压入几何的位置、几何法向；光学法向为 $-n_{d,\kappa}$ |
| $\mathcal M_{\kappa,a}$ | 同时满足源样本范围、目标接触、参考有效域、相机可见性、取样邻域及未饱和条件的材料点 |
| $F_0,c_\kappa^{\mathrm{obs}}$ | 已发布平面 LUT、目标实际观测色差；目标观测仅用于查询与评价 |

物理量取源 LUT 运行保存的配置快照。2026-10-28 历史运行使用 704 个源姿态，固定训练/验证划分为 563/141；其他输入的姿态数与划分以其 manifest 为准。

## 2. 计算公式

### 2.1 连续点 RGB 响应

$C_{\kappa,a}(X,n)\in[0,1]^3$ 表示曲率 $\kappa$、压入状态 $a$ 下，在世界坐标点 $X$、单位几何法向 $n$ 处计算的 RGB。它对应 [`PhotometryModel.evaluate()`](../experiment/curve_lut_compensation/photometry.py) 返回的 `display_rgb`：**曝光与 Reinhard 之后、sRGB 编码之前的连续 RGB**。端点指同一材料点在压入态与无压入参考态的两次求值。

记 $[x]_+=\max(x,0)$，向量的截断、乘除和 Reinhard 均逐通道进行。当前实现要求 `surface.view_side_z=-1`、`indentation.direction_normal=-1`，所以参与受光计算的光学法向为 $n_{\mathrm{opt}}=-n$。将全部灯带离散为 $N$ 个光积分点；第 $j$ 点的位置、朝内发光单位法向、RGB 功率和余弦指数分别为 $\ell_{\kappa,j}$、$e_{\kappa,j}$、$P_{\kappa,j}\in\mathbb R_+^3$（W）和 $m_j$。对点与光源不重合的几何，定义：

$$
r_{\kappa,j}=\lVert\ell_{\kappa,j}-X\rVert,
\qquad
\omega_{\kappa,j}=\frac{\ell_{\kappa,j}-X}{r_{\kappa,j}},
\qquad
\psi_m(\mu)=
\begin{cases}
\mu^m,&\mu>0,\\
0,&\mu\leq0.
\end{cases}
$$

$\omega_{\kappa,j}$ 从表面点指向光源。$V_{\kappa,a,j}(X,n)\in\{0,1\}$ 是该状态下表面点到光源的阴影可见性。入射 RGB 辐照度和 Lambert 出射 RGB 辐亮度为：

$$
\mathcal E_{\kappa,a}(X,n)
=\sum_{j=1}^{N}
P_{\kappa,j}\,
\frac{m_j+1}{2\pi}\,
\psi_{m_j}(-e_{\kappa,j}\cdot\omega_{\kappa,j})\,
[-n\cdot\omega_{\kappa,j}]_+\,
\frac{V_{\kappa,a,j}(X,n)}{r_{\kappa,j}^2},
\qquad
L_{\kappa,a}(X,n)
=\frac{10^6}{\pi}\,
\bigl(\rho\odot\mathcal E_{\kappa,a}(X,n)\bigr).
$$

其中 $\rho$ 为 `surface.reflectance_rgb`，$\odot$ 表示逐通道乘法；$(m_j+1)/(2\pi)$ 将发光角分布归一到前半球总功率。距离使用 mm，因此 $\mathcal E$ 的单位为 W/mm²；$10^6$ 将其转换为 W/m²，$L$ 的单位为 W/(m²·sr)。表面反射项为 Lambert 漫反射。

令相机位置为 $O$，朝前单位轴为 $f_{\mathrm{cam}}$，针孔半宽为 $h$、焦距为 $f$，相机响应矩阵为 $M\in\mathbb R^{3\times3}$，曝光系数为 $\eta$。相机因子及最终连续 RGB 为：

$$
v(X)=\frac{X-O}{\lVert X-O\rVert},
\qquad
g(X)=[v(X)\cdot f_{\mathrm{cam}}]_+^4
\left(\frac{2h}{f}\right)^2,
$$

$$
u_{\kappa,a}(X,n)
=\left[\eta\,M\bigl(g(X)L_{\kappa,a}(X,n)\bigr)\right]_+,
\qquad
\boxed{
C_{\kappa,a}(X,n)
=\operatorname{clip}_{[0,1]}
\left(\frac{u_{\kappa,a}(X,n)}{1+u_{\kappa,a}(X,n)}\right).
}
$$

列向量写法 $M(gL)$ 对应代码中的行向量写法 `sensor @ camera_response.T`。公式量与配置来源如下：

| 公式量 | 配置或实现来源 |
|---|---|
| $\ell_{\kappa,j},e_{\kappa,j},P_{\kappa,j}$ | [`analytic_light_samples()`](../utils/light_distribution.py)，曲面由 [`curve_light_samples()`](../utils/curve_geometry.py) 生成 |
| $m_j$ | 所属灯带的 `lights[].cosine_exponent` |
| $V_{\kappa,a,j}$ | `render.shadows` 开启时，按状态 $a$ 的完整变形触觉表面求交；关闭时取 1 |
| $O,f_{\mathrm{cam}}$ | `camera.position_mm` 与 [`camera_frame()`](../utils/geometry.py) 的第三列 |
| $h,f,M,\eta$ | `camera.pinhole_half_width_mm`、`camera.focal_length_mm`、`camera.response_rgb_3x3`、`render.exposure_scale` |

光积分点的功率已经包含积分权重，不再额外乘灯带面积。本次历史配置使用 `discrete_rgb_points_v1`：每个纵向发光点的 `point_power_rgb_w` 均分到 `samples_width` 个宽度积分点，保持该发光点的总功率；均匀灯带模式则按积分权重分配 `power_rgb_w`。光源位置和朝向随曲率 $\kappa$ 更新，在同一曲率下对参考态和压入态保持固定；压入状态 $a$ 通过求值位置、法向及阴影影响 $C$。

阴影计算沿用平面/曲面的共享渲染核：射线起点为 $X+10^{-6}n_{\mathrm{opt}}$ mm，方向为 $\omega_{\kappa,j}$，最近交点距离满足 $t_{\mathrm{hit}}\geq r_{\kappa,j}-10^{-5}$ mm 时取 $V=1$。材料点是否位于相机可见区域另由 $\mathcal M_{\kappa,a}$ 筛选；$g(X)$ 是光度因子。

2026-10-28 历史运行[保存快照](../results_lut/analytic/evaluation/curve_lut_compensation/20261028_endpoint_hpt_01/config_snapshot.json)中，$N=52$、全部 $m_j=1$、$\rho=(0.8,0.8,0.8)^T$、$M=I_3$、$\eta=6000$、$h=0.004$ mm、$f=3$ mm、$O=(0,0,-60)^T$ mm、$f_{\mathrm{cam}}=(0,0,1)^T$，且阴影开启。此时发光角权重简化为 $[-e_{\kappa,j}\cdot\omega_{\kappa,j}]_+/\pi$。各点功率及几何参数以所选源运行的配置快照为准。

### 2.2 端点求差、目标几何与迁移拟合

令 $a=0$ 为无压入参考状态。同一材料点的连续点响应端点差为：

$$
H_{\kappa,a}^{\mathrm{pt}}
=C_{\kappa,a}(X_{d,\kappa},n_{d,\kappa})
-C_{\kappa,0}(X_{0,\kappa},n_{0,\kappa}).
$$

两端分别完成曝光与 Reinhard 后再相减；不先对辐亮度或曝光前 RGB 求差，也不经过 sRGB 编码、8 位量化或图像插值。源观测 $c_{0,a}^{\mathrm{obs}}$ 则由保存的 sRGB8 解码，并按材料点配对求差；解码恢复的是 Reinhard 后 RGB 域。

目标迁移颜色与目标几何坡度标签为：

$$
c_{\kappa,a}^{\mathrm{mig}}
=c_{0,a}^{\mathrm{obs}}+H_{\kappa,a}^{\mathrm{pt}}-H_{0,a}^{\mathrm{pt}},
\qquad
\beta_{\kappa,a}=(p,q)^T
=-\frac{E_\kappa^Tn_{d,\kappa}}{n_{0,\kappa}\cdot n_{d,\kappa}}.
$$

其中 $X_{0,\kappa},n_{0,\kappa},E_\kappa$ 来自目标参考面几何，$X_{d,\kappa},n_{d,\kappa}$ 来自目标球压入几何；法向符号遵循几何法向约定。

只在有效迁移集合 $\mathcal M_{\kappa,a}$ 上组成拟合样本，并用源运行固定的完整训练/验证姿态划分及验证阈值拟合目标 LUT：

$$
\mathcal S_\kappa
=\{(c_{\kappa,a}^{\mathrm{mig}},\beta_{\kappa,a})_i:
 i\in\mathcal M_{\kappa,a}\},
\qquad
F_\kappa=\operatorname{Fit}(\mathcal S_\kappa),
\qquad
\widehat\beta=F_\kappa(c_\kappa^{\mathrm{obs}}).
$$

目标实际观测 $c_\kappa^{\mathrm{obs}}$ 只用于预测查询和测试评价，不参与迁移颜色、拟合样本、参数估计或超参数选择。拟合通过原验证规则时保存正式 LUT；失败时保留诊断 LUT 和失败状态。

## 3. 输出

| 输出 | 内容 |
|---|---|
| 迁移样本 | 每姿态的 $c^{\mathrm{mig}}$、目标坡度 $\beta$、材料坐标、网格索引、参考面积权重及有效域 |
| 端点迁移 LUT | 每个曲率条件的 $F_\kappa$、拟合配置、逐姿态验证与发布/诊断状态 |
| 查询与评价 | 三种 LUT 方法的预测坡度 $\widehat\beta$；`full` 和跨曲率 `common` 材料域上的逐姿态及汇总误差、LUT 支持、材料覆盖与图表 |

输出目录为源 LUT 结果目录下的 `evaluation/curve_lut_compensation/<run>/`。当前实现不生成局部响应矩阵或矩阵补偿结果。2026-10-28 的历史对照见[结果 v1](平面-切平面LUT迁移补偿结果_v1.md)；该历史报告保留当时的矩阵实验记录，不表示矩阵方法仍属于当前实现。
