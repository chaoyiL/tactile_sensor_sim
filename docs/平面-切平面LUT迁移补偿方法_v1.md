# 平面 LUT 向曲面迁移的补偿方法

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
| $D_0,D_\kappa\in\mathbb R^{3\times2}$ | 无接触参考点的局部 RGB—坡度响应；求导固定位置、活动光源和可见性 |
| $\lambda$ | 固定 ridge 系数，本次为 $0.001$ |

物理量取源配置快照；源样本共 704 个姿态，固定训练/验证划分为 563/141，沿用原拟合方法和阈值。

## 2. 计算公式

### 端点补偿

#### 连续点 RGB 函数 C 的具体形式

$C_{\kappa,a}(X,n)\in[0,1]^3$ 表示曲率 $\kappa$、压入状态 $a$ 下，在世界坐标点 $X$、单位几何法向 $n$ 处计算的 RGB。它对应 [`PhotometryModel.evaluate()` 返回的 `display_rgb`](../experiment/curve_lut_compensation/photometry.py)：**曝光与 Reinhard 之后、sRGB 编码之前的连续 RGB**。这里的端点指同一材料点的压入态和无压入参考态。

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

$\omega_{\kappa,j}$ 从表面点指向光源。$V_{\kappa,a,j}(X,n)\in\{0,1\}$ 是该状态下表面点到光源的阴影可见性。于是入射 RGB 辐照度和 Lambert 出射 RGB 辐亮度为：

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

其中 $\rho$ 为 `surface.reflectance_rgb`，$\odot$ 表示逐通道乘法；$(m_j+1)/(2\pi)$ 将发光角分布归一到前半球总功率。距离使用 mm，因此 $\mathcal E$ 的单位为 W/mm²，$10^6$ 将其转换为 W/m²；$L$ 的单位为 W/(m²·sr)。当前模型的表面反射项为 Lambert 漫反射。

再令相机位置为 $O$，相机朝前单位轴为 $f_{\mathrm{cam}}$，针孔半宽为 $h$、焦距为 $f$，相机响应矩阵为 $M\in\mathbb R^{3\times3}$，曝光系数为 $\eta$。相机因子及最终 RGB 为：

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

列向量写法 $M(gL)$ 对应代码中的行向量写法 `sensor @ camera_response.T`。配置对应关系为：

| 公式量 | 配置或实现来源 |
|---|---|
| $\ell_{\kappa,j},e_{\kappa,j},P_{\kappa,j}$ | [`analytic_light_samples()`](../utils/light_distribution.py)，曲面由 [`curve_light_samples()`](../utils/curve_geometry.py) 生成 |
| $m_j$ | 所属灯带的 `lights[].cosine_exponent` |
| $V_{\kappa,a,j}$ | `render.shadows` 开启时，按状态 $a$ 的完整变形触觉表面求交；关闭时取 1 |
| $O,f_{\mathrm{cam}}$ | `camera.position_mm` 与 [`camera_frame()`](../utils/geometry.py) 的第三列 |
| $h,f,M,\eta$ | `camera.pinhole_half_width_mm`、`camera.focal_length_mm`、`camera.response_rgb_3x3`、`render.exposure_scale` |

光积分点的功率已经包含积分权重，不再额外乘灯带面积。本次使用 `discrete_rgb_points_v1`：每个纵向发光点的 `point_power_rgb_w` 均分到 `samples_width` 个宽度积分点，保持每个发光点的总功率；均匀灯带模式则按积分权重分配 `power_rgb_w`。光源位置及朝向随曲率 $\kappa$ 更新，在同一曲率下对参考态和压入态保持固定；压入状态 $a$ 通过求值位置、法向及阴影影响 $C$。

阴影计算沿用平面/曲面的共享渲染核：射线起点为 $X+10^{-6}n_{\mathrm{opt}}$ mm，方向为 $\omega_{\kappa,j}$，最近交点距离满足 $t_{\mathrm{hit}}\geq r_{\kappa,j}-10^{-5}$ mm 时取 $V=1$。相机能否看到该材料点另由 $\mathcal M_{\kappa,a}$ 筛选；$g(X)$ 是光度因子。

2026-10-28 结果使用的[保存快照](../results_lut/analytic/evaluation/curve_lut_compensation/20261028_endpoint_hpt_01/config_snapshot.json)中，$N=52$、全部 $m_j=1$、$\rho=(0.8,0.8,0.8)^T$、$M=I_3$、$\eta=6000$、$h=0.004$ mm、$f=3$ mm、$O=(0,0,-60)^T$ mm、$f_{\mathrm{cam}}=(0,0,1)^T$，阴影开启。此时发光权重简化为 $[-e_{\kappa,j}\cdot\omega_{\kappa,j}]_+/\pi$。各点功率及几何参数以该快照为准；复核历史结果时使用保存快照。

#### 端点求差与迁移

连续点响应差（$a=0$ 为无压入参考状态）：

$$
H_{\kappa,a}^{\mathrm{pt}}
=C_{\kappa,a}(X_{d,\kappa},n_{d,\kappa})
-C_{\kappa,0}(X_{0,\kappa},n_{0,\kappa}).
$$

上述两次求值各自完成曝光与 Reinhard 后再相减；不能先对辐亮度或曝光前 RGB 求差再做 Reinhard。它们不经过 sRGB 编码、8 位量化或图像插值。源观测 $c_{0,a}^{\mathrm{obs}}$ 则由保存的 sRGB8 解码并按材料点配对求差；解码恢复的是 Reinhard 后的 RGB 域。

迁移颜色与目标坡度标签：

$$
c_{\kappa,a}^{\mathrm{mig}}
=c_{0,a}^{\mathrm{obs}}+H_{\kappa,a}^{\mathrm{pt}}-H_{0,a}^{\mathrm{pt}},
\qquad
\beta_{\kappa,a}=(p,q)^T
=-\frac{E_\kappa^Tn_{d,\kappa}}{n_{0,\kappa}\cdot n_{d,\kappa}}.
$$

拟合目标 LUT，查询目标观测：

$$
\mathcal S_\kappa
=\{(c_{\kappa,a}^{\mathrm{mig}},\beta_{\kappa,a})_i:
 i\in\mathcal M_{\kappa,a}\},
\qquad
F_\kappa=\operatorname{Fit}(\mathcal S_\kappa),
\qquad
\widehat\beta=F_\kappa(c_\kappa^{\mathrm{obs}}).
$$

### 局部响应矩阵

参考点附近的几何法向及响应导数：

$$
n_\kappa(\beta)
=\frac{n_{0,\kappa}-E_\kappa\beta}{\sqrt{1+\lVert\beta\rVert^2}},
\qquad
D_\kappa
=\left.\frac{\partial C_{\kappa,0}(X_{0,\kappa},n_\kappa(\beta))}
{\partial\beta}\right|_{\beta=0}.
$$

$D_\kappa^+$ 为伪逆，$c=c_\kappa^{\mathrm{obs}}$：

| 方法 | 计算 | 输出坡度 |
|---|---|---|
| 投影颜色 | $c'=D_0D_\kappa^+c$ | $F_0(c')$ |
| 保留残差颜色 | $c'=[I_3+(D_0-D_\kappa)D_\kappa^+]c$ | $F_0(c')$ |
| 直接反演 | $R=D_\kappa^+$ | $Rc$ |

三种 ridge 版本将上表中的 $D_\kappa^+$ 替换为：

$$
R_\lambda=(D_\kappa^TD_\kappa+\lambda I_2)^{-1}D_\kappa^T.
$$

## 3. 输出

| 输出 | 内容 |
|---|---|
| 迁移样本 | 每姿态的 $c^{\mathrm{mig}}$、$\beta$、材料坐标、网格索引、面积权重及有效域 |
| 目标 LUT | 每曲率一张 $F_\kappa$；原划分验证通过后保存正式表，失败保存诊断表与失败状态 |
| 局部矩阵 | $D_0,D_\kappa$、三类补偿矩阵及 ridge 版本、奇异值、秩、条件数和有效掩膜 |
| 查询与评价 | 预测坡度 $\widehat\beta$；逐姿态及汇总误差、支持率、覆盖率和图表，见[对照结果](平面-切平面LUT迁移补偿.md) |

输出目录：源 LUT 结果目录下的 evaluation/curve_lut_compensation/运行子目录。
