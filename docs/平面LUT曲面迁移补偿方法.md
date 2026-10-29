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

连续点响应差（$a=0$ 为无压入参考状态）：

$$
H_{\kappa,a}^{\mathrm{pt}}
=C_{\kappa,a}(X_{d,\kappa},n_{d,\kappa})
-C_{\kappa,0}(X_{0,\kappa},n_{0,\kappa}).
$$

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
