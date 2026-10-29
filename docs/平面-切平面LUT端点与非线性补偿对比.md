# 平面—切平面 LUT：端点与非线性补偿对比

日期：2026-10-29。

**加入接收活动集更新的非线性 GN3，在本次静态档案上比端点 LUT 更准确，但查询更慢。** R=200/100/50 mm 的共同域 RMSE 分别降低 **12.39% / 10.97% / 18.50%**；端点查询 P50 为 1.55–1.95 ms，新法为 16.00–20.29 ms。新构型响应场更新约 0.57 s，端点法历史重建 LUT 约 138–151 s。R=50 的最差姿态仍有退化。

## 方法与输入

对应 [v2 设计](平面-切平面LUT迁移补偿方法_v2.md) 式 (11)–(15)，实现见 [nonlinear.py](../experiment/curve_lut_compensation/nonlinear.py)，比较流程见 [comparison.py](../experiment/curve_lut_compensation/comparison.py)。

| 项目 | 实现 |
|---|---|
| 输入 | 源快照的光源、相机、反射率；目标参考位置 $X_0$、法向 $n_0$、切向基 $E$、材料身份；实际有符号 RGB 色差 $c$ |
| 初值 | $\beta^0=F_0(c)$，查询冻结平面 LUT |
| 固定分支对照 | 固定参考位置、参考阴影和接收活动集，运行 GN1/GN2/GN3 |
| 活动集更新 GN3 | 每步按当前坡度重选正接收余弦光源，求和形成当前 $B$，计算该分支 Jacobian，更新坡度；共 3 步 |
| 闭式对照 | 用绝对颜色 $C_0+c$ 反演；安全色域、条件数、方向或活动集检查失败时，回退固定活动集 GN3 |
| 输出 | 直接输出局部坡度 $(p,q)$，不再查询平面 LUT |

$$
s=\sqrt{1+\lVert\beta\rVert^2},\qquad
n_o(\beta)=\frac{-n_0+E\beta}{s},\qquad
\Phi_{\mathcal A}(\beta)=T(B_{\mathcal A}n_o(\beta))-C_0.
$$

$$
\Delta\beta=-\left(J^TJ+\lambda I\right)^{-1}J^T(\Phi-c),\qquad
\beta^{k+1}=\beta^k+\Delta\beta.
$$

代码使用线性求解；阻尼 $\lambda=10^{-4}$，每步增量范数上限 0.5，均匀 RGB 权重，先验权重 0。参数未按目标测试结果调整。位置与阴影保持参考态；不使用测试球心、位移或真实法向作为推理输入。

## 精度

复用已发布端点 LUT 和历史测试观测，没有重新扫描或重新拟合端点 LUT。四条件各 126 姿态，共 504 个文件、1,696,754 个有效材料样本；平面/R200/R100/R50 分别为 432,998 / 427,808 / 422,776 / 413,172 点。源数据为 704 姿态，原训练/验证划分 563/141。

`full` 为每条件全部有效材料点；`common` 为同姿态跨四条件的交集。所有方法保留相同测试点，每姿态按参考面积归一，再对姿态等权。坡度向量 RMSE/P95 无量纲，越低越好。

**共同材料域 RMSE：**

| 方法 | 平面 | R=200 mm | R=100 mm | R=50 mm |
|---|---:|---:|---:|---:|
| 冻结平面 LUT | 0.139938 | 0.149990 | 0.163866 | 0.196348 |
| 端点 LUT | 0.139938 | 0.128104 | 0.128551 | 0.151807 |
| 固定活动集 GN1 | 0.192265 | 0.195550 | 0.189391 | 0.208909 |
| 固定活动集 GN2 | 0.266692 | 0.243263 | 0.219507 | 0.197994 |
| 固定活动集 GN3 | 0.327158 | 0.262019 | 0.223448 | 0.217436 |
| **活动集更新 GN3** | **0.110886** | **0.112231** | **0.114450** | **0.123728** |
| 闭式 / 固定 GN3 回退 | 0.327136 | 0.260572 | 0.220427 | 0.216798 |

**完整域与尾部误差：** 法向角 RMS 和 P95 使用共同域。

| 构型 | 端点 full RMSE | 新法 full RMSE | 端点 P95 | 新法 P95 | 端点角 RMS（°） | 新法角 RMS（°） |
|---|---:|---:|---:|---:|---:|---:|
| 平面 | 0.149781 | 0.126506 | 0.290162 | 0.218407 | 7.059 | 5.770 |
| R200 | 0.138136 | 0.126753 | 0.264875 | 0.219662 | 6.375 | 5.828 |
| R100 | 0.137065 | 0.126087 | 0.261983 | 0.228445 | 6.356 | 5.881 |
| R50 | 0.151807 | 0.123728 | 0.317409 | 0.252284 | 7.362 | 6.164 |

共同域的配对 bootstrap 按 63 个锚点聚类、保留各深度，重复 2,000 次，seed=42。新法减端点的 RMSE 差值 95% 区间为：平面 `[-0.036512,-0.021447]`，R200 `[-0.021964,-0.009348]`，R100 `[-0.020027,-0.008238]`，R50 `[-0.033683,-0.022187]`；均低于 0。

**R50 的最差姿态 RMSE 从端点的 0.270619 增至新法的 0.287803**，新法最差为 `test_000113`。平均和 P95 改善不表示每个姿态都改善。逐姿态位置、深度、偏差及相对端点输出的差异均保存在原始报告。

## 速度

实际使用 RTX 5050 Laptop GPU、PyTorch 2.12.1+cu130；光度与反演用 CUDA，端点 LUT 查询和非线性初值查表用 CPU。几何 float64、辐射 float32、反演工作精度 float64；52 个离散光积分点，阴影开启。

热身 2 次，各姿态测量 5 次，每方法/条件共 630 次；GPU 前后显式同步。每次是平均约 3,279–3,436 点的接触斑块。主机口径包含 CPU 初值查表、GPU 场取样、传输、反演及诊断；排除 LUT 文件加载、磁盘、指标和绘图。GPU 常驻口径使用相同算法及诊断，排除 CPU 查表、场取样和 H2D/D2H。

| 构型 | 端点主机 P50（ms） | 新法主机 P50（ms） | 新法 GPU 常驻 P50（ms） | 新法主机 P95（ms） | 新法主机 P99（ms） |
|---|---:|---:|---:|---:|---:|
| 平面 | 1.44 | 14.80 | 12.26 | 18.59 | 23.97 |
| R200 | 1.55 | 16.00 | 13.08 | 22.38 | 27.63 |
| R100 | 1.85 | 20.14 | 16.34 | 34.10 | 60.23 |
| R50 | 1.95 | 20.29 | 16.14 | 29.32 | 37.37 |

新构型准备成本如下。端点历史计时含有效域、迁移、压缩/检查点、拟合及验证；新法为本次全 122,640 点的参考几何、可见性、活动集、系数场及条件诊断。这是两种方法各自所需的准备工作，不能解释为同一个 GPU 内核的加速比。

| 构型 | 端点迁移+拟合（历史实测，s） | 新法响应场更新 P50（s） |
|---|---:|---:|
| 平面 | 98.208 | 0.387 |
| R200 | 137.948 | 0.576 |
| R100 | 150.559 | 0.575 |
| R50 | 146.863 | 0.567 |

首次 CUDA 模型初始化另计；完整对比进程耗时 **365.33 s**，包含全部条件、重复计时、指标和绘图，不是一次在线补偿耗时。上述查询不含曝光、采集/解码、整体几何估计或位移积分，也不是整幅图耗时。场更新约 0.57 s、R100 查询 P99 约 60 ms，当前实现尚不能宣称动态变曲面达到 30/60 Hz。

## 分支诊断与验证

固定分支 GN 均未超过端点法。闭式在平面/R200/R100/R50 仅有 0.14% / 0.57% / 1.05% / 1.91% 的点直接有效，其余回退固定 GN3，表中是混合结果。活动集更新版本的求解失败与非有限回退均为 0；活动集相对参考频繁变化是算法处理的分支变化，不是预测失败。位移、阴影及像素残差的各自误差贡献尚未隔离。

48 项相关测试通过；504/504 预测完成，56 组指标有效，冻结/端点两基线在 full/common 共 16 组 RMSE 与历史逐值一致。completed 续跑不改写产物，也不导入 PyTorch。实际 GPU 的四条件共 16 个有限法向小样例，对连续点光度实现的线性/显示 RGB 最大绝对差分别为 $2.38\times10^{-7}$ / $8.03\times10^{-8}$。

## 运行与原始数据

配置为 [curve_lut_compensation.json](../configs/curve_lut_compensation.json)，默认入口继续执行端点法。新方法使用独立比较模式，`--out` 选择新的空运行目录：

```powershell
.\.venv\Scripts\python.exe -m experiment.curve_lut_compensation --lut-results results_lut/analytic --transfer-results results_lut/analytic/evaluation/curve_lut_transfer/20261024T034322_868406Z_8b6a0400 --compare-nonlinear --endpoint-results results_lut/analytic/evaluation/curve_lut_compensation/timing_20261029_d5331c8c --out results_lut/analytic/evaluation/curve_lut_compensation/<new_run>
```

- [全部指标与配对区间](../results_lut/analytic/evaluation/curve_lut_compensation/nonlinear_active_20261029_aa5b0333/metrics.json)、[逐姿态结果](../results_lut/analytic/evaluation/curve_lut_compensation/nonlinear_active_20261029_aa5b0333/per_pose.csv)、[全部方法计时](../results_lut/analytic/evaluation/curve_lut_compensation/nonlinear_active_20261029_aa5b0333/runtime.json)、[分支诊断](../results_lut/analytic/evaluation/curve_lut_compensation/nonlinear_active_20261029_aa5b0333/diagnostics.json)、[对比图](../diagnostics/curve_lut_compensation/nonlinear_active_20261029_aa5b0333/figures/accuracy_speed.png)。
- [验证记录](../diagnostics/curve_lut_compensation/nonlinear_active_20261029_aa5b0333/verification.json)、[GPU 光度核对](../diagnostics/curve_lut_compensation/nonlinear_active_20261029_aa5b0333/gpu_photometry_validation.json)、[固定分支首轮档案](../results_lut/analytic/evaluation/curve_lut_compensation/nonlinear_20261029_37214964/metrics.json)。

新法用已知材料位置做局部反演，端点查询是全局 RGB LUT，两者使用位置的方式不同。测试档案已用于前期分析，本次属于静态档案比较；尚未实现/验证位移耦合、实际接触阴影更新、源残差学习、一般曲面网格或动态序列，不能外推到真实传感器或任意变曲面。
