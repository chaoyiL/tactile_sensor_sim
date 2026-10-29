# 端点补偿实验裁剪验证

日期：2026-10-29。当前 experiment.curve_lut_compensation 只保留端点补偿；对照为冻结平面 LUT 和历史同曲率重标定 LUT。六种局部响应矩阵补偿的实现、配置、预测分支、专属指标、图表与计时阶段已移除。

## 验证方式

- 43 项相关测试通过；范围转换与完整小样例流程在收尾修改后又单独复测，2 项通过。
- 复用原已发布 LUT 和全部 504 份独立测试样本，对三种 LUT、四个条件、full/common 两个材料域重新评价。
- 使用 CUDA 在原生产配置下，重新生成源姿态 000000 在四个条件下的端点迁移样本，与历史样本逐项对照。
- 本次未重新执行完整扫描，也未重新拟合正式 704 姿态 LUT；测试内的拟合使用独立小型 fixture，保留原阈值和失败状态规则。

## 数值结果

共同材料域坡度向量 RMSE：

| 条件 | 冻结平面 LUT | 端点补偿 LUT | 历史同曲率重标定 LUT |
|---|---:|---:|---:|
| 平面 | 0.139938 | 0.139938 | 0.139938 |
| R=200 mm | 0.149990 | 0.128104 | 0.128401 |
| R=100 mm | 0.163866 | 0.128551 | 0.128940 |
| R=50 mm | 0.196348 | 0.151807 | 0.152471 |

保留的 24 组指标对象和 8 组配对 bootstrap 与历史报告对应条目逐值一致；材料域和原验证标准未改变。四次 GPU 回放分别保留 1,789 / 1,775 / 1,757 / 1,707 个样本，色差、坡度、材料坐标、网格索引和面积权重均与历史数组完全一致。受检历史文件的哈希与修改时间未变。

记录与输出：[validation.json](../diagnostics/curve_lut_compensation/endpoint_only_20261029_adb483ee/validation.json)、[重新评价指标](../diagnostics/curve_lut_compensation/endpoint_only_20261029_adb483ee/evaluation/metrics.json)、[LUT 对比图](../diagnostics/curve_lut_compensation/endpoint_only_20261029_adb483ee/evaluation/figures/rmse_lut_comparison.png)。

## 配置、输出与续跑

- 配置中的 matrix 块已删除；显式传入旧矩阵配置会在读取实验数据前报错。
- 新输出只含三种 LUT 方法、full/common 域、材料覆盖，以及 LUT 对比图和原始数值；不生成矩阵文件或矩阵图。
- 已完成的旧运行仍只读校验。未完成的旧运行可复用兼容端点检查点；方法范围切换会归档旧报告、更新有效配置/方法语义并记录待评价状态。原退役文件保留在历史路径，不列为当前端点输出。
- 原正式 v1 运行实际执行完成态续跑后，受检报告哈希和修改时间不变，未导入 torch、未创建新计时会话，见[只读续跑复核](../diagnostics/curve_lut_compensation/endpoint_only_20261029_adb483ee/legacy_resume_verification.json)。
- [历史结果 v1](平面-切平面LUT迁移补偿结果_v1.md)、[历史方法 v1](平面-切平面LUT迁移补偿方法_v1.md) 和原 handoff 保持历史快照。v1 全套耗时包含矩阵对照，不能直接作为当前版本从零耗时。

当前方法：[端点补偿公式与输出](平面-切平面LUT迁移补偿方法.md)。运行入口与输出位置：[实验 README](../experiment/curve_lut_compensation/README.md)。
