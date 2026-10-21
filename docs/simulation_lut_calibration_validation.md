# 仿真 LUT 标定验收记录

验收日期：2026-10-21。

环境：NVIDIA GeForce RTX 5050 Laptop GPU（8151 MiB）、PyTorch 2.12.1+cu130、CUDA runtime 13.0。依赖来自提交内的 `uv.lock`。

## 数值与功能检查

- `python -m unittest discover -s tests -v`：17 项通过。
- GPU 渲染与 NumPy CPU 基线在 float64 下对照通过，包含球冠求交、面积灯积分和阴影射线；阈值为 `rtol=2e-11, atol=1e-13`。
- 移植的 LUT 构建、方差、有效节点、样本计数和 raw 三线性查询，与 `recon_stable` 固定提交 `e02f95f119aa42d5aaf1ebae1f2304d60f045e8c` 逐值一致。
- 默认配置 dry-run 得到 1316 个姿态；各深度数量为 `308, 270, 270, 234, 234`。
- 8 姿态小规模完整扫描通过：8 个姿态全部接受，共 1212 个样本；正式 LUT、训练 LUT、整图留出报告、8 张 verification 单图和全部 visualization 图表均成功生成。
- 第二遍 visualization 重新渲染的网格索引、signed RGB 色差和几何坡度与第一遍保存样本一致。

小规模完整扫描使用 `tests/config.calibration.small.json`，其验证阈值特意放宽，只用于覆盖发布路径；它不代表正式示例配置的标定精度结论。

## 渲染性能

命令：

```powershell
uv run --cache-dir .uv-cache --locked --extra gpu python tests/benchmark_renderer.py --config config.example.json --scale 1 --frames 3
```

条件：`640 × 480`、192 个面积灯采样点、开启阴影、float64 几何和辐射计算。CPU 与 GPU 使用相同物理配置，每侧先完成 GPU 预热；三帧结果为：

| 项目 | 结果 |
| --- | ---: |
| CPU 帧耗时 | 3.152 s, 3.115 s, 3.310 s |
| GPU 帧耗时 | 0.203 s, 0.191 s, 0.189 s |
| CPU 中位耗时 | 3.152 s |
| GPU 中位耗时 | 0.191 s |
| 中位加速比 | 16.50× |
| CUDA 峰值分配显存 | 145,393,152 bytes（约 138.7 MiB） |

基准脚本会在计时后再次比较 CPU/GPU 传感器数组；数值不一致时直接失败。当前峰值远低于 8 GB 设备的 70% 预算，且未触发自动 OOM 分块重试。

小扫描在锁定环境中的采样姿态 GPU 渲染中位耗时为 0.0167 s，材料采样中位耗时为 0.00454 s，峰值显存为 37,673,984 bytes；包含绘图的总流程为 9.00 s。
