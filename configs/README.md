# 配置目录

所有可编辑的运行配置集中于此。路径在 JSON 中相对于配置文件所在目录解析；命令行相对路径相对于当前工作目录解析。

| 文件 | 用途 |
|---|---|
| `analytic.json` | 解析仿真的几何、灯带、相机和渲染参数 |
| `zemax.json` | Zemax 仿真的几何、灯带、相机、渲染和 OpticStudio 参数 |
| `xense.json` | Xense/MuJoCo 的胶体、球和无 marker 的 SimpleSensor 参数 |
| `calibration.json` | 三种后端共用的球扫描、材料网格、采样、LUT 拟合和留出验证参数 |
| `norm_diff.json` | 已有 LUT 的 NormDiff 和可选加噪 LUT 验证参数 |
| `lut_quality.json` | 已有 LUT 的多值歧义与实际查表质量参数 |
| `curve_lut_compensation.json` | 已知模型 `H_pt` 端点迁移、固定划分与指标，以及 v2 有限坡度非线性反演和计时参数 |
| `length_normdiff.json`、`length_lutqual.json` | 独立长度扫描实验的参数和固定物理配置快照 |

标定入口通过 `--backend` 选择一个仿真器，通过 `--calibration-config` 选择共用流程参数；模拟器配置不包含标定参数。检验入口通过 `--lut-results` 选择已有标定结果；`lut_quality` 也接受验证失败但样本和训练 LUT 完整的运行，并保留其失败状态。标定运行时保存的 manifest 包含实际生效的合并配置；后续检验以该快照和样本签名为准。`legacy/` 是重构前配置的原样归档，不作为新运行的默认配置。

`curve_lut_compensation.json` 的可选 `nonlinear` 块控制 GN 步数、阻尼/步长、闭式条件与计时重复数，由 `--compare-nonlinear --endpoint-results <run>` 启用比较；默认入口继续执行端点迁移。旧配置若包含退役的 `matrix` 字段会报错；旧运行续跑兼容由实验入口处理。

Xense 的 `mujoco.python_executable` 是相对于 `configs/xense.json` 的路径。Zemax 的 `zemax.install_dir`、`zemax.nethelper_dll` 依安装环境设置；这两个后端需要各自的外部运行环境。

## 曲面渲染与统一位置坐标

`analytic.json` 新增 `surface.bending`：`axis=x`、`radius_mm=null` 表示平面、`direction_z=-1` 表示两侧向 −Z 弯曲。运行 `simulate_curve.py --radius-mm 100 --reference` 可生成曲面及同曲率参考帧。三个仿真器的位置统一使用 `indentation.center_uv∈[-1,1]²`，方向使用 `direction_normal`。旧平面毫米坐标仍可读取，新旧字段冲突时报错。

analytic 的针孔半宽和导出网格步长分别位于 `camera.pinhole_half_width_mm`、`render.mesh_step_mm`；旧位置仅作同文件兼容读取。详见[曲面仿真验证结果](../docs/解析曲面仿真验证结果.md)。

省略 `--out` 时，无弯曲的渲染写入 `results_render/<backend>/<唯一运行标识>/`，有弯曲的渲染写入 `results_curve_render/analytic/<唯一运行标识>/`。目录分类依据 CLI 覆盖后的曲率，零压入参考不改变分类；显式输出路径仍然优先。
