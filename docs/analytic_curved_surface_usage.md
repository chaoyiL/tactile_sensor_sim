# analytic 弯曲表面仿真

`simulate_curve.py` 支持沿 X 的等曲率圆弧、随边缘运动的灯带，以及沿局部法向的真实球面压入。CPU、普通 CUDA、编译照明内核和 CUDA Graph 使用相同几何语义。Zemax、Xense 和 LUT 标定目前仍只支持平面。

## 运行

从仓库根目录运行：

```powershell
# 校验参数和工作量，不初始化 GPU，不写入文件
.\.venv\Scripts\python.exe simulate_curve.py --radius-mm 100 --dry-run

# 压入图及同曲率、零压入的参考图
.\.venv\Scripts\python.exe simulate_curve.py --radius-mm 100 --reference --out results_curve_render/analytic/run_01

# 偏心压入，球从指定位置沿局部法向推进 1 mm
.\.venv\Scripts\python.exe simulate_curve.py --radius-mm 100 --center-uv 0.3 -0.2 --depth-mm 1 --out results_curve_render/analytic/run_02

# 相同流程的平面对照，可显式选 CPU
.\.venv\Scripts\python.exe simulate_curve.py --flat --device cpu --reference --out results_render/analytic/flat_01
```

默认配置为 `configs/analytic.json`，也可通过 `--config` 指定另一份 analytic 配置。CLI 覆盖配置值；`--flat` 与 `--radius-mm` 互斥。省略 `--out` 时，按覆盖后的实际曲率在仓库根目录下创建独立运行目录：

- 不弯曲（`radius_mm=null` 或 `--flat`）：`results_render/analytic/<UTC 时间和唯一标识>/`。
- 存在弯曲：`results_curve_render/analytic/<UTC 时间和唯一标识>/`。

普通入口 `simulate.py` 的默认目录同样为 `results_render/<backend>/<UTC 时间和唯一标识>/`，后端目录名为 `analytic`、`zemax` 或 `xense`。显式 `--out` 直接指定本次运行目录，相对路径以当前工作目录为基准，不再附加后端或运行标识。两个入口均要求指定目录不存在或为空。已有 `results/`、`results_curve/` 等历史结果保留原路径，不自动搬移。

`--reference` 只把局部压入深度设为零，保持弯曲半径、相机、灯带及曝光。它与取消全局弯曲的 `--flat` 不同。

## 配置

弯曲参数位于 `configs/analytic.json` 的 `surface.bending`：

```json
"bending": {
  "axis": "x",
  "radius_mm": null,
  "direction_z": -1
}
```

`radius_mm=null` 为精确平面，也是迁移后的默认值。修改为正数可直接运行 `simulate_curve.py`；或者保留 null，通过 `--radius-mm` 临时指定。首版只接受沿 X、向 −Z 弯曲。

`surface.size_mm=[W,H]` 是未压入表面的展开长度，W 为弧长而不是弦长。曲面中心保持在世界原点，Y 不弯曲。当前相机在 `(0,0,-60) mm`，光轴沿 +Z；两侧向相机弯曲，但相机位置、内参和曝光不自动调整。压入时相机也不追随凹坑。

`W=55.9 mm, R=100 mm` 时，边缘转角约 `16.0142°`，下沉约 `3.88065 mm`。圆弧支持范围为 `W/(2R)<π/2`，还必须满足完整接触区、灯带和相机的合法性要求。配置不合法时明确报错，不自动裁掉接触区或减小压入深度。

## 压入位置和深度

三个后端的平面配置以及 analytic 曲面配置统一使用：

```json
"indentation": {
  "radius_mm": 6.0,
  "depth_mm": 1.0,
  "center_uv": [0.3, -0.2],
  "direction_normal": -1
}
```

`center_uv=(u,v)` 位于 `[-1,1]²`，以未局部压入的基准曲面为参照。`(0,0)` 是中心；u 从左到右，v 从 top 到 bottom，对应 −Y 到 +Y。它不是图像像素坐标，也不是球心世界坐标。

归一化坐标对应展开毫米位置 `s=uW/2, t=vH/2`。弯曲时以 `α=s/R` 转换到世界位置：

```text
B = (R sin α, t, −2R sin²(α/2))
N = (sin α, 0, cos α)
```

N 是几何正向，中心朝 +Z。曲面压入使用 `direction_normal=-1`，推进方向为 `D=-N`；球心为 `C=B+(depth_mm-radius_mm)D`。球从相切位置沿 D 移动 `depth_mm`，并始终保持真实球形。接触区用球面替换基准圆柱面，区外不变。

接触区不是固定的平面圆盘。程序使用完整球面与圆柱交线的解析范围检查越界和分支连续性；不支持过深压入导致的折叠、回绕或跨越表面边缘。该模型是几何规定的压入，不求解弹性、膜张力或接触边缘平滑过渡。

## 灯带

灯带随全局弯曲运动，不跟随局部球压入。左右条仍沿 Y 保持直线，整体随边缘转动；上下条沿 X 成为弧形带。

- `offset_mm`：沿局部外向切线的边缘安装间隙。
- `z_mm`：沿基准几何法向 N 的有符号高度，负值位于光学侧。
- `width_mm`：沿 N 的完整带宽。
- `length_mm`、`point_spacing_mm`：灯带中心线自身的实际弧长和 LED 间距。
- `along_offset_mm`：灯带中心沿基准边缘的定位弧长。

发光面始终朝局部内侧，不自动瞄准原点。连续灯带按实际面积积分并保持各通道总功率；离散 RGB LED 保持各点功率和数量，宽度积分不会增加 LED。比有限曲面略长的灯带沿圆柱支架延伸，不截断端部。

## 输出

每帧保存：

| 文件 | 内容 |
|---|---|
| `image.png` | 固定曝光、响应矩阵和色调映射后的 sRGB 显示图 |
| `sensor_irradiance_rgb_w_m2.npy` | 传感器辐照度，RGB 三通道，W/m²，存储为 float32 |
| `radiance_rgb_w_m2_sr.npy` | 表面辐亮度，W/(m²·sr)，存储为 float32 |
| `geometry_gt.npz` | 相机有效像素、世界位置、法向、曲面坐标和接触区 |
| `surface.stl` | 按展开网格导出的曲面；图像渲染使用解析求交，不依赖此网格 |
| `metadata.json` | 本帧生效配置、球心、光源样本、模型版本、签名、代码哈希和执行模式 |

`--reference` 将帧放到 `indented/` 和 `reference/`。根目录 `run.json` 记录输入配置、覆盖参数、运行状态、各帧信息和实际耗时；失败会保留 `failed` 状态及错误，不覆盖后重跑。

几何真值的 `normal_world` 保持原有光学侧法向含义；`normal_geometric_world` 是几何正向。`surface_uv` 与 `base_position_mm`、`base_normal_world` 指向未压入基准曲面；`normal_displacement_mm` 是局部法向位移，`depth_mm` 是相机轴向深度。`contact_mask` 和 `surface_part` 区分接触区/基准面，无效像素的几何数值为 NaN。

GPU 默认首帧走普通 CUDA；复用同一上下文时，`compile_lighting=auto` 才触发编译。因此首次生成参考帧可能包含较长的编译准备时间。元数据分别记录编译、执行图及回退状态；后续复用的耗时不能与首次编译混为一谈。`required` 编译失败会报错，`auto` 回退时仍留在 CUDA 并明确报告。改变曲率、灯带或相机必须创建新上下文。

## 旧配置与标定兼容

旧平面的 `center_xy_mm=[x,y]` 仍可读取，对应 `u=2x/W, v=2y/H`；旧 `direction_z` 对应新 `direction_normal` 的同一符号。新旧位置字段或方向字段不能同时出现。有限曲率不接受旧 XY/世界 Z 输入，以防把展开坐标误当世界坐标。

原 `[1.5,-0.8] mm` 在 `55.9×21.8 mm` 表面上对应 `[0.053667262969588556,-0.07339449541284404]`，迁移没有移动原压入位置。Xense 的内部毫米通信协议保持不变，由公共适配器换算。

analytic 的针孔半宽和导出网格步长现分别位于 `camera.pinhole_half_width_mm`、`render.mesh_step_mm`。旧 analytic 配置同文件 `zemax` 段的这些值仍可兼容读取；新旧值冲突时报错，绝不读取另一后端的配置补值。

`simulate.py` 继续渲染平面；遇到非零曲率会提示使用新入口。平面 LUT 扫描保留原 `xy_step_mm`、完整姿态数量、序列与验证划分，同时在姿态记录中增加 `center_uv`。曲面 LUT 重标定和迁移评估现由独立入口 `python -m experiment.curve_lut_transfer` 提供，详见 [实验 README](../experiment/curve_lut_transfer/README.md)；通用平面标定与旧 NormDiff 入口保持原限制。

历史 manifest、配置快照、样本和 LUT 保留原格式及原状态。生成曲面结果必须重新渲染曲面和同曲率参考，不复用平面图像或 LUT。设计公式与实现阶段见 [实施计划](plan_analytic_curved_surface.md)，实际验证见 [验证记录](analytic_curved_surface_validation.md)。
