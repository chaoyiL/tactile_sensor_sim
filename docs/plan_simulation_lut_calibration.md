# 仿真 LUT 标定实现计划

> Historical design record: commands and config paths below predate the repository refactor. See [the current workflow](repository_workflow.md).

本文记录 LUT 标定实现规格。2026-10-21 流程更新：第一遍保存完整歧义输入，后处理不再重渲染；歧义诊断仅通过显式 stride 控制采样，并加入上游的 branch classification / continuation。移除逐帧 color_difference / pq_mapping 输出及其显示缓存。具体命令见 `simulation_lut_calibration_usage.md`。

## 1. 目标、范围与必须遵守的约定

新增一个通过仿真自动采集颜色—坡度样本并标定 LUT 的入口：

1. 使用 analytic 模型渲染一次 `depth_mm=0` 的 reference，并在整个标定批次中冻结。
2. 按配置扫描球的压入深度和接触面上的位置。每个姿态都使用同一 analytic 模型渲染。
3. 用已知球面几何直接得到接触范围及坡度标签，替代手动圈选、边界球拟合及整体曲面重建。
4. 按 `recon_stable` 的 `calibrate/calibrate_norm.py` 调用链构建、验证、保存 signed RGB residual → 局部坡度 LUT。
5. 将 analytic 的密集渲染计算改为 GPU 执行；CPU 仅负责配置调度、上游原样 LUT 拟合、绘图和落盘。
6. 输出 LUT、必要样本及数值报告；每个姿态只保存一张圈出标定范围的色差预览；`visualization` 沿用上游图表。

以下约定直接作为实现决定，避免编码时自行改变需求：

- “与两侧边缘相切”指当前压入深度对应的**接触圆**与接触面边缘相切。接触圆半径不是球半径，且随深度变化。
- “左上角”指接触面材料坐标中的 `x` 最小、`y` 最小处；本项目默认相机中正好对应图像左上。相机旋转时扫描顺序仍以材料坐标为准。
- “一张色差图”按每个扫描姿态理解。`source_image` 字段和 verification 共用同一个预览文件，不分别保存原图、reference、副本或拼图。不同姿态各有自己的预览。
- `visualization` 是用户单独要求保留的诊断图集，包含上游的色差多面板及坡度多面板；它不受 verification 单图要求限制。
- 全局只有一张 `RGB → (p,q)` LUT；位置和深度提供采样覆盖，不把 LUT 改为带位置索引的多维表。
- 不实现 marker 检测、marker mask、人工标注界面、整体曲面拟合、Poisson 高度重建或逐帧整体曲面资产。
- 不改变物理模型、曝光、色调映射、光源采样密度来换取速度。GPU 化必须包含阴影射线求交，不能通过关闭阴影提速。
- 不对上游 LUT 算法另做“优化版”。尤其不能替换成颜色均值回归、单纯最近邻、神经网络、绝对色差或逐通道独立 LUT。

## 2. 已核对的参考实现

### 2.1 固定版本

调研日期：2026-10-21。本文依据参考仓库提交：

`e02f95f119aa42d5aaf1ebae1f2304d60f045e8c`

后续移植必须以此提交为基线，避免上游变动使本计划中的算法、字段和输出名称失配。[参考仓库固定版本](https://github.com/chaoyiL/recon_stable/tree/e02f95f119aa42d5aaf1ebae1f2304d60f045e8c)

### 2.2 实际调用链及移植边界

| 上游文件 / 函数 | 实际职责 | 本项目处理 |
| --- | --- | --- |
| `calibrate/calibrate_norm.py:main` | `prepare → annotate → samples → fit` 的入口，也支持外部样本拟合 | 用自动扫描入口替代采集和标注调度 |
| `calibrate/norm_pipeline.py:compute_frame_colors` | 在线性 RGB 中、同一材料网格上计算 observed-reference | 保留语义；投影与几何由仿真直接提供 |
| `fit_ball_center`、`make_sphere_labels` | 人工边界结合整体几何拟合球，射线求交后生成局部坡度 | 已知球心/半径/深度，直接解析计算 |
| `export_normal_samples` | 导出逐点配对颜色、坡度及图像分组 | 保留必要样本字段及分组，去除对重建缓存的依赖 |
| `utils/normal_mapping.py:build_normal_lut` | 分箱、MAD 剔除、角度 RMS 筛选、近邻填充和零点约束 | 原样移植，见第 9 节 |
| `NormalCalibration.save/load` | LUT NPZ 数据格式及校验 | 保留版本 5 及核心字段 |
| `lookup_slopes_raw` | LUT 三线性插值、混合方差和置信度 | 原样移植 |
| `_fit_and_validate_samples` | 整图分组留出验证，通过后用全部样本拟合正式 LUT | 保留划分和判定算法，改为接收仿真样本及内嵌坐标声明 |
| `calibrate/visualize_norm.py` | 全样本拟合、整图留出、LUT 切片、歧义图 | 保留全局图表，删除逐帧色差/坡度输出 |
| `calibrate/analyze_ambiguities.py` | 外围分支延续及主支持/多分支分类 | 复现 branch classification、branch continuation 图及数学规则 |

源码定位：

- [入口](https://github.com/chaoyiL/recon_stable/blob/e02f95f119aa42d5aaf1ebae1f2304d60f045e8c/calibrate/calibrate_norm.py)
- [pipeline](https://github.com/chaoyiL/recon_stable/blob/e02f95f119aa42d5aaf1ebae1f2304d60f045e8c/calibrate/norm_pipeline.py)
- [LUT 核心与序列化](https://github.com/chaoyiL/recon_stable/blob/e02f95f119aa42d5aaf1ebae1f2304d60f045e8c/utils/normal_mapping.py)
- [线性 RGB 与 signed difference](https://github.com/chaoyiL/recon_stable/blob/e02f95f119aa42d5aaf1ebae1f2304d60f045e8c/utils/color_difference.py)
- [法线/坡度转换](https://github.com/chaoyiL/recon_stable/blob/e02f95f119aa42d5aaf1ebae1f2304d60f045e8c/utils/surface_geometry.py)
- [双线性采样与展示光栅化](https://github.com/chaoyiL/recon_stable/blob/e02f95f119aa42d5aaf1ebae1f2304d60f045e8c/utils/surface_sampling.py)
- [可视化](https://github.com/chaoyiL/recon_stable/blob/e02f95f119aa42d5aaf1ebae1f2304d60f045e8c/calibrate/visualize_norm.py)

注意区分上游两种阈值：`maximum_rms_angle_degrees=5` 是单个 LUT 节点的保留门限；整图验证的配置示例为 `validation_rms_degrees=20`，而 settings 的缺省值是 `5`。本文显式使用 `20` 做批次验证，不把它误当成节点门限。

## 3. 当前代码与落点

### 3.1 当前需要处理的问题

- `simulate.py` 是现有入口，内部还包含 `load_config()`。
- `utils/geometry.py` 使用 NumPy 计算球冠、相机射线求交及灯带采样。
- `utils/renderer.py:render()` 在 CPU 上循环灯带和每个采样光源；每个光源又对所有可见点进行遮挡求交，是扫描的主要耗时路径。
- `encode_display()` 的顺序是传感器辐照度 → 相机响应矩阵 → 固定曝光 → tone map → sRGB → uint8。
- `save_result()` 默认输出几何 NPZ、辐照度 NPY、图像及 metadata，不能直接用于精简的标定输出。
- `load_config()` 目前要求接触圆严格位于接触面内部，会拒绝用户要求的相切位置。
- 当前示例为 `55.9 × 21.8 mm` 接触面、`640 × 480` 图像、球半径 `6 mm`、相机位于 `z=-60 mm`。
- 调研时本机检测到 `NVIDIA GeForce RTX 5050 Laptop GPU`，总显存约 `8151 MiB`；现有 `.venv` 尚未安装 PyTorch。硬件信息只是容量规划依据，不代表 GPU 代码已经验证。

### 3.2 目标目录

根目录只放可直接运行的入口；辅助实现继续放 `utils/`，全部测试继续放已 gitignore 的 `tests/`。

```text
calibrate_lut.py                  # 新主入口：参数解析与 main()
simulate.py                       # 现有入口；analytic 默认走 GPU
config.json              # 统一配置，包含物理、渲染、Zemax 和 LUT 标定参数
utils/
  config.py                      # 提取共享配置读取/校验
  geometry.py                    # 保留 CPU 几何参考与 STL 导出
  geometry_gpu.py                # CUDA 球冠/相机/光源/遮挡求交
  renderer.py                    # 公共 API、结果保存、设备派发
  renderer_cpu.py                # 原 NumPy 渲染器，作对照与显式 CPU 模式
  renderer_gpu.py                # 全 GPU analytic 渲染与可复用上下文
  calibration_scan.py            # 深度、x/y 轴、稳定姿态编号
  calibration_samples.py         # 材料网格、几何标签、配对色差
  calibration_ambiguity.py       # 完整输入保存、stride 采样和独立歧义分析
  ambiguity_patterns.py          # 外围支持计数、分类、半径敏感性和两类分支图
  ambiguity_patterns_compiled.py # Numba 精确树遍历与支持图像位集合
  calibration_pipeline.py        # reference/采样/拟合/验证/发布调度
  normal_mapping.py              # 上游 LUT 核心、NPZ 保存/读取/插值
  calibration_visualization.py   # 上游图表与仿真数据适配
tests/
  test_physics.py
  test_calibration_scan.py
  test_calibration_samples.py
  test_normal_mapping.py
  test_renderer_gpu.py
  test_calibration_pipeline.py
docs/
  plan_simulation_lut_calibration.md
```

`load_config()` 从入口移到 `utils/config.py`。`simulate.py` 导入该函数，现有外部 `from simulate import load_config` 可暂时继续使用这一导入别名；新模块不要反向依赖入口。

不要导入上游的整个 pipeline，它会连带引入 marker、曲面重建、JAX、Qt 和真实相机依赖。只移植 LUT 数值核心、必要数据结构和图表函数；法线/坡度转换放在 `normal_mapping.py` 或共享小模块中，不复制整套曲面工具。

## 4. 配置及命令行

### 4.1 配置示例

`config.json` 是仿真和 LUT 标定共用的完整配置，包含 `surface/indentation/lights/camera/render/zemax/calibration` 段。下方是其中的 `render/calibration` 部分：

```json
{
  "render": {
    "shadows": true,
    "exposure_scale": 6000.0,
    "tone_map": "reinhard",
    "device": "cuda:0",
    "geometry_dtype": "float64",
    "radiometry_dtype": "float32",
    "pixel_chunk_size": 16384,
    "light_chunk_size": 16,
    "cuda_graphs": true,
    "compile_lighting": "auto"
  },
  "calibration": {
    "output_dir": "results_lut",
    "scan": {
      "xy_step_mm": [2.0, 2.0],
      "depth_min_mm": 0.2,
      "depth_max_mm": 1.0,
      "depth_step_mm": 0.2,
      "include_endpoints": true
    },
    "material_grid": {
      "step_xy_mm": [0.1, 0.1]
    },
    "sampling": {
      "contact_margin_mm": 0.0,
      "minimum_reference_dot": 0.1,
      "min_samples_per_pose": 16
    },
    "lut": {
      "size": 64,
      "maximum_rms_angle_degrees": 5.0
    },
    "validation": {
      "fraction": 0.2,
      "rms_degrees": 20.0,
      "maximum_out_of_range_fraction": 0.05
    },
    "visualization": {
      "enabled": true,
      "difference_gain": 2.0,
      "color_distance_threshold": 0.01,
      "normal_angle_threshold_degrees": 20.0,
      "ambiguity_stride": 5,
      "branch_dominance_fraction": 0.8,
      "export_ambiguity_pairs": false
    }
  }
}
```

规则：

- 标定球半径唯一来源是 `indentation.radius_mm`；每个扫描姿态只覆盖 `indentation.center_xy_mm/depth_mm`。
- `indentation.direction_z` 沿用物理配置；它和 `surface.view_side_z` 是不同概念。
- 标定入口仅支持 analytic；不能把 `--backend zemax` 作为可选加速方式。
- `xy_step_mm`、材料网格间距和 GPU 分块是不同参数，不能复用一个变量。
- 所有步长/半径必须有限且 `>0`，`0<depth_min<=depth_max<R`。
- 每个深度都检查 `2*a(d)<=min(surface.size_mm)`，否则该深度无法完整放入接触圆，配置直接报错；不要静默截掉球冠。
- `size` 为整数且 `>=2`；所有验证比例、阈值沿用上游合法区间；不接受 NaN/Inf、布尔值冒充整数。
- `contact_margin_mm>=0` 且小于最小深度的接触半径。它仅是可选边缘排除，不改变渲染几何；默认 `0`，不额外缩小接触范围。
- `minimum_reference_dot` 本期固定为上游的 `0.1`；若字段存在则校验其值为 `0.1`，避免改变标签定义。
- 所有配置相对路径相对于配置文件所在目录。CLI `--out` 相对于当前工作目录并覆盖配置，写入 manifest 前转换为规范路径。
- `output_dir` 不进入输入配置签名；物理配置、扫描规则、坐标约定、颜色编码、数值精度进入签名。

### 4.2 入口约定

```powershell
# 仅生成扫描列表与资源估计；不渲染、不拟合、不写图。
uv run --cache-dir .uv-cache --extra gpu --extra calibration python calibrate_lut.py --config config.json --dry-run

# 完整标定。默认执行采样、整图验证、全样本拟合和可视化。
uv run --cache-dir .uv-cache --extra gpu --extra calibration --extra gpu-compile python calibrate_lut.py --config config.json --out results_lut

# 原入口继续支持，analytic 默认使用 GPU。
uv run --cache-dir .uv-cache --extra gpu python simulate.py --config config.json --backend analytic --out results_gpu

# CPU 仅作显式对照或开发验证。
uv run --cache-dir .uv-cache python simulate.py --config config.json --backend analytic --device cpu --out results_cpu
```

完整标定使用 `--config`、`--out`、`--dry-run`。独立离线诊断使用 `--ambiguity-only`，可选 `--samples` 指定已有 normal_samples 或 ambiguity_inputs 文件，`--ambiguity-stride` 覆盖诊断采样设置；此模式无需 CUDA、不会拟合 LUT。一般渲染入口使用 `--device` 覆盖配置。完整标定若没有可用 CUDA，明确失败。

## 5. 扫描几何与可重复顺序

### 5.1 接触半径及扫描域

设接触面宽高为 `W,H`，球半径为 `R`，当前深度为 `d`：

```text
a(d) = sqrt(2*R*d - d*d)
x_min = -W/2 + a(d)
x_max = +W/2 - a(d)
y_min = -H/2 + a(d)
y_max = +H/2 - a(d)
```

每个深度的第一个姿态是 `(x_min,y_min,d)`，其接触圆分别与左、上边相切。先从左到右扫完整行，再从上到下扫下一行；深度是最外层循环，由浅到深。不要蛇形扫描，也不要使用最大深度半径固定全部深度的扫描域。[本项目球冠定义](../utils/geometry.py)

### 5.2 端点规则

实现公用 `closed_axis(start, stop, step)`：

1. 在 float64 中计算 `n=floor((stop-start)/step + tolerance)`，生成 `start+k*step, k=0..n`。
2. 容差使用与轴尺度相关的量，例如 `1e-10 * max(1, abs(start), abs(stop), abs(step))`；整数除法判定使用相应无量纲容差。
3. 去除因浮点误差超过 `stop` 的值；与 `stop` 接近的末项直接设为 `stop`。
4. 默认 `include_endpoints=true`，若最后一项尚未到达 `stop`，追加一个精确 `stop`。最后一段可以短于步长。
5. 若 `start==stop`，只返回这一个点，不重复。
6. 显式设为 `false` 时只保留规则步长网格，不额外追加末端；深度轴、x/y 轴都遵守此规则。

禁止仅用 `np.arange(start, stop+step, step)`，它可能生成越界球心或超过最大深度的姿态。

每个姿态分配稳定编号与键，例如：

```text
pose_id = 0,1,2,...
image_key = "000000", "000001", ...  # 至少六位，补零宽度整个批次固定
depth_index, row_index, column_index
center_xy_mm, depth_mm, contact_radius_mm
```

宽度取 `max(6, len(str(total_pose_count-1)))`，保证键的字典序等于扫描顺序，也与上游按 image key 排序的验证划分一致。

使用第 4 节默认扫描参数、当前 `55.9 × 21.8 mm` 接触面，应得到下表，作为 dry-run 回归基准：

| 深度 mm | 接触半径 mm（约） | x 点数 | y 点数 | 姿态数 |
| --- | --- | --- | --- | --- |
| 0.2 | 1.536229 | 28 | 11 | 308 |
| 0.4 | 2.154066 | 27 | 10 | 270 |
| 0.6 | 2.615339 | 27 | 10 | 270 |
| 0.8 | 2.993326 | 26 | 9 | 234 |
| 1.0 | 3.316625 | 26 | 9 | 234 |
| 合计 | — | — | — | 1316 |

加一次 reference，采样阶段是 `1317` 次渲染；第 12 节的完整逐姿态可视化需要再流式渲染这些压入姿态，reference 仍复用，不重复标定采样。

### 5.3 相切校验

将配置校验拆为“通用物理校验”和“单姿态校验”。标定单姿态允许：

```text
abs(cx)+a <= W/2 + geometric_tolerance
abs(cy)+a <= H/2 + geometric_tolerance
```

`geometric_tolerance` 仅吸收舍入误差，不作为可见越界宽容。一般模拟入口可以保留严格内部规则；标定显式传 `allow_contact_tangent=True`，不要通过偷偷平移球心一个 epsilon 绕过相切要求。

`d=0` 的 reference 单独校验，不受扫描最小深度约束；它与接触球位置无关。深拷贝配置后设置零深度，不更改原始配置对象。

## 6. 材料坐标、精确范围和坡度标签

### 6.1 固定参考网格

建立覆盖平面矩形的规则材料网格，列沿 `+x`，行沿 `+y`。用第 5 节的闭区间规则和 `material_grid.step_xy_mm` 生成 x/y 坐标轴。

每个网格节点的永久身份是 `(x,y)`。参考位置与当前位置为：

```text
X0(x,y) = (x,y,0)
Xd(x,y) = (x,y,h(x,y))
st(x,y) = (x+W/2, y+H/2)  # 平面参考弧长坐标，单位 mm
```

这个固定 `(x,y)` 身份是本球冠高度场的材料对应假设；本期没有弹性体切向位移。所有姿态共用坐标轴、grid index、chart ID，不能按接触圆重新铺一套局部网格。

参考平面是平坦的，因此可保留上游声明：`coordinate_kind='reference_arclength'`、`frame_convention='st_grid_columns_rows'`、`residual_domain='material_grid'`、`color_space='linear_rgb'`。但不创建上游依赖的外置 `reference_coordinates.npz`；在样本/LUT metadata 中明确内嵌轴坐标及 chart ID。原上游整套重建消费者仍需额外适配，不宣称仅复制 LUT 就已接好它的重建工程。

### 6.2 解析高度、法线与局部坡度

令 `s=indentation.direction_z`，`dx=x-cx`、`dy=y-cy`、`r2=dx²+dy²`，接触圆内部：

```text
root = sqrt(R*R-r2)
h = s*(d-R+root)
hx = -s*dx/root
hy = -s*dy/root
```

接触圆外 `h=hx=hy=0`。判断球冠内部使用 `r2<a*a`，圆周上的高度属于平面，避免斜率不连续处的标签歧义。

LUT 使用固定参考材料框架：

```text
e1=(1,0,0), e2=(0,1,0), n0=(0,0,1)
n_geom=normalize((-hx,-hy,1))
p=-(n_geom·e1)/(n_geom·n0)=hx
q=-(n_geom·e2)/(n_geom·n0)=hy
```

这些坡度是 mm/mm，无量纲；不是 `dz/du`、`dz/dv`，不要乘像素焦距，也不要把 `p,q` 当作角度或法线 x/y 分量。

照明使用的法线为 `n_optical=view_side_z*n_geom`。标签和验证用 `n_geom`，不要把光学朝向为 `-z` 的法线直接送入以 `+z` 为参考的上游坡度转换；否则会把可见标签全判为背向。

如从渲染 GT 的 `normal_world` 获取标签，则先乘 `view_side_z` 恢复 `n_geom`，再转换；应与解析 `hx,hy` 一致。

球心为 `(cx,cy,s*(d-R))`。如记录上游 `sphere_side`：当前 `n0=+z` 下，`s=-1` 对应 `positive_normal`，`s=+1` 对应 `negative_normal`。边界模型记录 `zero_boundary_displacement`。

必须保留上游标签约束 `n_geom·n0>0.1`。不要另外添加 marker、纹理、色差阈值或人工“压入区域检测”。

### 6.3 几何范围和有效样本

精确接触域由 `r2<a²` 给出；可选训练内域由 `r2<(a-contact_margin_mm)²` 给出。默认两者相同。

有效样本至少同时满足：

1. 位于训练内域，所有几何/坡度有限，`n_geom·n0>0.1`。
2. reference/current 投影在图像内且相机前方，投影对应表面的光学朝向可见。
3. 从相机朝 `Xd` 发出的射线，其最近表面交点确为 `Xd`，排除被球冠/其他表面部分遮住的点。用 float64 比较，初始容差 `atol=1e-6 mm, rtol=1e-7`；以 CPU 对照验证后再确定最终阈值。
4. 双线性采样的四个图像支撑像素都属于有效可见表面，并且 current/reference 的三个 uint8 通道都 `<255`，与上游饱和剔除语义一致。
5. current/reference 采样结果及 signed difference 有限。

这里仍需要几何、可见性和有效颜色布尔数组；用户排除的是 marker 等人为 mask，不是取消数学有效性检查。不要将接触圆外的平面、受阴影影响的背景点混入球面标签，也不要对所有圆外像素强制补零坡度样本。

不足 `min_samples_per_pose` 的姿态写 `rejected_insufficient_samples` 和原因，其预览仍可保存，不能用空数据凑数。默认完整运行最终应至少有两个接受姿态，否则整图留出验证无法成立。

## 7. reference 与颜色链路：必须复现上游数据域

### 7.1 先渲染 reference

流程启动时创建一个 GPU 场景上下文；复制物理配置，仅把 `depth_mm` 设为 `0`，渲染一次 reference。将 reference 图像及线性解码图、有效像素、材料网格上的参考颜色保存在显存中，在两个扫描 pass 中共用。

相机、分辨率、灯带采样、反射率、响应矩阵、曝光和 tone map 在整个批次保持一致。签名记录这些设置，不能每个姿态归一化亮度或自动调整曝光。

### 7.2 使用与“读入渲染 PNG”一致的颜色

上游输入是 uint8 sRGB 图像，经逆 sRGB 得到线性 RGB，再采样和相减。因此仿真也采用该模式，而非直接使用 `sensor_irradiance` 相减：

```text
E_sensor (物理 RGB)
→ response_rgb_3x3 与 exposure_scale
→ tone map 与 [0,1] 裁剪
→ sRGB 编码
→ round(255*srgb) 得到 uint8
→ inverse_sRGB(uint8/255) 得到 image_linear_rgb
```

对于归一化 sRGB 值 `u`，逆变换为：

```text
u<=0.04045: u/12.92
otherwise: ((u+0.055)/1.055)^2.4
```

`linear_rgb` 在这里是**展示图经逆 sRGB 后的线性 RGB**；tone map 和 8 位量化已经包含在成像系统里，不等于 tone map 之前的物理辐照度。metadata 记录 `image_encoding='srgb8_roundtrip'` 和完整曝光配置。

所有图像在本项目内部统一为 RGB；上游 OpenCV 的 BGR 输入转换只移植语义。不得照抄 `[...,::-1]` 后把本来就是 RGB 的 Pillow/GPU 图像反转两遍。

### 7.3 在同一材料点取色

相机矩阵按现有 `camera_frame()`：其列是 camera→world 的右、下、前三轴。对世界点 `X`：

```text
Xc = (X-camera_position) @ camera_frame
u = fx*Xc.x/Xc.z + cx
v = fy*Xc.y/Xc.z + cy
```

分别投影 `X0`、`Xd`，获得 `uv0`、`uvd`：

```text
Cref(x,y) = bilinear(reference_linear_rgb, uv0)
Cobs(x,y) = bilinear(observed_linear_rgb, uvd)
delta(x,y) = Cobs(x,y)-Cref(x,y)
```

先逆 sRGB，再双线性采样；不能先对 uint8/sRGB 插值再解码。四邻域有效性按上游 `bilinear_sample()`，不能用“插值得到的 valid 值大于 0.5”替代四点全部有效。

即使正对平面的相机，压入后投影也会随高度变化；同像素直接相减不会得到同材料点色差。不得将 `observed_linear_rgb-reference_linear_rgb` 的普通图像差直接作为训练样本。[上游材料网格取色实现](https://github.com/chaoyiL/recon_stable/blob/e02f95f119aa42d5aaf1ebae1f2304d60f045e8c/calibrate/norm_pipeline.py#L337)

保留色差正负号；不能使用 `abs`、`cv2.absdiff`、截断到非负值、逐帧 min/max 缩放或 `0.5+gain*delta` 的预览值作为 LUT 输入。

## 8. analytic 全 GPU 改造

### 8.1 技术选型与依赖

使用原生 Windows 的 PyTorch CUDA tensor 实现。新代码不依赖上游 JAX/Qt；上游使用 JAX 不等于必须移植它的运行环境。

在 `pyproject.toml` 中新增：

- `gpu` extra：CUDA 版 `torch`。
- `calibration` extra：`scipy`、`matplotlib`；图像保存继续使用 Pillow。
- 将项目支持 Python 下界调整为与选用 PyTorch 相符的范围，目标 `>=3.10,<3.14`，保留当前 Python 3.13 的运行能力。
- 为 torch 配置独立、显式的官方 CUDA wheel index，保证 `uv sync --extra gpu --extra calibration --locked` 能重现相同构建，不通过临时 `uv pip install` 留下与 lock 不一致的环境。

实施时按 [PyTorch Windows 安装说明](https://pytorch.org/get-started/locally/) 选择支持本机 GPU 的稳定版 CUDA wheel，并将实际版本和 index 锁入 `uv.lock`。可先从官方 CUDA 12.8 wheel 系列检查，但不能只根据驱动存在就假设任意 wheel 支持该 GPU 架构。

安装后检查 `torch.cuda.is_available()`、GPU 名称、device capability、`torch.cuda.get_arch_list()`，再实际运行一个 CUDA 张量算子并 synchronize。以实际执行成功为准；当前计划不声称某个 torch/CUDA 组合已测试。[架构查询 API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_arch_list.html)

### 8.2 API 与设备边界

建议接口：

```python
class AnalyticGpuContext:
    # 固定相机、光源采样、材料网格、常量和显存缓冲区。
    ...

def render_tensors(context, pose):
    # 返回仍驻留 CUDA 的 sensor、必要的几何 GT 和成像结果。
    ...

def render(config, device=None):
    # 兼容现有 simulate.py 的 host 数组返回接口。
    # GPU 完整渲染结束后，才在此边界复制一次所需数组到 host。
    ...
```

标定必须直接调用 `render_tensors()`，不要每个姿态先调用 host 包装器再把数组搬回 GPU。NumPy CPU 渲染器以显式设备模式保留，用于数值基准；默认 analytic 设备为 CUDA，CUDA 不可用则报错。

公共派发模块延迟导入 `renderer_gpu`/torch；仅 CPU 测试、显式 CPU 运行和 Zemax 入口不应因尚未安装 `gpu` extra 而在 import 阶段失败。GPU 路径缺依赖时给出对应 extra 的安装说明。

`simulate.py` 的 `save_result()` 可以继续保留现有模拟输出；标定入口使用独立轻量 writer，绝不调用 `export_stl()` 或通用 `save_result()`。

### 8.3 必须驻留 GPU 的运算

| 运算 | 实现要求 |
| --- | --- |
| 相机像素网格、方向归一化、视角因子 | 初始化时在 GPU 缓存 |
| 相机与有限平面/球冠最近交点 | `geometry_gpu.py`，保留两球根、有限矩形、球冠侧约束 |
| 高度与光学法线 | CUDA tensor，单位、符号与 CPU 一致 |
| 灯带位置、发射坐标系及采样 | 固定 GPU 常量；采样位置必须与现有中点积分完全一致 |
| 距离衰减、余弦项、光谱功率求和 | 像素块 × 光源块广播、归约 |
| 每条阴影射线的最近交点 | CUDA；不能转 NumPy/scipy 循环 |
| BRDF、辐亮度、针孔通光与 cos⁴ | CUDA，保留原单位换算 |
| 响应矩阵、tone map、sRGB、量化、逆 sRGB | CUDA |
| 材料点投影、双线性采样、接触域/坡度、样本筛选 | CUDA |

允许在 CPU 做少量配置校验和扫描列表生成。每个姿态只将最终接受的紧凑样本、少量计数和需要写出的预览传回 CPU；允许为 CPU 绘图传输诊断数组，禁止在逐光源/逐像素热循环中 `.cpu()`、`.numpy()` 或逐个 `.item()`。

### 8.4 保留光学方程

把 CPU 公式逐项改写为 tensor，不省略遮挡：

```text
v = light_sample - surface_point
d2 = dot(v,v)
l = v/sqrt(d2)
ci = max(dot(n_optical,l),0)
ce = max(-dot(l,emission_normal),0)
visible = nearest_hit(point+n_optical*1e-6,l) >= sqrt(d2)-1e-5
factor = (p_light+1)/(2*pi) * ce^p_light * (ce>0) * ci * visible/d2
surface_irradiance += factor * strip_power_rgb/strip_sample_count
surface_radiance = surface_irradiance * reflectance_rgb/pi
L_per_m2 = surface_radiance * 1e6
sensor = L_per_m2 * (2*pinhole_half_width_mm)^2/focal_length_mm^2 * cos_view^4
```

`p_light` 是灯带余弦指数，不能与坡度 `p` 混名。无效射线要先换成安全占位值再算乘法，不能依赖 `0*Inf` 或 `where` 自动消除 NaN。

### 8.5 精度与分块

- 几何坐标、射线求交、root 判别、接触边界和可见性默认 float64；照度累加、展示 RGB 默认 float32。
- 保留现有 `1e-6 mm` 自遮挡偏移和 `1e-5 mm` 可见性比较；不要直接在 float32 大世界坐标上使用这些值，舍入可能让偏移失效并造成自阴影。
- 先做纯 float64 GPU 对照，再启用上述几何/辐射混合精度。禁用 autograd 与半精度 autocast；不引入 FP16/BF16，不允许未经误差核验的 TF32 快捷计算。[数值精度注意事项](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- 用 `torch.inference_mode()` 包围渲染。Python 循环只控制灯带块/光源块/像素块，单个块内部全部广播计算。
- 起始 `pixel_chunk_size=16384`、`light_chunk_size=16`，默认启用 `cuda_graphs`；主要中间张量为 `[P,L,3]`，不要构造 `[全部姿态,H,W,全部光源,3]`。扫描复用上下文时，从第二帧开始捕获固定大小光照/阴影块，以设备张量更新球参数；尾块补零、仅复制有效输出。完整实现和复测命令见 [GPU 渲染优化](gpu_renderer_optimization.md)。
- 固定姿态一次只处理一帧。光源可以合并为点、normal、power、exponent 四组 tensor，仍保留每个 strip 的采样权重。
- OOM 时释放当前块临时量并将像素块减半，重试当前姿态；到最小可执行块仍失败则报错。不能悄悄降采样光源、降低图像分辨率或退回 CPU。
- 整个扫描仅初始化一次 CUDA 上下文，缓存固定射线和 reference。不要每帧 `empty_cache()`，不要每个深度重新编译/分配全套常量。
- `grid_sample` 可用于 bilinear 取色，但必须显式设 `mode='bilinear', align_corners=True`，并用 `2*u/(W-1)-1`、`2*v/(H-1)-1` 转坐标；四邻有效性单独 gather 判定。也可直接以 CUDA gather 实现上游四点公式，便于逐值对照。[grid_sample 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)
- GPU 计时在起止点同步或使用 CUDA event；不能把异步 kernel 提交时间当作渲染耗时。

## 9. LUT 核心：逐项保持上游算法

本节是必须照搬的算法契约，来源于固定版本的 [`build_normal_lut()` 与 `lookup_slopes_raw()`](https://github.com/chaoyiL/recon_stable/blob/e02f95f119aa42d5aaf1ebae1f2304d60f045e8c/utils/normal_mapping.py)。优先直接移植这两个函数及 `NormalCalibration`，仅移除不需要的导入。拟合在 CPU/NumPy/SciPy 执行，不纳入“渲染全 GPU”的改造范围，以免改变 median、排序和 KDTree 行为。

### 9.1 输入与颜色坐标

- `colors: [N,3]` 为 signed linear RGB residual，`slopes: [N,2]` 为逐点对应的 `p,q`。
- 函数内部转 float64，去除任一颜色/坡度分量非有限的整对样本；正常 pipeline 应在导出前已保证全部有限。
- 颜色范围：`color_min=min(channel_min,0)`、`color_max=max(channel_max,0)`，逐通道包含零点。任一通道跨度 `<=0` 则失败，不能填一个 epsilon 继续。
- 节点坐标：`rint(clip((color-color_min)/(color_max-color_min),0,1)*(S-1))`。
- `rint` 采用 NumPy 的最近偶数舍入规则；不能换成 floor、`+0.5` 后取整或整数截断。
- 轴顺序为 `[R,G,B]`，C-order 展平为 `((r*S)+g)*S+b`；按展平节点 ID 排序并分组。

### 9.2 节点鲁棒估计

对每个非空颜色节点：

1. 将该节点的坡度逐个转为 `normalize((-p,-q,1))`。
2. 将法线求和并归一化，得到初始平均法线。归一化分母下限 `1e-12`。
3. 计算各法线与初始平均法线的夹角，单位弧度，dot 先 clip 到 `[-1,1]`。
4. `m=median(angles)`；`MAD=1.4826*median(abs(angles-m))`。
5. 保留 `angle<=m+3*MAD+1e-12` 的样本。
6. 使用保留法线重新求和归一化，计算与新平均法线的角度 RMS。
7. 若 RMS **大于** `deg2rad(maximum_rms_angle_degrees)`，整节点不保留；等于门限时保留。
8. 保留节点的 LUT 值为保留坡度的算术均值，**不是**从平均法线反算的坡度。
9. 方差为 `mean((p-p_mean)^2+(q-q_mean)^2)`，是单个标量，不是两通道方差数组。
10. `sample_counts` 记录 MAD 后的保留数，`original_valid=True`；原始样本数另写报告，不覆盖这个定义。

上游没有“至少若干样本才保留节点”的规则：单样本节点可保留，方差为零。不要在移植时自行添加最小节点样本数。

### 9.3 空节点填充

- 至少有一个有效节点，否则失败。
- `sigma_ref2=max(median(variance[original_valid]),1e-8)`，在填充和零点覆盖之前计算。
- 有效节点坐标除以 `S-1` 后建 `scipy.spatial.cKDTree`，距离是归一化颜色立方体中的欧氏距离。
- 对所有无效节点找 `k=min(8,有效节点数)` 个最近有效节点；每批最多 `65536` 个待补节点。
- 权重 `w_i=1/max(distance_i,1e-12)^2`，随后归一化到和为一。
- 填充坡度 `mu=sum(w_i*mu_i)`。
- 填充方差 `v=sum(w_i*(v_i+||mu_i-mu||²))`，包含邻居均值间分歧。
- 填充完成不改变 `original_valid`；这些位置的 `sample_counts` 继续是零。

不得把填充改成单最近邻，也不能在计算新方差时只插值旧方差。

### 9.4 零色差约束

求零色差的连续 LUT 坐标，取各轴 floor 和相邻上界，覆盖其最多八个角节点：

```text
slopes[corner]=(0,0)
variances[corner]=sigma_ref2
```

不修改这些节点的 `original_valid`、`sample_counts`。这个覆盖步骤保留在原算法位置，也意味着节点值可能不再等于该节点原先的均值。

不要引入上游下游重建模块的 `zero_color_protection` 内/外半径、正负门限或平滑衰减；本期查询和所有图表都使用 raw LUT，只有这里规定的节点零点约束。

### 9.5 查询与不确定性

对有限输入：归一化并 clip 到 LUT 范围，取八个角节点进行三线性插值，得到 `mu`。

```text
variance = sum(w_i*(node_variance_i+||node_mu_i-mu||²))
confidence = 1/(1+variance/sigma_ref2)
```

越界色差仍 clip 查询，但另行记录是否越界；不能把 clip 后不越界的结果用来统计越界率。非有限输入或显式无效 mask 对应的坡度/方差/置信度都返回零。

查询计算及 float32 输出行为保留上游实现。训练阶段和验证阶段使用相同函数。

## 10. 样本组织与整图留出验证

### 10.1 紧凑样本

每个姿态的接受样本按固定材料网格的 row-major 顺序输出；不同姿态按接受顺序连接。记录原始 `pose_id`，另用从 `0` 连续编号的 `image_index` 对应接受姿态列表。拒绝姿态不占 `image_index`，但仍在 manifest/report 中。

建议 `normal_samples.npz` 字段：

| 字段 | 类型/形状 | 说明 |
| --- | --- | --- |
| `format_version` | int32 scalar，1 | 样本格式 |
| `color_residual_samples` | float32 `[N,3]` | signed linear RGB |
| `slope_samples` | float32 `[N,2]` | `p,q` |
| `image_index` | int32 `[N]` | 接受姿态连续编号 |
| `image_paths` | Unicode `[M]` | 稳定 image key，沿用上游字段名，不冒充不存在的原图路径 |
| `material_coordinates` | float64 `[N,2]` | 材料 `(x,y)` mm |
| `st_samples` | float32 `[N,2]` | 参考平面弧长坐标 |
| `material_grid_index` | int32 `[N,2]` | 网格 row/column，用于再次渲染核验与绘图 |
| `reference_normal_samples` | float32 `[N,3]` | 相同的 `(0,0,1)`，保留上游配对字段 |
| `tangent_frame_samples` | float32 `[N,3,2]` | 相同的 `[e1,e2]` |
| `material_grid_x_mm/y_mm` | float64 一维轴 | 内嵌参考 chart，不是整体曲面重建结果 |
| `metadata` | Unicode JSON scalar | 物理签名、颜色/坐标声明、pose 对应表等 |

公共声明可以同时保存为顶层 Unicode scalar，便于移植上游可视化校验。`reference_image` 使用明确的内存引用，例如 `analytic_reference:<run_signature>`，并记录 `reference_storage='rendered_once_in_memory'`，不填写不存在的 PNG 路径。

`coordinate_chart_file` 不保留虚假路径，替换为 `coordinate_chart_storage='embedded_flat_axes'`。改造 grouped validator 的输入校验，不能直接调用需要外置曲面/坐标 NPZ 的原 `_fit_and_validate_samples()`。

压缩 NPZ 只保存这些实际拟合样本及小型坐标信息，不包含每帧 H×W 的 XYZ、depth、法线、渲染照度或整体曲面。

采样可以逐姿态将紧凑数组拷到 host。写入前预估总样本与主机内存；数据较大时使用可清理的临时分块或 memmap 汇总，严禁把所有帧完整图像/GT 常驻显存。原 LUT 拟合要准确取 median/MAD，因此不能为了流式处理只累计均值和方差而丢弃原样本。

### 10.2 分组划分：与上游一致

1. 取真正有样本的 image index，按其 `image_paths` 键排序。
2. 使用 `np.random.default_rng(42)` 打乱该 index 列表，种子本期固定为 `42`。
3. `held_count=min(M-1,max(1,ceil(validation_fraction*M)))`。
4. 打乱列表前 `held_count` 张整图为验证集，其余整图为训练集。
5. 只用训练图样本构建 `normal_lut_train.npz`。
6. 验证图中的每一个样本必须完全未参与这张训练 LUT 的颜色范围、MAD、填充和零点约束计算。

不能随机拆同一张图的像素，也不能先用全样本算颜色范围后再声称留出验证。多深度扫描的相关性仍然存在；这里沿用上游的整图验证定义，不把它描述为按空间位置完全独立的泛化试验。

### 10.3 误差与判定

对 held-out 色差调用**训练 LUT** 的 `lookup_slopes_raw()`；用样本的共同参考框架将预测/真值坡度转成法线，得到角度误差。

每张验证图报告：角度 RMS、median、p95、色差越界样本比例、样本数，以及：

```text
passed = (rms_degrees<=validation.rms_degrees)
         and (out_of_range_fraction<=validation.maximum_out_of_range_fraction)
```

仅当所有 held-out 图都通过时，批次才通过。全局同时记录 median、p95、RMS 和越界率，不能只用总体均值替代逐图门限。

- **通过**：再使用全部接受样本重新构建正式 `normal_lut.npz`，记录 `validation_status='passed'`、划分及报告。
- **失败**：保存训练 LUT、`verification/validation.json` 和诊断；不发布/覆盖正式 LUT。需要失败可视化时可构建 `normal_lut_diagnostic.npz`，显式声明 `validation_status='failed'`，图中标明未发布。
- 若旧正式 LUT 已存在且本次失败，保留旧文件，当前运行报告明确失败，避免旧文件被误认为新输出。
- 可视化生成失败不反向改写已通过的数值验证状态；独立记录 `visualization_status` 和错误。

失败诊断的触发规则固定为：当 `visualization.enabled=true` 且训练 LUT 已成功构建但 held-out 判定失败时，用全部样本构建一次诊断 LUT 并保存到上述 diagnostic 文件，再绘制失败图表；可视化关闭时不额外构建诊断 LUT。若拟合本身因零跨度/无有效节点失败，则只写原因和已能生成的预览，不伪造诊断 LUT。

## 11. 输出白名单与 NPZ 兼容约定

### 11.1 成功运行的默认输出

```text
results_lut/
  normal_lut.npz
  normal_lut_train.npz
  normal_samples.npz
  ambiguity_inputs.npz           # 完整未采样的歧义输入，在拟合前写入
  manifest.json
  verification/
    report.csv
    validation.json
    000000.png                 # 单张 signed 色差预览 + 标定范围圈线
    000001.png
    ...
  visualization/
    summary.json
    calibration_summary.png
    color_pq_mapping.png
    lut_slices.png
    pq_ambiguities.png
    ambiguity_summary.json
    ambiguity_analysis.npz       # 分析子集、原始索引及端点 mask
    ambiguity_patterns/
      analysis.json
      branch_classification.png
      branch_continuation.png
      branch_classification.npz
```

manifest 中每个姿态的 `source_image`、`verification_image` 指向**同一个** `verification/<pose_key>.png`，并声明它是诊断色差预览，不能再作为原始 sRGB 输入反读来拟合。LUT 拟合输入只能来自内存采样或 `normal_samples.npz`。

默认不保存 `reference.png`、逐姿态原始 observation PNG、逐帧 JSON、STL、irradiance NPY、`frames/*.npz`、`overall/*.npz`、`prepared/*.npz`、`reference_surface.npz`、`reference_coordinates.npz`、`lut_material_grid.npz`，不建立整套 `assets/` 镜像。

### 11.2 正式 LUT 字段

沿用上游 `NormalCalibration.save()`，所有文件可用 `np.load(...,allow_pickle=False)` 加载：

| 字段 | 类型/形状 | 约束 |
| --- | --- | --- |
| `format_version` | int32 scalar | `5` |
| `color_residual_mode` | Unicode scalar | `signed` |
| `slopes` | float32 `[S,S,S,2]` | 所有节点有限 |
| `variances` | float32 `[S,S,S]` | 有限、非负 |
| `color_min` / `color_max` | float32 `[3]` | 每通道含零且 max>min |
| `sigma_ref2` | float32 scalar | 严格正 |
| `original_valid` | bool `[S,S,S]` | 填充前支持节点 |
| `sample_counts` | int32 `[S,S,S]` | MAD 后保留数；空节点仍为零 |
| `metadata` | Unicode JSON scalar | 见下文 |
| `total_sample_count` | integer scalar | 与输入紧凑样本数一致 |

metadata 至少包含：上游提交、仿真配置签名、渲染设备/库版本/精度、sphere 参数、扫描参数及姿态映射、参考颜色来源、RGB/signed/材料坐标声明、chart ID、固定局部框架、验证划分及报告、最终样本数量、原始支持节点数、填充比例、拒绝姿态原因统计。

图表使用的元数据不能依赖尚未生成的曲面文件。不要为“兼容”写出有路径但无实际文件的 `coordinate_chart_file`。

### 11.3 manifest 和姿态报告

`manifest.json` 保存一次完整有效配置及签名、参考渲染记录、全部扫描姿态、接受/拒绝状态、`pose_id ↔ image_index`、source/verification 路径及总体性能统计。

`report.csv` 每姿态一行，至少有：`pose_id,image_key,depth_index,row_index,column_index,cx_mm,cy_mm,depth_mm,contact_radius_mm,status,sample_count,rejected_reason,gpu_render_ms,sampling_ms,preview_file`。当前未执行的项目以明确空值表示，不伪造为通过。

NPZ 通过临时文件写完后替换正式文件；整个运行先绑定 manifest 签名，禁止把不同配置的样本和 LUT 混到一个可视化报告中。

## 12. verification 单图与 visualization 适配

### 12.1 verification 只画一张色差图

每个姿态的单图必须由第 7 节的材料配对色差生成，显示规则沿用上游：

```text
preview_rgb = round(255*clip(0.5+difference_gain*delta,0,1))
```

零色差为中灰；无效投影为黑色。先把材料网格的 residual 光栅化到 current 图像位置，或按当前像素交点的 `(x,y)` 反查 reference 色彩，保证与训练的材料对应一致；预览插值只用于显示，不重新生成训练数据。

接触圆是世界平面上的真实边界，投影后未必是图像中的圆。取至少 `360` 个边界点：

```text
theta = 0..2*pi
B(theta)=(cx+a*cos(theta), cy+a*sin(theta), 0)
```

投影成 polyline 叠加到图像上。默认绿色/亮黄色线，宽 `2 px`；如 `contact_margin_mm>0`，再画实际训练内域的投影轮廓，使用不同颜色，并在图中说明哪个轮廓参与拟合。

对训练内域轮廓，世界点应使用 `h(x,y)`，不能也硬放在 `z=0`；最稳妥的是投影 sample_valid mask 的边界或内域解析曲线。可见性导致的无效部分也应明确显示，不能让完整圆线暗示所有圈内像素都已接受。

图上附 `pose_id, center_xy_mm, d, a, sample_count, status` 简洁文字。不要做 current/reference/normal/height 拼图，不画整体曲面或输出其 NPZ。Pillow 绘制即可，无需 OpenCV 依赖。

### 12.2 visualization 保留的图表

按固定版本 `visualize_norm.py` 移植布局、轴方向、色表、标题意义与统计口径：

| 文件 | 必须保留的内容 |
| --- | --- |
| `calibration_summary.png` | 2×3：全样本拟合 p/q 对比、法线角度直方图；训练 LUT 的 held-out p/q 对比及逐图误差 |
| `color_pq_mapping.png` | 两个三维 signed RGB 点云，分别按 raw LUT p、q 着色；最多 6000 个等序抽样点仅用于绘制 |
| `lut_slices.png` | 2×3：逐一将 R/G/B 色差固定为零后的 p/q 切片；包含补全节点 |
| `pq_ambiguities.png` | p/q 平面中红色星号标记有歧义的原始样本；q 轴向下、等比例；需要时加中央 `[-2,2]` 视图 |
| `ambiguity_patterns/branch_classification.png` | RGB 投影、p/q 分类、外围半径敏感性三面板，与上游同样颜色和轴方向 |
| `ambiguity_patterns/branch_continuation.png` | RGB 三维和 p/q 支持点、锚点星号，展示两支/一支外围延续 |
| `summary.json` | 支持节点、保留样本数、全样本拟合/held-out 指标、歧义统计及分支图路径 |

绘图沿用 Matplotlib Agg，不弹交互窗口；缺失区域为黑色。坡度使用对称 `coolwarm`，模长 `viridis`，法线角误差 `magma`；全部姿态共享同一 slope/color limits。

上游 summary 的绘图抽样上限分别是 mapping 点云 `6000`、p/q 散点 `8000`，误差直方图 `50 bins`，保存 DPI `140`。训练、留出验证和拟合误差使用全部有效样本；歧义诊断可独立采样，必须声明统计范围。

所有 held-out 图表都必须从 `normal_lut_train.npz` 查询；正式全样本 LUT 的图只能标为 calibration fit，不能标为留出验证结果。p/q 图保持 raw LUT，不套用零色差保护或额外后处理。

verification 文件名统一使用原扫描 `image_key`；样本的连续 `image_index` 仅用于分组，通过 manifest 映射到该 key。分支分类 NPZ 的 `source_sample_index` 对应完整输入中的原始样本。

### 12.3 移除逐帧显示与几何依赖

删除 visualization 中的 color_difference 和 pq_mapping 目录创建、逐帧 LUT 查询、面板绘图、逐图路径记录及显示缓存写入。上游 `_frame_paths()` / `_project_frame()` 的 frames/overall 读取路径均不移植。

全部全局图和分支图只使用第一遍的配对 RGB、坡度、image_index、image_paths；不需要 image-space residual、逐帧 UV 或几何缓存。verification 中带接触圈线的单图继续在第一遍生成。总渲染次数固定为 `1 + pose_count`，`visualization_render_count=0`。

### 12.4 歧义统计必须可扩展且不改变定义

歧义定义沿用上游 `find_ambiguous_color_pairs()`：任意无序对 `i<j`，signed linear RGB 欧氏距离 `<=0.01`，且两个 `normalize((-p,-q,1))` 的夹角严格 `>20°`。同图与跨图都计入，两个端点都标为 ambiguous，使用 MAD 之前的原始配对样本。

第一遍扫描完成、拟合之前，将全部原始色差、坡度、image_index、image_paths、运行签名及元数据写入 `ambiguity_inputs.npz`。所有输入文件保留完整原始数据；stride 不应用于保存、训练或留出验证。

分析时对每个姿态按有效样本顺序取 `0,stride,2*stride,...`，当前示例 stride=5；省略配置字段时回退到 100。保留所有选中的点，不设置总样本数上限、不进行二次抽样；stride=1 恢复完整统计。报告记录采样方法 `per_image_stride`、原始/分析样本数、姿态覆盖、保留比例及 `count_scope`；采样结果仅是所选集合内的精确计数，不得冒充全量结果或直接放大推算。

对选定集合使用精确 KD-tree，`eps=0`，仅枚举 `j>i`，统计同图/跨图、pairs 及两端 mask。查询批大小随分析点数调整，使一批最坏情况下约不超过 262144 个邻域项（单行例外）；每约 5 秒输出进度。默认不导出 pair 明细，仅保存分析子集、原始样本索引、mask、报告和 PNG。显式开启 export 时，pair 索引对应分析子集，原始索引由 `ambiguity_analysis.npz` 中的 `source_sample_index` 映射。

### 12.5 branch classification 与 continuation

按固定提交 `calibrate/analyze_ambiguities.py` 的外围支持规则，对每个冲突对在 RGB 中点的 `2r/3r/4r` 球内查询；严格排除任意一个原始 r 球内的点及锚点。分别统计与两个锚点法线夹角 `<= angle_threshold/2` 的支持点、不同来源图数和其他图支持数。不设置最小内圈/外圈样本数。

主分类使用 `3r`。支持总数为零标 no_outer_support；某支支持数占总数至少 `branch_dominance_fraction=0.8` 标 main_support；其余标 multiple_branches。点级使用所有冲突对的最大类别编码，多分支重叠优先，唯一端点只计一次。报告保留不同半径稳定性、0.7/0.8/0.9 阈值敏感性、点角色和按图数计数的分歧；不删除训练样本。

优化只能改变执行顺序：按锚点复用颜色/法线候选，利用六维包围盒保证不漏掉真实候选，再严格做颜色欧氏距离和角度检查；对每个伙伴分块计算外围球和两个内圈排除。所有冲突对均处理，不再次抽样。冲突对、支持计数暂存磁盘映射数组，结束后清除；图和点级 NPZ/JSON 保留。

大规模集合使用 Numba 的并行精确树遍历，树节点保存颜色/法线包围盒和来源图像位集合。节点全部满足外圈、法线锥且在两个内圈之外时，直接累加节点点数并合并位集合；无法整体判定的节点继续下降，叶节点精确检查。边界剪枝保留浮点余量，最终逐点门槛不改变。不启用 fastmath；支持点数和唯一图数必须与上游 compare_branch_context 完全一致。Numba 随 calibration extra 安装并锁定。

两张图沿用上游版式、颜色、q 轴向下和 DPI=160。continuation 优先选择两端内圈各两点的上游实例；不存在时使用有支撑的稠密实例，并展示真实内圈计数。完全没有支持实例时仍生成说明图。

## 13. 主流程伪代码与函数边界

以下是调度说明，不是可以直接运行的实现；其中 LUT 数值函数以第 9 节的移植版本为准。

```python
def run_calibration(config_path, output_override=None, dry_run=False):
    cfg, settings = load_and_validate_calibration_config(config_path)
    poses = build_scan_poses(cfg, settings)
    grid = build_fixed_material_grid(cfg, settings)
    describe_scan_and_memory(poses, grid)
    if dry_run:
        return

    context = AnalyticGpuContext(cfg, grid)
    require_actual_cuda_execution(context)
    reference = render_zero_depth_reference_once(context)
    rows, compact_samples = [], SampleCollector()

    for pose in poses:
        gpu_frame = render_tensors(context, pose)
        paired = make_material_color_and_slope_samples(
            context, reference, gpu_frame, pose, grid)
        row = classify_pose_and_record_timing(pose, paired)
        save_one_annotated_difference_preview(row, paired)
        if row.accepted:
            compact_samples.append_paired_arrays_to_host(paired, pose)
        rows.append(row)
        release_pose_temporaries()

    samples = compact_samples.finalize_in_scan_order()
    save_normal_samples_and_manifest(samples, rows, cfg, settings)
    save_complete_ambiguity_inputs(samples)
    train_ids, validation_ids = upstream_group_split(samples, seed=42)
    training_lut = build_normal_lut(training_samples(samples, train_ids))
    save_training_lut(training_lut)
    report = validate_whole_images(training_lut, samples, validation_ids)
    save_validation_report(report)

    if report.passed:
        final_lut = build_normal_lut(all_samples(samples))
        publish_formal_lut(final_lut, report)
    else:
        final_lut = (build_failed_diagnostic(samples, report)
                     if settings.visualization.enabled else None)

    if settings.visualization.enabled and final_lut is not None:
        save_global_visualization_figures(samples, final_lut, training_lut)
        save_branch_classification_and_continuation(samples)

    finalize_manifest_status_and_timings()
    if not report.passed:
        raise CalibrationValidationFailed(report)
```

代码 API 至少明确下列边界：

| 函数/结构 | 输入 | 输出 |
| --- | --- | --- |
| `build_scan_poses` | 已校验物理配置、scan settings | 稳定 `ScanPose` 列表 |
| `build_fixed_material_grid` | W/H、step | 固定 axes、XY/ST、chart ID |
| `AnalyticGpuContext` | 场景配置、网格 | GPU 固定常量和缓冲区 |
| `render_tensors` | context、pose | CUDA sensor、必要 GT、图像 |
| `make_material_color_and_slope_samples` | reference/current、pose/grid | 全网格临时色差、标签、valid 与紧凑样本 |
| `build_normal_lut` | CPU colors/slopes、上游两个超参 | `NormalCalibration` |
| `validate_whole_images` | train LUT、sample groups | 原始 held-out 查询结果与逐图报告 |
| `compute_branch_support` | 配对颜色/法线/来源、全部冲突对、外围半径 | 与上游精确一致的外围支持与来源图计数 |
| `run_ambiguity_analysis` | 第一遍样本文件、stride、阈值、输出目录 | 独立报告/NPZ/PNG，渲染次数 0 |

错误必须带可定位原因：无 CUDA、球冠放不下、投影不可见、饱和/有效性导致无样本、颜色通道零跨度、无有效 LUT 节点、验证未通过、可视化失败。不要 catch 全部异常后仍返回成功。

## 14. 验证计划与验收标准

全部新测试放 `tests/`，继续保持 `.gitignore` 中的 `tests/`。只对实际数值、几何、数据泄漏和输出合同写测试，不写仅重复函数调用的测试。

### 14.1 扫描

- 当前配置和第 4 节默认参数严格得到 `1316` 个姿态；每个深度首点与左/上边相切。
- 最后 x/y 点与右/下边相切；普通段步长正确，只有追加末段允许更短。
- 整除、不整除、start==stop、极小浮点误差等轴生成情况，无重复、无越界。
- 所有姿态满足 `abs(center)+a<=half_size`，包含临界相切；略微真实越界应拒绝。
- 深度轴不包含 `0` 的 reference，不超过 `depth_max` 或球半径；任意非法步长立即报错。

### 14.2 几何与材料配对

- 中心 `p=q=0`；x/y 正负方向的坡度符号符合 `-s*dx/root`、`-s*dy/root`。
- `direction_z=±1` 与 `view_side_z=±1` 各自变化时，照明朝向和标签框架保持独立。
- 用几何解析坡度 → `normalize((-p,-q,1))` → 反算坡度往返一致。
- 接触边界 `h=0`，标签不混入圆外；可选 margin 不改变渲染图像或扫描球心。
- 非正对相机、`fx!=fy`、主点偏移时仍用同材料点计算色差。
- 在不压入的自比较中，所有有效材料点色差接近零。
- 使用有已知空间梯度的合成 reference 测试 `uv0!=uvd`，证明实现未退化为同像素相减。
- 构造一个无效/饱和邻居，四点采样必须拒绝该样本。

### 14.3 GPU 与原 CPU 模型对照

保留现有 `7` 项物理测试；CPU 基线不改变其物理含义。新增小分辨率 GPU 对照覆盖：平面、球冠、阴影开/关、斜相机、非中心主点、相切球冠、不同灯带指数和功率。

建议初始验收容差：

- GPU float64 几何与 CPU：有效 mask 一致；同一稳定分支的点/深度误差 `atol<=1e-7 mm`，法线分量 `atol<=1e-8`。
- 物理辐照度的 float64 对照：`rtol<=1e-7`，绝对容差根据暗区尺度选取并写入测试说明。
- 默认混合精度：RGB sensor `rtol<=3e-5, atol<=1e-9 W/m²`；全图各通道能量相对误差 `<=1e-4`，uint8 图绝大多数像素一致、最大误差 `<=1` 灰度级。
- 接近圆周、切线或遮挡边界的判别点单独设测试，明确容差的几何原因；不允许通过删除所有阴影/边缘像素来掩盖错误。
- 对任意 light/pixel chunk 设置，数值应在约定容差内一致；显存分块不能改变球心、光源位置或样本权重。

这些是实施验收目标；若现有模型在判别边界无法满足，先定位条件数/浮点原因、建立针对性用例并记录调整理由，不能无解释地放宽全部容差。

### 14.4 上游 LUT 对照

以固定 NumPy seed 生成同一批 colors/slopes，分别调用固定提交的原函数和移植函数：

- `color_min/max`、`original_valid`、`sample_counts` 一致。
- LUT slopes、variances、sigma_ref2 和 raw 查询结果一致，优先逐值一致；不得因换算法出现系统偏差。
- 覆盖单样本节点、偶数个元素 median、MAD=0、混合异常法线、RMS 恰在门限附近、只有一个有效节点、颜色范围单侧、零色差落在节点/节点间等情况。
- 同色异坡度节点应按原法线离散度处理，不能只取其中一个模式。
- 查询方差要含节点间均值差异，零点周围最多八节点确实被覆盖。
- 格式版本 5 的结果能被原 `NormalCalibration.load()` 加载，原 raw lookup 在相同查询下返回相同数组；这项测试不要求启动上游重建工程。

参考对照必须能离线重跑：在已忽略的 `tests/` 内保存固定提交的参考数值源码或由其生成的小型 golden arrays，并记录 SHA 与生成输入。若执行原源码，用独立测试进程/模块命名空间隔离上游 `utils` 名称，提供原版法线/坡度转换所需的最小依赖；不要为了跑对照引入整套 Qt/JAX/marker 工程，也不要每次测试联网获取 main 分支。

### 14.5 验证拆分与失败行为

- 同一 image index 的样本只属于训练或 held-out 一方。
- 测试有意在 held-out 加入训练范围外颜色，验证越界率用 train 的原始范围计算。
- 制造一张验证失败图，即使其他图误差很小，整体仍失败且不覆盖正式 LUT。
- 全样本重新拟合仅发生在通过之后；图表区分 final fit 与 held-out train LUT。
- 只有一个接受姿态、所有姿态无样本、某个颜色通道零跨度等情况有可读错误。

### 14.6 精简输出与可视化

- 小规模完整运行有 LUT、样本、manifest、report、每姿态单图和全部规定图表。
- 每个 `source_image` 与 `verification_image` 路径相同，没有 reference/raw observation 的重复 PNG。
- 输出目录不含 `frames/overall/prepared`、STL、逐帧 GT NPZ 或 irradiance NPY。
- 圈线按投影真实边界绘制；测试一个斜相机，不能仍画固定半径图像圆。
- visualization 在没有任何整体曲面文件的情况下独立完成；实际渲染调用次数仅为姿态数加一，不产生逐帧显示目录及诊断缓存。
- branch 支持计数逐对逐半径与上游 compare_branch_context 一致，点级重叠优先和 80% 占优边界均有回归测试。
- 歧义的流式实现与小数据全量 `query_pairs()` 返回的端点 mask、within/cross-image 数量一致。
- 保存文件均可 `allow_pickle=False` 读取；不存在引用缺失曲面文件的 metadata。

### 14.7 性能验收

先预热 GPU，再对相同配置至少测多帧，分别记录 reference、单姿态渲染、材料采样、CPU LUT 拟合、CPU 绘图、缓存 I/O 和单次扫描的耗时。固定相同分辨率、相同灯带点数和相同阴影配置。

报告 CPU/GPU 单帧中位耗时、扫描总耗时及 `torch.cuda.max_memory_allocated()`。要求渲染热路径没有 CPU 数组转换，显存峰值有余量；约 8 GB 设备起始预算不超过总显存的 `70%`，通过分块控制。

加速目标为相同物理配置下 GPU 稳态渲染至少约 `5×`；此数值是目标，非当前实测。若未达目标，使用 profiler 定位小 kernel、float64 求交、重复分配和同步开销，再调整分块；不得用少算阴影或低分辨率的结果替代对照。

GPU 不可用的环境可运行 CPU/LUT/扫描测试，并将 GPU 测试明确 skip；这不能代替实际 GPU 验收。真实本机必须运行 GPU parity、完整小扫描及性能检查。

## 15. 实施顺序与每阶段完成条件

按以下顺序执行，每阶段完成后再进入下一阶段：

1. **移植 LUT 核心。** 新建 `normal_mapping.py`，移植保存/加载/fit/raw lookup，建立与固定提交的数值对照。此时不做渲染，也不改算法。
2. **共享配置与扫描。** 提取 `utils/config.py`，实现相切校验及稳定闭区间扫描；dry-run 验证 `1316` 姿态和首末相切。
3. **GPU 基础渲染。** 先复制 CPU 基线，再实现 CUDA 几何、光源积分、阴影及成像；通过小图 CPU/GPU 对照后将一般 analytic 默认切到 GPU。
4. **材料配对和解析标签。** 建固定网格，渲染一次 reference 与少量手选姿态；通过斜相机、符号、四邻采样测试，生成第一批紧凑样本。
5. **采样 pipeline 与精简 writer。** 跑小扫描，输出样本、manifest、report 和单图预览；确认没有通用 `save_result()` 的多余资产。
6. **整图验证与发布。** 训练 LUT、held-out 图判定、正式全样本 LUT；实现失败不覆盖和元数据一致性。
7. **可视化移植。** 使用第一遍配对样本生成全局图、分支分类和延续图；独立歧义分析通过确定的 stride 采样，并明确声明统计范围。
8. **性能与完整扫描。** 在实际 CUDA 设备测基线、显存、一次渲染扫描及缓存读取耗时，再跑默认或用户指定完整扫描；如颜色→坡度存在空间歧义导致失败，如实报告，不放宽门限伪造通过。
9. **收尾。** 更新 `docs/使用说明.md` 的新入口、配置、CUDA 安装与输出说明，修正其中仍可能存在的旧测试命令；锁定依赖，执行全部适用检查。

实施完成后的验收命令至少包括：

```powershell
uv sync --cache-dir .uv-cache --locked --extra gpu --extra calibration
uv run --cache-dir .uv-cache --extra gpu --extra calibration python -m unittest discover -s tests -v
uv run --cache-dir .uv-cache --extra gpu --extra calibration python calibrate_lut.py --config config.json --dry-run
uv run --cache-dir .uv-cache --extra gpu --extra calibration --extra gpu-compile python calibrate_lut.py --config config.json --out results_lut
```

小规模集成测试使用测试内临时配置覆盖更粗扫描和较低分辨率，不修改正式示例物理参数。正式性能与默认扫描验收仍使用原分辨率及灯带采样参数。

最终交付应能明确回答：使用了哪些姿态、样本是否有精确几何对应、GPU 计算是否保持原物理模型、LUT 是否严格遵守上游算法、held-out 是否通过、没有输出哪些逐帧大资产，以及实际速度和显存是多少。
