# Xensim

Xensim 是一个视觉触觉传感器仿真库，支持基于深度图的快速渲染（SimpleSensor）和基于有限元的物理仿真（FemSensor）。

## 1. 安装

### 环境要求

- Python 3.9/3.10

### 安装步骤

1. **安装依赖库**

```bash
pip install xensesdk
```

2. **安装 xensim**

```bash
cd xensim
pip install -e .
```

## 2. 快速开始

Xensim 提供两种传感器仿真方式：

- **SimpleSensor**: 轻量级传感器，仅基于深度图渲染，不进行物理仿真
- **FemSensor**: 完整的传感器仿真，包含有限元计算、Marker 追踪和接触力计算

---

## SimpleSensor - 简单传感器

### 概述

`SimpleSensor` 适用于只需要快速生成触觉图像，而不需要物理仿真的场景。它直接从深度图渲染传感器图像，性能高，延迟低。

### 主要特点

- ✅ 快速深度图渲染
- ✅ 基于标定文件的 RGB 输出
- ✅ 可调节的网格平滑
- ❌ 无物理仿真
- ❌ 无 Marker 追踪
- ❌ 无接触力计算

### 初始化参数

```python
from xensim.core import SimpleSensor

sensor = SimpleSensor(
    calibrate_file,           # RGB 标定文件路径 (.npz)
    mesh_size=(100, 175),     # 网格分辨率 (width, height)
    render_size=(400, 700),   # 输出图像分辨率 (width, height)
    gel_width_mm=17.3,        # 传感器宽度 (mm)
    gel_height_mm=29.14,      # 传感器高度 (mm)
    visible=True,             # 是否显示可视化窗口
    parent=None,              # 父场景
    title="SimpleSensor"      # 窗口标题
)
```

### 基本用法

```python
import numpy as np
from xensim.core import SimpleSensor

# 创建传感器实例
sensor = SimpleSensor(
    calibrate_file="path/to/calibrate_table.npz",
    visible=True
)

# 生成深度图 (单位: mm, 负值表示侵入)
depth_map = np.random.randn(100, 175) * -0.5  # shape: (height, width)

# 执行仿真步骤
sensor.step(
    depth=depth_map,      # 深度图, 单位 mm, <0 表示侵入
    smooth_norm=2,        # 法线平滑级别 (0-10)
    smooth_mesh=2         # 网格平滑级别 (0-10)
)

# 获取渲染图像
image = sensor.get_image()        # RGB 图像, shape: (700, 400, 3), uint8
diff_image = sensor.get_diff_image()  # 差分图像（去除背景）
depth = sensor.get_depth()        # 当前深度, 单位 mm, >0 表示侵入
```

### API 方法

| 方法                                      | 说明           | 返回类型                       |
| ----------------------------------------- | -------------- | ------------------------------ |
| `step(depth, smooth_norm, smooth_mesh)` | 更新传感器状态 | `None`                       |
| `get_image()`                           | 获取 RGB 图像  | `np.ndarray (H, W, 3) uint8` |
| `get_diff_image()`                      | 获取差分图像   | `np.ndarray (H, W, 3) uint8` |
| `get_depth()`                           | 获取当前深度   | `np.ndarray (H, W) float32`  |
| `align_view()`                          | 对齐相机视角   | `None`                       |
| `update()`                              | 更新可视化窗口 | `None`                       |

---

## FemSensor - 有限元传感器

### 概述

`FemSensor` 是完整的物理仿真传感器，基于有限元方法 (FEM) 计算接触变形、Marker 位移和接触力分布。适用于需要高精度物理仿真的场景。

### 主要特点

- ✅ 有限元物理仿真
- ✅ Marker 追踪和渲染
- ✅ 接触力计算
- ✅ 摩擦力模拟
- ✅ 高精度变形计算

### 初始化参数

```python
from xensim.core import FemSensor

sensor = FemSensor(
    calibrate_file,                 # 标定文件路径 (.npz)
    fem_file,                       # 有限元模型文件路径 (.npz)
    depth_size,                     # 深度图分辨率 (width, height)
    mesh_size=(100, 175),           # 渲染网格分辨率 (width, height)
    render_size=(400, 700),         # 输出图像分辨率 (width, height)
    marker_row_col=(20, 11),        # Marker 网格 (行, 列)
    marker_dx_dy_mm=(1.42, 1.36),   # Marker 间距 (mm)
    marker_radius=2,                # Marker 渲染半径 (像素)
    visible=True,                   # 是否显示可视化窗口
    parent=None,
    title="FemSensor"
)
```

### 基本用法

```python
import numpy as np
from xensim.core import FemSensor
from xensesdk.ezgl import Matrix4x4

# 创建传感器实例
sensor = FemSensor(
    calibrate_file="path/to/calibrate_table.npz",
    fem_file="path/to/fem_data.npz",
    depth_size=(100, 175),  # 必须与深度图大小一致
    visible=True
)

# 定义物体和传感器位姿 (世界坐标系, 单位: m)
object_pose = Matrix4x4.fromVector6d(x=0, y=0.02, z=0, a=0, b=0, c=0)
sensor_pose = Matrix4x4.fromVector6d(x=0, y=0, z=0, a=0, b=0, c=0)

# 深度图 (单位: mm, <0 表示侵入)
depth_map = np.random.randn(175, 100) * -0.5  # shape: (height, width)

# 执行仿真 (多步迭代以提高收敛性)
sensor.step(
    obj_pose=object_pose,
    sensor_pose=sensor_pose,
    depth=depth_map,
    nstep=3  # 仿真迭代步数
)

# 获取仿真结果
image = sensor.get_image()         # RGB 图像
marker = sensor.get_marker()       # Marker 坐标, shape: (20, 11, 3)
force = sensor.get_force()         # 节点力, shape: (n, m, 3)
distance = sensor.get_distance()   # 最小距离 (mm)
```

### 可视化设置

```python
# 配置仿真可视化选项
sensor.set_sim(
    show_marker=True,      # 显示 Marker
    show_contact=False,    # 显示接触点
    show_force=False,      # 显示接触力矢量
    smooth_mesh=1,         # 网格平滑级别 (0-10)
    smooth_norm=5,         # 法线平滑级别 (0-10)
    rgb_gain=1.0,          # RGB 增益 (>1 增强, <1 减弱)
    bg_color=True,         # 显示背景颜色
    fric_coef=0.4          # 摩擦系数
)
```

### API 方法

| 方法                                          | 说明                   | 返回类型                           |
| --------------------------------------------- | ---------------------- | ---------------------------------- |
| `step(obj_pose, sensor_pose, depth, nstep)` | 执行有限元仿真         | `None`                           |
| `get_image()`                               | 获取 RGB 图像          | `np.ndarray (H, W, 3) uint8`     |
| `get_diff_image()`                          | 获取差分图像（去背景） | `np.ndarray (H, W, 3) uint8`     |
| `get_depth()`                               | 获取深度图             | `np.ndarray (H, W) float32`      |
| `get_marker()`                              | 获取 Marker 坐标       | `np.ndarray (20, 11, 3) float32` |
| `get_marker_image()`                        | 获取 Marker 二值图     | `np.ndarray (H, W, 3) uint8`     |
| `get_force()`                               | 获取节点力             | `np.ndarray (n, m, 3) float32`   |
| `get_distance()`                            | 获取最小距离           | `float`                          |
| `set_sim(...)`                              | 配置仿真选项           | `None`                           |
| `align_view()`                              | 对齐相机视角           | `None`                           |
| `update()`                                  | 更新可视化窗口         | `None`                           |

---

## 两种传感器对比

| 特性                  | SimpleSensor       | FemSensor            |
| --------------------- | ------------------ | -------------------- |
| **渲染速度**    | 🚀 极快            | ⏱️ 较慢            |
| **物理仿真**    | ❌ 无              | ✅ 有限元仿真        |
| **Marker 追踪** | ❌ 无              | ✅ 支持              |
| **接触力计算**  | ❌ 无              | ✅ 支持              |
| **摩擦力模拟**  | ❌ 无              | ✅ 支持              |
| **适用场景**    | 快速原型、数据增强 | 高精度仿真、力学分析 |

---

## 3. 示例代码

完整的示例代码位于 `examples/` 目录：

- `example_simple_sensor.py` - SimpleSensor 使用示例
- `example_fem_sensor.py` - FemSensor 使用示例（带 UI）

---

## 许可证

Xense Simulator SDK – Binary Runtime License (Non-Commercial) 
