# xense-mujoco

xense-mujoco 是一个将 Xense 传感器仿真能力与 MuJoCo（和相关可视化）集成的库，方便在 MuJoCo 环境中使用视触觉传感器、渲染与仿真工具进行研究与开发。

## 主要功能

- 在 MuJoCo 场景中集成 xense 传感器与渲染工具
- 提供演示脚本（examples/）用于快速上手
- 包含一些 URDF/资产与示例资源（见 `src/xensemujoco/assets`）

## 系统与依赖

- Python: 3.9/3.10
- 依赖库
  - `mujoco`
  - `numpy`
  - `xensesdk`
  - `xensim`

## 安装

在一个已安装并配置好 MuJoCo 的 Python 环境中：

1. 安装依赖

```shell
pip install xensesdk, xensim
```

2. 克隆仓库并安装：

```shell
git clone <repository-url> && cd xense-mujoco
pip install -e .
```

## 快速开始

仓库中包含若干示例脚本，最常用的示例位于 `examples/`：

- `examples/example_tac_sensor.py` — 触觉传感器示例

运行示例（在已安装依赖并配置 MuJoCo 的环境下）：

```shell
python ./examples/example_tac_sensor.py
```

示例会展示如何创建传感器对象、加载标定/资产文件并在 MuJoCo 场景中渲染或运行仿真步骤。
