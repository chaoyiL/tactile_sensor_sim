"""
ImGui Blueprint Framework - 类似 Rerun Blueprint 的声明式 ImGui 界面定义

Author: Jin Liu
Date: 2025/10/18
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Any, Union
import imgui
import numpy as np
import OpenGL.GL as gl

try:
    import glfw
    from imgui.integrations.glfw import GlfwRenderer
    HAS_GLFW = True
except ImportError:
    HAS_GLFW = False


@dataclass
class WidgetState:
    """控件状态基类"""
    pass


@dataclass
class SliderState(WidgetState):
    """滑块状态"""
    value: float = 0.0


@dataclass
class CheckboxState(WidgetState):
    """复选框状态"""
    checked: bool = False


@dataclass
class InputTextState(WidgetState):
    """文本输入状态"""
    text: str = ""


@dataclass
class ImageState(WidgetState):
    """图像状态"""
    texture_id: Optional[int] = None
    width: int = 0
    height: int = 0


# region 辅助函数：变量绑定 
def _get_bound_value(bind_to: tuple, default_value: Any) -> Any:
    """从绑定目标获取当前值
    
    参数：
    - bind_to: (对象, 属性/索引) 元组
    - default_value: 默认值（当无法获取时使用）
    
    返回：
    - 绑定目标的当前值
    """
    obj, attr = bind_to
    if isinstance(obj, dict):
        return obj.get(attr, default_value)
    elif isinstance(obj, (list, np.ndarray)):
        return obj[attr]
    else:
        return getattr(obj, attr, default_value)

def _set_bound_value(bind_to: tuple, new_value: Any) -> None:
    """设置绑定目标的值
    
    参数：
    - bind_to: (对象, 属性/索引) 元组
    - new_value: 新值
    """
    obj, attr = bind_to
    if isinstance(obj, dict):
        obj[attr] = new_value
    elif isinstance(obj, (list, np.ndarray)):
        obj[attr] = new_value
    else:
        setattr(obj, attr, new_value)
# endregion ------------------------------------------------------------------


# region : widgets
class Widget:
    """控件基类"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self._path = ""  # 控件的完整路径，由 Blueprint 自动设置
    
    def render(self, state_dict: dict) -> Any:
        """渲染控件，返回值表示状态是否改变"""
        raise NotImplementedError
    
    def _collect_widgets(self, widget_dict: dict, path_prefix: str = ""):
        """收集所有子控件到字典中（用于 Blueprint 的 widget 查找）"""
        if self.name:
            full_path = f"{path_prefix}/{self.name}" if path_prefix else self.name
            self._path = full_path
            widget_dict[full_path] = self


class Text(Widget):
    """文本控件"""
    
    def __init__(self, text: str, name=""):
        super().__init__(name)
        self.text = text
    
    def render(self, state_dict: dict) -> Any:
        imgui.text(self.text)
        return None


class Separator(Widget):
    """分隔线"""
    
    def render(self, state_dict: dict) -> Any:
        imgui.separator()
        return None

class Spacing(Widget):

    def render(self, state_dict: dict) -> Any:
        imgui.spacing()
        return None

class Button(Widget):
    """按钮控件"""
    
    def __init__(self, name: str, callback: Optional[Callable[[], None]] = None, 
                 size: tuple = (0, 0)):
        super().__init__(name)
        self.callback = callback
        self.size = size
    
    def render(self, state_dict: dict) -> Any:
        clicked = imgui.button(self.name, *self.size)
        if clicked and self.callback:
            self.callback()
        return clicked


class SameLine(Widget):
    """同行布局"""
    
    def __init__(self, spacing: float = -1.0, offset: float = 0.0):
        super().__init__()
        self.spacing = spacing
        self.offset = offset
    
    def render(self, state_dict: dict) -> Any:
        if self.spacing >= 0:
            imgui.same_line(spacing=self.spacing, offset_from_start_x=self.offset)
        else:
            imgui.same_line()
        return None


class SliderFloat(Widget):
    """浮点数滑块
    
    支持两种使用方式：
    1. 内部状态管理（传入 default_value）
    2. 外部绑定（传入 bind_to）- 直接同步到外部变量
    
    示例：
        # 方式1：内部状态
        SliderFloat(name="volume", min_value=0, max_value=100, default_value=50)
        
        # 方式2：外部绑定
        config = {"distance": 2.5}
        SliderFloat(name="distance", min_value=0, max_value=10, bind_to=(config, "distance"))
    """
    
    def __init__(self, name: str, min_value: float = 0.0, max_value: float = 1.0,
                 default_value: float = 0.0, 
                 callback: Optional[Callable[[float], None]] = None,
                 bind_to: Optional[tuple] = None):
        super().__init__(name)
        self.min_value = min_value
        self.max_value = max_value
        self.default_value = default_value
        self.callback = callback
        self.bind_to = bind_to  # (对象, 属性名) 或 (字典, 键)
    
    def render(self, state_dict: dict) -> Any:
        # 确定当前值
        if self.bind_to:
            current_value = _get_bound_value(self.bind_to, self.default_value)
        else:
            if self.name not in state_dict:
                state_dict[self.name] = SliderState(value=self.default_value)
            state = state_dict[self.name]
            current_value = state.value
        
        # 渲染滑块
        changed, new_value = imgui.slider_float(
            self.name, current_value, self.min_value, self.max_value
        )
        
        # 更新值
        if changed:
            if self.bind_to:
                _set_bound_value(self.bind_to, new_value)
            else:
                state_dict[self.name].value = new_value
            
            if self.callback:
                self.callback(new_value)
        
        return changed


class Checkbox(Widget):
    """复选框
    
    支持两种使用方式：
    1. 内部状态管理（传入 default_checked）
    2. 外部绑定（传入 bind_to）- 直接同步到外部变量
    
    示例：
        # 方式1：内部状态
        Checkbox(name="enable", default_checked=True)
        
        # 方式2：外部绑定（类似 ImGui 原生用法）
        flags = {"show_contacts": True}
        Checkbox(name="Show Contacts", bind_to=(flags, "show_contacts"))
        
        # 方式3：绑定到对象属性
        Checkbox(name="Show Contacts", 
                bind_to=(viewer.mjv_option.flags, mjtVisFlag.mjVIS_CONTACTPOINT))
    """
    
    def __init__(self, name: str, default_checked: bool = False,
                 callback: Optional[Callable[[bool], None]] = None,
                 bind_to: Optional[tuple] = None):
        super().__init__(name)
        self.default_checked = default_checked
        self.callback = callback
        self.bind_to = bind_to  # (对象, 属性名) 或 (字典/列表/数组, 键/索引)
    
    def render(self, state_dict: dict) -> Any:
        # 确定当前值
        if self.bind_to:
            current_value = bool(_get_bound_value(self.bind_to, self.default_checked))
        else:
            if self.name not in state_dict:
                state_dict[self.name] = CheckboxState(checked=self.default_checked)
            state = state_dict[self.name]
            current_value = state.checked
        
        # 渲染复选框
        changed, new_checked = imgui.checkbox(self.name, current_value)
        
        # 更新值
        if changed:
            if self.bind_to:
                _set_bound_value(self.bind_to, new_checked)
            else:
                state_dict[self.name].checked = new_checked
            
            if self.callback:
                self.callback(new_checked)
        
        return changed


class InputText(Widget):
    """文本输入框
    
    支持两种使用方式：
    1. 内部状态管理（传入 default_text）
    2. 外部绑定（传入 bind_to）- 直接同步到外部变量
    
    示例：
        # 方式1：内部状态
        InputText(name="username", default_text="guest")
        
        # 方式2：外部绑定
        config = {"username": "admin"}
        InputText(name="username", bind_to=(config, "username"))
    """
    
    def __init__(self, name: str, default_text: str = "",
                 buffer_size: int = 256,
                 callback: Optional[Callable[[str], None]] = None,
                 bind_to: Optional[tuple] = None):
        super().__init__(name)
        self.default_text = default_text
        self.buffer_size = buffer_size
        self.callback = callback
        self.bind_to = bind_to
    
    def render(self, state_dict: dict) -> Any:
        # 确定当前值
        if self.bind_to:
            current_text = _get_bound_value(self.bind_to, self.default_text)
        else:
            if self.name not in state_dict:
                state_dict[self.name] = InputTextState(text=self.default_text)
            state = state_dict[self.name]
            current_text = state.text
        
        # 渲染输入框
        changed, new_text = imgui.input_text(
            self.name, current_text, self.buffer_size
        )
        
        # 更新值
        if changed:
            if self.bind_to:
                _set_bound_value(self.bind_to, new_text)
            else:
                state_dict[self.name].text = new_text
            
            if self.callback:
                self.callback(new_text)
        
        return changed


class ImageWidget(Widget):
    """图像显示控件
    
    用于在 ImGui 中显示图像。自动管理 OpenGL 纹理的创建和更新。
    
    参数：
    - name: 控件名称（用于状态管理）
    - image: numpy 数组，形状为 (height, width, 3) 或 (height, width, 4)，dtype=uint8
    - width: 显示宽度（可选，默认使用图像宽度）
    - height: 显示高度（可选，默认使用图像高度）
    
    示例：
        import numpy as np
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        ImageWidget(name="my_image", image=image)
    """
    
    def __init__(self, name: str, image: Optional[np.ndarray] = None,
                 width: Optional[int] = None, height: Optional[int] = None):
        super().__init__(name)
        self.image = image
        self.display_width = width
        self.display_height = height
        self._needs_update = True  # 标记是否需要更新纹理
    
    def set_image(self, image: np.ndarray):
        """更新图像数据"""
        self.image = image
        self._needs_update = True  # 标记需要更新纹理
        return self
    
    def _create_texture(self, image: np.ndarray) -> int:
        """创建 OpenGL 纹理"""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        # 确定 OpenGL 格式
        if channels == 3:
            gl_format = gl.GL_RGB
        elif channels == 4:
            gl_format = gl.GL_RGBA
        elif channels == 1:
            gl_format = gl.GL_RED
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
        
        # 创建纹理
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl_format, width, height, 0, 
                       gl_format, gl.GL_UNSIGNED_BYTE, image)
        
        return texture_id
    
    def _update_texture(self, texture_id: int, image: np.ndarray):
        """更新纹理数据"""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        if channels == 3:
            gl_format = gl.GL_RGB
        elif channels == 4:
            gl_format = gl.GL_RGBA
        elif channels == 1:
            gl_format = gl.GL_RED
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
        
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, width, height, 
                          gl_format, gl.GL_UNSIGNED_BYTE, image)
    
    def render(self, state_dict: dict) -> Any:
        if self.image is None:
            imgui.text(f"[{self.name}] No image set")
            return None
        
        img_height, img_width = self.image.shape[:2]
        
        # 确定显示尺寸
        display_width = self.display_width if self.display_width else img_width
        display_height = self.display_height if self.display_height else img_height
        
        # 初始化或更新纹理状态
        if self.name not in state_dict:
            state_dict[self.name] = ImageState()
        
        state = state_dict[self.name]
        
        # 检查是否需要重新创建纹理（尺寸改变）
        if state.texture_id is None or state.width != img_width or state.height != img_height:
            # 删除旧纹理
            if state.texture_id is not None:
                gl.glDeleteTextures([state.texture_id])
            
            # 创建新纹理
            state.texture_id = self._create_texture(self.image)
            state.width = img_width
            state.height = img_height
            self._needs_update = False  # 重置更新标志
        elif self._needs_update:
            # 只有在 set_image() 被调用后才更新纹理
            self._update_texture(state.texture_id, self.image)
            self._needs_update = False  # 重置更新标志
        
        # 渲染图像
        imgui.image(state.texture_id, display_width, display_height)
# endregion ------------------------------------------------------------------


# region: layouts
class Layout:
    """布局基类"""
    
    def __init__(self, *children: Union[Widget, 'Layout'], name: str = ""):
        self.children = list(children)
        if len(children) > 0 and isinstance(children[0], str):
            raise ValueError(f"First argument to Layout must be Widget or Layout, not str: '{children[0]}'")
            
        self.name = name
        self._path = ""  # 布局的完整路径
    
    def add(self, *children: Union[Widget, 'Layout']) -> 'Layout':
        """添加子控件/布局，支持链式调用"""
        self.children.extend(children)
        return self
    
    def render(self, state_dict: dict):
        """渲染布局"""
        raise NotImplementedError
    
    def _collect_widgets(self, widget_dict: dict, path_prefix: str = ""):
        """递归收集所有子控件"""
        if self.name:
            full_path = f"{path_prefix}/{self.name}" if path_prefix else self.name
            self._path = full_path
            
            if isinstance(self, Panel) and full_path in widget_dict:
                raise ValueError(f"Duplicate Panel name detected: {full_path}")
            
            widget_dict[full_path] = self
            current_prefix = full_path
        else:
            current_prefix = path_prefix + "/."
        
        for child in self.children:
            if hasattr(child, '_collect_widgets'):
                child._collect_widgets(widget_dict, current_prefix)


class Vertical(Layout):
    """垂直布局"""
    
    def render(self, state_dict: dict):
        for child in self.children:
            child.render(state_dict)


class Horizontal(Layout):
    """水平布局"""
    
    def __init__(self, *children: Union[Widget, Layout], name: str = "", 
                 spacing: float = 8.0):
        super().__init__(*children, name=name)
        self.spacing = spacing
    
    def render(self, state_dict: dict):
        for i, child in enumerate(self.children):
            if i > 0:
                imgui.same_line(spacing=self.spacing)
            child.render(state_dict)


class Group(Layout):
    """分组容器"""
    
    def render(self, state_dict: dict):
        imgui.begin_group()
        for child in self.children:
            child.render(state_dict)
        imgui.end_group()


class CollapsingHeader(Layout):
    """可折叠的标题"""
    
    def __init__(self, *children: Union[Widget, Layout], name: str = "",
                 default_open: bool = True):
        super().__init__(*children, name=name)
        self.default_open = default_open
    
    def render(self, state_dict: dict):
        expanded, visible = imgui.collapsing_header(
            self.name, 
            flags=imgui.TREE_NODE_DEFAULT_OPEN if self.default_open else 0
        )
        if expanded:
            for child in self.children:
                child.render(state_dict)


class Panel(Layout):
    """面板（ImGui 窗口）
    
    Blueprint 中的顶层容器，对应 ImGui 的窗口。
    支持固定位置、大小、停靠等功能。
    """
    
    def __init__(self, *children,
                 name: str = "Panel",
                 pos_x: Optional[float] = None,
                 pos_y: Optional[float] = None,
                 width: float = 0,
                 height: float = 0,
                 fixed_position: bool = False,
                 resizable: bool = True,
                 no_collapse: bool = False,
                 no_title_bar: bool = False,
                 closable: bool = True,
                 dock_to: Optional[str] = None):
        """
        参数：
        - children: 子控件
        - name: 窗口名称
        - pos_x, pos_y: 窗口位置
        - width, height: 窗口大小
        - fixed_position: 固定位置（不可移动）
        - resizable: 是否可调整大小（停靠模式下支持拖动边缘调整）
        - no_collapse: 禁止折叠
        - no_title_bar: 隐藏标题栏
        - closable: 显示关闭按钮
        - dock_to: 停靠位置 ("left", "right", "top", "bottom")
        """
        super().__init__(*children, name=name)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.width = width
        self.height = height
        self.fixed_position = fixed_position
        self.resizable = resizable
        self.no_collapse = no_collapse
        self.no_title_bar = no_title_bar
        self.closable = closable
        self.dock_to = dock_to
        self._is_open = True
        
        # 用于拖动调整大小的状态
        self._is_dragging = False
        self._drag_start_pos = 0.0
    
    def render(self, state_dict: dict) -> Any:
        if not self._is_open:
            return False
        
        # 构建窗口标志
        flags = self._build_window_flags()
        
        # 设置窗口位置和大小
        if self.dock_to:
            self._setup_docked_window()
        else:
            self._setup_normal_window(flags)
        
        # 开始渲染窗口
        expanded, self._is_open = imgui.begin(self.name, closable=self.closable, flags=flags)
        
        if expanded:
            # 渲染子控件
            for child in self.children:
                child.render(state_dict)
            
            # 如果停靠且可调整大小，绘制拖动条
            if self.dock_to and self.resizable:
                self._render_resize_handle()
        
        imgui.end()
        return self._is_open
    
    def _build_window_flags(self) -> int:
        """构建窗口标志"""
        flags = 0
        if self.no_collapse:
            flags |= imgui.WINDOW_NO_COLLAPSE
        if self.no_title_bar:
            flags |= imgui.WINDOW_NO_TITLE_BAR
        if self.dock_to or self.fixed_position:
            flags |= imgui.WINDOW_NO_MOVE
        if self.dock_to or not self.resizable:
            flags |= imgui.WINDOW_NO_RESIZE  # 无右下角调整大小手柄
        return flags
    
    def _setup_docked_window(self):
        """设置停靠窗口的位置和大小"""
        io = imgui.get_io()
        display_width, display_height = io.display_size
        
        # 根据停靠方向计算位置和大小
        if self.dock_to == "left":
            pos = (0, self.pos_y or 0)
            size = (self.width, self.height or display_height - (self.pos_y or 0))
        elif self.dock_to == "right":
            pos = (display_width - self.width, self.pos_y or 0)
            size = (self.width, self.height or display_height - (self.pos_y or 0))
        elif self.dock_to == "top":
            pos = (self.pos_x or 0, 0)
            size = (self.width or display_width - (self.pos_x or 0), self.height)
        elif self.dock_to == "bottom":
            pos = (self.pos_x or 0, display_height - self.height)
            size = (self.width or display_width - (self.pos_x or 0), self.height)
        else:
            return  # 无效的停靠方向    

        imgui.set_next_window_position(*pos)
        imgui.set_next_window_size(*size)

    def _setup_normal_window(self, flags: int):
        """设置普通窗口的位置和大小"""
        if self.pos_x is not None and self.pos_y is not None:
            imgui.set_next_window_position(self.pos_x, self.pos_y, imgui.ALWAYS if self.fixed_position else imgui.ONCE)
        
        if self.width > 0 and self.height > 0:
            imgui.set_next_window_size(self.width, self.height, imgui.ONCE if self.resizable else imgui.ALWAYS)
    
    def _render_resize_handle(self):
        """渲染可拖动的调整大小手柄"""
        io = imgui.get_io()
        window_pos = imgui.get_window_position()
        window_size = imgui.get_window_size()
        
        # 配置拖动条参数
        handle_thickness = 4
        handle_config = self._get_handle_config(window_pos, window_size, handle_thickness)
        if not handle_config:
            return
        
        handle_pos, handle_size, cursor_type, is_horizontal = handle_config
        
        # 检查鼠标悬停
        mouse_pos = io.mouse_pos
        is_hovering = (handle_pos[0] <= mouse_pos.x <= handle_pos[0] + handle_size[0] and
                      handle_pos[1] <= mouse_pos.y <= handle_pos[1] + handle_size[1])
        
        # 绘制拖动条（悬停或拖动时显示）
        if is_hovering or self._is_dragging:
            draw_list = imgui.get_foreground_draw_list()
            color = imgui.get_color_u32_rgba (0.4, 0.6, 1.0, 0.8) if is_hovering else imgui.get_color_u32_rgba (0.3, 0.5, 0.8, 0.5)
            draw_list.add_rect_filled(
                handle_pos[0], handle_pos[1],
                handle_pos[0] + handle_size[0], handle_pos[1] + handle_size[1],
                color
            )
        
        # 处理拖动交互
        self._handle_drag_interaction(mouse_pos, cursor_type, is_hovering, is_horizontal)
    
    def _get_handle_config(self, window_pos, window_size, thickness):
        """获取拖动条配置（位置、大小、光标类型、是否水平）
        
        拖动条放置在窗口内部边缘，避免超出窗口范围导致 want_capture_mouse 失效
        """
        if self.dock_to == "left":
            # 右侧内部拖动条
            return (
                (window_pos.x + window_size.x - thickness - 1, window_pos.y),
                (thickness, window_size.y),
                imgui.MOUSE_CURSOR_RESIZE_EW,
                True
            )
        elif self.dock_to == "right":
            # 左侧内部拖动条
            return (
                (window_pos.x + 1, window_pos.y),
                (thickness, window_size.y),
                imgui.MOUSE_CURSOR_RESIZE_EW,
                True
            )
        elif self.dock_to == "top":
            # 底部内部拖动条
            return (
                (window_pos.x, window_pos.y + window_size.y - thickness - 1),
                (window_size.x, thickness),
                imgui.MOUSE_CURSOR_RESIZE_NS,
                False
            )
        elif self.dock_to == "bottom":
            # 顶部内部拖动条
            return (
                (window_pos.x, window_pos.y + 1),
                (window_size.x, thickness),
                imgui.MOUSE_CURSOR_RESIZE_NS,
                False
            )
        else:
            return None
    
    def _handle_drag_interaction(self, mouse_pos, cursor_type, is_hovering, is_horizontal):
        """处理拖动交互逻辑"""
        # 开始拖动
        if is_hovering:
            imgui.set_mouse_cursor(cursor_type)
            if imgui.is_mouse_clicked(0):
                self._is_dragging = True
                self._drag_start_pos = mouse_pos.x if is_horizontal else mouse_pos.y
        
        # 拖动中
        if self._is_dragging:
            imgui.set_mouse_cursor(cursor_type)
            
            if imgui.is_mouse_down(0):
                current_pos = mouse_pos.x if is_horizontal else mouse_pos.y
                delta = current_pos - self._drag_start_pos
                if self.dock_to == "right" or self.dock_to == "bottom":
                    delta = -delta
                
                # 更新宽度或高度
                if is_horizontal:
                    self.width = max(100, self.width + delta)
                else:
                    self.height = max(50, self.height + delta)
                
                self._drag_start_pos = current_pos
            else:
                # 拖动结束
                self._is_dragging = False



class Tabs(Layout):
    """标签页容器"""
    
    def __init__(self, *tabs: 'TabItem', active_tab: int = 0):
        super().__init__(*tabs)
        self.active_tab = active_tab
        self.tabs = tabs
    
    def render(self, state_dict: dict):
        if imgui.begin_tab_bar("tabs"):
            for tab in self.tabs:
                tab.render(state_dict)
            imgui.end_tab_bar()


class TabItem(Layout):
    """标签页项"""
    
    def __init__(self, *children: Union[Widget, Layout], name: str = "Tab"):
        super().__init__(*children, name=name)
    
    def render(self, state_dict: dict):
        if imgui.begin_tab_item(self.name)[0]:
            for child in self.children:
                child.render(state_dict)
            imgui.end_tab_item()
# endregion ------------------------------------------------------------------


class Blueprint:
    """ImGui Blueprint - 主入口类
    
    用于管理和渲染 ImGui 界面
    """

    def __init__(self, *panels: Union[Layout, Panel], window=None, imgui_renderer=None):
        """初始化 Blueprint
        
        参数：
        - panels: 一个或多个 Panel 对象
        - window: GLFW 窗口（可选，用于集成到现有应用）
        - imgui_renderer: ImGui 渲染器实例（可选，用于集成到现有应用）
        """
        self.panels = list(panels)
        self.state_dict = {}
        self._widget_dict = {}
        self._window = None
        self._imgui_renderer = imgui_renderer
        self.set_window(window)
        
        self._owns_window = False  # 标记是否由 Blueprint 创建窗口
        self._build_widget_dict()
    
    def set_window(self, window, attach_callbacks=True):
        self._window = window
        if window is not None and self._imgui_renderer is None:
            glfw.make_context_current(self._window)
            imgui.create_context()
            self._imgui_renderer = GlfwRenderer(window, attach_callbacks=attach_callbacks)

    def __enter__(self):
        if self._window:
            glfw.make_context_current(self._window)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"An exception occurred: {exc_type}: {exc_value}")
        # No need to unset context, as GLFW handles it per-thread

    @classmethod
    def create_standalone(cls, *panels: Union[Layout, Panel],
                         window_title: str = "ImGui Blueprint",
                         width: int = 1280, height: int = 720,
                         background_color: tuple = (0.1, 0.1, 0.1, 1.0)):
        """创建独立的 Blueprint 应用（自动创建 glfw 窗口）
        
        参数：
        - panels: 一个或多个 Panel 对象
        - window_title: 窗口标题
        - width, height: 窗口大小
        - background_color: 背景颜色 (R, G, B, A)
        
        返回：
        - Blueprint 实例
        
        示例：
            app = Blueprint.create_standalone(...)
            app.run()  # 开始运行
        """
        if not HAS_GLFW:
            raise ImportError(
                "GLFW is required for standalone mode. \nInstall it with: pip install glfw imgui[glfw]"
            )
        
        # 初始化 GLFW
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        
        # 设置 OpenGL 版本
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        
        # 创建窗口
        window = glfw.create_window(width, height, window_title, None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")
        glfw.make_context_current(window)
        
        # 创建 Blueprint 实例
        blueprint = cls(*panels, window=window)
        blueprint._owns_window = True
        blueprint._background_color = background_color
        
        return blueprint
    
    def _build_widget_dict(self):
        """构建控件字典，建立路径到控件的映射"""
        self._widget_dict.clear()
        for panel in self.panels:
            if hasattr(panel, '_collect_widgets'):
                panel._collect_widgets(self._widget_dict)
    
    def render(self):
        """渲染所有面板"""
        for panel in self.panels:
            panel.render(self.state_dict)
    
    def render_frame(self):
        """完整的 ImGui 帧渲染（仅在设置了 window 时可用）"""
        with self:
            if self._imgui_renderer is None:
                raise RuntimeError("Cannot call render_frame() without imgui_renderer.")
            self._imgui_renderer.process_inputs()
            imgui.new_frame()
            self.render()
            imgui.render()
            self._imgui_renderer.render(imgui.get_draw_data())
    
    def run(self, on_frame: Optional[Callable[[], bool]] = None, time_interval: float = 1/50):
        """运行独立应用的主循环（仅用于 create_standalone() 创建的实例）
        
        参数：
        - on_frame: 可选的每帧回调函数，返回 False 可以退出循环
        """
        if not self._owns_window:
            raise RuntimeError(
                "run() can only be called on Blueprint instances created with create_standalone(). "
                "For manual integration, use render() or render_frame() in your own loop."
            )
        
        try:
            while not glfw.window_should_close(self._window):
                glfw.poll_events()
                
                # 清空屏幕
                r, g, b, a = self._background_color
                gl.glClearColor(r, g, b, a)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                
                # 调用用户回调
                if on_frame and not on_frame():
                    break
                
                # 渲染 Blueprint
                self.render_frame()
                
                glfw.swap_buffers(self._window)
                time.sleep(time_interval)

        finally:
            # 清理资源
            self.shutdown()
    
    def shutdown(self):
        """清理资源"""
        if self._imgui_renderer:
            self._imgui_renderer.shutdown()
        if self._owns_window and self._window:
            glfw.terminate()
        
    def add(self, *panels: Panel):
        """添加多个面板，支持链式调用"""
        for panel in panels:
            self.panels.append(panel)
        self._build_widget_dict()  # 重新构建控件字典
        return self

    def get_widget(self, path: str) -> Optional[Union[Widget, Layout]]:
        """通过路径或名称获取控件
        
        参数：
        - path: 控件路径（如 "Panel/Layout/Widget"）或直接名称（如 "Widget"）
        
        返回：
        - 找到的控件实例，如果不存在则返回 None
        """
        if path in self._widget_dict:
            return self._widget_dict[path]
        
        # 如果不是完整路径，尝试按名称查找（找到第一个匹配的）
        for widget_path, widget in self._widget_dict.items():
            if widget_path.endswith(f"/{path}"):
                return widget
        
        return None
    
    def get_widgets_by_name(self, name: str) -> List[Union[Widget, Layout]]:
        """获取所有指定名称的控件（处理重名情况）"""
        results = []
        for widget_path, widget in self._widget_dict.items():
            if widget.name == name:
                results.append(widget)
        return results
    
    def list_widgets(self) -> List[str]:
        """列出所有控件的路径"""
        return sorted(self._widget_dict.keys())
    
    def get_state(self, label: str) -> Optional[WidgetState]:
        """获取控件状态"""
        return self.state_dict.get(label)
    
    def set_state(self, label: str, state: WidgetState):
        """设置控件状态"""
        self.state_dict[label] = state
    
    def is_any_panel_dragging(self) -> bool:
        """检查是否有任何 Panel 正在拖动调整大小"""
        for panel in self.panels:
            if isinstance(panel, Panel) and panel._is_dragging:
                return True
        return False


# 示例用法
if __name__ == "__main__":
    def on_button_click():
        print("Button clicked!")
    
    def on_slider_change(value):
        print(f"Slider value: {value}")
    
    def on_checkbox_change(checked):
        print(f"Checkbox: {checked}")
    
    # 方式 1: 单面板模式
    single_panel_blueprint = Blueprint(
        Panel(
            name="Control Panel",
            width=400,
            height=600
        ).add(
            Vertical().add(
                Text("Single Panel Example"),
                Separator(),
                Button("Button 1", callback=on_button_click),
                SliderFloat("Value", 0, 100, 50, callback=on_slider_change),
            )
        )
    )
    
    # 方式 2: 多面板模式（推荐）
    multi_panel_blueprint = Blueprint(
        # 左侧边栏
        Panel(
            name="Sidebar",
            width=250,
            dock_to="left",
            closable=False
        ).add(
            Vertical().add(
                Text("Sidebar"),
                Separator(),
                Button("Action 1"),
                Button("Action 2"),
            )
        ),
        
        # 顶部工具栏
        Panel(
            name="Toolbar",
            height=60,
            dock_to="top",
            no_title_bar=True,
            closable=False
        ).add(
            Horizontal(spacing=10).add(
                Button("New"),
                Button("Open"),
                Button("Save"),
            )
        ),
        
        # 主面板
        Panel(
            name="Main Panel",
            pos_x=300,
            pos_y=100,
            width=400,
            height=500
        ).add(
            Vertical().add(
                Text("Main Content"),
                Separator(),
                
                Horizontal(
                    Button("Button 1", callback=on_button_click),
                    Button("Button 2", callback=on_button_click),
                ),
                
                Separator(),
                
                Tabs(
                    TabItem(
                        Text("Tab 1 Content"),
                        Button("Action 1"),
                        name="Tab 1"
                    ),
                    TabItem(
                        Text("Tab 2 Content"),
                        SliderFloat("Value", 0, 100),
                        name="Tab 2"
                    ),
                    active_tab=0
                ),
            ),
            name="My Control Panel",
            width=400,
            height=600
        )
    )
    
    # 在渲染循环中调用
    # 单面板模式：
    # single_panel_blueprint.render()
    
    # 多面板模式（一次渲染所有面板）：
    # multi_panel_blueprint.render()
