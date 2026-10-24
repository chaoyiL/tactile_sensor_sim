import mujoco
import numpy as np
from collections import deque
from .transform3d import Matrix4x4


class DataBuffer:
    """
    使用 np.array 模拟最大长度为 maxlen 的循环缓冲区，用于存储 shape 的 np.array 数据。
    """
    def __init__(self, maxlen, shape, dtype=np.float32):
        self.maxlen = maxlen
        self.shape = shape
        self.buffer = np.zeros((maxlen,) + shape, dtype=dtype)
        self.start = 0
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")
        return self.buffer[(self.start + index) % self.maxlen]
    
    def __iter__(self):
        for i in range(self.size):
            yield self.buffer[(self.start + i) % self.maxlen]
    
    def clear(self):
        self.start = 0
        self.size = 0
            
    def extend(self, data):
        """一次性添加多个数据到缓冲区"""
        if data is None:
            return
        
        # 转换为数组并验证形状
        data_array = np.asarray(data).reshape(-1, *self.shape)
        n = len(data_array)
        if n == 0:
            return
        
        if n >= self.maxlen:
            # 数据足够多，直接覆盖整个缓冲区
            self.buffer[:] = data_array[-self.maxlen:]
            self.start = 0
            self.size = self.maxlen
        else:
            # 数据较少，需要保留部分旧数据
            keep_max = self.maxlen - n
            end = (self.start + self.size) % self.maxlen
            if keep_max < self.size:
                self.start = (self.start + (self.size - keep_max)) % self.maxlen
                self.size = self.maxlen
            else:
                self.size += n

            self._copy_to_buffer(data_array, end)
    
    def _copy_to_buffer(self, data_slice, start_pos):
        """将数据切片拷贝到缓冲区，从 start_pos 开始，至多两次拷贝"""
        n = len(data_slice)
        if n == 0:
            return
        
        # 计算从 start_pos 到缓冲区末尾的空间
        space_to_end = self.maxlen - start_pos
        
        if n <= space_to_end:
            self.buffer[start_pos:start_pos + n] = data_slice
        else:
            self.buffer[start_pos:] = data_slice[:space_to_end]
            self.buffer[:n - space_to_end] = data_slice[space_to_end:]
    
    def to_array(self):
        """返回当前缓冲区内容的数组副本"""
        if self.size == 0:
            return np.empty((0,) + self.shape)
        
        end = (self.start + self.size) % self.maxlen
        if end > self.start:
            return self.buffer[self.start:end].copy()
        else:
            return np.concatenate([self.buffer[self.start:], self.buffer[:end]])


class ColorGradient:
    """
    颜色渐变生成器
    """
    def __init__(self, colors, maxstep):
        self.colors = self.color_gradient(colors=colors, maxstep=maxstep)
        self.length = len(self.colors)
        self.idx = 0
        if self.length == 1:
            self.next = lambda: self.colors[0]
    
    def next(self):
        color = self.colors[self.idx]
        self.idx = (self.idx + 1) % self.length
        return color
    
    def reset(self):
        self.idx = 0

    @staticmethod
    def color_gradient(colors, maxstep):
        """
        生成颜色渐变数组

        Parameters:
        - colors: np.ndarray 或 list, 形状为 (4,) 或 (n, 4)，表示 RGBA 颜色值。
        - maxstep: int, 生成的渐变数组长度。

        Returns:
        - np.ndarray, 形状为 (maxstep, 4)，渐变颜色数组, 如果输入是单个颜色，则返回形状为 (1, 4) 的数组。
        """
        colors = np.asarray(colors)
        if colors.ndim == 1:
            colors = colors.reshape(1, -1)
        
        if colors.shape[0] == 1:
            # 单个颜色，重复 maxstep 次
            return colors

        elif colors.shape[1] == 4:
            n = colors.shape[0]
            # 多个颜色，向量化线性插值
            x = np.arange(n)
            x_new = np.linspace(0, n-1, maxstep)
            gradient = np.zeros((maxstep, 4))
            for c in range(4):
                gradient[:, c] = np.interp(x_new, x, colors[:, c])
            return gradient
        else:
            raise ValueError("colors 必须是形状 (4,) 或 (n, 4) 的数组")



class VisualItem:

    def __init__(self, parent=None):
        self.visible = True
        self._transform = Matrix4x4()
        self._parent: VisualItem | None = None
        self._children = set()
        self._draw_once = False
        if parent is not None:
            parent.add_child(self)

    def draw_once(self, draw_once: bool=True):
        self._draw_once = draw_once
        return self
        
    # items tree 
    def add_child(self, child: 'VisualItem'):
        if child._parent is not None:
            child._parent.remove_child(child)
        child._parent = self
        self._children.add(child)
    
    def remove_child(self, child: 'VisualItem'):
        self._children.discard(child)
        
    def set_parent(self, parent: 'VisualItem'):
        if parent is None:
            return
        parent.add_child(self)

    @property
    def parent(self):
        return self._parent
    
    def iter_tree(self):
        yield self
        for child in self._children:
            yield from child.iter_tree()

    def iter_children(self):
        for child in self._children:
            yield from child.iter_tree()

    # -- attributes and transform
    def set_visible(self, visible: bool):
        self.visible = visible

    def set_transform(self, pose, local=True):
        """
        Set the transformation matrix for the visual item.

        Parameters:
        - pose : (list, tuple, np.ndarray), 3D position (x, y, z) or a Matrix4x4 instance
                If a 3D position is provided, the orientation is set to identity.
        - local : bool, if True, the pose is relative to the parent item; otherwise, it's in world coordinates.
        """
        if isinstance(pose, (list, tuple, np.ndarray)) and len(pose) == 3:
            pose = Matrix4x4.fromTranslation(*pose)
    
        if local or self._parent is None:
            self._transform = pose.copy()
        else:
            self._transform = self._parent.transform(False).inverse() * pose
            
    def transform(self, local=True) -> Matrix4x4:
        if local:
            return self._transform.copy()
        else:
            tf = self._transform
            parent = self.parent
            while parent is not None:
                tf = parent.transform() * tf
                parent = parent.parent
            return tf
    
    def moveto(self, x: float, y: float, z: float):
        self._transform.moveto(x, y, z)
        return self
    
    def translate(self, dx: float, dy: float, dz: float, local: bool = True):
        self._transform.translate(dx, dy, dz, local)
        return self
    
    def rotate(self, angle, x, y, z, local: bool = True):
        self._transform.rotate(angle, x, y, z, local)
        return self
    
    def apply_transform(self, mat: Matrix4x4, local: bool = True):
        self._transform = self._transform.applyTransform(mat, local)
        return self

    @staticmethod
    def next_geom(mjv_scene: mujoco.MjvScene):
        ngeom = mjv_scene.ngeom
        if ngeom >= mjv_scene.maxgeom:
            raise ValueError("Warning: Maximum number of geoms reached in the scene.")
        geom = mjv_scene.geoms[ngeom]
        mjv_scene.ngeom += 1
        return geom

    # add to scene
    def add_self_to_scene(self, mjv_scene):
        # raise NotImplementedError("This method should be implemented by subclasses.")
        pass
    
    def add_to_scene(self, mjv_scene):
        if not self.visible:
            return

        self.add_self_to_scene(mjv_scene)

        for child in self.iter_children():
            child.add_to_scene(mjv_scene)


class GeomItem(VisualItem):

    def __init__(
        self, 
        pose=Matrix4x4(), 
        size=(0.1, 0.1, 0.1), 
        rgba=(1, 0, 0, 1), 
        type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_BOX,
        label=None,
        parent=None
    ):
        super().__init__(parent)
        self.set_transform(pose)
        self._size = size
        self._rgba = rgba
        self._type = type
        self._label = label
    
    def add_self_to_scene(self, mjv_scene):
        geom = self.next_geom(mjv_scene)
        
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.type = self._type
        geom.size[:] = self._size
        geom.rgba[:] = self._rgba

        tf = self.transform(local=False)
        geom.mat[:] = tf.matrix33
        geom.pos[:] = tf.xyz

        if self._label:
            geom.label = self._label
        else:
            geom.label = ""  # No label
        geom.specular = 0.5
        geom.shininess = 0.5
        
    @classmethod
    def create_box(cls, pose=Matrix4x4(), size=(0.1, 0.1, 0.1), rgba=(1, 0, 0, 1), label=None, parent=None):
        if isinstance(size, (int, float)):
            size = (size, size, size) 
        return cls(pose=pose, size=(size[0]/2, size[1]/2, size[2]/2), rgba=rgba, type=mujoco.mjtGeom.mjGEOM_BOX, label=label, parent=parent)
    
    @classmethod
    def create_sphere(cls, pose=Matrix4x4(), radius=0.1, rgba=(1, 0, 0, 1), label=None, parent=None):
        return cls(pose=pose, size=(radius, radius, radius), rgba=rgba, type=mujoco.mjtGeom.mjGEOM_SPHERE, label=label, parent=parent)
    
    @classmethod
    def create_cylinder(cls, pose=Matrix4x4(), radius=0.1, length=0.2, rgba=(1, 0, 0, 1), label=None, radius2=None, parent=None):
        return cls(pose=pose, size=(radius, radius2 if radius2 is not None else radius, length/2), rgba=rgba, type=mujoco.mjtGeom.mjGEOM_CYLINDER, label=label, parent=parent)

    @classmethod
    def create_capsule(cls, pose=Matrix4x4(), radius=0.1, length=0.2, rgba=(1, 0, 0, 1), label=None, radius2=None, parent=None):
        return cls(pose=pose, size=(radius, radius2 if radius2 is not None else radius, length/2), rgba=rgba, type=mujoco.mjtGeom.mjGEOM_CAPSULE, label=label, parent=parent)

    @classmethod
    def create_label(cls, pose=Matrix4x4(), label=None, rgba=(1, 0, 0, 1), parent=None):
        return cls(pose=pose, size=(0, 0, 0), rgba=rgba, type=mujoco.mjtGeom.mjGEOM_LABEL, label=label, parent=parent)


class ConnectorItem(VisualItem):

    def __init__(
        self, 
        start: list | tuple | np.ndarray = (0, 0, 0), 
        end: list | tuple | np.ndarray = (1, 0, 0), 
        width: float = 0.02, 
        rgba=(0, 1, 0, 1), 
        type=mujoco.mjtGeom.mjGEOM_LINE,
        label=None,
        parent=None
    ):
        super().__init__(parent)
        self._type = type
        self._rgba = rgba
        self._label = label
        self._start = np.array(start)
        self._end = np.array(end)
        self._direction_mat = Matrix4x4.fromZaxisToDirection(self._end - self._start).matrix33
        length = np.linalg.norm(self._end - self._start)
        self._size = (width, width, length)
    
    def set_data(self, start: list | tuple | np.ndarray, end: list | tuple | np.ndarray):
        self._start = np.array(start)
        self._end = np.array(end)
        self._direction_mat = Matrix4x4.fromZaxisToDirection(self._end - self._start).matrix33
        length = np.linalg.norm(self._end - self._start)
        self._size[2] = length
        
    def add_self_to_scene(self, mjv_scene):
        geom = self.next_geom(mjv_scene)
        
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.type = self._type
        geom.size[:] = self._size
        geom.rgba[:] = self._rgba

        if self._label:
            geom.label = self._label
        else:
            geom.label = ""  # No label
        geom.specular = 0.5
        geom.shininess = 0.5
        
        # mat / pos        
        tf = self.transform(local=False)
        start = tf.matrix33 @ self._start + tf.xyz
        
        geom.mat[:] = tf.matrix33 @ self._direction_mat
        geom.pos[:] = start
    
    @classmethod
    def create_line(cls, start=(0, 0, 0), end=(1, 0, 0), width=0.02, rgba=(0, 1, 0, 1), label=None, parent=None):
        return cls(start=start, end=end, width=width, rgba=rgba, type=mujoco.mjtGeom.mjGEOM_LINE, label=label, parent=parent)
    
    @classmethod
    def create_arrow(cls, start=(0, 0, 0), end=(1, 0, 0), width=0.02, rgba=(0, 1, 0, 1), label=None, parent=None):
        end = start + (np.array(end) - np.array(start)) * 2
        return cls(start=start, end=end, width=width, rgba=rgba, type=mujoco.mjtGeom.mjGEOM_ARROW, label=label, parent=parent)

    @classmethod
    def create_arrow1(cls, start=(0, 0, 0), end=(1, 0, 0), width=0.02, rgba=(0, 1, 0, 1), label=None, parent=None):
        end = start + (np.array(end) - np.array(start)) * 2
        return cls(start=start, end=end, width=width, rgba=rgba, type=mujoco.mjtGeom.mjGEOM_ARROW1, label=label, parent=parent)

    @classmethod
    def create_arrow2(cls, start=(0, 0, 0), end=(1, 0, 0), width=0.02, rgba=(0, 1, 0, 1), label=None, parent=None):
        return cls(start=start, end=end, width=width, rgba=rgba, type=mujoco.mjtGeom.mjGEOM_ARROW2, label=label, parent=parent)

    

class AxisItem(VisualItem):
    
    def __init__(
        self,
        size = (0.15, 0.15, 0.15),
        width = 0.005,
        show_label=False,
        parent=None        
    ):
        super().__init__(parent)
        if isinstance(size, (int, float)):
            size = (size, size, size)
            
        self.z_axis = ConnectorItem.create_arrow(start=(0,0,0), end=(0,0,size[2]), width=width, rgba=(0,0,1,1), parent=self)
        self.y_axis = ConnectorItem.create_arrow(start=(0,0,0), end=(0,size[1],0), width=width, rgba=(0,1,0,1), parent=self)
        self.x_axis = ConnectorItem.create_arrow(start=(0,0,0), end=(size[0],0,0), width=width, rgba=(1,0,0,1), parent=self)
        if show_label:
            self.z_label = GeomItem.create_label(pose=(0,0,size[2]), label="Z", rgba=(0,0,1,1), parent=self)
            self.y_label = GeomItem.create_label(pose=(0,size[1],0), label="Y", rgba=(0,1,0,1), parent=self)
            self.x_label = GeomItem.create_label(pose=(size[0],0,0), label="X", rgba=(1,0,0,1), parent=self)
  
    

class MultiGeomItem(VisualItem):
    
    def __init__(
        self,
        item: VisualItem,
        pose = None,
        maxlen = 500,
        rgba = ((1,0.2,0.1,1), (0.4,0,0.7,1)),
        parent = None
    ):
        """
        A visual item that maintains a deque of positions.

        Parameters:
        - item : VisualItem, the visual item to be managed
        - pose :list or tuple or np.ndarray, optional, default: None, initial positions, can be:
            * Matrix4x4
            * List[Matrix4x4]
            * np.ndarray, shape=(n, 3), 只定义 xyz
            * np.ndarray, shape=(n, 4, 4), 完整的位姿矩阵
        - maxlen : int, optional, default: 500, maximum length of the deque
        """
    
        super().__init__(parent)
        self.maxlen = maxlen
        self.item = item
        self.item._label = None
        item.set_parent(self)

        self.deque = deque(maxlen=maxlen)
        self.add_data(pose)

        # 颜色渐变
        self.color_grad = ColorGradient(rgba, maxlen)

    def _make_pose(self, pose):
        """
        规范化输入的位姿矩阵
        """
        if pose is None:
            return None
        if isinstance(pose, Matrix4x4):
            return [pose]
        elif isinstance(pose, list) and len(pose) > 0 and isinstance(pose[0], Matrix4x4):
            return pose
        else:
            pose = np.asarray(pose)
            if pose.ndim == 1 and pose.shape[0] == 3:
                pose = pose.reshape(1, 3)
            
            if pose.ndim == 2 and pose.shape[1] == 3:
                return [Matrix4x4.fromTranslation(*p) for p in pose]
            elif pose.ndim == 3 and pose.shape[1:] == (4, 4):
                return [Matrix4x4(m) for m in pose]
            else:
                raise ValueError("Invalid pose data")

    def add_data(self, pose):
        """
        Add new positions to the deque.
        """
        poses = self._make_pose(pose)
        if poses is not None:
            self.deque.extend(poses)
    
    def set_data(self, pose):
        """
        Set the deque to new positions.
        """
        poses = self._make_pose(pose)
        if poses is not None:
            self.deque.clear()
            self.deque.extend(poses)
    
    def clear(self):
        self.deque.clear()

    def add_self_to_scene(self, mjv_scene):
        self.color_grad.reset()
        for pose in self.deque:
            self.item._transform = pose
            self.item._rgba = self.color_grad.next()
            self.item.add_self_to_scene(mjv_scene)


class LineItem(VisualItem):
    
    def __init__(
        self,
        pts: list | tuple | np.ndarray = None,
        width: int = 1,
        maxlen: int = 500,
        rgba = (0,1,0,1),
        parent=None
    ):
        super().__init__(parent)
        self._width = width
        self._rgba = rgba
        self._buffer = DataBuffer(maxlen=maxlen, shape=(3, 5), dtype=np.float32)
        self._last_pt = None
        self._data = deque(maxlen=maxlen)
        self.add_data(pts)
        
    def add_data(self, pts: list | tuple | np.ndarray):
        # ensure pts is (n, 3), n >= 1
        if pts is None:
            return
        pts = np.asarray(pts).reshape(-1, 3)
        if len(pts) == 0:
            return
        
        if self._last_pt is None:
            self._last_pt = pts[[0]].copy()
        
        if len(pts) > 1:
            starts = np.vstack([self._last_pt, pts[:-1]])
            ends = pts
        else:
            starts = self._last_pt
            ends = pts
    
        data3x5 = self._direction_matrixs(starts, ends)
        self._buffer.extend(data3x5)
        self._last_pt = pts[[-1]].copy()
    
    def set_data(self, pts: list | tuple | np.ndarray):
        self._buffer.clear()
        self._last_pt = None
        self.add_data(pts)
    
    def add_self_to_scene(self, mjv_scene):
        for data in self._buffer:
            geom = self.next_geom(mjv_scene)
        
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            geom.type = mujoco.mjtGeom.mjGEOM_LINE
            geom.size[:] = data[:, 4]
            geom.rgba[:] = self._rgba
            geom.label = ""  # No label
            geom.specular = 0.5
            geom.shininess = 0.5
            
            # mat / pos
            mat, start = data[:, :3], data[:, 3]
            tf = self.transform(local=False)
            
            geom.mat[:] = tf.matrix33 @ mat
            geom.pos[:] = tf.matrix33 @ start + tf.xyz

    def _direction_matrixs(self, starts, ends):
        """
        返回 shape=(n, 3, 5) 的变换矩阵, 用于 LineItem
        """
        starts = np.array(starts)
        ends = np.array(ends)
        arrows = ends - starts
        arrows = arrows.reshape(-1, 3)
        # 处理零向量，归一化
        arrow_lens = np.linalg.norm(arrows, axis=1)  # (n,)
        zero_idxs = arrow_lens < 1e-3
        arrows[zero_idxs] = [0, 0, 1]

        arrow_lens_den = arrow_lens.copy()
        arrow_lens_den[zero_idxs] = 1
        arrows = arrows / arrow_lens_den[:, np.newaxis]
        # 构造标准箭头到目标箭头的旋转矩阵
        T = np.zeros_like(arrows)
        B = np.zeros_like(arrows)
        mask = arrows[:, 2] < -0.99999
        T[mask, 1] = -1
        B[mask, 0] = -1
        mask = np.logical_not(mask)
        a = 1 / (1 + arrows[mask, 2])
        b = -arrows[mask, 0] * arrows[mask, 1] * a
        T[mask] = np.stack((
            1 - arrows[mask, 0] * arrows[mask, 0] * a,
            b,
            -arrows[mask, 0],
        ), axis=1)
        B[mask] = np.stack((
            b,
            1 - arrows[mask, 1] * arrows[mask, 1] * a,
            -arrows[mask, 1],
        ), axis=1)

        # 转化成齐次变换矩阵
        sizes = np.zeros_like(starts)
        sizes[:, :2] = self._width
        sizes[:, 2] = arrow_lens
        data3x5 = np.stack((T, B, arrows, starts.reshape(-1, 3), sizes), axis=2)  # (n, 3, 5)
        return data3x5
    
    
    
class VisualItemManager:
    
    def __init__(self):
        self.items = set()
    
    def add_to_scene(self, mjv_scene, enable_draw_once=True):
        for item in self.items.copy():
            item.add_to_scene(mjv_scene)
            if item._draw_once and enable_draw_once:
                self.items.remove(item)
    
    def add_item(self, *items):
        for item in items:
            self.items.add(item)

    def remove_item(self, *items):
        for item in items:
            if item in self.items:
                self.items.remove(item)
    
    def clear(self):
        self.items.clear()