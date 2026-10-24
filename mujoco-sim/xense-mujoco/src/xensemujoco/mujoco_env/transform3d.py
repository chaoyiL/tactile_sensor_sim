"""
Description: Module for 3D transformations using NumPy, including Quaternion and Matrix4x4 classes.

Author: Jin Liu
Date: 2025/09/28
"""

import numpy as np
from math import acos, degrees, atan2, asin, radians, cos, sin
from typing import Any, Union, List, Tuple, Sequence
from functools import singledispatchmethod, update_wrapper

class dispatchmethod(singledispatchmethod):
    """Dispatch a method to different implementations
    depending upon the type of its first argument.
    If there is no argument, use 'object' instead.
    """
    def __get__(self, obj, cls=None):
        def _method(*args, **kwargs):
            if len(args) > 0:
                class__ = args[0].__class__
            elif len(kwargs) > 0:
                class__ = next(kwargs.values().__iter__()).__class__
            else:
                class__ = object

            method = self.dispatcher.dispatch(class__)
            return method.__get__(obj, cls)(*args, **kwargs)

        _method.__isabstractmethod__ = self.__isabstractmethod__
        _method.register = self.register
        update_wrapper(_method, self.func)
        return _method


def mat33ToEuler(mat33: np.ndarray) -> np.ndarray:
    """ zyx euler system (degree): Rot(z) * Rot(y) * Rot(x), [degrees] """
    if mat33[2, 0] > 0.9999:
        x = 0
        y = -np.pi/2
        z = atan2(-mat33[0, 1], -mat33[0, 2])
    elif mat33[2, 0] < -0.9999:
        x = 0
        y = np.pi/2
        z = atan2(-mat33[0, 1], mat33[0, 2])
    else:
        x = atan2(mat33[2, 1], mat33[2, 2])
        y = atan2(-mat33[2, 0], np.sqrt(mat33[2, 1]**2 + mat33[2, 2]**2))
        z = atan2(mat33[1, 0], mat33[0, 0])
    angle = np.rad2deg(np.array([x, y, z]))
    eq_angle = np.array([angle[0]-180, 180-angle[1], angle[2]-180])
    if eq_angle[0] < -180:
        eq_angle[0] += 360
    if eq_angle[1] > 180:
        eq_angle[1] -= 360
    if eq_angle[2] < -180:
        eq_angle[2] += 360
    if np.linalg.norm(eq_angle) < np.linalg.norm(angle):
        return eq_angle
    return angle

def eulerToMat33(x, y, z) -> np.ndarray:
    "zyx euler system (degree): Rot(z) * Rot(y) * Rot(x), [degrees] "
    cx, sx = cos(radians(x)), sin(radians(x))
    cy, sy = cos(radians(y)), sin(radians(y))
    cz, sz = cos(radians(z)), sin(radians(z))
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def axisAngleToRotationMatrix(x: float, y: float, z: float, angle: float) -> np.ndarray:
    """
    Convert axis-angle representation to 3x3 rotation matrix using Rodrigues' formula.
    
    Parameters:
    - x, y, z: axis components
    - angle: rotation angle in degrees
    
    Returns:
    - 3x3 rotation matrix
    """
    axis = np.array([x, y, z], dtype=np.float32)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        return np.eye(3, dtype=np.float32)  # Identity matrix for zero rotation
    
    axis = axis / axis_norm
    angle_rad = np.radians(angle)
    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)
    
    # Create skew-symmetric matrix K
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=np.float32)
    
    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    K2 = K @ K
    return np.eye(3, dtype=np.float32) + sin_a * K + (1 - cos_a) * K2


def r2w(R):
    """
    R to \omega
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()     


class Quaternion:
    """
    Pure NumPy implementation of Quaternion with the same API as Qt version.
    """

    def __init__(self, *args):
        """Initialize quaternion with various input formats"""
        if len(args) == 0:
            # Default constructor
            self._data = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif len(args) == 1:
            # Single argument - array-like
            if isinstance(args[0], (list, tuple, np.ndarray)):
                quat_values = np.array(args[0], dtype=np.float32).flatten()
                if len(quat_values) == 4:
                    self._data = quat_values
                else:
                    raise ValueError("Array must have 4 elements for quaternion")
            elif isinstance(args[0], Quaternion):
                self._data = args[0]._data.copy()
            else:
                raise ValueError("Invalid argument type for quaternion")
        elif len(args) == 4:
            # Four scalar arguments: w, x, y, z
            self._data = np.array(args, dtype=np.float32)
        else:
            raise ValueError("Invalid number of arguments for quaternion")

    def __repr__(self) -> str:
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        return f"Quaternion(w: {self.scalar()}, x: {self.x()}, y: {self.y()}, z: {self.z()})"

    def copy(self):
        return Quaternion(*self._data)

    def conjugate(self):
        """conjugate of quaternion"""
        return Quaternion(self._data[0], -self.x(), -self.y(), -self.z())

    def inverse(self):
        """inverse of quaternion (same as conjugate for unit quaternion)"""
        return self.conjugate()

    def normalize(self):
        norm = np.linalg.norm(self._data)
        if norm > 0:
            return Quaternion(self._data / norm)
        return self.copy()

    def toEulerAngles(self):
        """ zyx euler system (degree): Rot(z) * Rot(y) * Rot(x), [degrees] """
        # Normalize quaternion
        norm = np.linalg.norm(self._data)
        if norm == 0:
            return np.array([0.0, 0.0, 0.0])
        w, x, y, z = self._data / norm

        # Compute Euler angles directly from quaternion
        # roll (x)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = atan2(sinr_cosp, cosr_cosp)

        # pitch (y)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = asin(sinp)

        # yaw (z)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        yaw = atan2(siny_cosp, cosy_cosp)

        angle = np.rad2deg(np.array([roll, pitch, yaw]))

        # Equivalent angle calculation to match mat33ToEuler behavior
        eq_angle = np.array([angle[0] - 180, 180 - angle[1], angle[2] - 180])
        if eq_angle[0] < -180:
            eq_angle[0] += 360
        if eq_angle[1] > 180:
            eq_angle[1] -= 360
        if eq_angle[2] < -180:
            eq_angle[2] += 360
        if np.linalg.norm(eq_angle) < np.linalg.norm(angle):
            return eq_angle
        return angle

    def toRotationMatrix(self) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix"""
        w, x, y, z = self._data

        # Normalize quaternion
        norm = np.linalg.norm(self._data)
        if norm > 0:
            w, x, y, z = self._data / norm

        # Compute rotation matrix
        xx, xy, xz = x*x, x*y, x*z
        yy, yz, zz = y*y, y*z, z*z
        wx, wy, wz = w*x, w*y, w*z

        mat33 = np.array([
            [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
            [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
        ], dtype=np.float32)

        return mat33

    def toMatrix4x4(self):
        matrix3x3 = self.toRotationMatrix()
        matrix = np.identity(4, dtype=np.float32)
        matrix[:3, :3] = matrix3x3
        return Matrix4x4(matrix)

    @classmethod
    def fromEulerAngles(cls, x, y, z):
        "zyx euler system (degree): Rot(z) * Rot(y) * Rot(x), [degrees] "
        return cls.fromMatrix3x3(eulerToMat33(x, y, z))

    @classmethod
    def fromEulerAnglesOrder(cls, x, y, z, order=[2, 1, 0]):
        " euler system (degree), default zyx system "
        cx, sx = cos(radians(x)), sin(radians(x))
        cy, sy = cos(radians(y)), sin(radians(y))
        cz, sz = cos(radians(z)), sin(radians(z))
        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        rots = [rot_x, rot_y, rot_z]
        mat33 = rots[order[0]] @ rots[order[1]] @ rots[order[2]]
        return cls.fromMatrix3x3(mat33)

    @classmethod
    def fromAxisAndAngle(cls, x=0., y=0., z=0., angle=0.):
        """Create quaternion from axis-angle representation"""
        axis = np.array([x, y, z], dtype=np.float32)
        axis_norm = np.linalg.norm(axis)

        if axis_norm == 0 or angle == 0:
            return cls()

        axis = axis / axis_norm
        half_angle = radians(angle) / 2.0
        sin_half = sin(half_angle)
        cos_half = cos(half_angle)

        return cls(cos_half, axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half)

    @classmethod
    def fromMatrix4x4(cls, matrix: 'Matrix4x4'):
        matrix3x3 = matrix.matrix33
        return cls.fromMatrix3x3(matrix3x3)

    @classmethod
    def fromMatrix3x3(cls, matrix: np.ndarray):
        """Convert 3x3 rotation matrix to quaternion"""
        m = matrix.astype(np.float32)

        # Compute trace
        trace = np.trace(m)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        else:
            if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
                w = (m[2, 1] - m[1, 2]) / s
                x = 0.25 * s
                y = (m[0, 1] + m[1, 0]) / s
                z = (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
                w = (m[0, 2] - m[2, 0]) / s
                x = (m[0, 1] + m[1, 0]) / s
                y = 0.25 * s
                z = (m[1, 2] + m[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
                w = (m[1, 0] - m[0, 1]) / s
                x = (m[0, 2] + m[2, 0]) / s
                y = (m[1, 2] + m[2, 1]) / s
                z = 0.25 * s

        return cls(w, x, y, z)

    # operators
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            # Quaternion multiplication
            w1, x1, y1, z1 = self._data
            w2, x2, y2, z2 = other._data

            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2

            return Quaternion(w, x, y, z)

        # apply rotation to vectors np.array([[x1,y1,z1], [x2,y2,z2], ...])
        mat = self.toMatrix4x4()
        return mat * other

    def slerp(self, other, t) -> 'Quaternion':
        """
        计算两个四元数之间的插值点, t in [0, 1]

        Parameters:
        - other : Quaternion,
        - t : float in [0, 1]
        """
        # Normalize quaternions
        q1 = self.normalize()
        q2 = other.normalize()

        # Compute dot product
        dot = np.dot(q1._data, q2._data)

        # If dot product is negative, negate one quaternion to take shorter path
        if dot < 0.0:
            q2 = Quaternion(-q2.scalar(), -q2.x(), -q2.y(), -q2.z())
            dot = -dot

        # Clamp dot product to avoid numerical errors
        dot = np.clip(dot, -1.0, 1.0)

        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1._data + t * (q2._data - q1._data)
            return Quaternion(result).normalize()

        # Spherical linear interpolation
        theta_0 = acos(dot)
        theta = theta_0 * t
        sin_theta = sin(theta)
        sin_theta_0 = sin(theta_0)

        s1 = cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0

        result = s1 * q1._data + s2 * q2._data
        return Quaternion(result)

    def dot(self, other) -> float:
        return np.dot(self._data, other._data)

    def __neg__(self):
        return Quaternion(-self._data[0], -self.x(), -self.y(), -self.z())

    def scalar(self):
        return self._data[0]

    def x(self):
        return self._data[1]

    def y(self):
        return self._data[2]

    def z(self):
        return self._data[3]

    def vector(self):
        return self._data[1:4]


class Matrix4x4:
    """
    Pure NumPy implementation of 4x4 transformation matrix with the same API as Qt version.
    """

    def __init__(self, data=None):
        """Initialize 4x4 matrix"""
        if data is None:
            self._data = np.eye(4, dtype=np.float32)
        elif isinstance(data, Matrix4x4):
            self._data = data._data.copy()
        else:
            data = np.array(data, dtype=np.float32)
            if data.shape == (4, 4):
                self._data = data.copy()
            elif data.size == 16:
                self._data = data.reshape(4, 4)
            else:
                raise ValueError("Data must be 4x4 matrix or 16-element array")

            # Validate that this is a valid homogeneous transformation matrix
            # self._validate_homogeneous_matrix()

    def _validate_homogeneous_matrix(self):
        """Validate that the matrix is a proper homogeneous transformation matrix"""
        # Check that the last row is [0, 0, 0, 1]
        expected_last_row = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if not np.allclose(self._data[3, :], expected_last_row, atol=1e-6):
            raise ValueError(f"Invalid homogeneous transformation matrix: last row must be [0, 0, 0, 1], "
                           f"but got {self._data[3, :]}")

        # Check that the top-left 3x3 is a valid rotation matrix
        rotation_matrix = self._data[:3, :3]

        # Check orthogonality: R^T * R should be close to identity
        rt_r = rotation_matrix.T @ rotation_matrix
        identity = np.eye(3, dtype=np.float32)
        if not np.allclose(rt_r, identity, atol=1e-6):
            raise ValueError(f"Invalid homogeneous transformation matrix: top-left 3x3 must be orthogonal, "
                           f"but R^T * R = \n{rt_r}")

        # Check determinant is close to 1 (right-handed rotation)
        det = np.linalg.det(rotation_matrix)
        if not np.isclose(det, 1.0, atol=1e-6):
            raise ValueError(f"Invalid homogeneous transformation matrix: determinant of rotation matrix "
                           f"must be 1.0 (right-handed), but got {det}")

    def copy(self):
        return Matrix4x4(self._data.copy())

    def __len__(self):
        return 16

    def __getitem__(self, i):
        if isinstance(i, tuple):
            row, col = i
            return self._data[row, col]
        return self._data.flatten()[i]

    def __setitem__(self, i, value):
        if isinstance(i, tuple):
            row, col = i
            self._data[row, col] = value
        else:
            self._data.flat[i] = value

    @property
    def matrix33(self):
        return self._data[:3, :3].copy()

    @matrix33.setter
    def matrix33(self, mat33):
        self._data[:3, :3] = mat33
        
    @property
    def matrix44(self):
        return self._data.copy()

    @property
    def array(self):
        return self._data.flatten().copy()

    @property
    def xyz(self):
        return self._data[:3, 3].copy()

    @xyz.setter
    def xyz(self, vec3):
        self._data[:3, 3] = vec3

    @property
    def quat(self):
        """convert to quaternion"""
        return Quaternion.fromMatrix4x4(self)

    @property
    def quat_data(self):
        """(qw, qx, qy, qz)"""
        return self.quat._data        
    
    @property
    def euler(self):
        """degree"""
        return self.toEulerAngles()

    @property
    def glData(self):
        """convert to column-major order for use with OpenGL"""
        return self._data.T.flatten()

    @classmethod
    def dist(cls, m1, m2):
        return np.linalg.norm(m1.xyz - m2.xyz)

    def __repr__(self) -> str:
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        return f"Matrix4x4(\n{self.matrix44}\n)"

    def rotate(self, *args, local=True):
        """rotate by quaternion or (angle, x, y, z) in local space, it will change the current matrix"""
        if len(args) > 4:
            local = args[4]
        
        if len(args) == 1 and isinstance(args[0], Quaternion):
            mat44 = args[0].toMatrix4x4()._data
        
        elif len(args) >= 4:
            angle = args[0]
            x, y, z = args[1], args[2], args[3]
            
            mat44 = np.eye(4, dtype=np.float32)
            mat44[:3, :3] = axisAngleToRotationMatrix(x, y, z, angle)
            
        else:
            raise ValueError("Invalid arguments for rotate method")
            
        if local:
            # Local rotation: M' = M * R
            self._data[:] = self._data @ mat44
        else:
            # Global rotation: M' = R * M
            self._data[:] = mat44 @ self._data
        return self

    def rotate_inplace(self, *args):
        """move to origin and rotate, then move back, it will change the current matrix"""
        xyz = self.xyz.copy()
        self.xyz = (0, 0, 0)
        self.rotate(*args, local=False)
        self.xyz = xyz
        return self
    
    def translate(self, x, y, z, local=True):
        """translate by x,y,z in local space, it will change the current matrix"""
        trans = np.array([x, y, z], dtype=np.float32)
        if local:
            # Local translation: M' = M * T, so translation is transformed by current rotation
            self._data[:3, 3] += self._data[:3, :3] @ trans
        else:
            # Global translation: M' = T * M, so translation is added directly
            self._data[:3, 3] += trans
        return self

    def scale(self, x, y, z, local=True):
        """scale by x,y,z in local space, it will change the current matrix"""
        scale_matrix = np.diag([x, y, z, 1.0]).astype(np.float32)
        if local:
            # Local scaling: M' = M * S
            self._data = self._data @ scale_matrix
        else:
            # Global scaling: M' = S * M
            self._data = scale_matrix @ self._data
        return self

    def moveto(self, x, y, z):
        """move to x,y,z in world space, it will change the current matrix"""
        self._data[:3, 3] = np.array([x, y, z], dtype=np.float32)
        return self

    def setData(self, matrix: Union[np.ndarray, 'Matrix4x4']):
        """set matrix data"""
        if isinstance(matrix, Matrix4x4):
            self._data = matrix._data.copy()
        else:
            self._data = np.array(matrix, dtype=np.float32).reshape(4, 4)

    def applyTransform(self, matrix, local=False):
        """apply transform to self"""
        if isinstance(matrix, Matrix4x4):
            mat_data = matrix._data
        else:
            mat_data = np.array(matrix, dtype=np.float32).reshape(4, 4)

        if local:
            return Matrix4x4(self._data @ mat_data)
        else:
            return Matrix4x4(mat_data @ self._data)

    @classmethod
    def fromRotTrans(cls, R: np.ndarray, t=None):
        """rotate by R(3x3) and translate by t"""
        if t is None:
            t = np.zeros(3, dtype='f4')
        data = np.eye(4, dtype=np.float32)
        data[:3, :3] = R.astype(np.float32)
        data[:3, 3] = np.array(t, dtype=np.float32)
        return cls(data)

    @classmethod
    def fromEulerAngles(cls, x, y, z):
        """ zyx euler system (degree): Rot(z) * Rot(y) * Rot(x), [degrees] """
        mat44 = np.eye(4, dtype=np.float32)
        mat44[:3, :3] = eulerToMat33(x, y, z)
        return cls(mat44)

    @classmethod
    def fromTranslation(cls, x=0., y=0., z=0.):
        """translate by x,y,z"""
        mat = cls()
        mat.moveto(x, y, z)
        return mat

    @classmethod
    def fromScale(cls, x=1., y=1., z=1.):
        """scale by x,y,z"""
        mat = cls()
        mat.scale(x, y, z)
        return mat

    @classmethod
    def fromAxisAndAngle(cls, x=0., y=0., z=0., angle=0.):
        """rotate by angle(degree) around x,y,z - optimized version"""
        if angle == 0 or (x == 0 and y == 0 and z == 0):
            return cls()
        
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = axisAngleToRotationMatrix(x, y, z, angle)
        return cls(mat)

    @classmethod
    def fromQuaternion(cls, q):
        """rotate by quaternion"""
        mat = cls()
        mat.rotate(q)
        return mat

    @classmethod
    def fromTransRotVector(cls, x, y, z, a, b, c) -> "Matrix4x4":
        """ axis-angle system (radius): x, y, z with mod """
        axis_angle = np.array([a, b, c], dtype=np.float32)
        angle = np.linalg.norm(axis_angle)
        if angle == 0:
            axis = np.array([1, 0, 0], dtype=np.float32)
        else:
            axis = axis_angle / angle
        return cls.fromAxisAndAngle(axis[0], axis[1], axis[2], np.degrees(angle)).moveto(x, y, z)

    @classmethod
    def fromZaxisToDirection(cls, direction) -> "Matrix4x4":
        """ create rotation matrix to rotate z-axis to given direction """
        direction = np.array(direction, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm == 0 or np.allclose(direction, [0, 0, 1]):
            return cls()
        e3 = direction / norm
        
        # 选择临时向量
        temp = np.array([0, 0, 1]) if abs(e3[2]) < 0.8 else np.array([1, 0, 0])
        # 构造 e1 (Gram-Schmidt)
        e1 = temp - np.dot(temp, e3) * e3
        e1 = e1 / np.linalg.norm(e1)
        
        # e2 = e3 × e1
        e2 = np.cross(e3, e1)
        R = np.column_stack((e1, e2, e3))
        return cls.fromRotTrans(R)

    @classmethod
    def fromVector6d(cls, x, y, z, a, b, c) -> "Matrix4x4":
        """ zyx euler system (degree): Rot(z) * Rot(y) * Rot(x), [degrees] """
        return cls.fromEulerAngles(a, b, c).moveto(x, y, z)

    @classmethod
    def fromVector7d(cls, x, y, z, qw, qx, qy, qz) -> "Matrix4x4":
        """rotate by quaternion"""
        return cls.fromQuaternion(Quaternion([qw, qx, qy, qz])).moveto(x, y, z)

    def inverse(self):
        """Return inverse of the matrix"""
        inv_data = np.linalg.inv(self._data)
        return Matrix4x4(inv_data)

    def transpose(self) -> 'Matrix4x4':
        """transpose the matrix"""
        return Matrix4x4(self._data.T)

    def toQuaternion(self):
        """convert to quaternion"""
        return Quaternion.fromMatrix4x4(self)

    def toEulerAngles(self):
        """degree"""
        return mat33ToEuler(self.matrix33)

    def toTranslation(self):
        return self.xyz.copy()

    def toTransRotVector(self):
        """
        [translation, rotation vector]: 3D 轴角向量 (x, y, z, rvx, rvy, rvz), 单位为弧度，长度为旋转角度
        """
        R = self.matrix33
        # 计算旋转角度 theta
        trace = np.trace(R)
        theta = np.arccos(np.clip((trace - 1) / 2.0, -1.0, 1.0))

        # 如果角度非常小，说明是单位阵（旋转角为0），轴角为零向量
        if np.isclose(theta, 0):
            return np.concatenate([self.xyz, np.zeros(3)])

        # 若角度接近 pi，需要特殊处理（方向可能不唯一）
        if np.isclose(theta, np.pi):
            # 对角线元素最大的一项作为轴
            R_plus = (R + np.eye(3)) / 2
            axis = np.sqrt(np.diag(R_plus))
            axis *= np.sign([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
            axis = axis / np.linalg.norm(axis)
        else:
            # 正常情况，根据反对称部分求轴
            axis = np.array([
                R[2,1] - R[1,2],
                R[0,2] - R[2,0],
                R[1,0] - R[0,1]
            ]) / (2 * np.sin(theta))
        T = self.toTranslation()
        return np.concatenate([T, axis * theta])

    def toVector6d(self):
        """zyx euler system (degree), return [x, y, z, pitch, yaw, roll]"""
        t = self.toTranslation()
        abc = self.toEulerAngles()
        return np.array([*t, *abc])

    def toVector7d(self):
        """return list [x, y, z, qw, qx, qy, qz]"""
        t = self.toTranslation()
        q = self.toQuaternion()
        return np.array([*t, q.scalar(), q.x(), q.y(), q.z()])

    def __mul__(self, other):
        if isinstance(other, Matrix4x4):
            return Matrix4x4(self._data @ other._data)
        elif isinstance(other, Quaternion):
            mat = Matrix4x4(self._data.copy())
            mat.rotate(other)
            return mat
        # Check if other is a Matrix4x4 from another library (by class name)
        elif type(other).__name__.endswith('Matrix4x4'):
            other_mat4x4 = Matrix4x4(other.matrix44)
            return Matrix4x4(self._data @ other_mat4x4._data)
        
        # Convert to numpy array if needed
        if isinstance(other, (list, tuple)):
            other = np.array(other, dtype=np.float32)
        assert isinstance(other, np.ndarray), f"unsupported type {type(other)}"

        # apply rotation to vectors v = np.array([[x1,y1,z1], [x2,y2,z2], ...]), n,3
        # v * R.T + t
        origin_shape = other.shape
        if other.ndim == 1:
            other = other[None, :]

        channels = other.shape[1]

        if channels == 3:
            # Pad with homogeneous coordinate
            other_padded = np.pad(other, ((0, 0), (0, 1)), constant_values=1.0)
            ret = other_padded @ self._data.T
            # Normalize homogeneous coordinates
            ret /= np.maximum(ret[:, [3]], 1e-8)
            return ret[:, :3].reshape(origin_shape)
        else:
            # Assume homogeneous coordinates
            ret = other @ self._data.T
            ret /= np.maximum(ret[:, [3]], 1e-8)
            ret[:, 3] = 1
            return ret.reshape(origin_shape)

    def __sub__(self, m2):
        if isinstance(m2, Matrix4x4):
            return Matrix4x4(self._data - m2._data)
        return Matrix4x4(self._data - np.array(m2, dtype=np.float32).reshape(4, 4))

    def __add__(self, m2):
        if isinstance(m2, Matrix4x4):
            return Matrix4x4(self._data + m2._data)
        return Matrix4x4(self._data + np.array(m2, dtype=np.float32).reshape(4, 4))

    def interp(self, other: 'Matrix4x4', t) -> 'Matrix4x4':
        """
        计算两个点之间的插值点， t in [0, 1]

        Parameters:
        - other : Matrix4x4, 另一个点
        - t : float, 插值系数, other 所占的比例

        Returns:
        - Matrix4x4
        """
        quat1 = self.quat
        quat2 = other.quat
        quat_interp = quat1.slerp(quat2, t)
        xyz = self.xyz * (1 - t) + other.xyz * t

        return Matrix4x4().rotate(quat_interp).moveto(*xyz)

    @classmethod
    def perspective(cls, angle: float, aspect: float, near: float, far: float) -> 'Matrix4x4':
        """Create perspective projection matrix"""
        fov_rad = np.radians(angle)
        f = 1.0 / np.tan(fov_rad / 2.0)

        proj = np.eye(4, dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        proj[3, 3] = 0.0

        return cls(proj)

    @classmethod
    def ortho(cls, left, right, bottom, top, near, far) -> 'Matrix4x4':
        """Create orthographic projection matrix"""
        ortho = np.eye(4, dtype=np.float32)
        ortho[0, 0] = 2.0 / (right - left)
        ortho[0, 3] = -(right + left) / (right - left)
        ortho[1, 1] = 2.0 / (top - bottom)
        ortho[1, 3] = -(top + bottom) / (top - bottom)
        ortho[2, 2] = -2.0 / (far - near)
        ortho[2, 3] = -(far + near) / (far - near)

        return cls(ortho)

    @classmethod
    def lookAt(cls, eye: Sequence, center: Sequence, up: Sequence):
        """Create look-at matrix"""
        eye = np.array(eye, dtype=np.float32)
        center = np.array(center, dtype=np.float32)
        up = np.array(up, dtype=np.float32)

        # Forward vector
        f = center - eye
        f = f / np.linalg.norm(f)

        # Right vector
        r = np.cross(f, up)
        r = r / np.linalg.norm(r)

        # Up vector
        u = np.cross(r, f)

        # Translation
        trans = -np.dot(r, eye), -np.dot(u, eye), np.dot(f, eye)

        # Create matrix
        mat = np.eye(4, dtype=np.float32)
        mat[0, :3] = r
        mat[1, :3] = u
        mat[2, :3] = -f
        mat[:3, 3] = trans

        return cls(mat)
    
    def diff(self, other: 'Matrix4x4') -> tuple:
        """计算两个矩阵的平移和旋转差异"""
        t_err = self.xyz - other.xyz
        
        R_err = other.matrix33.T @ self.matrix33
        w_err = other.matrix33 @ r2w(R_err)
        return np.concatenate([t_err, w_err])

     


class Vector3():

    @dispatchmethod
    def __init__(self, x: float = 0., y: float = 0., z: float = 0.):
        self._data = np.array([x, y, z], dtype='f4')

    @__init__.register(np.ndarray)
    @__init__.register(list)
    @__init__.register(tuple)
    def _(self, data):
        self._data = np.array(data, dtype='f4').flatten()[:3]

    def copy(self):
        return Vector3(self._data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return 3

    def __repr__(self):
        return f"Vec3({self.x:.3g}, {self.y:.3g}, {self.z:.3g})"

    def __sub__(self, other):
        return Vector3(self._data - other._data)

    def __add__(self, other):
        return Vector3(self._data + other._data)

    def __isub__(self, other):
        self._data -= other._data
        return self

    def __iadd__(self, other):
        self._data += other._data
        return self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector3(self._data * other)
        elif isinstance(other, (Vector3, np.ndarray, list, tuple)):
            return Vector3(self._data[0] * other[0],
                           self._data[1] * other[1],
                           self._data[2] * other[2])
        else:
            raise TypeError(f"unsupported type {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self._data *= other
        elif isinstance(other, (Vector3, np.ndarray, list, tuple)):
            self._data[0] *= other[0]
            self._data[1] *= other[1]
            self._data[2] *= other[2]
        else:
            raise TypeError(f"unsupported type {type(other)}")
        return self

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector3(self._data / other)
        elif isinstance(other, (Vector3, np.ndarray, list, tuple)):
            return Vector3(self._data[0] / other[0],
                           self._data[1] / other[1],
                           self._data[2] / other[2])
        else:
            raise TypeError(f"unsupported type {type(other)}")

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            self._data /= other
        elif isinstance(other, (Vector3, np.ndarray, list, tuple)):
            self._data[0] /= other[0]
            self._data[1] /= other[1]
            self._data[2] /= other[2]
        else:
            raise TypeError(f"unsupported type {type(other)}")
        return self

    def __neg__(self):
        return Vector3(-self._data)

    @property
    def xyz(self):
        return self._data

    @property
    def x(self):
        return self._data[0]

    @x.setter
    def x(self, val):
        self._data[0] = val

    @property
    def y(self):
        return self._data[1]

    @y.setter
    def y(self, val):
        self._data[1] = val

    @property
    def z(self):
        return self._data[2]

    @z.setter
    def z(self, val):
        self._data[2] = val

    @property
    def norm(self):
        return np.linalg.norm(self._data)

    def normalize(self, length=1):
        self._data = self._data * (length / self.norm)
        return self

    def __getitem__(self, i):
        if i > 2:
            raise IndexError("Point has no index %s" % str(i))
        return self._data[i]

    def __setitem__(self, i, x):
        self._data[i] = x

    @classmethod
    def fromPolar(cls, r, theta, phi) -> 'Vector3':
        """theta, phi in degrees"""
        theta = np.radians(theta)
        phi = np.radians(phi)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return cls(x, y, z)

    def toPolar(self) -> tuple:
        """theta, phi in degrees"""
        r = np.linalg.norm(self._data)
        theta = degrees(acos(self._data[0] / r))
        phi = degrees(acos(self._data[2] / r))
        return r, theta, phi

@Vector3.__init__.register
def _(self, v: Vector3):
    self._data = np.array(v._data, dtype='f4')