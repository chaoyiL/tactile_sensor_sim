import os
import time
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Optional, Dict, List, Tuple, Any, Union
import logging
import copy
import contextlib

import mujoco
import glfw
import cv2
import numpy as np
import imgui

from .visual_item import VisualItemManager, AxisItem, GeomItem, LineItem, ConnectorItem, MultiGeomItem
from .imgui_blueprint import Blueprint
from .transform3d import Matrix4x4

MUJOCO_VERSION = tuple(map(int, mujoco.__version__.split('.')))


class DisplayPosition(Enum):
    TOP_LEFT = mujoco.mjtGridPos.mjGRID_TOPLEFT
    TOP = mujoco.mjtGridPos.mjGRID_TOP
    TOP_RIGHT = mujoco.mjtGridPos.mjGRID_TOPRIGHT
    BOTTOM_LEFT = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
    BOTTOM = mujoco.mjtGridPos.mjGRID_BOTTOM
    BOTTOM_RIGHT = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT


class FigureManager:
    def __init__(self):
        self.figures = {}  # dict: DisplayPosition -> MjvFigure

    def plot(self, position: DisplayPosition, x, y, line_idx=0, linergb=(0,0,1), linename='Line', **params):
        if position not in self.figures:
            fig = mujoco.MjvFigure()
            mujoco.mjv_defaultFigure(fig)
            fig.flg_extend = 1
            fig.figurergba = (1, 1, 1, 0)
            fig.panergba = (1, 1, 1, 0.2)
            self.figures[position] = fig
        else:
            fig = self.figures[position]
        
        L = len(x)  #  this cannot exceed 'mujoco.mjMAXLINEPNT'
        if L > mujoco.mjMAXLINEPNT:
            raise ValueError(f"Length of x ({L}) exceeds mujoco.mjMAXLINEPNT ({mujoco.mjMAXLINEPNT})")
        
        data = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1).flatten()
        fig.linedata[line_idx][:2*L] = data
        fig.linergb[line_idx] = linergb
        fig.linename[line_idx] = linename
        fig.linepnt[line_idx] = L

        for key, value in params.items():
            if hasattr(fig, key):
                setattr(fig, key, value)

    def render(self, ctx, win_width, win_height):
        for pos, fig in self.figures.items():
            if pos.value == mujoco.mjtGridPos.mjGRID_TOPLEFT:
                left = 0
                bottom = int(win_height * 0.75)
            elif pos.value == mujoco.mjtGridPos.mjGRID_TOPRIGHT:
                left = int(win_width * 0.75)
                bottom = int(win_height * 0.75)
            elif pos.value == mujoco.mjtGridPos.mjGRID_BOTTOMLEFT:
                left = 0
                bottom = 0
            else:
                left = int(win_width * 0.75)
                bottom = 0
            viewport = mujoco.MjrRect(left, bottom, int(win_width / 4), int(win_height / 4))
            mujoco.mjr_figure(viewport, fig, ctx)

    def clear(self):
        self.figures.clear()


class ImageManager:
    def __init__(self):
        self.images = {}

    def add_image(self, position: DisplayPosition, image):
        self.images[position.value] = image

    def render(self, ctx, win_width, win_height):
        h_overlay, w_overlay = int(win_height // 4), int(win_width // 4)

        for gridpos, img in self.images.items():
            h_raw, w_raw = img.shape[:2]
            scale = min(w_overlay / w_raw, h_overlay / h_raw)
            w_new = int(w_raw * scale)
            h_new = int(h_raw * scale)
            
            if gridpos == mujoco.mjtGridPos.mjGRID_TOPLEFT:
                left, bottom = 0, win_height - h_new
            elif gridpos == mujoco.mjtGridPos.mjGRID_TOPRIGHT:
                left, bottom = win_width - w_new, win_height - h_new
            elif gridpos == mujoco.mjtGridPos.mjGRID_BOTTOMLEFT:
                left, bottom = 0, 0
            elif gridpos == mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT:
                left, bottom = win_width - w_new, 0
            else:
                continue  # Skip unsupported positions for images

            viewport = mujoco.MjrRect(left, bottom, w_new, h_new)
            img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, gridpos, viewport, "", "", ctx)  # HACK: 如果不先调用这个函数，图像无法显示
            mujoco.mjr_drawPixels(np.flipud(img).flatten(), None, viewport, ctx)

    def clear(self):
        self.images.clear()


class TextManager:
    def __init__(self):
        self.overlays = {}

    def add_text(self, position: DisplayPosition, text1='', text2='', clear_buffer=False):
        gridpos = position.value
        if gridpos not in self.overlays or clear_buffer:
            self.overlays[gridpos] = ['', '']
        if text1:
            self.overlays[gridpos][0] += ('' if not self.overlays[gridpos][0] else '\n') + text1
        if text2:
            self.overlays[gridpos][1] += ('' if not self.overlays[gridpos][1] else '\n') + text2

    def render(self, mjr_context, viewport):
        for gridpos, [t1, t2] in self.overlays.items():
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, gridpos, viewport, t1, t2, mjr_context)

    def clear(self):
        self.overlays.clear()


class MinimalCallbacks:
    def __init__(self):
        self.gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._key_callbacks = {}
        
    def add_key_callback(self, key, callback):
        self._key_callbacks[key] = callback

    def _char_callback(self, window, char):
        if self.control_panel is not None and imgui.get_io().want_capture_keyboard:
            imgui.get_io().char_callback(window, char)
            return

    def _key_callback(self, window, key, scancode, action, mods):
        # if self.control_panel is not None and imgui.get_io().want_capture_keyboard:
        #     imgui.get_io().keyboard_callback(window, key, scancode, action, mods)
        #     return
        
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        if key in self._key_callbacks and (action == glfw.PRESS or action == glfw.REPEAT):
            self._key_callbacks[key]()

    def _cursor_pos_callback(self, window, xpos, ypos):
        if self.control_panel is not None and imgui.get_io().want_capture_mouse:
            return
        
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                    glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

        if self._button_right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        dx = int(xpos) - self._last_mouse_x
        dy = int(ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self.gui_lock:
            mujoco.mjv_moveCamera(self.mj_model, action, dx/width, dy/height, self.mjv_scene, self.mjv_camera)

        self._last_mouse_x = int(xpos)
        self._last_mouse_y = int(ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        if self.control_panel is not None and imgui.get_io().want_capture_mouse:
            return
        
        self._button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        self._button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(x)
        self._last_mouse_y = int(y)

    def _scroll_callback(self, window, x_offset, y_offset):
        if self.control_panel is not None and imgui.get_io().want_capture_mouse:
            imgui.get_io().mouse_wheel += y_offset
            return
        
        with self.gui_lock:
            mujoco.mjv_moveCamera(self.mj_model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset, self.mjv_scene, self.mjv_camera)




class MujocoViewer(MinimalCallbacks):
    def __init__(
        self, 
        mj_model, 
        mj_data, 
        win_width=800, 
        win_height=600, 
        maxgeom=10000, 
        offscreen=False,
        title="MuJoCo Viewer",
        control_panel: Blueprint = None
    ):
        super().__init__()

        self.mj_model: mujoco.MjModel = mj_model
        self.mj_data: mujoco.MjData = mj_data
        self.offscreen = offscreen
        self.is_alive = True
        self.gui_lock = Lock()
        self._time_per_render = 1/60.        
        self.control_panel = None

        # -- GLFW initialization --
        if not glfw.init():
            raise Exception("GLFW can't be initialized")
        glfw.window_hint(glfw.VISIBLE, not offscreen)
        
        self.window = glfw.create_window(win_width, win_height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window can't be created")
        
        self.makeCurrent()
        glfw.swap_interval(1)

        if not self.offscreen:
            # -- ImGui initialization --
            self.set_panel(control_panel)
            glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
            glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
            glfw.set_key_callback(self.window, self._key_callback)
            glfw.set_char_callback(self.window, self._char_callback)

        # -- Viewer state initialization --
        self.mjv_option = mujoco.MjvOption()
        self.mjv_camera = mujoco.MjvCamera()  # mjCAMERA_FREE
        self.mjv_scene = mujoco.MjvScene(self.mj_model, maxgeom=maxgeom)
        self.mjr_context = mujoco.MjrContext(self.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.mjr_cam_context = mujoco.MjrContext(self.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.mjv_cam_scene = mujoco.MjvScene(self.mj_model, maxgeom=10000)
        self.mjv_cam_option = mujoco.MjvOption()

        # Initialize managers
        self.figure_manager = FigureManager()
        self.image_manager = ImageManager()
        self.text_manager = TextManager()
        self.item_manager = VisualItemManager()

    def set_panel(self, panel: Blueprint):
        if panel is not None and not self.offscreen:
            self.control_panel = panel
            self.control_panel.set_window(self.window, attach_callbacks=False)

    def add_plot(self, position: DisplayPosition, x, y, line_idx=0, linergb=(0,0,1), linename='Line', **params):
        self.figure_manager.plot(position, x, y, line_idx, linergb, linename, **params)

    def add_image(self, position: DisplayPosition, image):
        self.image_manager.add_image(position, image)

    def add_text(self, position: DisplayPosition, text1='', text2='', clear_buffer=False):
        self.text_manager.add_text(position, text1, text2, clear_buffer)

    def add_item(self, *items):
        self.item_manager.add_item(*items)

    def _update(self):
        render_start = time.time()
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        
        with self.gui_lock:
            mujoco.mjv_updateScene(self.mj_model, self.mj_data, self.mjv_option, None, self.mjv_camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.mjv_scene)
            self.item_manager.add_to_scene(self.mjv_scene)
            mujoco.mjr_render(viewport, self.mjv_scene, self.mjr_context)

            self.text_manager.render(self.mjr_context, viewport)
            self.figure_manager.render(self.mjr_context, width, height)
            self.image_manager.render(self.mjr_context, width, height)
            
            if self.control_panel is not None:
                self.control_panel.render_frame()
            
            glfw.swap_buffers(self.window)

        glfw.poll_events()
        self._time_per_render = 0.9 * self._time_per_render + 0.1 * (time.time() - render_start)

    def render(self):
        if not self.is_alive:
            return
        
        with self:
            if glfw.window_should_close(self.window):
                self.close()
                return

            self._update()

    def close(self):
        self.is_alive = False
        if self.control_panel is not None:
            self.control_panel.shutdown()
        glfw.terminate()
        self.mjr_context.free()

    def set_viewer(
        self,
        azimuth       = None,
        distance      = None,
        elevation     = None,
        lookat        = None,
        transparent   = None,
        contactpoint  = None,
        contactwidth  = None,
        contactheight = None,
        contactrgba   = None,
        joint         = None,
        jointlength   = None,
        jointwidth    = None,
        jointrgba     = None,
        geomgroup_0   = None,
        geomgroup_1   = None,
        geomgroup_2   = None,
        geomgroup_3   = None,
        geomgroup_4   = None,
        geomgroup_5   = None,
        black_sky     = None,
        convex_hull   = None,
    ):
        """
        Set or update the viewer’s camera and visualization parameters.
        
        Parameters:
            azimuth (float): Camera azimuth angle.
            distance (float): Camera distance.
            elevation (float): Camera elevation angle.
            lookat (list or np.array): Camera look-at position.
            transparent (bool): Flag for making dynamic geometries transparent.
            contactpoint (bool): Flag for displaying contact points.
            contactwidth (float): Contact point width.
            contactheight (float): Contact point height.
            contactrgba (list): RGBA color for contact points.
            joint (bool): Flag for joint visualization.
            jointlength (float): Length of joint visuals.
            jointwidth (float): Width of joint visuals.
            jointrgba (list): RGBA color for joints.
            geomgroup_0 ~ geomgroup_5: Visibility flags for geometry groups.
            black_sky (bool): Flag for enabling/disabling skybox.
            convex_hull (bool): Flag for convex hull visualization.
        
        Returns:
            None
        """
        # Basic viewer setting (azimuth, distance, elevation, and lookat)
        if azimuth is not None: self.mjv_camera.azimuth = azimuth
        if distance is not None: self.mjv_camera.distance = distance
        if elevation is not None: self.mjv_camera.elevation = elevation
        if lookat is not None: self.mjv_camera.lookat = lookat
        # Make dynamic geoms more transparent
        if transparent is not None: self.mjv_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
        # Contact point
        if contactpoint is not None: self.mjv_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contactpoint
        if contactwidth is not None: self.mj_model.vis.scale.contactwidth = contactwidth
        if contactheight is not None: self.mj_model.vis.scale.contactheight = contactheight
        if contactrgba is not None: self.mj_model.vis.rgba.contactpoint = contactrgba
        # Joint
        if joint is not None: self.mjv_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = joint
        if jointlength is not None: self.mj_model.vis.scale.jointlength = jointlength
        if jointwidth is not None: self.mj_model.vis.scale.jointwidth = jointwidth
        if jointrgba is not None: self.mj_model.vis.rgba.joint = jointrgba
        # Geom group
        if geomgroup_0 is not None: self.mjv_option.geomgroup[0] = geomgroup_0
        if geomgroup_1 is not None: self.mjv_option.geomgroup[1] = geomgroup_1
        if geomgroup_2 is not None: self.mjv_option.geomgroup[2] = geomgroup_2
        if geomgroup_3 is not None: self.mjv_option.geomgroup[3] = geomgroup_3
        if geomgroup_4 is not None: self.mjv_option.geomgroup[4] = geomgroup_4
        if geomgroup_5 is not None: self.mjv_option.geomgroup[5] = geomgroup_5
        # Skybox
        if black_sky is not None: self.mjv_scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = not black_sky
        # Convex hull
        if convex_hull is not None: self.mjv_option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = convex_hull
    
    def render_offscreen(self, depth=False, size=None):
        with self:
            if size is None:
                size = glfw.get_framebuffer_size(self.window)

            viewport = mujoco.MjrRect(0, 0, *size)
            
            with self.gui_lock:
                mujoco.mjv_updateScene(self.mj_model, self.mj_data, self.mjv_option, None, self.mjv_camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.mjv_scene)
            mujoco.mjr_render(viewport, self.mjv_scene, self.mjr_context)

            if depth:
                rgb_img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                depth_img = np.zeros((size[1], size[0], 1), dtype=np.float32)
                mujoco.mjr_readPixels(rgb_img, depth_img, viewport, self.mjr_context)
                return (np.flipud(rgb_img), np.flipud(depth_img))
            else:
                img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                mujoco.mjr_readPixels(img, None, viewport, self.mjr_context)
                return np.flipud(img)

    def __enter__(self):
        self.makeCurrent()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"An exception occurred: {exc_type}: {exc_value}")
        self.doneCurrent()

    def makeCurrent(self):
        glfw.make_context_current(self.window)

    def doneCurrent(self):
        glfw.make_context_current(None)


class ArrayView:
    
    def __init__(self, array: np.ndarray, ids=None):
        self.array = array
        self.ids = ids
    
    def get(self):
        if self.ids is not None:
            return self.array[self.ids]
        return self.array

    def set(self, values):
        self.array[self.ids] = values
        return self

    def __repr__(self):
        return f"ArrayView({self.array[self.ids]})"



class MujocoEnv:
    """
    A modern, minimal MuJoCo environment wrapper.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = None,
        model_xml: Optional[str] = None,
        assets: Optional[Dict[str, bytes]] = None,
        timestep: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize MuJoCo environment.

        Args:
            model_path: Path to MJCF/URDF file
            model_xml: XML string content
            assets: Asset dictionary for XML loading
            timestep: Override simulation timestep
            logger: Custom logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        # Load model
        if model_path:
            self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        elif model_xml:
            self.mj_model = mujoco.MjModel.from_xml_string(model_xml, assets)
        else:
            raise ValueError("Must provide either model_path or model_xml")
        
        # Override timestep if specified
        if timestep is not None:
            self.mj_model.opt.timestep = timestep

        # Create data
        self.mj_data = mujoco.MjData(self.mj_model)
        self.init_qpos = self.mj_data.qpos.copy()
        self.mj_forward()
        
        # camera
        self.cams = []
        self.cam_option = mujoco.MjvOption()
        for cam_id in range(self.mj_model.ncam):
            cam = mujoco.MjvCamera()
            cam.fixedcamid = cam_id
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cams.append(cam)

        # viewer
        self.viewer = MujocoViewer(self.mj_model, self.mj_data, win_width=800, win_height=600, offscreen=False)
        self.logger.info(f"MuJoCo model loaded with {self.mj_model.nbody} bodies, {self.mj_model.njnt} joints, {self.mj_model.nu} actuators.")

        # state
        self.tick = 0
        self.dt = self.mj_model.opt.timestep
        self.state_stored = None    
        self._joint_to_actuator: dict = {}  # joint_id -> actuator_id
        for i in range(self.mj_model.nu):
            jnt_id = self.actuator_to_joint_id(i)
            self._joint_to_actuator[jnt_id] = i

    def set_panel(self, panel: Blueprint):
        self.viewer.set_panel(panel)
    
    # region : get/set state --------------------------------------------------
    def set_init_qpos(self, joint_ids, qpos, forward=False):
        """
        Set the initial joint positions for specified joints.
        """
        self.init_qpos[self.joint_qposadr(joint_ids)] = qpos
        if forward:
            self.reset()

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:] = self.init_qpos.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def get_state(self):
        """ 
        Retrieve the current simulation state including time, joint positions, velocities, and actuator states.
        
        Returns:
            dict: A dictionary containing state information.
        """
        state = {
            'tick':self.tick,
            'time':self.mj_data.time,
            'qpos':self.mj_data.qpos.copy(), # [self.model.nq]
            'qvel':self.mj_data.qvel.copy(), # [self.model.nv]
            'act':self.mj_data.act.copy(),
        }
        return state
    
    def store_state(self):
        """
        Store the current simulation state for later restoration.
        """
        state = self.get_state()
        self.state_stored = copy.deepcopy(state) # deep copy
    
    def restore_state(self):
        """
        Restore the simulation state from a previously stored state.
        """
        state = self.state_stored
        self.set_state(
            qpos = state['qpos'],
            qvel = state['qvel'],
            act  = state['act'],
            step = True
        )
        
    def set_state(
            self,
            tick = None,
            time = None,
            qpos = None,
            qvel = None,
            act  = None, # used for simulating tendons and muscles
            ctrl = None,
            step = False
        ):
        """
        Set the simulation state with provided values.
        
        Parameters:
            tick (int): Simulation tick.
            time (float): Simulation time.
            qpos (np.array): Joint positions.
            qvel (np.array): Joint velocities.
            act (np.array): Actuator states.
            ctrl (np.array): Control signals.
            step (bool): If True, perform a simulation step after setting the state.
        """
        if tick is not None: self.tick = tick
        if time is not None: self.mj_data.time = time
        if qpos is not None: self.mj_data.qpos = qpos.copy()
        if qvel is not None: self.mj_data.qvel = qvel.copy()
        if act is not None: self.mj_data.act = act.copy()
        if ctrl is not None: self.mj_data.ctrl = ctrl.copy()
        # Forward dynamics
        if step: 
            mujoco.mj_step(self.mj_model, self.mj_data)
    
    @contextlib.contextmanager
    def state_context(self, restore_state = True):
        """
        Context manager for saving and restoring simulation state.
        Automatically stores state on enter and restores on exit.
        """
        if restore_state:
            self.store_state()
        try:
            yield
        finally:
            if restore_state:
                self.restore_state()
    # endregion ---------------------------------------------------------------

    # region : get/set joint state --------------------------------------------
    def joint_names(self) -> List[str]:
        """
        Get a list of all joint names in the model.
        """
        return [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]

    def joint_qposadr(self, joint_names: Union[list, tuple, int, str]) -> Union[list[int], int]:
        """
        start addr in 'qpos' for joint's data  (qpos: nq x 1)
        """
        if isinstance(joint_names, (int, str)):
            return self.mj_model.joint(joint_names).qposadr[0]
        return [self.mj_model.joint(name).qposadr[0] for name in joint_names]

    def joint_dofadr(self, joint_names: Union[list, tuple, int, str]) -> Union[list[int], int]:
        """
        start addr in 'qvel' for joint's data  (qvel: nv x 1)
        """
        if isinstance(joint_names, (int, str)):
            return self.mj_model.joint(joint_names).dofadr[0]
        return [self.mj_model.joint(name).dofadr[0] for name in joint_names]

    def joint_id(self, joint_names: Union[list, tuple, int, str]) -> Union[list[int], int]:
        """
        joint id in 'mj_model.joint'  (njnt x 1)
        """
        if isinstance(joint_names, (int, str)):
            return self.mj_model.joint(joint_names).id
        return [self.mj_model.joint(name).id for name in joint_names]

    def joint_qpos(self, ids) -> ArrayView:
        """
        Get a view of the joint positions for specified joint IDs/Names. 
        Assuming the joints' qpos are all 1D

        Parameters:
        - ids : list | tuple | int | str, Joint IDs or Names

        Returns:
        - ArrayView, A view of the joint positions.
        """
        return ArrayView(self.mj_data.qpos, self.joint_qposadr(ids))
        
    def joint_qvel(self, ids) -> ArrayView:
        return ArrayView(self.mj_data.qvel, self.joint_dofadr(ids))
    
    def joint_qacc(self, ids) -> ArrayView:
        return ArrayView(self.mj_data.qacc, self.joint_dofadr(ids))
    
    def joint_range(self, ids):
        return ArrayView(self.mj_model.jnt_range, self.joint_id(ids))
    
    def set_qpos(self, joint_ids, qpos, forward=True):
        self.joint_qpos(joint_ids).set(qpos)
        if forward:
            self.mj_forward()
    
    # endregion----------------------------------------------------------------

    # region : get/set actuator state -----------------------------------------
    def actuator_names(self) -> List[str]:
        return [self.mj_model.actuator(i).name for i in range(self.mj_model.nu)]
    
    def actuator_id(self, names: Union[list, tuple, int, str]) -> Union[list[int], int]:
        if isinstance(names, (int, str)):
            return self.mj_model.actuator(names).id
        return [self.mj_model.actuator(name).id for name in names]
    
    def actuator_ctrl(self, ids) -> ArrayView:
        return ArrayView(self.mj_data.ctrl, self.actuator_id(ids))
    
    def actuator_to_joint_id(self, actuator_ids: Union[list, tuple, int, str]) -> Union[list[int], int]:
        if isinstance(actuator_ids, (int, str)):
            return self.mj_model.actuator(actuator_ids).trnid[0]
        return [self.mj_model.actuator(name).trnid[0] for name in actuator_ids]
    
    def joint_to_actuator_id(self, joint_ids: Union[list, tuple, int, str]) -> List[int]:
        joint_ids = self.joint_id(joint_ids)
        if isinstance(joint_ids, int):
            return [self._joint_to_actuator.get(joint_ids, -1)]
        return [self._joint_to_actuator.get(jnt_id, -1) for jnt_id in joint_ids]
    # endregion ---------------------------------------------------------------

    # region: forward, step, set - sim ----------------------------------------
    def mj_forward(self, qpos=None, joint_ids=None):
        if qpos is not None and joint_ids is not None:
            self.joint_qpos(joint_ids).set(qpos)
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def mj_step(self, nstep=1):
        mujoco.mj_step(self.mj_model, self.mj_data, nstep)
    
    def step_sim(
        self,
        ctrl: Optional[np.ndarray] = None,
        act_ids: Optional[Union[list, tuple]] = None,
        joint_ids: Optional[Union[list, tuple]] = None,
        nstep: int = 1,
    ):
        if act_ids is None:
            if joint_ids is not None:
                act_ids = self.joint_to_actuator_id(joint_ids)
            else:
                assert len(ctrl) == self.mj_model.nu, f"ctrl length {len(ctrl)} does not match actuator number {self.mj_model.nu}"
                act_ids = range(self.mj_model.nu)

        self.actuator_ctrl(act_ids).set(ctrl)

        if nstep>0:
            mujoco.mj_step(self.mj_model, self.mj_data, nstep)
            self.tick += nstep
    
    def loop_every(self, Hz=None, ntick=None):
        """
        Determine if the simulation loop should execute based on simulation tick or frequency.
        """
        if Hz is not None:
            return self.tick % int(1 / (Hz * self.dt)) == 0
        if ntick is not None:
            return self.tick % ntick == 0
        return False

    # endregion ---------------------------------------------------------------
    
    # region : get/set pose for body, geom, site, sensor, camera --------------
    def get_data_view(self, name, type: mujoco.mjtObj):
        if type == mujoco.mjtObj.mjOBJ_BODY:
            return self.mj_data.body(name)
        elif type == mujoco.mjtObj.mjOBJ_GEOM:
            return self.mj_data.geom(name)
        elif type == mujoco.mjtObj.mjOBJ_SITE:
            return self.mj_data.site(name)
        elif type == mujoco.mjtObj.mjOBJ_SENSOR:
            return self.mj_data.sensor(name)
        elif type == mujoco.mjtObj.mjOBJ_CAMERA:
            return self.mj_data.cam(name)
        elif type == mujoco.mjtObj.mjOBJ_ACTUATOR:
            return self.mj_data.actuator(name)
        elif type == mujoco.mjtObj.mjOBJ_JOINT:
            return self.mj_data.joint(name)
        else:
            raise ValueError("Unsupported type for get_elem_view")

    def get_model_view(self, name, type: mujoco.mjtObj):
        if type == mujoco.mjtObj.mjOBJ_BODY:
            return self.mj_model.body(name)
        elif type == mujoco.mjtObj.mjOBJ_GEOM:
            return self.mj_model.geom(name)
        elif type == mujoco.mjtObj.mjOBJ_SITE:
            return self.mj_model.site(name)
        elif type == mujoco.mjtObj.mjOBJ_SENSOR:
            return self.mj_model.sensor(name)
        elif type == mujoco.mjtObj.mjOBJ_CAMERA:
            return self.mj_model.cam(name)
        elif type == mujoco.mjtObj.mjOBJ_ACTUATOR:
            return self.mj_model.actuator(name)
        elif type == mujoco.mjtObj.mjOBJ_JOINT:
            return self.mj_model.joint(name)
        else:
            raise ValueError("Unsupported type for get_elem_view")
    
    def get_pose(self, name, type: mujoco.mjtObj):
        data_view = self.get_data_view(name, type)
        pose = Matrix4x4()
        pose.matrix33 = data_view.xmat.reshape(3,3)
        pose.xyz = data_view.xpos
        return pose
    
    def geom_pose(self, geom_name):
        return self.get_pose(geom_name, mujoco.mjtObj.mjOBJ_GEOM)
    
    def body_pose(self, body_name):
        return self.get_pose(body_name, mujoco.mjtObj.mjOBJ_BODY)
    
    def site_pose(self, site_name):
        return self.get_pose(site_name, mujoco.mjtObj.mjOBJ_SITE)
    
    def sensor_pose(self, sensor_name):
        return self.get_pose(sensor_name, mujoco.mjtObj.mjOBJ_SENSOR)
    
    def camera_pose(self, cam_name):
        return self.get_pose(cam_name, mujoco.mjtObj.mjOBJ_CAMERA)
    
    def set_pose(self, name, type: mujoco.mjtObj, pose: Matrix4x4, forward=True):
        """
        set pos, quat in mj_model, if forward=True, then call mj_forward() to update
        """
        model_view = self.get_model_view(name, type)
        model_view.pos = pose.xyz
        model_view.quat = pose.quat_data
        if forward:
            mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def camera_names(self) -> List[str]:
        return [self.mj_model.cam(i).name for i in range(self.mj_model.ncam)]
    
    def body_names(self) -> List[str]:
        return [self.mj_model.body(i).name for i in range(self.mj_model.nbody)]
    
    def geom_names(self) -> List[str]:
        return [self.mj_model.geom(i).name for i in range(self.mj_model.ngeom)]
    
    def site_names(self) -> List[str]:
        return [self.mj_model.site(i).name for i in range(self.mj_model.nsite)]
    
    def sensor_names(self) -> List[str]:
        return [self.mj_model.sensor(i).name for i in range(self.mj_model.nsensor)]

    def get_nearest_body(self, xyz, body_names=None):
        if body_names is None:
            body_names = self.body_names()
        
        nearest_name = ""
        min_dist = float('inf')
        
        for body_name in body_names:
            body_pos = self.get_data_view(body_name, mujoco.mjtObj.mjOBJ_BODY).xpos
            dist = np.linalg.norm(body_pos - xyz)
            if dist < min_dist:
                min_dist = dist
                nearest_name = body_name        

        return nearest_name

    # endregion ---------------------------------------------------------------
    
    # region : inverse kinematics ---------------------------------------------
    def _get_J(self, name, type: mujoco.mjtObj):
        n_dof = self.mj_model.nv
        J_p = np.zeros((3, n_dof))
        J_R = np.zeros((3, n_dof))
        if type == mujoco.mjtObj.mjOBJ_BODY:
            mujoco.mj_jacBody(self.mj_model, self.mj_data, J_p, J_R, self.mj_data.body(name).id)
        elif type == mujoco.mjtObj.mjOBJ_GEOM:
            mujoco.mj_jacGeom(self.mj_model, self.mj_data, J_p, J_R, self.mj_data.geom(name).id)
        elif type == mujoco.mjtObj.mjOBJ_SITE:
            mujoco.mj_jacSite(self.mj_model, self.mj_data, J_p, J_R, self.mj_data.site(name).id)
        J_full = np.vstack([J_p, J_R])
        return J_p, J_R, J_full

    def get_J_site(self, site_name):
        """
        Compute the Jacobian matrices (position and rotation) for the specified site.

        Parameters:
            site_name (str): Name of the site.

        Returns:
            tuple: (J_p, J_R, J_full) where
                J_p: 3 x n_dof, translational Jacobian
                J_R: 3 x n_dof, rotational Jacobian
                J_full: 6 x n_dof, stacked Jacobian
        """
        return self._get_J(site_name, mujoco.mjtObj.mjOBJ_SITE)

    def get_J_body(self, body_name):
        """
        Compute the Jacobian matrices (position and rotation) for the specified body.
        """
        return self._get_J(body_name, mujoco.mjtObj.mjOBJ_BODY)

    def get_J_geom(self, geom_name):
        """
        Compute the Jacobian matrices for the specified geometry.
        """
        return self._get_J(geom_name, mujoco.mjtObj.mjOBJ_GEOM)

    def damped_ls(self, J, err, eps=1e-2, stepsize=1.0, th=5*np.pi/180.0):
        """
        Solve the inverse kinematics using the damped least squares method.
        
        Parameters:
            J (np.array): Jacobian matrix.
            err (np.array): Error vector.
            eps (float): Damping factor.
            stepsize (float): Step size multiplier.
            th (float): Threshold for scaling the result.
        
        Returns:
            np.array: The computed joint increments (dq).
        """
        dq = stepsize * np.linalg.solve(a=(J.T@J) + eps * np.eye(J.shape[1]), b=J.T@err)

        # trim_scale
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > th:
            dq = dq * th / dq_abs_max

        return dq
    
    def ik_solve(
        self,
        site_name: str,
        joint_names: List[str],
        pose_target: Matrix4x4,
        qpos_init: np.array = None,
        atol: float = 1e-2,
        ik_th: float = np.radians(1.0),
        max_iter: int = 100,
        restore_state: bool = True,
        verbose: bool = False
    ):
        joint_dofadrs = self.joint_dofadr(joint_names)
        qpos_range = self.joint_range(joint_names).get().reshape(-1,2)
        qpos_min, qpos_max = qpos_range[:,0], qpos_range[:,1]
        qpos_view = self.joint_qpos(joint_names)
        
        if qpos_init is None:
            qpos_tmp = qpos_view.get()
        else:
            qpos_tmp = np.clip(qpos_init, qpos_min, qpos_max)
            
        with self.state_context(restore_state):
            for iter_cnt in range(max_iter):
                # break condition
                qpos_view.set(qpos_tmp)
                self.mj_forward()
                    
                pose_curr = self.site_pose(site_name)
                pose_err = pose_target.diff(pose_curr)
                pose_err_norm = np.linalg.norm(pose_err)
                if pose_err_norm < atol:
                    break
                if verbose:
                    self.logger.debug(f"[{iter_cnt}/{max_iter}] ik_err: {pose_err_norm:.5f}")

                J = self.get_J_site(site_name=site_name)[2]  # 6 x ndof
                J = J[:, joint_dofadrs]  # 6 x len(joint_names)
                dq = self.damped_ls(J, pose_err, th=ik_th)
                qpos_tmp = np.clip(qpos_tmp + dq, qpos_min, qpos_max)

        return qpos_tmp, pose_err_norm, (iter_cnt < max_iter)
    # endregion ---------------------------------------------------------------

    # region: visualization ---------------------------------------------------
    def is_alive(self):
        return self.viewer.is_alive

    def set_camera(self,  azimuth=None, elevation=None, distance=None, lookat=None):
        self.viewer.set_viewer(azimuth=azimuth, distance=distance, elevation=elevation, lookat=lookat)

    def render(self):
        self.viewer.render()
    
    def plot_pose(self, pose, length=0.15, width=0.005):
        """
        Visualize a 3D pose in the MuJoCo viewer using lines.
        
        Parameters:
            pose (Matrix4x4): The pose to visualize.
        """
        self.viewer.add_item(
            AxisItem((length, length, length), width).apply_transform(pose).draw_once()
        )
    
    def get_cam_rgbd(self, cam_id, width, height, render_item: bool = False):
        """
            Get Depth of fixed cam
        """
        with self.viewer:
            cam_id  = self.mj_model.cam(cam_id).id
            viewport = mujoco.MjrRect(0,0, int(width), int(height))
            
            # Update
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.viewer.mjr_cam_context)
            mujoco.mjv_updateScene(
                self.mj_model, self.mj_data, self.viewer.mjv_cam_option, None,
                self.cams[cam_id], mujoco.mjtCatBit.mjCAT_ALL, self.viewer.mjv_cam_scene
            )
            if render_item:
                self.viewer.item_manager.add_to_scene(self.viewer.mjv_cam_scene, enable_draw_once=False)
            mujoco.mjr_render(viewport, self.viewer.mjv_cam_scene, self.viewer.mjr_cam_context)

            # Grab RGBD
            rgb = np.zeros((viewport.height, viewport.width, 3), dtype=np.uint8)
            depth_raw = np.zeros((viewport.height, viewport.width), dtype=np.float32)
            mujoco.mjr_readPixels(rgb, depth_raw, viewport, self.viewer.mjr_cam_context)

            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.viewer.mjr_context)
            rgb, depth_raw = np.flipud(rgb), np.flipud(depth_raw)

            extent  = self.mj_model.stat.extent
            znear   = self.mj_model.vis.map.znear * extent
            zfar    = self.mj_model.vis.map.zfar * extent

            depth_m = znear * (1.0 - depth_raw) + zfar * depth_raw
            return rgb, depth_m

    def plot_camera(self, cam_id, length=0.1, width=0.02, aspect=4/3):
        cam = self.mj_model.cam(cam_id)
        pose = self.camera_pose(cam_id)
        
        axis_item = AxisItem((length, length, length)).apply_transform(pose).draw_once()
        
        h = cam.fovy[0]
        w = h * aspect
        near, far = -0.0, -0.1
        near_corners = np.array([
            [-w/2, -h/2, near],
            [ w/2, -h/2, near],
            [ w/2,  h/2, near],
            [-w/2,  h/2, near],
        ])
        far_corners = np.array([
            [-w/2, -h/2, far], 
            [ w/2, -h/2, far], 
            [ w/2,  h/2, far], 
            [-w/2,  h/2, far], 
        ])
        pts = np.vstack([near_corners, far_corners]) # shape: (8,3)
        
        line_item = LineItem(pts[[0,1,2,3,0,4,5,6,7,4,5,1,2,6,7,3]], width=width, rgba=(0,1,0,1), parent=axis_item)
        self.viewer.add_item(axis_item)
        
    # endregion ---------------------------------------------------------------
