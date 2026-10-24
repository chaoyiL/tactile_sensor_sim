"""Python 3.9 worker: MuJoCo depth images followed by Xensim tactile RGB.

This process deliberately imports no project modules. JSON lines are the control
channel; NumPy archives carry arrays without pickle across the ABI boundary.
"""
from __future__ import print_function

import hashlib
import json
import sys
import traceback
from pathlib import Path


PREFIX = 'XENSE_JSON:'
PROTOCOL_VERSION = 1


def _reply(value):
    sys.stdout.write(PREFIX + json.dumps(value, allow_nan=False) + '\n')
    sys.stdout.flush()


class Runtime:
    def __init__(self, config):
        import mujoco
        import numpy as np
        from xensim.core import SimpleSensor

        self.mujoco = mujoco
        self.np = np
        self.config = config
        if config['mujoco']['sensor_mode'] != 'simple':
            raise ValueError('Only the validated marker-free SimpleSensor mode is supported')
        sensor_config = config['mujoco']
        width_mm, height_mm = map(float, config['surface']['size_mm'])
        depth_width, depth_height = map(int, sensor_config['depth_size_wh'])
        camera_gap_mm = float(sensor_config['camera_gap_mm'])
        radius_mm = float(config['indentation']['radius_mm'])
        self.camera_gap_m = camera_gap_mm / 1000.0
        self.radius_mm = radius_mm
        self.depth_width, self.depth_height = depth_width, depth_height
        if sensor_config['calibrate_file'] == 'bundled:g1-ws_table':
            from xensim import PROJ_DIR
            table_path = Path(PROJ_DIR) / 'assets' / 'fem' / 'g1-ws_table.npz'
        else:
            table_path = Path(sensor_config['calibrate_file'])
        if not table_path.is_file():
            raise FileNotFoundError('Xensim calibration table: {}'.format(table_path))
        self.asset_sha256 = hashlib.sha256(table_path.read_bytes()).hexdigest()
        self.sensor = SimpleSensor(
            str(table_path),
            mesh_size=tuple(sensor_config['mesh_size_wh']),
            render_size=tuple(sensor_config['render_size_wh']),
            gel_width_mm=width_mm,
            gel_height_mm=height_mm,
            visible=False,
        )

        # Orthographic fovy has length units in MuJoCo 3.3.x. The camera
        # orientation makes image rows increase with sensor Y.
        xml = ('<mujoco model="xense_sphere_scan">'
               '<option gravity="0 0 0"/>'
               '<visual><global offwidth="{dw}" offheight="{dh}"/>'
               '<quality numslices="256" numstacks="128"/></visual>'
               '<worldbody>'
               '<geom name="reference_plane" type="plane" pos="0 0 0" '
               'size="{pw:.12g} {ph:.12g} 0.001" contype="0" conaffinity="0"/>'
               '<body name="calibration_sphere" mocap="true" pos="0 0 {initial:.12g}">'
               '<geom type="sphere" size="{radius:.12g}" contype="0" conaffinity="0"/>'
               '</body>'
               '<camera name="tactile_depth" pos="0 0 -{gap:.12g}" '
               'xyaxes="1 0 0 0 -1 0" orthographic="true" fovy="{fovy:.12g}"/>'
               '</worldbody></mujoco>').format(
                   dw=depth_width, dh=depth_height,
                   pw=width_mm / 2000.0, ph=height_mm / 2000.0,
                   initial=(radius_mm + 100.0) / 1000.0,
                   radius=radius_mm / 1000.0, gap=self.camera_gap_m,
                   fovy=height_mm / 1000.0)
        self.scene_xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.gl = mujoco.GLContext(depth_width, depth_height)
        self.gl.make_current()
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.context.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        self.scene = mujoco.MjvScene(self.model, maxgeom=100)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera.fixedcamid = self.model.camera('tactile_depth').id
        self.option = mujoco.MjvOption()
        self.viewport = mujoco.MjrRect(0, 0, depth_width, depth_height)
        self.near_m = float(self.model.vis.map.znear * self.model.stat.extent)
        self.far_m = float(self.model.vis.map.zfar * self.model.stat.extent)
        if not 0 < self.near_m < self.camera_gap_m < self.far_m:
            raise ValueError('MuJoCo near/far planes do not include the gel surface')
        self.versions = {
            'mujoco': mujoco.__version__,
            'numpy': np.__version__,
            'python': sys.version.split()[0],
        }
        try:
            from importlib import metadata
            self.versions['xensim'] = metadata.version('xensim')
            self.versions['xensesdk'] = metadata.version('xensesdk')
        except (ImportError, metadata.PackageNotFoundError):
            from importlib_metadata import version
            self.versions['xensim'] = version('xensim')
            self.versions['xensesdk'] = version('xensesdk')

    def _depth_image(self, center_xy_mm, depth_mm):
        mj = self.mujoco
        np = self.np
        center = np.asarray(center_xy_mm, dtype=np.float64)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError('center_xy_mm must contain two finite coordinates')
        if not np.isfinite(depth_mm) or not 0 <= depth_mm < self.radius_mm:
            raise ValueError('Require 0 <= depth_mm < sphere radius')
        center_z = ((self.radius_mm - depth_mm) if depth_mm > 0
                    else self.radius_mm + 100.0)
        self.data.mocap_pos[0] = (center[0] / 1000.0, center[1] / 1000.0,
                                  center_z / 1000.0)
        mj.mj_forward(self.model, self.data)
        self.gl.make_current()
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        mj.mjv_updateScene(self.model, self.data, self.option, None,
                           self.camera, mj.mjtCatBit.mjCAT_ALL, self.scene)
        mj.mjr_render(self.viewport, self.scene, self.context)
        raw = np.empty((self.depth_height, self.depth_width), np.float32)
        mj.mjr_readPixels(None, raw, self.viewport, self.context)
        raw = np.flipud(raw)
        # MuJoCo 3.3.x uses reversed Z. Renderer.enable_depth_rendering()
        # applies a perspective inverse even to orthographic cameras; read the
        # public depth buffer and use the orthographic inverse instead.
        depth_m = self.near_m + (self.far_m - self.near_m) * (1.0 - raw)
        if not np.isfinite(depth_m).all():
            raise RuntimeError('MuJoCo produced non-finite depth')
        return depth_m

    def render(self, center_xy_mm, depth_mm, output_path):
        np = self.np
        depth_m = self._depth_image(center_xy_mm, float(depth_mm))
        depth_input_mm = np.minimum((depth_m - self.camera_gap_m) * 1000.0, 0.0)
        self.sensor.step(depth_input_mm.astype(np.float32),
                         smooth_norm=int(self.config['mujoco']['smooth_norm']),
                         smooth_mesh=int(self.config['mujoco']['smooth_mesh']))
        image = np.asarray(self.sensor.get_image()).copy()
        expected = (self.config['mujoco']['render_size_wh'][1],
                    self.config['mujoco']['render_size_wh'][0], 3)
        if image.dtype != np.uint8 or image.shape != expected:
            raise RuntimeError('Xensim RGB has unexpected dtype or shape: {} {}'.format(
                image.dtype, image.shape))
        path = Path(output_path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, rgb8=image,
                                depth_input_mm=depth_input_mm.astype(np.float32),
                                depth_m=depth_m.astype(np.float32))
        temporary.replace(path)
        return {'path': str(path), 'depth_min_mm': float(depth_input_mm.min()),
                'depth_max_mm': float(depth_input_mm.max())}

    def close(self):
        if hasattr(self, 'gl'):
            self.gl.free()


def main():
    runtime = None
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            if request.get('protocol_version') != PROTOCOL_VERSION:
                raise ValueError('Unsupported worker protocol version')
            operation = request['operation']
            if operation == 'open':
                if runtime is not None:
                    raise RuntimeError('Worker is already open')
                runtime = Runtime(request['config'])
                payload = {'versions': runtime.versions,
                           'calibration_table_sha256': runtime.asset_sha256,
                           'near_m': runtime.near_m, 'far_m': runtime.far_m,
                           'camera_gap_m': runtime.camera_gap_m,
                           'scene_xml': runtime.scene_xml}
            elif operation == 'render':
                if runtime is None:
                    raise RuntimeError('Worker is not open')
                payload = runtime.render(request['center_xy_mm'],
                                         request['depth_mm'], request['output_path'])
            elif operation == 'close':
                if runtime is not None:
                    runtime.close()
                    runtime = None
                payload = {'closed': True}
            else:
                raise ValueError('Unknown operation: {}'.format(operation))
            _reply({'id': request['id'], 'ok': True, 'payload': payload})
            if operation == 'close':
                break
        except Exception as exc:
            _reply({'id': request.get('id') if isinstance(request, dict) else None,
                    'ok': False, 'error': '{}: {}'.format(type(exc).__name__, exc),
                    'traceback': traceback.format_exc()})


if __name__ == '__main__':
    main()
