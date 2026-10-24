import time
import glfw
import mujoco
import numpy as np
import cv2

from xensemujoco import PROJ_DIR
from xensemujoco.mujoco_env.base_env import MujocoEnv
from xensemujoco.mujoco_env.visual_item import *
from xensemujoco.mujoco_env.imgui_blueprint import *
from xensim.core import FemSensor
from xensim import PROJ_DIR as XENSIM_PROJ_DIR


class CustomEnv(MujocoEnv):
    ur5_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]   
    ctrl_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
        "finger_pos"
    ]
    ee_site_name = "ee"
    
    def __init__(self, xml_path):
        super().__init__(xml_path)
        self.viewer.set_viewer(geomgroup_5=True)
        self.set_camera(azimuth=-100, elevation=-40, distance=2.7)
        self._ctrl = np.zeros(len(self.ctrl_names))
        self.set_init_qpos(self.ur5_joint_names, [-1.8581, -0.9167,  1.5921,  0.8550,  1.5413, -3.4274], forward=True)
        self.reset()
        self.init_teleop()

    def reset(self):
        super().reset()
        self.finger_pos = -0.02
        self.target_pose = self.get_ee_pose()
        self.step()  # udpate _ctrl
        
    def step(self):
        qpos, err, ret = self.ik_solve(self.ee_site_name, self.ur5_joint_names, self.target_pose, atol=0.001)
        self._ctrl = np.concatenate([qpos, [self.finger_pos]])
    
    def step_sim(self, nstep=1):
        super().step_sim(self._ctrl, act_ids=self.ctrl_names, nstep=nstep)
    
    def get_ee_pose(self):
        return self.site_pose(self.ee_site_name)
    
    def init_teleop(self):
        self.viewer.add_key_callback(glfw.KEY_A, lambda: self.target_pose.translate( 0.01, 0, 0, False))
        self.viewer.add_key_callback(glfw.KEY_D, lambda: self.target_pose.translate(-0.01, 0, 0, False))
        self.viewer.add_key_callback(glfw.KEY_W, lambda: self.target_pose.translate(0, -0.01, 0, False))
        self.viewer.add_key_callback(glfw.KEY_S, lambda: self.target_pose.translate(0,  0.01, 0, False))
        self.viewer.add_key_callback(glfw.KEY_Q, lambda: self.target_pose.translate(0, 0,  0.01, False))
        self.viewer.add_key_callback(glfw.KEY_E, lambda: self.target_pose.translate(0, 0, -0.01, False))
        self.viewer.add_key_callback(glfw.KEY_R, lambda: self.target_pose.rotate( 2, 1, 0, 0))
        self.viewer.add_key_callback(glfw.KEY_F, lambda: self.target_pose.rotate(-2, 1, 0, 0))
        self.viewer.add_key_callback(glfw.KEY_T, lambda: self.target_pose.rotate( 2, 0, 1, 0))
        self.viewer.add_key_callback(glfw.KEY_G, lambda: self.target_pose.rotate(-2, 0, 1, 0))
        self.viewer.add_key_callback(glfw.KEY_Y, lambda: self.target_pose.rotate( 2, 0, 0, 1))
        self.viewer.add_key_callback(glfw.KEY_H, lambda: self.target_pose.rotate(-2, 0, 0, 1))
        self.viewer.add_key_callback(glfw.KEY_SPACE, lambda: setattr(self, 'finger_pos', 0.0 if self.finger_pos>0.02 else 0.04))


env = CustomEnv(PROJ_DIR / "assets/universal_robots_ur5e/scene.xml")

sensor = FemSensor(
    calibrate_file=XENSIM_PROJ_DIR/"assets/fem/g1-ws_table.npz",
    fem_file=XENSIM_PROJ_DIR/"assets/fem/g1-ws.npz",
    depth_size=(100, 175),
    visible=True,
    title="FemSensor Simulation"
)


# 定义左侧控制面板
panel_blueprint = Blueprint(
    Panel(name="Control", width=200, dock_to="left", no_collapse=True, closable=False, no_title_bar=False, resizable=True).add(
        Vertical().add(
            CollapsingHeader(name="Visualization Options").add(
                Checkbox("show marker", True, callback=lambda v: sensor.set_sim(show_marker=v)),
                Checkbox("show contact", callback=lambda v: sensor.set_sim(show_contact=v)),
                Checkbox("show force", callback=lambda v: sensor.set_sim(show_force=v)),
                Checkbox("bg color", True, callback=lambda v: sensor.set_sim(bg_color=v)),
                SliderFloat("smooth level", 0, 10, default_value=5, callback=lambda v: sensor.set_sim(smooth_norm=int(v))),
                Button("align view", callback=sensor.align_view),
                Button("reset", callback=env.reset),
            ),
          
            Spacing(), Spacing(),
            Text("Depth"),
            ImageWidget("depth_image", width=200, height=350),
        )
    ),
    Panel(name="Help", width=200, closable=True).add(
        Vertical(Text("WASD/QE: move the end-effector. \nSpace: open/close gripper."))
    )
)
env.set_panel(panel_blueprint)

# 添加轨迹点
traj_item = MultiGeomItem(GeomItem.create_sphere(radius=0.005, rgba=(1,0,0,1)), maxlen=50)
env.viewer.add_item(traj_item)

# 主循环
while env.is_alive():
    env.step_sim()
    
    # 渲染和传感器更新
    if env.loop_every(Hz=30):
        env.step()
        
        # 获取深度并可视化
        rgb, depth = env.get_cam_rgbd(1, 200, 350)
        depth = depth - 0.01  # unit m, 因相机设置在距离表面 1cm 处, 需要减去 0.01m, 《0 为侵入
        
        depth_vis = (-np.minimum(depth, 0) * 125_000).astype(np.uint8)
        depth_vis = cv2.applyColorMap(255-depth_vis, cv2.COLORMAP_JET)
        panel_blueprint.get_widget("depth_image").set_image(depth_vis)

        # step sensor
        obj_pose = env.body_pose("block")
        sensor_pose = env.camera_pose("left_sensor")
        sensor.step(obj_pose, sensor_pose, depth*1000, nstep=2)
        sensor.update()
        
        # 仿真环境渲染        
        env.render()

    # 显示末端轨迹
    if env.loop_every(Hz=10):
        traj_item.add_data(env.get_ee_pose())