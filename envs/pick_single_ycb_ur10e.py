from pathlib import Path
import importlib
import numpy as np
import copy
import sapien
import torch

import envs.agents
from mani_skill.envs.tasks.tabletop.pick_single_ycb import PickSingleYCBEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.sapien_env import BaseEnv
from transforms3d.euler import euler2quat

from utils.const import ASSET_PATH

SODA_CAN_PATH = ASSET_PATH / "soda_can.glb"
TABLE_SCENE_BUILDER_MODULE = importlib.import_module(
    "mani_skill.utils.scene_builder.table.scene_builder"
)


class LargeXYTableSceneBuilder(TableSceneBuilder):      # enlarge the table
    def build(self):
        builder = self.scene.create_actor_builder()
        model_dir = Path(TABLE_SCENE_BUILDER_MODULE.__file__).parent / "assets"
        table_model_file = str(model_dir / "table.glb")
        scale = 1.75
        xy_scale = 2.0
        table_height = 0.9196429

        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, table_height / 2]),
            half_size=(xy_scale * 2.418 / 2, xy_scale * 1.209 / 2, table_height / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file,
            scale=[scale * xy_scale, scale * xy_scale, scale],
            pose=table_pose,
        )
        builder.initial_pose = sapien.Pose(
            p=[-0.12, 0, -table_height], q=euler2quat(0, 0, np.pi / 2)
        )
        self.table = builder.build_kinematic(name="table-workspace")
        self.table_length = 2 * 1.2090764
        self.table_width = 2 * 2.4178784
        self.table_height = table_height

        floor_width = 500 if self.scene.parallel_in_single_scene else 100
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        self.scene_objects: list[sapien.Entity] = [self.table, self.ground]


@register_env("PickSingleYCBUR10e-v1", max_episode_steps=300)
class PickSingleYCBUR10eEnv(PickSingleYCBEnv):
    SUPPORTED_ROBOTS = ["ur10e_robotiq_2f85"]
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_INTRINSIC = np.array([
        [461.49, 0.0, 322.39],
        [0.0, 461.65, 241.174],
        [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    def __init__(
        self,
        *args,
        robot_uids="ur10e_robotiq_2f85",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )

    @property
    def _default_sensor_configs(self):
        pose = self._default_camera_pose()
        return [
            CameraConfig(
                uid="base_camera",
                pose=pose,
                width=self.CAMERA_WIDTH,
                height=self.CAMERA_HEIGHT,
                intrinsic=self.CAMERA_INTRINSIC,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = self._default_camera_pose()
        return CameraConfig(
            uid="render_camera",
            pose=pose,
            width=self.CAMERA_WIDTH,
            height=self.CAMERA_HEIGHT,
            intrinsic=self.CAMERA_INTRINSIC,
            near=0.01,
            far=100,
        )

    def _default_camera_pose(self):
        # Camera extrinsic: world-frame camera pose.
        return sapien_utils.look_at(
            eye=[0.35, 0.45, 0.4],
            target=[-0.05, 0.45, 0.1],
        )

    def _load_scene(self, options: dict):
        self.table_scene = LargeXYTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        builder = self.scene.create_actor_builder()
        can_pose = sapien.Pose(q=euler2quat(0, 0, 0))
        builder.add_convex_collision_from_file(
            filename=str(SODA_CAN_PATH),
            density=1000,
            pose=can_pose,
        )
        builder.add_visual_from_file(
            filename=str(SODA_CAN_PATH),
            pose=can_pose,
        )
        builder.initial_pose = sapien.Pose()
        self.obj = builder.build(name="soda_can")

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _after_reconfigure(self, options: dict):
        collision_mesh = self.obj.get_first_collision_mesh()
        object_z = -collision_mesh.bounding_box.bounds[0, 2]
        self.object_zs = common.to_tensor(
            [object_z] * self.num_envs, device=self.device
        )

    def _load_agent(self, options: dict):
        BaseEnv._load_agent(        # set robot base pose
            self,
            options,
            sapien.Pose(p=[-0.615, 0, 0], q=euler2quat(-90, 0, 0, "szyx")),
        )

    def _set_base_camera_look_at_obj(self, obj_pos: torch.Tensor):
        obj_pos = common.to_tensor(obj_pos, device=self.device)
        if obj_pos.ndim == 1:
            obj_pos = obj_pos[None, :]

        target = obj_pos.detach().clone()
        target[:, 2] += 0.2
        eye = target.detach().clone()
        eye[:, 0] = 1.
        # print("target:", target)
        # print("eye:", eye)
        pose = sapien_utils.look_at(eye=eye, target=obj_pos, device=self.device)

        sensor = self._sensors["base_camera"]
        for i, render_camera in enumerate(sensor.camera._render_cameras):
            sapien_pose = pose[i].sp
            render_camera.set_local_pose(sapien_pose)
        sensor.camera._cached_model_matrix = None
        sensor.camera._cached_extrinsic_matrix = None
        sensor.camera._cached_local_pose = None
        sensor.config.pose = pose

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.1 + torch.tensor([-0.1, 0.4])
            xyz[:, 2] = self.object_zs[env_idx]
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            self._set_base_camera_look_at_obj(xyz)

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = copy.deepcopy(xyz[:, :2])
            goal_xyz[:, 2] = torch.rand((b)) * 0.05 + xyz[:, 2] + 0.12
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            robot_root_pose = sapien.Pose(
                [-0.615, 0, 0], euler2quat(-1.5708, 0, 0, "szyx")
            )
            self.agent.robot.set_root_pose(robot_root_pose)

            qpos = torch.zeros((b, self.agent.robot.max_dof), device=self.device)
            ik_seed_arm_qpos = common.to_tensor(
                [-1.5708, -1.5708, -1.5708, -1.5708, 1.5708, 1.5708],
                device=self.device,
            )
            qpos[:, : len(self.agent.arm_joint_names)] = ik_seed_arm_qpos

            # Define the desired initial TCP pose in the world frame here.
            tcp_pose_world = Pose.create_from_pq(
                p=torch.tensor([[-0.1, 0.4, 0.3]], device=self.device).repeat(
                    b, 1
                ) + torch.rand((b, 3), device=self.device) * 0.05,
                q=torch.from_numpy(euler2quat(3.1416, 0, 0, "sxyz")).repeat(b, 1),
            )
            tcp_pose_base = self.agent.robot.root.pose.inv() * tcp_pose_world

            original_control_mode = self.agent.control_mode
            if original_control_mode != "pd_ee_delta_pose":
                self.agent.set_control_mode("pd_ee_delta_pose")
            try:
                arm_controller = self.agent.controller.controllers["arm"]
                arm_qpos = arm_controller.kinematics.compute_ik(
                    pose=tcp_pose_base,
                    q0=qpos,
                    solver_config=arm_controller.config.delta_solver_config,
                )
            finally:
                if original_control_mode != self.agent.control_mode:
                    self.agent.set_control_mode(original_control_mode)
            if arm_qpos is None:
                raise RuntimeError("Failed to compute IK for initial TCP pose")

            qpos[:, : len(self.agent.arm_joint_names)] = arm_qpos

            gripper_qpos = np.zeros(6)
            qpos[:, len(self.agent.arm_joint_names) :] = common.to_tensor(
                gripper_qpos, device=self.device
            )
            self.agent.reset(qpos)

    def evaluate(self):
        obj_to_goal_pos = self.goal_site.pose.p - self.obj.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        is_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static(0.2)
        return dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasping=is_grasped,
            success=torch.logical_and(is_obj_placed, is_robot_static),
        )

    def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel(), axis=1)
        )
        reward += static_reward * info["is_obj_placed"]
        reward += info["is_grasped"]
        reward[info["success"]] = 4
        return reward

    def compute_normalized_dense_reward(self, obs, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4
