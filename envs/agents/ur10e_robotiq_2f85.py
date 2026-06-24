from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDEEPosControllerConfig,
    PDEEPoseControllerConfig,
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
    PassiveControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


ASSET_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets/robots/ur10e_robotiq_2f85/ur10e_robotiq_2f85.xml"
)
URDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets/robots/ur10e_robotiq_2f85/ur10e_robotiq_2f85.urdf"
)


@register_agent(asset_download_ids=["ur10e", "robotiq_2f"])
class UR10eRobotiq2F85(BaseAgent):
    uid = "ur10e_robotiq_2f85"
    # mjcf_path = str(ASSET_PATH)
    urdf_path = str(URDF_PATH)
    disable_self_collisions = True

    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    active_gripper_joint_names = [
        "left_outer_knuckle_joint",
        "right_outer_knuckle_joint",
    ]
    passive_gripper_joint_names = [
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "left_inner_finger_joint",
        "right_inner_finger_joint",
    ]
    ee_link_name = "robotiq_arg2f_base_link"
    tcp_link_name = "tcp_link"

    keyframes = dict(       # define initial pose
        rest=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0]),
            qpos=np.array(  # qpos sort follows SAPIEN loading (may change the order of joints in the MJCF file)
                [
                    -1.5708,
                    -1.5708,
                    1.5708,
                    -1.5708,
                    -1.5708,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
    )

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=1000,
            damping=100,
            force_limit=330,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,     # define the range of joint delta position
            upper=0.1,
            stiffness=1e4,
            damping=1e3,
            force_limit=330,
            normalize_action=True,
            use_delta=True,
        )
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=1e4,
            damping=1e3,
            force_limit=330,
            ee_link=self.tcp_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(    # Rx @ Ry @ Rz
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=1e4,
            damping=1e3,
            force_limit=330,
            ee_link=self.tcp_link_name,
            urdf_path=self.urdf_path,
            frame="root_translation:body_aligned_body_rotation"     # root trans, body rot
        )

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.active_gripper_joint_names,        # define the gripper action joint
            lower=0.0,
            upper=0.81,
            stiffness=1e5,
            damping=1e3,
            force_limit=100,
            friction=0.05,
            normalize_action=True,
            mimic={
                "left_outer_knuckle_joint": {
                    "joint": "right_outer_knuckle_joint",
                    "multiplier": 1.0,
                    "offset": 0.0,
                }
            },
        )
        gripper_pd_joint_delta_pos = deepcopy(gripper_pd_joint_pos)
        gripper_pd_joint_delta_pos.lower = -0.15        # define the range of gripper delta position
        gripper_pd_joint_delta_pos.upper = 0.15
        gripper_pd_joint_delta_pos.normalize_action = True
        gripper_pd_joint_delta_pos.use_delta = True

        passive_gripper_joints = PassiveControllerConfig(
            joint_names=self.passive_gripper_joint_names,
            damping=0,
            friction=0,
        )

        return deepcopy_dict(
            dict(
                pd_joint_delta_pos=dict(
                    arm=arm_pd_joint_delta_pos,
                    gripper_active=gripper_pd_joint_delta_pos,
                    gripper_passive=passive_gripper_joints,
                ),
                pd_joint_pos=dict(
                    arm=arm_pd_joint_pos,
                    gripper_active=gripper_pd_joint_pos,
                    gripper_passive=passive_gripper_joints,
                ),
                pd_ee_delta_pos=dict(
                    arm=arm_pd_ee_delta_pos,
                    gripper_active=gripper_pd_joint_pos,
                    gripper_passive=passive_gripper_joints,
                ),
                pd_ee_delta_pose=dict(
                    arm=arm_pd_ee_delta_pose,
                    gripper_active=gripper_pd_joint_pos,
                    gripper_passive=passive_gripper_joints,
                ),
            )
        )

    def _after_loading_articulation(self):
        # Add mimic drive (motion constraints) for the gripper
        right_inner_finger = self.robot.active_joints_map["right_inner_finger_joint"]
        right_inner_knuckle = self.robot.active_joints_map["right_inner_knuckle_joint"]
        right_pad = right_inner_finger.get_child_link()
        right_lif = right_inner_knuckle.get_child_link()

        p_f_right = [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02]
        p_p_right = [1.3578170e-09, -1.7901104e-02, 6.5159947e-03]
        right_drive = self.scene.create_drive(
            right_lif, sapien.Pose(p_f_right), right_pad, sapien.Pose(p_p_right)
        )
        right_drive.set_limit_x(0, 0)
        right_drive.set_limit_y(0, 0)
        right_drive.set_limit_z(0, 0)

        left_inner_finger = self.robot.active_joints_map["left_inner_finger_joint"]
        left_inner_knuckle = self.robot.active_joints_map["left_inner_knuckle_joint"]
        left_pad = left_inner_finger.get_child_link()
        left_lif = left_inner_knuckle.get_child_link()

        p_f_left = [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02]
        p_p_left = [-1.4041154e-08, -1.7901093e-02, 6.5159872e-03]
        left_drive = self.scene.create_drive(
            left_lif, sapien.Pose(p_f_left), left_pad, sapien.Pose(p_p_left)
        )
        left_drive.set_limit_x(0, 0)
        left_drive.set_limit_y(0, 0)
        left_drive.set_limit_z(0, 0)

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.tcp_link_name
        )
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_inner_finger_pad"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_inner_finger_pad"
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        left_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        right_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        left_force = torch.linalg.norm(left_forces, axis=1)
        right_force = torch.linalg.norm(right_forces, axis=1)

        left_dir = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        right_dir = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        left_angle = common.compute_angle_between(left_dir, left_forces)
        right_angle = common.compute_angle_between(right_dir, right_forces)
        left_flag = torch.logical_and(
            left_force >= min_force, torch.rad2deg(left_angle) <= max_angle
        )
        right_flag = torch.logical_and(
            right_force >= min_force, torch.rad2deg(right_angle) <= max_angle
        )
        return torch.logical_and(left_flag, right_flag)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., : len(self.arm_joint_names)]
        return torch.max(torch.abs(qvel), dim=1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
