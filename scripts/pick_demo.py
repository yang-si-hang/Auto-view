import argparse
import sys
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
import sapien
import torch
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_euler_angles,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_matrix,
)
from mani_skill.utils.wrappers.record import RecordEpisode

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import envs.pick_single_ycb_ur10e
from utils.const import DATA_PATH


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DATA_PATH / "demos" / "PickSingleYCBUR10e-v1")
    parser.add_argument("--trajectory-name", default="trajectory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--save-video", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--move-camera", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--waypoint-spacing", type=float, default=0.02)
    parser.add_argument("--waypoint-tolerance", type=float, default=0.005)
    return parser.parse_args()


def set_sensor_camera_pose(env, camera_name, pose):
    base_env = env.unwrapped
    sensor = base_env._sensors[camera_name]
    if not isinstance(pose, sapien.Pose):
        for i, render_camera in enumerate(sensor.camera._render_cameras):
            render_camera.set_local_pose(pose[i].sp)
    else:
        sensor.camera.set_local_pose(pose)
    sensor.camera._cached_model_matrix = None
    sensor.camera._cached_extrinsic_matrix = None
    sensor.camera._cached_local_pose = None
    sensor.config.pose = pose


def build_waypoints(env):
    base_env = env.unwrapped
    device = base_env.device
    obj_pos = base_env.obj.pose.p
    goal_pos = base_env.goal_site.pose.p
    xy = obj_pos[:, :2]
    n = xy.shape[0]
    grasp_quat = torch.tensor(
        [0.0, -1.0, 0.0, 0.0], dtype=torch.float32, device=device
    ).repeat(n, 1)

    positions = torch.stack(
        [
            torch.cat(
                [xy, torch.full((n, 1), 0.15, dtype=torch.float32, device=device)],
                dim=1,
            ),
            torch.cat(
                [xy, torch.full((n, 1), 0.09, dtype=torch.float32, device=device)],
                dim=1,
            ),
            torch.cat(
                [xy, torch.full((n, 1), 0.09, dtype=torch.float32, device=device)],
                dim=1,
            ),
            goal_pos.to(dtype=torch.float32),
        ],
        dim=1,
    )
    orientations = torch.stack([grasp_quat, grasp_quat, grasp_quat, grasp_quat], dim=1)
    grippers = torch.stack(
        [
            -torch.ones(n, dtype=torch.float32, device=device),
            -torch.ones(n, dtype=torch.float32, device=device),
            0.7*torch.ones(n, dtype=torch.float32, device=device),
            0.7*torch.ones(n, dtype=torch.float32, device=device),
        ],
        dim=1,
    )
    return dict(position=positions, orientation=orientations, gripper=grippers)


def slerp_quat(q0, q1, t):
    q0 = q0 / torch.linalg.norm(q0, dim=1, keepdim=True).clamp_min(1e-8)
    q1 = q1 / torch.linalg.norm(q1, dim=1, keepdim=True).clamp_min(1e-8)

    dot = torch.sum(q0 * q1, dim=1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.sum(q0 * q1, dim=1, keepdim=True).clamp(-1.0, 1.0)

    linear = dot > 0.9995
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp_min(1e-8)
    theta = theta_0 * t
    s0 = torch.cos(theta) - dot * torch.sin(theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    quat = s0 * q0 + s1 * q1
    quat = torch.where(linear, (1 - t) * q0 + t * q1, quat)
    return quat / torch.linalg.norm(quat, dim=1, keepdim=True).clamp_min(1e-8)


def interpolate_waypoints(env, waypoints, spacing):
    if spacing <= 0:
        raise ValueError("--waypoint-spacing must be positive")

    base_env = env.unwrapped
    start_pos = base_env.agent.tcp.pose.p
    start_quat = base_env.agent.tcp.pose.q
    start_gripper = waypoints["gripper"][:, :1]

    positions = torch.cat([start_pos[:, None, :], waypoints["position"]], dim=1)
    orientations = torch.cat([start_quat[:, None, :], waypoints["orientation"]], dim=1)
    grippers = torch.cat([start_gripper, waypoints["gripper"]], dim=1)

    dense_positions = []
    dense_orientations = []
    dense_grippers = []
    for idx in range(positions.shape[1] - 1):
        p0 = positions[:, idx]
        p1 = positions[:, idx + 1]
        q0 = orientations[:, idx]
        q1 = orientations[:, idx + 1]
        g0 = grippers[:, idx]
        g1 = grippers[:, idx + 1]

        max_dist = torch.linalg.norm(p1 - p0, dim=1).max()
        num_steps = int(torch.ceil((max_dist / spacing).clamp_min(1.0)).item())
        for step in range(1, num_steps + 1):
            t = torch.full(
                (positions.shape[0], 1),
                step / num_steps,
                dtype=torch.float32,
                device=base_env.device,
            )
            dense_positions.append((1 - t) * p0 + t * p1)
            dense_orientations.append(slerp_quat(q0, q1, t))
            dense_grippers.append((1 - t[:, 0]) * g0 + t[:, 0] * g1)

    return dict(
        position=torch.stack(dense_positions, dim=1),
        orientation=torch.stack(dense_orientations, dim=1),
        gripper=torch.stack(dense_grippers, dim=1),
    )


def waypoint_action(env, target_pos, target_quat_world, target_gripper):
    base_env = env.unwrapped

    arm_controller = base_env.agent.controller.controllers["arm"]
    arm_pos_action_scale = float(arm_controller.config.pos_upper)
    arm_rot_action_scale = float(arm_controller.config.rot_lower)

    tcp_pos = base_env.agent.tcp.pose.p
    tcp_quat_world = base_env.agent.tcp.pose.q
    delta_pos_world = target_pos - tcp_pos

    root_quat = base_env.agent.robot.root.pose.q
    root_quat_inv = quaternion_invert(root_quat)
    delta_pos_base = quaternion_apply(root_quat_inv, delta_pos_world)

    delta_quat_body = quaternion_multiply(quaternion_invert(tcp_quat_world), target_quat_world)
    delta_rot_body = matrix_to_euler_angles(quaternion_to_matrix(delta_quat_body), "XYZ")

    action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=base_env.device)
    delta_action = delta_pos_base / arm_pos_action_scale
    scale = torch.maximum(
        torch.max(torch.abs(delta_action), dim=1, keepdim=True).values,
        torch.ones((delta_action.shape[0], 1), dtype=torch.float32, device=base_env.device),
    )
    delta_action = delta_action / scale
    rot_action = delta_rot_body / arm_rot_action_scale
    rot_scale = torch.maximum(
        torch.linalg.norm(rot_action, dim=1, keepdim=True),
        torch.ones((rot_action.shape[0], 1), dtype=torch.float32, device=base_env.device),
    )
    rot_action = rot_action / rot_scale
    if action.ndim == 1:
        action[:3] = delta_action[0]
        action[3:6] = rot_action[0]
        action[6] = target_gripper[0]
    else:
        action[:, :3] = delta_action
        action[:, 3:6] = rot_action
        action[:, 6] = target_gripper
    return action, torch.linalg.norm(delta_pos_world, axis=1)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if isinstance(args.output_dir, str) else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "PickSingleYCBUR10e-v1",
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        num_envs=args.num_envs,
    )
    env = RecordEpisode(
        env,
        output_dir=str(output_dir),
        trajectory_name=args.trajectory_name,
        save_trajectory=True,
        save_video=args.save_video,
        max_steps_per_video=args.num_steps,
        save_on_reset=False,
        source_type="waypoint_policy",
        source_desc="One waypoint-following trajectory for LeRobot conversion testing.",
    )

    obs, info = env.reset(seed=args.seed)
    print("action_space:", env.action_space)

    waypoints = interpolate_waypoints(
        env,
        build_waypoints(env),
        spacing=args.waypoint_spacing,
    )
    device = env.unwrapped.device
    waypoint_idx = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    env_ids = torch.arange(args.num_envs, dtype=torch.long, device=device)
    num_waypoints = waypoints["position"].shape[1]
    print(f"interpolated waypoints: {num_waypoints} targets")

    for step in range(args.num_steps):
        if args.move_camera:
            angle = torch.tensor(
                2 * torch.pi * step / max(args.num_steps, 1),
                dtype=torch.float32,
                device=device,
            )
            eye = torch.stack(
                [
                    0.05 + 0.45 * torch.cos(angle),
                    0.20 + 0.45 * torch.sin(angle),
                    torch.tensor(0.55, dtype=torch.float32, device=device),
                ]
            )
            eye = eye[None, :].repeat(args.num_envs, 1)
            target = env.unwrapped.obj.pose.p
            camera_pose = sapien_utils.look_at(eye=eye, target=target)
            set_sensor_camera_pose(env, "base_camera", camera_pose)

        target_pos = waypoints["position"][env_ids, waypoint_idx]
        target_quat = waypoints["orientation"][env_ids, waypoint_idx]
        target_gripper = waypoints["gripper"][env_ids, waypoint_idx]
        action, dist = waypoint_action(env, target_pos, target_quat, target_gripper)
        reached = (dist < args.waypoint_tolerance) & (
            waypoint_idx < num_waypoints - 1
        )
        if torch.any(reached):
            waypoint_idx[reached] += 1
            target_pos = waypoints["position"][env_ids, waypoint_idx]
            target_quat = waypoints["orientation"][env_ids, waypoint_idx]
            target_gripper = waypoints["gripper"][env_ids, waypoint_idx]
            action, dist = waypoint_action(env, target_pos, target_quat, target_gripper)

        print(f"step {step} =====: \naction={action}, dist={dist}")
        obs, reward, terminated, truncated, info = env.step(action)
        if bool(torch.as_tensor(terminated, device=device).any()):
            break
        if bool(torch.as_tensor(truncated, device=device).any()):
            break

    env.flush_trajectory()
    if args.save_video:
        env.flush_video()

    h5_path = output_dir / f"{args.trajectory_name}.h5"
    json_path = output_dir / f"{args.trajectory_name}.json"
    env.close()

    print(f"saved trajectory: {h5_path}")
    print(f"saved metadata:   {json_path}")


if __name__ == "__main__":
    main()
