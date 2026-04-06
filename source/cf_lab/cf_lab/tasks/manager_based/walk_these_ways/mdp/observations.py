from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, ContactSensor, Imu, RayCaster, RayCasterCamera, TiledCamera
from isaaclab.sensors import CameraData
import cv2

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.envs.base_env import BaseEnv


def robot_joint_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint torque of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.applied_torque.to(device)


def robot_joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint acc of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.joint_acc.to(device)


def robot_feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """contact force of the robot feet"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    contact_force_tensor = contact_sensor.data.net_forces_w_history.to(device)
    return contact_force_tensor.view(contact_force_tensor.shape[0], -1)


def robot_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_mass.to(device)


def robot_inertia(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """inertia of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inertia_tensor = asset.data.default_inertia.to(device)
    return inertia_tensor.view(inertia_tensor.shape[0], -1)


def robot_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint positions of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_pos.to(device)


def robot_joint_stiffness(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint stiffness of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_stiffness.to(device)


def robot_joint_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint damping of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_damping.to(device)


def robot_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)


def robot_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """velocity of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_vel_w.to(device)


def robot_material_properties(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """material properties of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    material_tensor = asset.root_physx_view.get_material_properties().to(device)
    return material_tensor.view(material_tensor.shape[0], -1)


def robot_center_of_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """center of mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    com_tensor = asset.root_physx_view.get_coms().clone().to(device)
    return com_tensor.view(com_tensor.shape[0], -1)


def get_clock_inputs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-foot clock signals with stance/swing warping.

    Returns sin(2*pi*warped_foot_index) for each of the 4 feet, matching
    the original Walk These Ways ``clock_inputs``.

    Returns:
        torch.Tensor: Shape (num_envs, 4).
    """
    command_term = env.command_manager.get_term("gait_command")
    durations = command_term.command[:, 1]
    foot_indices = command_term.foot_indices.clone()

    dur = durations.unsqueeze(1).expand_as(foot_indices)
    stance = foot_indices < dur
    swing = foot_indices >= dur
    foot_indices[stance] = foot_indices[stance] * (0.5 / dur[stance])
    foot_indices[swing] = 0.5 + (foot_indices[swing] - dur[swing]) * (0.5 / (1 - dur[swing]))

    return torch.sin(2 * torch.pi * foot_indices)


def get_gait_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current gait command parameters as observation.

    Returns:
        torch.Tensor: The gait command parameters [frequency, offset, duration].
                     Shape: (num_envs, 3).
    """
    return env.command_manager.get_command(command_name)


def base_external_force_torque(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """External force and torque applied to the base body.

    Returns the concatenated force (3) and torque (3) vectors. Shape: (num_envs, 6).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    forces = asset.permanent_wrench_composer.composed_force_as_torch[:, asset_cfg.body_ids, :]
    torques = asset.permanent_wrench_composer.composed_torque_as_torch[:, asset_cfg.body_ids, :]
    return torch.cat([forces.reshape(env.num_envs, -1), torques.reshape(env.num_envs, -1)], dim=-1)


def friction_coefficients(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Average static and dynamic friction coefficients across all shapes.

    Returns the mean (static_friction, dynamic_friction). Shape: (num_envs, 2).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # material_properties shape: (num_envs, total_num_shapes, 3)
    # channels: (static_friction, dynamic_friction, restitution)
    materials = asset.root_physx_view.get_material_properties()
    # Average across all shapes, keep only static + dynamic friction -> (num_envs, 2)
    return materials[:, :, :2].mean(dim=1)


def robot_base_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot base"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)


def process_depth_image(env: BaseEnv, sensor_cfg: SceneEntityCfg, data_type: str, visualize=False) -> torch.Tensor:
    """Process the depth image."""
    # extract the used quantities (to enable type-hinting)
    sensor: CameraData = env.scene.sensors[sensor_cfg.name].data

    output = sensor.output[data_type].clone().unsqueeze(1)
    near_clip = 0.3
    far_clip = 2.0
    output[torch.isnan(output)] = far_clip
    output[torch.isinf(output)] = far_clip

    output = torch.clip(output, near_clip, far_clip)
    output = output - near_clip

    if visualize:
        depth_image_size = (output.shape[2], output.shape[3])
        output_clone = output.clone().reshape(env.num_envs, depth_image_size[0], depth_image_size[1])[0,:,:]
        window_name = "Depth Image"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow("Depth Image", output_clone.cpu().numpy())
        cv2.waitKey(1)

    return output.reshape(env.num_envs, -1)
