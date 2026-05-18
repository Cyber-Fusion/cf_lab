# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

from .rough_env_cfg import AygRoughEnvCfg, AygRoughEnvCfg_PLAY

# D555 depth contract — must match Gazebo + real driver in Ayg/.
_D555_HFOV_RAD = 1.5184  # 87 deg, datasheet v1.1
_FOCAL_LENGTH = 24.0  # Isaac Lab default; horizontal_aperture is solved from HFoV
_HORIZ_APERTURE = 2.0 * _FOCAL_LENGTH * math.tan(_D555_HFOV_RAD / 2.0)


@configclass
class StudentObservationsCfg(ObservationsCfg):
    """Two-group obs for DAgger distillation.

    `policy` is what the student network consumes at training and deploy time:
    proprioception (no base_lin_vel — real robot can't measure cleanly) + depth.

    `teacher` mirrors what the locked Phase 1 teacher was trained on (ego + height_scan)
    and is used only at training time to generate supervision targets via the frozen teacher.
    """

    @configclass
    class TeacherCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    teacher: TeacherCfg = TeacherCfg()


def _apply_student_overlay(cfg) -> None:
    # D555 depth sensor. Spawned as a child sub-prim of the URDF link
    # (parent prim already exists after URDF import; spawning at the link path
    # itself would raise "A prim already exists"). Zero offset inherits the
    # link's pose, which is the ROS optical convention baked into the URDF.
    cfg.scene.depth_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Camera_depth_optical_frame/depth_sensor",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=_HORIZ_APERTURE,
            clipping_range=(0.26, 9.0),
        ),
        width=80,
        height=45,
        update_period=1.0 / 30.0,
        depth_clipping_behavior="zero",
    )

    # Student `policy` group: strip privileged terms, add depth with sim-to-real noise.
    cfg.observations.policy.base_lin_vel = None
    cfg.observations.policy.height_scan = None
    cfg.observations.policy.depth = ObsTerm(
        func=mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            "data_type": "distance_to_image_plane",
            "normalize": True,
        },
        noise=Gnoise(mean=0.0, std=0.02),
    )
    # depth is (N, H, W, 1); can't concatenate with (N, K) vector terms.
    cfg.observations.policy.concatenate_terms = False


@configclass
class AygRoughStudentEnvCfg(AygRoughEnvCfg):
    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_student_overlay(self)


@configclass
class AygRoughStudentEnvCfg_PLAY(AygRoughEnvCfg_PLAY):
    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_student_overlay(self)
