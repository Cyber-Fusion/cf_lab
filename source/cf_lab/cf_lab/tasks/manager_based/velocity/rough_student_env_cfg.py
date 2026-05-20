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
    # itself would raise "A prim already exists").
    #
    # The OffsetCfg is interpreted in ROS body convention (X forward, Y left,
    # Z up) per `convention="ros"`. A 30 deg "pitch-down" — the camera optical
    # axis tilting toward the ground — is a +30 deg rotation about the +Y axis
    # (right-hand rule: +X rotates toward -Z, i.e., forward goes down).
    #
    # Quaternion (w, x, y, z) for +30 deg about Y: (cos(15), 0, sin(15), 0).
    # If a visual check shows the camera looking *up* instead, flip the sign
    # of the y component to (cos(15), 0, -sin(15), 0).
    cfg.scene.depth_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Camera_depth_optical_frame/depth_sensor",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.9659258262890683, 0.0, 0.25881904510252074, 0.0),
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
    # history_length=10 + flatten_history_dim=True gives the depth term a 10-frame
    # history that the env flattens into a single (N, T*H*W*1) vector. The vision
    # StudentTeacher subclass (cf_lab.learning.student_teacher_vision) reshapes this
    # back to (N, T, H, W) inside the depth CNN, after slicing past the leading ego
    # terms in the concatenated policy obs.
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
        history_length=10,
        flatten_history_dim=True,
    )
    # All policy terms now share rank — single concatenated tensor for the
    # RSL-RL DistillationRunner. Term order (set by __dict__ insertion):
    # base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
    # + joint_pos(12) + joint_vel(12) + actions(12) + depth(10*45*80=36000) = 36045.
    cfg.observations.policy.concatenate_terms = True

    # Forward-biased command sampling for the vision student.
    # Teacher was trained on [-1, 1] symmetric ranges (see rough_env_cfg.py:133-135)
    # using its full 187-ray height_scan. The student's forward depth cone can't
    # see what's behind/beside the feet, so asking it to imitate teacher actions
    # on backward/lateral commands is asking it to copy without the information
    # the teacher used. Subset stays in-distribution for the teacher (so labels
    # remain valid) while limiting the student to motions its camera supports.
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


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
