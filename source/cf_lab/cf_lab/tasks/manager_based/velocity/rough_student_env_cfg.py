# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

from . import mdp as ayg_mdp
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
    # Geometry source of truth: ayg_description/urdf/camera.xacro. The URDF
    # places the Camera link origin at (0.2475, 0, 0.105) in Base (raised 1 cm
    # vs. earlier value of 0.095 to mirror the planned hardware bracket
    # revision) and translates Camera_body by (0.0527, -0.024, 0) in Camera-
    # local axes so the D450 optical-module reference sits at (+5.27 cm fwd,
    # -2.4 cm vertical) of the Camera link in robot body frame — i.e., inside
    # the D555 housing where the depth/IR/IMU sensors physically live. The
    # depth optical center in Base is therefore (0.3002, 0, 0.081).
    #
    # The OffsetCfg below adds only the pitch-down rotation. pos stays
    # (0,0,0) because the URDF already encodes the correct optical-center
    # position. The OffsetCfg is interpreted in ROS body convention
    # (X forward, Y left, Z up) per `convention="ros"`. A 30 deg "pitch-down"
    # — the camera optical axis tilting toward the ground — is a +30 deg
    # rotation about the +Y axis (right-hand rule: +X rotates toward -Z,
    # i.e., forward goes down).
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
        height=60,
        update_period=1.0 / 30.0,
        depth_clipping_behavior="zero",
    )

    # Student `policy` group: strip privileged terms, add a strided depth history +
    # an aligned strided proprio history.
    #
    # Why strided (not contiguous): the env steps at 50 Hz (decimation 4 * dt 0.005)
    # but the depth camera refreshes at 30 Hz (update_period 1/30), so a contiguous
    # 20-frame stack spans only ~0.4 s and ~8 of its frames are duplicates. Sampling
    # 10 frames every `stride=4` control steps (12.5 Hz < 30 Hz) yields 10 *distinct*
    # frames spanning ~0.72 s. DepthHistoryStrided keeps its own on-device ring buffer
    # and emits only the 10 strided frames, so the rollout storage holds 10 frames
    # (not the ~37 it rings internally) — halving the previous depth obs footprint.
    #
    # ProprioHistoryStrided emits a matching 10-frame stack of a 21-dim proprio vector
    # so the temporal encoder can infer inter-frame ego-motion (camera registration).
    cfg.observations.policy.base_lin_vel = None
    cfg.observations.policy.height_scan = None
    cfg.observations.policy.depth = ObsTerm(
        func=ayg_mdp.DepthHistoryStrided,
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            "data_type": "distance_to_image_plane",
            "num_frames": 10,
            "stride": 4,
            "far_clip": 9.0,
        },
        noise=Gnoise(mean=0.0, std=0.02),
    )
    cfg.observations.policy.proprio_history = ObsTerm(
        func=ayg_mdp.ProprioHistoryStrided,
        params={"num_frames": 10, "stride": 4, "command_name": "base_velocity"},
    )
    # All policy terms now share rank — single concatenated tensor for the RSL-RL
    # DistillationRunner. Active term order (set by __dict__ insertion, base_lin_vel
    # and height_scan are None -> skipped):
    #   base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
    #   + joint_pos(12) + joint_vel(12) + actions(12)            = 45 (ego, sliced first)
    #   + depth(10*60*80 = 48000)
    #   + proprio_history(10*21 = 210)                           => total 48255.
    cfg.observations.policy.concatenate_terms = True

    # Forward-only command sampling for the vision student.
    # Teacher was trained on [-1, 1] symmetric ranges (see rough_env_cfg.py:133-135)
    # using its full 187-ray height_scan. The student's forward depth cone can't
    # see what's behind/beside the feet, so asking it to imitate teacher actions
    # on backward/lateral commands is asking it to copy without the information
    # the teacher used. Earlier narrowed ranges (lat ±0.3, yaw ±0.5) still didn't
    # beat the blind baseline, so per team-lead suggestion we zero out lateral
    # and yaw entirely: isolate the question "does depth help us walk forward
    # over rough terrain at all?" before reintroducing the other axes.
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


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
