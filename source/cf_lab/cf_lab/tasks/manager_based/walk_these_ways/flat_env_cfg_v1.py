# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .flat_env_cfg import AygFlatWTWEnvCfg


@configclass
class AygFlatWTWEnvCfgV1(AygFlatWTWEnvCfg):
    """Flat WTW variant without observation history and with the `wtw_no_hist` reward tweaks.

    Differences from `AygFlatWTWEnvCfg` (v0):
    - Policy observation history disabled (current-step only).
    - Slight obs scaling changes (base_ang_vel policy/critic, base_lin_vel/joint_vel critic).
    - `stand_*_when_zero_command` rewards gate on the velocity command only (no body-velocity check).
    - Bumped weights for the four "no command" rewards.
    """

    def __post_init__(self):
        super().__post_init__()

        # ========================== Observations ============================ #
        # Drop the 5-step flattened history — policy sees only the current timestep.
        self.observations.policy.history_length = None

        # Match the obs scales of the trained no_hist export.
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.critic.base_lin_vel.scale = 2.0
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.joint_vel.scale = 0.05

        # ============================ Rewards =============================== #
        # Re-enable and bump the no-command penalties (v0 flat zeroed two of these).
        self.rewards.stand_still_when_zero_command.weight = -0.1
        self.rewards.zero_vel_when_zero_command.weight = -20.0
        self.rewards.zero_ang_vel_when_zero_command.weight = -10.0

        # Gate stand/stand_still on the velocity command only (don't wait for the body to slow down).
        self.rewards.stand_when_zero_command.params["use_cmd_only_gate"] = True
        self.rewards.stand_still_when_zero_command.params["use_cmd_only_gate"] = True


@configclass
class AygFlatWTWEnvCfgV1_PLAY(AygFlatWTWEnvCfgV1):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # Skip the velocity curriculum so play sees the full per-gait ranges immediately.
        self.curriculum.gait_velocity_curriculum.params["anneal_steps"] = 1
