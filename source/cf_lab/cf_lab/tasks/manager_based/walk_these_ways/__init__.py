# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-WTW-Flat-Ayg-v0",
    entry_point="cf_lab.envs:WTWManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AygFlatWTWEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AygFlatWTWPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WTW-Flat-Ayg-Play-v0",
    entry_point="cf_lab.envs:WTWManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AygFlatWTWEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AygFlatWTWPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WTW-Rough-Ayg-v0",
    entry_point="cf_lab.envs:WTWManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:AygRoughWTWEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AygRoughWTWPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WTW-Rough-Ayg-Play-v0",
    entry_point="cf_lab.envs:WTWManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:AygRoughWTWEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AygRoughWTWPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WTW-Cobblestone-Ayg-v0",
    entry_point="cf_lab.envs:WTWManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cobblestone_env_cfg:AygCobblestoneWTWEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AygCobblestoneWTWPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WTW-Cobblestone-Ayg-Play-v0",
    entry_point="cf_lab.envs:WTWManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cobblestone_env_cfg:AygCobblestoneWTWEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AygCobblestoneWTWPPORunnerCfg",
    },
)
