# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from cf_lab.tasks.manager_based.walk_these_ways.wtw_params import WalkTheseWaysParams as Params

from .flat_env_cfg import AygFlatWTWEnvCfg
from .flat_env_cfg_v1 import AygFlatWTWEnvCfgV1

##
# Pre-defined configs
##
from cf_lab.tasks.manager_based.velocity.spot_inspired_env_cfg import COBBLESTONE_ROAD_CFG  # isort: skip


def _apply_cobblestone_overrides(cfg) -> None:
    """Swap to cobblestone terrain and restore the height scanner that Flat removed."""
    # swap terrain to cobblestone road (same as Spot Like env)
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Restore the height scanner that Flat nulled out — cobblestone bumps make
    # terrain-relative measurements meaningful again for these rewards.
    cfg.scene.height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + Params.base_name,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    cfg.rewards.base_height_l2.params["sensor_cfg"] = Params.height_scanner
    cfg.rewards.footswing_height.params["height_scanner_cfg"] = Params.height_scanner
    cfg.rewards.foot_clearance.params["height_scanner_cfg"] = Params.height_scanner

    # No height-scan in the policy/critic observations.
    cfg.observations.policy.height_scan = None
    cfg.observations.critic.height_scan = None

    # Cobblestone has no terrain levels.
    cfg.curriculum.terrain_levels = None


@configclass
class AygCobblestoneWTWEnvCfg(AygFlatWTWEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        _apply_cobblestone_overrides(self)


@configclass
class AygCobblestoneWTWEnvCfg_PLAY(AygCobblestoneWTWEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None


# =================================== v1 (no-history) =================================== #
@configclass
class AygCobblestoneWTWEnvCfg_V1(AygFlatWTWEnvCfgV1):
    def __post_init__(self):
        # post init of parent (v1 flat base)
        super().__post_init__()
        _apply_cobblestone_overrides(self)


@configclass
class AygCobblestoneWTWEnvCfg_V1_PLAY(AygCobblestoneWTWEnvCfg_V1):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
