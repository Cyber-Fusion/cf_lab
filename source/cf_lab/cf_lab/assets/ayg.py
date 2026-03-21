"""Configuration for the Ayg robots.

The following configuration parameters are available:

* :obj:`AYG_CFG`: The AYG robot

Reference:

* https://github.com/AYG/ayg_description

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from cf_lab import CF_LAB_DATA_DIR

##
# Configuration - Actuators.
##

AYG_MOTOR_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    # Robstride 03 motor specs:
    #   Source: Ayg/src/libs/robstride_sdk/include/robstride_sdk/types.hpp
    #   Source: kscale metadata.json (vendor-confirmed values)
    #   Max torque: 60 Nm, Max velocity: 18.849 rad/s (~6π), Max Kp: 5000, Max Kd: 100
    #   Armature (rotor inertia): 0.02 kg·m², Friction loss (Coulomb): 0.2 Nm
    # Real HW PD gains (from Ayg/src/control/ayg_control/config/ayg_hw_controllers.yaml):
    #   Kp: 50.0, Kd: 3.0, Ki: 0.0 — all joints
    # Real HW control frequency: 160 Hz, HW velocity safety limit: 5.0 rad/s
    saturation_effort=60.0,
    effort_limit=50.0,
    velocity_limit=18.849,
    stiffness=50.0,
    damping=3.0,
    armature=0.02,
    friction=0.2,
)
"""Configuration for AYG's motor with DC actuator model (Robstride 03)."""


##
# Configuration - Articulation.
##

AYG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{CF_LAB_DATA_DIR}/Robots/ayg_description/urdf/ayg.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*HAA": 0.0,       # all HAA
            ".*HFE": 0.0,       # all HFE
            "LF_KFE": -0.2,     # front knees slightly bent
            "RF_KFE": -0.2,
            "LH_KFE": -0.25,    # hind knees slightly more bent
            "RH_KFE": -0.25,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={"legs": AYG_MOTOR_SIMPLE_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of Ayg robot using DC actuator."""
