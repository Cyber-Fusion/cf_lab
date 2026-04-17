# Domain Randomization Reference for Legged Locomotion

## Common Randomization Terms

### Friction Coefficient

Makes the policy robust to variations in the friction coefficient, which can be caused by different terrain types or changes in the environment.

```python
physics_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "static_friction_range": (0.3, 1.0),
        "dynamic_friction_range": (0.3, 0.8),
        "restitution_range": (0.0, 0.15),
        "num_buckets": 64,
    },
)
```

### Base Mass

Makes the policy robust to variations in the robot's base mass, which can be caused by carrying payloads or due to modeling inaccuracies.

```python
add_base_mass = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=Params.base_name),
        "mass_distribution_params": (-5.0, 5.0),
        "operation": "add",
    },
)
```

### Force and Torque Perturbations

Makes the policy robust to continuously applied external forces and torques.

```python
base_external_force_torque = EventTerm(
    func=mdp.apply_external_force_torque,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=Params.base_name),
        "force_range": (10.0, 10.0),
        "torque_range": (-5.0, 5.0),
    },
)
```

### Reset Base

**IMPORTANT**: randomizing the base state and the joint positions and velocities increases exploration in the initial part of the episode. Therefore, it proved to be very useful to achieve better performance and faster convergence in the Spot-Like and WTW environments.

```python
reset_base = EventTerm(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {
            "x": (-1.5, 1.5),
            "y": (-1.0, 1.0),
            "z": (-0.5, 0.5),
            "roll": (-0.7, 0.7),
            "pitch": (-0.7, 0.7),
            "yaw": (-1.0, 1.0),
        },
    },
)
```

### Reset Robot Joints

```python
reset_robot_joints = EventTerm(
    func=ayg_mdp.reset_joints_around_default,
    mode="reset",
    params={
        "position_range": (-0.2, 0.2),
        "velocity_range": (-2.5, 2.5),
        "asset_cfg": SceneEntityCfg("robot"),
    },
)
```

## Push

Makes the robot robust to external instantaneous pushes.
```python
push_robot = EventTerm(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(10.0, 15.0),
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
    },
)
```

## Uncommon Randomization Terms