# Observations Reference for Legged Locomotion

In Isaac Lab, the observations are split into two categories: policy observations (for the actor) and critic observations (for the critic). The policy observations are used by the policy to make decisions, while the critic observations are used by the critic to evaluate the value of the current state.

Two common RL frameworks can be used:
- **Shared observations**: the policy and the critic share the same observation space. This is the default configuration in Isaac Lab.
  - With shared observations, a Teacher policy can be trained with privileged information not available to the real robot. Then, a Student policy is trained is a supervised manner to mimic the Teacher policy, but with access only to the observations available on the real robot.
- **Separate observations**: the policy and the critic have different observation spaces. This is called asymmetric actor-critic. The policy uses only the observations available on the real robot and uses noisy observations. The critic has access to privileged information and is trained with clean observations. This configuration is also supported in Isaac Lab. With asymmetric actor-critic, the policy benefits from history of past observations, while the critic can use privileged information to have an almost full state representation (and thus it benefits less from history of past observations).

## Policy Observations

```python
@configclass
class PolicyCfg(ObsGroup):
    """Observations for the actor (no privileged info like ground-truth linear velocity)."""

    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
        scale=0.2,
        clip=(-100, 100),
        noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        params={"asset_cfg": SceneEntityCfg("robot")},
        clip=(-100, 100),
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot")},
        clip=(-100, 100),
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot")},
        scale=0.05,
        clip=(-100, 100),
        noise=Unoise(n_min=-0.5, n_max=0.5),
    )
    actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
```

The base linear velocity is **not** included in the policy observations because it is not directly measurable on the real robot and it would require very good estimation accuracy. Additionally, the estimated base linear velocity noise is not zero-mean and uncorrelated.

## Critic Observations

```python
@configclass
class CriticCfg(ObsGroup):
    """Privileged observations for the critic (includes ground-truth velocity, foot state)."""

    # -- shared terms (no noise, no scaling)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
    projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
    actions = ObsTerm(func=mdp.last_action)
    # -- privileged terms
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
    foot_heights = ObsTerm(
        func=ayg_mdp.foot_heights,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=Params.feet_names)},
    )
    foot_air_time = ObsTerm(
        func=ayg_mdp.foot_air_time,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=Params.feet_names)},
    )
    foot_contact = ObsTerm(
        func=ayg_mdp.foot_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=Params.feet_names), "threshold": 1.0},
    )
    foot_contact_forces = ObsTerm(
        func=ayg_mdp.foot_contact_forces,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=Params.feet_names)},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
```