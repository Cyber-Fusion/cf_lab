# CLAUDE.md

## Conventions

All the manager-based environments follow the same structure:
```
├── <env_name>/
    ├── agents  # Config files for different RL libraries (RSL-RL, RL-Games, SKRL)
    │   ├── rl_games_flat_ppo_cfg.yaml
    │   ├── rl_games_rough_ppo_cfg.yaml
    │   └── rsl_rl.py
    │   ├── skrl_flat_ppo_cfg.yaml
    │   └── skrl_rough_ppo_cfg.yaml
    ├── mdp/  # Custom MDP logic (optional)
    │   ├── commands/  # custom commands for the environment
    │   ├── curriculums.py
    │   ├── events.py
    │   ├── observations.py
    │   ├── terminations.py
    │   └── rewards.md
    ├── <robot_name>_params.py  # optional robot-specific parameters and names
    ├── flat_env_cfg.py  # Environment config for flat terrain (overrides rough_env_cfg.py)
    ├── rough_env_cfg.py  # Environment config for rough terrain (overrides <env_name>_cfg.py)
    └── <env_name>_cfg.py  # Environment config (will be overridden by flat/rough configs)
```
**Important**: The reward weights of the main environment config are overridden by the flat/rough configs. This allows us to have different reward weightings for flat vs rough terrain.