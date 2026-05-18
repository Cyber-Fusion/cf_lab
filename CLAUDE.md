# CLAUDE.md

## Project Overview

**cf_lab** is an Isaac Lab extension for RL-based locomotion control of the AYG quadruped robot (12-DOF, 4 legs × 3 joints: HAA, HFE, KFE). It runs on NVIDIA Isaac Sim/Omniverse.

## Common Commands

### Installation
```bash
python -m pip install -e source/cf_lab
```

### Training

**Important:** You must set BLAS/OpenMP threading variables to avoid crashes (illegal instruction / segfault from OpenBLAS thread conflicts):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Then run training:

```bash
# RSL-RL (primary RL library)
python scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-Ayg-v0

# RL-Games
python scripts/rl_games/train.py --task=Isaac-Velocity-Flat-Ayg-Direct-v0

# SKRL
python scripts/skrl/train.py --task=Isaac-Velocity-Flat-Ayg-Direct-v0
```

### Sanity Checks
```bash
python scripts/list_envs.py           # List registered environments
python scripts/zero_agent.py --task=<TASK>    # Zero-action agent
python scripts/random_agent.py --task=<TASK>  # Random-action agent
```

### Linting & Formatting
```bash
pip install pre-commit
pre-commit run --all-files
```

Ruff is the linter/formatter. Config in root `pyproject.toml`: line length 120, Python 3.10+, Google-style docstrings. Isort has custom section ordering for Omniverse/Isaac Lab imports.

## Architecture

### Two Environment Paradigms

1. **Direct environments** (`tasks/direct/ayg/`): Subclass `DirectRLEnv`. Custom reward/observation/reset logic in `AygEnv`. Registered as `Isaac-Velocity-*-Ayg-Direct-v0`.

2. **Manager-based environments** (`tasks/manager_based/velocity/`): Use Isaac Lab's `ManagerBasedRLEnv` with modular observation, action, reward, and termination managers defined in `*_env_cfg.py` files. Custom MDP logic in `mdp/rewards.py` and `mdp/events.py`. Registered as `Isaac-Velocity-*-Ayg-v0`.

Each environment has flat/rough terrain variants, and manager-based ones also have `-Play-v0` variants (reduced environments for inference).

### Environment Registration

Environments are registered via `gymnasium.register()` in each task's `__init__.py`. Each registration bundles the env config, and agent configs for all three RL libraries (RSL-RL, RL-Games, SKRL). Importing `cf_lab` triggers registration via `tasks/__init__.py`.

#### Registered Task IDs

**Direct** (2):
- `Isaac-Velocity-Flat-Ayg-Direct-v0`
- `Isaac-Velocity-Rough-Ayg-Direct-v0`

**Manager-based velocity** (6):
- `Isaac-Velocity-Flat-Ayg-v0` / `-Play-v0`
- `Isaac-Velocity-Rough-Ayg-v0` / `-Play-v0`
- `Isaac-Velocity-Spot-Like-Flat-Ayg-v0` / `-Play-v0`

**Walk-These-Ways** (4):
- `Isaac-WTW-Flat-Ayg-v0` / `-Play-v0`
- `Isaac-WTW-Rough-Ayg-v0` / `-Play-v0`

### Robot Asset

- URDF: `source/cf_lab/data/Robots/ayg_description/urdf/ayg.urdf`
- USD config: `source/cf_lab/data/Robots/ayg/config.yaml`
- Articulation config: `cf_lab/assets/ayg.py` defines `AYG_CFG` with actuator model, initial state, and physics properties
- Joint naming: `{LF|RF|LH|RH}_{HAA|HFE|KFE}` (12 joints). Body names: `Base`, `{LF|RF|LH|RH}_{Hip|Thigh|Shank|Foot}` (17 bodies)
- Common regex patterns: joints `".*HAA"`, `".*HFE"`, `".*KFE"`; bodies `".*_Foot"`, `".*_Shank"`, `".*_Thigh"`, `".*_Hip"`, `"Base"`
- Leg prefixes: `LF` (left front), `RF` (right front), `LH` (left hind), `RH` (right hind)

### Key Directories

- `source/cf_lab/cf_lab/assets/` — Robot articulation configs
- `source/cf_lab/cf_lab/tasks/` — All environment definitions
- `source/cf_lab/cf_lab/tasks/*/agents/` — RL algorithm hyperparameter configs (YAML + Python)
- `source/cf_lab/data/Robots/` — URDF/USD robot descriptions
- `scripts/` — Training, play, and utility scripts

### Adding a New Environment

#### Manager-based (recommended)

1. Create `*_env_cfg.py` with a `@configclass` config inheriting from an existing env cfg (e.g., `AygFlatEnvCfg` in `tasks/manager_based/velocity/flat_env_cfg.py`). Override scenes, rewards, terminations, etc. in `__post_init__`.
2. Create a `_PLAY` variant that inherits from the training config and reduces `num_envs`.
3. Add agent config files in an `agents/` subdirectory (at minimum `rsl_rl_ppo_cfg.py`; optionally RL-Games and SKRL YAML files).
4. Register both training and play variants with `gym.register()` in the task's `__init__.py`. Include `env_cfg_entry_point` and agent config entry points.
5. Import the task package from `cf_lab/tasks/__init__.py`.

Templates: `tasks/manager_based/velocity/` (full example with flat/rough/spot-inspired variants).

#### Direct

1. Create `*_env.py` subclassing `DirectRLEnv` with custom `_setup_scene`, `_pre_physics_step`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`.
2. Create `*_env_cfg.py` with the scene and simulation config.
3. Add agent configs and register as above, but use `entry_point=f"{__name__}.your_env:YourEnvClass"`.

Template: `tasks/direct/ayg/` (flat + rough variants).

### Distillation Policy

A student-teacher distillation pipeline is wired into the rough velocity task (`Isaac-Velocity-Rough-Ayg-v0` / `-Play-v0`).

- **Teacher**: a previously trained PPO checkpoint (same `experiment_name="ayg_rough"`). Its inputs are the original `policy` observation group — privileged proprio + `base_lin_vel` + `height_scan`.
- **Student**: proprio only (no `base_lin_vel`, no `height_scan`), optionally augmented with a sparse forward depth point cloud from a `RayCasterCameraCfg` (`front_depth_camera`) attached to the robot's base. Defined as the `student` group in `AygObservationsCfg`.
- **Routing** (in `AygRoughDistillationRunnerCfg`): `obs_groups = {"policy": ["student"], "teacher": ["policy"]}` — the env's `student` group feeds the student network, and the original `policy` group (teacher inputs) feeds the teacher network.
- **Config**: `tasks/manager_based/velocity/agents/rsl_rl_distillation_cfg.py` defines `AygRoughDistillationRunnerCfg` (subclass of `RslRlDistillationRunnerCfg`) using `RslRlDistillationStudentTeacherCfg` + `RslRlDistillationAlgorithmCfg`. Registered as `rsl_rl_distillation_cfg_entry_point` in the task's `__init__.py`.
- **Training**: pass `--agent=rsl_rl_distillation_cfg_entry_point` plus `--load_run <timestamp> --checkpoint <model_*.pt>` to load the teacher. `scripts/rsl_rl/train.py` detects `algorithm.class_name == "Distillation"` and uses `DistillationRunner` instead of `OnPolicyRunner`, auto-loading the checkpoint as the teacher.

When adding distillation to another task: (1) extend the env's `ObservationsCfg` with a `student: ObsGroup` defining the deployable observations; (2) add any sensors the student needs to the scene cfg; (3) create `rsl_rl_distillation_cfg.py` mirroring `AygRoughDistillationRunnerCfg` (matching `experiment_name` to the teacher's PPO run); (4) register `rsl_rl_distillation_cfg_entry_point` in the gym kwargs.

## Code Style

- License header required on all `.py` and `.yaml` files (enforced by pre-commit)
- `E402` (imports not at top) is intentionally allowed — Isaac Lab requires specific import ordering
- `F401` (unused imports) is allowed in `__init__.py` files
- Import ordering (enforced by isort sections in `pyproject.toml`): stdlib > third-party > omniverse (`isaacsim`, `omni`, `pxr`, `carb`) > `isaaclab` core > isaaclab extensions (`isaaclab_assets`, `isaaclab_rl`, etc.) > `cf_lab` (first-party) > local-folder (relative imports)
- Use `@configclass` (from `isaaclab.utils`) for env configs, not `@dataclass`
