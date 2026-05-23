# WTW Env Changelog

## Commands

- Add `MultiGaitVelocityCommand` with per-gait sampling ranges (trot / pace / bound / pronk).
- Add a linear curriculum on the maximum commanded `|vx|`, ramping from `initial_max_lin_vel_x` to the per-gait final range over `anneal_steps`.
- Couple secondary commands to `|vx|`: scale `vy`, `omega`, body pitch, body roll, and base-height deviation by `1 / max(1, |vx|/knee)` so the policy is not asked to track aggressive secondary objectives at high forward speed.
- Couple gait frequency and contact duration to `|vx|`: interpolate between low-`|vx|` and high-`|vx|` ranges so high speeds enable shorter stance / higher cadence (aerial trot). Stance fraction: `(0.5, 0.75)` at low `|vx|` → `(0.35, 0.5)` at high `|vx|` (was fixed at `0.5`); frequency: `(1.5, 2.0)` → `(2.5, 3.5)` (was `(1.5, 3.0)`).
- Track canonical gait id per env on `GaitCommandQuad` (`gait_ids`) so sibling terms can read it.

## Rewards

- Switch `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` back to the standard squared-error exp kernel (`mdp.track_lin_vel_xy_exp` / `track_ang_vel_z_exp`) with tighter `std=0.5`.
- Replace `track_lin_vel_xy_exp` with `track_lin_vel_xy_exp_speed_adaptive`: sigma linearly interpolates `0.25 → 0.5` over commanded `|v_xy| ∈ [1.0, 2.0]` (looser tracking at high speeds), and the reward is scaled by `max(1.0, 1.0 + 0.5 * (|v_cmd| - 1.0))` like the spot-like `base_linear_velocity_reward` ramp.
- Add `use_cmd_only_gate` option to `stand_when_zero_command` and `stand_still_when_zero_command` to gate on the velocity command only (no body-velocity check).
- Tighten `gait_vel_sigma` (1.0 → 0.25); bump `foot_clearance` (-150 → -200) and `undesired_contacts` (-10 → -50) in the rough config.

## Configs

- Add `Isaac-WTW-Flat-Ayg-v1` / `-Play-v1` (no observation history, tweaked obs scales, re-enabled no-command rewards, smaller PPO net 128×128×128).
- Flat env: pin the gait to **trot** (`multi_gait=False`, phase offsets fixed to LF/RH in phase, RF/LH half-cycle out); per-gait vx envelope up to 4 m/s.
- Cobblestone env: now inherits from flat and restores the height scanner + dependent reward params via a shared override helper.
- Bump flat PPO `max_iterations` 10000 → 20000.
- Shorten exp-neg sigma anneal (48000 → 12000 steps); add new `gait_velocity_curriculum` term.
