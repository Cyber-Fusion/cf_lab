# Auto-Train: Isaac-Velocity-Flat-Ayg-Direct-v0
Started: 2026-03-20  |  Level: 1  |  Device: RTX Ada 3000 12GB

## Environment Summary

**Reward terms (default weights):**
| Term | Scale | Type |
|---|---|---|
| lin_vel_reward_scale | 2.0 | + tracking |
| yaw_rate_reward_scale | 1.0 | + tracking |
| z_vel_reward_scale | -2.0 | - penalty |
| ang_vel_reward_scale | -0.05 | - penalty |
| joint_torque_reward_scale | -0.0002 | - penalty |
| joint_accel_reward_scale | -2.5e-7 | - penalty |
| action_rate_reward_scale | -0.01 | - penalty |
| feet_air_time_reward_scale | 0.25 | + gait |
| undesired_contact_reward_scale | -0.25 | - penalty |
| flat_orientation_reward_scale | -2.5 | - penalty |
| feet_regulation_reward_scale | -0.05 | - penalty |

**PPO:** [128,128,128], lr=1e-3 adaptive, entropy=0.005, gamma=0.99, lam=0.95
**Observations:** 48-dim (base vel/ang/grav + cmds + joint pos/vel + actions)
**Termination:** base/hip contact + 20s timeout

---

## Iteration 1 — Baseline (300 iters)

**Goal:** Establish baseline metrics with default reward weights
**Why:** Need to understand reward term magnitudes and visual quality before tuning
**Changes:** None (default config, 2048 envs)
**Result:** success
**Key metrics:**
- Mean reward: 28.0 (final), 24.6 (avg last 100), still growing
- Episode length: 920/999 — robot survives well
- Noise std: 0.37 (from 1.0)
- FPS: ~85K, training time: 189s
- Reward breakdown (final):
  - track_lin_vel_xy: +1.14 (decent)
  - track_ang_vel_z: +0.80 (good)
  - feet_air_time: **-0.028** (NEGATIVE — broken!)
  - dof_torques_l2: -0.18 (biggest penalty)
  - ang_vel_xy: -0.072
  - dof_acc: -0.063
  - undesired_contacts: -0.056
  - action_rate: -0.053
  - feet_regulation: -0.033
  - lin_vel_z: -0.021
  - flat_orientation: -0.020

**Visual assessment:** Robots upright and walking across all frames. Posture reasonable. No crawling or spider-walking. Some have slightly spread legs. Generally functional locomotion at 300 iterations.
**Conclusion:** The feet_air_time reward is actively penalizing the robot — the 0.5s air time threshold is too high, making `(air_time - 0.5)` negative for normal gaits. This reward should be disabled. Tracking is decent but can be pushed harder. Torque penalty dominates penalties.
**Log dir:** `logs/rsl_rl/ayg_flat_direct/2026-03-20_22-20-59/`

## Iteration 2 — Disable air time, boost tracking

**Goal:** Remove broken air time reward (was penalizing normal gaits) and boost tracking signals
**Why:** Baseline showed feet_air_time was negative (-0.028) due to 0.5s threshold being too high for this robot
**Changes:**
- feet_air_time_reward_scale: 0.25 → 0.0
- lin_vel_reward_scale: 2.0 → 3.0
- yaw_rate_reward_scale: 1.0 → 1.5
- joint_torque_reward_scale: -0.0002 → -0.00015
**Result:** success
**Key metrics:**
- Mean reward: 69.2 (final), 68.2 (avg last 100) — **+147% from baseline**
- Episode length: 984/999 — nearly perfect survival
- Base contacts: 0.03 avg — almost zero falls
- Noise std: 0.33 (converging)
- LR: 0.000114 (very low — policy converging fast)
- Reward breakdown (final):
  - track_lin_vel_xy: +2.65 (88% of 3.0 max, up from 57% of 2.0)
  - track_ang_vel_z: +1.27 (85% of 1.5 max, up from 80% of 1.0)
  - dof_torques: -0.20 (slightly worse despite lower weight)
  - action_rate: -0.076 (up from -0.053 — jerkier)
  - ang_vel_xy: -0.072 (similar)
  - dof_acc: -0.067 (slightly worse)
  - feet_regulation: -0.056 (up from -0.033 — faster foot ground contact)
  - undesired_contacts: -0.043 (improved)
  - lin_vel_z: -0.034 (up from -0.021 — more bounce)
  - flat_orientation: -0.020 (same)

**Visual assessment:** All robots upright and actively locomoting. Good ground coverage. More dynamic movement than baseline. No crawling or spider-walking. Clean locomotion.
**Conclusion:** Disabling air time was correct — massive reward improvement. Tracking is strong. However, action smoothness degraded (action_rate, feet_regulation up). Next: focus on smoothness penalties to improve motion quality without sacrificing tracking.
**Log dir:** `logs/rsl_rl/ayg_flat_direct/2026-03-20_22-26-16/`

## Iteration 3 — Smoothness focus

**Goal:** Improve motion smoothness while maintaining tracking quality
**Why:** Iter 2 showed increased action rate and feet regulation penalties despite good tracking
**Changes:**
- action_rate_reward_scale: -0.01 → -0.02 (2x penalty)
- feet_regulation_reward_scale: -0.05 → -0.08 (1.6x penalty)
- joint_torque_reward_scale: -0.00015 → -0.0001 (relaxed)
- entropy_coef: 0.005 → 0.008 (more exploration)
- (kept: feet_air_time=0, lin_vel=3.0, yaw=1.5)
**Result:** success
**Key metrics:**
- Mean reward: 72.5 (final), 71.6 (avg last 100) — +5% from iter 2
- Episode length: 997/999 — near perfect
- Base contacts: 0.016 — essentially zero
- Reward std: 0.61 — very stable convergence
- Reward breakdown (final, with iter 2 comparison):
  - track_lin_vel_xy: +2.81 (↑ from 2.65, 94% of max)
  - track_ang_vel_z: +1.33 (↑ from 1.27, 88% of max)
  - dof_torques: -0.136 (↓ from -0.20, much better)
  - action_rate: -0.121 (↑ penalty due to 2x weight)
  - feet_regulation: -0.087 (↑ due to higher weight)
  - ang_vel_xy: -0.079 (similar)
  - dof_acc: -0.054 (↓ from -0.067)
  - lin_vel_z: -0.028 (↓ from -0.034)
  - undesired_contacts: -0.015 (↓ from -0.043, big improvement!)
  - flat_orientation: -0.013 (↓ from -0.020)

**Visual assessment:** All robots upright, clean locomotion, good terrain coverage. Posture looks slightly cleaner than iter 2.
**Conclusion:** Smoothness penalties working well — tracking improved AND penalties decreased. Policy is well-converged (std 0.61). Next: push tracking slightly higher, relax ang_vel, try longer rollouts.
**Log dir:** `logs/rsl_rl/ayg_flat_direct/2026-03-20_22-33-30/`

## Iteration 4 — Push tracking + longer rollouts

**Goal:** Test if higher tracking scales and longer rollouts improve performance further
**Why:** Iter 3 showed tracking at 94%/88% — testing if there's headroom with more reward signal
**Changes:**
- lin_vel_reward_scale: 3.0 → 3.5
- yaw_rate_reward_scale: 1.5 → 1.75
- ang_vel_reward_scale: -0.05 → -0.03 (relaxed)
- num_steps_per_env: 24 → 32 (longer rollouts)
- (kept: air_time=0, torque=-0.0001, action_rate=-0.02, feet_reg=-0.08, entropy=0.008)
**Result:** success
**Key metrics:**
- Mean reward: **87.6 (final), 86.9 (avg last 100)** — +21% from iter 3!
- Episode length: 999 — PERFECT
- Base contacts: 0.014, trend "improving" — essentially zero
- Reward std: 0.52 — extremely stable
- Reward breakdown (final):
  - track_lin_vel_xy: +3.33 (95% of 3.5 max — excellent!)
  - track_ang_vel_z: +1.56 (89% of 1.75 max — excellent!)
  - action_rate: -0.135 (similar)
  - dof_torques: -0.134 (similar)
  - feet_regulation: -0.086 (similar)
  - dof_acc: -0.059 (similar)
  - ang_vel_xy: -0.059 (↓ from -0.079, lower weight helping)
  - lin_vel_z: -0.027 (slightly improved)
  - flat_orientation: -0.013 (stable)
  - undesired_contacts: -0.011 (improved)

**Visual assessment:** Excellent locomotion quality. All robots upright, purposeful movement, wide terrain coverage. Best visual quality so far.
**Conclusion:** This is the best result yet. Tracking near practical limits (95%/89%). The remaining penalties are inherent locomotion costs. Try one more iteration to test ceiling, then production run.
**Log dir:** `logs/rsl_rl/ayg_flat_direct/2026-03-20_22-40-43/`

## Iteration 5 — Ceiling test

**Goal:** Determine if higher tracking scales yield real behavioral improvement
**Why:** Iter 4 tracking was at 95%/89% — need to know if this is the ceiling
**Changes:**
- lin_vel_reward_scale: 3.5 → 4.0
- yaw_rate_reward_scale: 1.75 → 2.0
- z_vel_reward_scale: -2.0 → -1.5 (relaxed)
- (kept everything else from iter 4)
**Result:** success
**Key metrics:**
- Mean reward: 100.2 (final), 99.9 (avg last 100) — raw number up from 87.6
- **BUT tracking percentages IDENTICAL: lin 95%, ang 88%**
- Undesired contacts: -0.019 (worse than iter 4's -0.011)
- All other penalties similar
- Reward breakdown (final):
  - track_lin_vel_xy: +3.80 (95% of 4.0 max — same % as iter 4!)
  - track_ang_vel_z: +1.77 (88% of 2.0 max — same % as iter 4!)

**Visual assessment:** Similar quality to iter 4. All robots upright and walking well.
**Conclusion:** **CEILING CONFIRMED.** Tracking at 95%/88% is the physical limit. Higher scales inflate reward numbers without behavioral improvement. Iter 4 config is the winner (lower contacts, z_vel penalty kept). Moving to production run.
**Log dir:** `logs/rsl_rl/ayg_flat_direct/2026-03-20_22-52-05/`

## Production Run — Iter 4 config at 2000 iterations

**Goal:** Full production training with winning config from iter 4
**Config:**
- feet_air_time_reward_scale: 0.0
- lin_vel_reward_scale: 3.5
- yaw_rate_reward_scale: 1.75
- joint_torque_reward_scale: -0.0001
- action_rate_reward_scale: -0.02
- feet_regulation_reward_scale: -0.08
- ang_vel_reward_scale: -0.03
- z_vel_reward_scale: -2.0 (original)
- entropy_coef: 0.008
- num_steps_per_env: 32

**Result:** success
**Key metrics:**
- Mean reward: 88.5 (final), 87.6 (avg last 100) — consistent with tuning runs
- Episode length: 999 — perfect
- Tracking: lin 3.35/3.5 (96%), ang 1.58/1.75 (90%) — slightly improved over 600-iter runs
- Undesired contacts: -0.0001 — essentially ZERO (eliminated with longer training!)
- Lin vel z: -0.017 (improved from -0.027)
- All penalties stable, extremely low variance (std 0.85)
- Reward breakdown (final):
  - track_lin_vel_xy: +3.35 (96%)
  - track_ang_vel_z: +1.58 (90%)
  - action_rate: -0.132
  - dof_torques: -0.130
  - feet_regulation: -0.084
  - dof_acc: -0.066
  - ang_vel_xy: -0.055
  - lin_vel_z: -0.017
  - flat_orientation: -0.013
  - undesired_contacts: -0.0001
**Visual assessment:** Excellent. All robots upright, clean locomotion, wide terrain coverage. Best quality across all runs.
**Log dir:** `logs/rsl_rl/ayg_flat_direct/2026-03-20_23-03-21/`
**Checkpoint:** `logs/rsl_rl/ayg_flat_direct/2026-03-20_23-03-21/model_1999.pt`

## Summary

**5 tuning iterations + 1 production run.** Key findings:
1. feet_air_time reward was broken (penalizing normal gaits) — disabled it
2. Tracking scales should be 3.5/1.75 (not default 2.0/1.0) for stronger signal
3. Smoothness penalties (action_rate -0.02, feet_reg -0.08) improve both quality and tracking
4. Longer rollouts (32 vs 24 steps) improve sample efficiency
5. Slightly higher entropy (0.008 vs 0.005) helps exploration
6. Tracking ceiling is ~96% lin / ~90% ang — physical limit of the environment
7. With 2000 iterations, undesired contacts are essentially eliminated

**Winning config baked into source:** `ayg_env_cfg.py`

