# Auto-Train: Isaac-WTW-Flat-Ayg-v0
Started: 2026-03-21  |  Level: 2  |  Device: RTX 3000 Ada 8GB
Constraint: 2048 envs, production ≥ 10000 iterations

## Pre-Training Reward Analysis

### Robot Physical Parameters
- Init height: 0.36m, ~10kg mass, leg length ~0.3m
- Default joints: HAA=0, HFE=0, KFE_front=-0.2, KFE_hind=-0.25
- Actuator: effort_limit=50Nm, velocity_limit=18.849 rad/s, Kp=50, Kd=3
- Control dt = 0.02s (decimation=4, sim.dt=0.005)

### Current Config (AygRoughWTWEnvCfg → AygFlatWTWEnvCfg)

**Reward formula (hybrid):** `r_total = r_additive + r_gait * exp(c_aux * r_footswing)`
- c_aux anneals from 0.02 → 0.3 over 80000 steps
- Only "gait" and "footswing_height" go through exp; everything else is additive

### Per-Term Analysis (dt=0.02)

| Term | Weight | Type | Max per-step | Issue? |
|------|--------|------|-------------|--------|
| track_lin_vel_xy_exp | +1.0 | ADD | +0.020 | OK |
| track_ang_vel_z_exp | +1.0 | ADD | +0.020 | OK |
| lin_vel_z_l2 | -0.02 | ADD | -0.0004 | Negligible |
| ang_vel_xy_l2 | -0.02 | ADD | -0.002 | Weak |
| **flat_orientation_l2** | **-1.0** | ADD | **-0.020** | **CONFLICTS with body_pitch_cmd! Penalizes ALL tilt including commanded pitch.** |
| joint_deviation_l1 | -0.05 | ADD | -0.020 | OK |
| joint_vel_l2 | -2e-5 | ADD | ~0 | Negligible |
| joint_acc_l2 | -5e-9 | ADD | ~0 | Negligible |
| joint_torques_l2 | -2e-5 | ADD | -0.012 | Moderate |
| **base_height_l2** | **-0.2** | ADD | **-0.0001** | **CRITICALLY WEAK! Crawling at 0.15m penalty is only 0.01% of tracking reward. Robot has zero incentive to maintain height.** |
| feet_slip | -8e-4 | ADD | ~0 | Negligible |
| action_rate_l2 | -0.01 | ADD | moderate | OK |
| action_smoothness_l2 | -0.01 | ADD | moderate | OK |
| undesired_contacts | -1.0 | ADD | -0.16 | OK (also terminated) |
| gait | +0.5 | GAIT_TASK | +0.08 | **Too weak** — co-worker uses 16.0 |
| footswing_height | -2.0 | GAIT_AUX | varies | OK |
| body_pitch_tracking | -0.1 | ADD | -0.0007 | **Too weak** to override flat_orientation |
| stand_when_zero_command | -0.01 | ADD | ~0 | Negligible |
| stand_still_when_zero_command | -0.01 | ADD | ~0 | Negligible |

### Critical Issues Found

1. **base_height_l2 is ~300x too weak** (w=-0.2): Even extreme crawling produces penalty < 0.01% of tracking reward. Co-worker uses -160 through exp. Need ≥ -30 additive to create meaningful gradient.

2. **flat_orientation_l2 conflicts with body_pitch command**: The gait command specifies target pitch (-0.3 to 0.3 rad), but flat_orientation_l2 penalizes ALL tilt equally. When pitch_cmd ≠ 0, these two signals fight each other. Need to replace with separate pitch tracking + roll penalty.

3. **gait reward too weak** (w=0.5): Co-worker uses 16.0. At 0.5, the gait signal is only 4x the tracking reward. The robot may ignore gait structure entirely and just optimize velocity tracking.

4. **body_pitch_tracking too weak** (w=-0.1): 10x weaker than the conflicting flat_orientation. The robot will prioritize staying flat over tracking the commanded pitch.

5. **Action scale uniform 0.25**: HAA joints control lateral leg spread. Smaller HAA scale (0.125) prevents spider stance while allowing full HFE/KFE range for gait diversity. Co-worker uses HAA=0.125.

6. **Episode length too short** (10s): Co-worker uses 20s. Longer episodes let the robot learn sustained gaits and provide more data per episode for gait tracking rewards.

### Plan for Iteration 1 (batch of obvious fixes)

Source edits (Level 2):
1. Add `body_roll_l2` reward function (penalize roll, allow commanded pitch)
2. Per-joint action scale: HAA=0.125, HFE/KFE=0.25
3. base_height_l2: -0.2 → -30.0
4. flat_orientation_l2: -1.0 → 0.0 (disabled)
5. body_pitch_tracking: -0.1 → -3.0
6. body_roll_l2: -3.0
7. gait: 0.5 → 4.0
8. undesired_contacts: -1.0 → -2.0
9. episode_length_s: 10.0 → 20.0
10. Reduce add_base_mass: (1.5, 4.0) → (-1.0, 2.0) for fairer starting point

---

## Iteration 1 — Baseline with critical fixes

**Hypothesis:** Fixing weak base_height penalty (-0.2→-30), replacing conflicting flat_orientation with separate pitch+roll tracking, increasing gait weight (0.5→4.0), and per-joint action scale (HAA=0.125) will produce an upright walking baseline.
**Variable under test:** Batch of obvious fixes from pre-training analysis (Level 2 source edits).
**Changes:**
- Source edits: body_roll_l2 added, base_height_l2 -0.2→-30, flat_orientation_l2 disabled, body_pitch -0.1→-3, body_roll -3, gait 0.5→4, undesired_contacts -1→-2, HAA scale 0.125, episode 10→20s, add_base_mass (1.5,4)→(-1,2)
- Override: agent.max_iterations=1500
**Result:** success (completed 1500 iters)
**Key metrics:**
- Mean reward: 183.9 (last 100), peak 212.9
- Episode length: 928/1000 steps (93% survival)
- Terminations: 88% timeout, 8.5% bad_orient, 2.5% shank/thigh
- Velocity tracking: error_xy=1.52, error_yaw=4.27 (POOR)
- Gait reward: 10.12 (good contact schedule)
- Learning rate COLLAPSED to 1.5e-5 (from 1e-3)
- Policy noise INCREASED 0.5→0.8 (not converging)
**Visual assessment:** Robots standing upright on all 4 feet. No crawling, no knee-walking, no spider stance. Some robots falling (~8.5%). Legs look reasonable, not spread too wide. Gait visually appears functional but rough.
**Conclusion:** Baseline gait is solid! Robot walks upright. But velocity tracking is poor because learning rate collapsed due to desired_kl=0.008 being too tight. Policy stopped learning early. Need to relax KL constraint.
**Log dir:** `logs/rsl_rl/ayg_wtw_flat/2026-03-21_22-42-39/`

---

## Iteration 2 — Relax KL constraint (resume from iter 1)

**Hypothesis:** Learning rate collapsed in iter 1 because desired_kl=0.008 is too tight. Relaxing to 0.01 will maintain LR and improve velocity tracking.
**Variable under test:** agent.algorithm.desired_kl (0.008 → 0.01)
**Changes:** Override: desired_kl=0.01, resumed from iter 1 checkpoint, 3000 total (1500 new)
**Result:** success (completed 4498 total iterations)
**Key metrics (vs iter 1):**
- Mean reward: 183.9 → 186.6 (slight improvement)
- Episode length: 928 → 945 (better survival)
- Timeouts: 88.2% → 91.3%, bad_orient: 8.5% → 5.0%
- error_vel_xy: 1.52 → 1.49 (barely improved)
- **error_vel_yaw: 4.27 → 1.75 (MAJOR improvement)**
- track_ang_vel_z: 0.118 → 0.263 (more than doubled!)
- Gait: 10.1 → 11.0 (improved)
- Learning rate: still low at 5e-5 (better than 1.5e-5)
- **Policy noise: 0.80 → 1.06 (still increasing — problematic)**
**Visual assessment:** Robots walking upright with active dynamic gaits. Clear leg lift, proper stride patterns. No crawling, no knee-walking. Better than iter 1 visually. ~8% robots in fallen/reset states.
**Conclusion:** Relaxing KL helped yaw tracking dramatically. But adaptive LR still collapses over long runs, causing noise drift. Need fixed LR for production run.
**Log dir:** `logs/rsl_rl/ayg_wtw_flat/2026-03-21_23-04-26/`

---

## Iteration 3 — Fixed LR experiment (fresh start)

**Hypothesis:** Fixed LR at 3e-4 will prevent adaptive LR collapse and enable consistent learning.
**Variable under test:** schedule (adaptive → fixed), learning_rate (1e-3 → 3e-4)
**Changes:** Override: schedule=fixed, lr=3e-4, fresh start 2000 iters
**Result:** success but WORSE than iter 2
**Key metrics (vs iter 2):**
- Mean reward: 186.6 → 176.1 (worse)
- error_vel_yaw: 1.75 → 4.22 (REGRESSION — lost yaw tracking improvement)
- Timeout: 91.3% → 85.7% (worse survival)
- base_contact: 0.9% → 3.6% (more crashes)
- Policy noise: 1.06 → 1.24 (EVEN WORSE — noise grew faster with fixed LR)
- Action smoothness: -0.58 → -0.80 (more jitter)
**Visual assessment:** Still upright walking but messier than iter 2. More robots in awkward/fallen states.
**Conclusion:** Fixed LR was a bad idea. Adaptive schedule is better for this problem. The yaw improvement in iter 2 came from the resumed checkpoint + adaptive schedule. The noise growth problem must be solved by reducing entropy_coef, not by changing the LR schedule.
**Log dir:** `logs/rsl_rl/ayg_wtw_flat/2026-03-21_23-40-25/`

---

## Production Run — Resume from iter 2 with reduced entropy

**Hypothesis:** Reducing entropy_coef from 0.005 to 0.001 will control noise growth while maintaining good velocity tracking. Combined with iter 2's adaptive schedule (desired_kl=0.01).
**Variable under test:** entropy_coef (0.005 → 0.001) for long production run
**Config:** Resume from iter 2 (model_4498), 10500 new iterations, adaptive schedule, desired_kl=0.01, entropy_coef=0.001
**Result:** success (completed 14997 total iterations, ~10500 new)
**Key metrics (final):**
- Mean reward: 204.5 (last 100), peak 225.2 — best result
- Episode length: 964.7/1000 (96.5% survival)
- Timeouts: 95.5%, bad_orient: 3.0%, shank_thigh: 0.3%
- error_vel_xy: 1.50 (stable), error_vel_yaw: 4.47 (regressed from iter 2's 1.75)
- Gait reward: 12.42 (best yet, +12.5% from iter 2)
- track_lin_vel: 0.257, track_ang_vel: 0.131
- **Noise std: 0.731 (DOWN from 1.06) — entropy reduction worked!**
- sigma_exp_neg reached maximum 0.3
**Visual assessment:** Robots walking upright with refined dynamic gaits. Very few falls. Leg extension and stride patterns look natural. No crawling, no knee-walking, no spider stance. Best visual quality of all iterations.
**Conclusion:** Production run successful. Noise growth fixed by entropy_coef=0.001. Gait tracking significantly improved. Survival near-perfect. Yaw tracking regressed — the policy converged to prioritize gait compliance over turning. This is a known trade-off in WTW; Phase 2 with observation history may help.
**Winning config baked into source:** entropy_coef=0.001, desired_kl=0.01, max_iterations=15000
**Log dir:** `logs/rsl_rl/ayg_wtw_flat/2026-03-22_00-05-09/`
**Checkpoint:** `model_14997.pt`

---

## CRITICAL BUG FOUND: Hip-walking exploit

**User caught it from close-up video:** Robots were walking on their HIPS, not their feet. The overhead play frames made it look like proper walking, but close inspection reveals the robot found a local optimum where it shuffles on upper leg segments.

**Root cause:** Hip bodies (`.*_Hip`) were excluded from BOTH:
- `base_contact` termination (only checked `"Base"`, not hips)
- `undesired_contact_names` (only `[".*_Shank", ".*_Thigh"]`, not hips)

The `termination_contact_names` in wtw_params.py correctly included `["Base", "Camera", ".*_Hip"]` but was NEVER wired into the actual termination config. All previous iterations had this bug — the robot was exploiting hip contact with zero penalty.

**Lesson learned:** Default play video camera angles cannot reliably detect hip-walking vs foot-walking. Must rely on correct terminations — and verify they cover ALL non-foot body parts.

**Fix applied (Level 2 source edits):**
1. `base_contact` termination now uses `Params.termination_contact_names` = `["Base", "Camera", ".*_Hip"]`
2. `undesired_contact_names` now includes `".*_Hip"` — penalizes hip contact in reward
3. Previous checkpoint is INVALID — must train from scratch

---

## Iteration 4 — Fresh start with hip termination fix

**Hypothesis:** Adding hip contact termination + penalty will force the robot to walk on feet only. Previous reward/PPO config is otherwise solid (entropy_coef=0.001, desired_kl=0.01).
**Variable under test:** Hip contact termination + reward penalty
**Changes:** Source edits: base_contact uses termination_contact_names (includes hips), undesired_contact_names includes hips. Fresh start 2000 iters.

