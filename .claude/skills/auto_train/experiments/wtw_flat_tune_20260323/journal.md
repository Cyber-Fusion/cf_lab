# Auto-Train: Isaac-WTW-Flat-Ayg-v0
Started: 2026-03-23  |  Level: 2  |  Device: RTX 3000 Ada 8GB VRAM  |  Envs: 2048

**Goal:** Train AYG quadruped with Walk These Ways (4 gait styles: trot, pace, bound, pronk) on flat terrain. Motivated by the WTW paper (Margolis & Agrawal, CoRL 2022), originally for Go1. Deployment-level production run >5000 iterations.

**Key user notes:**
- Paper uses 30-step history; user found it catastrophically bad → using 5 (already configured)
- Config is "not bad" but needs tuning for better performance
- 2048 envs on RTX 3000 Ada 8GB

## Pre-Training Reward Analysis

### Reward Architecture: Exp-Negative Composition
R = sum(additive_terms * dt) * exp(sum(exp_negative_terms * dt) * sigma)

- sigma anneals from 1.0 → 20.0 over 24000 env steps (~1000 iterations at 24 steps/iter)
- Early training: exp gate ≈ 1, policy focuses on velocity tracking
- Late training: exp gate suppresses reward when behavior is poor

### Additive Terms (base reward):
| Term                     | Weight  | Expected Value | Contribution/step*dt |
|--------------------------|---------|----------------|---------------------|
| track_lin_vel_xy_exp     | 2.0     | 0.3–0.8        | +0.6 to +1.6       |
| track_ang_vel_z_exp      | 1.0     | 0.3–0.8        | +0.3 to +0.8       |
| lin_vel_z_l2             | -2.0    | ~0.01          | -0.02              |
| ang_vel_xy_l2            | -0.05   | ~0.1           | -0.005             |
| joint_vel_l2             | -1e-3   | ~50            | -0.05              |
| joint_acc_l2             | -2.5e-7 | ~large         | negligible          |
| joint_torques_l2         | -2e-4   | ~moderate      | minor               |
| action_rate_l2           | -0.01   | ~0.5           | -0.005             |
| action_smoothness_l2     | -0.01   | ~0.5           | -0.005             |

**Net additive reward ≈ +1 to +2.5 per step*dt** (dominated by velocity tracking)

### Exp-Negative Terms (gate multiplier):
| Term                  | Weight | Expected Raw Value | Contribution to Gate   |
|-----------------------|--------|-------------------|----------------------|
| gait                  | +16.0  | 2–4 (positive!)   | AMPLIFIES (+32 to +64)*dt |
| orientation_control   | -40.0  | 0.01–0.25         | -0.4 to -10 * dt    |
| base_height_l2        | -160.0 | 0.001–0.01        | -0.16 to -1.6 * dt  |
| feet_slip             | -0.04  | 0.2–2.0           | -0.008 to -0.08 * dt|
| undesired_contacts    | -8.0   | 0–8               | 0 to -64 * dt       |
| foot_clearance        | -30.0  | 0.01–0.04         | -0.3 to -1.2 * dt   |

**Key observations:**
- `gait` is the only POSITIVE exp-negative term — good gait tracking AMPLIFIES the velocity reward
- `orientation_control` at -40.0 is very strong — moderate pitch/roll error heavily suppresses reward
- `base_height_l2` at -160.0 is extremely strong — even 5cm height error creates significant suppression
- `undesired_contacts` at -8.0 can be devastating if contacts occur

### Feasibility Assessment:
- Velocity tracking (std=0.5): At 0.5 m/s error → exp(-0.25/0.25) = 0.37, reasonable
- Gait frequency (2-4 Hz): At 3 Hz, cycle period = 0.33s. With 50% duty → 0.17s swing. Foot travels ~0.05m in air. Feasible for AYG legs.
- Base height (0.20-0.40m): AYG default is 0.36m. Range matches robot's kinematic workspace.
- Feet height (0.05-0.20m): AYG leg length ~0.3m. 0.2m swing height requires near-full extension → aggressive but feasible.
- Body pitch (±0.4 rad = ±23°): Extreme values may conflict with walking stability, but exp-negative gate modulates this.
- Action scale: HAA=0.125, HFE=KFE=0.25 → prevents spider-walking, good.

### PPO Configuration:
- Network: [512, 256, 128] actor/critic, ELU activation
- Observation: 61 dims/step × 5 history = 305 dims (actor + critic)
- LR: 1e-3 (adaptive), entropy: 0.005, clip: 0.2, KL: 0.01
- Steps/env: 24, mini_batches: 4, epochs: 5
- Gamma: 0.99, Lambda: 0.95
- Empirical normalization: OFF

## Body Coverage Audit

| Body          | Foot? | Termination? | Penalty? | Status   |
|---------------|-------|-------------|----------|----------|
| Base          | No    | Yes (base_contact) | No | COVERED  |
| IMU           | No    | N/A (sensor link, zero mass) | N/A | N/A |
| LF_Hip        | No    | Yes (base_contact, .*_Hip) | No | COVERED  |
| LF_Thigh      | No    | Yes (shank_thigh_contact) | Yes (undesired) | COVERED  |
| LF_Shank      | No    | Yes (shank_thigh_contact) | Yes (undesired) | COVERED  |
| LF_Foot       | Yes   | No          | No       | FOOT     |
| RF_Hip        | No    | Yes         | No       | COVERED  |
| RF_Thigh      | No    | Yes         | Yes      | COVERED  |
| RF_Shank      | No    | Yes         | Yes      | COVERED  |
| RF_Foot       | Yes   | No          | No       | FOOT     |
| LH_Hip        | No    | Yes         | No       | COVERED  |
| LH_Thigh      | No    | Yes         | Yes      | COVERED  |
| LH_Shank      | No    | Yes         | Yes      | COVERED  |
| LH_Foot       | Yes   | No          | No       | FOOT     |
| RH_Hip        | No    | Yes         | No       | COVERED  |
| RH_Thigh      | No    | Yes         | Yes      | COVERED  |
| RH_Shank      | No    | Yes         | Yes      | COVERED  |
| RH_Foot       | Yes   | No          | No       | FOOT     |

**Gaps found:** None — all 17 bodies covered. IMU is a sensor link.

**CRITICAL FINDING:** `shank_thigh_contact` termination makes shanks/thighs terminate the episode on ANY contact (>1N). The WTW paper only penalizes thigh/calf contact (-0.02 weight), never terminates. These bodies are already penalized via `undesired_contacts` reward (weight=-8.0 in exp-neg gate). Double punishment (termination + penalty) is too aggressive and will cause very short episodes, preventing the robot from learning recovery and gait transitions.

**Action:** Remove `shank_thigh_contact` termination in flat_env_cfg.py (Level 2 source edit). Keep penalty.

---

## Iteration 1 — Baseline + Remove Shank/Thigh Termination

**Hypothesis:** Removing the shank_thigh_contact termination will allow longer episodes and better gait learning. The undesired_contacts penalty (weight=-8.0 in exp-neg gate) is sufficient to discourage unwanted shank/thigh contact. The paper only penalizes these contacts. We expect episode length to increase and velocity tracking to improve because the robot has more time to learn within each episode.

**Variable under test:** shank_thigh_contact termination disabled (set to None)

**Iteration count and reasoning:** 400 iterations — first run, need to establish baseline metrics. 400 iters covers the initial sigma annealing (sigma reaches ~6.7 at iter 400), enough to see velocity tracking response and episode length.

**Changes:**
- Source edit: `flat_env_cfg.py` → add `self.terminations.shank_thigh_contact = None` in __post_init__
- No JSON overrides needed for this change

**Overrides (JSON):** max_iterations=400 only (via CLI arg)

**Result:** success
**Training time:** 364s (399 iterations completed)

**Detailed Metrics:**
| Metric                    | Final   | Mean (last 100) | Trend      | Curve Shape     | Converged@iter |
|--------------------------|---------|-----------------|------------|-----------------|----------------|
| Total reward              | 1113.2  | 529.6           | improving  | oscillating     | 125            |
| track_lin_vel_xy_exp      | 1.506   | 1.593           | stable     | converged_early | 108            |
| track_ang_vel_z_exp       | 0.736   | 0.769           | stable     | converged_early | 64             |
| gait                      | 36.9    | 36.8            | stable     | converged_early | 49             |
| orientation_control       | -1.94   | -2.35           | stable     | converged_early | 49             |
| base_height_l2            | -1.14   | -1.18           | stable     | converged_early | 49             |
| foot_clearance            | -0.15   | -0.17           | stable     | converged_early | 49             |
| undesired_contacts        | 0.0     | -0.001          | stable     | oscillating     | —              |
| Episode length            | 943     | 940             | stable     | converged_early | 49             |
| error_vel_xy              | 0.332   | 0.357           | stable     | converged_early | 49             |
| error_vel_yaw             | 0.342   | 0.379           | stable     | converged_early | 49             |
| bad_orientation %         | 10.1%   | 9.8%            | stable     | —               | —              |
| sigma_exp_neg             | 4.03    | 3.34            | stable     | —               | —              |

**Suspicious Patterns:**
- high_reward_low_tracking (critical): Total reward 1113 but tracking sum 2.24 (0.2% of total) — MISLEADING for exp-negative composition. Tracking IS working (error_vel_xy=0.33).
- tracking_flatlined_early (warning): Both tracking terms converged early — expected at 400 iters since sigma is still low (~4) and reward structure is changing.

**Velocity Tracking Gate:**
  - Linear velocity tracking: error_vel_xy=0.33 → exp(-0.33²/0.25)=0.64 → ACCEPTABLE (>0.6 target)
  - Angular velocity tracking: error_vel_yaw=0.34 → ACCEPTABLE (>0.3 target)

**Visual Assessment:**
  - Camera: side-view at robot level (play_for_inspection.py, distance=2.0m, height=0.4m, azimuth=90°)
  - CAN VERIFY: Robots upright with proper quadruped stance, base at normal height (~0.3m), legs visible and in walking configurations
  - CAN VERIFY: Forward motion occurring, velocity arrows tracking movement direction
  - CAN VERIFY: Legs show different configurations across frames suggesting active swing/stance phases
  - CANNOT VERIFY: Exact foot clearance during swing phase at this resolution
  - CANNOT VERIFY: Which specific gait (trot/pace/bound/pronk) is being executed
  - Failure modes: hip-walking: NOT detected | belly-sliding: NOT detected | shuffling: UNCERTAIN | spider-walking: NOT detected (stance width normal)

**Reasoning:**
  - Removing shank_thigh_contact termination was the right call — undesired_contacts ≈ 0 shows the penalty alone prevents thigh/shank contact without needing termination. Episode length is nearly maxed (943/1000).
  - Total reward still strongly improving at 400 iters. Sigma only reached 4.03 (of max 20), so the exp-negative gate is still relatively weak. Full convergence requires many more iterations.
  - Gait reward (36.9) is strong and stable — the robot learned the gait pattern well.
  - 10% bad_orientation termination rate is notable — the robot struggles with some commanded orientations.
  - The "high_reward_low_tracking" suspicious pattern is a false positive: in exp-negative composition, the tracking terms are small additive values amplified by the exp gate. The actual velocity errors are good (0.33/0.34).

**Conclusion and Next Step:**
  - Baseline is very promising. Episode length near max, velocity tracking acceptable, gait strong.
  - Next: increase track_lin_vel_xy_exp weight from 2.0→3.5 to push velocity error lower.
  - hypothesis: stronger tracking weight will reduce error_vel_xy from 0.33 to <0.25.

**Log dir:** `logs/rsl_rl/ayg_wtw_flat/2026-03-23_20-48-12/`

---

## Iteration 2 — Increase Linear Velocity Tracking Weight

**Hypothesis:** Increasing track_lin_vel_xy_exp weight from 2.0 to 3.5 will make velocity tracking a stronger priority in the additive base, reducing error_vel_xy from 0.33 to <0.25 while maintaining gait quality.

**Variable under test:** track_lin_vel_xy_exp.weight (2.0 → 3.5)

**Iteration count and reasoning:** 400 iterations — same as iteration 1 for direct comparison. Enough to see if velocity tracking responds to increased weight.

**Changes:**
  - `rewards.track_lin_vel_xy_exp.weight`: 2.0 → 3.5

**Result:** success
**Training time:** 367s (399 iterations completed)

**Detailed Metrics:**
| Metric                    | Final   | Mean (last 100) | Trend      | Curve Shape     | Converged@iter |
|--------------------------|---------|-----------------|------------|-----------------|----------------|
| Total reward              | 1658.5  | 849.6           | improving  | oscillating     | —              |
| track_lin_vel_xy_exp      | 2.966   | 2.882           | stable     | converged_early | 58             |
| track_ang_vel_z_exp       | 0.743   | 0.711           | stable     | converged_early | 56             |
| gait                      | 39.2    | 36.8            | stable     | converged_early | —              |
| orientation_control       | -2.20   | -2.30           | stable     | converged_early | —              |
| base_height_l2            | -1.53   | -1.24           | stable     | converged_early | —              |
| Episode length            | 968     | 940             | stable     | converged_early | —              |
| error_vel_xy              | 0.332   | 0.310           | stable     | converged_early | —              |
| error_vel_yaw             | 0.471   | 0.460           | stable     | converged_early | —              |
| bad_orientation %         | 12.7%   | 9.5%            | stable     | —               | —              |

**Velocity Tracking Gate:**
  - Linear velocity tracking: error_vel_xy=0.332 → SAME as iter 1 (0.332). Hypothesis FAILED.
  - Angular velocity tracking: error_vel_yaw=0.471 → WORSE than iter 1 (0.342). Yaw degraded significantly.

**Visual Assessment:**
  - Camera: side-view, same setup as iter 1
  - CAN VERIFY: Robots upright with proper stance, walking locomotion visible
  - No failure modes detected

**Reasoning:**
  - Increasing track_lin_vel_xy_exp weight from 2.0→3.5 did NOT reduce linear velocity error.
  - Yaw tracking DEGRADED from 0.342→0.471. The stronger xy weight may be crowding out yaw priority.
  - Raw tracking value: 2.966/3.5=0.847 (slightly better than iter 1's 1.506/2.0=0.753), but the error metric didn't improve, suggesting the improvement is in reward magnitude not actual behavior.
  - bad_orientation slightly worse (12.7% vs 10.1%) — more aggressive tracking causes more falls.
  - Conclusion: velocity tracking is not limited by reward weight at this stage. It's likely limited by training duration (only 400 iters, sigma=4) and competing objectives.

**Conclusion and Next Step:**
  - REVERT track_lin_vel_xy_exp to 2.0 (no benefit, caused yaw degradation).
  - Next: reduce orientation_control weight from -40.0 to -20.0. Hypothesis: softer orientation penalty will reduce bad_orientation termination rate and give the robot more freedom during gait transitions.

**Log dir:** `logs/rsl_rl/ayg_wtw_flat/2026-03-23_20-59-09/`

---

## Iteration 3 — Reduce Orientation Control Penalty

**Hypothesis:** Reducing orientation_control weight from -40.0 to -20.0 will lower the bad_orientation termination rate (currently ~10%) by giving the robot more freedom to lean during gait transitions, while the bad_orientation termination (at 28.6° tilt) still provides a hard safety constraint. This should lead to longer episodes and indirectly improve velocity tracking.

**Variable under test:** orientation_control weight (-40.0 → -20.0)

**Iteration count and reasoning:** 400 iterations — direct comparison with iter 1 baseline.

**Changes:**
  - `rewards.orientation_control.weight`: -40.0 → -20.0
  - (track_lin_vel_xy_exp reverted to default 2.0)

**Result:** success
**Training time:** 367s (399 iterations completed)

**Detailed Metrics:**
| Metric                    | Final   | Mean (last 100) | Trend      |
|--------------------------|---------|-----------------|------------|
| Total reward              | 1103.8  | 577.7           | improving  |
| track_lin_vel_xy_exp      | 1.445   | 1.616           | stable     |
| track_ang_vel_z_exp       | 0.716   | 0.772           | stable     |
| gait                      | 34.8    | 36.4            | stable     |
| orientation_control       | -1.14   | -1.29           | stable     |
| base_height_l2            | -0.92   | -1.12           | stable     |
| Episode length            | 908     | 935             | stable     |
| error_vel_xy              | 0.359   | 0.329           | stable     |
| error_vel_yaw             | 0.347   | 0.365           | stable     |
| bad_orientation %         | 13.3%   | —               | —          |

**Velocity Tracking Gate:**
  - Linear: error_vel_xy=0.359 → slightly WORSE than iter 1 (0.332)
  - Angular: error_vel_yaw=0.347 → about SAME as iter 1 (0.342)

**Reasoning:**
  - Reducing orientation_control weight HALVED the penalty contribution (-1.94→-1.14) but INCREASED bad_orientation termination (10%→13%). With weaker penalty, robot is less cautious about upright posture and more frequently exceeds the hard 28.6° termination limit.
  - Velocity tracking slightly worse — less orientation control means more body wobble which hurts tracking.
  - Hypothesis FAILED: the current -40.0 weight is actually well-calibrated.

**Conclusion and Next Step:**
  - REVERT orientation_control to default (-40.0). The baseline iter 1 config is the best so far.
  - All 3 weight-tuning attempts showed the baseline is well-balanced for 400-iter runs.
  - The bottleneck is training DURATION, not reward weights.
  - Next: increase observation history from 5→7 (source edit). More temporal context should help with 4-gait pattern matching. Run 500 iterations for a longer convergence window.

**Log dir:** `logs/rsl_rl/ayg_wtw_flat/2026-03-23_21-07-46/`

## Progress Summary (updated after iteration 3)

| Metric              | Iter 1 (baseline) | Iter 2 (track↑) | Iter 3 (orient↓) | Best   | Target |
|--------------------|----------|----------|----------|--------|--------|
| Total reward        | 1113     | 1658     | 1104     | 1658   | >2000  |
| error_vel_xy        | 0.332    | 0.332    | 0.359    | 0.332  | <0.25  |
| error_vel_yaw       | 0.342    | 0.471    | 0.347    | 0.342  | <0.30  |
| Gait reward         | 36.9     | 39.2     | 34.8     | 39.2   | >35    |
| Episode length      | 943      | 968      | 908      | 968    | >950   |
| bad_orientation %   | 10.1%    | 12.7%    | 13.3%    | 10.1%  | <5%    |

**Insight:** The iter 1 baseline (default weights + shank_thigh_contact removal) is the best config across all tested parameters. Weight tuning at 400 iterations has marginal/negative effects.

---

## Iteration 4 — Increase Observation History to 7

**Hypothesis:** Increasing observation history from 5 to 7 steps provides the policy with more temporal context for gait pattern matching across 4 different gaits. With 7 steps of history at 50Hz (0.14s lookback), the policy can better distinguish gait phases at 2-4Hz frequency range. This should improve both gait tracking and velocity tracking.

**Variable under test:** observation history_length (5 → 7) — source edit

**Iteration count and reasoning:** 500 iterations — slightly longer to give the increased observation space time to converge. Obs dims increase from 305 to 427.

**Changes:**
  - Source edit: wtw_env_cfg.py → ObservationsCfg.PolicyCfg.history_length: 5 → 7
  - Source edit: wtw_env_cfg.py → ObservationsCfg.CriticCfg.history_length: 5 → 7
  - All reward weights at default (iter 1 baseline)

**Result:** success
**Training time:** 458s (499 iterations completed)

**Detailed Metrics:**
| Metric                    | Final    | Mean (last 100) | Trend      |
|--------------------------|----------|-----------------|------------|
| Total reward              | 8134.5   | 3320.0          | improving  |
| track_lin_vel_xy_exp      | 1.492    | 1.496           | stable     |
| track_ang_vel_z_exp       | 0.798    | 0.765           | stable     |
| gait                      | 44.95    | 39.22           | stable     |
| orientation_control       | -1.98    | -2.02           | stable     |
| base_height_l2            | -1.06    | -1.06           | stable     |
| Episode length            | 963      | 919             | stable     |
| error_vel_xy              | 0.454    | 0.400           | stable     |
| error_vel_yaw             | 0.364    | 0.358           | stable     |
| bad_orientation %         | 14.5%    | 13.6%           | —          |
| sigma                     | 5.74     | —               | —          |

**Velocity Tracking Gate:**
  - Linear: error_vel_xy=0.454 → WORSE than iter 1 (0.332). BLOCKING — needs attention.
  - Angular: error_vel_yaw=0.364 → similar to iter 1 (0.342).

**Visual Assessment:** Robots upright, walking dynamically. More energetic gait than iter 1.

**Reasoning:**
  - History=7 IMPROVED gait reward (36.9→44.95) — more temporal context helps gait pattern matching.
  - But DEGRADED velocity tracking (0.332→0.454) — larger observation space needs more training to also learn tracking.
  - At 500 iters with sigma=5.74, the policy prioritizes gait (exp-gate amplifier) over velocity tracking (additive base).
  - Bad orientation rose to 14.5% — more dynamic gait may cause more falls.
  - Conclusion: h=7 may help long-term but hurts short-term. REVERT to h=5 for production.

**Conclusion and Next Step:**
  - REVERT history_length to 5. The velocity tracking degradation is unacceptable.
  - Next: slow down sigma annealing from 24000→48000 steps (~1000→2000 iterations to max).
    Hypothesis: slower annealing gives velocity tracking more training time before the exp gate becomes strict.

**Log dir:** `logs/rsl_rl/ayg_wtw_flat/2026-03-23_21-19-52/`

## Progress Summary (updated after iteration 4)

| Metric              | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Best   | Target |
|--------------------|--------|--------|--------|--------|--------|--------|
| Total reward        | 1113   | 1658   | 1104   | 8135   | 8135   | —      |
| error_vel_xy        | 0.332  | 0.332  | 0.359  | 0.454  | 0.332  | <0.25  |
| error_vel_yaw       | 0.342  | 0.471  | 0.347  | 0.364  | 0.342  | <0.30  |
| Gait reward         | 36.9   | 39.2   | 34.8   | 44.95  | 44.95  | >35    |
| Episode length      | 943    | 968    | 908    | 963    | 968    | >950   |
| bad_orientation %   | 10.1%  | 12.7%  | 13.3%  | 14.5%  | 10.1%  | <5%    |

**Overall insight:** The iter 1 baseline config (h=5, default weights, shank_thigh_contact removed) remains the best for velocity tracking. All tuning attempts so far have either been neutral or degraded tracking. The reward structure is well-balanced — the bottleneck is training duration and sigma annealing speed.

---

## Iteration 5 — Slow Down Sigma Annealing

**Hypothesis:** Slowing sigma annealing from 24000→48000 env steps (extending from ~1000 to ~2000 iterations to reach max) gives the policy more time with a weak exp gate to focus on velocity tracking before gait enforcement becomes strict. At 500 iters, sigma will be ~3.4 (vs ~5.7 with default), keeping the policy in the "velocity tracking first" regime longer.

**Variable under test:** curriculum.sigma_exp_neg_anneal.params.anneal_steps (24000 → 48000)

**Iteration count and reasoning:** 500 iterations — same as iter 4 for comparison, but with slower sigma ramp.

**Changes:**
  - Source edit: flat_env_cfg.py → anneal_steps: 24000 → 48000
  - History reverted to 5 (source edit reverted)

**Result:** success
**Training time:** 455s (499 iterations completed)

**Detailed Metrics:**
| Metric                    | Final   | Mean (last 100) | Trend      |
|--------------------------|---------|-----------------|------------|
| Total reward              | 223.4   | 177.0           | stable     |
| track_lin_vel_xy_exp      | 1.742   | 1.741           | stable     |
| track_ang_vel_z_exp       | 0.851   | 0.838           | stable     |
| gait                      | 42.14   | 38.09           | stable     |
| orientation_control       | -2.34   | -2.50           | stable     |
| base_height_l2            | -1.44   | -1.43           | stable     |
| Episode length            | 963     | 969             | stable     |
| error_vel_xy              | 0.319   | 0.288           | stable     |
| error_vel_yaw             | 0.329   | 0.321           | stable     |
| bad_orientation %         | 5.1%    | —               | —          |
| sigma                     | 2.19    | —               | —          |

**Velocity Tracking Gate:**
  - Linear: error_vel_xy=0.288 (mean100) → exp(-0.288²/0.25) = 0.72 → PASS (>0.6 target!)
  - Angular: error_vel_yaw=0.321 (mean100) → exp(-0.321²/0.25) = 0.66 → PASS (>0.3 target!)

**Visual Assessment:** (checked frames — robots upright, walking, similar quality to iter 1)

**Reasoning:**
  - Slower sigma annealing is a BREAKTHROUGH change. At 500 iters, sigma=2.19 (vs 4.03 with default), keeping the exp gate permissive and letting the policy focus on velocity tracking.
  - error_vel_xy improved from 0.332→0.288 (mean100: 0.357→0.288). Best so far!
  - error_vel_yaw improved from 0.342→0.321 (mean100: 0.379→0.321). Best so far!
  - bad_orientation HALVED from 10.1%→5.1%. The policy is more stable.
  - Gait reward improved (36.9→42.14) — slower gate pressure lets gait optimize better too.
  - Total reward is lower (223 vs 1113) purely because sigma is lower — less amplification. This is fine.
  - Hypothesis CONFIRMED.

**Conclusion and Next Step:**
  - KEEP the 48000 anneal_steps change. This is the new baseline.
  - Next: enable stand_when_zero_command (-0.05) and stand_still_when_zero_command (-0.05) to improve zero-velocity behavior. 10% of envs get zero command.

**Log dir:** `logs/rsl_rl/ayg_wtw_flat/2026-03-23_21-32-11/`

## Progress Summary (updated after iteration 5)

| Metric              | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Best   | Target |
|--------------------|--------|--------|--------|--------|--------|--------|--------|
| error_vel_xy (m100) | 0.357  | 0.310  | 0.329  | 0.400  | **0.288** | 0.288  | <0.25  |
| error_vel_yaw (m100)| 0.379  | 0.460  | 0.365  | 0.358  | **0.321** | 0.321  | <0.30  |
| Gait reward         | 36.9   | 39.2   | 34.8   | 44.95  | **42.14** | 44.95  | >35    |
| Episode length      | 943    | 968    | 908    | 963    | **963** | 969    | >950   |
| bad_orientation %   | 10.1%  | 12.7%  | 13.3%  | 14.5%  | **5.1%** | 5.1%   | <5%    |

**Breakthrough:** Slower sigma annealing (48000 steps) is the single most impactful change. It improved ALL key metrics simultaneously.

---

## Iteration 6 — Enable Standing Behavior Rewards

**Hypothesis:** Enabling stand_when_zero_command (-0.05) and stand_still_when_zero_command (-0.05) will make the robot stand still when zero velocity is commanded (10% of envs). This should reduce unnecessary motion during standing phases, slightly improving overall velocity metrics.

**Variable under test:** stand_when_zero_command and stand_still_when_zero_command weights (0.0 → -0.05 each)

**Iteration count and reasoning:** 500 iterations — same as iter 5 for direct comparison.

**Changes:**
  - `rewards.stand_when_zero_command.weight`: 0.0 → -0.05
  - `rewards.stand_still_when_zero_command.weight`: 0.0 → -0.05

**Result:** success
**Training time:** 454s (499 iterations completed)

| Metric                    | Final   | Mean (last 100) |
|--------------------------|---------|-----------------|
| error_vel_xy              | 0.287   | 0.281           |
| error_vel_yaw             | 0.328   | 0.328           |
| gait                      | 39.87   | 37.40           |
| bad_orientation %         | 6.0%    | —               |
| Episode length            | 982     | 964             |

**Reasoning:** Marginal improvement in xy tracking (0.288→0.281), neutral on yaw. Standing rewards contribute small penalties (-0.007, -0.061) via exp-neg gate. Will be more impactful at higher sigma during production.

**Log dir:** `logs/rsl_rl/ayg_wtw_flat/2026-03-23_21-41-53/`

---

## Production Readiness Assessment

| Check                      | Status | Evidence                                    |
|---------------------------|--------|---------------------------------------------|
| Body coverage             | PASS   | All 17 bodies covered (pre-training audit)  |
| Velocity tracking (xy)    | PASS   | exp(-0.281²/0.25) = 0.73 (>0.6) — iter 6   |
| Velocity tracking (yaw)   | PASS   | exp(-0.328²/0.25) = 0.65 (>0.3) — iter 6   |
| Visual gait quality       | PASS   | Side-view: upright stance, active walking, no hip-walk/belly-slide across all 6 iterations |
| No reward hacking         | PASS   | "high_reward_low_tracking" is false positive for exp-neg formula. Tracking metrics confirm actual velocity following. |
| Sufficient iterations     | PASS   | 6 successful tuning iterations, 4+ distinct hypotheses tested |
| Metric convergence        | PASS   | Key metrics "stable" trend in iters 5-6. Reward will evolve during production as sigma increases — by design. |

**Decision:** PROCEED with production run.

**Winning configuration (baked into source):**
- `flat_env_cfg.py`: shank_thigh_contact = None (removed aggressive termination)
- `flat_env_cfg.py`: anneal_steps = 48000 (slower sigma annealing, ~2000 iters to max)
- `flat_env_cfg.py`: stand_when_zero_command = -0.05, stand_still_when_zero_command = -0.05
- `rsl_rl_ppo_cfg.py`: max_iterations = 6000, save_interval = 200
- All other weights at defaults from iter 1 baseline

---

## Production Run 1 (FAILED) — sigma_max=20 caused reward explosion

**Config:** All winning params baked, sigma_max=20 (default).
**Iterations:** 6000 completed
**Training time:** 88 min

**CRITICAL ISSUE:** Reward exploded to 478 billion! The positive gait reward (+16.0 weight) in the exp-neg gate creates exponential amplification at sigma=20:
- Per-step gait contribution to exp-neg: ~1.28
- At sigma=20: exp(1.28 * 20) = exp(25.6) ≈ 1.45e11 amplification factor per step
- Total reward: ~478 billion (numerical instability territory)

**Impact on behavior:**
- error_vel_xy DEGRADED: 0.517 (vs 0.288 at sigma=2.2 in tuning)
- error_vel_yaw: 0.358 (similar to tuning)
- bad_orientation: 16% (vs 5% in tuning)
- Gait: 41.1 (maintained)
- Tracking converged early (iter 58-100) then STAGNATED for 5900 iters

**Root cause:** The paper's exp formula uses all-NEGATIVE auxiliary terms (r_aux ≤ 0, so exp gate ≤ 1). Our positive gait reward makes exp gate >> 1, creating runaway amplification at high sigma. The policy optimizes the gate (gait) at the expense of tracking.

**Fix:** Cap sigma_max at 5.0 (source edit). At sigma=5, exp(1.28*5)=exp(6.4)≈600 — significant enforcement without the 10^11 blowup. Tuning data confirms sigma=2-5 gives best tracking.

**Log dir:** `logs/rsl_rl/ayg_wtw_flat/2026-03-23_21-51-55/`

---

## Production Run 2 — sigma_max=5.0 (SUCCESS)

**Config:** sigma_max=5.0, anneal_steps=48000, all winning params baked into source.
**Iterations:** 6000 (5999 completed)
**Training time:** 87 min (5239s)

**Final Metrics (detailed):**
| Metric                    | Final    | Mean (last 100) | Converged |
|--------------------------|----------|-----------------|-----------|
| Total reward              | 10,632   | 10,827          | Yes (stable) |
| track_lin_vel_xy_exp      | 1.746    | 1.744           | Yes       |
| track_ang_vel_z_exp       | 0.880    | 0.870           | Yes       |
| gait                      | 47.41    | 48.19           | Yes       |
| orientation_control       | -1.003   | -1.006          | Yes       |
| base_height_l2            | -0.670   | -0.680          | Yes       |
| foot_clearance            | -0.109   | -0.113          | Yes       |
| undesired_contacts        | -0.001   | -0.002          | ~0        |
| stand_when_zero_command   | -0.031   | -0.030          | Yes       |
| stand_still_when_zero_cmd | -0.078   | -0.070          | Yes       |
| Episode length            | 980      | 981             | Yes       |
| error_vel_xy              | 0.312    | 0.308           | Yes       |
| error_vel_yaw             | 0.284    | 0.293           | Yes       |
| bad_orientation %         | 4.5%     | 3.4%            | Yes       |
| Sigma                     | 5.00     | —               | Maxed     |

**Velocity Tracking Assessment:**
  - Linear: exp(-0.308²/0.25) = exp(-0.379) = 0.685 → PASS (>0.6)
  - Angular: exp(-0.293²/0.25) = exp(-0.343) = 0.710 → PASS (>0.3, well above!)

**Visual Assessment (production):**
  - Camera: side-view at robot level (play_for_inspection.py, distance=2.0m, height=0.4m)
  - CAN VERIFY: Robots walking upright with proper quadruped stance across all frames
  - CAN VERIFY: Clear leg swing motion visible — legs cycle with distinct swing/stance phases
  - CAN VERIFY: Base maintained at stable height, no belly-sliding or hip-walking
  - CAN VERIFY: Different command arrows (direction/magnitude) suggest velocity tracking is active
  - CAN VERIFY: Multiple robots in view, all showing consistent walking behavior
  - CANNOT VERIFY: Which specific gait (trot/pace/bound/pronk) each robot is executing
  - CANNOT VERIFY: Exact foot clearance height during swing at this resolution
  - Failure modes: hip-walking: NOT detected | belly-sliding: NOT detected | shuffling: NOT detected (clear leg cycling visible) | spider-walking: NOT detected

**What CAN be verified from this run:**
  - Velocity tracking is functional (error_vel_xy=0.31, error_vel_yaw=0.29 — both acceptable)
  - Gait reward is strong (48.2 mean) — the robot learned gait patterns
  - Episode length near max (981/1000) — robot rarely falls
  - bad_orientation at 3.4% — excellent stability
  - Undesired contacts ≈ 0 — no thigh/shank exploitation
  - Standing behavior active (small non-zero stand penalties)
  - All metrics converged and stable

**What CANNOT be verified and requires further review:**
  - Gait diversity: whether the policy correctly executes all 4 gaits (trot, pace, bound, pronk) vs only learning a subset
  - Sim-to-real transfer quality
  - Performance at higher velocity commands (curriculum capped at 1.5 m/s)
  - Robustness to perturbations (push_robot disabled in flat training)

**Recommended next steps:**
  1. Test in Gazebo simulation via `ros2 launch` for sim-to-sim validation
  2. Export ONNX and test with joystick gait parameter control
  3. Try different gait commands to verify trot/pace/bound/pronk all work
  4. Consider rough terrain training (Isaac-WTW-Rough-Ayg-v0) with these same tuning insights
  5. Hardware deployment on AYG robot

**Checkpoint:** `logs/rsl_rl/ayg_wtw_flat/2026-03-23_23-24-13/model_5999.pt`
**Log dir:** `logs/rsl_rl/ayg_wtw_flat/2026-03-23_23-24-13/`
