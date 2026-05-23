# Reward Reference for Legged Locomotion

A general-purpose reference for reward functions used in RL-based quadruped locomotion.
Each entry includes the mathematical formula, key parameters, typical weight ranges from our configurations, and design insights.
This catalog is environment-agnostic — individual environments compose subsets of these rewards with their own weights.

**Notation.** Subscripts: $x, y, z$ are body-frame axes; $i$ indexes joints or feet.
Superscript $cmd$ denotes a commanded quantity.
$\Delta t$ is the control timestep.
$\mathbf{q}$ is joint positions, $\dot{\mathbf{q}}$ velocities, $\ddot{\mathbf{q}}$ accelerations, $\boldsymbol{\tau}$ applied torques.
$\mathbf{v}$ is base linear velocity, $\boldsymbol{\omega}$ is base angular velocity.

---

## Part 1: Cross-cutting Concepts

### 1.1 Reward Structure in Locomotion RL

The total reward at each timestep is a weighted sum of individual terms:

$$r_t = \sum_i w_i \cdot r_i(s_t, a_t) \cdot \Delta t$$

where $w_i$ is the weight (positive for rewards, negative for penalties) and $\Delta t$ is the control timestep (`sim.dt × decimation`).
Multiplying by $\Delta t$ ensures **timestep invariance**: reward magnitudes stay consistent if you change the simulation frequency, so hyperparameters transfer between setups with different dt values.
In practice, the positive tracking rewards provide the "carrot" that drives the policy toward task completion, while the negative penalty terms act as "sticks" that suppress undesirable behaviors.
A useful mental model is the **reward budget**: the policy must earn enough from tracking rewards to offset the cost of penalties, so only efficient locomotion strategies survive.

### 1.2 Penalty Kernels: L2 Quadratic vs Exponential

Most reward terms use one of two kernel shapes to transform raw errors into reward signals.

**L2 Quadratic (squared) penalty:**

$$r = \sum_i e_i^2$$

Properties:
zero gradient at zero error (tolerant of small deviations),
gradient grows linearly with error (strong pull back from large deviations),
unbounded output.
Used with negative weight to suppress a quantity.
The quadratic shape makes it lenient near zero but aggressive against large values.

**Exponential (exp) kernel** — two common variants:

$$r = \exp\!\Bigl(-\frac{\|\mathbf{e}\|^2}{\sigma^2}\Bigr) \quad \text{(Gaussian)}
\qquad\qquad
r = \exp\!\Bigl(-\frac{\|\mathbf{e}\|}{\sigma}\Bigr) \quad \text{(Laplacian)}$$

Properties:
output bounded in $[0, 1]$,
equals 1.0 at zero error,
$\sigma$ controls the **tolerance radius** — the error magnitude at which reward drops to $\approx 0.37$.

- The **Gaussian** variant has zero gradient at zero error and is smoother near the optimum — good as a default.
- The **Laplacian** variant has a constant gradient magnitude near zero, providing stronger shaping signal when the policy is already close to the target — useful when precise tracking matters.

**Gradient saturation trade-off.** The exp kernel saturates at 0 for large errors, meaning the gradient vanishes far from the target.
If the policy starts very far from the goal (e.g., a random initialization), exp rewards provide almost no learning signal.
L2 penalties do not have this problem — their gradient grows with error.
This is why a typical reward set uses exp kernels for tracking objectives (bounded, well-shaped near target) and L2 penalties for regularization (always provides gradient).

**Other kernels** (less common):
**L1** ($\sum |e_i|$) — constant gradient regardless of error magnitude, useful for default-pose regularization;
**L2 norm** ($\|\mathbf{e}\|$, unsquared) — constant-magnitude gradient like L1 but considers all dimensions jointly;
**tanh** ($\tanh(\alpha \|\mathbf{e}\|)$) — saturating function used as a gating signal (e.g., weighting clearance reward by foot velocity) rather than as a standalone reward.

### 1.3 Reward Scaling Considerations

Reward weights are **not portable** across robots or configurations. Key interactions:

- **Mass dependence.** Torque and acceleration penalties scale with the robot's mass (heavier robots require more torque).
  For a ~20 kg robot with 30 N·m effort limit, `joint_torques_l2` weights around $-10^{-4}$; a 50+ kg robot would need a smaller magnitude to avoid over-penalizing necessary forces.
- **Action scale interaction.** If `action_scale` = 0.25 rad, the raw action is multiplied by 0.25 before being applied as a joint position offset.
  Doubling `action_scale` quadruples `action_rate_l2` (since it is squared), so the penalty weight must decrease accordingly.
- **Command range interaction.** Wider velocity command ranges (e.g., $[-2, 3]$ m/s vs $[-1, 1]$ m/s) make the tracking reward harder to saturate, increasing its average magnitude.
  Tracking weights may need adjustment when command ranges change.
- **Terrain interaction.** Some rewards should be enabled or disabled per terrain type.
  For example, `flat_orientation_l2` is typically active on flat terrain ($w = -5.0$) but disabled on rough terrain ($w = 0.0$) where some body tilt is unavoidable.

### 1.4 Terminations and the Survival Incentive

Isaac Lab distinguishes two termination types, and understanding the difference is critical for reward design:

- **`is_terminated`** (e.g., illegal body contact): the episode ends and the value function is set to zero.
  The agent loses all future reward — this acts as an **implicit penalty** worth approximately $\gamma \cdot V(s)$, which can be very large.
  No explicit penalty weight is needed; the termination itself teaches avoidance.
- **`time_out`**: the episode ends but the value function is **bootstrapped** (not zeroed).
  There is no implicit penalty — the agent treats it as if the episode continues.
  Used for maximum episode length or out-of-bounds conditions.

An **explicit survival reward** (a small positive constant each step, e.g., $+1.0$) can help early in training to prevent the policy from immediately falling over, since it creates a direct incentive to keep the episode alive.
However, it compounds over long episodes and can dominate the reward signal, potentially encouraging overly conservative behavior.
Use sparingly and consider annealing it as training progresses.

### 1.5 Exp-Negative Rewards: Multiplicative Gating

The standard reward paradigm (Section 1.1) sums all terms additively. A more expressive alternative is the **exp-negative** structure, where penalty terms *multiplicatively gate* the tracking rewards instead of being summed with them.

This multiplicative terms have the advantage of never causing negative rewards and, thus, early termination. **Empirically**, they make reward tuning easier but convergence slower.

**Total reward formula:**

$$R = \underbrace{\sum_i w_i^{+} r_i^{+} \Delta t}_{\text{additive terms}} \;\cdot\; \exp\!\Bigl(\underbrace{\sum_j w_j^{-} r_j^{-} \Delta t}_{\text{exp-negative terms}} \cdot \sigma\Bigr)$$

The additive terms (e.g., velocity tracking) accumulate into a positive reward as usual.
The exp-negative terms (e.g., height error, gait violations, undesired contacts) accumulate separately into a negative sum, which is then exponentiated to produce a **multiplicative gate** in $[0, 1]$.

**How the gate works:**

- **Good behavior** — exp-negative penalties are small, so $\exp(\text{small negative} \cdot \sigma) \approx 1$, and the full tracking reward passes through.
- **Bad behavior** — exp-negative penalties are large, so $\exp(\text{large negative} \cdot \sigma) \to 0$, and the tracking reward is suppressed regardless of how well the policy tracks velocity.

This creates a fundamentally different optimization landscape from additive penalties:

| Property | Additive penalties | Exp-negative penalties |
|---|---|---|
| Interaction with tracking | Independent — penalty and tracking are summed | Coupled — penalty gates how much tracking reward is received |
| Bad-behavior signal | Negative reward (can outweigh tracking) | Tracking reward approaches zero (cannot go negative) |
| Multi-penalty scaling | Each penalty adds independently; total penalty grows linearly | Penalties compound multiplicatively; violating *any* term suppresses the whole reward |
| Reward range | Unbounded below | Always $\geq 0$ (bounded by the tracking reward) |

**Sigma annealing.** The $\sigma$ coefficient controls how strict the gate is.
It is typically annealed via curriculum during training:

$$\sigma(t) = \sigma_{min} + (\sigma_{max} - \sigma_{min}) \cdot \min\!\left(\frac{t}{T_{anneal}},\; 1\right)^2$$

- **Early training** ($\sigma$ low): the gate is weak ($\exp(\cdot) \approx 1$), so the policy focuses on learning to track velocity without being overwhelmed by secondary constraints.
- **Late training** ($\sigma$ high): the gate becomes strict, and the policy must satisfy all exp-negative constraints to receive meaningful reward.

Typical annealing: $\sigma_{min} = 1$, $\sigma_{max} = 20$, $T_{anneal} = 48000$ steps (quadratic schedule).

**When to use exp-negative vs additive.**
Exp-negative penalties are well-suited for secondary behavioral constraints (gait pattern, body height, foot placement) that should not interfere with early velocity-tracking learning but must be strictly enforced later.
Additive penalties remain better for smooth regularization terms (torque, acceleration, action rate) where the gradient should always be present.

#### Implementation

The exp-negative reward structure is implemented in `cf_lab/envs/wtw_env.py` as `ExpNegativeRewardManager`, a custom `RewardManager` subclass.
Each reward term is tagged with a `RewardType` — either `ADDITIVE` (default) or `EXP_NEGATIVE` — via the `WtwRewTerm` config class in the WTW environment config.
The sigma annealing curriculum is in `walk_these_ways/mdp/curriculums.py:anneal_sigma_exp_neg`.

#### Example: Walk-These-Ways reward budget

From the WTW rough environment (`rough_env_cfg.py`):

| Term | Type | Weight | Purpose |
|------|------|--------|---------|
| `track_lin_vel_xy_exp` | Additive | +2.0 | Primary task reward |
| `track_ang_vel_z_exp` | Additive | +1.0 | Yaw tracking |
| `gait` | Exp-negative | $-16.0$ | Enforce commanded gait pattern |
| `foot_clearance` | Exp-negative | $-150.0$ | Penalize scuffing during swing |
| `base_height_l2` | Exp-negative | $-100.0$ | Maintain target body height |
| `undesired_contacts` | Exp-negative | $-10.0$ | Penalize thigh/shank contact |
| `zero_vel_when_zero_command` | Exp-negative | $-10.0$ | Stand still when commanded |
| `stand_when_zero_command` | Exp-negative | $-1.0$ | Return to default pose when stopped |
| `feet_slip` | Exp-negative | $-0.04$ | Penalize foot sliding |
| `orientation_control` | Exp-negative | $-40.0$ | Track commanded body pitch/roll |
| `lin_vel_z_l2` | Additive | $-2.0$ | Suppress bouncing |
| `action_rate_l2` | Additive | $-0.01$ | Smooth actions |
| `joint_torques_l2` | Additive | $-2 \times 10^{-4}$ | Energy efficiency |

Note: exp-negative weights are much larger in magnitude (e.g., $-150$) than typical additive penalties because the raw reward values are small and then further attenuated by the $\exp(\cdot)$ operation.
The large weights ensure that violations produce a meaningful suppression of the gate.

---

## Part 2: Reward Catalog

### Source Locations

| Label | Path |
|-------|------|
| **Built-in** | `isaaclab/envs/mdp/rewards.py` |
| **Custom** | `source/cf_lab/cf_lab/tasks/manager_based/velocity/mdp/rewards.py` |
| **WTW** | `source/cf_lab/cf_lab/tasks/manager_based/walk_these_ways/mdp/rewards.py` |
| **Direct** | `source/cf_lab/cf_lab/tasks/direct/velocity/ayg_env.py` |
| **Exp-neg manager** | `source/cf_lab/cf_lab/envs/wtw_env.py` |

---

### A. Velocity Tracking

#### `track_lin_vel_xy_exp`

Reward tracking of commanded XY linear velocity using a Gaussian exponential kernel.

$$r = \exp\!\left(-\frac{\|\mathbf{v}^{cmd}_{xy} - \mathbf{v}_{xy}\|^2}{\sigma^2}\right)$$

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `std` ($\sigma$) | Tolerance — error magnitude at which reward $\approx 0.37$ | 0.25–1.0 |

**Typical weight:** +1.0 to +2.0

This is the primary task reward for velocity-tracking locomotion.
The Gaussian kernel is very forgiving near perfect tracking (zero gradient at zero error) but drops sharply as error exceeds $\sigma$.
A smaller $\sigma$ demands more precise tracking but can make learning harder early on.
This is the single most important reward — without sufficient weight here, the policy has no incentive to move.

**Variant — `base_linear_velocity_reward`** (custom): Uses a Laplacian kernel $\exp(-\|\mathbf{e}\|/\sigma)$ which provides stronger gradient near perfect tracking.
Includes a **velocity-dependent scaling ramp**: above `ramp_at_vel`, the reward is multiplied by $\max(1,\; 1 + \text{rate} \cdot (\|\mathbf{v}^{cmd}\| - v_{ramp}))$, boosting the incentive to track high-speed commands.
Typical weight: +10.0, $\sigma = 1.0$, `ramp_at_vel` = 1.0 m/s, `ramp_rate` = 0.5.

*Built-in: `isaaclab/envs/mdp/rewards.py`. Custom: `velocity/mdp/rewards.py:70`.*

---

#### `track_ang_vel_z_exp`

Reward tracking of commanded yaw angular velocity using a Gaussian exponential kernel.

$$r = \exp\!\left(-\frac{(\omega^{cmd}_z - \omega_z)^2}{\sigma^2}\right)$$

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `std` ($\sigma$) | Tolerance radius | 0.25–2.0 |

**Typical weight:** +0.5 to +1.0. Typically weighted exactly half of the linear velocity reward.

Analogous to linear velocity tracking but for yaw rate.
Usually weighted lower than linear velocity because precise yaw control is less critical for stable locomotion.
The scalar error (1D) means this reward is easier to saturate than the 2D linear velocity tracking.

**Variant — `base_angular_velocity_reward`** (custom): Laplacian kernel $\exp(-|e|/\sigma)$ with non-zero gradient at perfect tracking.
Typical weight: +10.0, $\sigma = 2.0$.

*Built-in: `isaaclab/envs/mdp/rewards.py`. Custom: `velocity/mdp/rewards.py:60`.*

---

### B. Body Stability

#### `flat_orientation_l2`

Penalize non-flat base orientation via the XY components of the projected gravity vector.

$$r = g_x^2 + g_y^2$$

where $\mathbf{g}_{xy}$ is the gravity vector projected into the robot's body frame (XY components only).
When the robot is perfectly level, $\mathbf{g}_{xy} = \mathbf{0}$.

| Parameter | Description |
|-----------|-------------|
| (none) | No tunable parameters |

**Typical weight:** $-5.0$ (flat), $0.0$ (rough)

Essential on flat terrain where the robot should remain level.
Disabled on rough terrain because terrain slopes make perfect leveling impossible and penalizing it fights the terrain curriculum.
The L2-squared kernel tolerates small tilts (near-zero gradient) but penalizes large ones aggressively.

**Variant — `base_orientation_penalty`** (custom): Uses L2 norm $\|\mathbf{g}_{xy}\|$ instead of L2-squared.
Provides a constant-magnitude corrective gradient even at small tilts.
Typical weight: $-3.0$.

*Built-in: `isaaclab/envs/mdp/rewards.py`. Custom: `velocity/mdp/rewards.py:221`.*

---

#### `base_height_l2`

Penalize deviation of base height from a target value.

$$r = (h - h_{target})^2$$

On rough terrain with a height scanner, the target adjusts to local terrain:

$$r = \bigl(h - (h_{target} + \bar{z}_{terrain})\bigr)^2$$

where $\bar{z}_{terrain}$ is the mean terrain height from ray-cast sensor readings.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `target_height` | Desired base height (m) | 0.35 |
| `sensor_cfg` | Height scanner for terrain adjustment (optional) | — |

**Typical weight:** $-5.0$

Prevents the robot from crouching or bouncing.
The target should match the robot's nominal standing height (0.35 m for AYG at default joint positions).
On rough terrain, the sensor adjustment is critical — without it, the reward penalizes the robot for walking on elevated terrain.

*Built-in: `isaaclab/envs/mdp/rewards.py`. Direct: `ayg_env.py:226`.*

---

#### `lin_vel_z_l2`

Penalize vertical (z-axis) base linear velocity.

$$r = v_z^2$$

| Parameter | Description |
|-----------|-------------|
| (none) | No tunable parameters |

**Typical weight:** $-2.0$

Suppresses vertical bouncing and hopping.
Works with `ang_vel_xy_l2` to keep base motion smooth.
The squared kernel tolerates small vertical velocities from normal walking dynamics but strongly penalizes hopping or jumping.

*Built-in: `isaaclab/envs/mdp/rewards.py`.*

---

#### `ang_vel_xy_l2`

Penalize roll and pitch angular velocity of the base.

$$r = \omega_x^2 + \omega_y^2$$

| Parameter | Description |
|-----------|-------------|
| (none) | No tunable parameters |

**Typical weight:** $-0.05$

Suppresses body rocking and rolling.
The low weight (relative to other penalties) reflects that some roll/pitch velocity is inevitable during dynamic locomotion — this term only needs to prevent excessive oscillation, not eliminate it entirely.

*Built-in: `isaaclab/envs/mdp/rewards.py`.*

---

#### `base_motion`

Combined penalty for vertical velocity and roll/pitch rate with asymmetric weighting.

$$r = 0.8 \cdot v_z^2 + 0.2 \cdot \bigl(|\omega_x| + |\omega_y|\bigr)$$

| Parameter | Description |
|-----------|-------------|
| (none) | Hardcoded 0.8/0.2 weighting |

**Typical weight:** $-2.0$

A convenience term that bundles vertical velocity and angular velocity penalties into one reward with opinionated weighting.
The 0.8/0.2 split emphasizes vertical velocity suppression (bouncing is worse than rocking).
Note the angular terms use L1 (absolute value) rather than L2, providing a constant corrective gradient even for small oscillations.

*Custom: `velocity/mdp/rewards.py:212`.*

---

### C. Energy & Smoothness

#### `joint_torques_l2`

Penalize applied joint torques to encourage energy-efficient locomotion.

$$r = \sum_{i \in \mathcal{J}} \tau_i^2$$

where $\mathcal{J}$ is the set of selected joint indices.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `asset_cfg.joint_ids` | Which joints to penalize | `".*"` (all 12) |

**Typical weight:** $-10^{-4}$

Encourages energy efficiency and reduces actuator stress.
The very small weight reflects that torque values can be large (up to 30 N·m effort limit, 60 N·m saturation for AYG).
The weight must scale inversely with $(\text{effort\_limit})^2$ — doubling actuator capacity requires quartering the weight to maintain the same effective penalty.

**Variant — `joint_torques_penalty`** (custom): Uses L2 norm $\|\boldsymbol{\tau}\|$ instead of sum-of-squares.
The unsquared norm provides a more uniform penalty gradient across all torque magnitudes.
Typical weight: $-5 \times 10^{-4}$.

*Built-in: `isaaclab/envs/mdp/rewards.py`. Custom: `velocity/mdp/rewards.py:266`.*

---

#### `joint_acc_l2`

Penalize joint accelerations to promote smooth joint motion.

$$r = \sum_{i \in \mathcal{J}} \ddot{q}_i^2$$

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `asset_cfg.joint_ids` | Which joints to penalize | `".*"` or `".*_HAA", ".*_HFE"` |

**Typical weight:** $-2.5 \times 10^{-7}$

Joint accelerations can be very large, hence the extremely small weight.
Smooths out jerky motion that would be harsh on real hardware.
Sometimes applied only to HAA and HFE joints (hip abduction/flexion), leaving KFE (knee) free to accelerate quickly for foot clearance.

**Variant — `joint_acceleration_penalty`** (custom): Uses L2 norm $\|\ddot{\mathbf{q}}\|$ applied to HAA+HFE joints only.
Typical weight: $-10^{-4}$.

*Built-in: `isaaclab/envs/mdp/rewards.py`. Custom: `velocity/mdp/rewards.py:248`.*

---

#### `joint_vel_l2`

Penalize joint velocities.

$$r = \sum_{i \in \mathcal{J}} \dot{q}_i^2$$

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `asset_cfg.joint_ids` | Which joints to penalize | `".*_HAA", ".*_HFE"` |

**Typical weight:** $-10^{-2}$

Limits joint speeds to reduce mechanical stress and energy consumption.
Often applied selectively to hip joints (HAA, HFE), allowing the knee (KFE) higher velocities needed for foot clearance during swing phase.

**Variant — `joint_velocity_penalty`** (custom): Uses L2 norm $\|\dot{\mathbf{q}}\|$.
Typical weight: $-10^{-2}$.

*Custom: `velocity/mdp/rewards.py:273`.*

---

#### `action_rate_l2`

Penalize the rate of change of the policy's output actions.

$$r = \sum_i (a_{t,i} - a_{t-1,i})^2$$

| Parameter | Description |
|-----------|-------------|
| (none) | No tunable parameters |

**Typical weight:** $-0.01$

Smooths the policy output over time, reducing jitter that would stress actuators on real hardware.
This operates on the raw policy output, not the actual joint commands — if `action_scale` is large, the same raw change produces a larger joint motion, so the weight may need to increase with smaller action scales (or decrease with larger ones).

*Built-in: `isaaclab/envs/mdp/rewards.py`.*

---

#### `action_smoothness`

Penalize the magnitude of action changes using L2 norm (unsquared).

$$r = \|\mathbf{a}_t - \mathbf{a}_{t-1}\|$$

| Parameter | Description |
|-----------|-------------|
| (none) | No tunable parameters |

**Typical weight:** $-1.0$

Similar intent to `action_rate_l2` but uses the unsquared norm — the gradient magnitude is constant regardless of how large the action change is.
This provides a steadier smoothing signal: large jumps aren't penalized disproportionately, but small changes aren't ignored either.
The higher weight magnitude (vs $-0.01$ for the squared variant) reflects that the unsquared output is much smaller.

*Custom: `velocity/mdp/rewards.py:192`.*

---

### D. Foot & Contact Management

#### `feet_air_time`

Reward feet for spending time in the air, gated by velocity command.

$$r = \sum_{i \in \text{feet}} (t^{air}_i - t_{thresh}) \cdot \mathbb{1}[\text{first\_contact}_i] \cdot \mathbb{1}[\|\mathbf{v}^{cmd}_{xy}\| > 0.1]$$

Rewards the excess air time above a threshold at the moment each foot lands.
Negative contributions (air time below threshold) penalize rapid stomping.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `threshold` | Minimum air time to earn reward (s) | 0.4 |

**Typical weight:** +0.01 (rough) to +0.25 (flat)

Encourages the robot to lift its feet and take proper strides rather than shuffling.
Higher weight on flat terrain where air time is easier to achieve; lower on rough terrain where cautious stepping is preferred.
The command gating ensures the robot isn't penalized for standing still.

**Variant — `air_time_reward`** (custom, spot-inspired): More sophisticated.
When moving ($\|\mathbf{v}^{cmd}\| > 0$), rewards time in the current phase (air or contact) up to `mode_time`, then stops rewarding to encourage phase cycling.
When standing ($\|\mathbf{v}^{cmd}\| = 0$), rewards contact time minus air time, encouraging the robot to keep all feet on the ground.
Parameters: `mode_time` = 0.3 s, typical weight: +5.0.

*Built-in: `isaaclab_tasks/.../velocity/mdp`. Custom: `velocity/mdp/rewards.py:31`.*

---

#### `foot_clearance`

Reward swing feet for achieving a target height, using an exp kernel modulated by foot velocity.

**Variant 1 — `foot_clearance_reward`** (spot-inspired):

$$r = \exp\!\left(-\frac{\sum_i (z_i - h_{target})^2 \cdot \tanh(\alpha \|\mathbf{v}^{xy}_i\|)}{\sigma}\right)$$

The tanh gating ensures that only feet that are actively moving (swinging) contribute — feet at rest on the ground have near-zero velocity and are effectively ignored.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `target_height` | Desired swing foot height (m) | 0.10 |
| `std` ($\sigma$) | Exp kernel tolerance | 0.05 |
| `tanh_mult` ($\alpha$) | Velocity gating steepness | 2.0 |

**Typical weight:** +1.0

**Variant 2 — `foot_clearance_swing`** (rough/flat manager-based):

$$r = \sum_{i \in \text{feet}} \exp\!\left(-\frac{(z_i - h_{target})^2}{\sigma}\right) \cdot \mathbb{1}[t^{air}_i > 0] \;\cdot\; \mathbb{1}[\|\mathbf{v}^{cmd}_{xy}\| > v_{thresh}]$$

Uses the contact sensor's air time to identify swing phase rather than velocity-based gating.
Only active when the robot is commanded to move.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `target_height` | Desired swing foot height (m) | 0.10 |
| `sigma` ($\sigma$) | Exp kernel tolerance | 0.005 |
| `velocity_threshold` | Min command speed to activate (m/s) | 0.1 |

**Typical weight:** +0.25

Both variants prevent foot scuffing, which is critical for sim-to-real transfer.
The target height determines the gait style — 0.10 m produces a moderate trot; higher values yield more exaggerated stepping.

*Custom: `velocity/mdp/rewards.py:176` (variant 1), `velocity/mdp/rewards.py:280` (variant 2).*

---

#### `feet_regulation`

Penalize foot XY velocity when feet are near the ground (stance phase proxy).

$$r = \sum_{i \in \text{feet}} \|\mathbf{v}^{xy}_i\|^2 \cdot \exp\!\left(-\frac{z_i}{0.025 \cdot h_{desired}}\right)$$

The exponential height gate produces a large multiplier when feet are near the ground ($z \approx 0$) and decays to near-zero when feet are at swing height.
This acts as a continuous approximation of "penalize velocity only during stance."

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `desired_body_height` | Robot standing height for normalization (m) | 0.35 |
| `sensor_cfg` | Height scanner for terrain adjustment (optional) | — |

**Typical weight:** $-0.15$

Encourages clean foot placement — feet should be nearly stationary when they touch the ground.
Complements `foot_slip` but uses a softer, differentiable height-based gate rather than binary contact detection.
The $0.025 \cdot h_{desired}$ denominator in the exponential means the gate is very sharp: at $z = 0$, $\exp(0) = 1$; at $z = h_{desired}$, $\exp(-40) \approx 0$.

*Custom: `velocity/mdp/rewards.py:306`. Direct: `ayg_env.py:209`.*

---

#### `foot_slip`

Penalize foot planar velocity when in contact with the ground.

$$r = \sum_{i \in \text{feet}} \mathbb{1}[\|\mathbf{F}_i\| > F_{thresh}] \cdot \|\mathbf{v}^{xy}_i\|$$

Uses contact force magnitude to determine if a foot is on the ground, then penalizes its XY velocity.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `threshold` ($F_{thresh}$) | Contact force threshold (N) | 1.0 |

**Typical weight:** $-0.5$

Directly penalizes sliding feet, which wastes energy and causes instability on real surfaces with limited friction.
The binary contact detection (via force threshold) means this reward has a discontinuous gate — feet transition sharply between penalized and not-penalized states, which can make optimization noisier than the smooth `feet_regulation` alternative.

*Custom: `velocity/mdp/rewards.py:231`.*

---

#### `undesired_contacts`

Count contact violations on forbidden body parts (thighs, shanks).

$$r = \sum_{i \in \mathcal{B}} \mathbb{1}\bigl[\max_t \|\mathbf{F}_i^{(t)}\| > F_{thresh}\bigr]$$

where $\mathcal{B}$ is the set of forbidden body indices and the max is over the contact force history buffer.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `threshold` ($F_{thresh}$) | Force threshold to count as contact (N) | 1.0 |
| `sensor_cfg.body_names` | Bodies to monitor | `".*_Shank"`, `".*_Thigh"` |

**Typical weight:** $-1.0$

A discrete count of how many forbidden bodies are touching the ground.
Combined with `body_contact` termination (which ends episodes on base/hip contact), this creates a spectrum: mild body parts get a per-step penalty, severe contacts terminate the episode entirely.
The count-based (rather than force-based) formulation means even a light brush is penalized equally to a hard collision.

*Built-in: `isaaclab/envs/mdp/rewards.py`.*

---

### E. Gait Quality

#### `gait_sync`

Enforce a trot gait by rewarding synchronized and anti-synchronized foot pairs.

For **synced pairs** (diagonal: LF+RH, RF+LH in a trot):

$$r_{sync} = \exp\!\left(-\frac{(t^{air}_{f_0} - t^{air}_{f_1})^2 + (t^{contact}_{f_0} - t^{contact}_{f_1})^2}{\sigma}\right)$$

For **async pairs** (e.g., LF vs RF — should be in opposite phases):

$$r_{async} = \exp\!\left(-\frac{(t^{air}_{f_0} - t^{contact}_{f_1})^2 + (t^{contact}_{f_0} - t^{air}_{f_1})^2}{\sigma}\right)$$

Total reward is the product of all sync and async terms, gated by velocity command:

$$r = r_{sync,0} \cdot r_{sync,1} \cdot r_{async,0} \cdot r_{async,1} \cdot r_{async,2} \cdot r_{async,3} \cdot \mathbb{1}[\|\mathbf{v}^{cmd}\| > 0]$$

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `std` ($\sigma$) | Exp kernel tolerance | 0.1 |
| `max_err` | Clamp on timing difference (s) | 0.2 |
| `synced_feet_pair_names` | Which feet should be in sync | (LF,RH), (RF,LH) for trot |

**Typical weight:** +20.0

The high weight reflects that this reward's output is very small (product of 6 exp terms, each $\leq 1$).
The multiplicative structure means all pairs must be synchronized simultaneously — a single out-of-phase pair zeros out the entire reward.
The `max_err` clamp prevents extreme timing differences from producing near-zero exp outputs that would vanish numerically.

*Custom: `velocity/mdp/rewards.py:85`.*

---

#### `air_time_variance`

Penalize variance in air/contact times across all feet.

$$r = \text{Var}\bigl(\text{clip}(t^{air}_{1..4},\; 0.5)\bigr) + \text{Var}\bigl(\text{clip}(t^{contact}_{1..4},\; 0.5)\bigr)$$

Uses `last_air_time` and `last_contact_time` (from the most recently completed phase), clipped to 0.5 s to prevent outliers from dominating.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `sensor_cfg.body_names` | Feet to include | `".*_Foot"` (all 4) |

**Typical weight:** $-1.0$

Encourages all four feet to have similar stepping patterns — prevents the robot from favoring one side or developing asymmetric gaits.
Complements `gait_sync` which enforces specific pair relationships, while this term ensures overall symmetry.
The 0.5 s clip prevents a stuck foot (very long contact time) from overwhelming the variance calculation.

*Custom: `velocity/mdp/rewards.py:197`.*

---

#### `joint_position_penalty`

Penalize deviation from the default joint pose, amplified when standing still.

$$r = \|\mathbf{q} - \mathbf{q}_{default}\| \cdot \begin{cases} k_{stand} & \text{if } \|\mathbf{v}^{cmd}\| = 0 \\ 1 & \text{otherwise} \end{cases}$$

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `stand_still_scale` ($k_{stand}$) | Multiplier when velocity command is zero | 5.0 |
| `velocity_threshold` | Command magnitude below which standing mode activates | 0.5 |

**Typical weight:** $-1.4$

During locomotion, this provides mild regularization toward the default pose, preventing extreme joint configurations.
When the robot is commanded to stand still, the $5\times$ amplification strongly pulls joints to the default position, ensuring a clean neutral stance rather than a random frozen pose.
Uses L2 norm (not squared), so the gradient is constant regardless of deviation magnitude.

**Related — `joint_deviation_l1`** (built-in): Uses L1 norm $\sum_i |q_i - q^{default}_i|$ with no standing amplification.
Provides uniform per-joint gradient regardless of deviation size.
Typically disabled ($w = 0.0$) in favor of the command-gated custom variant.

*Custom: `velocity/mdp/rewards.py:255`. Built-in: `isaaclab/envs/mdp/rewards.py`.*

---

## Part 3: Quick Reference Table

| Name | Category | Kernel | Formula (shorthand) | Typical Weight | Key Parameter |
|------|----------|--------|---------------------|----------------|---------------|
| `track_lin_vel_xy_exp` | Tracking | Exp (Gaussian) | $\exp(-\|\mathbf{e}_{xy}\|^2 / \sigma^2)$ | +1.0 to +2.0 | $\sigma$: 0.25–1.0 |
| `base_linear_velocity_reward` | Tracking | Exp (Laplacian) | $\exp(-\|\mathbf{e}_{xy}\| / \sigma) \cdot s$ | +10.0 | $\sigma$: 1.0, ramp: 0.5 |
| `track_ang_vel_z_exp` | Tracking | Exp (Gaussian) | $\exp(-e_z^2 / \sigma^2)$ | +1.0 | $\sigma$: 0.25–2.0 |
| `base_angular_velocity_reward` | Tracking | Exp (Laplacian) | $\exp(-\|e_z\| / \sigma)$ | +10.0 | $\sigma$: 2.0 |
| `flat_orientation_l2` | Stability | L2 squared | $\|\mathbf{g}_{xy}\|^2$ | $-5.0$ / $0.0$ | flat / rough |
| `base_orientation_penalty` | Stability | L2 norm | $\|\mathbf{g}_{xy}\|$ | $-3.0$ | — |
| `base_height_l2` | Stability | L2 squared | $(h - h_t)^2$ | $-5.0$ | $h_t$: 0.35 m |
| `lin_vel_z_l2` | Stability | L2 squared | $v_z^2$ | $-2.0$ | — |
| `ang_vel_xy_l2` | Stability | L2 squared | $\|\boldsymbol{\omega}_{xy}\|^2$ | $-0.05$ | — |
| `base_motion` | Stability | Mixed | $0.8 v_z^2 + 0.2\|\boldsymbol{\omega}_{xy}\|_1$ | $-2.0$ | — |
| `joint_torques_l2` | Energy | L2 squared | $\sum \tau_i^2$ | $-10^{-4}$ | joint selection |
| `joint_acc_l2` | Energy | L2 squared | $\sum \ddot{q}_i^2$ | $-2.5 \times 10^{-7}$ | joint selection |
| `joint_vel_l2` | Energy | L2 squared | $\sum \dot{q}_i^2$ | $-10^{-2}$ | joint selection |
| `action_rate_l2` | Smoothness | L2 squared | $\sum (a_t - a_{t-1})^2$ | $-0.01$ | — |
| `action_smoothness` | Smoothness | L2 norm | $\|\mathbf{a}_t - \mathbf{a}_{t-1}\|$ | $-1.0$ | — |
| `feet_air_time` | Contact | Threshold | $\sum (t_{air} - t_{th}) \cdot \mathbb{1}_{land}$ | +0.01 to +0.25 | $t_{th}$: 0.4 s |
| `air_time_reward` | Contact | Phase-gated | phase time up to $t_{mode}$ | +5.0 | $t_{mode}$: 0.3 s |
| `foot_clearance` | Contact | Exp + tanh | $\exp(-\sum e_z^2 \tanh(\alpha v) / \sigma)$ | +0.25 to +1.0 | $h_t$: 0.10 m |
| `feet_regulation` | Contact | Exp height gate | $\sum v_{xy}^2 \exp(-z / 0.025h)$ | $-0.15$ | $h$: 0.35 m |
| `foot_slip` | Contact | Binary gate | $\sum \mathbb{1}_{contact} \cdot \|v_{xy}\|$ | $-0.5$ | $F_{th}$: 1.0 N |
| `undesired_contacts` | Contact | Count | $\sum \mathbb{1}[F > F_{th}]$ | $-1.0$ | $F_{th}$: 1.0 N |
| `gait_sync` | Gait | Exp (product) | $\prod \exp(-\Delta t^2 / \sigma)$ | +20.0 | $\sigma$: 0.1 |
| `air_time_variance` | Gait | Variance | $\text{Var}(t_{air}) + \text{Var}(t_{contact})$ | $-1.0$ | clip: 0.5 s |
| `joint_position_penalty` | Gait | L2 norm | $\|\mathbf{q} - \mathbf{q}_0\| \cdot k$ | $-1.4$ | $k_{stand}$: 5.0 |
