# Exploiting Bilateral Symmetry in Legged RL

A reference for the mechanisms used to inject left–right (sagittal) symmetry priors into RL training for legged robots, the papers that motivate them, and the rationale for selecting one as the canonical approach in cf_lab.

**Notation.**
$G = C_2 = \{e, g\}$ is the reflection group ($g$ is the sagittal mirror, $g \circ g = e$).
$\rho_S$, $\rho_A$ are linear representations of $G$ on the state space $S$ and action space $A$, so $g \triangleright s = \rho_S(g)\, s$ and $g \triangleright a = \rho_A(g)\, a$.
$\pi_\theta : S \to A$ is the policy, $V_\phi : S \to \mathbb{R}$ the critic.

---

## 1. The property we want

For a robot whose morphology is sagittally symmetric, the underlying MDP is $G$-symmetric: the transition density is $G$-invariant, the reward function is $G$-invariant, and so the **optimal policy is $G$-equivariant** and the **optimal value function is $G$-invariant**:

$$
\pi^*(g \triangleright s) \;=\; g \triangleright \pi^*(s),
\qquad
V^*(g \triangleright s) \;=\; V^*(s).
$$

Equivariance is the right property — it says "if you mirror the observation (and command), mirror the action."
It is **not** the same as requiring $q_L(t) = \pm q_R(t)$ at every timestep, which is a strictly stronger and generally wrong constraint (it rules out trot, pace, turning, and side-stepping).

A vanilla actor-critic does not get equivariance for free: PPO explores asymmetrically, settles into one of the symmetric modes, and produces limping or asymmetric gaits.
The mechanisms below all aim to inject the equivariance prior into training.

---

## 2. References

### Abdolhosseini et al. 2019 — *On Learning Symmetric Locomotion*

ACM SIGGRAPH Symposium on Computer Animation (SCA).
The foundational comparison.
Studies four mechanisms for inducing symmetric gaits in character animation RL (called DUP, LOSS, NET, PHASE in their nomenclature).
Main empirical finding: **DUP (≈ data augmentation) and NET (equivariant network) consistently outperform LOSS (mirror loss) and PHASE (phase-mirrored rollouts)** on bipedal locomotion benchmarks.

### Mittal et al. 2024 — *Symmetry Considerations for Learning Task Symmetric Robot Policies*

ICRA 2024, arXiv [2403.04359](https://arxiv.org/abs/2403.04359).
Investigates two mechanisms inside PPO: data augmentation of rollouts with mirrored transitions, and an auxiliary mirror loss on the policy.
Provides a theoretical justification for on-policy data augmentation despite the off-policy nature of the augmented samples.
Tested on ANYmal box-climbing and dexterous manipulation.
**Conclusion: data augmentation is the most effective approach** for task-symmetric policies, and improves both convergence speed and final task behavior.

### Su et al. 2024 — *Leveraging Symmetry in RL-based Legged Locomotion Control*

IROS 2024, arXiv [2403.17320](https://arxiv.org/abs/2403.17320).
Compares two PPO variants for quadrupedal loco-manipulation and bipedal locomotion on real hardware (CyberDog2, Go1): **PPOaug** (data augmentation) and **PPOeqic** (strictly equivariant/invariant MLP via EMLP).
Explicitly opts out of the mirror-loss approach, citing Abdolhosseini 2019.
**Findings:** PPOeqic wins in pure-simulation metrics (sample efficiency, task return, symmetry index).
PPOaug is more robust to environment asymmetries (domain randomization, asymmetric actuator dynamics) and transfers more reliably to hardware.

---

## 3. The four strategies

Four mechanisms have been proposed for enforcing $G$-equivariance / invariance in legged RL.
They differ in *where* the symmetry pressure is applied (data, loss, reward, or model) and in *what* property they constrain.

### 3.1 Data augmentation (PPOaug)

**Where:** the rollout buffer feeding the PPO update.

**How:** for every collected transition $(s, a, s', r)$, the symmetric transition $(g \triangleright s,\; g \triangleright a,\; g \triangleright s',\; r)$ is generated on the fly and added to the mini-batch.
Both actor and critic gradients are computed on the union.

$$
\mathcal{L}^{\text{PPO}}_{\text{aug}}(\theta, \phi)
\;=\;
\mathbb{E}_{(s,a) \sim \mathcal{D}}\!\left[\mathcal{L}^{\text{PPO}}(s,a;\theta,\phi)\right]
+
\mathbb{E}_{(s,a) \sim \mathcal{D}}\!\left[\mathcal{L}^{\text{PPO}}(g \triangleright s,\; g \triangleright a;\theta,\phi)\right]
$$

**What it constrains:** $\pi_\theta(g \triangleright s) \approx g \triangleright \pi_\theta(s)$ *indirectly*, by training the policy to fit both halves of every symmetric pair to the same return.

**Strengths.**
- Soft constraint: hypothesis class is unrestricted, so the policy can adapt to environment asymmetries (asymmetric mass distribution, terrain, actuator dynamics).
- No extra hyperparameter beyond the doubled batch size — no loss weight to tune.
- Gradient signal comes from the actual PPO objective on a state the policy genuinely needs to handle.
- Critic and actor benefit jointly (the critic learns approximate $G$-invariance, the actor approximate equivariance).
- Works with asymmetric commands: the command is part of $s$ and is mirrored by $\rho_S(g)$, so a "turn left" rollout supplies a "turn right" sample.

**Weaknesses.**
- Doubles the effective batch size during the PPO update (compute cost, not wall-clock-per-step).
- Approximately on-policy only — Mittal 2024 §III.A addresses this formally and shows it is well-behaved when the policy is initialized near-equivariant (small variance, zero mean).
- Requires a correct, layout-aware permutation/sign transform on observations, actions, *and* critic-side privileged terms.
  Bugs here silently train the policy on a scrambled mirror.

### 3.2 Mirror loss

**Where:** an auxiliary term inside the PPO loss.

**How:** an L2 penalty on the deviation from equivariance, weighted by a scalar $w$:

$$
\mathcal{L}^{\text{sym}}_{g}(\theta)
\;=\;
\mathbb{E}_{s \sim \mathcal{D}}\!\left[\;
\bigl\lVert\, \rho_A(g)\, \pi_\theta(s) \;-\; \pi_\theta\bigl(\rho_S(g)\, s\bigr) \,\bigr\rVert_2^2
\;\right],
\qquad
\mathcal{L}_{\text{total}} = \mathcal{L}^{\text{PPO}} + w \cdot \mathcal{L}^{\text{sym}}_{g}
$$

**What it constrains:** the same equivariance property as data augmentation, but enforced as an explicit regularizer rather than via the data distribution.

**Strengths.**
- Same target property as PPOaug (policy equivariance), so it is *principled* in the same sense.
- Cheap: one extra forward pass through the policy on $g \triangleright s$.

**Weaknesses.**
- Adds a competing gradient: the regularizer gradient can fight the policy gradient, especially early in training.
  Requires careful weight tuning per task.
- Empirically dominated by data augmentation (Abdolhosseini 2019, Mittal 2024, and the reason Su 2024 omits it).
- Does not regularize the critic — Mittal 2024's data-aug variant improves the critic too.
- Can be partially satisfied by degenerate policies (e.g., outputs near the symmetric fixed point), wasting capacity.

### 3.3 Mirror reward (per-timestep)

**Where:** an additional reward term inside the environment, summed into $r_t$.

**How:** a per-timestep L2 penalty on the deviation of mirrored joint pairs from instantaneous mirror equality, with a sign $s_i \in \{+1, -1\}$ depending on the axis convention of each joint pair:

$$
r^{\text{mirror}}_t
\;=\;
-\sum_{i \in \text{pairs}} \bigl(\,q_{L_i}(t) \;+\; s_i \, q_{R_i}(t)\,\bigr)^2
$$

For AYG, $s_i = +1$ for HAA (axes mirrored across legs, true symmetry requires $q_L = -q_R$), $s_i = -1$ for HFE / KFE (axes co-aligned, true symmetry requires $q_L = q_R$).

**What it constrains:** $q_L(t) = \pm q_R(t)$ at every timestep — i.e., the *current state* must be kinematically mirrored.
This is **not** the same property as policy equivariance: it constrains state, not policy.

**Strengths.**
- Trivially cheap (two index lookups and a sum per env per step).
- Easy to layer on top of any algorithm — no PPO modification needed.
- Useful as a sanity-check or last-resort shaping signal.

**Weaknesses.**
- *Wrong target property.* Equivariance does not require the current state to be mirrored; it requires the policy to respond consistently to mirrored states.
  Gaits like trot and pace **require** $q_{LF}(t) \neq q_{RF}(t)$ by construction, so penalizing the difference fights the gait reward.
- Hostile to asymmetric commands: penalizes legitimate HAA divergence during turning, side-stepping, lateral velocity tracking.
- Distorts the optimal value landscape — unlike loss-side or data-side mechanisms, this *changes the problem* the policy is solving.
- Not endorsed by Abdolhosseini 2019, Mittal 2024, or Su 2024.
  Su §VI-D notes reward-side symmetry tuning is the *workaround* vanilla PPO needs precisely because the other mechanisms are unavailable.

### 3.4 Equivariant network (PPOeqic)

**Where:** the architecture of the actor and critic networks.

**How:** the actor MLP is replaced by an equivariant MLP (EMLP) whose weight matrices commute with the group representation, so $\pi_\theta(g \triangleright s) = g \triangleright \pi_\theta(s)$ holds *exactly* by construction.
The critic uses an invariant MLP that maps the regular representation to the trivial representation in its final layer, so $V_\phi(g \triangleright s) = V_\phi(s)$ holds exactly.

**What it constrains:** equivariance is a hard architectural constraint, not a learning signal — the policy *cannot* express non-equivariant functions.

**Strengths.**
- Sample-efficient: Su 2024 reports PPOeqic outperforms PPOaug and vanilla PPO in sim on every benchmarked task.
- Zero runtime overhead at inference (the network is just a structured MLP).
- No hyperparameters beyond ordinary MLP sizing.
- Symmetry guaranteed by construction — no risk of approximation error.

**Weaknesses.**
- *Rigid.* When the environment is not perfectly $G$-symmetric (asymmetric mass, asymmetric actuator dynamics, domain randomization, asymmetric terrain — i.e., almost always on real hardware), the policy cannot compensate.
  Su 2024 §VI-F documents this: PPOeqic loses to PPOaug on bipedal walking when deployed on hardware with non-trivial dynamics mismatch.
- Implementation cost: EMLP layers are non-trivial to build and integrate.
  Requires a library like ESCNN or MorphoSymm and bespoke weight construction.
- Less mature in the locomotion stack: no RSL-RL / SKRL / RL-Games built-in support; cf_lab would have to ship its own EMLP modules.
- Constrains the *function class*, which is harder to debug than a data or loss intervention (a buggy permutation in PPOaug produces a loud divergence; a subtly wrong representation in EMLP can train to completion with a silently sub-optimal policy).

---

## 4. Comparison

| Property                                              | Data aug (PPOaug)        | Mirror loss          | Mirror reward          | Equivariant net (PPOeqic) |
|-------------------------------------------------------|--------------------------|----------------------|------------------------|---------------------------|
| Property enforced                                     | Policy equivariance      | Policy equivariance  | Instantaneous mirror   | Policy equivariance       |
| Hard or soft constraint                               | Soft                     | Soft                 | Soft (via reward)      | Hard (architectural)      |
| Restricts policy class                                | No                       | No                   | No                     | Yes                       |
| Survives asymmetric commands (turn, side-step)        | Yes                      | Yes                  | **No**                 | Yes                       |
| Survives asymmetric gaits (trot, pace, amble)         | Yes                      | Yes                  | **No** (`*_all` term)  | Yes                       |
| Survives asymmetric dynamics (DR, real hardware)      | Yes                      | Yes                  | Yes (but distorts)     | Degrades                  |
| Hyperparameters introduced                            | None                     | Loss weight $w$      | Reward weight per term | None                      |
| Training compute overhead                             | 2× minibatch in PPO step | One extra fwd pass   | Negligible             | Comparable to plain MLP   |
| Inference compute overhead                            | None                     | None                 | None                   | None                      |
| Improves critic too                                   | Yes                      | No                   | N/A                    | Yes                       |
| Empirical ranking in cited literature                 | **1st (tied with NET)**  | 3rd                  | Not studied            | **1st (sim only)**        |
| Implementation maturity in RSL-RL                     | Built-in                 | Built-in             | Manual reward term     | Not supported             |

---

## 5. Why PPOaug is the canonical choice for cf_lab

PPOaug (data augmentation) is the mechanism cf_lab adopts as the default symmetry tool.
The case in short:

1. **It targets the right property.**
   Equivariance, not instantaneous mirror, is the symmetry that the underlying MDP actually has.
   This rules out the mirror-reward approach.

2. **It is the strongest soft constraint.**
   Across Abdolhosseini 2019, Mittal 2024, and Su 2024, data augmentation outperforms the mirror loss every time the two are directly compared.
   The mirror loss is a weaker version of the same idea (regularizer instead of data) with the additional cost of a tuning knob.

3. **It is robust to environment asymmetries.**
   Unlike PPOeqic, which forces an exact equivariant policy and cannot compensate for asymmetric mass, actuator dynamics, or terrain, PPOaug encodes equivariance as a prior that the policy can override when the data says otherwise.
   Su 2024 shows this is the difference between a policy that transfers to hardware and one that does not.

4. **It is cheap to ship.**
   RSL-RL exposes data augmentation behind a single flag (`RslRlSymmetryCfg(use_data_augmentation=True, ...)`) given a `compute_symmetric_states` function.
   There is no new architecture, no new loss, no new reward weight to tune.

5. **It improves the critic at the same time.**
   When the privileged observation group is also mirrored, the value function learns approximate $G$-invariance, which tightens the advantage estimate for both halves of every state pair.
   The mirror-loss and mirror-reward routes do not give this for free.

6. **It composes cleanly with existing rewards.**
   PPOaug does not change the reward function, so it does not perturb any of the carefully tuned tracking, regulation, and penalty terms in [rewards.md](rewards.md).
   The mirror-reward route, in contrast, adds a term that fights gait rewards on trot/pace.

**Caveat — the correctness pre-requisite.**
PPOaug's benefits depend entirely on a correct symmetry transform.
The `compute_symmetric_states` function must permute joints by the actual articulation order, flip signs on the correct subset, mirror the command vector, mirror per-foot critic terms when present, and mirror any spatial observations (height scans).
A bug here trains the policy on a corrupted mirror without any error signal.
cf_lab's velocity implementations guard against this with a runtime joint-order assertion (`_EXPECTED_JOINT_ORDER`); any new layout should follow the same pattern.

**When to consider the alternatives.**
- *Mirror loss* is a defensible second knob if PPOaug-alone leaves visible asymmetry in the learned gait, but expect to sweep $w$ and expect a small marginal gain.
  Set it only with `use_data_augmentation=True, use_mirror_loss=True` — never alone.
- *PPOeqic* is worth investigating for tasks where the simulator is a near-perfect $G$-symmetric MDP (no DR on actuator gains, no terrain asymmetry) and sample efficiency matters more than hardware transfer.
  It would require non-trivial implementation work in cf_lab.
- *Mirror reward* is not recommended for any cf_lab task.
  The `*_all` variant fights trot and pace by construction; the `*_haa` variant penalizes turning.
  Both are dominated by PPOaug.
