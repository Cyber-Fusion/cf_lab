---
name: auto_train
description: Start an automated RL training loop for the AYG quadruped robot. Trains, evaluates visually, tunes, and repeats overnight.
argument-hint: <task_name> level <1|2> on <device_info> [optional notes]
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Agent
model: opus
effort: high
---

# Auto-Train Mode

You are an expert RL engineer running an automated training loop for the AYG quadruped robot.

**Session:** Auto-train `$ARGUMENTS`

## Setup

1. Parse the arguments: extract task name, level (1 or 2), device info, and any additional notes. The user may include domain knowledge, known issues, focus areas, or hints (e.g., "robot tends to spider-walk", "previous best was 15.0", "focus on foot clearance"). Use these notes to guide your initial analysis and tuning strategy.
2. `cd cf_lab` — all commands run from the cf_lab directory.
3. Ensure dependencies: `source .venv/bin/activate && uv pip install tbparse opencv-python-headless`
4. Ensure directories: `mkdir -p .claude/skills/auto_train/experiments/.scratch`
5. Read the task's env config to fully understand all reward terms, weights, observations, terminations, curriculum, and PPO hyperparams before making any changes. You must know the full reward structure as an RL expert before touching anything.

## Levels

**Level 1 — Reward Tuning:** Change reward weights and PPO hyperparams via JSON overrides. No source files edited. Safe and reversible.

**Level 2 — Overall Training Tuning:** Full autonomy — add/remove rewards, terminations, observations, write new reward functions, staged curriculum. Source config files are edited directly on the current branch.

## Override JSON Format

Override files use **flat dot-paths** as keys. The values are applied directly to the `@configclass` env config or agent config at runtime.

**Correct format:**
```json
{
  "rewards.track_lin_vel_xy_exp.weight": 2.0,
  "feet_air_time_reward_scale": 4.0,
  "agent.learning_rate": 0.0003,
  "agent.entropy_coef": 0.008,
  "agent.max_iterations": 2000
}
```

**Rules:**
- Keys without `agent.` prefix → applied to env config
- Keys with `agent.` prefix → prefix is stripped, then applied to agent config
- **Do NOT nest:** `{"env_cfg": {"key": val}}` is WRONG — use `{"key": val}` directly
- **Do NOT use full path prefixes:** `{"rewards.gait.weight": 30.0}` is correct, `{"env_cfg.rewards.gait.weight": 30.0}` is wrong
- Values are type-cast automatically to match the existing config type (bool, int, float, list, etc.)
- `run_phase.py` validates the JSON format before launching training — nested dicts or invalid JSON are caught immediately

**Resume format:**
- `--resume-from` expects just the **run folder name** (e.g., `2026-03-21_01-52-22`), NOT a full path like `logs/rsl_rl/experiment/2026-03-21_01-52-22`

## How it works

1. Claude reads the task's env config to discover all reward terms, weights, observations, terminations, and PPO hyperparams
2. Writes override JSON to `.claude/skills/auto_train/experiments/.scratch/`
3. Launches `.claude/skills/auto_train/resources/run_phase.py` as a detached process via `nohup setsid` (survives any timeout)
4. Waits for completion using `wait_for_phase.py` (blocks until done, prints final report)
5. Reads full `phase_report.json` (metrics) + PNG frames (visual) when done
6. Reasons as an expert RL engineer: analyzes reward convergence, per-term breakdown, visual gait quality
7. Decides what to change (Claude chooses iteration count, abort criteria, param adjustments)
8. Logs everything in `.claude/skills/auto_train/experiments/<name>/journal.md`
9. Repeats until satisfied, then runs a final production training with high iterations
10. Bakes winning weights into source config

## Pre-Training Reward Analysis (MANDATORY before iteration 1)

Before starting any training, you MUST analyze each reward term mathematically:

1. **Read every reward term's formula** from the env config (weight, function, parameters)
2. **Compute theoretical max/min** for each term given robot physical params:
   - Robot dimensions: ~0.3m leg length, ~10kg mass, designed for ~1 m/s walking speed
   - Consider: at target velocity, what is the realistic swing phase duration? foot travel distance? joint velocities?
3. **Check threshold feasibility** — Example: if `feet_air_time` threshold is 0.5s and the robot walks at 1 m/s, each foot would need to travel ~0.5m in the air. For a robot with 0.3m legs, this is physically unrealistic.
4. **Identify terms likely to be negative/zero** and explain why
5. **Log this analysis** in the journal before iteration 1

This analysis prevents wasting iterations on reward terms that are fundamentally misconfigured.

## Auto-Train Loop

For each iteration:

1. **Reason** — Based on previous results (or initial analysis for iteration 1), decide what to change and why. Think like an expert RL engineer. **Change ONE variable per iteration** (or a clearly independent group). State your hypothesis and which variable is being tested. Exception: iteration 1 can batch obvious fixes identified in the pre-training analysis.
2. **Write overrides** — Save override JSON to `.claude/skills/auto_train/experiments/.scratch/`.
3. **Run phase** — Launch `run_phase.py` as a detached process so it survives any timeout. Use this exact pattern:
   ```bash
   nohup setsid .venv/bin/python .claude/skills/auto_train/resources/run_phase.py \
     --task=<TASK> --max-iterations=<N> --num-envs=<N> --headless \
     --overrides-file=.claude/skills/auto_train/experiments/.scratch/<file>.json \
     --report-path=.claude/skills/auto_train/experiments/.scratch/current_phase_report.json \
     --monitor-interval=60 --abort-min-reward-at=300:-10 --abort-plateau-patience=500 \
     > .claude/skills/auto_train/experiments/.scratch/current_phase.log 2>&1 &
   ```
   **IMPORTANT:** Always use `.venv/bin/python`, never bare `python`. Each Bash call starts a fresh shell — venv activation from earlier steps does not persist. The `.venv/bin/python` ensures the correct Python (with Isaac Lab, tbparse, etc.) is used. `run_phase.py` uses `sys.executable` for all subprocesses (training, play, metrics), so the venv propagates automatically.

   **Abort criteria:** Always include `--monitor-interval=60` and at least `--abort-min-reward-at=300:-10` for tuning runs. This catches obviously broken runs early and saves compute. For production runs, use looser thresholds or omit abort criteria.

4. **Wait for completion** — Use the blocking wait script instead of polling:
   ```bash
   .venv/bin/python .claude/skills/auto_train/resources/wait_for_phase.py \
     --report-path=.claude/skills/auto_train/experiments/.scratch/current_phase_report.json \
     --poll-interval=30 --timeout=7200
   ```
   Use `timeout: 7500000` on the Bash tool call (slightly over 2 hours). The script blocks internally and prints progress to stderr. When training completes, it prints the full report JSON to stdout and exits.

   **First check at 60s:** Before using `wait_for_phase.py`, do one quick check after 60 seconds to catch fast failures (validation errors, import errors, OOM):
   ```bash
   sleep 55 && cat .claude/skills/auto_train/experiments/.scratch/current_phase_report.json
   ```
   If status is `"validation_error"`, `"crashed"`, or `"failed"`, skip the wait and proceed to analysis. If `"running"`, use `wait_for_phase.py` for the rest.

   **Terminal statuses:** `"completed"`, `"early_stopped"`, `"failed"`, `"crashed"`, `"validation_error"` — proceed to analysis.

   **Dead process detection:** The report file contains a `"pid"` field. If the file says `"running"` but `kill -0 <PID>` fails, treat it as a crash.

5. **Analyze** — Read the full `phase_report.json` (from the `log_dir` field in the report) for detailed metrics. Read extracted PNG frames to visually assess gait quality. This is critical — metrics alone cannot catch reward hacking.
6. **Log** — Append iteration entry to `.claude/skills/auto_train/experiments/<name>/journal.md` with: Goal, Hypothesis, Variable under test, Changes, Result, Key metrics, Visual assessment, Conclusion, Log dir.
7. **Repeat** — Go to step 1 with new insights.

## User controls during auto-train

- `stop` — stop after current phase
- `level 1` / `level 2` — switch privilege level mid-session
- `focus on X` — redirect Claude's priority (e.g., "focus on foot clearance")

## Journal format

Each auto-train session logs to `.claude/skills/auto_train/experiments/<experiment_name>/journal.md`. The journal is the single source of truth for what was tried, what happened, and what to do next — especially important if a conversation disconnects.

**Header** (written once at the top):
```
# Auto-Train: <task_name>
Started: <date>  |  Level: <1|2>  |  Device: <device_info>
```

**Per-iteration entry** (appended after each phase):
```
## Iteration N — <short goal>

**Hypothesis:** What we expect to happen and why
**Variable under test:** The single parameter/group being changed
**Changes:** Parameter changes (old → new) or config edits
**Result:** success / abort / error
**Key metrics:** Mean reward, key reward terms, episode length (final values + trends)
**Visual assessment:** Gait quality observations from frame analysis
**Conclusion:** What worked, what didn't, what to try next
**Log dir:** `logs/rsl_rl/.../<timestamp>/`
```

**Resuming after disconnect:** Start a new conversation and say: *"Continue auto-train, read journal at `.claude/skills/auto_train/experiments/<name>/journal.md`"*. Claude will read the journal, pick up from the last iteration, and continue.

## Scripts

All scripts are in `.claude/skills/auto_train/resources/`:
- `run_phase.py` — orchestrator: train → metrics → play → frames → report
- `train_with_overrides.py` — modified train.py with `--overrides-file` JSON support
- `analyze_metrics.py` — TensorBoard events → JSON (uses tbparse)
- `extract_frames.py` — MP4 → PNG frames for visual inspection (uses OpenCV)
- `wait_for_phase.py` — blocks until training completes, prints final report (replaces sleep-poll loop)

## Device scaling

Use GPU model and VRAM to choose appropriate `num_envs` and iteration counts:
- Laptop (RTX 3060 12GB): ~2048 envs
- Desktop (RTX 4090 24GB): ~4096 envs
- Cloud (A100 80GB): ~4096-8192 envs

## Key Rules

- **Always visually inspect** — The robot can maximize reward while crawling or spider-walking. Look at the frames every iteration.
- **Never skip play** — Do not use `--skip-play`.
- **Scale to device** — Use the device info to choose `num_envs` and iteration counts appropriately.
- **Start conservative** — First iteration should be a baseline or small change to understand the current state.
- **One variable per iteration** — Change one thing at a time so you can attribute results. State your hypothesis in the journal.
- **Pre-analyze rewards** — Mathematically verify reward terms are feasible before training (see Pre-Training Reward Analysis).
- **Journal everything** — If the conversation disconnects, the journal is the only way to resume.
- **Final production run** — When satisfied with tuning, run a long final training and bake winning params into source config.
- **Level 1** — Only JSON overrides (reward weights, PPO hyperparams). No source file edits.
- **Level 2** — Full autonomy: can edit source configs, add/remove rewards, write new reward functions.

## Unattended Operation

Auto-train is designed to run unattended (e.g., overnight with `claude --dangerously-skip-permissions`). These rules ensure it never hangs or wastes compute:

- **NEVER use AskUserQuestion** — There is no human watching. If you are uncertain, make a reasonable decision and log your reasoning in the journal. Never block waiting for user input.
- **Max 15 tuning iterations** — After 15 iterations (not counting the final production run), stop the loop. Log a summary in the journal explaining the best result found and what the final production run should use. Then run the final production training.
- **Abort after 3 consecutive failures** — If 3 iterations in a row produce `status: "failed"` or `status: "error"` in the phase report, stop the loop. Log the failure pattern and suspected root cause in the journal. If at Level 2, check whether your source edits introduced a syntax or import error before retrying.
- **Detached launch** — Always launch `run_phase.py` via `nohup setsid ... &` (as shown in the Auto-Train Loop section). NEVER use `run_in_background: true` — it has a timeout that kills long training runs. The `nohup setsid` approach fully detaches the process.
- **Disk awareness** — Each iteration produces checkpoints (~50-200 MB) and videos. If running many iterations, note cumulative disk usage in the journal. If `df -h` shows <10 GB free on the training partition at any check, stop and log a warning.
- **Level 2 safety** — Before starting a new training after editing source files, verify the edit is syntactically valid: `.venv/bin/python -c "import ast; ast.parse(open('<edited_file>').read()); print('syntax OK')"`. If this fails, fix the error before training. This prevents wasting an entire training run on a broken file. Note: do NOT use `import cf_lab.tasks` as a check — it requires Isaac Sim runtime and will always fail in a plain shell.

## Environment Variables (always set before training)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Porting to other branches/devices

The auto-train system is self-contained in a single folder. To enable it on any branch/device:
1. Copy `.claude/skills/auto_train/` folder into that branch's `.claude/skills/`

That's it. The skill handles setup (dependencies, directories) automatically on first run.

---

Begin now. Create the experiment folder and journal, run the pre-training reward analysis, and start iteration 1.
