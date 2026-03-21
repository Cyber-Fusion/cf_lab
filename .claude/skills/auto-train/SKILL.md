---
name: auto-train
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
4. Ensure directories: `mkdir -p experiments/.scratch`
5. Read the task's env config to fully understand all reward terms, weights, observations, terminations, curriculum, and PPO hyperparams before making any changes. You must know the full reward structure as an RL expert before touching anything.

## Levels

**Level 1 — Reward Tuning:** Change reward weights and PPO hyperparams via JSON overrides. No source files edited. Safe and reversible.

**Level 2 — Overall Training Tuning:** Full autonomy — add/remove rewards, terminations, observations, write new reward functions, staged curriculum. Source config files are edited directly on the current branch.

## How it works

1. Claude reads the task's env config to discover all reward terms, weights, observations, terminations, and PPO hyperparams
2. Writes override JSON to `experiments/.scratch/`
3. Launches `scripts/auto_train/run_phase.py` as a detached process via `nohup setsid` (survives any timeout)
4. Polls `experiments/.scratch/current_phase_report.json` every 120s until completion
5. Reads full `phase_report.json` (metrics) + PNG frames (visual) when done
6. Reasons as an expert RL engineer: analyzes reward convergence, per-term breakdown, visual gait quality
7. Decides what to change (Claude chooses iteration count, abort criteria, param adjustments)
8. Logs everything in `experiments/<name>/journal.md`
9. Repeats until satisfied, then runs a final production training with high iterations
10. Bakes winning weights into source config

## Auto-Train Loop

For each iteration:

1. **Reason** — Based on previous results (or initial analysis for iteration 1), decide what to change and why. Think like an expert RL engineer.
2. **Write overrides** — Save override JSON to `experiments/.scratch/`.
3. **Run phase** — Launch `run_phase.py` as a detached process so it survives any timeout. Use this exact pattern:
   ```bash
   nohup setsid .venv/bin/python scripts/auto_train/run_phase.py \
     --task=<TASK> --max-iterations=<N> --num-envs=<N> --headless \
     --overrides-file=experiments/.scratch/<file>.json \
     --report-path=experiments/.scratch/current_phase_report.json \
     > experiments/.scratch/current_phase.log 2>&1 &
   ```
   **IMPORTANT:** Always use `.venv/bin/python`, never bare `python`. Each Bash call starts a fresh shell — venv activation from earlier steps does not persist. The `.venv/bin/python` ensures the correct Python (with Isaac Lab, tbparse, etc.) is used. `run_phase.py` uses `sys.executable` for all subprocesses (training, play, metrics), so the venv propagates automatically.
4. **Poll for completion** — Check the report file every 120 seconds until it no longer says `"status": "running"`. Use this exact pattern (the timeout must exceed the sleep duration):
   ```bash
   sleep 115 && cat experiments/.scratch/current_phase_report.json
   ```
   Use `timeout: 180000` on the Bash tool call (3 minutes, to cover the 115s sleep + read time). Do NOT use `run_in_background` for polling.
   - When the status changes to `"completed"`, `"early_stopped"`, `"failed"`, or `"crashed"`, proceed to analysis.
   - The report file contains a `"pid"` field (written at launch). If the file still says `"running"` but the process is dead (`kill -0 <PID>` fails), treat it as a crash.
   - If the report file doesn't exist yet, wait — the process may still be starting up.
5. **Analyze** — Read the full `phase_report.json` (from the `log_dir` field in the report) for detailed metrics. Read extracted PNG frames to visually assess gait quality. This is critical — metrics alone cannot catch reward hacking.
6. **Log** — Append iteration entry to `experiments/<name>/journal.md` with: Goal, Why, Changes, Result, Key metrics, Visual assessment, Conclusion, Log dir.
7. **Repeat** — Go to step 1 with new insights.

## User controls during auto-train

- `stop` — stop after current phase
- `level 1` / `level 2` — switch privilege level mid-session
- `focus on X` — redirect Claude's priority (e.g., "focus on foot clearance")

## Journal format

Each auto-train session logs to `experiments/<experiment_name>/journal.md`. The journal is the single source of truth for what was tried, what happened, and what to do next — especially important if a conversation disconnects.

**Header** (written once at the top):
```
# Auto-Train: <task_name>
Started: <date>  |  Level: <1|2>  |  Device: <device_info>
```

**Per-iteration entry** (appended after each phase):
```
## Iteration N — <short goal>

**Goal:** What this iteration is trying to achieve
**Why:** Reasoning for this change based on previous results
**Changes:** Parameter changes (old → new) or config edits
**Result:** success / abort / error
**Key metrics:** Mean reward, key reward terms, episode length (final values + trends)
**Visual assessment:** Gait quality observations from frame analysis
**Conclusion:** What worked, what didn't, what to try next
**Log dir:** `logs/rsl_rl/.../<timestamp>/`
```

**Resuming after disconnect:** Start a new conversation and say: *"Continue auto-train, read journal at `experiments/<name>/journal.md`"*. Claude will read the journal, pick up from the last iteration, and continue.

## Scripts

All scripts are in `scripts/auto_train/`:
- `run_phase.py` — orchestrator: train → metrics → play → frames → report
- `train_with_overrides.py` — modified train.py with `--overrides-file` JSON support
- `analyze_metrics.py` — TensorBoard events → JSON (uses tbparse)
- `extract_frames.py` — MP4 → PNG frames for visual inspection (uses OpenCV)

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
- **Journal everything** — If the conversation disconnects, the journal is the only way to resume.
- **Final production run** — When satisfied with tuning, run a long final training and bake winning params into source config.
- **Level 1** — Only JSON overrides (reward weights, PPO hyperparams). No source file edits.
- **Level 2** — Full autonomy: can edit source configs, add/remove rewards, write new reward functions.

## Unattended Operation

Auto-train is designed to run unattended (e.g., overnight with `claude --dangerously-skip-permissions`). These rules ensure it never hangs or wastes compute:

- **NEVER use AskUserQuestion** — There is no human watching. If you are uncertain, make a reasonable decision and log your reasoning in the journal. Never block waiting for user input.
- **Max 15 tuning iterations** — After 15 iterations (not counting the final production run), stop the loop. Log a summary in the journal explaining the best result found and what the final production run should use. Then run the final production training.
- **Abort after 3 consecutive failures** — If 3 iterations in a row produce `status: "failed"` or `status: "error"` in the phase report, stop the loop. Log the failure pattern and suspected root cause in the journal. If at Level 2, check whether your source edits introduced a syntax or import error before retrying.
- **Detached launch** — Always launch `run_phase.py` via `nohup setsid ... &` (as shown in the Auto-Train Loop section). NEVER use `run_in_background: true` — it has a timeout that kills long training runs. The `nohup setsid` approach fully detaches the process. Poll `--report-path` for completion.
- **Disk awareness** — Each iteration produces checkpoints (~50-200 MB) and videos. If running many iterations, note cumulative disk usage in the journal. If `df -h` shows <10 GB free on the training partition at any check, stop and log a warning.
- **Level 2 safety** — Before starting a new training after editing source files, verify the edit is syntactically valid: `.venv/bin/python -c "import ast; ast.parse(open('<edited_file>').read()); print('syntax OK')"`. If this fails, fix the error before training. This prevents wasting an entire training run on a broken file. Note: do NOT use `import cf_lab.tasks` as a check — it requires Isaac Sim runtime and will always fail in a plain shell.

## Environment Variables (always set before training)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Porting to other branches/devices

The auto-train system is self-contained. To enable it on any branch/device:
1. Copy `scripts/auto_train/` folder into that branch's `scripts/`
2. Copy `.claude/skills/auto-train/` folder into that branch's `.claude/skills/`

That's it. The skill handles setup (dependencies, directories) automatically on first run.

---

Begin now. Create the experiment folder and journal, analyze the task, and start iteration 1.
