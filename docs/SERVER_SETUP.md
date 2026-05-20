# Rented GPU Server Setup (vast.ai-style)

How to provision a vast.ai (or similar) instance to run cf_lab training without
the Isaac Sim 5.1 RTX renderer crashing.

## The crash you're avoiding

Symptom inside the container:

```
[Error] [carb] [Plugin: libomni.hydra.rtx.plugin.so] could not load.
*** Error in `python`: free(): invalid pointer in librtx.scenedb.plugin.so
UsdManager::createHydraEngine -> SIGABRT
```

Root cause: Isaac Sim **5.1** ships RTX renderer plugins that link against
glibc from Ubuntu 22.04. vast.ai's default template (and many community
templates) are Ubuntu **24.04** with glibc 2.39, which breaks the bundled
plugins. The crash happens before the first env.step(); training never starts.

## The fix: Ubuntu 22.04 base image

Pick one of the two paths below when creating the vast.ai instance template.
**B is recommended** — fewer moving parts, no image rebuild.

### Option A — rebuild cf_lab's Dockerfile

Use cf_lab's `docker/Dockerfile` (base `nvcr.io/nvidia/isaac-lab:2.3.2`,
already Ubuntu 22.04). Build locally, push to your registry, point vast.ai at it.

```bash
# local
cd cf_lab
make build                                            # builds the cf-lab image
docker tag cf-lab:latest <your-registry>/cf-lab:latest
docker push <your-registry>/cf-lab:latest
# vast.ai instance template -> image: <your-registry>/cf-lab:latest
```

Pros: same image as your local container, reproducible. Cons: ~1–2 hours
(image build + push), and the Dockerfile uses isaac-lab:2.3.2 which lags
the local pip install (isaaclab 0.54.3, isaacsim 5.1.0.0 as of 2026-05-19).

### Option B — provision Ubuntu 22.04 + install cf_lab per-instance (recommended)

Pick the official NVIDIA Isaac Sim image as the vast.ai template:

```
nvcr.io/nvidia/isaac-sim:5.1.0
```

NVIDIA's image is Ubuntu 22.04 + Isaac Sim 5.1 + CUDA, all matched. cf_lab
is small — install it per-instance after `deploy.sh` rsyncs the tree.

#### One-time per instance

```bash
# inside the running container, as root
apt-get update && apt-get install -y rsync ssh
cd /workspace/cf_lab     # populated by scripts/server/deploy.sh
python -m venv .venv && source .venv/bin/activate
# cf_lab pulls the right Isaac Lab + Isaac Sim versions transitively
uv pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com  # adjust if you've bumped versions locally
uv pip install -e source/cf_lab
uv pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then immediately run the smoke test to confirm the RTX renderer loads:

```bash
scripts/server/smoke_test.sh
```

Expected output (the obs-shape line is the critical one):

```
[CHECK] policy dim=36045 (expected 36045 = 45 ego + 36000 depth)
[CHECK] teacher dim=235 (expected 235 = 48 ego + 187 height_scan rays)
[CHECK] depth.shape=(1, 45, 80, 1) dtype=torch.float32
[CHECK] non-zero pixels: ~1600/3600
[OK] wrote /tmp/d555_smoke.png
```

If you get a SIGABRT inside `librtx.scenedb.plugin.so`, the template was
not Ubuntu 22.04. Recreate the instance with a different template.

## Day-to-day workflow

From your laptop:

```bash
# 1. Push cf_lab to the server (rerun whenever code changes)
scripts/server/deploy.sh --port <vast-ssh-port>

# 2. ssh in and run the training
ssh -p <vast-ssh-port> root@213.181.123.15
cd /workspace/cf_lab && source .venv/bin/activate
scripts/server/run_tilt30_training.sh --max_iterations 2000 --run_name tilt30_first

# 3. Pull logs back when training is done
scripts/server/sync_logs_back.sh --port <vast-ssh-port>
```

For TensorBoard while training is live:

```bash
ssh -p <vast-ssh-port> -L 8080:localhost:8080 root@213.181.123.15
# inside:
tensorboard --logdir logs/rsl_rl/ayg_rough --port 8080
# then on laptop open http://localhost:8080
```

## When ports / instances change

vast.ai SSH ports are ephemeral. Recreate instance → new port (sometimes
new IP). Verify on the vast.ai dashboard before deploying. The memory at
`.claude/.../reference_gpu_training_server.md` captures the most recent
known port; trust the dashboard over the memory.

## Issue #16 branch matrix

| Branch | Purpose | Task ID |
|---|---|---|
| `16-improve-rough-terrain` | Long deployment-level training (final) | `Isaac-Velocity-Rough-Ayg-v0` (teacher) |
| `16-rough-student-blind` | Blind baseline (no depth, proprio only) | `Isaac-Velocity-Rough-Ayg-Student-Blind-v0` |
| `16-rough-student-vision-tilt30-f10` | Vision student, 30° pitch-down, 10-frame stack | `Isaac-Velocity-Rough-Ayg-Student-v0` |

After camera-position experiments converge on a winner, merge the chosen
variant into `16-improve-rough-terrain` for the long final run.
