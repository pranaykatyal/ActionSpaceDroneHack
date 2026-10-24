# Local environment setup (this machine)

Record of every environment change made to set up `Action_Space` on this machine, why each was
needed, and how to reproduce or revert it.

- **Date:** 2026-10-23
- **Repo:** https://github.com/jaromero3rd/Action_Space @ `cf62a13`, cloned to `/home/shibuwu/Desktop/Action space`
- **Machine:** NVIDIA RTX A5500 24 GB, driver 560.28.03, Ubuntu 22.04.5, no passwordless sudo
- **Repo files modified:** no tracked files changed. This file is new and untracked; `.venv/` and `outputs/` are gitignored.

## Quick start

Simulation (Isaac Sim / Isaac Lab). Run in every new shell:

```bash
cd ~/Action_Space/simulation_infra
conda deactivate 2>/dev/null; unset CONDA_PREFIX CONDA_DEFAULT_ENV
source ~/isaac/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$HOME/isaac/assets/tello.usd
```

Real drones (`drone_deployment_infra`):

```bash
cd ~/Action_Space/drone_deployment_infra
../.venv/bin/python <script>.py
```

## 1. Repo location and symlink

| Change | Why |
|---|---|
| Cloned into `/home/shibuwu/Desktop/Action space` | Working directory chosen by the user |
| Symlink `~/Action_Space` → `/home/shibuwu/Desktop/Action space` | The real path contains a space, which breaks unquoted shell scripts (e.g. `simulation_infra/scripts/run_iter.sh`). The symlink also matches the paths in `README.md` / `CLAUDE.md`. **Always use `~/Action_Space`.** |

## 2. Simulation environment: `~/isaac`

Installed with the repo's own `simulation_infra/setup.sh ~/isaac`. It is completely separate from
the existing Isaac Sim 6.0 install at `~/thesiswork/isaacsim`, which was **not touched**.

### Why not reuse the existing Isaac Sim 6.0

`~/thesiswork/isaacsim` is Isaac Sim **6.0.0-rc.59** (Python 3.12, no Isaac Lab). This repo is
written against **Isaac Lab v2.3.2 + Isaac Sim 5.1 + Python 3.11**:

- Isaac Sim 6.0 requires Isaac Lab 3.0.0-beta2, which dropped Sim 5.1 support and has breaking API
  changes, so the kit would need porting.
- The shipped room/peer policies only load on Python 3.11 (see section 3), and Sim 6.0 bundles 3.12.

### How `setup.sh` was run (three workarounds, no script edits)

```bash
cd ~/Action_Space/simulation_infra
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PYTHON_EXE CONDA_EXE
export CMAKE_POLICY_VERSION_MINIMUM=3.5
sudo() { echo "skip (no passwordless sudo): sudo $*"; }; export -f sudo
./setup.sh ~/isaac > ~/isaac/setup.log 2>&1; echo "EXIT=$?" >> ~/isaac/setup.log
```

| Workaround | Why |
|---|---|
| Unset `CONDA_*` | Conda `base` is active by default; `isaaclab.sh` would use conda's Python instead of the venv |
| `sudo` shell function that only prints | `setup.sh` runs `sudo apt-get install` for `vulkan-tools`, the only missing apt package. Without passwordless sudo, `set -e` would abort the script. `vulkan-tools` is just the `vulkaninfo` diagnostic tool, and `libvulkan1` is already installed. |
| `CMAKE_POLICY_VERSION_MINIMUM=3.5` | `~/.local/bin/cmake` is **CMake 4.3.2** and shadows `/usr/bin/cmake` 3.22. CMake 4 refuses `egl-probe`'s `cmake_minimum_required(VERSION 2.8)` (pulled in via `isaaclab-mimic` → `robomimic`), which failed the Isaac Lab install. This env var makes CMake 4 accept it. Changing PATH doesn't help because `setup.sh` re-prepends `~/.local/bin`. |

Logs: `~/isaac/setup.log` (successful run), `~/isaac/setup.attempt1.log` (stopped by hand),
`~/isaac/setup.attempt2.log` (failed on egl-probe).

### Package changes after `setup.sh`

```bash
uv pip install --python ~/isaac/env_isaaclab/bin/python \
  numpy==2.4.6 opencv-python==5.0.0.93 pupil-apriltags==1.0.4.post11
```

| Package | `setup.sh` left | Changed to | Why |
|---|---|---|---|
| numpy | 1.26.0 | **2.4.6** | The shipped checkpoints (`model_vecnormalize.pkl`, SB3 zips) were pickled under numpy 2. On 1.26, `sb3/play.py` and `record_approach.py` fail with `ModuleNotFoundError: No module named 'numpy._core'` |
| opencv-python | 4.11.0.86 | **5.0.0.93** | Matches the authors' training env |
| pupil-apriltags | — | **1.0.4.post11** | Matches the authors' training env; used by `check_room_render.py` (real localizer on sim renders) |

Target versions come from the authors' recorded training env,
`simulation_infra/policies/peer_qualified_84M/params/run.json` → `dependencies`. All other
packages match it, or differ only at the patch level. To check for drift again:

```bash
cd ~/Action_Space/simulation_infra && ~/isaac/env_isaaclab/bin/python -I -c "
import json, importlib.metadata as md
ref = {k.lower().replace('_','-'): v for k, v in json.load(open('policies/peer_qualified_84M/params/run.json'))['dependencies'].items()}
mine = {d.metadata['Name'].lower().replace('_','-'): d.version for d in md.distributions()}
[print(f'{k:30s} ref={ref[k]:16s} mine={mine.get(k)}') for k in sorted(ref) if mine.get(k) != ref[k]]"
```

### Resulting versions (`~/isaac/env_isaaclab`)

| Package | Version |
|---|---|
| Python | 3.11.0rc1 (system `/usr/bin/python3.11`; see Known issues) |
| isaacsim | 5.1.0.0 |
| Isaac Lab | v2.3.2 checkout at `~/isaac/IsaacLab` (isaaclab 0.54.2, isaaclab_rl 0.4.7, isaaclab_tasks 0.11.12) |
| torch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 (CUDA available) |
| stable_baselines3 / skrl / rl_games / gymnasium | 2.8.0 / 2.1.1 / 1.6.1 / 1.2.1 |
| numpy / opencv-python / pupil-apriltags | 2.4.6 / 5.0.0.93 / 1.0.4.post11 |
| action_space_kit | 0.1.0, editable from `~/Action_Space/simulation_infra/source/action_space_kit` |
| djitellopy | 2.5.0 |

| Path | Size | What |
|---|---|---|
| `~/isaac/env_isaaclab/` | 20 GB | the venv |
| `~/isaac/IsaacLab/` | 220 MB | Isaac Lab v2.3.2 |
| `~/isaac/assets/tello.usd` (+ `configuration/`) | 36 KB | drone model converted from `assets/tello/tello.urdf` |
| `~/isaac/.cache/` | 3.0 GB | pip/uv caches (speeds up reinstalls) |
| `~/isaac/tmp/` | 7.5 GB | `TMPDIR` used during install; safe to delete |

## 3. Drone environment: repo-root `.venv`

`drone_deployment_infra` expects a venv at the repo root (`../.venv`). Built with **uv-managed
Python 3.11.17**:

```bash
cd ~/Action_Space
printf "numpy==2.4.6\nav==18.1.0\n" > /tmp/drone-overrides.txt
uv venv --managed-python --python 3.11 --seed .venv
uv pip install --python .venv/bin/python --override /tmp/drone-overrides.txt \
  -r drone_deployment_infra/cal_and_map/requirements.txt \
  -r drone_deployment_infra/swarm_ops/requirements.txt
uv pip install --python .venv/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python --override /tmp/drone-overrides.txt \
  stable_baselines3==2.8.0 gymnasium==1.2.1 cloudpickle==3.1.2
.venv/bin/pip install -e drone_deployment_infra/cal_and_map
```

### Why Python 3.11, and why the overrides

- **The policies only load on Python 3.11.** SB3 checkpoints that use the residual policy
  (e.g. `peer_qualified_84M`) pickle the `residual_policy` class **by value**, i.e. as Python 3.11
  bytecode. On Python 3.12, `PPO.load` **segfaults** inside `core/residual_policy.py`. This
  affects `core/room_policy.py`, the real-flight inference path.
- The repo pins `numpy==2.5.3` and `av==19.0.1`, which both require Python ≥ 3.12. They are
  overridden to `numpy==2.4.6` and `av==18.1.0`, the versions in the authors' training env.
- `torch` and `stable_baselines3` are not in the repo's requirements files, but `core/room_policy.py`
  and `core/residual_policy.py` import them. A CPU torch build is enough for 4-drone MLP inference.
- uv must be told `--managed-python`; otherwise it picks Ubuntu's `/usr/bin/python3.11`, which is 3.11.0rc1.

### Resulting versions (`.venv`)

Python 3.11.17 · numpy 2.4.6 · opencv-python 5.0.0.93 · av 18.1.0 · pyyaml 6.0.3 ·
pupil-apriltags 1.0.4.post11 · djitellopy 2.5.0 · pytest 9.1.1 · torch 2.7.0+cpu ·
stable_baselines3 2.8.0 · gymnasium 1.2.1 · cloudpickle 3.1.2 · tello_calibration 0.1.0 (editable)

## 4. Not changed

- Existing Isaac Sim 6.0 at `~/thesiswork/isaacsim` and all conda envs (`base`, `isaacsim`, `phase0`, `robocasa`, `stretch_sim`)
- System packages, NVIDIA driver, `~/.local/bin/cmake`, shell profile, PATH
- Any file tracked in the repo

## Verification results

All runs were made from `~/Action_Space/simulation_infra` with the quick-start prefix. Logs are in `simulation_infra/outputs/`.

| Check | Command | Result |
|---|---|---|
| Installer | `setup.sh` | `EXIT=0`, `wrote ~/isaac/assets/tello.usd`, `isaaclab ok, cuda: True`, all `AS-*` tasks registered |
| Approach env (`CLAUDE.md` Step 3) | `scripts/check_approach_env.py --headless --test all` | **22/22 `[ok]`**, 0 FAIL. Scripted pilot 100% success, team return +1542.0; zero actions 0%, −806.8 (reference ~+1540 / ~−800) |
| Shipped policy score (Step 4) | `scripts/sb3/play.py ... --checkpoint policies/approach_it5_room_fixed/model.zip` (+ the 3 `env.*` overrides) | **256 episodes: success 0.898, near_base_frac 0.990**, gates 3.996/4, landed 3.867/4, collision 0.008, illegal entry 0.004, crash 0.000. Passes the ≥ 0.80 bar; reference was success 0.864 / near_base 0.897 |
| Demo video (Step 5) | `scripts/record_approach.py --pilot policy ...` | Wrote `simulation_infra/outputs/videos/approach_policy.mp4` (1.07 MB, 1280×720 @ 20 fps, 31.5 s, 630 frames). Episode 1 SUCCESS (gates 4/4, landed 4/4, 8.6 s); episode 2 "cut at 20 s" (the recorder's cap, not an error). The frame at 7 s shows the floor grid, keep-out sphere, 4 gate rings and touchdown disc, so **rendering works on driver 560** |
| Drone Kalman filter | `.venv/bin/python drone_deployment_infra/core/positional_kalman_filter.py` | `SELFTEST PASSED` |
| Drone policy load | `RoomPolicy('simulation_infra/policies/peer_qualified_84M/model.zip')` in `.venv` | Loads (4 drones, obs 384 = 48 × 8-frame stack, action 5) and returns finite actions |
| Swarm CLI | `drone_deployment_infra/swarm_ops/swarm_policies.py --help` | OK |

## Known issues and follow-ups

1. **NVIDIA driver 560.28.03 is below Isaac Sim 5.1's listed 580.65.06.** Physics, the Tello URDF
   conversion, policy scoring and viewport video capture all work on it. Not yet tested: the GUI
   window, WebRTC livestream, and Replicator camera sensors. If any of those fails with Vulkan/RTX
   errors, upgrade the driver first. This needs sudo and a reboot: `sudo apt install nvidia-driver-580`.
2. **`vulkan-tools` is not installed.** It's optional; to add it: `sudo apt-get install -y vulkan-tools`.
3. **`~/isaac/env_isaaclab` runs on Python 3.11.0rc1**, a release candidate: `setup.sh`'s
   `uv venv --python 3.11` picked the system interpreter. Everything verified works. If an odd
   interpreter bug ever shows up, rebuild the venv with `uv venv --managed-python --python 3.11`.
   That means reinstalling Isaac Sim into it; the cached downloads make it faster.
4. **Upstream repo issues worth reporting:**
   - `simulation_infra/setup.sh` leaves numpy 1.26, so the shipped checkpoints don't load until numpy 2.4.6 is installed.
   - `drone_deployment_infra/cal_and_map/requirements.txt` pins numpy 2.5.3 and av 19.0.1 (Python ≥ 3.12),
     but the shipped residual-policy checkpoints segfault on anything but Python 3.11.
   - Neither drone requirements file lists `torch` / `stable_baselines3`, which `core/room_policy.py` needs.
   - **Deploy laptops will hit the same problems.** Build their `.venv` exactly as in section 3.

## Reverting

```bash
# undo the post-setup package changes in the sim env
uv pip install --python ~/isaac/env_isaaclab/bin/python numpy==1.26.0 opencv-python==4.11.0.86
uv pip uninstall --python ~/isaac/env_isaaclab/bin/python pupil-apriltags

# remove everything this setup added (repo clone itself left in place)
rm -rf ~/isaac                              # ~31 GB: sim venv, Isaac Lab, caches, tello.usd
rm -rf "/home/shibuwu/Desktop/Action space/.venv"
rm ~/Action_Space                           # the symlink only
```
