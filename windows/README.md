# Running the Action Space kit on Windows

The kit (`Action_Space/`) and the SF Compute guide are written for Linux. This folder holds
the Windows equivalents. Tested on Windows 11, RTX 4080 Laptop (12 GB), driver 617.

| Piece | Where it runs | Environment |
|---|---|---|
| Isaac Sim / Isaac Lab (local tests, watching policies) | native Windows | conda env `isaaclab` (Python 3.11), installed into `C:\isaac` |
| SF Compute `autor` CLI (cloud H100 training) | WSL Ubuntu (`autor` has no Windows build) | `~/.local/bin/autor` |
| Drone side: calibration, mapping, single-drone flying | native Windows | conda env `tello` (Python 3.12; the repo pins numpy 2.5.3) |
| Multi-drone flying with USB Wi-Fi dongles | **Linux only** (`nmcli`, `SO_BINDTODEVICE`) | not possible on Windows or WSL |

## 1. Local Isaac Sim (about 30 GB, 30-60 min)

One-time setup in an **admin** PowerShell (Isaac Sim's file paths exceed Windows' 260-character limit):

```powershell
Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled 1
```

Then in a normal PowerShell, from the repo root:

```powershell
New-Item -ItemType Directory -Force C:\isaac | Out-Null
powershell -ExecutionPolicy Bypass -File windows\setup_isaac.ps1 *> C:\isaac\setup.log
Select-String "^== |wrote|isaaclab ok|AS-Tello-Approach|EXIT" C:\isaac\setup.log
```

It passes when the log shows `wrote C:\isaac\assets\tello.usd`, `isaaclab ok, cuda: True`, and the
`AS-*` task table. It's safe to re-run. The `pip's dependency resolver ... packaging` error is harmless.

Every new terminal (the PowerShell equivalent of the Step 2 block in `Action_Space/CLAUDE.md`):

```powershell
conda activate isaaclab
cd Action_Space\simulation_infra
$env:OMNI_KIT_ACCEPT_EULA='YES'; $env:TELLO_USD_PATH='C:\isaac\assets\tello.usd'
python -u scripts\check_approach_env.py --headless --test all
```

Verified on Windows (2026-10-23): `check_approach_env.py` passes all 22 checks (scripted 100%,
team return +1545). `sb3/play.py` on `policies/approach_it5_room_fixed` scores near_base_frac 0.982,
success 0.887. If a run dies with `Unable to open the usd file at path: https://omniverse-content-production...`,
the Wi-Fi dropped while Isaac was fetching a stock asset. Just re-run.

`TELLO_USD_PATH` is required; without it the code looks in `/mnt/data`. Use the commands in
`Action_Space/CLAUDE.md` with `\` paths; drop `setsid nohup ... &` (start a second terminal instead).
12 GB of VRAM fits a few hundred envs. Do real training on SF Compute.

## 2. SF Compute (`sf_compute_quick_guide_sachi.pdf`)

`autor` runs inside WSL. One-time setup:

```powershell
wsl --update            # admin; older WSL (2.3.x) fails to set up Ubuntu 26.04 with "Catastrophic failure"
wsl --install -d Ubuntu
ubuntu                  # first run: create a Linux user
```

Inside Ubuntu:

```bash
curl -fsSL https://autoresearch.sfcompute.com/cli.sh | bash
cat >> ~/.bashrc <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
N=team1
E="--env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env PYTHONUNBUFFERED=1"
EOF
source ~/.bashrc
autor login             # the team account
autor node ls           # team1 should be listed
```

The team node **`team1` already exists**, so skip PDF step 2. It bills every minute it runs:
`autor node stop $N` when done, and tell the team when you wake it. In WSL the repo is at
`/mnt/c/Users/<you>/OneDrive/Desktop/Action Space Hackathon`; run the PDF's `tar` and `autor cp`
commands from there.

## 3. Drone side

```powershell
conda create -y -n tello -c conda-forge --override-channels python=3.12
conda activate tello
pip install -r Action_Space\drone_deployment_infra\cal_and_map\requirements.txt
pip install -e Action_Space\drone_deployment_infra\cal_and_map
```

The docs say `../.venv/bin/python`; use `python` from this env instead. Single-drone scripts
that talk to the Tello over the laptop's own Wi-Fi (djitellopy, 192.168.10.1) should work.
Anything using `core/tello_link.py`, `tello_dongle_setup.py` or `swarm_ops/connect.py` needs Linux.

## Pulling organizer updates

`Action_Space/` is a git subtree of `jaromero3rd/Action_Space`:

```bash
git remote add upstream https://github.com/jaromero3rd/Action_Space.git   # once
git subtree pull --prefix=Action_Space upstream main
```

Also run `git config core.longpaths true` once per clone, because some kit paths are long.
