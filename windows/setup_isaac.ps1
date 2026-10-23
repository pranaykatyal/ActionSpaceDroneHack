# Windows port of Action_Space/simulation_infra/setup.sh: Isaac Sim 5.1 + Isaac Lab 2.3.2 +
# the Action Space kit in a Python 3.11 conda env. Safe to re-run.
#
#   powershell -ExecutionPolicy Bypass -File windows\setup_isaac.ps1 [-IsaacRoot C:\isaac] [-EnvName isaaclab]
#
# Needs: NVIDIA driver, Miniconda, git, and Win32 long paths ON (admin, once):
#   Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled 1
# IsaacRoot must be a short path with no spaces and outside OneDrive (~30 GB lands there).
param(
    [string]$IsaacRoot = "C:\isaac",
    [string]$EnvName = "isaaclab",
    [string]$IsaacLabVersion = "v2.3.2",
    [string]$IsaacSimVersion = "5.1.0"
)
$ErrorActionPreference = "Stop"
$KitDir = Resolve-Path "$PSScriptRoot\..\Action_Space\simulation_infra"

function Step($msg) { Write-Host "== $msg" }
function Run { & $args[0] $args[1..($args.Count - 1)]; if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $($args -join ' ')" } }

Step "checking GPU"
Run nvidia-smi "--query-gpu=name,driver_version,memory.total" "--format=csv,noheader"

Step "long paths"
if ((Get-ItemPropertyValue 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled) -ne 1) {
    throw "Win32 long paths are off; enable them as admin (see the header of this script)"
}
git config --global core.longpaths true

Step "workspace: $IsaacRoot"
New-Item -ItemType Directory -Force "$IsaacRoot\assets", "$IsaacRoot\tmp" | Out-Null
$env:PIP_CACHE_DIR = "$IsaacRoot\.cache\pip"
$env:TMP = $env:TEMP = "$IsaacRoot\tmp"
$env:OMNI_KIT_ACCEPT_EULA = "YES"   # NVIDIA Omniverse EULA: https://docs.omniverse.nvidia.com/eula/

Step "python 3.11 conda env: $EnvName"
$conda = (Get-Command conda -ErrorAction Stop).Source
$envPath = Join-Path (& $conda info --base) "envs\$EnvName"
$py = "$envPath\python.exe"
if (-not (Test-Path $py)) { Run $conda create -y -n $EnvName -c conda-forge --override-channels python=3.11 }
# isaaclab.bat resolves python from the active conda env; make this one active for child processes
$env:CONDA_PREFIX = $envPath
$env:PATH = "$envPath;$envPath\Scripts;$envPath\Library\bin;$env:PATH"

Step "PyTorch (CUDA 12.8)"
Run $py -m pip install -q --upgrade pip
Run $py -m pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

Step "Isaac Sim $IsaacSimVersion"
Run $py -m pip install "isaacsim[all,extscache]==$IsaacSimVersion.0" --extra-index-url https://pypi.nvidia.com

Step "Isaac Lab $IsaacLabVersion"
if (-not (Test-Path "$IsaacRoot\IsaacLab")) { Run git clone -q https://github.com/isaac-sim/IsaacLab.git "$IsaacRoot\IsaacLab" }
Push-Location "$IsaacRoot\IsaacLab"
Run git checkout -q $IsaacLabVersion
# same flatdict trap as setup.sh: it fails to build with setuptools >= 81 and isaaclab is skipped
Run $py -m pip install -q "setuptools<81" wheel
Run $py -m pip install -q --no-build-isolation flatdict==4.0.1
Run cmd /d /c "$IsaacRoot\IsaacLab\isaaclab.bat" --install
Run $py -m pip install -q --editable source\isaaclab
Pop-Location

Step "this kit"
Run $py -m pip install -q -e "$KitDir\source\action_space_kit"
Run $py -m pip install -q djitellopy

Step "Tello asset"
$usd = "$IsaacRoot\assets\tello.usd"
if (-not (Test-Path $usd)) {
    Push-Location "$IsaacRoot\IsaacLab"
    Run cmd /d /c "$IsaacRoot\IsaacLab\isaaclab.bat" -p "$IsaacRoot\IsaacLab\scripts\tools\convert_urdf.py" "$KitDir\assets\tello\tello.urdf" $usd --merge-joints --headless
    Pop-Location
}
if (-not (Test-Path $usd)) { throw "tello.usd was not written" }
Write-Host "wrote $usd"

Step "verifying"
Run $py -c "import isaaclab, isaacsim, torch; print('isaaclab ok, cuda:', torch.cuda.is_available())"
Push-Location $KitDir
& $py scripts\list_envs.py 2>$null | Select-String "AS-"
Pop-Location

Write-Host @"

Done. In every new terminal:

  conda activate $EnvName
  cd "$KitDir"
  `$env:OMNI_KIT_ACCEPT_EULA='YES'; `$env:TELLO_USD_PATH='$usd'
  python -u scripts\check_approach_env.py --headless --test all
"@
