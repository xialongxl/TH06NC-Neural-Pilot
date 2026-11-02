$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
Set-Location -LiteralPath $taskRoot
$env:PYTHONUTF8 = '1'
$env:UV_CACHE_DIR = Join-Path $taskRoot '.cache\uv'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv first.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    uv venv --python 3.10 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the project environment.' }
}
& '.venv\Scripts\python.exe' 'tools\fetch_torch_cuda.py'
if ($LASTEXITCODE -ne 0) { throw 'Official CUDA wheel download/verification failed.' }
$taskWheel = Join-Path $taskRoot '.cache\downloads\pytorch-cu130\torch-2.14.1+cu130-cp310-cp310-win_amd64.whl'
uv pip install --python '.venv\Scripts\python.exe' $taskWheel 'numpy==2.2.6' 'pillow==12.3.0' 'mss==10.2.0' 'pefile==2024.8.26' 'capstone==5.0.9' 'nvidia-ml-py==13.615.71' 'gymnasium==1.4.0' 'stable-baselines3==2.9.2' 'tensorboard==2.21.0' 'pytest==9.1.1' 'fastapi==0.162.0' 'uvicorn==0.54.0' 'httpx2==2.14.0'
if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
if (-not (Test-Path -LiteralPath 'config.json')) {
    Copy-Item -LiteralPath 'config.example.json' -Destination 'config.json'
    Write-Output 'Created config.json from the portable example. Set game_exe to your own th06nc.exe before calibration.'
}
Write-Output 'Installed. Launch the local web console with 启动AI.cmd. Existing checkpoints and calibration remain available.'
