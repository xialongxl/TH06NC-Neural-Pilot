param([string] $Python)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$launcherCache = Join-Path $taskRoot '.cache\launcher'
$null = New-Item -ItemType Directory -Path $launcherCache -Force
$previousCache = $env:PYINSTALLER_CONFIG_DIR
$env:PYINSTALLER_CONFIG_DIR = Join-Path $launcherCache 'config'
Push-Location -LiteralPath $taskRoot
try {
    $dist = Join-Path $launcherCache 'dist'
    $buildOptions = @(
        '--noconfirm', '--onefile', '--windowed', '--noupx', '--name', 'LaunchAI',
        '--paths', $taskRoot, '--add-data', "${taskRoot}\th06ai\messages.json:th06ai",
        '--distpath', $dist, '--workpath', (Join-Path $launcherCache 'build'),
        '--specpath', $launcherCache, 'tools/launch_console.py'
    )
    if ($Python) {
        & $Python -m PyInstaller @buildOptions
    } else {
        uv tool run --cache-dir (Join-Path $taskRoot '.cache\uv') --python 3.10 --from 'pyinstaller==6.22.2' pyinstaller @buildOptions
    }
    if ($LASTEXITCODE -ne 0) { throw 'Launcher build failed.' }
    Copy-Item -LiteralPath (Join-Path $dist 'LaunchAI.exe') -Destination (Join-Path $taskRoot 'LaunchAI.exe') -Force
    Write-Output 'Built LaunchAI.exe in the project root.'
} finally {
    Pop-Location
    $env:PYINSTALLER_CONFIG_DIR = $previousCache
}
