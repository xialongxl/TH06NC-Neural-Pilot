$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
Set-Location -LiteralPath (Join-Path $taskRoot 'frontend')
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'Node.js 22.12+ or 24 LTS is required to build the frontend.' }
npm.cmd ci --cache (Join-Path $taskRoot '.cache\npm') --no-fund --no-audit
if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
npm.cmd run build
if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
Write-Output 'Web console built. Run tools/build-launcher.ps1 to create LaunchAI.exe if needed.'
