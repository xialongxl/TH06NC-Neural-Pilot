@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Please run install.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m th06ai open %*
if errorlevel 1 (
  pause
  exit /b 1
)
exit /b 0
