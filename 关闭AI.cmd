@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Please run install.ps1 first.
  pause
  exit /b 1
)
echo 正在关闭 AI；训练会先保存，请等待完成。
".venv\Scripts\python.exe" -m th06ai shutdown %*
if errorlevel 1 (
  pause
  exit /b 1
)
timeout /t 3 >nul 2>nul
exit /b 0
