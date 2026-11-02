@echo off
chcp 65001 >nul
rem Compatibility alias. The normal launcher starts both the API and the web UI.
call "%~dp0启动AI.cmd" %*
