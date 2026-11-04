@echo off
rem Compatibility alias for the local console launcher.
call "%~dp0StartAI.cmd" %*
exit /b %errorlevel%
