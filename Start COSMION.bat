@echo off
setlocal
cd /d "%~dp0"
title COSMION
color 0B

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   COSMION is not installed in this folder.
    echo   Follow the Quickstart in README.md, then try again.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "scripts\one_click.py"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo   COSMION exited with code %EXIT_CODE%.
    echo   See artifacts\one-click-worker.log for details.
    echo.
    pause
)
endlocal
