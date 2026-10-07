@echo off
rem Double-click to start the Glitch Hunter dashboard.
rem This window is centred on the screen; the browser opens
rem http://localhost:5000 (also centred) the moment the dashboard is ready.
rem While it runs, the laptop is kept at full speed even on battery, and your
rem own power mode comes back when it stops (see desktop.py).
rem Close this window (or press Ctrl+C in it) to stop the dashboard.

cd /d "%~dp0"
title Glitch Hunter dashboard
rem No __pycache__ folders in the project (see common/__init__.py).
set "PYTHONPYCACHEPREFIX=%~dp0generated\cache\pycache"

if not exist "venv_gpu\Scripts\python.exe" (
    echo venv_gpu was not found in %CD%.
    echo Create it first - see "Installation" in README.md.
    pause
    exit /b 1
)

venv_gpu\Scripts\python.exe desktop.py center-console
start "" venv_gpu\Scripts\pythonw.exe desktop.py open-dashboard
venv_gpu\Scripts\python.exe app.py --desktop %*
rem A stop with an error: say in plain words what is wrong (a library that will
rem not load, for example Windows blocking PyTorch), instead of leaving only a
rem traceback. Silent when everything loads (tools\check_environment.py).
if errorlevel 1 venv_gpu\Scripts\python.exe tools\check_environment.py

echo.
echo The dashboard has stopped.
pause
