@echo off
rem Double-click to start the Glitch Hunter dashboard.
rem This window is centred on the screen; the browser opens
rem http://localhost:5000 (also centred) the moment the dashboard is ready.
rem While it runs, the laptop is kept at full speed even on battery, and your
rem own power mode comes back when it stops (see dashboard/desktop.py).
rem To stop the dashboard press Esc in this window (it closes by itself), or
rem press Esc on the dashboard page and choose Yes. Ctrl+C and the window's X
rem work too.

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

venv_gpu\Scripts\python.exe -m dashboard.desktop center-console
start "" venv_gpu\Scripts\pythonw.exe -m dashboard.desktop open-dashboard
venv_gpu\Scripts\python.exe app.py --desktop %*
rem A normal stop (Esc, Ctrl+C, Stop on the page) closes this window.
if not errorlevel 1 exit /b 0
rem A stop with an error: say in plain words what is wrong (a library that will
rem not load, for example Windows blocking PyTorch), instead of leaving only a
rem traceback. Silent when everything loads (tools\check_environment.py).
venv_gpu\Scripts\python.exe tools\check_environment.py

echo.
echo The dashboard stopped with an error (see above).
pause
