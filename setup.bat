@echo off
rem ====================================================================
rem  Glitch Hunter - one-click setup for Windows.
rem  Double-click this file. It installs everything, then opens the
rem  dashboard. Safe to run again: it only adds what is missing.
rem
rem    setup.bat          install (small CPU build of PyTorch), then run
rem    setup.bat gpu      install the NVIDIA GPU build instead (for training)
rem    setup.bat norun    install only, do not start the dashboard (CI uses it)
rem ====================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title Glitch Hunter setup

set "TORCH=torch==2.14.0"
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "RUN=1"
for %%A in (%*) do (
    if /i "%%A"=="gpu" set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
    if /i "%%A"=="norun" set "RUN=0"
)

echo.
echo  [1/5] Looking for Python 3.12 ...
call :find_python
if not defined PY (
    echo        Not found. Installing Python 3.12 with winget ...
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    call :find_python
)
if not defined PY (
    echo.
    echo  Python 3.12 could not be found or installed automatically.
    echo  Install it from https://www.python.org/downloads/release/python-3120/
    echo  ^(tick "Add python.exe to PATH"^), then double-click setup.bat again.
    goto :fail
)
echo        Using %PY%
rem Windows limits a file path to 260 characters, and PyTorch's files sit deep
rem inside venv_gpu: a long project path makes the install fail half-way.
%PY% -c "import os, sys; sys.exit(len(os.getcwd()) > 100)" || (
    echo.
    echo  This folder's path is too long for Windows ^(over 100 characters^):
    echo  %CD%
    echo  Move the project folder somewhere shorter, for example C:\glitch_hunter_project,
    echo  then double-click setup.bat again.
    goto :fail
)

echo  [2/5] Creating the environment (venv_gpu) ...
if not exist "venv_gpu\Scripts\python.exe" (
    %PY% -m venv venv_gpu || goto :fail
)
set "VPY=venv_gpu\Scripts\python.exe"
"%VPY%" -m pip install --quiet --upgrade pip || goto :fail

echo  [3/5] Installing PyTorch (the biggest download, please wait) ...
"%VPY%" -m pip install --quiet %TORCH% --index-url %TORCH_INDEX% || goto :fail

echo  [4/5] Installing the other libraries ...
"%VPY%" -m pip install --quiet -r requirements.txt || goto :fail

echo  [5/5] Checking the AI brain files ...
set "MISSING="
for %%F in (glitch_hunter_main_brain.zip glitch_hunter_main_brain_coverage.npz
            checkpoints_qa\final_objective2_16000000\FINAL_OBJECTIVE2.json
            exploration_data\reachable_mask.npz) do if not exist "%%F" set "MISSING=1"
if defined MISSING (
    echo        Some brain files are missing. Downloading them ...
    "%VPY%" tools\final_brain.py install || goto :fail
)
"%VPY%" app.py --help >nul || goto :fail

echo.
echo  Setup finished. Next time, just double-click run_dashboard.bat.
echo.
if "%RUN%"=="1" call run_dashboard.bat
exit /b 0

:find_python
set "PY="
py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12" && exit /b 0
python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1 && set "PY=python" && exit /b 0
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe""
exit /b 0

:fail
echo.
echo  Setup stopped because of the error above. Nothing was broken: fix it
echo  (or check your internet connection) and double-click setup.bat again.
if "%RUN%"=="1" pause
exit /b 1
