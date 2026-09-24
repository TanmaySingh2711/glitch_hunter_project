@echo off
rem Double-click to start the Glitch Hunter dashboard.
rem The browser opens http://localhost:5000 once the server has had time to load.
rem Close this window (or press Ctrl+C in it) to stop the dashboard.

cd /d "%~dp0"

if not exist "venv_gpu\Scripts\python.exe" (
    echo venv_gpu was not found in %CD%.
    echo Create it first - see "12. Installation" in README.md.
    pause
    exit /b 1
)

start "" cmd /c "timeout /t 12 /nobreak >nul & start http://localhost:5000"
venv_gpu\Scripts\python.exe app.py %*

echo.
echo The dashboard has stopped.
pause
