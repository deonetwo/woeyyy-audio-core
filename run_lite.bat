@echo off
setlocal
title Woeyyy Lite - Mic Boost, Voice Changer & Soundboard
cd /d "%~dp0"

echo ============================================================
echo   Woeyyy Lite - Minimalist Mic Enhancer, Voice Changer & Soundboard
echo ============================================================

:: 1. Verify that Python is installed and accessible
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python was not found on your system!
    echo Please download and install Python from:
    echo    https://www.python.org/downloads/
    echo.
    echo [IMPORTANT] Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

:: 2. If .venv already exists, activate it
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: 3. Check if all required dependencies are available
python -c "import customtkinter, sounddevice, scipy, numpy" >nul 2>&1
if not errorlevel 1 (
    goto :launch
)

:: 4. If dependencies are missing, set up .venv and install requirements
echo.
echo [*] Missing dependencies detected. Setting up virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
echo [*] Installing required audio engine packages from requirements.txt...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies! Please check your internet connection.
    pause
    exit /b 1
)
echo.
echo [*] Setup complete!

:launch
echo [*] Launching Woeyyy Lite...
python gui_lite.py %*

:: If the app exited with an error, keep console open for troubleshooting
if errorlevel 1 (
    echo.
    echo [NOTICE] Woeyyy Lite exited with an error. See details above.
    pause
)
