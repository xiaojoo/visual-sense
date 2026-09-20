@echo off

setlocal

cd /d "%~dp0"

echo ============================================
echo VisualSense V0.6  控制台
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Virtual environment not found.
    echo [INFO] Creating .venv ...
    echo.

    python -m venv .venv

    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to create virtual environment.
        echo.
        pause
        exit /b 1
    )
)

echo [INFO] Checking dependencies...
echo.

".venv\Scripts\python.exe" -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies.
    echo.
    pause
    exit /b 1
)

echo.
echo [INFO] Starting VisualSense...
echo.

".venv\Scripts\python.exe" -m app.server

echo.
echo ============================================
echo VisualSense stopped.
echo ============================================
echo.

pause