@echo off

setlocal

cd /d "%~dp0"

rem CUDA wheel channel for torch / torchvision.
rem Driver 591.x supports cu130. If torch import fails, try cu126 here.
set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"

echo ============================================
echo VisualSense V0.6  控制台 (AI)
echo ============================================
echo.

if exist ".venv-ai\Scripts\python.exe" goto venv_ok

echo [INFO] AI environment not found.
echo [INFO] Creating .venv-ai ...
echo.

python -m venv .venv-ai

if errorlevel 1 goto err_venv

:venv_ok

".venv-ai\Scripts\python.exe" -c "import torch, torchvision" 1>nul 2>nul

if not errorlevel 1 goto torch_ok

echo [INFO] Installing torch + torchvision (CUDA)...
echo [INFO] Index : %TORCH_INDEX%
echo [INFO] This download is around 3 GB, please wait.
echo.

".venv-ai\Scripts\python.exe" -m pip install --disable-pip-version-check torch torchvision --index-url %TORCH_INDEX%

if errorlevel 1 goto err_torch

:torch_ok

echo [INFO] Checking dependencies...
echo.

".venv-ai\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt -r requirements-ai.txt

if errorlevel 1 goto err_deps

echo.
echo [INFO] Starting VisualSense V0.6 控制台...
echo.

".venv-ai\Scripts\python.exe" -m app.server

goto end

:err_venv

echo.
echo [ERROR] Failed to create .venv-ai.
goto pause_end

:err_torch

echo.
echo [ERROR] Failed to install torch.
echo [ERROR] Check %TORCH_INDEX%, or switch TORCH_INDEX to cu126.
goto pause_end

:err_deps

echo.
echo [ERROR] Failed to install requirements.
goto pause_end

:pause_end

echo.

pause

goto end

:end

echo.
echo ============================================
echo VisualSense stopped.
echo ============================================
echo.

pause
