@echo off
setlocal enabledelayedexpansion

echo === Cars Detector + Face Recognition — Install ===
echo.

REM ---- Python check ----
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM ---- Create venv ----
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

REM ---- Upgrade pip ----
python -m pip install --upgrade pip

REM ---- Detect CUDA ----
set CUDA_AVAILABLE=0
nvcc --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set CUDA_AVAILABLE=1
)

echo.
if %CUDA_AVAILABLE% equ 1 (
    echo CUDA detected ^(GPU mode^)
) else (
    echo CUDA not detected ^(CPU mode^)
)

REM ---- Install PyTorch (CPU or CUDA) ----
if %CUDA_AVAILABLE% equ 1 (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

REM ---- Install GPU packages if CUDA available ----
if %CUDA_AVAILABLE% equ 1 (
    pip install onnxruntime-gpu faiss-gpu
) else (
    pip install onnxruntime faiss-cpu
)

REM ---- Install everything else ----
echo Installing dependencies...

REM v4
pip install opencv-python numpy scikit-image fastapi uvicorn ultralytics python-dotenv google-genai

REM cars_detector
pip install opencv-python numpy fastapi uvicorn ultralytics

REM ---- Done ----
echo.
echo === Install complete ===
echo.
echo To run:
echo   v4:  python v4\app.py --source 0
echo   cars:  python cars_detector\app.py --source 0
echo.
pause
