#!/usr/bin/env bash
set -euo pipefail

echo "=== Cars Detector + Face Recognition — Install ==="
echo

# ---- Check Python ----
if ! command -v python3 &>/dev/null; then
    echo "Python 3 not found. Install Python 3.10+ first."
    exit 1
fi

# ---- Create venv ----
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# ---- Upgrade pip ----
python3 -m pip install --upgrade pip

# ---- Detect CUDA ----
CUDA_AVAILABLE=0
if command -v nvidia-smi &>/dev/null; then
    CUDA_AVAILABLE=1
fi

echo
if [ "$CUDA_AVAILABLE" = 1 ]; then
    echo "CUDA detected (GPU mode)"
else
    echo "CUDA not detected (CPU mode)"
fi

# ---- System deps ----
if [ "$(uname)" = "Linux" ]; then
    echo "Installing system dependencies..."
    sudo apt-get update -qq && sudo apt-get install -y -qq libgl1 libglib2.0-0 2>/dev/null || true
fi

# ---- Install PyTorch (CPU or CUDA) ----
if [ "$CUDA_AVAILABLE" = 1 ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# ---- Install GPU packages if CUDA available ----
if [ "$CUDA_AVAILABLE" = 1 ]; then
    pip install onnxruntime-gpu faiss-gpu
else
    pip install onnxruntime faiss-cpu
fi

# ---- Install everything else ----
echo "Installing dependencies..."

# v4
pip install opencv-python numpy scikit-image fastapi uvicorn ultralytics python-dotenv google-genai

# cars_detector
pip install opencv-python numpy fastapi uvicorn ultralytics

# ---- Done ----
echo
echo "=== Install complete ==="
echo
echo "To run:"
echo "  v4:    python v4/app.py --source 0"
echo "  cars:  python cars_detector/app.py --source 0"
echo
