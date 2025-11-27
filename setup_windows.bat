@echo off
REM =============================================
REM Z-Image Turbo Q-DiT Setup Script (Windows)
REM =============================================

REM 1. OPTIONAL: Create virtual environment (skip if using Comfy embedded env)
python -m venv venv
call venv\Scripts\activate

REM 2. Set environment variables
set HF_TOKEN=rbaiSlhMqLcFZBhIiQbSSQPWWMLLuXvnuQ
REM Optional: set CUDA paths only if needed
REM set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

REM 3. Upgrade pip
python -m pip install --upgrade pip

REM 4. Install PyTorch with CUDA 12.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

REM 5. Upgrade diffusers to latest (>=0.35.2)
REM pip install --upgrade diffusers
pip install git+https://github.com/huggingface/diffusers.git

REM 6. Install other dependencies
pip install -r requirements.txt

REM 7. Run quantization helper
python quantization_helper.py

REM 8. Optional calibration
set /p RUN_CALIBRATION="Run calibration? (y/n): "
if /I "%RUN_CALIBRATION%"=="y" (
    python qdit_calibration.py
)

echo Setup complete.