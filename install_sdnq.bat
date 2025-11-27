@echo off
echo ========================================
echo SDNQ Installation Script for ComfyUI
echo ========================================
echo.

set COMFY_ROOT=D:\ComfyUI7\ComfyUI
set PYTHON=%COMFY_ROOT%\..\python_embeded\python.exe
set TEMP_DIR=%TEMP%\sdnq_install

echo [1/5] Cleaning up previous installation attempts...
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"

echo [2/5] Cloning SDNQ repository...
git clone https://github.com/Disty0/sdnq "%TEMP_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to clone repository. Make sure git is installed.
    pause
    exit /b 1
)

echo [3/5] Fixing pyproject.toml...
cd /d "%TEMP_DIR%"

:: Use PowerShell to fix multiple issues in pyproject.toml
powershell -Command "$content = Get-Content pyproject.toml -Raw; $content = $content -replace 'license = \"GPL-3.0-only\"', 'license = {text = \"GPL-3.0-only\"}'; $content = $content -replace 'license-files = \[\"LICENSE\"\]', ''; $content | Set-Content pyproject.toml"

echo Fixed pyproject.toml
echo [4/5] Installing SDNQ with ComfyUI's Python...
"%PYTHON%" -m pip install . --no-cache-dir
if errorlevel 1 (
    echo ERROR: Installation failed.
    pause
    exit /b 1
)

echo [5/5] Cleaning up...
cd /d "%COMFY_ROOT%"
rmdir /s /q "%TEMP_DIR%"

echo.
echo ========================================
echo SDNQ installed successfully!
echo Please restart ComfyUI.
echo ========================================
pause
