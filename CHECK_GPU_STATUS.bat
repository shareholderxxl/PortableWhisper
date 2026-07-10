@echo off
echo ====================================
echo PortableWhisper GPU Diagnostic Tool
echo ====================================
echo.

echo [1/5] Checking for NVIDIA GPU...
wmic path win32_VideoController get name | findstr /i "nvidia geforce quadro"
if %errorlevel% equ 0 (
    echo ✓ NVIDIA GPU detected
) else (
    echo ✗ No NVIDIA GPU found
)
echo.

echo [2/5] Checking Python installation...
where python >nul 2>&1
if %errorlevel% equ 0 (
    python --version
    echo ✓ Python found
) else (
    echo ✗ Python not found in PATH
    echo   This may prevent GPU library installation
)
echo.

echo [3/5] Checking pip installation...
where pip >nul 2>&1
if %errorlevel% equ 0 (
    pip --version
    echo ✓ pip found
) else (
    echo ✗ pip not found in PATH
    py -m pip --version >nul 2>&1
    if %errorlevel% equ 0 (
        py -m pip --version
        echo ✓ pip found via 'py -m pip'
    ) else (
        echo ✗ pip not available
        echo   GPU libraries cannot be auto-installed
    )
)
echo.

echo [4/5] Checking downloaded GPU libraries...
if exist "%~dp0gpu_libs\nvidia" (
    echo ✓ GPU libraries folder exists
    dir "%~dp0gpu_libs\nvidia" /b
    echo.
    echo Checking cuDNN DLLs...
    dir "%~dp0gpu_libs\nvidia\cudnn\bin\*.dll" 2>nul
    if %errorlevel% equ 0 (
        echo ✓ cuDNN DLLs found
    ) else (
        echo ✗ cuDNN DLLs NOT found
    )
) else (
    echo ✗ GPU libraries NOT installed
    echo   Location: %~dp0gpu_libs\nvidia
    echo.
    echo   ACTION REQUIRED:
    echo   1. Run PortableWhisper
    echo   2. Go to Settings
    echo   3. Click "Install GPU Libraries" (~600MB download)
)
echo.

echo [5/5] Checking system CUDA installation...
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" (
    echo ✓ System CUDA found
    dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" /b
) else (
    echo ℹ System CUDA NOT installed (this is OK if using downloaded libs)
)
echo.

if exist "C:\Program Files\NVIDIA\CUDNN" (
    echo ✓ System cuDNN found
    dir "C:\Program Files\NVIDIA\CUDNN" /b
) else (
    echo ℹ System cuDNN NOT installed (this is OK if using downloaded libs)
)
echo.

echo ====================================
echo Diagnostic Complete
echo ====================================
echo.
echo Save this output and share if you need support.
pause
