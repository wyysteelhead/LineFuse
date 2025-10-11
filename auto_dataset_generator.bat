@echo off
REM LineFuse Automatic Dataset Generator - Windows Batch Version (English)
REM Compatible with all Windows versions, no PowerShell required

chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

setlocal enabledelayedexpansion

REM Configuration parameters
set STAGE1_SAMPLES=8000
set STAGE2_SAMPLES=25000
set BASE_OUTPUT_DIR=dataset\training_dataset
set CONDA_ENV=linefuse
set LOG_DIR=logs

echo === LineFuse Automatic Dataset Generator - Windows Batch Version ===
echo Stage 1: %STAGE1_SAMPLES% samples (U-Net baseline)
echo Stage 2: %STAGE2_SAMPLES% samples (Diffusion model)
echo Output directory: %BASE_OUTPUT_DIR%
echo.

REM Create necessary directories
if not exist "%BASE_OUTPUT_DIR%" (
    mkdir "%BASE_OUTPUT_DIR%"
    echo Created output directory: %BASE_OUTPUT_DIR%
)

if not exist "%LOG_DIR%" (
    mkdir "%LOG_DIR%"
    echo Created log directory: %LOG_DIR%
)

REM Generate timestamp
for /f "tokens=1-6 delims=/:. " %%a in ("%date% %time%") do (
    set TIMESTAMP=%%c%%a%%b_%%d%%e%%f
)
set TIMESTAMP=!TIMESTAMP: =0!

set STAGE1_LOG=%LOG_DIR%\stage1_generation_!TIMESTAMP!.log
set STAGE2_LOG=%LOG_DIR%\stage2_generation_!TIMESTAMP!.log

@REM echo Starting Stage 1 dataset generation...
echo Activating conda environment: %CONDA_ENV%
echo Log file: %STAGE1_LOG%
@REM echo Command: python main.py generate --samples %STAGE1_SAMPLES% --output %BASE_OUTPUT_DIR%\stage1_unet --style-level 0.6 --image-size 1024
@REM echo.

@REM REM Stage 1: U-Net dataset generation
@REM echo [%date% %time%] Starting Stage 1 dataset generation > "%STAGE1_LOG%"
@REM call conda activate %CONDA_ENV% && python main.py generate --samples %STAGE1_SAMPLES% --output %BASE_OUTPUT_DIR%\stage1_unet --style-level 0.6 --image-size 1024 >> "%STAGE1_LOG%" 2>&1

@REM if !errorlevel! neq 0 (
@REM     echo [ERROR] Stage 1 execution failed, exit code: !errorlevel!
@REM     echo Check log file: %STAGE1_LOG%
@REM     pause
@REM     exit /b 1
@REM )

@REM echo [%date% %time%] Stage 1 dataset generation completed >> "%STAGE1_LOG%"
@REM echo Stage 1 completed successfully!
@REM echo.

@REM echo Starting Stage 2 dataset generation...
@REM echo Activating conda environment: %CONDA_ENV%
@REM echo Log file: %STAGE2_LOG%
@REM echo Command: python main.py generate --samples %STAGE2_SAMPLES% --output %BASE_OUTPUT_DIR%\stage2_diffusion --style-level 1.0 --image-size 1024
@REM echo.

REM Stage 2: Diffusion model dataset generation
echo [%date% %time%] Starting Stage 2 dataset generation > "%STAGE2_LOG%"
call conda activate %CONDA_ENV% && python main.py generate --samples %STAGE2_SAMPLES% --output %BASE_OUTPUT_DIR%\stage2_diffusion_line_only --style-level 1.0 --image-size 1024 --pure-line-only >> "%STAGE2_LOG%" 2>&1

if !errorlevel! neq 0 (
    echo [ERROR] Stage 2 execution failed, exit code: !errorlevel!
    echo Check log file: %STAGE2_LOG%
    pause
    exit /b 1
)

echo [%date% %time%] Stage 2 dataset generation completed >> "%STAGE2_LOG%"
echo Stage 2 completed successfully!
echo.

echo.
echo === Dataset Generation Completed! ===
echo Stage 1 log: %STAGE1_LOG%
echo Stage 2 log: %STAGE2_LOG%
echo Dataset location: %BASE_OUTPUT_DIR%
echo Completion time: %date% %time%
echo.

echo Press any key to exit...
pause > nul