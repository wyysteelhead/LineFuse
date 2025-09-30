@echo off
REM LineFuse Automatic Dataset Generator - Windows Batch Version (English)
REM Compatible with all Windows versions, no PowerShell required

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

echo Starting Stage 1 dataset generation...
echo Activating conda environment: %CONDA_ENV%
echo Log file: %STAGE1_LOG%
echo Command: python main.py generate --samples %STAGE1_SAMPLES% --output %BASE_OUTPUT_DIR%\stage1_unet --style-level 0.6 --image-size 1024
echo.

REM Stage 1: U-Net dataset generation
echo [%date% %time%] Starting Stage 1 dataset generation > "%STAGE1_LOG%"
call conda activate %CONDA_ENV% && python main.py generate --samples %STAGE1_SAMPLES% --output %BASE_OUTPUT_DIR%\stage1_unet --style-level 0.6 --image-size 1024 >> "%STAGE1_LOG%" 2>&1

if !errorlevel! neq 0 (
    echo [ERROR] Stage 1 execution failed, exit code: !errorlevel!
    echo Check log file: %STAGE1_LOG%
    pause
    exit /b 1
)

echo [%date% %time%] Stage 1 dataset generation completed >> "%STAGE1_LOG%"
echo Stage 1 completed successfully!
echo.

echo Starting Stage 2 dataset generation...
echo Activating conda environment: %CONDA_ENV%
echo Log file: %STAGE2_LOG%
echo Command: python main.py generate --samples %STAGE2_SAMPLES% --output %BASE_OUTPUT_DIR%\stage2_diffusion --style-level 1.0 --image-size 1024
echo.

REM Stage 2: Diffusion model dataset generation
echo [%date% %time%] Starting Stage 2 dataset generation > "%STAGE2_LOG%"
call conda activate %CONDA_ENV% && python main.py generate --samples %STAGE2_SAMPLES% --output %BASE_OUTPUT_DIR%\stage2_diffusion --style-level 1.0 --image-size 1024 >> "%STAGE2_LOG%" 2>&1

if !errorlevel! neq 0 (
    echo [ERROR] Stage 2 execution failed, exit code: !errorlevel!
    echo Check log file: %STAGE2_LOG%
    pause
    exit /b 1
)

echo [%date% %time%] Stage 2 dataset generation completed >> "%STAGE2_LOG%"
echo Stage 2 completed successfully!
echo.

echo Validating generated datasets...

REM Create dataset validation script
echo import os > temp_validation.py
echo stage1_path = '%BASE_OUTPUT_DIR%/stage1_unet/final_dataset'.replace('\\', '/') >> temp_validation.py
echo stage2_path = '%BASE_OUTPUT_DIR%/stage2_diffusion/final_dataset'.replace('\\', '/') >> temp_validation.py
echo. >> temp_validation.py
echo def count_images(path): >> temp_validation.py
echo     count = 0 >> temp_validation.py
echo     if os.path.exists(path): >> temp_validation.py
echo         for root, dirs, files in os.walk(path): >> temp_validation.py
echo             count += len([f for f in files if f.endswith('.png')]) >> temp_validation.py
echo     return count >> temp_validation.py
echo. >> temp_validation.py
echo stage1_count = count_images(stage1_path) >> temp_validation.py
echo stage2_count = count_images(stage2_path) >> temp_validation.py
echo print(f'Stage 1 dataset: {stage1_count} images') >> temp_validation.py
echo print(f'Stage 2 dataset: {stage2_count} images') >> temp_validation.py
echo print(f'Total: {stage1_count + stage2_count} images') >> temp_validation.py
echo. >> temp_validation.py
echo # Check dataset integrity >> temp_validation.py
echo expected_stage1 = %STAGE1_SAMPLES% * 2  # clean + blur >> temp_validation.py
echo expected_stage2 = %STAGE2_SAMPLES% * 2  # clean + blur >> temp_validation.py
echo. >> temp_validation.py
echo if stage1_count ^< expected_stage1 * 0.95:  # Allow 5%% error margin >> temp_validation.py
echo     print(f'WARNING: Stage 1 image count insufficient (expected: {expected_stage1}, actual: {stage1_count})') >> temp_validation.py
echo     exit(1) >> temp_validation.py
echo. >> temp_validation.py
echo if stage2_count ^< expected_stage2 * 0.95:  # Allow 5%% error margin >> temp_validation.py
echo     print(f'WARNING: Stage 2 image count insufficient (expected: {expected_stage2}, actual: {stage2_count})') >> temp_validation.py
echo     exit(1) >> temp_validation.py
echo. >> temp_validation.py
echo print('Dataset validation passed!') >> temp_validation.py
echo print('Dataset generation verification completed!') >> temp_validation.py

REM Run validation script
call conda activate %CONDA_ENV% && python temp_validation.py
if !errorlevel! neq 0 (
    echo [ERROR] Dataset validation failed
    del temp_validation.py
    pause
    exit /b 1
)

REM Clean up temporary files
del temp_validation.py

echo.
echo === Dataset Generation Completed! ===
echo Stage 1 log: %STAGE1_LOG%
echo Stage 2 log: %STAGE2_LOG%
echo Dataset location: %BASE_OUTPUT_DIR%
echo Completion time: %date% %time%
echo.

echo Press any key to exit...
pause > nul