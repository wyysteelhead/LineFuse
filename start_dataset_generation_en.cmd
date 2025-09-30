@echo off
REM LineFuse Dataset Generation Launcher - English Version
REM Double-click this file to run the dataset generation

echo === LineFuse Dataset Generation Launcher ===
echo.
echo Starting dataset generation...
echo This will generate a large dataset, please ensure you have:
echo - At least 20GB free disk space
echo - Stable power connection
echo - Several hours for completion
echo.

echo Choose your preferred method:
echo [1] PowerShell version (recommended, more features)
echo [2] Batch version (simple, more compatible)
echo [3] Exit
echo.

set /p choice="Enter your choice (1/2/3): "

if "%choice%"=="1" (
    echo.
    echo Starting PowerShell version...
    echo If you see execution policy errors, choose [Y] to allow execution
    echo.
    powershell.exe -ExecutionPolicy Bypass -File "%~dp0auto_dataset_generator.ps1"
    if %errorlevel% neq 0 (
        echo.
        echo PowerShell execution failed. Trying batch version as fallback...
        echo.
        pause
        call "%~dp0auto_dataset_generator_en.bat"
    )
) else if "%choice%"=="2" (
    echo.
    echo Starting batch version...
    echo.
    call "%~dp0auto_dataset_generator_en.bat"
) else if "%choice%"=="3" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice. Using batch version by default...
    echo.
    call "%~dp0auto_dataset_generator_en.bat"
)

echo.
echo Dataset generation process completed!
pause