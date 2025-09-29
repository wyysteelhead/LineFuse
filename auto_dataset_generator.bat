@echo off
REM LineFuse 自动数据集生成脚本 - Windows批处理版本
REM 适用于基础Windows环境，无需PowerShell

setlocal enabledelayedexpansion

REM 配置参数
set STAGE1_SAMPLES=8000
set STAGE2_SAMPLES=25000
set BASE_OUTPUT_DIR=dataset\training_dataset
set PYTHON_PATH=/root/miniconda3/envs/linefuse/bin/python
set LOG_DIR=logs

echo === LineFuse 自动数据集生成 - Windows批处理版本 ===
echo Stage 1: %STAGE1_SAMPLES% 样本 (U-Net baseline)
echo Stage 2: %STAGE2_SAMPLES% 样本 (Diffusion model)
echo 输出目录: %BASE_OUTPUT_DIR%
echo.

REM 创建必要目录
if not exist "%BASE_OUTPUT_DIR%" (
    mkdir "%BASE_OUTPUT_DIR%"
    echo 创建输出目录: %BASE_OUTPUT_DIR%
)

if not exist "%LOG_DIR%" (
    mkdir "%LOG_DIR%"
    echo 创建日志目录: %LOG_DIR%
)

REM 生成时间戳
for /f "tokens=1-6 delims=/:. " %%a in ("%date% %time%") do (
    set TIMESTAMP=%%c%%a%%b_%%d%%e%%f
)
set TIMESTAMP=!TIMESTAMP: =0!

set STAGE1_LOG=%LOG_DIR%\stage1_generation_!TIMESTAMP!.log
set STAGE2_LOG=%LOG_DIR%\stage2_generation_!TIMESTAMP!.log

echo 开始Stage 1数据集生成...
echo 日志文件: %STAGE1_LOG%
echo 命令: %PYTHON_PATH% main.py generate --samples %STAGE1_SAMPLES% --output %BASE_OUTPUT_DIR%\stage1_unet --style-level 0.6 --image-size 1024
echo.

REM Stage 1: U-Net数据集生成
echo [%date% %time%] 开始Stage 1数据集生成 > "%STAGE1_LOG%"
%PYTHON_PATH% main.py generate --samples %STAGE1_SAMPLES% --output %BASE_OUTPUT_DIR%\stage1_unet --style-level 0.6 --image-size 1024 >> "%STAGE1_LOG%" 2>&1

if !errorlevel! neq 0 (
    echo [错误] Stage 1 执行失败，退出码: !errorlevel!
    echo 检查日志文件: %STAGE1_LOG%
    pause
    exit /b 1
)

echo [%date% %time%] Stage 1数据集生成完成 >> "%STAGE1_LOG%"
echo Stage 1 完成成功!
echo.

echo 开始Stage 2数据集生成...
echo 日志文件: %STAGE2_LOG%
echo 命令: %PYTHON_PATH% main.py generate --samples %STAGE2_SAMPLES% --output %BASE_OUTPUT_DIR%\stage2_diffusion --style-level 1.0 --image-size 1024
echo.

REM Stage 2: 扩散模型数据集生成
echo [%date% %time%] 开始Stage 2数据集生成 > "%STAGE2_LOG%"
%PYTHON_PATH% main.py generate --samples %STAGE2_SAMPLES% --output %BASE_OUTPUT_DIR%\stage2_diffusion --style-level 1.0 --image-size 1024 >> "%STAGE2_LOG%" 2>&1

if !errorlevel! neq 0 (
    echo [错误] Stage 2 执行失败，退出码: !errorlevel!
    echo 检查日志文件: %STAGE2_LOG%
    pause
    exit /b 1
)

echo [%date% %time%] Stage 2数据集生成完成 >> "%STAGE2_LOG%"
echo Stage 2 完成成功!
echo.

echo 验证生成的数据集...

REM 创建数据集验证脚本
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
echo print(f'Stage 1 数据集: {stage1_count} 张图像') >> temp_validation.py
echo print(f'Stage 2 数据集: {stage2_count} 张图像') >> temp_validation.py
echo print(f'总计: {stage1_count + stage2_count} 张图像') >> temp_validation.py
echo. >> temp_validation.py
echo # 检查数据集完整性 >> temp_validation.py
echo expected_stage1 = %STAGE1_SAMPLES% * 2  # clean + blur >> temp_validation.py
echo expected_stage2 = %STAGE2_SAMPLES% * 2  # clean + blur >> temp_validation.py
echo. >> temp_validation.py
echo if stage1_count ^< expected_stage1 * 0.95:  # 允许5%%的误差 >> temp_validation.py
echo     print(f'WARNING: Stage 1 图像数量不足 (期望: {expected_stage1}, 实际: {stage1_count})') >> temp_validation.py
echo     exit(1) >> temp_validation.py
echo. >> temp_validation.py
echo if stage2_count ^< expected_stage2 * 0.95:  # 允许5%%的误差 >> temp_validation.py
echo     print(f'WARNING: Stage 2 图像数量不足 (期望: {expected_stage2}, 实际: {stage2_count})') >> temp_validation.py
echo     exit(1) >> temp_validation.py
echo. >> temp_validation.py
echo print('✅ 数据集验证通过!') >> temp_validation.py
echo print('数据集生成验证完成!') >> temp_validation.py

REM 运行验证脚本
%PYTHON_PATH% temp_validation.py
if !errorlevel! neq 0 (
    echo [错误] 数据集验证失败
    del temp_validation.py
    pause
    exit /b 1
)

REM 清理临时文件
del temp_validation.py

echo.
echo === 数据集生成完成! ===
echo Stage 1 日志: %STAGE1_LOG%
echo Stage 2 日志: %STAGE2_LOG%
echo 数据集位置: %BASE_OUTPUT_DIR%
echo 完成时间: %date% %time%
echo.

echo 按任意键退出...
pause > nul