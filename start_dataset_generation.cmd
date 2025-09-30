@echo off
REM LineFuse数据集生成启动器
REM 双击此文件即可运行PowerShell脚本

echo === LineFuse数据集生成启动器 ===
echo.
echo 正在启动PowerShell脚本...
echo 如果出现执行策略错误，请选择 [Y] 允许执行
echo.

REM 设置执行策略并运行PowerShell脚本
echo 尝试方法1: 直接执行PowerShell脚本
powershell.exe -ExecutionPolicy Bypass -File "%~dp0auto_dataset_generator.ps1"
if %errorlevel% equ 0 goto success

echo.
echo 方法1失败，尝试方法2: 通过命令行参数执行
powershell.exe -ExecutionPolicy Bypass -Command "& '%~dp0auto_dataset_generator.ps1'"
if %errorlevel% equ 0 goto success

echo.
echo 方法2失败，尝试方法3: 设置执行策略后执行
powershell.exe -Command "Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process; & '%~dp0auto_dataset_generator.ps1'"
if %errorlevel% equ 0 goto success

:success

if %errorlevel% neq 0 (
    echo.
    echo 执行失败，可能的原因：
    echo 1. PowerShell执行策略限制
    echo 2. Python环境未正确配置
    echo 3. 缺少必要权限
    echo.
    echo 尝试备选方案：批处理版本
    echo.
    pause
    call "%~dp0auto_dataset_generator.bat"
) else (
    echo.
    echo PowerShell脚本执行完成！
)

pause