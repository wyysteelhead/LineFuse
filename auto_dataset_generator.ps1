# LineFuse 自动数据集生成脚本 - Windows版本
# 使用PowerShell后台任务管理长时间运行

param(
    [int]$Stage1Samples = 8000,    # U-Net训练数据集
    [int]$Stage2Samples = 25000,   # 扩散模型数据集
    [string]$BaseOutputDir = "dataset\training_dataset",
    [string]$PythonPath = "/root/miniconda3/envs/linefuse/bin/python"
)

# 设置错误处理
$ErrorActionPreference = "Stop"

Write-Host "=== LineFuse 自动数据集生成 - Windows版本 ===" -ForegroundColor Green
Write-Host "Stage 1: $Stage1Samples 样本 (U-Net baseline)" -ForegroundColor Yellow
Write-Host "Stage 2: $Stage2Samples 样本 (Diffusion model)" -ForegroundColor Yellow
Write-Host "输出目录: $BaseOutputDir" -ForegroundColor Yellow

# 创建输出目录
if (-not (Test-Path $BaseOutputDir)) {
    New-Item -ItemType Directory -Path $BaseOutputDir -Force | Out-Null
    Write-Host "创建输出目录: $BaseOutputDir" -ForegroundColor Green
}

# 日志文件路径
$LogDir = "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
$Stage1LogFile = "$LogDir\stage1_generation_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$Stage2LogFile = "$LogDir\stage2_generation_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Host "`n开始Stage 1数据集生成..." -ForegroundColor Cyan
Write-Host "日志文件: $Stage1LogFile" -ForegroundColor Gray

# Stage 1: U-Net数据集生成
$Stage1Job = Start-Job -ScriptBlock {
    param($PythonPath, $Samples, $OutputDir, $LogFile)

    $Command = "$PythonPath main.py generate --samples $Samples --output $OutputDir/stage1_unet --style-level 0.6 --image-size 1024"

    # 执行命令并捕获输出
    $Process = Start-Process -FilePath $PythonPath -ArgumentList "main.py", "generate", "--samples", $Samples, "--output", "$OutputDir/stage1_unet", "--style-level", "0.6", "--image-size", "1024" -NoNewWindow -PassThru -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.error"

    $Process.WaitForExit()
    return $Process.ExitCode
} -ArgumentList $PythonPath, $Stage1Samples, $BaseOutputDir, $Stage1LogFile

Write-Host "Stage 1 任务已启动 (Job ID: $($Stage1Job.Id))" -ForegroundColor Green
Write-Host "监控进度: Get-Job -Id $($Stage1Job.Id) | Receive-Job -Keep" -ForegroundColor Gray

# 等待Stage 1完成
Write-Host "`n等待Stage 1完成..." -ForegroundColor Yellow
do {
    Start-Sleep -Seconds 30
    $JobState = Get-Job -Id $Stage1Job.Id
    Write-Host "Stage 1 状态: $($JobState.State) - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

    # 显示最新的日志行数（如果日志文件存在）
    if (Test-Path $Stage1LogFile) {
        $LogLines = (Get-Content $Stage1LogFile | Measure-Object -Line).Lines
        if ($LogLines -gt 0) {
            Write-Host "日志行数: $LogLines" -ForegroundColor Gray
        }
    }
} while ($JobState.State -eq "Running")

$Stage1Result = Receive-Job -Id $Stage1Job.Id
Remove-Job -Id $Stage1Job.Id

if ($Stage1Result -eq 0) {
    Write-Host "`nStage 1 完成成功!" -ForegroundColor Green
} else {
    Write-Host "`nStage 1 执行失败 (退出码: $Stage1Result)" -ForegroundColor Red
    Write-Host "检查错误日志: $Stage1LogFile.error" -ForegroundColor Red
    exit 1
}

Write-Host "`n开始Stage 2数据集生成..." -ForegroundColor Cyan
Write-Host "日志文件: $Stage2LogFile" -ForegroundColor Gray

# Stage 2: 扩散模型数据集生成
$Stage2Job = Start-Job -ScriptBlock {
    param($PythonPath, $Samples, $OutputDir, $LogFile)

    $Process = Start-Process -FilePath $PythonPath -ArgumentList "main.py", "generate", "--samples", $Samples, "--output", "$OutputDir/stage2_diffusion", "--style-level", "1.0", "--image-size", "1024" -NoNewWindow -PassThru -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.error"

    $Process.WaitForExit()
    return $Process.ExitCode
} -ArgumentList $PythonPath, $Stage2Samples, $BaseOutputDir, $Stage2LogFile

Write-Host "Stage 2 任务已启动 (Job ID: $($Stage2Job.Id))" -ForegroundColor Green

# 等待Stage 2完成
Write-Host "`n等待Stage 2完成..." -ForegroundColor Yellow
do {
    Start-Sleep -Seconds 30
    $JobState = Get-Job -Id $Stage2Job.Id
    Write-Host "Stage 2 状态: $($JobState.State) - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

    # 显示最新的日志行数
    if (Test-Path $Stage2LogFile) {
        $LogLines = (Get-Content $Stage2LogFile | Measure-Object -Line).Lines
        if ($LogLines -gt 0) {
            Write-Host "日志行数: $LogLines" -ForegroundColor Gray
        }
    }
} while ($JobState.State -eq "Running")

$Stage2Result = Receive-Job -Id $Stage2Job.Id
Remove-Job -Id $Stage2Job.Id

if ($Stage2Result -eq 0) {
    Write-Host "`nStage 2 完成成功!" -ForegroundColor Green
} else {
    Write-Host "`nStage 2 执行失败 (退出码: $Stage2Result)" -ForegroundColor Red
    Write-Host "检查错误日志: $Stage2LogFile.error" -ForegroundColor Red
    exit 1
}

# 数据集验证
Write-Host "`n验证生成的数据集..." -ForegroundColor Cyan

$ValidationScript = @"
import os
import sys
stage1_path = '$BaseOutputDir/stage1_unet/final_dataset'.replace('\\', '/')
stage2_path = '$BaseOutputDir/stage2_diffusion/final_dataset'.replace('\\', '/')

def count_images(path):
    count = 0
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            count += len([f for f in files if f.endswith('.png')])
    return count

stage1_count = count_images(stage1_path)
stage2_count = count_images(stage2_path)

print(f'Stage 1 数据集: {stage1_count} 张图像')
print(f'Stage 2 数据集: {stage2_count} 张图像')
print(f'总计: {stage1_count + stage2_count} 张图像')
print('数据集生成验证完成!')

# 检查数据集完整性
expected_stage1 = $Stage1Samples * 2  # clean + blur
expected_stage2 = $Stage2Samples * 2  # clean + blur

if stage1_count < expected_stage1 * 0.95:  # 允许5%的误差
    print(f'WARNING: Stage 1 图像数量不足 (期望: {expected_stage1}, 实际: {stage1_count})')
    sys.exit(1)

if stage2_count < expected_stage2 * 0.95:  # 允许5%的误差
    print(f'WARNING: Stage 2 图像数量不足 (期望: {expected_stage2}, 实际: {stage2_count})')
    sys.exit(1)

print('✅ 数据集验证通过!')
"@

# 写入临时验证脚本
$TempValidationScript = "temp_validation.py"
$ValidationScript | Out-File -FilePath $TempValidationScript -Encoding UTF8

try {
    $ValidationResult = & $PythonPath $TempValidationScript
    Write-Host $ValidationResult -ForegroundColor Green
} catch {
    Write-Host "数据集验证失败: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    # 清理临时文件
    if (Test-Path $TempValidationScript) {
        Remove-Item $TempValidationScript -Force
    }
}

Write-Host "`n=== 数据集生成完成! ===" -ForegroundColor Green
Write-Host "Stage 1 日志: $Stage1LogFile" -ForegroundColor Gray
Write-Host "Stage 2 日志: $Stage2LogFile" -ForegroundColor Gray
Write-Host "数据集位置: $BaseOutputDir" -ForegroundColor Gray

# 显示磁盘使用情况
$DatasetSize = (Get-ChildItem -Path $BaseOutputDir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "数据集大小: $([math]::Round($DatasetSize, 2)) MB" -ForegroundColor Yellow

Write-Host "`n任务完成时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green