# 继续训练脚本 - 使用已训练的模型作为起点

param(
    [Parameter(Mandatory=$false)]
    [string]$BaseModel = "D:\chapter-llama\Llama-3.2-1B-Instruct",
    
    [Parameter(Mandatory=$false)]
    [string]$PreviousCheckpoint = "D:\chapter-llama\outputs\chapterize\Llama-3.2-1B-Instruct\asr\default\sml1k_train\default\model_checkpoints"
)

# Set environment variables
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True,max_split_size_mb:128'

Write-Host "=== 继续训练配置 ===" -ForegroundColor Cyan
Write-Host "Base Model: $BaseModel" -ForegroundColor Yellow
Write-Host "Previous Checkpoint: $PreviousCheckpoint" -ForegroundColor Yellow
Write-Host "========================" -ForegroundColor Cyan
Write-Host ""

Write-Host "注意事项:" -ForegroundColor Green
Write-Host "1. 确保已准备好新的训练数据（聊天记录等）" -ForegroundColor White
Write-Host "2. 数据应放在 dataset/docs/subset_data/ 目录下" -ForegroundColor White
Write-Host "3. 继续训练会加载之前的 LoRA 权重并在此基础上训练" -ForegroundColor White
Write-Host ""

# 选项 1: 从头开始训练新数据（推荐用于加入聊天数据）
Write-Host "选项 1: 合并训练（推荐）" -ForegroundColor Yellow
Write-Host "  - 加载之前训练的 LoRA 适配器" -ForegroundColor White
Write-Host "  - 在原有知识基础上学习新数据" -ForegroundColor White
Write-Host "  - 命令: python train.py data=asr_continue model=llama3.2_1B" -ForegroundColor Gray
Write-Host ""

# 选项 2: 完全重新训练
Write-Host "选项 2: 重新训练" -ForegroundColor Yellow
Write-Host "  - 使用原始基础模型" -ForegroundColor White
Write-Host "  - 同时训练原始数据和新数据" -ForegroundColor White  
Write-Host "  - 命令: python train.py data=asr_continue model=llama3.2_1B" -ForegroundColor Gray
Write-Host ""

$choice = Read-Host "选择训练方式 (1=合并训练, 2=重新训练, 其他=退出)"

if ($choice -eq "1") {
    Write-Host "开始合并训练..." -ForegroundColor Green
    conda activate chapter-llama
    
    # TODO: 需要修改 train.py 支持从检查点继续训练
    # 或者使用 PEFT 的 merge 功能先合并，再训练
    python train.py data=asr_continue model=llama3.2_1B
    
} elseif ($choice -eq "2") {
    Write-Host "开始重新训练..." -ForegroundColor Green
    conda activate chapter-llama
    python train.py data=asr_continue model=llama3.2_1B
    
} else {
    Write-Host "退出" -ForegroundColor Red
}

Write-Host ""
Write-Host "训练完成后，新模型将保存在:" -ForegroundColor Cyan
Write-Host "outputs/chapterize/Llama-3.2-1B-Instruct/asr/default/<subset_name>/default/model_checkpoints/" -ForegroundColor Yellow
