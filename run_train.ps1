# Run training script with memory optimizations for 16GB GPU
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True,max_split_size_mb:128'

Write-Host "=== Training Configuration ===" -ForegroundColor Cyan
Write-Host "Model: Llama-3.2-1B-Instruct" -ForegroundColor Yellow
Write-Host "Context Length: 2048 (reduced for 16GB GPU)" -ForegroundColor Yellow
Write-Host "LoRA Rank: 4 (reduced to save memory)" -ForegroundColor Yellow
Write-Host "Gradient Checkpointing: Enabled" -ForegroundColor Yellow
Write-Host "Gradient Accumulation Steps: 8" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# First, kill any existing Python processes to free GPU memory
Write-Host "Cleaning up GPU memory..." -ForegroundColor Green
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# Activate conda and run training
conda activate chapter-llama
python train.py 2>&1 | Tee-Object -FilePath "train_log.txt"
