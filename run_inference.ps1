# Inference script for local model
# Usage: .\run_inference.ps1 <video_path>

param(
    [Parameter(Mandatory=$true)]
    [string]$VideoPath,
    
    [Parameter(Mandatory=$false)]
    [string]$ModelPath = "D:\chapter-llama\outputs\chapterize\Llama-3.2-1B-Instruct\asr\default\sml1k_train\default\model_checkpoints",
    
    [Parameter(Mandatory=$false)]
    [string]$BaseModel = "D:\chapter-llama\Llama-3.2-1B-Instruct"
)

# Set environment variables
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

Write-Host "=== Chapter-Llama Inference ===" -ForegroundColor Cyan
Write-Host "Video: $VideoPath" -ForegroundColor Yellow
Write-Host "Model: $ModelPath" -ForegroundColor Yellow
Write-Host "Base Model: $BaseModel" -ForegroundColor Yellow
Write-Host "===============================" -ForegroundColor Cyan
Write-Host ""

# Activate conda and run inference
conda activate chapter-llama
python inference.py "$VideoPath" --model "$ModelPath" --base_model "$BaseModel"
