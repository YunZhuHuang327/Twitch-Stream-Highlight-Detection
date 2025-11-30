# Highlight Detection 推理脚本

# 设置环境变量
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

# 参数
$VIDEO_PATH = "D:\chapter-llama\v1.mp4"  # 替换为你的视频路径
$CHAT_FILE = "D:\chapter-llama\chat.json"  # 替换为你的聊天数据文件（可选）
$MODEL_PATH = "D:\chapter-llama\outputs\highlight\Llama-3.2-1B-Instruct-Highlight\highlight\highlight_detection\train\default\model_checkpoints"
$BASE_MODEL = "D:\chapter-llama\Llama-3.2-1B-Instruct"

# 激活 conda 环境
conda activate chapter-llama

# 检查聊天文件是否存在
if (Test-Path $CHAT_FILE) {
    Write-Host "使用聊天数据: $CHAT_FILE" -ForegroundColor Green
    python inference_highlight.py $VIDEO_PATH `
        --model $MODEL_PATH `
        --base_model $BASE_MODEL `
        --chat_file $CHAT_FILE `
        --window_size 35000 `
        --overlap 300
} else {
    Write-Host "未找到聊天数据，仅使用 ASR" -ForegroundColor Yellow
    python inference_highlight.py $VIDEO_PATH `
        --model $MODEL_PATH `
        --base_model $BASE_MODEL `
        --window_size 35000 `
        --overlap 300
}

Write-Host "`n推理完成！" -ForegroundColor Green
