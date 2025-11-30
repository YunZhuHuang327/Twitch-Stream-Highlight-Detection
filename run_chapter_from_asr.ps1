# PowerShell 脚本：运行 Chapter-Llama 分段
# 使用方法: .\run_chapter_from_asr.ps1

$env:KMP_DUPLICATE_LIB_OK = "TRUE"

python chapter_from_asr.py `
    --asr_file "dataset/highlights/123/asr.txt" `
    --video_title "TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY" `
    --output_dir "outputs/chapters/123"
