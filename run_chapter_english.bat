@echo off
REM Chapter segmentation using Chapter-Llama (English version - no encoding issues)

REM Set encoding to UTF-8
chcp 65001 >nul 2>&1

REM Set environment variables to fix encoding and OpenMP issues
set PYTHONIOENCODING=utf-8
set KMP_DUPLICATE_LIB_OK=TRUE

echo ============================================================
echo Chapter-Llama: Generate chapters from ASR text
echo ============================================================
echo.
echo ASR file: dataset/highlights/123/asr.txt
echo Output: outputs/chapters/123/
echo.

REM Use 4-bit quantization to reduce memory usage and avoid offloading issues
python chapter_from_asr_english.py ^
    --asr_file "dataset/highlights/123/asr.txt" ^
    --video_title "TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY" ^
    --output_dir "outputs/chapters/123" ^
    --quantization 4bit

echo.
pause
