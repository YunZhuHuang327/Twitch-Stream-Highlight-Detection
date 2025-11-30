@echo off
REM Chapter-Llama without quantization (may use more VRAM but should work)

echo ============================================================
echo Chapter-Llama: Generate chapters from ASR text (No Quant)
echo ============================================================
echo.
echo WARNING: This will use ~16GB VRAM or offload to CPU
echo If you have less than 16GB VRAM, this may be slow but should work
echo.
echo Using environment: chapter-llama
echo ASR file: dataset/highlights/123/asr.txt
echo Output: outputs/chapters/123/
echo.

REM Set encoding to UTF-8
chcp 65001 >nul 2>&1

REM Set environment variables
set PYTHONIOENCODING=utf-8
set KMP_DUPLICATE_LIB_OK=TRUE

REM Directly use the chapter-llama environment Python
set PYTHON_PATH=C:\Users\Huang\Miniconda3\envs\chapter-llama\python.exe

echo Using Python: %PYTHON_PATH%
echo.

REM Run WITHOUT quantization
"%PYTHON_PATH%" chapter_from_asr_english.py ^
    --asr_file "dataset/highlights/123/asr.txt" ^
    --video_title "TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY" ^
    --output_dir "outputs/chapters/123"

echo.
pause
