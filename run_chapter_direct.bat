@echo off
REM Direct activation of chapter-llama environment (bypassing broken conda)

echo ============================================================
echo Chapter-Llama: Generate chapters from ASR text
echo ============================================================
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

REM Check if Python exists
if not exist "%PYTHON_PATH%" (
    echo Error: Python not found at %PYTHON_PATH%
    echo.
    echo Please update PYTHON_PATH in this batch file to point to your chapter-llama Python.
    echo.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_PATH%
echo.

REM Run the script with the chapter-llama environment Python
"%PYTHON_PATH%" chapter_from_asr_english.py ^
    --asr_file "dataset/highlights/123/asr.txt" ^
    --video_title "TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY" ^
    --output_dir "outputs/chapters/123" ^
    --quantization 4bit

echo.
pause
