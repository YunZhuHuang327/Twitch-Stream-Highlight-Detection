@echo off
REM Quick Chapter Segmentation using GPT-4o-mini API
REM This avoids all encoding issues with local models

echo ============================================================
echo Quick Chapter Segmentation (GPT-4o-mini API)
echo ============================================================
echo.

REM Check if API key is set
if "%OPENAI_API_KEY%"=="" (
    echo Error: OPENAI_API_KEY environment variable not set!
    echo.
    echo Please set your API key:
    echo   set OPENAI_API_KEY=sk-your-key-here
    echo.
    echo Or pass it as a parameter:
    echo   python quick_chapter.py --api_key sk-your-key-here
    echo.
    echo Get your API key at: https://platform.openai.com/api-keys
    echo.
    pause
    exit /b 1
)

REM Run the script
python quick_chapter.py ^
    --asr_file "dataset/highlights/123/asr.txt" ^
    --video_title "TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY" ^
    --output_dir "outputs/chapters/123"

pause
