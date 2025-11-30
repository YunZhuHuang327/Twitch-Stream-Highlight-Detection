@echo off
REM Test V3 highlight scoring with STRICT title relevance detection
REM Fixes: False positives from assuming all interactions are with collaborator

echo ========================================
echo Testing V3 Highlight Scoring
echo ========================================
echo.
echo V3 Improvements over V2:
echo - STRICT title relevance detection (must have name confirmation)
echo - Negative example (false positive to avoid)
echo - Explicit evidence requirements for +2/+3 scoring
echo - WARNING prompts against common false positives
echo.

REM Find merged events file
set MERGED_EVENTS=D:\chapter-llama\outputs\merged_3_data\123\merged_events.json
set OUTPUT_DIR=D:\chapter-llama\outputs\prompt-v3

if not exist "%MERGED_EVENTS%" (
    echo [ERROR] merged_events.json not found at: %MERGED_EVENTS%
    echo Please run the full pipeline first to generate merged_3_data
    pause
    exit /b 1
)

echo [INFO] Found merged events: %MERGED_EVENTS%
echo.

REM Set video title
set VIDEO_TITLE=TWITCHCON W/ AGENT00 - EXPLORING LITTLE ITALY - EATING, SHOPS, AND YAPPING

echo [INFO] Video Title: %VIDEO_TITLE%
echo [INFO] Will strictly check for: "agent", "agent00" mentions in ASR/CHAT
echo [INFO] Will NOT assume random fan interactions are Agent00
echo.

REM Run V3 scoring
echo ========================================
echo Running V3 scoring (strict detection)...
echo ========================================
python tools\score_highlights_v3.py ^
    --merged_events "%MERGED_EVENTS%" ^
    --title "%VIDEO_TITLE%" ^
    --window_size 30 ^
    --stride 15 ^
    --delay 0.2

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] V3 scoring failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Scoring complete!
echo ========================================
echo.
echo Output: %OUTPUT_DIR%\highlight_scores_v3.json
echo.
echo Next steps:
echo.
echo 1. Check GT ranges scores:
echo    python -c "import json; data = json.load(open(r'%OUTPUT_DIR%\highlight_scores_v3.json')); gt1 = [w for w in data if 871 <= w['start_time'] < 1068]; print(f'GT#1 avg: {sum(w[\"highlight_score\"] for w in gt1)/len(gt1):.1f}' if gt1 else 'No windows')"
echo.
echo 2. Extract highlights (score >= 6):
echo    python tools\extract_highlights.py --scored_windows "%OUTPUT_DIR%\highlight_scores_v3.json" --min_score 6 --output "%OUTPUT_DIR%\extracted_v3"
echo.
echo 3. Evaluate:
echo    python tools\evaluate_highlights.py --groundtruth "label.csv" --predictions "%OUTPUT_DIR%\extracted_v3\extracted_highlights.json"
echo.

pause
