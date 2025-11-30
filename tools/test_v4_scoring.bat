@echo off
REM Test V4 highlight scoring with PRIMARY SUBJECT PRIORITY
REM V4 Fixes: Avoid false positives from agent-only segments without primary subject

echo ========================================
echo Testing V4 Highlight Scoring
echo ========================================
echo.
echo V4 Improvements over V3:
echo - PRIMARY SUBJECT PRIORITY (main streamer must be present)
echo - Weighted scoring: Primary 40%%, Title 25%%, Narrative 20%%, Chat 15%%
echo - Agent-only segments without primary subject capped at 5-6
echo - Generic template (no hardcoded names in prompt)
echo - Example 5: Explicit "other person only" false positive prevention
echo.

REM Find merged events file
set MERGED_EVENTS=D:\chapter-llama\outputs\merged_3_data\123\merged_events.json
set OUTPUT_DIR=D:\chapter-llama\outputs\prompt-v4

if not exist "%MERGED_EVENTS%" (
    echo [ERROR] merged_events.json not found at: %MERGED_EVENTS%
    echo Please run the full pipeline first to generate merged_3_data
    pause
    exit /b 1
)

echo [INFO] Found merged events: %MERGED_EVENTS%
echo.

REM Set video metadata
set VIDEO_TITLE=TWITCHCON W/ AGENT00 - EXPLORING LITTLE ITALY - EATING, SHOPS, AND YAPPING
set PRIMARY_SUBJECT=extraemily

echo [INFO] Video Title: %VIDEO_TITLE%
echo [INFO] Primary Subject: %PRIMARY_SUBJECT%
echo [INFO] Will prioritize segments where %PRIMARY_SUBJECT% appears
echo [INFO] Will check for title collaborators: "agent", "agent00"
echo [INFO] Agent-only segments (no %PRIMARY_SUBJECT%) will be capped at 5-6
echo.

REM Run V4 scoring
echo ========================================
echo Running V4 scoring (primary subject priority)...
echo ========================================
python tools\score_highlights_v4.py ^
    --merged_events "%MERGED_EVENTS%" ^
    --title "%VIDEO_TITLE%" ^
    --primary_subject "%PRIMARY_SUBJECT%" ^
    --window_size 30 ^
    --stride 15 ^
    --delay 0.2

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] V4 scoring failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Scoring complete!
echo ========================================
echo.
echo Output: %OUTPUT_DIR%\highlight_scores_v4.json
echo.
echo Next steps:
echo.
echo 1. Check GT ranges scores:
echo    python -c "import json; data = json.load(open(r'%OUTPUT_DIR%\highlight_scores_v4.json')); gt1 = [w for w in data if 871 <= w['start_time'] < 1068]; print(f'GT#1 avg: {sum(w[\"highlight_score\"] for w in gt1)/len(gt1):.1f}' if gt1 else 'No windows')"
echo.
echo 2. Extract highlights (score >= 7):
echo    python tools\extract_highlights.py --scored_windows "%OUTPUT_DIR%\highlight_scores_v4.json" --min_score 7 --output "%OUTPUT_DIR%\extracted_v4"
echo.
echo 3. Evaluate:
echo    python tools\evaluate_highlights.py --groundtruth "label.csv" --predictions "%OUTPUT_DIR%\extracted_v4\extracted_highlights.json"
echo.
echo 4. Compare V3 vs V4:
echo    Compare precision/recall between prompt-v3 and prompt-v4 outputs
echo.

pause
