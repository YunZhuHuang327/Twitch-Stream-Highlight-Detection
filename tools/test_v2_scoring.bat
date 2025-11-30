@echo off
REM Test V2 highlight scoring with data-driven weights
REM Based on groundtruth analysis: Title (40%) + Narrative (35%) + Chat (25%)

echo ========================================
echo Testing V2 Highlight Scoring
echo ========================================
echo.
echo Improvements over original:
echo - Data-driven weights from GT analysis
echo - Strict JSON format with examples
echo - Title entity extraction (auto-detect collaborators)
echo - Title relevance scoring (0-3)
echo - Consistency prompts
echo.

REM Find merged events file (CORRECTED: use merged_3_data)
set MERGED_EVENTS=D:\chapter-llama\outputs\merged_3_data\123\merged_events.json
set OUTPUT_DIR=D:\chapter-llama\outputs\highlights\123

REM Check if merged_events.json exists, if not create it from transcript
if not exist "%MERGED_EVENTS%" (
    echo [WARN] merged_events.json not found, converting from transcript...
    python convert_transcript_to_events.py
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Conversion failed!
        pause
        exit /b 1
    )
    echo [OK] Conversion complete
    echo.
)

echo [INFO] Found merged events: %MERGED_EVENTS%
echo.

REM Set video title (CRITICAL for V2 scoring)
set VIDEO_TITLE=TWITCHCON W/ AGENT00 - EXPLORING LITTLE ITALY - EATING, SHOPS, AND YAPPING

echo [INFO] Video Title: %VIDEO_TITLE%
echo [INFO] Will auto-extract keywords: agent00, little, italy
echo.

REM Run V2 scoring
echo ========================================
echo Running V2 scoring (improved prompt)...
echo ========================================
python tools\score_highlights_v2.py ^
    --merged_events "%MERGED_EVENTS%" ^
    --title "%VIDEO_TITLE%" ^
    --window_size 30 ^
    --stride 15 ^
    --delay 0.2 ^
    --output "%OUTPUT_DIR%\highlight_scores_v2.json"

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] V2 scoring failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Scoring complete!
echo ========================================
echo.
echo Output saved to: %OUTPUT_DIR%\highlight_scores_v2.json
echo.
echo Next steps:
echo.
echo 1. Extract highlights (score >= 7):
echo    python tools\extract_highlights.py --scored_windows "%OUTPUT_DIR%\highlight_scores_v2.json" --min_score 7 --output "%OUTPUT_DIR%\extracted_v2"
echo.
echo 2. Evaluate against groundtruth:
echo    python tools\evaluate_highlights.py --groundtruth "D:\chapter-llama\label.csv" --predictions "%OUTPUT_DIR%\extracted_v2\extracted_highlights.json"
echo.
echo 3. Compare original vs V2:
echo    python compare_scoring_versions.py
echo.

pause
