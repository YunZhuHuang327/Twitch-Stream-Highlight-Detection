@echo off
REM Download Llama 3.1 8B Instruct base model

echo ============================================================
echo Download Llama 3.1 8B Instruct Base Model
echo ============================================================
echo.
echo This script will download the base model needed for
echo chapter segmentation (~16GB download).
echo.
echo Prerequisites:
echo 1. Accept license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
echo 2. Login with: huggingface-cli login
echo.
echo ============================================================
echo.

python download_llama_base.py

pause
