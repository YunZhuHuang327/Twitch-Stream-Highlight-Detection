@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
chcp 65001
python chapter_from_asr.py --asr_file "dataset/highlights/123/asr.txt" --video_title "TwitchCon" --output_dir "outputs/chapters/123"
