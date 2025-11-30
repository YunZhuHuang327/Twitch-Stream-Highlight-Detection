from pathlib import Path

from lutils import openf, writef

from src.data.chapters import sec_to_hms

# from tools.extract.asr_whisperx import ASRProcessor
from tools.extract.asr_faster_whisper import ASRProcessor


class SingleVideo:
    """
    A simplified implementation of the src.data.chapters.Chapters interface for single video inference.

    This class mimics the behavior of the ChaptersASR class but is designed to work with
    a single video file rather than a dataset. It provides the necessary methods
    required by the PromptASR class for generating chapter timestamps and titles.

    Note: This class is intended for inference only and should not be used for
    training or evaluation purposes.
    """

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.video_ids = [video_path.stem]
        assert video_path.exists(), f"Video file {video_path} not found"
        self.asr, self.duration = get_asr(video_path, overwrite=True)

    def __len__(self):
        return len(self.video_ids)

    def __iter__(self):
        return iter(self.video_ids)

    def __contains__(self, vid_id):
        return vid_id in self.video_ids

    def get_duration(self, vid_id, hms=False):
        assert vid_id == self.video_ids[0], f"Invalid video ID: {vid_id}"
        if hms:
            return sec_to_hms(self.duration)
        return self.duration

    def get_asr(self, vid_id):
        assert vid_id == self.video_ids[0], f"Invalid video ID: {vid_id}"
        return self.asr


def get_asr(video_path, overwrite=False, return_timestamps=False):
    """
    Get ASR for a video
    
    Args:
        video_path: Path to video file (str or Path)
        overwrite: Whether to overwrite cached ASR
        return_timestamps: If True, return (asr_text, asr_with_timestamps, duration)
                          If False, return (asr_text, duration)
    
    Returns:
        If return_timestamps=False: (asr_text, duration)
        If return_timestamps=True: (asr_text, asr_data, duration)
            where asr_data is a list of dicts with 'timestamp' and 'text'
    """
    video_path = Path(video_path)  # Convert to Path if string
    output_dir = Path(f"outputs/inference/{video_path.stem}")
    asr_output = output_dir / "asr.txt"
    asr_json_output = output_dir / "asr.json"
    duration_output = output_dir / "duration.txt"
    
    if asr_output.exists() and duration_output.exists() and not overwrite:
        asr = openf(asr_output)
        asr = "\n".join(asr) + "\n"

        duration = openf(duration_output)
        assert isinstance(duration, list) and len(duration) == 1, (
            f"Duration is not a list of length 1: {duration}"
        )
        duration = float(duration[0])
        assert duration > 0, f"Duration is not positive: {duration}"
        
        if return_timestamps:
            # Load timestamp data if available
            if asr_json_output.exists():
                import json
                with open(asr_json_output, 'r', encoding='utf-8') as f:
                    asr_data = json.load(f)
            else:
                # Parse from text if JSON not available
                asr_data = parse_asr_timestamps(asr)
            return asr, asr_data, duration
        
        return asr, duration

    print(f"\n=== 🎙️ Processing ASR for {video_path} ===")
    asr_processor = ASRProcessor()
    asr, duration = asr_processor.get_asr(video_path)
    print(f"=== ✅ ASR processing complete for {video_path} ===\n")
    output_dir.mkdir(parents=True, exist_ok=True)
    writef(asr_output, asr)
    writef(duration_output, str(duration))
    
    # Save timestamp data
    asr_data = parse_asr_timestamps(asr)
    import json
    with open(asr_json_output, 'w', encoding='utf-8') as f:
        json.dump(asr_data, f, indent=2, ensure_ascii=False)
    
    if return_timestamps:
        return asr, asr_data, duration
    
    return asr, duration


def parse_asr_timestamps(asr_text: str):
    """
    Parse ASR text to extract timestamps
    
    Args:
        asr_text: ASR text with format "[HH:MM:SS] text"
    
    Returns:
        List of dicts with 'timestamp', 'timestamp_str', and 'text'
    """
    import re
    
    lines = asr_text.strip().split('\n')
    asr_data = []
    
    for line in lines:
        # Match [HH:MM:SS] or [MM:SS] or [timestamp]
        match = re.match(r'\[([^\]]+)\]\s*(.*)', line)
        if match:
            time_str = match.group(1)
            text = match.group(2)
            
            # Parse timestamp to seconds
            parts = time_str.split(':')
            try:
                if len(parts) == 3:
                    h, m, s = map(float, parts)
                    timestamp = h * 3600 + m * 60 + s
                elif len(parts) == 2:
                    m, s = map(float, parts)
                    timestamp = m * 60 + s
                else:
                    timestamp = float(time_str)
                
                asr_data.append({
                    'timestamp': timestamp,
                    'timestamp_str': time_str,
                    'text': text
                })
            except ValueError:
                # Skip lines with invalid timestamps
                continue
    
    return asr_data
