from pathlib import Path

import torch
from faster_whisper import WhisperModel

from src.data.chapters import sec_to_hms

# Set device and disable TF32 for consistent results
device = "cuda" if torch.cuda.is_available() else "cpu"


class ASRProcessor:
    """
    Automatic Speech Recognition processor using WhisperX.

    Transcribes audio files and returns time-aligned transcription segments.
    """

    def __init__(self, model_name="large-v2", compute_type="float16"):
        self.model_name = model_name
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def get_asr(self, audio_file, return_duration=True):
        assert Path(audio_file).exists(), f"File {audio_file} does not exist"
        segments, info = self.model.transcribe(
            audio_file, length_penalty=0.5, condition_on_previous_text=False
        )

        asr_clean = []
        for segment in segments:
            t = segment.text.strip()
            s = sec_to_hms(segment.start)
            asr_clean.append(f"{s}: {t}")

        if return_duration:
            return "\n".join(asr_clean) + "\n", info.duration
        else:
            return "\n".join(asr_clean) + "\n"
