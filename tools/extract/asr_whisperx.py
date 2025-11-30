from pathlib import Path

import torch
import whisperx
from whisperx.audio import SAMPLE_RATE

from src.data.chapters import sec_to_hms

# Set device and disable TF32 for consistent results
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class ASRProcessor:
    """
    Automatic Speech Recognition processor using WhisperX.

    Transcribes audio files and returns time-aligned transcription segments.
    """

    def __init__(self, model_name="large-v2", compute_type="float16"):
        self.model_name = model_name
        self.model = whisperx.load_model(model_name, device, compute_type=compute_type)

    def get_asr(self, audio_file, return_duration=True):
        assert Path(audio_file).exists(), f"File {audio_file} does not exist"
        audio = whisperx.load_audio(audio_file)
        result = self.model.transcribe(audio, batch_size=1)
        language = result["language"]
        duration = audio.shape[0] / SAMPLE_RATE

        # Align the transcription
        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        aligned_result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        # Format the output
        segments = [
            {field: segment[field] for field in ["start", "end", "text"]}
            for segment in aligned_result["segments"]
        ]

        asr_clean = []
        for segment in segments:
            t = segment["text"].strip()
            s = sec_to_hms(segment["start"])
            asr_clean.append(f"{s}: {t}")

        if return_duration:
            return "\n".join(asr_clean) + "\n", duration
        else:
            return "\n".join(asr_clean) + "\n"
