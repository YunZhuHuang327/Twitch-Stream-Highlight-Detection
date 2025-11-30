import os
import tempfile
from pathlib import Path

import gradio as gr
from llama_cookbook.inference.model_utils import load_model as load_model_llamarecipes
from llama_cookbook.inference.model_utils import load_peft_model
from transformers import AutoTokenizer

from src.data.single_video import SingleVideo
from src.data.utils_asr import PromptASR
from src.models.llama_inference import inference
from src.test.vidchapters import get_chapters
from tools.download.models import download_model

# Global variables to store loaded models
base_model = None
tokenizer = None
current_peft_model = None
inference_model = None

LLAMA_CKPT_PATH = "meta-llama/Llama-3.1-8B-Instruct"


def load_base_model():
    """Load the base Llama model and tokenizer once at startup."""
    global base_model, tokenizer

    if base_model is None:
        print(f"Loading base model: {LLAMA_CKPT_PATH}")
        base_model = load_model_llamarecipes(
            model_name=LLAMA_CKPT_PATH,
            device_map="auto",
            quantization=None,
            use_fast_kernels=True,
        )
        base_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(LLAMA_CKPT_PATH)
        tokenizer.pad_token = tokenizer.eos_token

        print("Base model loaded successfully")


class FastLlamaInference:
    def __init__(
        self,
        model,
        add_special_tokens: bool = True,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        top_p: float = 1.0,
        top_k: int = 50,
        use_cache: bool = True,
        max_padding_length: int = None,
        do_sample: bool = False,
        min_length: int = None,
        repetition_penalty: float = 1.0,
        length_penalty: int = 1,
        max_prompt_tokens: int = 35_000,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.use_cache = use_cache
        self.max_padding_length = max_padding_length
        self.do_sample = do_sample
        self.min_length = min_length
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.max_prompt_tokens = max_prompt_tokens

    def __call__(self, prompt: str, **kwargs):
        # Create a dict of default parameters from instance attributes
        params = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "add_special_tokens": self.add_special_tokens,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "use_cache": self.use_cache,
            "max_padding_length": self.max_padding_length,
            "do_sample": self.do_sample,
            "min_length": self.min_length,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "max_prompt_tokens": self.max_prompt_tokens,
        }

        # Update with any overrides passed in kwargs
        params.update(kwargs)

        return inference(**params)


def load_peft(model_name: str = "asr-10k"):
    """Load or switch PEFT model while reusing the base model."""
    global base_model, current_peft_model, inference_model

    # First make sure the base model is loaded
    if base_model is None:
        load_base_model()

    # Only load a new PEFT model if it's different from the current one
    if current_peft_model != model_name:
        print(f"Loading PEFT model: {model_name}")
        model_path = download_model(model_name)

        if not Path(model_path).exists():
            print(f"PEFT model does not exist at {model_path}")
            return False

        # Apply the PEFT model to the base model
        peft_model = load_peft_model(base_model, model_path)

        peft_model.eval()

        # Create the inference wrapper
        inference_model = FastLlamaInference(model=peft_model)
        current_peft_model = model_name

        print(f"PEFT model {model_name} loaded successfully")
        return True

    # Model already loaded
    return True


def download_from_url(url, output_path):
    """Download a video from a URL using yt-dlp and save it to output_path."""
    try:
        # Import yt-dlp Python package
        try:
            import yt_dlp
        except ImportError:
            print("yt-dlp Python package is not installed")
            return (
                False,
                "yt-dlp Python package is not installed. Please install it with 'pip install yt-dlp'.",
            )

        # Configure yt-dlp options
        ydl_opts = {
            "format": "best",
            "outtmpl": str(output_path),
            "noplaylist": True,
            "quiet": True,
        }

        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Check if the download was successful
        if not os.path.exists(output_path):
            return (
                False,
                "Download completed but video file not found. Please check the URL.",
            )

        return True, None
    except Exception as e:
        error_msg = f"Error downloading video: {str(e)}"
        print(error_msg)
        return False, error_msg


def process_video(
    video_file, video_url, model_name: str = "asr-10k", do_sample: bool = False
):
    """Process a video file or URL and generate chapters."""
    progress = gr.Progress()
    progress(0, desc="Starting...")

    # Check if we have a valid input
    if video_file is None and not video_url:
        return "Please upload a video file or provide a URL."

    # Load the PEFT model
    progress(0.1, desc=f"Loading LoRA parameters from {model_name}...")
    if not load_peft(model_name):
        return "Failed to load model. Please try again."

    # Create a temporary directory to save the uploaded or downloaded video
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_path = Path(temp_dir) / "temp_video.mp4"

        if video_file is not None:
            # Using uploaded file
            progress(0.2, desc="Processing uploaded video...")
            with open(temp_video_path, "wb") as f:
                f.write(video_file)
        else:
            # Using URL
            progress(0.2, desc=f"Downloading video from URL: {video_url}...")
            success, error_msg = download_from_url(video_url, temp_video_path)
            if not success:
                return f"Failed to download video: {error_msg}"

        # Process the video
        progress(0.3, desc="Extracting ASR transcript...")
        single_video = SingleVideo(temp_video_path)
        progress(0.4, desc="Creating prompt...")
        prompt = PromptASR(chapters=single_video)

        vid_id = single_video.video_ids[0]
        progress(0.5, desc="Creating prompt...")
        prompt = prompt.get_prompt_test(vid_id)

        transcript = single_video.get_asr(vid_id)
        prompt = prompt + transcript

        progress(0.6, desc="Generating chapters with Chapter-Llama...")
        _, chapters = get_chapters(
            inference_model,
            prompt,
            max_new_tokens=1024,
            do_sample=do_sample,
            vid_id=vid_id,
        )

        # Format the output
        progress(0.9, desc="Formatting results...")
        output = ""
        for timestamp, text in chapters.items():
            output += f"{timestamp}: {text}\n"

        progress(1.0, desc="Complete!")
        return output


# CSS for the submit button color
head = """
<head>
    <title>Chapter-Llama - VidChapters</title>
    <link rel="icon" type="image/x-icon" href="./favicon.ico">
</head>
"""

title_markdown = """
<div style="display: flex; justify-content: space-between; align-items: center; background: linear-gradient(90deg, rgba(72,219,251,0.1), rgba(29,209,161,0.1)); border-radius: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px;">
    <div style="display: flex; align-items: center;">
        <a href="https://github.com/lucas-ventura/chapter-llama" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
            <img src="https://imagine.enpc.fr/~lucas.ventura/chapter-llama/images/chapter-llama.png" alt="Chapter-Llama" style="max-width: 100px; height: auto; border-radius: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        </a>
        <div>
            <h1 style="margin: 0; background: linear-gradient(90deg, #8F68C3, #477EF4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5em; font-weight: 700;">Chapter-Llama</h1>
            <h2 style="margin: 10px 0; background: linear-gradient(90deg, #8F68C3, #477EF4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.8em; font-weight: 600;">Efficient Chaptering in Hour-Long Videos with LLMs</h2>           
            <div style="display: flex; gap: 15px; margin-top: 10px;">
                <a href="https://github.com/lucas-ventura/chapter-llama" style="text-decoration: none; color: #8F68C3; font-weight: 500; transition: color 0.3s;">GitHub</a> |
                <a href="https://imagine.enpc.fr/~lucas.ventura/chapter-llama/" style="text-decoration: none; color: #8F68C3; font-weight: 500; transition: color 0.3s;">Project Page</a> |
                <a href="https://arxiv.org/abs/2504.00072" style="text-decoration: none; color: #8F68C3; font-weight: 500; transition: color 0.3s;">Paper</a>
            </div>
        </div>
    </div>
    <div style="text-align: right; margin-left: 20px;">
        <h2 style="margin: 10px 0; color: #24467C; font-weight: 700; font-size: 2.5em;">CVPR 2025</h2>
    </div>
</div>
"""

# Citation from demo_sample.py
bibtext = """
### Citation
```
@InProceedings{ventura25chapter,
  title = {{Chapter-Llama}: Efficient Chaptering in Hour-Long Videos with {LLM}s},
  author = {Lucas Ventura and Antoine Yang and Cordelia Schmid and G{\"u}l Varol},
  booktitle = {CVPR},
  year = {2025}
}
```
"""

# Create the Gradio interface
with gr.Blocks(title="Chapter-Llama", head=head) as demo:
    gr.HTML(title_markdown)
    gr.Markdown(
        """
        This demo is currently using only the audio data (ASR), without frame information. 
        We will add audio+captions functionality in the near future, which will improve 
        chapter generation by incorporating visual content.
        """
    )

    with gr.Row():
        with gr.Column():
            with gr.Tab("Upload File"):
                video_input = gr.File(
                    label="Upload Video or Audio File",
                    file_types=["video", "audio"],
                    type="binary",
                )

            with gr.Tab("Video URL"):
                video_url_input = gr.Textbox(
                    label="YouTube or Video URL",
                    placeholder="https://youtube.com/watch?v=...",
                )

            model_dropdown = gr.Dropdown(
                choices=["asr-10k", "asr-1k"],
                value="asr-10k",
                label="Select Model",
            )
            do_sample = gr.Checkbox(
                label="Use random sampling", value=False, interactive=True
            )
            submit_btn = gr.Button("Generate Chapters")

        with gr.Column():
            status_area = gr.Markdown("**Status:** Ready to process video")
            output_text = gr.Textbox(
                label="Generated Chapters", lines=12, interactive=False
            )

    def update_status_and_process(video_file, video_url, model_name, do_sample):
        if video_file is None and not video_url:
            return (
                "**Status:** No video uploaded or URL provided",
                "Please upload a video file or provide a URL.",
            )
        else:
            return "**Status:** Processing video...", process_video(
                video_file, video_url, model_name, do_sample
            )

    # Load the base model at startup
    load_base_model()

    submit_btn.click(
        fn=update_status_and_process,
        inputs=[video_input, video_url_input, model_dropdown, do_sample],
        outputs=[status_area, output_text],
    )

    gr.Markdown(bibtext)


if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch(share=True)
