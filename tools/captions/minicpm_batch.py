import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class MiniCPM:
    def __init__(self, model_name="HwwwH/MiniCPM-V-2"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.stop_token_ids = [self.tokenizer.eos_id]

        messages = [
            {
                "role": "user",
                "content":
                # Number of images
                "(<image>./</image>)" + "\nWhat is the content of this image?",
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        self.prompt = prompt

        dtype = "auto" if torch.cuda.get_device_capability()[0] >= 8 else "float16"
        self.model = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            dtype=dtype,
            tensor_parallel_size=1,
        )

    def caption_frames(self, frames, batch_size=4):
        if not isinstance(frames, list):
            frames = [frames]

        # Process frames in batches to avoid CUDA out of memory
        all_outputs = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]

            inputs = [
                {
                    "prompt": self.prompt,
                    "multi_modal_data": {"image": frame},
                }
                for frame in batch_frames
            ]
            sampling_params = SamplingParams(
                stop_token_ids=self.stop_token_ids,
                temperature=0,
                max_tokens=1024,
            )
            outputs = self.model.generate(
                inputs, sampling_params=sampling_params, use_tqdm=False
            )
            all_outputs.extend([output.outputs[0].text for output in outputs])

        return all_outputs
