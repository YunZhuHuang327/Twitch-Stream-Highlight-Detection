# test.py
import torch
from transformers import AutoModel, AutoTokenizer, logging

logging.set_verbosity_error()


class MiniCPM:
    def __init__(self, model_name="openbmb/MiniCPM-V-2_6"):
        # Use bfloat16 for A100/H100, float16 for V100
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=dtype,
        )  # sdpa or flash_attention_2, no eager

        # self.model = self.model.to(device="cuda", dtype=dtype)
        self.model = self.model.to(device="cuda")
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    def chat(self, image, question="", msgs=None, sampling=False, **params):
        # msgs = [{"role": "user", "content": [image, question]}]
        # return self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        if msgs is None:
            assert question, "Either question or msgs must be provided"
            msgs = [{"role": "user", "content": question}]

        return self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            # temperature=0.6,
            sampling=sampling,  # if sampling=False, beam_search will be used by default
            **params,
        )
