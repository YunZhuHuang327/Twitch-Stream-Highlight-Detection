from pathlib import Path

import torch
from llama_cookbook.inference.model_utils import load_model as load_model_llamarecipes
from llama_cookbook.inference.model_utils import load_peft_model
from transformers import AutoTokenizer

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def load_model(
    ckpt_path, quantization=None, use_fast_kernels=False, peft_model=False, **kwargs
):
    model = load_model_llamarecipes(
        model_name=ckpt_path,
        quantization=quantization,
        use_fast_kernels=use_fast_kernels,
        device_map="auto",
        **kwargs,
    )
    if peft_model:
        model = load_peft_model(model, peft_model)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.pad_token = tokenizer.eos_token
    # special_tokens = {"additional_special_tokens": ["<image>"]}
    # tokenizer.add_special_tokens(special_tokens)

    return model, tokenizer


@torch.no_grad()
def inference(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    add_special_tokens: bool = True,
    temperature: float = 1.0,
    max_new_tokens=1024,
    top_p: float = 1.0,
    top_k: int = 50,
    use_cache: bool = True,
    max_padding_length: int = None,
    do_sample: bool = False,
    min_length: int = None,
    repetition_penalty: float = 1.0,
    length_penalty: int = 1,
    max_prompt_tokens: int = 35_000,
    **kwargs,
):
    """
    temperature: float, optional (default=1.0) The value used to module the next token probabilities.
    max_new_tokens: int, optional (default=1024) The maximum number of tokens to generate.
    top_p: float, optional (default=1.0) If set to float < 1 only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    top_k: int, optional (default=50) The number of highest probability vocabulary tokens to keep for top-k-filtering.
    use_cache: bool, optional (default=True) Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    max_padding_length: int, optional (default=None) the max padding length to be used with tokenizer padding the prompts.
    do_sample: bool, optional (default=True) Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int, optional (default=None) The minimum length of the sequence to be generated input prompt + min_new_tokens
    repetition_penalty: float, optional (default=1.0) The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int, optional (default=1) Exponential penalty to the length that is used with beam-based generation.
    """
    if add_special_tokens:
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        # prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    batch = tokenizer(
        prompt,
        truncation=True,
        max_length=max_padding_length,
        return_tensors="pt",
    )

    # if the input is too long, return the length of the input
    n_tokens = len(batch["input_ids"][0])
    if max_prompt_tokens is not None and n_tokens > max_prompt_tokens:
        return n_tokens

    batch = {k: v.to("cuda") for k, v in batch.items()}

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    try:
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        output = output_text.split("<|start_header_id|>assistant<|end_header_id|>")[1]
        output = output.strip()
        output = output.removesuffix("<|eot_id|>")

    except torch.cuda.OutOfMemoryError as e:
        log.error(f"CUDA out of memory error: {e}")
        torch.cuda.empty_cache()
        return n_tokens

    return output


class LlamaInference:
    def __init__(
        self,
        ckpt_path,
        quantization=None,
        use_fast_kernels=False,
        peft_model=False,
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
        **kwargs,
    ):
        # Check if LLaMA model exists
        # if not Path(ckpt_path).exists():
        #     log.warning(f"Model checkpoint does not exist at {ckpt_path}")
        #     return None

        # If PEFT model is specified, check if it exists
        if peft_model and not Path(peft_model).exists():
            log.warning(f"PEFT model does not exist at {peft_model}")
            return None
        if peft_model:
            log.info(f"PEFT model found at {peft_model}")

        # Bypass llama-recipes load_model to avoid offloading issues
        # llama-recipes uses device_map="auto" internally which causes PEFT loading to fail
        from transformers import AutoModelForCausalLM
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",  # Force all on GPU 0
            low_cpu_mem_usage=True,
        )
        if peft_model:
            model = load_peft_model(model, peft_model)

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        tokenizer.pad_token = tokenizer.eos_token

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
