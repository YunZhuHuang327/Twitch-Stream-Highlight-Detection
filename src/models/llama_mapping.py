from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn

# load processor and model
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.activations import GELUActivation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiModalProjector(nn.Module):
    def __init__(self, finetuned=True):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=1152, out_features=4096, bias=True)
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(in_features=4096, out_features=4096, bias=True)

        self.finetuned = finetuned
        if not finetuned:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x

    @classmethod
    def from_pretrained(cls, ckpt_path, finetuned=False):
        if ckpt_path is None or not Path(ckpt_path).exists():
            return cls(finetuned=finetuned)
        instance = cls(finetuned=finetuned)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        instance.load_state_dict(state_dict)
        return instance


class ImageProcessorSigLip:
    def __init__(self):
        model_name = "google/siglip-so400m-patch14-384"
        self.image_processor = AutoProcessor.from_pretrained(model_name)
        self.image_model = AutoModel.from_pretrained(
            model_name,
            # attn_implementation="flash_attention_2",
        )
        self.image_model.to(device)

    def __call__(self, images: List[Union[str, Image.Image]]):
        if isinstance(images[0], str):
            images = [Image.open(image) for image in images]

        pixel_values = self.image_processor(images=images, return_tensors="pt")[
            "pixel_values"
        ]
        pixel_values = pixel_values.to(self.image_model.device)

        with torch.no_grad():
            image_features = self.image_model.get_image_features(
                pixel_values=pixel_values
            )
        return image_features


# Code from huggingface transformers
def merge_input_ids_with_image_features(
    image_features,
    inputs_embeds,
    input_ids,
    attention_mask,
    labels=None,
    image_token_index=128002,
    ignore_index=-100,
):
    # image_features torch.Size([2, 728, 4096])
    # inputs_embeds torch.Size([1, 66, 4096])
    # input_ids torch.Size([1, 66])

    if len(image_features.shape) == 2:
        image_features = image_features.unsqueeze(1)

    num_images, num_image_patches, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape
    # 1. Create a mask to know where special image tokens are
    special_image_token_mask = input_ids == image_token_index
    # count the number of special image tokens
    n_image_tokens = torch.sum(special_image_token_mask, dim=-1).item()
    assert n_image_tokens == num_images, (
        f"Number of image tokens {n_image_tokens} does not match the number of images {num_images}"
    )

    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    # Compute the maximum embed dimension
    max_embed_dim = (
        num_special_image_tokens.max() * (num_image_patches - 1)
    ) + sequence_length
    batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

    # 2. Compute the positions where text should be written
    # Calculate new positions for text tokens in merged image-text sequence.
    # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
    # `torch.cumsum` computes how each image token shifts subsequent text token positions.
    # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
    new_token_positions = (
        torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
    )
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    # 3. Create the full embedding, already padded to the maximum position
    final_embedding = torch.zeros(
        batch_size,
        max_embed_dim,
        embed_dim,
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    final_attention_mask = torch.zeros(
        batch_size,
        max_embed_dim,
        dtype=attention_mask.dtype,
        device=inputs_embeds.device,
    )
    if labels is not None:
        final_labels = torch.full(
            (batch_size, max_embed_dim),
            ignore_index,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
    # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
    # set the corresponding tensors into their correct target device.
    target_device = inputs_embeds.device
    batch_indices, non_image_indices, text_to_overwrite = (
        batch_indices.to(target_device),
        non_image_indices.to(target_device),
        text_to_overwrite.to(target_device),
    )
    attention_mask = attention_mask.to(target_device)

    # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
    # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
        batch_indices, non_image_indices
    ]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
        batch_indices, non_image_indices
    ]
    if labels is not None:
        final_labels[batch_indices, text_to_overwrite] = labels[
            batch_indices, non_image_indices
        ]

    # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
    image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(
        target_device
    )

    if image_to_overwrite.sum() != image_features.shape[:-1].numel():
        raise ValueError(
            f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
            f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
        )

    final_embedding[image_to_overwrite] = (
        image_features.contiguous()
        .reshape(-1, embed_dim)
        .to(dtype=final_embedding.dtype, device=target_device)
    )
    final_attention_mask |= image_to_overwrite
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
        (final_attention_mask == 0), 1
    )

    if labels is None:
        final_labels = None

    return final_embedding, final_attention_mask, final_labels, position_ids


def generate_with_images(
    model,
    tokenizer,
    processor,
    prompt: str,
    images: Optional[List[Union[str, Image.Image]]] = None,
    max_length: Optional[int] = None,
    temperature: Optional[float] = None,
    do_sample: bool = False,
    top_p: float = 1.0,
):
    """
    Generate text with optional image inputs using a multimodal model.

    Args:
        model: The language model
        tokenizer: The tokenizer
        processor: Image processor for visual inputs
        prompt: Text prompt
        images: Optional list of image paths or PIL Images
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        top_p: Top-p sampling parameter
    """
    # Process text inputs
    batch = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    n_tokens = len(batch["input_ids"][0])

    # Move text inputs to device
    batch = {k: v.to(model.device) for k, v in batch.items()}

    # If we have images, process them
    if images is not None:
        # Convert string paths to PIL Images if needed
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            processed_images.append(img)

        # Process images with the processor
        image_inputs = processor(
            images=processed_images,
            text=prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        # Move image inputs to device
        for k, v in image_inputs.items():
            if v is not None:
                if isinstance(v, torch.Tensor):
                    image_inputs[k] = v.to(model.device)
                elif isinstance(v, list):
                    image_inputs[k] = [x.to(model.device) for x in v]

        # Get input embeddings
        input_ids = batch["input_ids"]
        inputs_embeds = model.model.embed_tokens(input_ids)

        # Merge text and image features
        merged_embeds, merged_attention_mask, _, position_ids = (
            merge_input_ids_with_image_features(
                image_features=image_inputs["image_features"],
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=batch["attention_mask"],
                image_token_index=processor.tokenizer.convert_tokens_to_ids(
                    "<|reserved_special_token_0|>"
                ),
            )
        )

        # Update batch with merged features
        batch["inputs_embeds"] = merged_embeds
        batch["attention_mask"] = merged_attention_mask
        batch["position_ids"] = position_ids
        del batch["input_ids"]
    else:
        # For text-only, convert input_ids to embeddings
        input_ids = batch["input_ids"]
        inputs_embeds = model.model.embed_tokens(input_ids)
        batch["inputs_embeds"] = inputs_embeds
        del batch["input_ids"]

    # Set up generation terminators
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    # Generate output
    outputs = model.generate(
        **batch,
        do_sample=do_sample,
        eos_token_id=terminators,
        temperature=temperature,
        max_length=n_tokens + 1000 if max_length is None else max_length,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    output_text = output_text.split("<|start_header_id|>assistant<|end_header_id|>")[1]
    output_text = output_text.strip()
    output_text = output_text.removesuffix("<|eot_id|>")

    return output_text
