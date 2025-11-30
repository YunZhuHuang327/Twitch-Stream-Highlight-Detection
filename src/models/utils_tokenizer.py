# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import itertools

B_INST, E_INST = "[INST]", "[/INST]"
EOT_ID = 128009  # <|eot_id|>


def mask_target(target, seq):
    for i in range(len(seq) - len(target)):
        if seq[i : i + len(target)] == target:
            seq[i : i + len(target)] = [-100] * len(target)
    return seq


def tokenize_dialog(dialog, tokenizer):
    if tokenizer.vocab_size >= 128000:
        dialog_tokens = tokenizer.apply_chat_template(dialog)
        eot_indices = [i for i, n in enumerate(dialog_tokens) if n == EOT_ID]
        labels = copy.copy(dialog_tokens)
        # determine token for system and user
        system_or_user = (tokenizer.encode("system")[-1], tokenizer.encode("user")[-1])
        labels[0] = -100  # bos token
        last_idx = 1
        for _, idx in enumerate(eot_indices):
            role_token = labels[last_idx + 1]
            if role_token in system_or_user:
                # Set labels to -100 for system and user tokens to ignore in loss function
                labels[last_idx : idx + 1] = [-100] * (idx - last_idx + 1)
            last_idx = idx + 1
        mask_target(
            tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>",
                add_special_tokens=False,
            ),
            labels,
        )

        dialog_tokens = [dialog_tokens]
        labels_tokens = [labels]
    else:
        prompt_tokens = [
            tokenizer.encode(
                f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}",
                add_special_tokens=False,
            )
            for prompt in dialog[::2]
        ]
        answer_tokens = [
            tokenizer.encode(
                f"{answer['content'].strip()} {tokenizer.eos_token}",
                add_special_tokens=False,
            )
            for answer in dialog[1::2]
        ]
        dialog_tokens = list(
            itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens))
        )

        # Add labels, convert prompt token to -100 in order to ignore in loss function
        labels_tokens = [
            len(c)
            * [
                -100,
            ]
            if i % 2 == 0
            else c
            for i, c in enumerate(dialog_tokens)
        ]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1] * len(combined_tokens["input_ids"]))


class Tokenizer:
    def __init__(
        self,
        base_dir="./",
        ckpt_path="checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct/",
    ):
        import os

        from transformers import AutoTokenizer

        ckpt_path = os.path.join(base_dir, ckpt_path)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, prompt):
        return self.tokenizer.encode(prompt)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def n_tokens(self, prompt):
        return len(self.encode(prompt))


if __name__ == "__main__":
    tokenizer = Tokenizer()
    prompt = "Hello, world!"
    prompt_tokens = tokenizer.encode(prompt)
    print(prompt_tokens)
    print(tokenizer.decode(prompt_tokens[1:]))
