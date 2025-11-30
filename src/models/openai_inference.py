import logging
import os
from pathlib import Path

from lutils import openf
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = openf(str(Path.home() / ".openai.txt"))[0]
logger = logging.getLogger("Assitant")
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class OpenAIInference:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model_name

    def get_response(self, prompt: str):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        chapters = completion.choices[0].message.content

        return chapters

    def __call__(self, prompt: str, **kwargs):
        return self.get_response(prompt)
