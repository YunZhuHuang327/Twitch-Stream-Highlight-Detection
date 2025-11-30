from pathlib import Path

from google import genai
from lutils import openf

api_key = openf(str(Path.home() / ".gemini.txt"))[0]


class GeminiInference:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model_name

    def get_response(self, prompt: str):
        completion = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        try:
            chapters = completion.text.strip()
        except Exception as e:  # noqa: F841
            chapters = ""
        return chapters

    def __call__(self, prompt: str, **kwargs):
        return self.get_response(prompt)
