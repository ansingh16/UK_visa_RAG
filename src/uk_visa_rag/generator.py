"""LLM generation backends."""

import logging
from typing import Dict, Any

import ollama

logger = logging.getLogger(__name__)


class OllamaGenerator:
    """Generate answers using a local Ollama model."""

    def __init__(
        self,
        model: str = "mistral:instruct",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ):
        self.model = model
        self.client = ollama.Client(host=base_url)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Send *prompt* to Ollama and return the response text."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                },
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            return "I apologize, but I'm having trouble generating a response right now."

    # LangChain compatibility
    def run(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return {"replies": [self.generate(prompt)]}
