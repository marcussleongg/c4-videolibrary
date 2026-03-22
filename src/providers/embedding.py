"""OpenAI embedding provider via OpenRouter."""
from __future__ import annotations

from openai import OpenAI

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_MODEL


class OpenAIEmbedder:
    """Implements EmbeddingProvider using OpenAI text-embedding models via OpenRouter."""

    def __init__(self, model: str | None = None):
        self._model = model or EMBEDDING_MODEL
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in resp.data]
