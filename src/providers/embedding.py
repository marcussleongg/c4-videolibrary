"""Embedding provider via Gemini API."""
from __future__ import annotations

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION


class GeminiEmbedder:
    """Implements EmbeddingProvider using Gemini embedding models."""

    def __init__(self, model: str | None = None):
        self._model = model or EMBEDDING_MODEL
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    def embed(self, text: str) -> list[float]:
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION,
            ),
        )
        return result.embeddings[0].values

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION,
            ),
        )
        return [e.values for e in result.embeddings]
