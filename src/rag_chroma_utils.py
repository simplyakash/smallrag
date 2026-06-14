"""Shared Chroma helpers that avoid hard Torch or ONNX dependencies."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re


LOGGER = logging.getLogger("rag_chroma_utils")
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
DEFAULT_HASH_EMBEDDING_DIMENSIONS = 384


class HashEmbeddingFunction:
    """Small deterministic embedding function for constrained demos.

    This is not as semantically strong as a transformer embedding model, but it
    keeps the AG News RAG example runnable without downloading Torch or ONNX.
    """

    def __init__(self, dimensions: int = DEFAULT_HASH_EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions

    @staticmethod
    def name() -> str:
        # Chroma treats "default" as compatible with collections created by its
        # built-in embedding function, while we still provide the callable here.
        return "default"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"\w+", text.lower())

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector

        return [value / norm for value in vector]


def build_embedding_function() -> object:
    """Return the configured Chroma embedding function.

    Set ``RAG_EMBEDDING_BACKEND=sentence-transformers`` to use
    ``all-MiniLM-L6-v2``. The default avoids Torch and ONNX by using a
    deterministic hash embedding function.
    """

    backend = os.getenv("RAG_EMBEDDING_BACKEND", "hash").strip().lower()
    if backend == "sentence-transformers":
        from chromadb.utils import embedding_functions

        model_name = os.getenv(
            "RAG_SENTENCE_TRANSFORMER_MODEL",
            DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        )
        LOGGER.info("[build_embedding_function] using sentence-transformers=%s", model_name)
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

    dimensions = int(
        os.getenv("RAG_HASH_EMBEDDING_DIMENSIONS", str(DEFAULT_HASH_EMBEDDING_DIMENSIONS))
    )
    LOGGER.info("[build_embedding_function] using hash embeddings dimensions=%d", dimensions)
    return HashEmbeddingFunction(dimensions=dimensions)
