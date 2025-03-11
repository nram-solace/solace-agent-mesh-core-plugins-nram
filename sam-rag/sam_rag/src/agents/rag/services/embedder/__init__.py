"""
Embedder package for converting text chunks to vector embeddings.
"""

from .embedder_base import EmbedderBase
from .local_embedder import (
    SentenceTransformerEmbedder,
    HuggingFaceEmbedder,
    OpenAICompatibleEmbedder,
)
from .cloud_embedder import (
    OpenAIEmbedder,
    AzureOpenAIEmbedder,
    CohereEmbedder,
    VertexAIEmbedder,
)
from .embedder_service import EmbedderService

__all__ = [
    "EmbedderBase",
    "SentenceTransformerEmbedder",
    "HuggingFaceEmbedder",
    "OpenAICompatibleEmbedder",
    "OpenAIEmbedder",
    "AzureOpenAIEmbedder",
    "CohereEmbedder",
    "VertexAIEmbedder",
    "EmbedderService",
]
