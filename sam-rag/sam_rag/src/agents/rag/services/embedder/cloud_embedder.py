"""
Cloud-based embedders that use external APIs.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from .embedder_base import EmbedderBase


class OpenAIEmbedder(EmbedderBase):
    """
    Embedder using the OpenAI API.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the OpenAI embedder.

        Args:
            config: A dictionary containing configuration parameters.
                - api_key: The OpenAI API key (required).
                - model: The model to use (default: "text-embedding-ada-002").
                - batch_size: The batch size to use (default: 32).
                - dimensions: The dimensions of the embeddings (default: None, which uses the model's default).
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.model = self.config.get("model", "text-embedding-ada-002")
        self.dimensions = self.config.get("dimensions", None)
        self.client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up the OpenAI client.
        """
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
            )
        except ImportError:
            raise ImportError(
                "The openai package is required for OpenAIEmbedder. "
                "Please install it with `pip install openai`."
            )

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding.
        """
        if not text:
            # Return a zero vector of the correct dimension
            dim = self.get_embedding_dimension()
            return [0.0] * dim

        # Get the embedding from the API
        kwargs = {"model": self.model, "input": text}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = self.client.embeddings.create(**kwargs)

        # Extract the embedding
        embedding = response.data[0].embedding

        return embedding

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of text strings.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        if not texts:
            return []

        # Filter out empty texts
        non_empty_indices = [i for i, text in enumerate(texts) if text]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            # Return zero vectors of the correct dimension
            dim = self.get_embedding_dimension()
            return [[0.0] * dim for _ in texts]

        # Get the embeddings from the API
        kwargs = {"model": self.model, "input": non_empty_texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = self.client.embeddings.create(**kwargs)

        # Extract the embeddings
        embeddings = [data.embedding for data in response.data]

        # Reinsert zero vectors for empty texts
        result = []
        non_empty_idx = 0
        for i in range(len(texts)):
            if i in non_empty_indices:
                result.append(embeddings[non_empty_idx])
                non_empty_idx += 1
            else:
                # Add a zero vector of the correct dimension
                dim = (
                    len(embeddings[0]) if embeddings else self.get_embedding_dimension()
                )
                result.append([0.0] * dim)

        return result


class AzureOpenAIEmbedder(EmbedderBase):
    """
    Embedder using the Azure OpenAI API.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Azure OpenAI embedder.

        Args:
            config: A dictionary containing configuration parameters.
                - api_key: The Azure OpenAI API key (required).
                - api_version: The Azure OpenAI API version (default: "2023-05-15").
                - endpoint: The Azure OpenAI endpoint (required).
                - deployment: The Azure OpenAI deployment name (required).
                - batch_size: The batch size to use (default: 32).
                - dimensions: The dimensions of the embeddings (default: None, which uses the model's default).
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        if not self.api_key:
            raise ValueError("Azure OpenAI API key is required")

        self.api_version = self.config.get("api_version", "2023-05-15")
        self.endpoint = self.config.get("endpoint")
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint is required")

        self.deployment = self.config.get("deployment")
        self.model = self.config.get("model", "text-embedding-ada-002")
        if not self.deployment:
            raise ValueError("Azure OpenAI deployment name is required")

        self.dimensions = self.config.get("dimensions", None)
        self.client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up the Azure OpenAI client.
        """
        try:
            from openai import AzureOpenAI

            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
            )
        except ImportError:
            raise ImportError(
                "The openai package is required for AzureOpenAIEmbedder. "
                "Please install it with `pip install openai`."
            )

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding.
        """
        if not text:
            # Return a zero vector of the correct dimension
            dim = self.get_embedding_dimension()
            return [0.0] * dim

        # Get the embedding from the API
        kwargs = {"model": self.model, "input": text}

        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = self.client.embeddings.create(**kwargs)

        # Extract the embedding
        embedding = response.data[0].embedding

        return embedding

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of text strings.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        if not texts:
            return []

        # Filter out empty texts
        non_empty_indices = [i for i, text in enumerate(texts) if text]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            # Return zero vectors of the correct dimension
            dim = self.get_embedding_dimension()
            return [[0.0] * dim for _ in texts]

        # Get the embeddings from the API
        # kwargs = {"deployment_id": self.deployment, "input": non_empty_texts}
        kwargs = {"model": self.model, "input": non_empty_texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = self.client.embeddings.create(**kwargs)

        # Extract the embeddings
        embeddings = [data.embedding for data in response.data]

        # Reinsert zero vectors for empty texts
        result = []
        non_empty_idx = 0
        for i in range(len(texts)):
            if i in non_empty_indices:
                result.append(embeddings[non_empty_idx])
                non_empty_idx += 1
            else:
                # Add a zero vector of the correct dimension
                dim = (
                    len(embeddings[0]) if embeddings else self.get_embedding_dimension()
                )
                result.append([0.0] * dim)

        return result


class CohereEmbedder(EmbedderBase):
    """
    Embedder using the Cohere API.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Cohere embedder.

        Args:
            config: A dictionary containing configuration parameters.
                - api_key: The Cohere API key (required).
                - model: The model to use (default: "embed-english-v3.0").
                - batch_size: The batch size to use (default: 32).
                - input_type: The input type (default: "search_document").
                - truncate: The truncation strategy (default: "END").
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        if not self.api_key:
            raise ValueError("Cohere API key is required")

        self.model = self.config.get("model", "embed-english-v3.0")
        self.input_type = self.config.get("input_type", "search_document")
        self.truncate = self.config.get("truncate", "END")
        self.client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up the Cohere client.
        """
        try:
            import cohere

            self.client = cohere.Client(self.api_key)
        except ImportError:
            raise ImportError(
                "The cohere package is required for CohereEmbedder. "
                "Please install it with `pip install cohere`."
            )

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding.
        """
        if not text:
            # Return a zero vector of the correct dimension
            dim = self.get_embedding_dimension()
            return [0.0] * dim

        # Get the embedding from the API
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type=self.input_type,
            truncate=self.truncate,
        )

        # Extract the embedding
        embedding = response.embeddings[0]

        return embedding

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of text strings.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        if not texts:
            return []

        # Filter out empty texts
        non_empty_indices = [i for i, text in enumerate(texts) if text]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            # Return zero vectors of the correct dimension
            dim = self.get_embedding_dimension()
            return [[0.0] * dim for _ in texts]

        # Get the embeddings from the API
        response = self.client.embed(
            texts=non_empty_texts,
            model=self.model,
            input_type=self.input_type,
            truncate=self.truncate,
        )

        # Extract the embeddings
        embeddings = response.embeddings

        # Reinsert zero vectors for empty texts
        result = []
        non_empty_idx = 0
        for i in range(len(texts)):
            if i in non_empty_indices:
                result.append(embeddings[non_empty_idx])
                non_empty_idx += 1
            else:
                # Add a zero vector of the correct dimension
                dim = (
                    len(embeddings[0]) if embeddings else self.get_embedding_dimension()
                )
                result.append([0.0] * dim)

        return result


class VertexAIEmbedder(EmbedderBase):
    """
    Embedder using the Google Vertex AI API.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Vertex AI embedder.

        Args:
            config: A dictionary containing configuration parameters.
                - project_id: The Google Cloud project ID (required).
                - location: The Google Cloud location (default: "us-central1").
                - model: The model to use (default: "textembedding-gecko@latest").
                - batch_size: The batch size to use (default: 32).
                - credentials_path: The path to the Google Cloud credentials file (optional).
        """
        super().__init__(config)
        self.project_id = self.config.get("project_id")
        if not self.project_id:
            raise ValueError("Google Cloud project ID is required")

        self.location = self.config.get("location", "us-central1")
        self.model = self.config.get("model", "textembedding-gecko@latest")
        self.credentials_path = self.config.get("credentials_path", None)
        self.client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up the Vertex AI client.
        """
        try:
            import os
            from google.cloud import aiplatform

            # Set credentials if provided
            if self.credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path

            # Initialize the Vertex AI client
            aiplatform.init(project=self.project_id, location=self.location)

            # Import the text embedding model
            from vertexai.language_models import TextEmbeddingModel

            self.client = TextEmbeddingModel.from_pretrained(self.model)
        except ImportError:
            raise ImportError(
                "The google-cloud-aiplatform and vertexai packages are required for VertexAIEmbedder. "
                "Please install them with `pip install google-cloud-aiplatform vertexai`."
            )

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding.
        """
        if not text:
            # Return a zero vector of the correct dimension
            dim = self.get_embedding_dimension()
            return [0.0] * dim

        # Get the embedding from the API
        embeddings = self.client.get_embeddings([text])

        # Extract the embedding
        embedding = embeddings[0].values

        return embedding

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of text strings.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        if not texts:
            return []

        # Filter out empty texts
        non_empty_indices = [i for i, text in enumerate(texts) if text]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            # Return zero vectors of the correct dimension
            dim = self.get_embedding_dimension()
            return [[0.0] * dim for _ in texts]

        # Get the embeddings from the API
        embeddings_response = self.client.get_embeddings(non_empty_texts)

        # Extract the embeddings
        embeddings = [embedding.values for embedding in embeddings_response]

        # Reinsert zero vectors for empty texts
        result = []
        non_empty_idx = 0
        for i in range(len(texts)):
            if i in non_empty_indices:
                result.append(embeddings[non_empty_idx])
                non_empty_idx += 1
            else:
                # Add a zero vector of the correct dimension
                dim = (
                    len(embeddings[0]) if embeddings else self.get_embedding_dimension()
                )
                result.append([0.0] * dim)

        return result
