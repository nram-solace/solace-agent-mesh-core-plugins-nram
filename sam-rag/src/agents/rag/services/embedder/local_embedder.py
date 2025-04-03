"""
Local embedders that run on the local machine.
"""

from typing import Dict, Any, List

from .embedder_base import EmbedderBase


class SentenceTransformerEmbedder(EmbedderBase):
    """
    Embedder using the sentence-transformers library.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the sentence-transformers embedder.

        Args:
            config: A dictionary containing configuration parameters.
                - model_name: The name of the sentence-transformers model to use (default: "all-MiniLM-L6-v2").
                - device: The device to use for inference (default: None, which uses the default device).
                - normalize_embeddings: Whether to normalize the embeddings (default: True).
        """
        super().__init__(config)
        self.model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
        self.device = self.config.get("device", None)
        self.normalize = self.config.get("normalize_embeddings", True)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the sentence-transformers model.
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            raise ImportError(
                "The sentence-transformers package is required for SentenceTransformerEmbedder. "
                "Please install it with `pip install sentence-transformers`."
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

        # Embed the text
        embedding = self.model.encode(text, normalize_embeddings=self.normalize)

        # Convert to list
        return embedding.tolist()

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
        non_empty_texts = [text for text in texts if text]

        if not non_empty_texts:
            # Return zero vectors of the correct dimension
            dim = self.get_embedding_dimension()
            return [[0.0] * dim for _ in texts]

        # Embed the texts
        embeddings = self.model.encode(
            non_empty_texts, normalize_embeddings=self.normalize
        )

        # Convert to list of lists
        embeddings_list = embeddings.tolist()

        # Reinsert zero vectors for empty texts
        result = []
        non_empty_idx = 0
        for text in texts:
            if text:
                result.append(embeddings_list[non_empty_idx])
                non_empty_idx += 1
            else:
                # Add a zero vector of the correct dimension
                dim = (
                    len(embeddings_list[0])
                    if embeddings_list
                    else self.get_embedding_dimension()
                )
                result.append([0.0] * dim)

        return result


class HuggingFaceEmbedder(EmbedderBase):
    """
    Embedder using the Hugging Face transformers library.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Hugging Face embedder.

        Args:
            config: A dictionary containing configuration parameters.
                - model_name: The name of the Hugging Face model to use (default: "sentence-transformers/all-MiniLM-L6-v2").
                - device: The device to use for inference (default: None, which uses the default device).
                - pooling_strategy: The pooling strategy to use (default: "mean").
                - normalize_embeddings: Whether to normalize the embeddings (default: True).
        """
        super().__init__(config)
        self.model_name = self.config.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.device = self.config.get("device", None)
        self.pooling_strategy = self.config.get("pooling_strategy", "mean")
        self.normalize = self.config.get("normalize_embeddings", True)
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the Hugging Face model and tokenizer.
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            # Set the device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load the model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        except ImportError:
            raise ImportError(
                "The transformers package is required for HuggingFaceEmbedder. "
                "Please install it with `pip install transformers`."
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

        # Import torch here to avoid importing it if not needed
        import torch

        # Tokenize the text
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Get the embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Pool the embeddings
        if self.pooling_strategy == "mean":
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            embedding = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.pooling_strategy == "cls":
            # CLS token pooling
            embedding = outputs.last_hidden_state[:, 0]
        else:
            # Default to mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            embedding = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Normalize if requested
        if self.normalize:
            embedding = embedding / embedding.norm(dim=1, keepdim=True)

        # Convert to list
        return embedding[0].cpu().numpy().tolist()

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

        # Import torch here to avoid importing it if not needed
        import torch

        # Tokenize the texts
        inputs = self.tokenizer(
            non_empty_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Get the embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Pool the embeddings
        if self.pooling_strategy == "mean":
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.pooling_strategy == "cls":
            # CLS token pooling
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            # Default to mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Normalize if requested
        if self.normalize:
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        # Convert to list of lists
        embeddings_list = embeddings.cpu().numpy().tolist()

        # Reinsert zero vectors for empty texts
        result = []
        non_empty_idx = 0
        for i in range(len(texts)):
            if i in non_empty_indices:
                result.append(embeddings_list[non_empty_idx])
                non_empty_idx += 1
            else:
                # Add a zero vector of the correct dimension
                dim = (
                    len(embeddings_list[0])
                    if embeddings_list
                    else self.get_embedding_dimension()
                )
                result.append([0.0] * dim)

        return result


class OpenAICompatibleEmbedder(EmbedderBase):
    """
    Embedder using a local model that is compatible with the OpenAI API.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the OpenAI-compatible embedder.

        Args:
            config: A dictionary containing configuration parameters.
                - base_url: The base URL of the API (default: "http://localhost:8000").
                - api_key: The API key to use (default: "").
                - model: The model to use (default: "text-embedding-ada-002").
                - batch_size: The batch size to use (default: 32).
        """
        super().__init__(config)
        self.base_url = self.config.get("base_url", "http://localhost:8000")
        self.api_key = self.config.get("api_key", "")
        self.model = self.config.get("model", "text-embedding-ada-002")
        self.client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up the OpenAI client.
        """
        try:
            from openai import OpenAI

            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        except ImportError:
            raise ImportError(
                "The openai package is required for OpenAICompatibleEmbedder. "
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
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

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
        response = self.client.embeddings.create(
            model=self.model,
            input=non_empty_texts,
        )

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
