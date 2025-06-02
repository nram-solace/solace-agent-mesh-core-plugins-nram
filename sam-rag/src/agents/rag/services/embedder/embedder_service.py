from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from typing import Dict, Any, List, Optional, Callable
import nltk
import re
import string  # For punctuation removal
from solace_ai_connector.common.log import log as logger

# Ensure NLTK stopwords are available during module load or first use.
# This is a common pattern but can be moved to __init__ if preferred for explicitness.
try:
    stopwords.words("english")  # Check for default language stopwords
except LookupError:
    nltk.download("stopwords")
"""
Service for embedding text chunks into vector representations.
"""

from typing import Dict, Any, List, Tuple, Optional
import random  # For potential future use or more complex placeholders
import numpy as np

from .embedder_base import EmbedderBase

from .litellm_embedder import LiteLLMEmbedder


class EmbedderService:
    """
    Service for embedding text chunks into vector representations.
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        hybrid_search_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the embedder service.

        Args:
            config: A dictionary containing configuration parameters for the embedder.
                - embedder_type: The type of embedder to use (default: "openai").
                - embedder_params: The parameters to pass to the embedder.
            hybrid_search_config: Optional dictionary containing hybrid search configuration.
                - enabled: Boolean flag to enable/disable hybrid search.
        """
        self.config = config or {}
        self.embedder_type = self.config.get("embedder_type", "openai")
        self.embedder_params = self.config.get("embedder_params", {})
        self.normalize = self.config.get("normalize_embeddings", True)

        _hybrid_search_config = hybrid_search_config or {}
        self.hybrid_search_enabled = _hybrid_search_config.get("enabled", False)

        # Sparse model related attributes
        self.sparse_model_config = {}
        self.sparse_model_type = None
        self.tfidf_vectorizer = None
        self.tfidf_vocabulary_ = None
        self.tfidf_idf_ = None
        self.tokenizer_options = {}
        self._stop_words_set = None  # For caching stopwords
        self._sample_corpus_fitted = False  # Track if we've fitted a sample corpus

        if self.hybrid_search_enabled:
            # Load sparse_model_config from the main 'embedding' config section,
            # under its 'hybrid_search' subsection, as per rag.yaml structure.
            self.sparse_model_config = self.config.get("hybrid_search", {}).get(
                "sparse_model_config", {}
            )
            self.sparse_model_type = self.sparse_model_config.get(
                "type", "tfidf"
            ).lower()

            # Default tokenizer options, can be overridden by sparse_model_config.tokenizer_options
            default_tokenizer_opts = {
                "lowercase": True,
                "remove_stopwords": True,
                "stopword_language": "english",
                "stemming_method": None,  # No stemming by default
                "remove_punctuation": True,
                "remove_numbers": False,
            }
            # Allow overrides from sparse_model_config.tokenizer_options
            config_tokenizer_opts = self.sparse_model_config.get(
                "tokenizer_options", {}
            )
            self.tokenizer_options = {**default_tokenizer_opts, **config_tokenizer_opts}

            if self.tokenizer_options.get("remove_stopwords"):
                try:
                    self._stop_words_set = set(
                        stopwords.words(
                            self.tokenizer_options.get("stopword_language", "english")
                        )
                    )
                except LookupError:
                    logger.warning(
                        f"NLTK stopwords for '{self.tokenizer_options.get('stopword_language', 'english')}' not found. Downloading..."
                    )
                    nltk.download("stopwords")
                    self._stop_words_set = set(
                        stopwords.words(
                            self.tokenizer_options.get("stopword_language", "english")
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error loading NLTK stopwords: {e}. Stopword removal will be skipped."
                    )
                    self._stop_words_set = set()

            logger.info(
                f"Hybrid search enabled. Sparse model type: '{self.sparse_model_type}'. "
                f"Tokenizer options: {self.tokenizer_options}"
            )
            if self.sparse_model_type != "tfidf":
                logger.warning(
                    f"Sparse model type '{self.sparse_model_type}' is configured, but only 'tfidf' is currently implemented for sparse vector generation."
                )
        else:
            logger.info(
                "Hybrid search disabled. Sparse model components will not be actively used."
            )

        self.embedder = self._create_embedder()  # For dense embeddings

        # If hybrid search is enabled, fit the sparse model with a sample corpus
        if (
            self.hybrid_search_enabled
            and self.sparse_model_type == "tfidf"
            and not self._sample_corpus_fitted
        ):
            self._fit_sample_corpus()

    def _create_embedder(self) -> EmbedderBase:
        """
        Create the appropriate embedder based on the configuration.

        Returns:
            The embedder instance.
        """
        # Check if we should use litellm for cloud embedders
        if self.embedder_type in ["openai", "azure_openai", "cohere", "vertex_ai"]:
            # Map embedder_type to litellm model format
            model_prefix = {
                "openai": "openai/",
                "azure_openai": "azure/",
                "cohere": "cohere/",
                "vertex_ai": "vertex_ai/",
            }

            # Create a copy of the embedder params
            params = self.embedder_params.copy()

            # Add the model prefix to the model name if not already present
            model_name = params.get("model", "")
            if model_name and not any(
                model_name.startswith(prefix) for prefix in model_prefix.values()
            ):
                params["model"] = model_prefix[self.embedder_type] + model_name

            return LiteLLMEmbedder(params)

        # Use local embedders
        elif self.embedder_type == "litellm":
            # Direct use of LiteLLM embedder
            return LiteLLMEmbedder(self.embedder_params)
        else:
            raise ValueError(
                f"Unsupported embedder type: {self.embedder_type}"
            ) from None

    def _create_tokenizer(self) -> Callable[[str], List[str]]:
        """
        Creates a tokenizer function based on self.tokenizer_options.
        This function will be passed to TfidfVectorizer if TF-IDF is used.
        """
        options = self.tokenizer_options
        # Ensure _stop_words_set is available if needed, it's initialized in __init__
        stop_words_list = list(self._stop_words_set) if self._stop_words_set else []

        # Placeholder for stemmer, can be added if configured
        # stemmer = None
        # if options.get("stemming_method") == "porter":
        #     from nltk.stem.porter import PorterStemmer
        #     stemmer = PorterStemmer()
        # elif options.get("stemming_method") == "snowball":
        #     from nltk.stem.snowball import SnowballStemmer
        #     stemmer = SnowballStemmer(options.get("stopword_language", "english"))

        # Pre-compile regex for punctuation if used frequently
        punct_re = None
        if options.get("remove_punctuation"):
            # Create a regex to match all punctuation characters
            # string.punctuation typically includes: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
            punct_to_remove = string.punctuation
            punct_re = re.compile(f"[{re.escape(punct_to_remove)}]")

        def tokenizer_function(text_content: str) -> List[str]:
            if not isinstance(text_content, str):
                # TfidfVectorizer can sometimes pass unexpected types if input is weird
                logger.warning(
                    f"Tokenizer received non-string input: {type(text_content)}. Returning empty list."
                )
                return []

            if options.get("lowercase"):
                text_content = text_content.lower()

            if punct_re:  # Use pre-compiled regex
                text_content = punct_re.sub(
                    " ", text_content
                )  # Replace punctuation with space to avoid merging words

            # Basic tokenization by space, handles multiple spaces due to previous replacements
            tokens = text_content.split()

            if options.get("remove_numbers"):
                tokens = [
                    token
                    for token in tokens
                    if not token.isdigit() and not re.match(r"^-?\d+(?:\.\d+)?$", token)
                ]

            if (
                options.get("remove_stopwords") and self._stop_words_set
            ):  # Check self._stop_words_set directly
                tokens = [
                    token for token in tokens if token not in self._stop_words_set
                ]

            # if stemmer:
            #     tokens = [stemmer.stem(token) for token in tokens]

            # Filter out any remaining empty strings or very short tokens (e.g., single chars if not desired)
            return [token for token in tokens if token and len(token) > 1]

        return tokenizer_function

    def fit_sparse_model(self, corpus_texts: List[str]) -> None:
        """
        Fits the sparse model (e.g., TfidfVectorizer) on the provided corpus texts.
        This method should be called once during the ingestion pipeline setup
        if hybrid search is enabled.

        Args:
            corpus_texts: A list of text strings representing the entire corpus
                          (or a representative sample) to fit the model on.
        """
        logger.debug(
            f"[HYBRID_SEARCH_DEBUG] fit_sparse_model called with hybrid_search_enabled={self.hybrid_search_enabled}, sparse_model_type={self.sparse_model_type}"
        )

        if not self.hybrid_search_enabled:
            logger.info("Hybrid search is disabled. Skipping sparse model fitting.")
            return

        if self.sparse_model_type == "tfidf":
            if not corpus_texts:
                logger.warning("Corpus texts are empty. TF-IDF model cannot be fitted.")
                return

            try:
                logger.info(
                    f"Fitting TF-IDF model with {len(corpus_texts)} documents..."
                )
                logger.debug(
                    f"[HYBRID_SEARCH_DEBUG] Sample corpus texts (first 3): {corpus_texts[:3] if len(corpus_texts) >= 3 else corpus_texts}"
                )

                custom_tokenizer = self._create_tokenizer()

                # Adjust parameters based on corpus size to avoid min_df/max_df conflicts
                num_docs = len(corpus_texts)

                # For small corpora, adjust min_df and max_df to avoid conflicts
                if num_docs == 1:
                    # With only 1 document, use min_df=1 and max_df=1.0
                    min_df_param = 1
                    max_df_param = 1.0
                    logger.info(
                        "Single document detected. Adjusting TF-IDF parameters: min_df=1, max_df=1.0"
                    )
                elif num_docs < 5:
                    # For very small corpora, use min_df=1 to include all terms
                    min_df_param = 1
                    max_df_param = 0.95
                    logger.info(
                        f"Small corpus ({num_docs} docs) detected. Adjusting TF-IDF parameters: min_df=1, max_df=0.95"
                    )
                else:
                    # Use original parameters for larger corpora
                    min_df_param = 2
                    max_df_param = 0.95

                self.tfidf_vectorizer = TfidfVectorizer(
                    tokenizer=custom_tokenizer,
                    max_df=max_df_param,
                    min_df=min_df_param,
                    max_features=10000,
                    ngram_range=(1, 1),
                    use_idf=True,
                    smooth_idf=True,
                    sublinear_tf=False,
                )

                self.tfidf_vectorizer.fit(corpus_texts)
                self.tfidf_vocabulary_ = self.tfidf_vectorizer.vocabulary_
                self.tfidf_idf_ = self.tfidf_vectorizer.idf_
                logger.info(
                    f"TF-IDF model fitted successfully. Vocabulary size: {len(self.tfidf_vocabulary_)}"
                )
                logger.debug(
                    f"[HYBRID_SEARCH_DEBUG] TF-IDF parameters: min_df={min_df_param}, max_df={max_df_param}, max_features=10000"
                )
                logger.debug(
                    f"[HYBRID_SEARCH_DEBUG] Sample vocabulary terms: {list(self.tfidf_vocabulary_.keys())[:10]}"
                )
            except Exception as e:
                logger.error(f"Error fitting TF-IDF model: {e}", exc_info=True)
                self.tfidf_vectorizer = None
                self.tfidf_vocabulary_ = None
                self.tfidf_idf_ = None
        else:
            logger.warning(
                f"Sparse model type '{self.sparse_model_type}' is configured, "
                "but no fitting logic is implemented for it. Skipping fitting."
            )

    def embed_text(self, text: str) -> Dict[str, Any]:
        """
        Embed a single text string, generating both dense and sparse (if enabled) vectors.

        Args:
            text: The text to embed.

        Returns:
            A dictionary containing the dense vector and optionally a sparse vector.
            Example: {"dense_vector": [0.1, ...], "sparse_vector": {"0": 1.0, ...}}
        """
        dense_vector: Optional[List[float]] = None
        sparse_vector_dict: Optional[Dict[int, float]] = (
            None  # Using int for keys as per TF-IDF output
        )

        logger.debug(
            f"[HYBRID_SEARCH_DEBUG] embed_text called with text length: {len(text) if text else 0}, hybrid_search_enabled: {self.hybrid_search_enabled}"
        )

        if not text:
            logger.warning("EmbedderService.embed_text received empty or None text.")
            # Try to get dimension for a zero vector if embedder is available
            try:
                if self.embedder and hasattr(self.embedder, "get_embedding_dimension"):
                    dim = self.embedder.get_embedding_dimension()
                    if dim:
                        dense_vector = [0.0] * dim
            except Exception as e:
                logger.error(
                    f"Could not determine embedding dimension for empty text: {e}"
                )
            # sparse_vector_dict remains None, which will become {} if Qdrant needs it later.
            # For now, None indicates no sparse vector could be generated.
            # Let's default to {} for sparse if hybrid search is on, to be safe with DBs.
            if self.hybrid_search_enabled:
                sparse_vector_dict = {}
            return {"dense_vector": dense_vector, "sparse_vector": sparse_vector_dict}

        # 1. Generate Dense Embedding
        try:
            dense_vector = self.embedder.embed_text(text)
            if self.normalize and dense_vector is not None:
                # Ensure normalization happens only if dense_vector is not None
                # and the embedder supports it or we have a utility.
                if hasattr(self.embedder, "normalize_embedding"):
                    dense_vector = self.embedder.normalize_embedding(dense_vector)
                elif hasattr(
                    self, "_normalize_vector"
                ):  # Fallback to local utility if embedder doesn't have it
                    dense_vector = self._normalize_vector(dense_vector)
        except Exception as e:
            logger.error(f"Error generating dense embedding: {e}", exc_info=True)
            # Attempt to return a zero vector of appropriate dimension if possible
            try:
                if self.embedder and hasattr(self.embedder, "get_embedding_dimension"):
                    dim = self.embedder.get_embedding_dimension()
                    if dim:
                        dense_vector = [0.0] * dim
            except Exception as dim_e:
                logger.error(
                    f"Could not determine embedding dimension after dense embedding error: {dim_e}"
                )

        # 2. Generate Sparse Embedding (conditionally)
        sparse_vector_dict = {}  # Default to empty dict for sparse if hybrid is on
        # Will be populated if successful, or remains {} if not/error

        if self.hybrid_search_enabled:
            if self.sparse_model_type == "tfidf":
                if (
                    self.tfidf_vectorizer
                    and hasattr(self.tfidf_vectorizer, "vocabulary_")
                    and self.tfidf_vectorizer.vocabulary_
                ):
                    try:
                        # The TfidfVectorizer's tokenizer (our custom_tokenizer) will be applied.
                        # transform expects an iterable of documents.
                        vector_transformed = self.tfidf_vectorizer.transform([text])

                        # Convert the sparse matrix row to {index: value} format
                        # vector_transformed is a csr_matrix of shape (1, num_features)
                        cx = vector_transformed.tocoo()
                        current_sparse_values = {}
                        for _, col_idx, val_data in zip(cx.row, cx.col, cx.data):
                            # col_idx is the feature index (term_index in our vocabulary)
                            # val_data is the tf-idf score
                            current_sparse_values[int(col_idx)] = float(val_data)

                        sparse_vector_dict = current_sparse_values  # This will be {} if no terms were found
                        # which is the desired format for Qdrant.

                        # Log the sparse vector details for debugging
                        if sparse_vector_dict:
                            logger.info(
                                f"Generated non-empty sparse vector with {len(sparse_vector_dict)} elements."
                            )
                            logger.debug(
                                f"[HYBRID_SEARCH_DEBUG] Sparse vector sample terms: {dict(list(sparse_vector_dict.items())[:5])}"
                            )

                            # Print detailed TF-IDF output
                            logger.info(
                                f"[HYBRID_SEARCH_DEBUG] TF-IDF OUTPUT for text: '{text[:50]}...'"
                            )
                            logger.info(
                                f"[HYBRID_SEARCH_DEBUG] Full sparse vector: {sparse_vector_dict}"
                            )

                            # # Map indices back to actual terms for better understanding
                            # if self.tfidf_vocabulary_:
                            #     reverse_vocab = {
                            #         v: k for k, v in self.tfidf_vocabulary_.items()
                            #     }
                            #     term_scores = {}
                            #     for idx, score in sparse_vector_dict.items():
                            #         if idx in reverse_vocab:
                            #             term_scores[reverse_vocab[idx]] = score
                            #     logger.info(
                            #         f"[HYBRID_SEARCH_DEBUG] TF-IDF terms with scores: {term_scores}"
                            #     )

                            #     # Show top scoring terms
                            #     sorted_terms = sorted(
                            #         term_scores.items(),
                            #         key=lambda x: x[1],
                            #         reverse=True,
                            #     )
                            #     logger.info(
                            #         f"[HYBRID_SEARCH_DEBUG] Top 10 TF-IDF terms: {sorted_terms[:10]}"
                            #     )
                        else:
                            logger.warning(
                                "Generated empty sparse vector. This may indicate an issue with the TF-IDF model or the input text."
                            )
                            logger.debug(
                                f"[HYBRID_SEARCH_DEBUG] Text preview for empty sparse vector: '{text[:100]}...'"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error generating TF-IDF sparse vector for text: {e}",
                            exc_info=True,
                        )
                        # sparse_vector_dict remains {}
                else:
                    logger.warning(
                        "Hybrid search enabled and TF-IDF type configured, but TF-IDF model is not fitted or vocabulary is empty. "
                        "Cannot generate TF-IDF sparse vector. Returning empty sparse vector."
                    )
                    # sparse_vector_dict remains {}
            else:
                logger.warning(
                    f"Hybrid search enabled, but sparse model type '{self.sparse_model_type}' "
                    "is not 'tfidf' or not implemented for generation. Returning empty sparse vector."
                )
                # sparse_vector_dict remains {}
        else:  # Hybrid search not enabled
            sparse_vector_dict = None  # Explicitly None if hybrid search is off

        return {"dense_vector": dense_vector, "sparse_vector": sparse_vector_dict}

    def embed_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Embed multiple text strings.

        Args:
            texts: The texts to embed.

        Returns:
            A list of dictionaries, where each dictionary contains dense and optional sparse vectors.
        """
        if not texts:
            return []

        results = []
        # TODO: Batching for dense embeddings is handled by LiteLLMEmbedder.
        # Consider if sparse vector generation needs its own batching or if it's efficient enough per text.
        for text_content in texts:
            results.append(self.embed_text(text_content))
        return results

    def embed_chunks(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Embed a list of text chunks.

        Args:
            chunks: The text chunks to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        return self.embed_texts(chunks)

    def embed_file_chunks(
        self, file_chunks: List[Tuple[str, List[str]]]
    ) -> Dict[str, List[List[float]]]:
        """
        Embed chunks from multiple files.

        Args:
            file_chunks: A list of tuples containing (file_path, chunks).

        Returns:
            A dictionary mapping file paths to lists of embeddings.
        """
        result = {}

        for file_path, chunks in file_chunks:
            # Embed the chunks for this file
            embeddings = self.embed_chunks(chunks)

            # Add to the result
            result[file_path] = embeddings

        return result

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this embedder.

        Returns:
            The dimension of the embeddings.
        """
        return self.embedder.get_embedding_dimension()

    def cosine_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Calculate the cosine similarity between two embeddings.

        Args:
            embedding1: The first embedding.
            embedding2: The second embedding.

        Returns:
            The cosine similarity between the embeddings.
        """
        return self.embedder.cosine_similarity(embedding1, embedding2)

    def _fit_sample_corpus(self) -> None:
        """
        Fits the sparse model with a sample corpus of common English words and phrases.
        This is used when no actual corpus is available yet, to ensure the TF-IDF model
        is initialized and can generate sparse vectors.
        """
        if not self.hybrid_search_enabled or self.sparse_model_type != "tfidf":
            return

        logger.info("Fitting TF-IDF model with sample corpus...")

        # Create a sample corpus with common English words and phrases
        sample_corpus = [
            "This is a sample document for TF-IDF model fitting.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning and natural language processing are fascinating fields.",
            "Vector databases store and retrieve high-dimensional vectors efficiently.",
            "Hybrid search combines dense and sparse vector representations.",
            "Embeddings capture semantic meaning of text in vector space.",
            "Information retrieval systems help find relevant documents.",
            "Document similarity can be measured using cosine distance.",
            "Query expansion improves search results by adding related terms.",
            "Sparse vectors represent text using term frequency statistics.",
            "Dense vectors capture contextual relationships between words.",
            "Tokenization splits text into meaningful units for processing.",
            "Stop words are common words that are filtered out during text processing.",
            "Stemming reduces words to their root form for better matching.",
            "TF-IDF weighs terms based on their frequency in documents and corpus.",
        ]

        # Fit the TF-IDF model with the sample corpus
        self.fit_sparse_model(sample_corpus)
        self._sample_corpus_fitted = True
        logger.info("Sample corpus fitted successfully.")

    def search_similar(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Search for the most similar embeddings to a query embedding.

        Args:
            query_embedding: The query embedding.
            embeddings: The embeddings to search.
            top_k: The number of results to return.

        Returns:
            A list of tuples containing (index, similarity score).
        """
        if not embeddings:
            return []

        # Calculate similarities
        similarities = [
            self.cosine_similarity(query_embedding, embedding)
            for embedding in embeddings
        ]

        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]

        # Get the top-k results
        top_k_indices = sorted_indices[:top_k]

        # Return the indices and scores
        return [(int(idx), similarities[idx]) for idx in top_k_indices]
