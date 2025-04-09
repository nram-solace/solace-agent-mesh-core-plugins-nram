"""
Unit tests for the Retriever class.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.agents.rag.services.rag.retriever import Retriever


class TestRetriever(unittest.TestCase):
    """Test cases for the Retriever class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock configuration
        self.config = {
            "embedding": {
                "embedder_type": "openai",
                "embedder_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "test-key",
                },
            },
            "vector_db": {
                "db_type": "qdrant",
                "db_params": {
                    "url": "http://localhost:6333",
                    "collection_name": "test_collection",
                },
            },
        }

        # Create patcher for EmbedderService
        self.embedder_patcher = patch(
            "src.agents.rag.services.rag.retriever.EmbedderService"
        )
        self.mock_embedder_service = self.embedder_patcher.start()
        self.mock_embedder_instance = self.mock_embedder_service.return_value

        # Create patcher for VectorDBService
        self.vector_db_patcher = patch(
            "src.agents.rag.services.rag.retriever.VectorDBService"
        )
        self.mock_vector_db_service = self.vector_db_patcher.start()
        self.mock_vector_db_instance = self.mock_vector_db_service.return_value

        # Create retriever instance
        self.retriever = Retriever(self.config)

    def tearDown(self):
        """Tear down test fixtures."""
        self.embedder_patcher.stop()
        self.vector_db_patcher.stop()

    def test_init(self):
        """Test initialization."""
        # Check that services were initialized with correct configs
        self.mock_embedder_service.assert_called_once_with(self.config["embedding"])
        self.mock_vector_db_service.assert_called_once_with(self.config["vector_db"])

    def test_get_query_embedding(self):
        """Test get_query_embedding method."""
        # Setup mock
        mock_embedding = [0.1, 0.2, 0.3]
        self.mock_embedder_instance.embed_texts.return_value = [mock_embedding]

        # Call the method
        result = self.retriever.get_query_embedding("test query")

        # Assertions
        self.mock_embedder_instance.embed_texts.assert_called_once_with(["test query"])
        self.assertEqual(result, mock_embedding)

    def test_get_query_embedding_error(self):
        """Test get_query_embedding method with error."""
        # Setup mock to return empty list
        self.mock_embedder_instance.embed_texts.return_value = []

        # Call the method and check for exception
        with self.assertRaises(ValueError):
            self.retriever.get_query_embedding("test query")

    def test_retrieve(self):
        """Test retrieve method."""
        # Setup mocks
        mock_embedding = [0.1, 0.2, 0.3]
        mock_results = [
            {"text": "doc1", "metadata": {"source": "file1"}, "score": 0.9},
            {"text": "doc2", "metadata": {"source": "file2"}, "score": 0.8},
        ]
        self.mock_embedder_instance.embed_texts.return_value = [mock_embedding]
        self.mock_vector_db_instance.search.return_value = mock_results

        # Call the method
        result = self.retriever.retrieve("test query", top_k=2)

        # Assertions
        self.mock_embedder_instance.embed_texts.assert_called_once_with(["test query"])
        self.mock_vector_db_instance.search.assert_called_once_with(
            query_embedding=mock_embedding, top_k=2, filter=None
        )
        self.assertEqual(result, mock_results)

    def test_retrieve_with_filter(self):
        """Test retrieve method with filter."""
        # Setup mocks
        mock_embedding = [0.1, 0.2, 0.3]
        mock_results = [
            {"text": "doc1", "metadata": {"source": "file1"}, "score": 0.9},
        ]
        mock_filter = {"metadata.source": "file1"}
        self.mock_embedder_instance.embed_texts.return_value = [mock_embedding]
        self.mock_vector_db_instance.search.return_value = mock_results

        # Call the method
        result = self.retriever.retrieve("test query", top_k=1, filter=mock_filter)

        # Assertions
        self.mock_vector_db_instance.search.assert_called_once_with(
            query_embedding=mock_embedding, top_k=1, filter=mock_filter
        )
        self.assertEqual(result, mock_results)

    def test_retrieve_with_scores(self):
        """Test retrieve_with_scores method."""
        # Setup mocks
        mock_results = [
            {"text": "doc1", "metadata": {"source": "file1"}, "score": 0.9},
            {"text": "doc2", "metadata": {"source": "file2"}, "score": 0.8},
        ]

        # Mock the retrieve method
        with patch.object(self.retriever, "retrieve", return_value=mock_results):
            # Call the method
            texts, metadatas, scores = self.retriever.retrieve_with_scores(
                "test query", top_k=2
            )

            # Assertions
            self.assertEqual(texts, ["doc1", "doc2"])
            self.assertEqual(metadatas, [{"source": "file1"}, {"source": "file2"}])
            self.assertEqual(scores, [0.9, 0.8])

    def test_retrieve_by_embedding(self):
        """Test retrieve_by_embedding method."""
        # Setup mocks
        mock_embedding = [0.1, 0.2, 0.3]
        mock_results = [
            {"text": "doc1", "metadata": {"source": "file1"}, "score": 0.9},
            {"text": "doc2", "metadata": {"source": "file2"}, "score": 0.8},
        ]
        self.mock_vector_db_instance.search.return_value = mock_results

        # Call the method
        result = self.retriever.retrieve_by_embedding(mock_embedding, top_k=2)

        # Assertions
        self.mock_vector_db_instance.search.assert_called_once_with(
            query_embedding=mock_embedding, top_k=2, filter=None
        )
        self.assertEqual(result, mock_results)

    def test_retrieve_multiple(self):
        """Test retrieve_multiple method."""
        # Setup mocks
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_results1 = [
            {"text": "doc1", "metadata": {"source": "file1"}, "score": 0.9}
        ]
        mock_results2 = [
            {"text": "doc2", "metadata": {"source": "file2"}, "score": 0.8}
        ]

        self.mock_embedder_instance.embed_texts.return_value = mock_embeddings
        self.mock_vector_db_instance.search.side_effect = [mock_results1, mock_results2]

        # Call the method
        result = self.retriever.retrieve_multiple(["query1", "query2"], top_k=1)

        # Assertions
        self.mock_embedder_instance.embed_texts.assert_called_once_with(
            ["query1", "query2"]
        )
        self.assertEqual(self.mock_vector_db_instance.search.call_count, 2)
        self.assertEqual(result, [mock_results1, mock_results2])

    def test_retrieve_and_rerank(self):
        """Test retrieve_and_rerank method."""
        # Setup mocks
        mock_results = [
            {"text": "doc1", "metadata": {"source": "file1"}, "score": 0.9},
            {"text": "doc2", "metadata": {"source": "file2"}, "score": 0.8},
            {"text": "doc3", "metadata": {"source": "file3"}, "score": 0.7},
        ]

        # Mock the retrieve method
        with patch.object(self.retriever, "retrieve", return_value=mock_results):
            # Call the method
            result = self.retriever.retrieve_and_rerank(
                "test query", top_k_retrieve=3, top_k_rerank=2
            )

            # Assertions
            self.assertEqual(len(result), 2)
            self.assertEqual(result, mock_results[:2])


if __name__ == "__main__":
    unittest.main()
