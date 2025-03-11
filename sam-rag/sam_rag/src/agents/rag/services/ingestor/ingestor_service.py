"""
Service for document ingestion.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from .ingestor import DocumentIngestor


class IngestorService:
    """
    Service for document ingestion.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ingestor service.

        Args:
            config: A dictionary containing configuration parameters.
                - preprocessor: Configuration for the preprocessor.
                - splitter: Configuration for the text splitter.
                - embedder: Configuration for the embedder.
                - vector_db: Configuration for the vector database.
        """
        self.config = config or {}
        self.ingestor = DocumentIngestor(self.config)

    def ingest_documents(
        self,
        file_paths: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest documents from file paths.

        Args:
            file_paths: List of file paths to ingest.
            metadata: Optional metadata for each document.

        Returns:
            A dictionary containing the ingestion results.
        """
        return self.ingestor.ingest_documents(file_paths, metadata)

    def ingest_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest texts directly.

        Args:
            texts: List of texts to ingest.
            metadata: Optional metadata for each text.
            ids: Optional IDs for each text.

        Returns:
            A dictionary containing the ingestion results.
        """
        return self.ingestor.ingest_texts(texts, metadata, ids)

    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector database.

        Args:
            ids: The IDs of the documents to delete.
        """
        self.ingestor.delete_documents(ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.

        Args:
            query: The query text.
            top_k: The number of results to return.
            filter: Optional filter to apply to the search.

        Returns:
            A list of dictionaries containing the search results.
        """
        return self.ingestor.search(query, top_k, filter)
