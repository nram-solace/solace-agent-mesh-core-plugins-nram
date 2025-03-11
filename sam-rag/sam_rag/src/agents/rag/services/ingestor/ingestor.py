"""
Implementation of document ingestor.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union, Tuple

from ..preprocessor.preprocessor_service import PreprocessorService
from ..splitter.splitter_service import SplitterService
from ..embedder.embedder_service import EmbedderService
from ..database.vector_db_service import VectorDBService
from .ingestor_base import IngestorBase

logger = logging.getLogger(__name__)


class DocumentIngestor(IngestorBase):
    """
    Document ingestor that integrates preprocessing, splitting, embedding, and vector database storage.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the document ingestor.

        Args:
            config: A dictionary containing configuration parameters.
                - preprocessor: Configuration for the preprocessor.
                - splitter: Configuration for the text splitter.
                - embedder: Configuration for the embedder.
                - vector_db: Configuration for the vector database.
        """
        super().__init__(config)

        # Initialize components
        self.preprocessor = PreprocessorService(self.config.get("preprocessor", {}))
        self.splitter = SplitterService(self.config.get("splitter", {}))
        self.embedder = EmbedderService(self.config.get("embedder", {}))
        self.vector_db = VectorDBService(self.config.get("vector_db", {}))

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
        logger.info(f"Ingesting {len(file_paths)} documents")

        # Create default metadata if not provided
        if metadata is None:
            metadata = [{"source": file_path} for file_path in file_paths]

        # Step 1: Preprocess documents
        preprocessed_docs = []
        preprocessed_metadata = []

        for i, (file_path, meta) in enumerate(zip(file_paths, metadata)):
            try:
                # Preprocess the document
                text = self.preprocessor.preprocess_file(file_path)

                if text:
                    preprocessed_docs.append(text)

                    # Add file path to metadata if not already present
                    if "source" not in meta:
                        meta["source"] = file_path

                    # Add file type to metadata
                    _, ext = os.path.splitext(file_path)
                    if ext and "file_type" not in meta:
                        meta["file_type"] = ext.lower()[1:]  # Remove the dot

                    preprocessed_metadata.append(meta)
                    logger.debug(f"Successfully preprocessed: {file_path}")
                else:
                    logger.warning(f"Failed to preprocess: {file_path}")
            except Exception as e:
                logger.error(f"Error preprocessing {file_path}: {str(e)}")

        # Return early if no documents were preprocessed
        if not preprocessed_docs:
            logger.warning("No documents were successfully preprocessed")
            return {
                "success": False,
                "message": "No documents were successfully preprocessed",
                "document_ids": [],
            }

        # Step 2: Split documents into chunks
        chunks = []
        chunks_metadata = []

        for i, (doc, meta) in enumerate(zip(preprocessed_docs, preprocessed_metadata)):
            try:
                # Get the file type from metadata or default to "text"
                data_type = meta.get("file_type", "text")

                # Split the document
                doc_chunks = self.splitter.split_text(doc, data_type)

                # Add chunks and metadata
                chunks.extend(doc_chunks)
                chunks_metadata.extend([meta.copy() for _ in range(len(doc_chunks))])

                logger.debug(f"Split document into {len(doc_chunks)} chunks")
            except Exception as e:
                logger.error(f"Error splitting document: {str(e)}")

        # Return early if no chunks were created
        if not chunks:
            logger.warning("No chunks were created from the documents")
            return {
                "success": False,
                "message": "No chunks were created from the documents",
                "document_ids": [],
            }

        # Step 3: Embed chunks
        try:
            embeddings = self.embedder.embed_texts(chunks)
            logger.debug(f"Created {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error embedding chunks: {str(e)}")
            return {
                "success": False,
                "message": f"Error embedding chunks: {str(e)}",
                "document_ids": [],
            }

        # Step 4: Store embeddings in vector database
        try:
            document_ids = self.vector_db.add_documents(
                documents=chunks,
                embeddings=embeddings,
                metadatas=chunks_metadata,
            )
            logger.info(f"Added {len(document_ids)} documents to vector database")
        except Exception as e:
            logger.error(f"Error storing embeddings in vector database: {str(e)}")
            return {
                "success": False,
                "message": f"Error storing embeddings in vector database: {str(e)}",
                "document_ids": [],
            }

        return {
            "success": True,
            "message": f"Successfully ingested {len(document_ids)} chunks from {len(preprocessed_docs)} documents",
            "document_ids": document_ids,
        }

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
        logger.info(f"Ingesting {len(texts)} texts")

        # Create default metadata if not provided
        if metadata is None:
            metadata = [{"source": "direct_text"} for _ in range(len(texts))]

        # Step 1: Preprocess texts
        preprocessed_texts = []
        preprocessed_metadata = []

        for i, (text, meta) in enumerate(zip(texts, metadata)):
            try:
                # Preprocess the text
                processed_text = self.preprocessor.preprocess_text(text)

                if processed_text:
                    preprocessed_texts.append(processed_text)
                    preprocessed_metadata.append(meta)
                    logger.debug(f"Successfully preprocessed text {i}")
                else:
                    logger.warning(f"Failed to preprocess text {i}")
            except Exception as e:
                logger.error(f"Error preprocessing text {i}: {str(e)}")

        # Return early if no texts were preprocessed
        if not preprocessed_texts:
            logger.warning("No texts were successfully preprocessed")
            return {
                "success": False,
                "message": "No texts were successfully preprocessed",
                "document_ids": [],
            }

        # Step 2: Split texts into chunks
        chunks = []
        chunks_metadata = []

        for i, (text, meta) in enumerate(
            zip(preprocessed_texts, preprocessed_metadata)
        ):
            try:
                # Get the data type from metadata or default to "text"
                data_type = meta.get("data_type", "text")

                # Split the text
                text_chunks = self.splitter.split_text(text, data_type)

                # Add chunks and metadata
                chunks.extend(text_chunks)
                chunks_metadata.extend([meta.copy() for _ in range(len(text_chunks))])

                logger.debug(f"Split text {i} into {len(text_chunks)} chunks")
            except Exception as e:
                logger.error(f"Error splitting text {i}: {str(e)}")

        # Return early if no chunks were created
        if not chunks:
            logger.warning("No chunks were created from the texts")
            return {
                "success": False,
                "message": "No chunks were created from the texts",
                "document_ids": [],
            }

        # Step 3: Embed chunks
        try:
            embeddings = self.embedder.embed_texts(chunks)
            logger.debug(f"Created {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error embedding chunks: {str(e)}")
            return {
                "success": False,
                "message": f"Error embedding chunks: {str(e)}",
                "document_ids": [],
            }

        # Step 4: Store embeddings in vector database
        try:
            document_ids = self.vector_db.add_documents(
                documents=chunks,
                embeddings=embeddings,
                metadatas=chunks_metadata,
                ids=ids,
            )
            logger.info(f"Added {len(document_ids)} documents to vector database")
        except Exception as e:
            logger.error(f"Error storing embeddings in vector database: {str(e)}")
            return {
                "success": False,
                "message": f"Error storing embeddings in vector database: {str(e)}",
                "document_ids": [],
            }

        return {
            "success": True,
            "message": f"Successfully ingested {len(document_ids)} chunks from {len(preprocessed_texts)} texts",
            "document_ids": document_ids,
        }

    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector database.

        Args:
            ids: The IDs of the documents to delete.
        """
        try:
            self.vector_db.delete(ids)
            logger.info(f"Deleted {len(ids)} documents from vector database")
        except Exception as e:
            logger.error(f"Error deleting documents from vector database: {str(e)}")
            raise

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
        try:
            # Preprocess the query
            processed_query = self.preprocessor.preprocess_text(query)

            # Embed the query
            query_embedding = self.embedder.embed_text(processed_query)

            # Search the vector database
            results = self.vector_db.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter=filter,
            )

            logger.info(f"Found {len(results)} results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error searching for query: {str(e)}")
            raise
