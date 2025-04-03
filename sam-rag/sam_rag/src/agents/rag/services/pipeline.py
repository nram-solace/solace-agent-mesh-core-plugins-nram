"""
RAG Pipeline that combines preprocessing, splitting, embedding, and ingestion components.

This module provides a complete pipeline for processing documents through the RAG workflow:
1. Preprocessing: Convert documents to clean text
2. Splitting: Split text into chunks
3. Embedding: Convert chunks to vector embeddings
4. Ingestion: Store embeddings in a vector database

Usage:
    python -m src.agents.rag.services.pipeline
"""

import os
import yaml
import logging
import threading
import time
from typing import Dict, List, Any, Optional
from solace_ai_connector.common.log import log

from ..services.preprocessor.document_processor import DocumentProcessor
from ..services.splitter.splitter_service import SplitterService
from ..services.embedder.embedder_service import EmbedderService
from ..services.ingestor.ingestor_service import IngestorService

from ..services.scanner.file_tracker import FileChangeTracker
from ..services.memory.memory_storage import memory_storage

SCANNER_AVAILABLE = True


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class RAGPipeline:
    """
    RAG Pipeline that combines preprocessing, splitting, embedding, and ingestion components.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the RAG pipeline.

        Args:
            config_path: Path to the configuration file. If None, uses the default path.
            use_memory_storage: Whether to use memory storage instead of database.
        """
        # Set default config path if not provided
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        )
                    )
                ),
                "configs",
                "agents",
                "rag.yaml",
            )

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self.preprocessor = DocumentProcessor(self.config.get("preprocessor", {}))
        self.splitter = SplitterService(self.config.get("splitter", {}))
        self.embedder = EmbedderService(self.config.get("embedding", {}))
        self.ingestor = IngestorService(
            {
                "preprocessor": self.config.get("preprocessor", {}),
                "splitter": self.config.get("splitter", {}),
                "embedder": self.config.get("embedding", {}),
                "vector_db": self.config.get("vector_db", {}),
            }
        )

        # Initialize file tracker if scanner is available
        self.file_tracker = None
        if SCANNER_AVAILABLE:
            scanner_config = self.config.get("scanner", {})
            if scanner_config:
                self.use_memory_storage = scanner_config["use_memory_storage"]
                self.batch = scanner_config["batch"]
                self.file_tracker = FileChangeTracker(scanner_config)
                logger.info(
                    "File tracker initialized with memory storage"
                    if self.use_memory_storage
                    else "File tracker initialized with database"
                )

        logger.info("RAG Pipeline initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            A dictionary containing the configuration.
        """
        try:
            with open(config_path, "r") as file:
                yaml_config = yaml.safe_load(file)

            # Extract the component configurations
            for flow in yaml_config.get("flows", []):
                if flow.get("name") == "rag_action_request_processor":
                    for component in flow.get("components", []):
                        if (
                            component.get("component_name")
                            == "action_request_processor"
                        ):
                            return component.get("component_config", {})

            logger.warning("Could not find RAG configuration in the YAML file")
            return {}
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            return {}

    def process_files(self, files: List[str]) -> Dict[str, Any]:
        """
        Process files through the RAG pipeline.

        Args:
            files: List of files to process.
            document_types: Optional list of document types (e.g., "pdf", "text", "html").
                If not provided, types will be inferred from file extensions.

        Returns:
            A dictionary containing the processing results.
        """
        logger.info(f"Processing {len(files)} files through the RAG pipeline")

        # Step 1: Preprocess files
        preprocessed_docs = []
        preprocessed_metadata = []

        for i, file in enumerate(files):
            try:
                file_path = file.get("path", None)  # Get the file path
                if not file_path:
                    logger.warning(f"Invalid file path: {file}")
                    continue

                # Verify the file exists
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue

                file_status = file.get("status", None)  # Get the file status
                if file_status not in {"modified", "new"}:
                    continue

                # Get document type from extension if not provided
                doc_type = self._get_file_type(file_path)

                # Process the file
                text = self.preprocessor.process_document(file_path)

                if text:
                    preprocessed_docs.append(text)
                    preprocessed_metadata.append(
                        {
                            "source": file_path,
                            "document_type": doc_type,
                            "file_name": os.path.basename(file_path),
                            "index": i,
                        }
                    )
                    logger.info(
                        f"Successfully preprocessed file: {file_path} ({doc_type})"
                    )
                else:
                    logger.warning(f"Failed to preprocess file: {file_path}")

                # update the file status in the tracker
                if SCANNER_AVAILABLE and self.file_tracker:
                    memory_storage.update_document(path=file_path, status="done")
            except Exception as e:
                logger.error(f"Error preprocessing file {file_path}: {str(e)}")

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
                # Get the document type
                doc_type = meta.get("document_type", "text")

                # Split the document
                doc_chunks = self.splitter.split_text(doc, doc_type)

                # Add chunks and metadata
                chunks.extend(doc_chunks)
                chunks_metadata.extend([meta.copy() for _ in range(len(doc_chunks))])

                logger.info(f"Split document {i} into {len(doc_chunks)} chunks")
            except Exception as e:
                logger.error(f"Error splitting document {i}: {str(e)}")

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
            logger.info(f"Created {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error embedding chunks: {str(e)}")
            return {
                "success": False,
                "message": f"Error embedding chunks: {str(e)}",
                "document_ids": [],
            }

        # Step 4: Ingest embeddings into vector database
        try:
            # Use the ingestor service to store the embeddings
            # Note: Using the new ingest_embeddings method that takes both chunks and embeddings
            result = self.ingestor.ingestor.ingest_embeddings(
                texts=chunks, embeddings=embeddings, metadata=chunks_metadata
            )
            logger.info(f"Ingestion result: {result['message']}")
            return result
        except Exception as e:
            logger.error(f"Error ingesting embeddings: {str(e)}")
            return {
                "success": False,
                "message": f"Error ingesting embeddings: {str(e)}",
                "document_ids": [],
            }

    def _get_file_type(self, file_path: str) -> str:
        """
        Get the file type from a file path.

        Args:
            file_path: Path to the file.

        Returns:
            The file type (e.g., "pdf", "text", "html").
        """
        _, ext = os.path.splitext(file_path.lower())
        return ext[1:] if ext else "text"  # Remove the leading dot

    def search(
        self, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None
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
        # Process and embed the query
        try:
            # Clean the query text
            processed_query = query  # self.preprocessor.(query)

            # Embed the query
            query_embedding = self.embedder.embed_text(processed_query)

            # Search using the query embedding
            return self.ingestor.ingestor.search(query_embedding, top_k, filter)
        except Exception as e:
            logger.error(f"Error searching for query: {str(e)}")
            return []

    def get_tracked_files(self) -> List[Dict[str, Any]]:
        """
        Get all tracked files from the file tracker.

        Returns:
            A list of tracked files with their metadata.
        """
        if SCANNER_AVAILABLE and self.file_tracker:
            return self.file_tracker.get_tracked_files()
        else:
            logger.warning(
                "Scanner module not available or file tracker not initialized"
            )
            return []

    def scan_files(self) -> Dict[str, List[str]]:
        """
        Scan for file changes using the file tracker.

        Returns:
            A dictionary containing the scan results.
        """
        if SCANNER_AVAILABLE and self.file_tracker:
            return self.file_tracker.scan()
        else:
            logger.warning(
                "Scanner module not available or file tracker not initialized"
            )
            return {"added": [], "removed": [], "changed": []}


def scan_files_thread(pipeline):
    """
    Run the file scanner in a separate thread.

    Args:
        pipeline: The RAGPipeline instance

    Returns:
        The started thread object
    """
    scan_thread = threading.Thread(target=pipeline.scan_files)
    scan_thread.daemon = True  # Thread will exit when main program exits
    scan_thread.start()
    return scan_thread


def main():
    """
    Main function to demonstrate the RAG pipeline with file processing.
    """

    # Initialize the RAG pipeline
    pipeline = RAGPipeline()

    run_scanner = input("Would you like to scan and ingest new documents? (Yes/No) ")

    if run_scanner.lower() == "yes":
        # Scan files in a background thread
        scan_files_thread(pipeline)

        # Ingest files
        for _ in range(1, 4):  # Loop from 1 to 3 inclusive
            time.sleep(2)  # Wait for file changes
            # Process the files
            files = pipeline.get_tracked_files()
            if not files:
                print("No files to process. Exiting...")
                continue

            result = pipeline.process_files(files)

            # Print the result
            print(f"\nProcessing result: {result['message']}")
            print(f"Document IDs: {result['document_ids']}")

    # Search
    while True:
        query = input("How can I help you? ")
        result = pipeline.search(query=query)
        print(f"output is {result}")


if __name__ == "__main__":
    main()
