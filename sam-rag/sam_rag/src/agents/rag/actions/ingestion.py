"""
Action for ingesting documents into the RAG system.
This action scans documents from various data sources and ingests them into a vector database.
"""

import os
from typing import Dict, List, Any
from solace_ai_connector.common.log import log

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse

# Adding imports for file tracking and ingestor functionality
from src.agents.rag.services.ingestor.ingestor_service import IngestorService
from src.agents.rag.services.database.model import init_db

# Add new imports for the RAG pipeline
from src.agents.rag.services.preprocessor.document_processor import DocumentProcessor
from src.agents.rag.services.splitter.splitter_service import SplitterService
from src.agents.rag.services.embedder.embedder_service import EmbedderService

# To import from a local file, like this file, use a relative path from the rag
# For example, to load this class, use:
#   from rag.actions.sample_action import SampleAction


class IngestionAction(Action):

    def __init__(self, **kwargs):
        super().__init__(
            {
                "name": "ingestion_action",
                "prompt_directive": (
                    "This action scans documents of data sources and ingests them into a vector database. "
                    "Examples include scanning a filesystem and indexing PDF documents."
                ),
                "params": [
                    {
                        "name": "scanner",
                        "desc": "Configuration for the document scanner.",
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "object",
                                "desc": "Document source configuration",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "desc": "Source type (e.g., filesystem)",
                                    },
                                    "directories": {
                                        "type": "array",
                                        "desc": "Directories to scan",
                                    },
                                    "filters": {
                                        "type": "object",
                                        "desc": "File filtering options",
                                    },
                                },
                            },
                            "use_memory_storage": {
                                "type": "boolean",
                                "desc": "Whether to use in-memory storage",
                            },
                            "schedule": {
                                "type": "object",
                                "desc": "Scanning schedule configuration",
                            },
                        },
                    },
                    {
                        "name": "preprocessor",
                        "desc": "Configuration for document preprocessing.",
                        "type": "object",
                        "properties": {
                            "default_preprocessor": {
                                "type": "object",
                                "desc": "Default preprocessing settings",
                            },
                            "preprocessors": {
                                "type": "object",
                                "desc": "File-specific preprocessors",
                            },
                        },
                    },
                    {
                        "name": "splitter",
                        "desc": "Configuration for text splitting.",
                        "type": "object",
                        "properties": {
                            "default_splitter": {
                                "type": "object",
                                "desc": "Default text splitter settings",
                            },
                            "splitters": {
                                "type": "object",
                                "desc": "File-specific text splitters",
                            },
                        },
                    },
                    {
                        "name": "embedding",
                        "desc": "Configuration for embedding generation.",
                        "type": "object",
                        "properties": {
                            "embedder_type": {
                                "type": "string",
                                "desc": "Type of embedder to use",
                            },
                            "embedder_params": {
                                "type": "object",
                                "desc": "Parameters for the embedder",
                            },
                            "normalize_embeddings": {
                                "type": "boolean",
                                "desc": "Whether to normalize embeddings",
                            },
                        },
                    },
                    {
                        "name": "vector_db",
                        "desc": "Configuration for vector database.",
                        "type": "object",
                        "properties": {
                            "db_type": {
                                "type": "string",
                                "desc": "Type of vector database",
                            },
                            "db_params": {
                                "type": "object",
                                "desc": "Parameters for the vector database",
                            },
                        },
                    },
                ],
                "required_scopes": ["rag:ingestion_action:write"],
            },
            **kwargs,
        )
        self.ingestor = None

    def invoke(self, params, meta={}) -> ActionResponse:
        log.debug("Starting document ingestion process")

        # Initialize ingestor if not already initialized
        if not self.ingestor:
            self.ingestor = IngestorService(params)

        # If no explicit directories, check scanner configuration
        if "scanner" in params:
            scanner_config = params.get("scanner", {})
            source_config = scanner_config.get("source", {})

            # Get directories from scanner configuration
            if (
                source_config.get("type") == "filesystem"
                and "directories" in source_config
            ):
                directories = source_config.get("directories", [])

            if directories:
                # Process files through the complete pipeline
                result = self._process_files(directories, params)
                return ActionResponse(
                    message=result.get("message", "Files processed successfully."),
                    error=not result.get("success", True),
                    result=result,
                )

        log.warning("No directories provided for ingestion.")
        return ActionResponse(
            message="No directories provided.",
            error=True,
            result=None,
        )

    def _process_files(self, file_paths: List[str], params=None) -> Dict[str, Any]:
        """
        Process files through a complete RAG pipeline: preprocess, chunk, embed, and ingest.

        Args:
            file_paths: List of file paths to process.
            params: Configuration parameters.

        Returns:
            A dictionary containing the processing results.
        """
        log.info(f"Processing {len(file_paths)} files through the RAG pipeline")

        # Initialize pipeline components with configuration from params
        preprocessor = DocumentProcessor(params.get("preprocessor", {}))
        splitter = SplitterService(params.get("splitter", {}))
        embedder = EmbedderService(params.get("embedding", {}))

        # Step 1: Preprocess files
        preprocessed_docs = []
        preprocessed_metadata = []

        for i, file_path in enumerate(file_paths):
            try:
                # Verify the file exists
                if not os.path.exists(file_path):
                    log.warning(f"File not found: {file_path}")
                    continue

                # Process the file
                text = preprocessor.process_document(file_path)
                doc_type = self._get_file_type(file_path)

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
                    log.info(
                        f"Successfully preprocessed file: {file_path} ({doc_type})"
                    )
                else:
                    log.warning(f"Failed to preprocess file: {file_path}")
            except Exception as e:
                log.error(f"Error preprocessing file {file_path}: {str(e)}")

        if not preprocessed_docs:
            log.warning("No documents were successfully preprocessed")
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
                doc_chunks = splitter.split_text(doc, doc_type)

                # Add chunks and metadata
                chunks.extend(doc_chunks)
                chunks_metadata.extend([meta.copy() for _ in range(len(doc_chunks))])

                log.info(f"Split document {i} into {len(doc_chunks)} chunks")
            except Exception as e:
                log.error(f"Error splitting document {i}: {str(e)}")

        if not chunks:
            log.warning("No chunks were created from the documents")
            return {
                "success": False,
                "message": "No chunks were created from the documents",
                "document_ids": [],
            }

        # Step 3: Embed chunks
        try:
            embeddings = embedder.embed_texts(chunks)
            log.info(f"Created {len(embeddings)} embeddings")
        except Exception as e:
            log.error(f"Error embedding chunks: {str(e)}")
            return {
                "success": False,
                "message": f"Error embedding chunks: {str(e)}",
                "document_ids": [],
            }

        # Step 4: Ingest embeddings into vector database
        try:
            # Use the ingestor to store the embeddings
            result = self.ingestor.ingest_embeddings(
                texts=chunks, embeddings=embeddings, metadata=chunks_metadata
            )
            log.info(f"Ingestion result: {result['message']}")
            return result
        except Exception as e:
            log.error(f"Error ingesting embeddings: {str(e)}")
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

    def do_action(self, action) -> ActionResponse:
        action += " Action performed"
        return ActionResponse(message=action)
