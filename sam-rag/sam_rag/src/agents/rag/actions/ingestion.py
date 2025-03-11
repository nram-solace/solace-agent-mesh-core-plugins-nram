"""
Action for ingesting documents into the RAG system.
This action scans documents from various data sources and ingests them into a vector database.
"""

import os
from typing import Dict, List, Tuple, Optional
from solace_ai_connector.common.log import log

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse

# Adding imports for file tracking and ingestor functionality
from src.agents.rag.services.scanner.file_tracker import FileChangeTracker
from src.agents.rag.services.ingestor.ingestor_service import IngestorService
from src.agents.rag.services.database.model import init_db

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
                        "name": "document",
                        "desc": "The document to be ingested.",
                        "type": "string",
                    },
                    {
                        "name": "file_paths",
                        "desc": "List of file paths to preprocess and ingest.",
                        "type": "array",
                    },
                ],
                "required_scopes": ["rag:ingestion_action:write"],
            },
            **kwargs,
        )
        self.ingestor = None

    def invoke(self, params, meta={}) -> ActionResponse:
        log.debug("Starting document ingestion process")

        # Initialize database if needed
        # init_db(self.config["database"])

        # Initialize ingestor if not already initialized
        if not self.ingestor:
            self.ingestor = IngestorService(self.config)

        # Check if specific file paths were provided
        file_paths = params.get("file_paths", [])
        if file_paths:
            # Preprocess and ingest specific files
            return self._process_specific_files(file_paths)
        else:
            # Scan for file changes using file tracker
            return self._process_file_changes()

    def _process_specific_files(self, file_paths: List[str]) -> ActionResponse:
        """
        Process specific files provided in the parameters.

        Args:
            file_paths: List of file paths to process.

        Returns:
            ActionResponse with the result of the processing.
        """
        log.debug(f"Processing {len(file_paths)} specific files")

        # Create metadata for each file
        metadata = [{"source": file_path} for file_path in file_paths]

        # Ingest the files
        result = self.ingestor.ingest_documents(file_paths, metadata)

        if result["success"]:
            log.info(result["message"])
            return ActionResponse(message=result["message"])
        else:
            log.error(f"Ingestion failed: {result['message']}")
            return ActionResponse(
                message=f"Ingestion failed: {result['message']}", error=True
            )

    def _process_file_changes(self) -> ActionResponse:
        """
        Process file changes detected by the file tracker.

        Returns:
            ActionResponse with the result of the processing.
        """
        # Create file tracker with configuration
        file_tracker = FileChangeTracker(self.config)

        # Scan for file changes
        changes = file_tracker.scan()

        # Process the changes if any were detected
        if changes:
            # Extract file paths from changes
            file_paths = []
            for change_type, paths in changes.items():
                if change_type in ["added", "modified"]:
                    file_paths.extend(paths)

            # Preprocess the changed files
            if file_paths:
                return self._process_specific_files(file_paths)

        log.debug("Detected %d file changes", len(changes) if changes else 0)

        return ActionResponse(
            message=f"Ingestion completed. Found {len(changes) if changes else 0} changes."
        )

    def do_action(self, sample) -> ActionResponse:
        sample += " Action performed"
        return ActionResponse(message=sample)
