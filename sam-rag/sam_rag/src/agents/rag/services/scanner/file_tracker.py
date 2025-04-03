from typing import List, Dict, Any, Optional, Set
from solace_ai_connector.common.log import log as logger

from .file_system import LocalFileSystemDataSource
from .cloud_storage import CloudStorageDataSource
from ..memory.memory_storage import memory_storage
from ..database.vector_db_service import VectorDBService

# Try to import database modules, but don't fail if they're not available
try:
    from ..database.connect import connect

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


# Class to track file changes
class FileChangeTracker:
    """
    Class to track file changes from different data sources.
    """

    def __init__(self, config: Dict):
        """
        Initialize the FileChangeTracker with the given configuration.

        Args:
            config: A dictionary containing the configuration.
        """
        self.config = config
        self.data_source = None
        self.use_memory_storage = config.get("use_memory_storage", False)
        self.batch = config.get("batch", False)

        # Initialize vector database service to check for existing documents
        self.vector_db = VectorDBService(config.get("vector_db", {}))

    def get_source_documents(self) -> List[str]:
        """
        Get a list of source document paths from the vector database.

        Returns:
            A list of document paths that are already in the vector database.
        """
        try:
            # Use a dummy embedding to search for all documents
            # We'll use a large number to get all documents
            # The actual similarity doesn't matter since we just want to extract metadata
            dummy_embedding = [0.0] * 768  # Common embedding dimension
            results = self.vector_db.search(
                query_embedding=dummy_embedding, top_k=10000
            )

            # Extract source paths from metadata
            sources = set()
            for result in results:
                if "metadata" in result and "source" in result["metadata"]:
                    sources.add(result["metadata"]["source"])

            logger.info(f"Found {len(sources)} existing documents in vector database")
            return list(sources)
        except Exception as e:
            logger.warning(
                f"Error getting source documents from vector database: {str(e)}"
            )
            return []

    def scan(self) -> None:
        """
        Detect added, removed, and changed files.

        Returns:
            A dictionary containing the scan results.
        """
        # If using database, connect to it
        if not self.use_memory_storage and DATABASE_AVAILABLE:
            db_config = self.config.get("database", {})
            if not db_config:
                logger.warning(
                    "Database configuration is missing, using memory storage"
                )
                self.use_memory_storage = True
            else:
                connect(db_config)

        # Get data source configuration
        source_config = self.config.get("source", {})
        if not source_config:
            raise ValueError("Source configuration is missing")

        # Set memory storage flag in source config
        source_config["use_memory_storage"] = self.use_memory_storage
        source_config["batch"] = self.batch

        # Get existing source documents from vector database
        existing_sources = self.get_source_documents()

        # Add existing sources to source config
        source_config["existing_sources"] = existing_sources

        # Create data source based on type
        source_type = source_config.get("type", "filesystem")
        if source_type == "filesystem":
            self.data_source = LocalFileSystemDataSource(source_config)
        elif source_type == "cloud":
            self.data_source = CloudStorageDataSource(source_config)
        else:
            raise ValueError(f"Invalid data source type: {source_type}")

        # Start scanning
        self.data_source.scan()

        # Return empty result for now
        # return {"added": [], "removed": [], "changed": []}

    def get_tracked_files(self) -> List[Dict[str, Any]]:
        """
        Get all tracked files.

        Returns:
            A list of tracked files with their metadata.
        """
        if self.data_source:
            return self.data_source.get_tracked_files()
        elif self.use_memory_storage:
            return memory_storage.get_all_documents()
        else:
            logger.warning("No data source available")
            return []
