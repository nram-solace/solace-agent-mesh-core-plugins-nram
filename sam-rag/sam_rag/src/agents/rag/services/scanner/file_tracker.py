from typing import List, Dict, Any, Optional
import logging

from .file_system import LocalFileSystemDataSource
from .cloud_storage import CloudStorageDataSource
from ..memory.memory_storage import memory_storage

# Try to import database modules, but don't fail if they're not available
try:
    from ..database.connect import connect

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


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
