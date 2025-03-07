from typing import List, Dict

from .file_system import LocalFileSystemDataSource
from .cloud_storage import CloudStorageDataSource
from ..database.connect import connect


# Class to track file changes
class FileChangeTracker:

    def __init__(self, config: Dict):
        self.config = config

    def scan(self) -> Dict[str, List[str]]:
        """
        Detect added, removed, and changed files.
        """
        db_config = self.config.get("database", {})
        if not db_config:
            raise ValueError("Database configuration is missing")
        connect(db_config)

        data_source = self.config.get("source", {})

        if not data_source:
            raise ValueError("Source configuration is missing")

        match data_source.get("type"):
            case "filesystem":
                data_source = LocalFileSystemDataSource(data_source)
            case "cloud":
                data_source = CloudStorageDataSource(data_source)
            case _:
                raise ValueError("Invalid data source type")

        data_source.scan()
