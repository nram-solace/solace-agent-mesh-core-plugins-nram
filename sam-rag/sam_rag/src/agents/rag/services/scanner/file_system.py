import os
import time
import threading
from typing import Dict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from solace_ai_connector.common.log import log

from .datasource_base import DataSource
from ..database.connect import get_db, insert_document, update_document, delete_document


# Concrete implementation for Local File System
class LocalFileSystemDataSource(DataSource):
    """
    A data source implementation for monitoring local file system changes.
    """

    def __init__(self, source: Dict):
        """
        Initialize the LocalFileSystemDataSource with the given source configuration.

        :param source: A dictionary containing the source configuration.
        :param metadata_db: The database connection for metadata storage.
        """
        super().__init__(source)
        self.directories = []
        self.formats = []
        self.max_file_size = None
        self.file_changes = []
        self.interval = 10
        self.process_config(source)

    def process_config(self, source: Dict = {}) -> None:
        """
        Process the source configuration to set up directories, file formats, and max file size.

        :param source: A dictionary containing the source configuration.
        """
        self.directories = source.get("directories", [])
        if not self.directories:
            log.info("No folder paths configured.")
            return
        filters = source.get("filters", {})
        if filters:
            self.formats = source.get("file_formats", [])
            self.max_file_size = source.get("max_file_size", None)

        schedule = source.get("schedule", {})
        if schedule:
            self.interval = schedule.get("interval", 10)

    def scan(self) -> None:
        """
        Monitor the configured directories for file system changes.
        """
        event_handler = FileSystemEventHandler()
        event_handler.on_created = self.on_created
        event_handler.on_deleted = self.on_deleted
        event_handler.on_modified = self.on_modified

        observer = Observer()
        for directory in self.directories:
            observer.schedule(event_handler, directory, recursive=True)
        observer.start()

        def run_periodically():
            while True:
                time.sleep(self.interval)

        thread = threading.Thread(target=run_periodically)
        thread.start()

        try:
            thread.join()
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    def on_created(self, event):
        """
        Handle the event when a file is created.

        :param event: The file system event.
        """
        if not self.is_valid_file(event):
            return

        insert_document(
            get_db(),
            status="new",
            path=event.src_path,
            file=os.path.basename(event.src_path),
        )
        log.info(f"Document inserted: {event.src_path}")

    def on_deleted(self, event):
        """
        Handle the event when a file is deleted.

        :param event: The file system event.
        """
        if not self.is_valid_file(event):
            return

        delete_document(get_db(), path=event.src_path)
        log.info(f"Document deleted: {event.src_path}")

    def on_modified(self, event):
        """
        Handle the event when a file is modified.

        :param event: The file system event.
        """
        if not self.is_valid_file(event):
            return

        update_document(get_db(), path=event.src_path, status="modified")
        log.info(f"Document updated: {event.src_path}")

    def is_valid_file(self, event: FileSystemEventHandler) -> bool:
        """
        Check if the file is valid based on the configured formats and size.

        :param event: The file system event.
        :return: True if the file is valid, False otherwise.
        """
        if os.path.isdir(event.src_path):
            return False
        if event.src_path.endswith(".DS_Store"):
            return False
        if self.formats and not any(
            event.src_path.endswith(fmt) for fmt in self.formats
        ):
            return False
        if (
            self.max_file_size is not None
            and os.path.getsize(event.src_path) > self.max_file_size * 1024
        ):
            return False
        return True
