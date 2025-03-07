from typing import Dict

from .datasource_base import DataSource


# Concrete implementation for Cloud Storage (Placeholder example)
class CloudStorageDataSource(DataSource):
    """
    A data source implementation for fetching files from cloud storage.
    """

    def process_config(self, source: Dict = {}) -> None:
        """
        Assume a method to connect to cloud and fetch the file list.
        The actual implementation will depend on the cloud provider API.

        :param source: A dictionary containing the source configuration.
        :return: A list of file paths from the cloud storage.
        """
        return ["cloud://bucket/file1.txt", "cloud://bucket/file2.txt"]  # Mock example

    def scan(self) -> None:
        """
        Monitor changes in the cloud storage.
        """
        pass
