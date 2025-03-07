from abc import ABC, abstractmethod
from typing import List, Dict


# Abstract base class for data sources
class DataSource(ABC):
    """
    Abstract base class for data sources.
    """

    def __init__(self, config: Dict):
        """
        Initialize the DataSource with the given configuration.

        :param config: A dictionary containing the configuration.
        """
        self.config = config

    @abstractmethod
    def process_config(self) -> List[str]:
        """
        Retrieve the list of files from the data source.

        :return: A list of file paths.
        """
        pass

    @abstractmethod
    def scan(self) -> None:
        """
        Monitor changes in the data source.

        This method should be implemented by concrete data source classes.
        """
        pass
