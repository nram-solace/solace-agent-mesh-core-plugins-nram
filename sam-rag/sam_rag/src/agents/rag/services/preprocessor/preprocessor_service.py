"""
Main preprocessor service for handling document preprocessing.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
from .preprocessor_base import PreprocessorBase
from .document_preprocessor import (
    TextFilePreprocessor,
    PDFPreprocessor,
    DocxPreprocessor,
    HTMLPreprocessor,
    ExcelPreprocessor,
    ODTPreprocessor,
)


class PreprocessorService:
    """
    Service for preprocessing documents of various formats.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the preprocessor service.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.preprocessors: List[PreprocessorBase] = []
        self._register_preprocessors()

    def _register_preprocessors(self) -> None:
        """
        Register all available preprocessors.
        """
        # Register preprocessors in order of preference
        self.preprocessors = [
            PDFPreprocessor(self.config),
            DocxPreprocessor(self.config),
            HTMLPreprocessor(self.config),
            ExcelPreprocessor(self.config),
            ODTPreprocessor(self.config),
            # TextFilePreprocessor should be last as it handles many generic formats
            TextFilePreprocessor(self.config),
        ]

    def _get_preprocessor(self, file_path: str) -> Optional[PreprocessorBase]:
        """
        Get the appropriate preprocessor for the given file.

        Args:
            file_path: Path to the file.

        Returns:
            The appropriate preprocessor, or None if no suitable preprocessor is found.
        """
        for preprocessor in self.preprocessors:
            if preprocessor.can_process(file_path):
                return preprocessor
        return None

    def preprocess_file(self, file_path: str) -> Optional[str]:
        """
        Preprocess a single file.

        Args:
            file_path: Path to the file.

        Returns:
            Preprocessed text content, or None if the file cannot be processed.
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        preprocessor = self._get_preprocessor(file_path)
        if preprocessor:
            return preprocessor.preprocess(file_path)
        else:
            print(f"No suitable preprocessor found for file: {file_path}")
            return None

    def preprocess_files(
        self, file_paths: List[str]
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Preprocess multiple files.

        Args:
            file_paths: List of file paths.

        Returns:
            List of tuples containing (file_path, preprocessed_text).
            If a file cannot be processed, the preprocessed_text will be None.
        """
        results = []
        for file_path in file_paths:
            preprocessed_text = self.preprocess_file(file_path)
            results.append((file_path, preprocessed_text))
        return results

    def get_supported_extensions(self) -> List[str]:
        """
        Get a list of all supported file extensions.

        Returns:
            List of supported file extensions.
        """
        extensions = []
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PDFPreprocessor):
                extensions.append(".pdf")
            elif isinstance(preprocessor, DocxPreprocessor):
                extensions.extend([".doc", ".docx"])
            elif isinstance(preprocessor, HTMLPreprocessor):
                extensions.extend([".html", ".htm"])
            elif isinstance(preprocessor, ExcelPreprocessor):
                extensions.extend([".xls", ".xlsx"])
            elif isinstance(preprocessor, ODTPreprocessor):
                extensions.append(".odt")
        return list(set(extensions))  # Remove duplicates
