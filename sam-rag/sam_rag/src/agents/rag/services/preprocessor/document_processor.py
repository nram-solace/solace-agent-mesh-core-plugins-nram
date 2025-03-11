"""
Document processor that converts files of various formats to clean text.

This module provides a high-level interface for processing documents from various
formats and converting them to clean text suitable for embedding.
"""

import sys
from typing import Dict, List, Optional, Any
from .enhanced_preprocessor import EnhancedPreprocessorService


class DocumentProcessor:
    """
    Document processor that converts files of various formats to clean text.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the document processor.

        Args:
            config: Configuration dictionary for text preprocessing options.
                - lowercase: Whether to convert text to lowercase (default: True).
                - normalize_unicode: Whether to normalize Unicode characters (default: True).
                - normalize_whitespace: Whether to normalize whitespace (default: True).
                - remove_punctuation: Whether to remove punctuation (default: True).
                - remove_special_chars: Whether to remove special characters (default: True).
                - remove_urls: Whether to remove URLs (default: True).
                - remove_html_tags: Whether to remove HTML tags (default: True).
                - remove_numbers: Whether to remove numbers (default: False).
                - remove_non_ascii: Whether to remove non-ASCII characters (default: False).
        """
        self.config = config or {}
        self.preprocessor = EnhancedPreprocessorService(self.config)

    def process_documents(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Process a list of documents and convert them to clean text.

        Args:
            file_paths: List of file paths to process.

        Returns:
            Dictionary mapping file paths to cleaned text.
        """
        # Process the files
        results = self.preprocessor.preprocess_file_list(file_paths)
        return results

    def process_document(self, file_path: str) -> Optional[str]:
        """
        Process a single document and convert it to clean text.

        Args:
            file_path: Path to the document file.

        Returns:
            Cleaned text from the document, or None if processing failed.
        """
        return self.preprocessor.preprocess_file(file_path)

    def get_supported_formats(self) -> List[str]:
        """
        Get a list of supported file formats.

        Returns:
            List of supported file formats (without the leading dot).
        """
        extensions = self.preprocessor.get_supported_extensions()
        return [ext[1:] for ext in extensions]  # Remove the leading dot

    def get_file_format(self, file_path: str) -> str:
        """
        Get the format of a file from its path.

        Args:
            file_path: Path to the file.

        Returns:
            The file format (e.g., "pdf", "docx", "txt").
        """
        return self.preprocessor.get_file_format(file_path)


def main():
    """
    Command-line interface for the document processor.
    """
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.agents.rag.services.preprocessor.document_processor file1 [file2 ...]"
        )
        return

    file_paths = sys.argv[1:]

    # Create document processor with default configuration
    processor = DocumentProcessor()

    # Print supported formats
    print("Supported formats:", ", ".join(processor.get_supported_formats()))
    print()

    # Process the documents
    print(f"Processing {len(file_paths)} documents...")
    results = processor.process_documents(file_paths)

    # Print results
    print(
        f"\nSuccessfully processed {len(results)} out of {len(file_paths)} documents:"
    )
    for file_path, text in results.items():
        format_name = processor.get_file_format(file_path)
        print(f"\nFile: {file_path}")
        print(f"Format: {format_name}")

        # Print preview of the processed text
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"Processed text preview: {preview}")
        print(f"Text length: {len(text)} characters")


if __name__ == "__main__":
    main()
