"""
Generic preprocessor component that gets a list of documents in various formats,
converts them to text, and trims text after conversion.

This script demonstrates the exact functionality requested in the task.
"""

import sys
from typing import Dict, List, Optional
from .document_processor import DocumentProcessor


def preprocess_documents(
    file_paths: List[str], config: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Preprocess a list of documents in various formats and convert them to clean text.

    Args:
        file_paths: List of file paths to process.
        config: Optional configuration dictionary for text preprocessing options.

    Returns:
        Dictionary mapping file paths to cleaned text.
    """
    # Create a document processor with the provided configuration
    processor = DocumentProcessor(config)

    # Process the documents
    results = processor.process_documents(file_paths)

    return results


def main():
    """
    Main function to demonstrate the generic preprocessor component.
    """
    # Check if file paths were provided
    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.agents.rag.services.preprocessor.generic_preprocessor file1 [file2 ...]"
        )
        print(
            "Example: python -m src.agents.rag.services.preprocessor.generic_preprocessor /path/to/document.pdf /path/to/document.docx"
        )
        return

    # Get file paths from command line arguments
    file_paths = sys.argv[1:]

    print(f"Processing {len(file_paths)} files...")

    # Define preprocessing configuration
    config = {
        "lowercase": True,  # Convert all text to lowercase
        "normalize_unicode": True,  # Normalize Unicode characters
        "normalize_whitespace": True,  # Remove excessive spaces, tabs, or newlines
        "remove_punctuation": True,  # Remove unnecessary punctuation
        "remove_special_chars": True,  # Remove symbols like @#$%^&*()
        "remove_urls": True,  # Strip out links
        "remove_html_tags": True,  # Strip out HTML tags
        "remove_numbers": False,  # Keep numbers (set to True to remove)
        "remove_non_ascii": False,  # Keep non-ASCII characters (set to True to remove)
    }

    # Process the files
    results = preprocess_documents(file_paths, config)

    # Print results
    print(f"\nSuccessfully processed {len(results)} out of {len(file_paths)} files:")

    # Create a processor to get file formats
    processor = DocumentProcessor(config)

    for file_path, text in results.items():
        # Get file format
        file_format = processor.get_file_format(file_path)

        print(f"\nFile: {file_path}")
        print(f"Format: {file_format}")

        # Print preview of the processed text
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"Processed text preview: {preview}")
        print(f"Text length: {len(text)} characters")


if __name__ == "__main__":
    main()
