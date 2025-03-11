"""
Example script demonstrating how to use text splitters.

This script shows how to:
1. Use different types of text splitters
2. Use the SplitterService to automatically select the appropriate splitter
3. Split text from different sources (string, file)
4. Compare the results of different splitters
"""

import os
import sys
import argparse
from typing import Dict, List, Any

# Add the parent directory to the path to allow importing from sibling packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from services.splitter.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from services.splitter.splitter_service import SplitterService


def demonstrate_character_splitter(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:
    """
    Demonstrate the CharacterTextSplitter.

    Args:
        text: The text to split.
        chunk_size: The size of each chunk.
        chunk_overlap: The overlap between chunks.

    Returns:
        A list of text chunks.
    """
    print("\n=== CharacterTextSplitter ===")
    print(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")

    # Create the splitter
    splitter = CharacterTextSplitter(
        {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "separator": "\n\n"}
    )

    # Split the text
    chunks = splitter.split_text(text)

    # Print the results
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} characters):")
        print(f"{chunk[:100]}..." if len(chunk) > 100 else chunk)

    return chunks


def demonstrate_recursive_character_splitter(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:
    """
    Demonstrate the RecursiveCharacterTextSplitter.

    Args:
        text: The text to split.
        chunk_size: The size of each chunk.
        chunk_overlap: The overlap between chunks.

    Returns:
        A list of text chunks.
    """
    print("\n=== RecursiveCharacterTextSplitter ===")
    print(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")

    # Create the splitter
    splitter = RecursiveCharacterTextSplitter(
        {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separators": ["\n\n", "\n", " ", ""],
        }
    )

    # Split the text
    chunks = splitter.split_text(text)

    # Print the results
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} characters):")
        print(f"{chunk[:100]}..." if len(chunk) > 100 else chunk)

    return chunks


def demonstrate_token_splitter(
    text: str, chunk_size: int = 500, chunk_overlap: int = 100
) -> List[str]:
    """
    Demonstrate the TokenTextSplitter.

    Args:
        text: The text to split.
        chunk_size: The size of each chunk in tokens.
        chunk_overlap: The overlap between chunks in tokens.

    Returns:
        A list of text chunks.
    """
    print("\n=== TokenTextSplitter ===")
    print(f"Chunk size: {chunk_size} tokens, Chunk overlap: {chunk_overlap} tokens")

    try:
        # Create the splitter
        splitter = TokenTextSplitter(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "encoding_name": "cl100k_base",
            }
        )

        # Split the text
        chunks = splitter.split_text(text)

        # Print the results
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1} ({len(chunk)} characters):")
            print(f"{chunk[:100]}..." if len(chunk) > 100 else chunk)

        return chunks
    except ImportError:
        print("TokenTextSplitter requires the tiktoken package.")
        print("Please install it with `pip install tiktoken`.")
        return []


def demonstrate_splitter_service(text: str, data_type: str) -> List[str]:
    """
    Demonstrate the SplitterService.

    Args:
        text: The text to split.
        data_type: The type of data to split.

    Returns:
        A list of text chunks.
    """
    print(f"\n=== SplitterService with data type: {data_type} ===")

    # Create the splitter service
    service = SplitterService()

    # Split the text
    chunks = service.split_text(text, data_type)

    # Print the results
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} characters):")
        print(f"{chunk[:100]}..." if len(chunk) > 100 else chunk)

    return chunks


def split_file(file_path: str) -> None:
    """
    Split a file using different splitters.

    Args:
        file_path: The path to the file to split.
    """
    print(f"\n=== Splitting file: {file_path} ===")

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Read the file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        print(f"File size: {len(text)} characters")

        # Get the file extension
        _, ext = os.path.splitext(file_path)
        data_type = ext[1:] if ext else "text"  # Remove the leading dot

        # Create the splitter service
        service = SplitterService()

        # Split the file
        chunks = service.split_file(file_path)

        # Print the results
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1} ({len(chunk)} characters):")
            print(f"{chunk[:100]}..." if len(chunk) > 100 else chunk)

    except Exception as e:
        print(f"Error splitting file: {str(e)}")


def compare_splitters(text: str) -> None:
    """
    Compare different splitters on the same text.

    Args:
        text: The text to split.
    """
    print("\n=== Comparing Splitters ===")

    # Define the splitters to compare
    splitters = {
        "CharacterTextSplitter": CharacterTextSplitter(
            {"chunk_size": 1000, "chunk_overlap": 200}
        ),
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter(
            {"chunk_size": 1000, "chunk_overlap": 200}
        ),
    }

    try:
        # Add TokenTextSplitter if tiktoken is available
        splitters["TokenTextSplitter"] = TokenTextSplitter(
            {"chunk_size": 500, "chunk_overlap": 100}
        )
    except ImportError:
        print(
            "TokenTextSplitter requires the tiktoken package and is not included in the comparison."
        )

    # Compare the splitters
    results = {}
    for name, splitter in splitters.items():
        print(f"\nSplitting with {name}...")
        chunks = splitter.split_text(text)
        results[name] = chunks
        print(f"Number of chunks: {len(chunks)}")
        print(
            f"Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0:.2f} characters"
        )

    # Print a summary
    print("\nSummary:")
    for name, chunks in results.items():
        print(
            f"{name}: {len(chunks)} chunks, avg size: {sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0:.2f} characters"
        )


def main():
    """
    Main function to demonstrate text splitters.
    """
    parser = argparse.ArgumentParser(description="Demonstrate text splitters.")
    parser.add_argument("--file", help="Path to a file to split")
    parser.add_argument("--text", help="Text to split")
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Chunk size (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="Chunk overlap (default: 200)"
    )
    parser.add_argument("--data-type", default="text", help="Data type (default: text)")

    args = parser.parse_args()

    # Use the provided text or a default example
    text = (
        args.text
        or """
    # Example Text for Splitting
    
    This is an example text that will be split into chunks using different text splitters.
    
    ## Section 1
    
    Text splitters are an important component of RAG (Retrieval-Augmented Generation) systems.
    They break down large documents into smaller chunks that can be embedded and stored in a vector database.
    
    ## Section 2
    
    There are different types of text splitters:
    
    1. CharacterTextSplitter: Splits text by characters.
    2. RecursiveCharacterTextSplitter: Splits text recursively by different separators.
    3. TokenTextSplitter: Splits text by tokens.
    
    ## Section 3
    
    The choice of text splitter depends on the specific requirements of your application.
    For most cases, RecursiveCharacterTextSplitter is a good default choice.
    
    ## Section 4
    
    When splitting text, it's important to consider:
    
    - Chunk size: The size of each chunk.
    - Chunk overlap: The overlap between chunks.
    - Separators: The separators to use for splitting.
    
    ## Conclusion
    
    Text splitting is a crucial step in the RAG pipeline, as it affects the quality of the embeddings and the retrieval performance.
    """
    )

    # Demonstrate different splitters
    demonstrate_character_splitter(text, args.chunk_size, args.chunk_overlap)
    demonstrate_recursive_character_splitter(text, args.chunk_size, args.chunk_overlap)

    try:
        demonstrate_token_splitter(
            text, args.chunk_size // 4, args.chunk_overlap // 4
        )  # Token sizes are typically smaller
    except ImportError:
        print("\n=== TokenTextSplitter ===")
        print("TokenTextSplitter requires the tiktoken package.")
        print("Please install it with `pip install tiktoken`.")

    demonstrate_splitter_service(text, args.data_type)

    # Split a file if provided
    if args.file:
        split_file(args.file)

    # Compare splitters
    compare_splitters(text)


if __name__ == "__main__":
    main()
