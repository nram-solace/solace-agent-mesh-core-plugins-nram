"""
Test script for the content_format_splitter module.

This script demonstrates how to use the content_format_splitter module
with various sample inputs of different formats.

Usage:
    python -m src.agents.rag.services.splitter.test_content_format_splitter
"""

import os
import sys
from typing import Dict, List, Any, Tuple

# Add the parent directory to the path to allow importing from sibling packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from services.splitter.content_format_splitter import (
    load_config_from_yaml,
    split_content_by_format,
    display_chunks,
)


def run_test_case(content: str, format_type: str, config: Dict[str, Any]) -> None:
    """
    Run a test case with the given content and format.

    Args:
        content: The content to split.
        format_type: The format of the content.
        config: The splitter configuration.
    """
    print(f"\n{'=' * 60}")
    print(f"Testing format: {format_type}")
    print(f"Content preview: {content[:100]}..." if len(content) > 100 else content)
    print(f"{'=' * 60}")

    # Split the content
    chunks = split_content_by_format(content, format_type, config)

    # Display the chunks
    display_chunks(chunks)


def main():
    """
    Main function to test the content_format_splitter module.
    """
    # Load the configuration from the YAML file
    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../../../..", "configs/agents/rag.yaml"
        )
    )
    config = load_config_from_yaml(config_path)

    if not config:
        print(f"Could not load configuration from {config_path}")
        return

    print(f"Loaded splitter configuration from {config_path}")

    # Test case 1: Plain text
    plain_text = """
    # Sample Document

    This is a sample document for testing the text splitter.

    ## Section 1

    Text splitters are an important component of RAG (Retrieval-Augmented Generation) systems.
    They break down large documents into smaller chunks that can be embedded and stored in a vector database.

    ## Section 2

    There are different types of text splitters:

    1. CharacterTextSplitter: Splits text by characters.
    2. RecursiveCharacterTextSplitter: Splits text recursively by different separators.
    3. TokenTextSplitter: Splits text by tokens.
    """
    run_test_case(plain_text, "text", config)

    # Test case 2: JSON
    json_text = """
    {
        "title": "Vector Databases",
        "description": "A comparison of vector databases for RAG applications",
        "databases": [
            {
                "name": "Qdrant",
                "description": "An open-source vector search engine",
                "features": ["High performance", "Scalable", "Python API"]
            },
            {
                "name": "FAISS",
                "description": "A library for efficient similarity search",
                "features": ["Fast", "Scalable", "Memory-efficient"]
            },
            {
                "name": "Milvus",
                "description": "An open-source vector database",
                "features": ["Cloud-native", "Scalable", "High performance"]
            }
        ]
    }
    """
    run_test_case(json_text, "json", config)

    # Test case 3: Markdown
    markdown_text = """
    # Retrieval-Augmented Generation (RAG)

    Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based and generation-based approaches for natural language processing tasks.

    ## How RAG Works

    1. **Retrieval**: Given a query, retrieve relevant documents from a corpus.
    2. **Augmentation**: Augment the query with the retrieved documents.
    3. **Generation**: Generate a response based on the augmented query.

    ## Benefits of RAG

    - Improved accuracy
    - Reduced hallucinations
    - Better factual grounding
    - More up-to-date information
    """
    run_test_case(markdown_text, "markdown", config)

    # Test case 4: HTML
    html_text = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Text Splitter Example</title>
    </head>
    <body>
        <h1>Text Splitter Example</h1>
        <p>This is an example of HTML content that will be split by the HTML splitter.</p>
        
        <h2>Features</h2>
        <ul>
            <li>Splits HTML content based on tags</li>
            <li>Preserves the structure of the HTML</li>
            <li>Handles nested elements</li>
        </ul>
        
        <h2>How it Works</h2>
        <p>The HTML splitter uses BeautifulSoup to parse the HTML and extract text from specified tags.</p>
        <p>It can be configured to split by different tags, such as paragraphs, divs, or sections.</p>
    </body>
    </html>
    """
    run_test_case(html_text, "html", config)

    # Test case 5: CSV
    csv_text = """
    name,age,city,occupation
    John Doe,30,New York,Software Engineer
    Jane Smith,28,San Francisco,Data Scientist
    Bob Johnson,35,Chicago,Product Manager
    Alice Brown,25,Boston,UX Designer
    Charlie Wilson,40,Seattle,DevOps Engineer
    """
    run_test_case(csv_text, "csv", config)

    print("\nAll test cases completed.")


if __name__ == "__main__":
    main()
