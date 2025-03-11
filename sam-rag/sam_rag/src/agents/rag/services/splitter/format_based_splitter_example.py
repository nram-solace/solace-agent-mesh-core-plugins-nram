"""
Example script demonstrating how to use the splitter service with configuration from rag.yaml.

This script shows how to:
1. Load configuration from rag.yaml
2. Initialize a SplitterService with the loaded configuration
3. Split text based on its format (json, text, pdf, etc.)
4. Display the resulting chunks

Usage:
    python -m src.agents.rag.services.splitter.format_based_splitter_example --content "Your content here" --format json
    python -m src.agents.rag.services.splitter.format_based_splitter_example --file path/to/your/file.json --format json
"""

import os
import sys
import argparse
import yaml
from typing import Dict, List, Any, Optional

# Add the parent directory to the path to allow importing from sibling packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from services.splitter.splitter_service import SplitterService


def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        A dictionary containing the configuration.
    """
    try:
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)

        # Extract the splitter configuration from the rag.yaml
        for flow in config.get("flows", []):
            if flow.get("name") == "rag_action_request_processor":
                for component in flow.get("components", []):
                    if component.get("component_name") == "action_request_processor":
                        return component.get("component_config", {}).get("splitter", {})

        print("Warning: Could not find splitter configuration in the YAML file.")
        return {}
    except Exception as e:
        print(f"Error loading configuration from {yaml_path}: {str(e)}")
        return {}


def split_text_by_format(
    content: str, format_type: str, config: Dict[str, Any]
) -> List[str]:
    """
    Split text based on its format using the SplitterService.

    Args:
        content: The text content to split.
        format_type: The format of the content (json, text, pdf, etc.).
        config: The splitter configuration.

    Returns:
        A list of text chunks.
    """
    # Initialize the SplitterService with the configuration
    splitter_service = SplitterService(config)

    # Split the text based on its format
    chunks = splitter_service.split_text(content, format_type)

    return chunks


def display_chunks(chunks: List[str], max_preview_length: int = 100) -> None:
    """
    Display the chunks with a preview of each.

    Args:
        chunks: The list of text chunks.
        max_preview_length: Maximum length of the preview.
    """
    print(f"\nTotal chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        preview = (
            chunk[:max_preview_length] + "..."
            if len(chunk) > max_preview_length
            else chunk
        )
        print(f"\nChunk {i+1}/{len(chunks)} ({len(chunk)} characters):")
        print(f"{preview}")


def main():
    """
    Main function to demonstrate the format-based text splitter.
    """
    parser = argparse.ArgumentParser(
        description="Split text based on its format using configuration from rag.yaml."
    )
    parser.add_argument("--content", help="Text content to split")
    parser.add_argument("--file", help="Path to a file containing the content to split")
    parser.add_argument(
        "--format", required=True, help="Format of the content (json, text, pdf, etc.)"
    )
    parser.add_argument(
        "--config",
        default="configs/agents/rag.yaml",
        help="Path to the configuration YAML file",
    )

    args = parser.parse_args()

    # Check if either content or file is provided
    if not args.content and not args.file:
        parser.error("Either --content or --file must be provided")

    # Load the configuration from the YAML file
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../..", args.config)
    )
    config = load_config_from_yaml(config_path)

    if not config:
        print(f"Could not load configuration from {config_path}")
        return

    print(f"Loaded splitter configuration from {config_path}")

    # Get the content from either the command line argument or the file
    content = args.content
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as file:
                content = file.read()
        except Exception as e:
            print(f"Error reading file {args.file}: {str(e)}")
            return

    # Split the text based on its format
    print(f"Splitting content with format: {args.format}")
    chunks = split_text_by_format(content, args.format, config)

    # Display the chunks
    display_chunks(chunks)


if __name__ == "__main__":
    main()
