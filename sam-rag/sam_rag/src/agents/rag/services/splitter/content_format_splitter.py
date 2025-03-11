"""
Example script for splitting text based on its format using configuration from rag.yaml.

This script takes input in the format of "<content>, <format>" where:
- content: The plain text content to split
- format: The format type (json, text, pdf, etc.)

Usage:
    python -m src.agents.rag.services.splitter.content_format_splitter

Example input:
    {"name": "John", "age": 30, "city": "New York"}, json
    This is a plain text document with multiple paragraphs.\n\nIt will be split by the text splitter., text
"""

import os
import sys
import yaml
from typing import Dict, List, Any, Tuple

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


def parse_input(input_text: str) -> Tuple[str, str]:
    """
    Parse the input text in the format "<content>, <format>".

    Args:
        input_text: The input text to parse.

    Returns:
        A tuple containing (content, format).
    """
    try:
        # Find the last comma in the input text
        last_comma_index = input_text.rstrip().rfind(",")
        if last_comma_index == -1:
            raise ValueError("Input must be in the format '<content>, <format>'")

        # Extract content and format
        content = input_text[:last_comma_index].strip()
        format_type = input_text[last_comma_index + 1 :].strip()

        return content, format_type
    except Exception as e:
        print(f"Error parsing input: {str(e)}")
        return "", ""


def split_content_by_format(
    content: str, format_type: str, config: Dict[str, Any]
) -> List[str]:
    """
    Split content based on its format using the SplitterService.

    Args:
        content: The content to split.
        format_type: The format of the content (json, text, pdf, etc.).
        config: The splitter configuration.

    Returns:
        A list of text chunks.
    """
    # Initialize the SplitterService with the configuration
    splitter_service = SplitterService(config)

    # Split the content based on its format
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
    Main function to demonstrate the content-format splitter.
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
    print("\nEnter input in the format '<content>, <format>'")
    print("Example: '{\"name\": \"John\"}, json' or 'This is a text document, text'")
    print("Enter 'exit' to quit")

    config.get

    while True:
        # Get input from the user
        print("\n> ", end="")
        user_input = input()

        if user_input.lower() == "exit":
            break

        # Parse the input
        content, format_type = parse_input(user_input)
        if not content or not format_type:
            print("Invalid input. Please use the format '<content>, <format>'")
            continue

        # Split the content based on its format
        print(f"Splitting content with format: {format_type}")
        chunks = split_content_by_format(content, format_type, config)

        # Display the chunks
        display_chunks(chunks)


if __name__ == "__main__":
    main()
