"""The ingestion agent component for the rag"""

import os
import copy
import sys
from typing import Any, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from solace_agent_mesh.agents.base_agent_component import (
    agent_info,
    BaseAgentComponent,
)

# Ingestion action import
from rag.actions.ingestion import IngestAction

info = copy.deepcopy(agent_info)
info.update(
    {
        "agent_name": "ingestion",
        "class_name": "IngestionAgentComponent",
        "description": "The agent scans documents of resources, ingests documents in a vector database, and "
        "provides relevant information when a user explicitly requests it.",
        "config_parameters": [
            {
                "name": "scanner",
                "desc": "Configuration for the document scanner.",
                "type": "object",
                "properties": {
                    "use_memory_storage": {
                        "type": "boolean",
                        "desc": "Whether to use in-memory storage",
                    },
                    "source": {
                        "type": "object",
                        "desc": "Document source configuration",
                        "properties": {
                            "type": {
                                "type": "string",
                                "desc": "Source type (e.g., filesystem)",
                            },
                            "directories": {
                                "type": "array",
                                "desc": "Directories to scan",
                                "items": {"type": "string"},
                            },
                            "filters": {
                                "type": "object",
                                "desc": "File filtering options",
                                "properties": {
                                    "file_formats": {
                                        "type": "array",
                                        "desc": "Supported file formats",
                                        "items": {"type": "string"},
                                    },
                                    "max_file_size": {
                                        "type": "number",
                                        "desc": "Maximum file size in KB",
                                    },
                                },
                            },
                        },
                    },
                    "database": {
                        "type": "object",
                        "desc": "Database configuration for metadata",
                        "properties": {
                            "type": {
                                "type": "string",
                                "desc": "Database type (e.g., postgresql)",
                            },
                            "dbname": {"type": "string", "desc": "Database name"},
                            "host": {"type": "string", "desc": "Database host"},
                            "port": {"type": "number", "desc": "Database port"},
                            "user": {"type": "string", "desc": "Database username"},
                            "password": {"type": "string", "desc": "Database password"},
                        },
                    },
                    "schedule": {
                        "type": "object",
                        "desc": "Scanning schedule configuration",
                        "properties": {
                            "interval": {
                                "type": "number",
                                "desc": "Scan interval in seconds",
                            }
                        },
                    },
                },
            },
            {
                "name": "preprocessor",
                "desc": "Configuration for document preprocessing.",
                "type": "object",
                "properties": {
                    "default_preprocessor": {
                        "type": "object",
                        "desc": "Default preprocessing settings",
                        "properties": {
                            "type": {"type": "string", "desc": "Preprocessor type"},
                            "params": {
                                "type": "object",
                                "desc": "Preprocessing parameters",
                                "properties": {
                                    "remove_stopwords": {
                                        "type": "boolean",
                                        "desc": "Whether to remove stopwords",
                                    },
                                    "remove_punctuation": {
                                        "type": "boolean",
                                        "desc": "Whether to remove punctuation",
                                    },
                                    "lowercase": {
                                        "type": "boolean",
                                        "desc": "Whether to convert text to lowercase",
                                    },
                                    "strip_html": {
                                        "type": "boolean",
                                        "desc": "Whether to strip HTML tags",
                                    },
                                    "strip_xml": {
                                        "type": "boolean",
                                        "desc": "Whether to strip XML tags",
                                    },
                                    "fix_unicode": {
                                        "type": "boolean",
                                        "desc": "Whether to fix unicode issues",
                                    },
                                    "normalize_whitespace": {
                                        "type": "boolean",
                                        "desc": "Whether to normalize whitespace",
                                    },
                                    "remove_urls": {
                                        "type": "boolean",
                                        "desc": "Whether to remove URLs",
                                    },
                                    "remove_emails": {
                                        "type": "boolean",
                                        "desc": "Whether to remove email addresses",
                                    },
                                    "language": {
                                        "type": "string",
                                        "desc": "Language code",
                                    },
                                },
                            },
                        },
                    },
                    "preprocessors": {
                        "type": "object",
                        "desc": "File-specific preprocessors",
                        "properties": {
                            "text": {
                                "type": "object",
                                "desc": "Text file preprocessor",
                            },
                            "pdf": {"type": "object", "desc": "PDF file preprocessor"},
                            "docx": {
                                "type": "object",
                                "desc": "DOCX file preprocessor",
                            },
                            "json": {
                                "type": "object",
                                "desc": "JSON file preprocessor",
                            },
                            "html": {
                                "type": "object",
                                "desc": "HTML file preprocessor",
                            },
                            "markdown": {
                                "type": "object",
                                "desc": "Markdown file preprocessor",
                            },
                            "csv": {"type": "object", "desc": "CSV file preprocessor"},
                        },
                    },
                },
            },
            {
                "name": "splitter",
                "desc": "Configuration for text splitting.",
                "type": "object",
                "properties": {
                    "default_splitter": {
                        "type": "object",
                        "desc": "Default text splitter settings",
                        "properties": {
                            "type": {"type": "string", "desc": "Splitter type"},
                            "params": {
                                "type": "object",
                                "desc": "Splitter parameters",
                                "properties": {
                                    "chunk_size": {
                                        "type": "number",
                                        "desc": "Size of each chunk",
                                    },
                                    "chunk_overlap": {
                                        "type": "number",
                                        "desc": "Overlap between chunks",
                                    },
                                    "separators": {
                                        "type": "array",
                                        "desc": "Text separators",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                    "splitters": {
                        "type": "object",
                        "desc": "File-specific text splitters",
                        "properties": {
                            "text": {"type": "object", "desc": "Text file splitter"},
                            "txt": {"type": "object", "desc": "TXT file splitter"},
                            "json": {"type": "object", "desc": "JSON file splitter"},
                            "html": {"type": "object", "desc": "HTML file splitter"},
                            "markdown": {
                                "type": "object",
                                "desc": "Markdown file splitter",
                            },
                            "csv": {"type": "object", "desc": "CSV file splitter"},
                        },
                    },
                },
            },
            {
                "name": "embedding",
                "desc": "Configuration for embedding generation.",
                "type": "object",
                "properties": {
                    "embedder_type": {
                        "type": "string",
                        "desc": "Type of embedder to use",
                    },
                    "embedder_params": {
                        "type": "object",
                        "desc": "Parameters for the embedder",
                    },
                    "normalize_embeddings": {
                        "type": "boolean",
                        "desc": "Whether to normalize embeddings",
                    },
                },
            },
            {
                "name": "vector_db",
                "desc": "Configuration for vector database.",
                "type": "object",
                "properties": {
                    "db_type": {
                        "type": "string",
                        "desc": "Type of vector database",
                    },
                    "db_params": {
                        "type": "object",
                        "desc": "Parameters for the vector database",
                    },
                },
            },
        ],
    }
)


class IngestionAgentComponent(BaseAgentComponent):
    info = info
    # ingest action
    actions = [IngestAction]

    def __init__(self, module_info: Dict[str, Any] = None, **kwargs):
        """Initialize the ingestion agent component.

        Args:
            module_info: Optional module configuration.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If required database configuration is missing.
        """
        module_info = module_info or info

        super().__init__(module_info, **kwargs)

        self.agent_name = self.get_config("agent_name")

        # Scanner configuration
        scanner_config = self.get_config("scanner", {})
        if scanner_config:
            source_config = scanner_config.get("source", {})
            if source_config:
                self.source_type = source_config.get("type")
                self.directories = source_config.get("directories", [])
                self.filters = source_config.get("filters", {})
            self.use_memory_storage = scanner_config.get("use_memory_storage", False)
            self.schedule = scanner_config.get("schedule")
            # Get database config if available
            self.db_config = scanner_config.get("database", {})

        # Preprocessor configuration
        preprocessor_config = self.get_config("preprocessor", {})
        if preprocessor_config:
            self.default_preprocessor = preprocessor_config.get(
                "default_preprocessor", {}
            )
            self.preprocessors = preprocessor_config.get("preprocessors", {})

        # Splitter configuration
        splitter_config = self.get_config("splitter", {})
        if splitter_config:
            self.default_splitter = splitter_config.get("default_splitter", {})
            self.splitters = splitter_config.get("splitters", {})

        # Embedding configuration
        embedding_config = self.get_config("embedding", {})
        if embedding_config:
            self.embedder_type = embedding_config.get("embedder_type")
            self.embedder_params = embedding_config.get("embedder_params", {})
            self.normalize_embeddings = embedding_config.get(
                "normalize_embeddings", False
            )

        # Vector DB configuration
        vector_db_config = self.get_config("vector_db", {})
        if vector_db_config:
            self.db_type = vector_db_config.get("db_type")
            self.db_params = vector_db_config.get("db_params", {})

        self.action_list.fix_scopes("<agent_name>", self.agent_name)

        module_info["agent_name"] = self.agent_name

    def get_agent_summary(self):
        """Get a summary of the agent's capabilities."""
        return {
            "agent_name": self.agent_name,
            "description": f"This agent ingests documents and retrieves relevant information.\n",
            "detailed_description": (
                "This agent ingests various types of documents in a vector database and retrieves relevant information.\n\n"
                f"Source of documents:\n{self.source_type}\n\n"
                f"Document types:\n{self.filters}\n\n"
            ),
            "always_open": self.info.get("always_open", False),
            "actions": self.get_actions_summary(),
        }
