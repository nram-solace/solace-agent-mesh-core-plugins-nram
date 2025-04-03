"""
Action for ingesting documents into the RAG system.
This action scans documents from various data sources and ingests them into a vector database.
"""

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse
from solace_ai_connector.common.log import log

# To import from a local file, like this file, use a relative path from the rag
# For example, to load this class, use:
#   from rag.actions.sample_action import SampleAction


class IngestionAction(Action):

    def __init__(self, **kwargs):
        super().__init__(
            {
                "name": "ingestion_action",
                "prompt_directive": (
                    "This action scans documents of data sources and ingests them into a vector database. "
                    "Examples include scanning a filesystem and indexing PDF documents."
                ),
                "params": [
                    {
                        "name": "scanner",
                        "desc": "Configuration for the document scanner.",
                        "type": "object",
                        "properties": {
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
                                    },
                                    "filters": {
                                        "type": "object",
                                        "desc": "File filtering options",
                                    },
                                },
                            },
                            "use_memory_storage": {
                                "type": "boolean",
                                "desc": "Whether to use in-memory storage",
                            },
                            "schedule": {
                                "type": "object",
                                "desc": "Scanning schedule configuration",
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
                            },
                            "preprocessors": {
                                "type": "object",
                                "desc": "File-specific preprocessors",
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
                            },
                            "splitters": {
                                "type": "object",
                                "desc": "File-specific text splitters",
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
                "required_scopes": ["rag:ingestion_action:write"],
            },
            **kwargs,
        )
        self.ingestor = None

    def invoke(self, params, meta={}) -> ActionResponse:
        log.debug("Starting document ingestion process")

    def do_action(self, action) -> ActionResponse:
        action += " Action performed"
        return ActionResponse(message=action)
