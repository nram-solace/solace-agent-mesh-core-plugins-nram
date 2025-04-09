"""
Action for ingesting documents into the RAG system.
This action scans documents from various data sources and ingests them into a vector database.
"""

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse
from solace_ai_connector.common.log import log

# To import from a local file, like this file, use a relative path from the rag
# For example, to load this class, use:
#   from rag.actions.ingestion_action import IngestionAction


class RAGAction(Action):

    def __init__(self, **kwargs):
        super().__init__(
            {
                "name": "rag_action",
                "prompt_directive": (
                    "This action ingest documents into a vector database."
                ),
                "params": [
                    {
                        "name": "document",
                        "desc": "The document name with the suffix.",
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "object",
                                "desc": "The original file",
                            },
                            "name": {
                                "type": "string",
                                "desc": "The name of the document with suffix",
                            },
                        },
                        "default": {},
                        "required": False,
                    },
                    {
                        "name": "query",
                        "desc": "The query to search for",
                        "type": "string",
                        "default": "",
                        "required": True,
                    },
                ],
                "required_scopes": ["rag:ingestion_action:write"],
            },
            **kwargs,
        )

    def invoke(self, params, meta={}) -> ActionResponse:
        log.debug("Starting document ingestion process")
        document = params.get("document")
        if (
            document
            and isinstance(document, dict)
            and "source" in document
            or "name" in document
        ):
            source = document["source"]
            document_name = document["name"]

        query = params.get("query")

        rag = self.get_agent().get_augmentation_handler()
        results = rag.augment(query, filter=None)
        if not results:
            return ActionResponse(message="No results found for the query")
        log.debug(f"Augmentation results: {results}")

        return ActionResponse(message=results)
