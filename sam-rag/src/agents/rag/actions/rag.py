"""
Action for retrieving documents from a RAG system.
This action retrieves documents from a vector database and augments the results.
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
                    "This action retrieves documents from a vector database."
                ),
                "params": [
                    {
                        "name": "query",
                        "desc": "The query to search for",
                        "type": "string",
                        "default": "",
                        "required": True,
                    },
                ],
                "required_scopes": ["rag:rag_action:write"],
            },
            **kwargs,
        )

    def invoke(self, params, meta={}) -> ActionResponse:
        log.debug("Starting document retrieval process")

        query = params.get("query")
        if not query:
            return ActionResponse(message="Query parameter is required")

        rag = self.get_agent().get_augmentation_handler()
        results = rag.augment(query, filter=None)
        if not results:
            return ActionResponse(message="No results found for the query")
        log.debug(f"Augmentation results: {results}")

        return ActionResponse(message=results)
