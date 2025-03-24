"""MongoDB Agent Component for handling database operations."""

import copy
from typing import Dict, Any, List, Tuple

from solace_agent_mesh.agents.base_agent_component import (
    agent_info,
    BaseAgentComponent,
)
from solace_ai_connector.components.general.db.mongo.mongo_handler import MongoHandler
from .actions.search_query import SearchQuery


info = copy.deepcopy(agent_info)
info.update(
    {
        "agent_name": "mongodb",
        "class_name": "MongoDBAgentComponent",
        "description": "MongoDB agent for executing queries based on natural language prompts",
        "config_parameters": [
            {
                "name": "always_open",
                "required": False,
                "description": "Whether this agent should always be open",
                "type": "boolean",
                "default": False
            },
            {
                "name": "database_host",
                "required": True,
                "description": "MongoDB host",
                "type": "string",
            },
            {
                "name": "database_port",
                "required": True,
                "description": "MongoDB port",
                "type": "integer",
            },
            {
                "name": "database_user",
                "required": False,
                "description": "MongoDB user",
                "type": "string",
            },
            {
                "name": "database_password",
                "required": False,
                "description": "MongoDB password",
                "type": "string",
            },
            {
                "name": "agent_name",
                "required": True,
                "description": "Name of this MongoDB agent",
            },
            {
                "name": "database_name",
                "required": True,
                "description": "Database name",
                "type": "string",
            },
            {
                "name": "database_collection",
                "required": False,
                "description": "Collection name - if not provided, all collections will be used",
            },
            {
                "name": "database_purpose",
                "required": True,
                "description": "Purpose of the database",
                "type": "string",
            },
            {
                "name": "data_description",
                "required": True,
                "description": "Detailed description of the data held in the database",
                "type": "string",
            },
            {
                "name": "auto_detect_schema",
                "required": False,
                "description": "Automatically create a schema based on the database structure",
                "type": "boolean",
                "default": False,
            },
            {
                "name": "database_schema",
                "required": False,
                "description": "Database schema if auto_detect_schema is False. Document structure is not required, just the collection names and fields.",
                "type": "string",
            },
            {
                "name": "max_inline_results",
                "required": False,
                "description": "Maximum number of results to return inline before using a file",
                "type": "integer",
                "default": 10,
            },
        ],
    }
)


class MongoDBAgentComponent(BaseAgentComponent):
    """Component for handling MongoDB database operations."""

    actions = [SearchQuery]

    def __init__(self, module_info: Dict[str, Any] = None, **kwargs):
        """Initialize the MongoDB agent component.

        Args:
            module_info: Optional module configuration.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If required database configuration is missing.
        """
        module_info = module_info or info

        super().__init__(module_info, **kwargs)
        self.info = copy.deepcopy(module_info)
        self.info["always_open"] = self.get_config("always_open", False)

        self.agent_name = self.get_config("agent_name")
        self.database_purpose = self.get_config("database_purpose")
        self.data_description = self.get_config("data_description")
        self.database_collection = self.get_config("database_collection")
        self.auto_detect_schema = self.get_config("auto_detect_schema", False)

        self.action_list.fix_scopes("<agent_name>", self.agent_name)
        self.database_schema = None

        module_info["agent_name"] = self.agent_name

        # Initialize MongoDB handler
        self.db_handler = MongoHandler(
            self.get_config("database_host"),
            self.get_config("database_port"),
            self.get_config("database_user"),
            self.get_config("database_password"),
            self.get_config("database_collection"),
            self.get_config("database_name"),
        )

        if self.auto_detect_schema:
            self.detailed_schema, self.summary_schema = self._detect_schema()
        else:
            self.detailed_schema = self.get_config("database_schema")
            self.summary_schema = self.get_config("database_schema")

    def _detect_schema(self) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, str]]:
        """Detect the database schema and include sample data.

        Returns:
            A tuple containing:
            - A dictionary where keys are collection names and values are dictionaries
              containing field names and sample data.
            - A dictionary where keys are collection names and values are comma-separated
              lists of field names.
        """
        detailed_schema = {}
        summary_schema = {}

        if self.database_collection:
            collections = [self.database_collection]
        else:
            collections = self.db_handler.get_collections()
        
        for collection in collections:
            fields = self.db_handler.get_fields(collection)
            sample_data = self._get_sample_data(collection, fields)
            detailed_schema[collection] = sample_data
            summary_schema[collection] = ", ".join(fields)
            
        return detailed_schema, summary_schema

    def _get_sample_data(self, collection: str, fields: List[str]) -> Dict[str, str]:
        """Get sample data for a collection.

        Args:
            collection: Name of the collection.
            fields: List of field names.

        Returns:
            A dictionary where keys are field names and values are strings of sample data.
        """
        sample_data = {}
        for field in fields:
            unique_values, all = self.db_handler.get_sample_values(collection, field)
            truncated_values = [value if len(value) <= 40 else value[:37] + "..." for value in unique_values]
            if all:
                sample_data[field] = f"(all unique values: {', '.join(truncated_values)})"
            else:
                sample_data[field] = f"(examples: {', '.join(truncated_values)})"

        return sample_data

    def get_db_handler(self):
        """Get the database handler instance."""
        return self.db_handler

    def get_agent_summary(self):
        """Get a summary of the agent's capabilities."""
        return {
            "agent_name": self.agent_name,
            "description": f"This agent provides read only access to this MongoDB database:\n\n{self.database_purpose}\n",
            "detailed_description": (
                "This agent provides read only access to this MongoDB database:\n\n"
                f"Purpose:\n{self.database_purpose}\n\n"
                f"Data Description:\n{self.data_description}\n\n"
                f"Summary Schema:\n{self.summary_schema}\n"
            ),
            "always_open": self.info.get("always_open", False),
            "actions": self.get_actions_summary(),
        }
