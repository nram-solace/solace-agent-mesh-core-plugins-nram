"""Responsible for processing the actions from the Event Mesh Agent"""

import copy
from typing import Dict, Any

from solace_agent_mesh.agents.base_agent_component import (
    agent_info,
    BaseAgentComponent,
)


info = copy.deepcopy(agent_info)
info.update(
    {
        "agent_name": "event_mesh",  # Default name, can be overridden in config
        "class_name": "SolaceEventMeshAgentComponent",
        "description": "Event Mesh agent for publishing requests to and receiving responses from the Solace event mesh",
        "config_parameters": [
            {
                "name": "agent_name", 
                "required": True,
                "description": "Name of this Event Mesh agent",
            },
            {
                "name": "max_response_size_before_file",
                "required": False,
                "description": "Maximum response payload size in bytes before forcing file output",
                "type": "integer",
                "default": 1024
            },
            {
                "name": "agent_description",
                "required": False,
                "description": "Description of this Event Mesh agent's purpose",
                "default": "Event Mesh agent for publishing requests to and receiving responses from the Solace event mesh",
            },
            {
                "name": "always_open",
                "required": False,
                "description": "Whether this agent should always be open",
                "type": "boolean",
                "default": False
            },
            {
                "name": "actions",
                "required": True,
                "description": "List of actions this agent can perform",
                "type": "list",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the action"
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of what the action does"
                        },
                        "parameters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Name of the parameter"
                                    },
                                    "required": {
                                        "type": "boolean",
                                        "description": "Whether this parameter is required"
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Description of the parameter"
                                    },
                                    "default": {
                                        "description": "Default value for the parameter"
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "Data type of the parameter"
                                    },
                                    "payload_path": {
                                        "type": "string",
                                        "description": "Path in the payload where this parameter value should be placed"
                                    },
                                },
                                "required": ["name", "required", "description", "type", "payload_path"]
                            }
                        },
                        "topic": {
                            "type": "string",
                            "description": "Topic to publish the action request to"
                        },
                        "response_timeout": {
                            "type": "number",
                            "description": "Timeout in seconds to wait for a response"
                        },
                        "response_format": {
                            "type": "string",
                            "description": "Expected format of response payload (json, yaml, text, or none)",
                            "default": "json"
                        },
                        "return_as_file": {
                            "type": "boolean",
                            "description": "Whether to return the response as a file using FileService",
                            "default": False
                        },
                        "required_scope": {
                            "type": "string",
                            "description": "Scope required to access this parameter",
                            "default": "<agent_name>:<action_name>:write"
                        },
                    },
                    "required": ["name", "description", "parameters", "topic", "response_topic", "response_timeout"]
                }
            }
        ],
    }
)


class SolaceEventMeshAgentComponent(BaseAgentComponent):
    """Component for handling Event Mesh operations."""

    actions = []  # Actions will be populated from configuration

    def __init__(self, module_info: Dict[str, Any] = None, **kwargs):
        """Initialize the Event Mesh agent component.

        Args:
            module_info: Optional module configuration.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If any action configuration is invalid.
        """
        module_info = module_info or info
        super().__init__(module_info, **kwargs)
        self.info = copy.deepcopy(module_info)

        self.agent_name = self.get_config("agent_name")

        self.agent_description = self.get_config("agent_description")
        module_info["agent_name"] = self.agent_name
        self.info["always_open"] = self.get_config("always_open", False)

        # Create action instances from configuration
        actions_config = self.get_config("actions", [])
        if not actions_config:
            raise ValueError("No actions configured for Event Mesh agent")

        from .actions.broker_request_response import BrokerRequestResponse
        
        self.action_list = self.get_actions_list(agent=self, config_fn=self.get_config)

        for action_config in actions_config:
            # Create a new action instance for each configured action
            action = BrokerRequestResponse(action_config, agent=self, config_fn=self.get_config)
            # Validate payload paths at startup
            action._validate_payload_paths()
            self.action_list.add_action(action)

        # Initialize the action list with the configured actions
        self.action_list.fix_scopes("<agent_name>", self.agent_name)
        self.action_list.set_agent(self)

    def get_agent_summary(self):
        """Get a summary of the agent's capabilities.

        Returns:
            Dict containing the agent's name, description and available actions.
        """
        return {
            "agent_name": self.agent_name,
            "description": self.agent_description,
            "always_open": self.info.get("always_open", False),
            "actions": self.get_actions_summary(),
        }
