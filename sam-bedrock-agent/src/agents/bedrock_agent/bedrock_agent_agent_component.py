"""The agent component for the bedrock agent"""

import os
import copy
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from solace_agent_mesh.agents.base_agent_component import (
    agent_info,
    BaseAgentComponent,
)

from bedrock_agent.actions.invoke_agent import InvokeAgent
from bedrock_agent.actions.invoke_flow import InvokeFlow

from bedrock_agent.bedrock_agent_runtime import BedrockAgentRuntime

info = copy.deepcopy(agent_info)
info["agent_name"] = "bedrock_agent"
info["class_name"] = "BedrockAgentAgentComponent"
info["description"] = "Runs the Amazon Bedrock agent"

class BedrockAgentAgentComponent(BaseAgentComponent):
    """The agent component for the bedrock agent"""

    info = info
    actions = [] # Actions will be populated from configuration


    def __init__(self, **kwargs):
        super().__init__(info, **kwargs)
        self.kwargs = kwargs
        self.agent_name = self.get_config("agent_name")
        self.agent_description = self.get_config("agent_description")
        self.always_open = self.get_config("always_open", False)

        info["agent_name"] = self.agent_name
        info["description"] = self.agent_description
        info["always_open"] = self.always_open

        amazon_bedrock_runtime_config = self.get_config("amazon_bedrock_runtime_config", {})
        boto3_config = amazon_bedrock_runtime_config.get("boto3_config", {})
        endpoint_url = amazon_bedrock_runtime_config.get("endpoint_url", None)

        if not boto3_config:
            raise ValueError("Missing boto3_config in amazon_bedrock_runtime_config.")
        
        # Initialize the Bedrock agent runtime with the provided configuration
        self.bedrock_agent_runtime = BedrockAgentRuntime(boto3_config, endpoint_url)
        
        # Create action instances from configuration
        bedrock_agents = self.get_config("bedrock_agents", [])
        bedrock_flows = self.get_config("bedrock_flows", [])

        if not bedrock_agents and not bedrock_flows:
            raise ValueError("No actions configured for Bedrock agent. bedrock_agents or bedrock_flows must be provided.")
        
        self.action_list = self.get_actions_list(agent=self, config_fn=self.get_config)

        for action_config in bedrock_agents:
            # Create a new action instance for each configured action
            action = InvokeAgent(
                {
                    "name": action_config["name"],
                    "description": action_config["description"],
                    "bedrock_agent_id": action_config["bedrock_agent_id"],
                    "bedrock_agent_alias_id": action_config["bedrock_agent_alias_id"],
                    "param_description": action_config.get("param_description"),
                },
                self.bedrock_agent_runtime,
                agent=self,
                config_fn=self.get_config
            )
            self.action_list.add_action(action)

        
        for action_config in bedrock_flows:
            # Create a new action instance for each configured action
            action = InvokeFlow(
                {
                    "name": action_config["name"],
                    "description": action_config["description"],
                    "bedrock_flow_id": action_config["bedrock_flow_id"],
                    "bedrock_flow_alias_id": action_config["bedrock_flow_alias_id"],
                    "param_description": action_config.get("param_description"),
                },
                self.bedrock_agent_runtime,
                agent=self,
                config_fn=self.get_config
            )
            self.action_list.add_action(action)

        # Initialize the action list with the configured actions
        self.action_list.fix_scopes("<agent_name>", self.agent_name)

    def get_agent_summary(self):
        """Get a summary of the agent's capabilities."""

        return {
            "agent_name": self.agent_name,
            "description": self.agent_description,
            "always_open": self.always_open,
            "actions": self.get_actions_summary(),
        }
