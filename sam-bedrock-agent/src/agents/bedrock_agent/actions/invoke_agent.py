"""Amazon Bedrock Agent invoke action."""

from solace_ai_connector.common.log import log
from typing import Dict, Any
from botocore.exceptions import ClientError

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo

class InvokeAgent(Action):
    """Invoke an Amazon Bedrock agent action."""

    def __init__(self, action_config: Dict[str, Any], bedrock_agent_runtime, **kwargs):
        """Initialize the bedrock agent InvokeAgent action.

        Args:
            action_config: Configuration dictionary containing action parameters.
            bedrock_agent_runtime: Runtime environment for the Bedrock agent.
            
        Raises:
            ValueError: If the action configuration is invalid or missing required parameters.
        """
        super().__init__({
            "name": action_config["name"],
            "prompt_directive": action_config["description"],
            "params": [
                {
                    "name": "input",
                    "desc": action_config.get("param_description", "Input to send to the action."),
                    "type": "string",
                    "required": True
                }
            ],
            "required_scopes": [action_config.get("required_scope", f"<agent_name>:{action_config['name']}:execute")],
        },
        **kwargs,
        )

        self.bedrock_agent_runtime = bedrock_agent_runtime

        self.bedrock_agent_id = action_config.get("bedrock_agent_id")
        self.bedrock_agent_alias_id = action_config.get("bedrock_agent_alias_id")

        if not self.bedrock_agent_id or not self.bedrock_agent_alias_id:
            raise ValueError("Missing required configuration for Bedrock agent ID or alias ID.")
        
    def invoke(self, params, meta={}) -> ActionResponse:
        user_input = params.get("input")
        session_id = meta.get("session_id")

        try:
            result =  self.bedrock_agent_runtime.invoke_agent(
                self.bedrock_agent_id,
                self.bedrock_agent_alias_id,
                session_id,
                user_input
            )
            return ActionResponse(message=result)
        except ClientError as e:
            log.error("Error invoking agent: %s", e)
            return ActionResponse(
                message=f"Error invoking bedrock agent {self.name}: {str(e)}",
                error_info=ErrorInfo(str(e))
            )

