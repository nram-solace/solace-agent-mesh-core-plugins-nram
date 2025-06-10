"""Amazon Bedrock Flow invoke action."""

from solace_ai_connector.common.log import log
from typing import Dict, Any
from botocore.exceptions import ClientError

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo

class InvokeFlow(Action):
    """Invoke an Amazon Bedrock flow action."""

    def __init__(self, action_config: Dict[str, Any], bedrock_agent_runtime, **kwargs):
        """Initialize the bedrock flow InvokeFlow action.

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

        self.bedrock_flow_id = action_config.get("bedrock_flow_id")
        self.bedrock_flow_alias_id = action_config.get("bedrock_flow_alias_id")

        if not self.bedrock_flow_id or not self.bedrock_flow_alias_id:
            raise ValueError("Missing required configuration for Bedrock flow ID or alias ID.")
        
    def invoke(self, params, meta={}) -> ActionResponse:
        user_input = params.get("input")

        try:
            inputs_data = [
                {
                    "content": {
                        "document": user_input
                    },
                    "nodeName": "FlowInputNode",
                    "nodeOutputName": "document"
                }
            ]

            result = self.bedrock_agent_runtime.invoke_flow(
                self.bedrock_flow_id,
                self.bedrock_flow_alias_id,
                inputs_data,
            )
            return ActionResponse(message=result)
        except ClientError as e:
            log.error("Error invoking flow: %s", e)
            return ActionResponse(
                message=f"Error invoking bedrock flow {self.name}: {str(e)}",
                error_info=ErrorInfo(str(e))
            )

