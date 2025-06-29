"""Action for model persistence (save/load models)."""

from typing import Dict, Any, List
import json
import yaml
from pathlib import Path
from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import (
    ActionResponse,
    ErrorInfo,
    InlineFile,
)
from solace_ai_connector.common.log import log

class ModelPersistence(Action):
    """Action for saving and loading trained models."""

    def __init__(self, **kwargs):
        super().__init__(
            {
                "name": "model_persistence",
                "prompt_directive": (
                    "Save or load trained machine learning models. "
                    "Supports saving models to disk and loading them back for inference."
                ),
                "params": [
                    {"name": "action", "desc": "Action to perform (save or load)", "type": "string", "required": True},
                    {"name": "model_name", "desc": "Name of the model", "type": "string", "required": True},
                    {"name": "file_path", "desc": "Path to save/load the model", "type": "string", "required": True},
                    {"name": "response_format", "desc": "Format of the response (yaml, json)", "type": "string", "required": False, "default": "yaml"}
                ],
                "required_scopes": ["<agent_name>:model_persistence:execute"],
            },
            **kwargs,
        )

    def invoke(self, params: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        try:
            action = params.get("action")
            model_name = params.get("model_name")
            file_path = params.get("file_path")
            response_format = params.get("response_format", "yaml")

            if not action or not model_name or not file_path:
                raise ValueError("action, model_name, and file_path are required")

            ml_service = self.get_agent().get_ml_service()

            if action.lower() == "save":
                # For save action, we would need a trained model
                # This is a placeholder - in practice, you'd need to pass the model object
                result = {"status": "save_not_implemented", "message": "Save functionality requires a trained model object"}
                message = "Model save functionality requires a trained model object"
            elif action.lower() == "load":
                try:
                    model = ml_service.load_model(file_path)
                    result = {"status": "loaded", "model_name": model_name, "file_path": file_path}
                    message = f"Model '{model_name}' loaded successfully from {file_path}"
                except Exception as e:
                    result = {"status": "error", "error": str(e)}
                    message = f"Failed to load model: {str(e)}"
            else:
                raise ValueError("Action must be 'save' or 'load'")

            # Prepare response
            if response_format.lower() == "json":
                content = json.dumps(result, indent=2, default=str)
                file = InlineFile(filename="model_persistence.json", content=content, content_type="application/json")
            else:
                content = yaml.dump(result, default_flow_style=False, allow_unicode=True)
                file = InlineFile(filename="model_persistence.yaml", content=content, content_type="text/yaml")

            return ActionResponse(
                message=message,
                files=[file],
                metadata={"action": action, "model_name": model_name, "file_path": file_path}
            )
        except Exception as e:
            log.error("Error in model persistence: %s", str(e))
            return ActionResponse(
                message=f"Error in model persistence: {str(e)}",
                error_info=ErrorInfo(str(e)),
            ) 