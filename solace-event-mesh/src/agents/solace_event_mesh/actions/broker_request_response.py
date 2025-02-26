"""Broker Request/Response action for Event Mesh agent."""

import json
import uuid
from typing import Dict, Any

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo
from solace_ai_connector.common.message import Message
from solace_ai_connector.common.log import log



class BrokerRequestResponse(Action):
    """Action for sending requests to and receiving responses from a broker."""

    def _validate_topic_template(self, topic: str) -> None:
        """Validate that a topic template is properly formatted.
        
        Valid formats include:
        - Plain text: my/topic/path
        - With parameter: my/topic/{{ param1 }}/path
        - With encoding: my/topic/{{ text://param1 }}/path
        
        Args:
            topic: The topic template string to validate.
            
        Raises:
            ValueError: If the topic template is invalid.
        """
        import re
        
        # Match {{ optional_encoding://param_name }}
        template_pattern = r'\{\{(\s*(text://)?\s*[a-zA-Z_][a-zA-Z0-9_]*\s*)\}\}'
        
        for match in re.finditer(template_pattern, topic):
            param_expr = match.group(1).strip()
            if '://' in param_expr:
                encoding, param = param_expr.split('://')
                if encoding.strip() != 'text':
                    raise ValueError(
                        f"Invalid encoding '{encoding}' in topic template '{topic}'. "
                        "Only 'text' encoding is supported."
                    )
                param = param.strip()
            else:
                param = param_expr
                
            # Verify parameter exists in config
            if not any(p['name'] == param for p in self._params):
                raise ValueError(
                    f"Topic template '{topic}' references undefined parameter '{param}'"
                )

    def __init__(self, action_config: Dict[str, Any], **kwargs):
        """Initialize the broker request/response action.

        Args:
            action_config: Configuration dictionary containing action parameters.
            
        Raises:
            ValueError: If topic templates are invalid.
        """
        super().__init__({
            "name": action_config["name"],
            "prompt_directive": action_config["description"],
            "params": [
                {
                    "name": param["name"],
                    "desc": param["description"],
                    "type": param["type"],
                    "required": param.get("required", False),
                    "default": param.get("default", None),
                } for param in action_config["parameters"]
            ],
            "required_scopes": [action_config.get("required_scope", f"<agent_name>:{action_config['name']}:write")],
        },
        **kwargs,
        )
        self.topic_template = action_config["topic"]
        self.response_timeout = action_config["response_timeout"]
        self.response_format = action_config.get("response_format", "json")
        
        if self.response_format not in ["json", "yaml", "text", "none"]:
            raise ValueError(
                f"Invalid response_format '{self.response_format}'. "
                "Must be one of: json, yaml, text, none"
            )
        self.parameters = {
            param["name"]: param for param in action_config["parameters"]
        }
        
        # Validate topic templates
        self._validate_topic_template(self.topic_template)

    def _validate_payload_paths(self):
        """Validate that all payload paths are properly formatted.
        
        Valid formats include:
        - Simple paths: field1.field2.field3
        - Array dot notation: field1.0.field2
        - Array bracket notation: field1[0].field2
        
        Raises:
            ValueError: If any payload path is invalid.
        """
        import re

        for param in self.parameters.values():
            path = param["payload_path"]
            if not isinstance(path, str) or not path:
                raise ValueError(
                    f"Invalid payload path '{path}' for parameter '{param['name']}'"
                )
            
            # Split on dots but preserve array brackets
            parts = []
            current = ''
            for char in path:
                if char == '.' and not (current.startswith('[') and ']' not in current):
                    if current:
                        parts.append(current)
                        current = ''
                else:
                    current += char
            if current:
                parts.append(current)

            if not all(parts):
                raise ValueError(
                    f"Invalid payload path '{path}' for parameter '{param['name']}'. "
                    "Path segments cannot be empty."
                )

            for part in parts:
                # Check if this part is an array index
                if part.startswith('[') and part.endswith(']'):
                    index_str = part[1:-1]
                    if not index_str.isdigit():
                        raise ValueError(
                            f"Invalid array index '{part}' in path '{path}' for parameter "
                            f"'{param['name']}'. Array indices must be non-negative integers."
                        )
                elif part.isdigit():
                    # Direct numeric index
                    pass
                elif not part.isalnum() and not '_' in part:
                    raise ValueError(
                        f"Invalid path segment '{part}' in path '{path}' for parameter "
                        f"'{param['name']}'. Path segments must be alphanumeric, "
                        "underscores, or array indices."
                    )

    def _build_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build the message payload using parameter values and their payload paths.
        
        Supports both dot notation and bracket notation for array indices:
        - field1.0.field2 
        - field1[0].field2
        
        Args:
            params: Dictionary of parameter values.
            
        Returns:
            Dict containing the structured payload.
            
        Raises:
            ValueError: If a negative array index is encountered.
        """
        payload = {}

        for name, value in params.items():
            if name in self.parameters:
                path = self.parameters[name]["payload_path"]
                current = payload
                
                # Split on dots but preserve array brackets
                parts = []
                current_part = ''
                for char in path:
                    if char == '.' and not (current_part.startswith('[') and ']' not in current_part):
                        if current_part:
                            parts.append(current_part)
                            current_part = ''
                    else:
                        current_part += char
                if current_part:
                    parts.append(current_part)
                
                # Navigate to the correct nested location
                for i, part in enumerate(parts[:-1]):
                    next_key = None
                    
                    # Handle array indices
                    if part.startswith('[') and part.endswith(']'):
                        # Bracket notation
                        index = int(part[1:-1])
                        if index < 0:
                            raise ValueError(f"Negative array index {index} not allowed")
                        if not isinstance(current, list):
                            current = []
                        while len(current) <= index:
                            current.append({})
                        next_key = index
                    elif part.isdigit():
                        # Dot notation for array index
                        index = int(part)
                        if index < 0:
                            raise ValueError(f"Negative array index {index} not allowed")
                        if not isinstance(current, list):
                            current = []
                        while len(current) <= index:
                            current.append({})
                        next_key = index
                    else:
                        # Regular dict key
                        if part not in current:
                            current[part] = {}
                        next_key = part
                    
                    current = current[next_key]
                
                # Handle the final part
                last_part = parts[-1]
                if last_part.startswith('[') and last_part.endswith(']'):
                    index = int(last_part[1:-1])
                    if index < 0:
                        raise ValueError(f"Negative array index {index} not allowed")
                    if not isinstance(current, list):
                        current = []
                    while len(current) <= index:
                        current.append(None)
                    current[index] = value
                elif last_part.isdigit():
                    index = int(last_part)
                    if index < 0:
                        raise ValueError(f"Negative array index {index} not allowed")
                    if not isinstance(current, list):
                        current = []
                    while len(current) <= index:
                        current.append(None)
                    current[index] = value
                else:
                    current[last_part] = value
                
        return payload

    def _fill_topic_template(self, template: str, params: Dict[str, Any]) -> str:
        """Fill a topic template with parameter values.
        
        Args:
            template: The topic template string.
            params: Dictionary of parameter values.
            
        Returns:
            The filled topic string.
            
        Raises:
            ValueError: If parameter substitution fails.
        """
        import re
        
        def replace_param(match):
            param_expr = match.group(1).strip()
            if '://' in param_expr:
                _, param = param_expr.split('//')
                param = param.strip()
            else:
                param = param_expr
                
            if param not in params:
                raise ValueError(f"Missing required parameter '{param}' for topic template")
            return str(params[param])
            
        return re.sub(r'\{\{(.*?)\}\}', replace_param, template)

    def invoke(self, params: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        """Execute the broker request/response action.

        Args:
            params: Action parameters from the user.
            meta: Additional metadata.

        Returns:
            ActionResponse containing the result or error information.
            For async requests, returns an ActionResponse with is_async=True
            and async_response_id set.
        """
        try:
            # Fill topic templates
            try:
                topic = self._fill_topic_template(self.topic_template, params)
            except ValueError as e:
                return ActionResponse(
                    message=f"Error filling topic template: {str(e)}",
                    error_info=ErrorInfo(str(e))
                )
            
                
            payload = self._build_payload(params)
            
            message = Message(
                payload=payload,
                topic=topic,
                # user_properties=meta or {}
            )

            agent = self.get_agent()
            if not agent.is_broker_request_response_enabled():
                raise ValueError("Broker request/response is not enabled for this agent")

            # Check if this should be an async request
            is_async = params.get("async", False)
            
            if is_async:
                
                # Generate a unique ID for this async request
                async_response_id = str(uuid.uuid4())
                
                # Store request context in cache
                cache_key = f"event_mesh_agent:async_request:{async_response_id}"
                cache_data = {
                    "params": params,
                    "meta": meta,
                    "response_format": self.response_format
                }
                agent.cache_service.add_data(
                    key=cache_key,
                    value=cache_data,
                    expiry=self.response_timeout,
                    component=self
                )
                
                # Send async request
                agent.do_broker_request_response(
                    message=message,
                    stream=False,
                    streaming_complete_expression=None,
                    async_response_id=async_response_id
                )
                
                return ActionResponse(
                    message="Request accepted for async processing",
                    is_async=True,
                    async_response_id=async_response_id
                )
            else:
                # Synchronous request
                response = agent.do_broker_request_response(
                    message=message,
                    stream=False,
                    streaming_complete_expression=None
                )

                if response is None:
                    return ActionResponse(
                        message=f"Request timed out after {self.response_timeout} seconds",
                        error_info=ErrorInfo(f"No response received within {self.response_timeout} seconds")
                    )

            payload = response.get_payload()
            
            try:
                if self.response_format == "json":
                    # Attempt JSON parsing even if format not specified
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                elif self.response_format == "yaml":
                    if isinstance(payload, str):
                        import yaml
                        payload = yaml.safe_load(payload)
                elif self.response_format == "text":
                    payload = str(payload)
                # For "none", return payload as-is
                
                return ActionResponse(message=str(payload))
                
            except Exception as e:
                error_msg = (
                    f"Error parsing response payload as {self.response_format}: {str(e)}"
                )
                log.error("%s\nPayload: %s", error_msg, payload)
                return ActionResponse(
                    message=error_msg,
                    error_info=ErrorInfo(error_msg)
                )

        except Exception as e:
            return ActionResponse(
                message=f"Error executing broker request: {str(e)}",
                error_info=ErrorInfo(str(e))
            )
