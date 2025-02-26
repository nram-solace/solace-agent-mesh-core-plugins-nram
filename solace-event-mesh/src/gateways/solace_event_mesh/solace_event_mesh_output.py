"""Output component for sending responses back to the Solace event mesh"""

import copy
import json
import re
import yaml
import base64


from solace_ai_connector.components.inputs_outputs.broker_output import BrokerOutput
from solace_ai_connector.components.inputs_outputs.broker_output import (
    info as broker_output_info,
)
from solace_ai_connector.common.log import log
from solace_ai_connector.common.utils import encode_payload
from solace_agent_mesh.common.utils import remove_config_parameter

# Deep copy the broker output info and update it
info = copy.deepcopy(broker_output_info)
# info = broker_output_info

# Remove the payload_encoding and payload_format from the config_parameters
remove_config_parameter(info, "payload_encoding")
remove_config_parameter(info, "payload_format")

# Update some fields
info.update(
    {
        "class_name": "EventMeshOutput",
        "description": "Component that sends responses back to the Solace event mesh",
    }
)

# Add the additional config parameters required for this component
info["config_parameters"].extend(
    [
        {
            "name": "output_handlers",
            "required": True,
            "description": "List of output handlers defining how to send messages",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique name for this output handler",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Topic to publish messages to",
                    },
                    "payload_encoding": {
                        "type": "string",
                        "description": "Encoding of the payload (utf-8, base64, gzip, none)",
                        "default": "utf-8",
                    },
                    "payload_format": {
                        "type": "string",
                        "description": "Format of the payload (json, yaml, text)",
                        "default": "json",
                    },
                    "attach_file_as_payload": {
                        "type": "boolean",
                        "description": "Whether to attach the first file in the message as the payload",
                        "default": False,
                    },
                },
                "required": ["name", "topic"],
            },
        },
        {
            "name": "output_handler_name_expression",
            "type": "string",
            "description": "Expression to select output handler name from message",
            "required": False,
        },
    ],
)


class EventMeshOutput(BrokerOutput):
    """Component that sends responses back to the Solace event mesh"""

    def __init__(self, **kwargs):
        # Payload encoding and format are not required for this component
        # Make sure they are set to none always
        kwargs["payload_encoding"] = "none"
        kwargs["payload_format"] = "none"
        super().__init__(module_info=info, **kwargs)

        # Validate output handlers configuration
        handlers = self.get_config("output_handlers", [])
        if not handlers:
            raise ValueError(f"{self.log_identifier}No output handlers configured")

        self.output_handlers = {}
        for handler in handlers:
            if not handler.get("name"):
                raise ValueError(
                    f"{self.log_identifier}Output handler missing required 'name' field"
                )
            if not handler.get("topic"):
                raise ValueError(
                    f"{self.log_identifier}Output handler '{handler['name']}' missing required 'topic' field"
                )
            self.output_handlers[handler["name"]] = handler

        self.output_handler_name_expression = self.get_config(
            "output_handler_name_expression"
        )

    def get_handler_for_message(self, message):
        """Get the appropriate output handler for a message"""
        handler_name = None

        # Try getting handler name from expression if configured
        if self.output_handler_name_expression:
            try:
                handler_name = message.get_data(self.output_handler_name_expression)
            except Exception as e:
                log.warning(
                    "%sFailed to get handler name from expression: %s",
                    self.log_identifier,
                    str(e),
                )

        # Fall back to user properties if no expression or it failed
        if not handler_name:
            handler_name = message.get_user_properties().get("output_handler_name")

        # Use default handler if no name found
        if not handler_name:
            handler_name = "default"

        return self.output_handlers.get(handler_name)

    def invoke(self, message, data):
        result = data.get("payload")

        if not result:
            log.debug("%sNo payload found in message", self.log_identifier)
            self.discard_current_message()
            return None
        
        if not result.get("last_chunk"):
            self.discard_current_message()
            return None

        handler = self.get_handler_for_message(message)
        if not handler:
            log.error(
                "%sNo matching output handler found for message", self.log_identifier
            )
            self.discard_current_message()
            return None

        # Determine payload based on handler config
        payload = None
        if handler.get("attach_file_as_payload"):
            if not result.get("files") or len(result["files"]) == 0:
                # It is possible to have no files in the message, since multiple messages can be sent
                self.discard_current_message()
                return None

            file = result["files"][0]
            if not file.get("content"):
                log.error(
                    "%sNo content found in file when attach_file_as_payload is True",
                    self.log_identifier,
                )
                self.discard_current_message()
                return None

            # File is base64 encoded, so we need to decode it
            content = file["content"]
            try:
                payload = base64.b64decode(content)
            except Exception as e:
                log.error(
                    "%sFailed to decode base64 content: %s", self.log_identifier, str(e)
                )
                self.discard_current_message()
                return None

        else:
            if "text" not in result:
                log.error("%sNo text field found in message", self.log_identifier)
                self.discard_current_message()
                return None
            payload = result["text"]


        # Encode payload according to handler config
        try:
            payload = self.parse_and_clean_content(payload, handler.get("payload_format", "text"))
            encoded_payload = encode_payload(
                payload,
                handler.get("payload_encoding", "utf-8"),
                handler.get("payload_format", "json"),
            )
        except Exception as e:
            log.error("%sFailed to encode payload: %s", self.log_identifier, str(e))
            self.discard_current_message()
            return None

        return {"payload": encoded_payload, "topic": handler["topic"]}

    def parse_and_clean_content(self, content, format):
        """Parse and clean the content based on the format"""
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        if "```" in content:
            parts = re.split(r"```[\w]*\n", content)
            if len(parts) > 1:
                content = parts[1]
        if "<response>" in content:
            parts = re.split(r"</?response>", content)
            if len(parts) > 1:
                content = parts[1]
        if format == "json":
            try:
                return json.loads(content)
            except Exception as e:
                log.error(
                    "%sFailed to parse content as JSON: %s", self.log_identifier, str(e)
                )
                return content
        elif format == "yaml":
            try:
                return yaml.safe_load(content)
            except Exception as e:
                log.error(
                    "%sFailed to parse content as YAML: %s", self.log_identifier, str(e)
                )
                return content
        else:
            return content
