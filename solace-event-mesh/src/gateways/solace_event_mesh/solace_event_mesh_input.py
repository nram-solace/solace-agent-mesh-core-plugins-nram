import uuid
import copy

from typing import Dict, Any
from solace_ai_connector.components.inputs_outputs.broker_input import (
    BrokerInput,
    info as broker_input_info,
)
from solace_ai_connector.common.message import Message
from solace_ai_connector.common.log import log
from solace_ai_connector.transforms.transforms import Transforms
from solace_ai_connector.common.utils import decode_payload
from solace_agent_mesh.common.utils import match_solace_topic, remove_config_parameter

DEFAULT_TIMEOUT_MS = 1000  # 1 second timeout for receiving messages

info = copy.deepcopy(broker_input_info)

# Remove the payload_encoding and payload_format from the config_parameters
remove_config_parameter(info, "payload_encoding")
remove_config_parameter(info, "payload_format")
remove_config_parameter(info, "broker_subscriptions")

info.update(
    {
        "class_name": "EventMeshInput",
        "description": "Component that receives events from Solace event mesh and prepares them for Solace Agent Mesh processing",
    }
)

info["config_parameters"].extend(
    [
        {
            "name": "event_handlers",
            "required": True,
            "description": "List of event handlers with their subscriptions and transformations",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique name for this event handler",
                    },
                    "subscriptions": {
                        "type": "array",
                        "description": "List of topic subscriptions for this handler",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic": {
                                    "type": "string",
                                    "description": "Topic pattern to subscribe to, supports wildcards * and >",
                                },
                                "qos": {
                                    "type": "integer",
                                    "description": "Quality of Service level (0, 1, or 2)",
                                    "default": 1,
                                },
                            },
                        },
                    },
                    "input_transforms": {
                        "type": "array",
                        "description": "List of transformations to apply to incoming messages",
                        "items": {"type": "object"},
                    },
                    "input_expression": {
                        "type": "string",
                        "description": "Expression template for processing the input payload",
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
                    "attach_payload_as_file": {
                        "type": "boolean",
                        "description": "Whether to attach the payload as a file",
                        "default": False,
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Name to use when attaching payload as file",
                        "default": "input_payload",
                    },
                    "file_mime_type": {
                        "type": "string",
                        "description": "MIME type to use when attaching payload as file",
                        "default": "application/octet-stream",
                    },
                    "output_handler_name": {
                        "type": "string",
                        "description": "Name of output handler to use for responses",
                        "required": False,
                    },
                },
            },
        },
        {
            "name": "output_handler_name_dest_expression",
            "type": "string",
            "description": "Expression to indicate where the output handler name should be stored in the message",
            "required": False,
            "default": "input.user_properties:output_handler_name",
        },
        {
            "name": "identity",
            "type": "string",
            "description": "Identity to put as the authenticated user for this request",
            "required": False,
            "default": None,
        },
    ]
)


class EventMeshInput(BrokerInput):
    """Component that receives events from Solace event mesh and prepares them for Solace Agent Mesh processing"""

    def __init__(self, **kwargs):
        # Initialize BrokerInput first to establish connection
        super().__init__(module_info=info, **kwargs)

        self.event_handlers = self.get_config("event_handlers", [])
        self.handler_transforms = {}

        # Add all subscriptions from event handlers to the broker
        # First ensure we have a queue to work with
        queue = self.get_config("broker_queue_name")
        if hasattr(self.messaging_service, "persistent_receivers"):
            queue = (
                self.messaging_service.persistent_receivers
                and self.messaging_service.persistent_receivers[0]
            )
        if not queue:
            raise ValueError(
                f"{self.log_identifier} Messaging service does not have a queue to subscribe to"
            )

        # Now add all subscriptions
        for handler in self.event_handlers:
            for subscription in handler.get("subscriptions", []):
                topic = subscription.get("topic")
                qos = subscription.get("qos", 1)

                if topic:
                    log.debug(
                        "%sAdding subscription for topic: %s with QoS: %d",
                        self.log_identifier,
                        topic,
                        qos,
                    )
                    try:
                        self.messaging_service.subscribe(topic, queue)
                    except Exception as e:
                        log.error(
                            "%sFailed to subscribe to topic %s: %s",
                            self.log_identifier,
                            topic,
                            str(e),
                        )
                        raise

        # Initialize transforms for each handler
        for handler in self.event_handlers:
            name = handler.get("name")
            if name:
                transforms = handler.get("input_transforms", [])
                self.handler_transforms[name] = Transforms(
                    transforms, log_identifier=f"{self.log_identifier}[{name}]"
                )

    def get_next_message(self, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> Message:
        """Get next message from broker and process it according to matching handler config"""
        broker_message = self.messaging_service.receive_message(
            timeout_ms, self.broker_properties["queue_name"]
        )
        if not broker_message:
            return None

        return self.process_message(broker_message)

    def process_message(self, broker_message: Dict[str, Any]) -> Message:

        self.current_broker_message = broker_message
        topic = broker_message.get("topic")
        user_properties = broker_message.get("user_properties") or {}

        # Find matching handler based on topic
        matching_handler = None
        for handler in self.event_handlers:
            for subscription in handler.get("subscriptions", []):
                if match_solace_topic(subscription.get("topic"), topic):
                    matching_handler = handler
                    break
            if matching_handler:
                break

        if not matching_handler:
            log.warning("%sNo handler found for topic: %s", self.log_identifier, topic)
            return None

        # Decode payload using handler-specific encoding
        payload = broker_message.get("payload")
        encoding = matching_handler.get("payload_encoding")
        payload_format = matching_handler.get("payload_format")
        try:
            decoded_payload = decode_payload(payload, encoding, payload_format)
        except Exception as e:
            log.error(
                "%sFailed to decode payload for handler: %s\nError: %s\nDiscarding message",
                self.log_identifier,
                matching_handler.get("name"),
                str(e),
            )
            self.discard_current_message()
            return None

        # Generate session ID
        session_id = str(uuid.uuid4())
        user_properties["session_id"] = session_id
        user_properties["input_type"] = "event_mesh"

        identity = self.get_config("identity")
        if identity:
            user_properties["identity"] = identity

        # Prepare output data structure
        output_data = {}

        if matching_handler.get("attach_payload_as_file"):
            # Add payload as file
            file_name = matching_handler.get("file_name", "input_payload")
            mime_type = matching_handler.get(
                "file_mime_type", "application/octet-stream"
            )

            output_data["files"] = [
                {
                    "name": file_name,
                    "content": payload,
                    "mime_type": mime_type,
                    "size": len(payload),
                }
            ]

        # Put input expression in text field
        input_expression = matching_handler.get("input_expression", "")

        if not input_expression:
            log.error(
                "%sNo input expression found for handler: %s",
                self.log_identifier,
                matching_handler.get("name"),
            )
            self.discard_current_message()
            return None

        message = Message(
            payload=decoded_payload, topic=topic, user_properties=user_properties
        )

        try:
            output_data["text"] = message.get_data(input_expression)
            output_data["identity"] = identity
            message.set_payload(output_data)
        except Exception as e:
            log.error(
                "%sFailed to process input expression for input event handler: %s\nEvent dump: %s",
                self.log_identifier,
                matching_handler.get("name"),
                str(e),
            )
            self.discard_current_message()
            return None

        # Store the output handler name in the message if there is an expression
        output_handler_name = matching_handler.get("output_handler_name")
        output_handler_name_dest_expression = self.get_config(
            "output_handler_name_dest_expression"
        )
        if output_handler_name and output_handler_name_dest_expression:
            message.set_data(
                output_handler_name_dest_expression,
                output_handler_name,
            )

        return message

    def invoke(self, message: Message, data: Dict[str, Any]) -> Dict[str, Any]:
        return message.get_payload()
