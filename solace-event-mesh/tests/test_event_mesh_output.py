import unittest
from solace_ai_connector.test_utils.utils_for_test_files import create_test_flows
from solace_ai_connector.common.message import Message


class TestEventMeshOutput(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        config = {
            "flows": [
                {
                    "name": "test_flow",
                    "components": [
                        {
                            "component_name": "solace_event_mesh_output",
                            "component_module": "src.gateways.solace_event_mesh.solace_event_mesh_output",
                            "component_config": {
                                "output_handlers": [
                                    {
                                        "name": "default",
                                        "topic": "default/topic",
                                        "payload_encoding": "utf-8",
                                        "payload_format": "json",
                                    },
                                    {
                                        "name": "custom",
                                        "topic": "custom/topic",
                                        "payload_encoding": "base64",
                                        "payload_format": "text",
                                    },
                                ],
                                "output_handler_name_expression": "input.user_properties:output_handler_name",
                                "broker_type": "dev_broker",
                                "broker_url": "tcp://localhost:55555",
                                "broker_username": "default",
                                "broker_password": "default",
                                "broker_vpn": "default",
                            },
                        }
                    ],
                }
            ]
        }
        self.connector, self.flows = create_test_flows(config)
        self.event_mesh_output = self.flows[0]["flow"].component_groups[0][0]

    def tearDown(self):
        if self.connector:
            self.connector.stop()

    def test_invoke_with_default_handler(self):
        message = Message(
            payload={"text": "Hello, World!"},
            user_properties={},
        )

        result = self.event_mesh_output.invoke(message, message.get_data("input"))

        self.assertIsNotNone(result)
        self.assertEqual(result["topic"], "default/topic")
        self.assertEqual(result["payload"], '"Hello, World!"'.encode())

    def test_invoke_with_custom_handler(self):
        message = Message(
            payload={"text": "Custom message"},
            user_properties={"output_handler_name": "custom"},
        )

        result = self.event_mesh_output.invoke(message, message.get_data("input"))

        self.assertIsNotNone(result)
        self.assertEqual(result["topic"], "custom/topic")
        self.assertEqual(
            result["payload"], b"Q3VzdG9tIG1lc3NhZ2U="
        )  # base64 encoded "Custom message"

    def test_invoke_with_no_payload(self):
        message = Message(
            payload={},
            user_properties={},
        )

        result = self.event_mesh_output.invoke(message, message.get_data("input"))

        self.assertIsNone(result)

    def test_invoke_with_attach_file_as_payload(self):
        self.event_mesh_output.output_handlers["default"][
            "attach_file_as_payload"
        ] = True
        message = Message(
            payload={
                "files": [
                    {
                        "content": "File content",
                        "name": "test.txt",
                        "mime_type": "text/plain",
                    }
                ]
            },
            user_properties={},
        )

        result = self.event_mesh_output.invoke(message, message.get_data("input"))

        self.assertIsNotNone(result)
        self.assertEqual(result["topic"], "default/topic")
        self.assertEqual(result["payload"], b'"File content"')

    def test_invoke_with_attach_file_as_payload_no_files(self):
        self.event_mesh_output.output_handlers["default"][
            "attach_file_as_payload"
        ] = True
        message = Message(
            payload={},
            user_properties={},
        )

        result = self.event_mesh_output.invoke(message, message.get_data("input"))

        self.assertIsNone(result)

    def test_invoke_with_no_text_field(self):
        message = Message(
            payload={"not_text": "This is not the text field"},
            user_properties={},
        )

        result = self.event_mesh_output.invoke(message, message.get_data("input"))

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
