"""Action description"""
from solace_ai_connector.common.log import log

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse
from solace_agent_mesh.services.file_service import FileService
import uuid
import os
import requests

class DrawAction(Action):
    def __init__(self, **kwargs):
        super().__init__(
            {
            "name": "draw",
            "prompt_directive": (
                "Draw a diagram or visualization using Mermaid.js syntax. It can"
                "process textual inputs to create flowcharts, sequence diagrams, "
                "Gantt charts, and other diagram types supported by Mermaid."
            ),
            "params": [
                {
                "name": "mermaid_code",
                "desc": "The input text using mermaid syntax to be rendered as a diagram.",
                "type": "string",
                }
            ],
            "required_scopes": ["mermaid:draw:create"],
            },
            **kwargs,
        )

    def invoke(self, params, meta={}) -> ActionResponse:
        log.debug("Doing mermaid draw action: %s", params["mermaid_code"])
        session_id = meta.get("session_id")
        return self.do_action(params["mermaid_code"], session_id)

    def do_action(self, mermaid_code, session_id) -> ActionResponse:
        # Generate the diagram as PNG
        output_filename = f"{str(uuid.uuid4())[:6]}_diagram.png"
        mermaid_response = generate_mermaid_png(mermaid_code, output_filename, session_id)

        # Check if error is raised when generating mermaid png
        if isinstance(mermaid_response, str):
            return ActionResponse(message=mermaid_response)

        return ActionResponse(files=[mermaid_response])

def generate_mermaid_png(mermaid_code, output_file, session_id):
    try:
        # Do a POST to the mermaid server to get a PNG
        mermaidServerService = os.getenv("MERMAID_SERVER_URL")
        
        # Handle URLs with or without trailing slash
        base_url = mermaidServerService.rstrip('/')
        request_url = f"{base_url}/generate?type=png"
        response = requests.post(request_url, data=mermaid_code)

        # Upload the PNG file to the file service
        file_service = FileService()
        image_meta = file_service.upload_from_buffer(
            response.content,
            file_name=output_file,
            session_id=session_id,
            data_source="Mermaid Agent - Draw Action",
        )
        return image_meta
    except Exception as e:
        log.error("Error generating Mermaid diagram: %s", str(e))
        return f"Error generating Mermaid diagram: {str(e)}"
