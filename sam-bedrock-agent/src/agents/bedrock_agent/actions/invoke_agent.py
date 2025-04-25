"""Amazon Bedrock Agent invoke action."""

from solace_ai_connector.common.log import log
from typing import Dict, Any
from botocore.exceptions import ClientError
import base64
import json

from solace_agent_mesh.services.file_service import FileService, FS_PROTOCOL
from solace_agent_mesh.services.file_service.file_utils import starts_with_fs_url
from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo

# Limit: 5 files (10MB total size)
# Format: .pdf, .txt, .doc, .csv, .xls, .xlsx
MAX_FILE_LENGTH = 10485760  # 10MB
MAX_NUM_FILES = 5
SUPPORTED_FILE_TYPES = [
    ".pdf",
    ".txt",
    ".doc",
    ".csv",
    ".xls",
    ".xlsx",
]

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
        self.allow_files = action_config.get("allow_files", False)
        params = [
                {
                    "name": "input",
                    "desc": action_config.get("param_description", "Input to send to the action."),
                    "type": "string",
                    "required": True
                }
            ]
        if self.allow_files:
            params.append(
                {
                    "name": "files",
                    "desc": f"[Optional] Files to append to the prompt. Only {FS_PROTOCOL} URLs. Max 5 files, 10MB total. Supported types: {SUPPORTED_FILE_TYPES}",
                    "type": "array of strings",
                    "required": False
                }
            )
        
        super().__init__({
            "name": action_config["name"],
            "prompt_directive": action_config["description"],
            "params": params,
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
        files = params.get("files")
        session_id = meta.get("session_id")

        session_state = None
        if self.allow_files and files:
            file_service = FileService()

            if isinstance(files, str):
                try:
                    files = json.loads(files)
                except json.JSONDecodeError:
                    return ActionResponse(
                        message="Invalid JSON format for files parameter.",
                        error_info=ErrorInfo("Invalid JSON format for files parameter")
                    )

            if not isinstance(files, list):
                return ActionResponse(
                    message="Files parameter must be a list of file URLs.",
                    error_info=ErrorInfo("Invalid files parameter")
                )
            if len(files) > MAX_NUM_FILES:
                return ActionResponse(
                    message=f"Too many files. Maximum is {MAX_NUM_FILES}.",
                    error_info=ErrorInfo(f"Too many files. Maximum is {MAX_NUM_FILES}.")
                )
            for file in files:
                if not starts_with_fs_url(file):
                    return ActionResponse(
                        message=f"Invalid file URL: {file}",
                        error_info=ErrorInfo(f"Invalid file URL: {file}")
                    )
                
            byte_contents = []
            total_size = 0
            for url in files:
                file_name, _ = file_service.get_parsed_url(url)
                if not file_name.endswith(tuple(SUPPORTED_FILE_TYPES)):
                    return ActionResponse(
                        message=f"Unsupported file type: {file_name}. Supported types are: {SUPPORTED_FILE_TYPES}",
                        error_info=ErrorInfo(f"Unsupported file type: {file_name}")
                    )

                try:
                    buffer, _, meta = file_service.resolve_url(url, session_id, return_extra=True)
                    media_type = meta.get("mime_type")

                    if isinstance(buffer, str):
                        buffer = buffer.encode('utf-8')
                        media_type = "text/plain"
                        
                except Exception as e:
                    log.error("Error resolving file URL: %s", e)
                    return ActionResponse(
                        message=f"Error resolving file URL: {str(e)}",
                        error_info=ErrorInfo(str(e))
                    )
                
                total_size += len(buffer)
                if total_size > MAX_FILE_LENGTH:
                    return ActionResponse(
                        message=f"Total file size exceeds {MAX_FILE_LENGTH} bytes.",
                        error_info=ErrorInfo(f"Total file size exceeds {MAX_FILE_LENGTH} bytes.")
                    )

                byte_content = base64.b64encode(buffer).decode('utf-8')
                byte_contents.append({
                    'name': meta.get("name"),
                    'source': {
                        'byteContent': {
                            'data': byte_content,
                            'mediaType': media_type,
                        },
                        'sourceType': 'BYTE_CONTENT'
                    },
                    'useCase': 'CHAT'
                })

            session_state = {
                'files': byte_contents,
            }
            

        try:
            result =  self.bedrock_agent_runtime.invoke_agent(
                self.bedrock_agent_id,
                self.bedrock_agent_alias_id,
                session_id,
                user_input,
                session_state
            )
            return ActionResponse(message=result)
        except ClientError as e:
            log.error("Error invoking agent: %s", e)
            return ActionResponse(
                message=f"Error invoking bedrock agent {self.name}: {str(e)}",
                error_info=ErrorInfo(str(e))
            )

