"""Data loader action for receiving data from other agents or loading from files."""

from typing import Dict, Any
import pandas as pd
import os
import json

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo
from solace_ai_connector.common.log import log


class DataLoaderAction(Action):
    """Action for receiving data from other agents or loading from files."""

    def __init__(self, **kwargs):
        """Initialize the action."""
        super().__init__(
            {
                "name": "load_data",
                "prompt_directive": (
                    "Receive data from other agents or load data from files for analysis. "
                    "This action supports collaborative workflows where data comes from SQL agents, "
                    "database agents, or other data sources. You can also load CSV, JSON, Excel, or Parquet files."
                ),
                "params": [
                    {
                        "name": "load_type",
                        "desc": "Type of data loading operation",
                        "type": "string",
                        "required": True,
                        "enum": ["file", "agent_data", "json_data"],
                    },
                    {
                        "name": "file_path",
                        "desc": "Path to the data file (required when load_type is 'file')",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "file_format",
                        "desc": "Format of the data file (csv, json, excel, parquet)",
                        "type": "string",
                        "required": False,
                        "enum": ["csv", "json", "excel", "parquet"],
                        "default": "csv",
                    },
                    {
                        "name": "json_data",
                        "desc": "JSON string containing data (required when load_type is 'json_data')",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "source_agent",
                        "desc": "Name of the agent that provided the data",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "description",
                        "desc": "Description of the data being loaded",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "help",
                        "desc": "Show help information for collaborative workflows",
                        "type": "boolean",
                        "required": False,
                        "default": False,
                    },
                ],
            }
        )

    def invoke(self, request_data: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        """Execute the data loader action."""
        try:
            # Get parameters
            load_type = request_data.get("load_type")
            file_path = request_data.get("file_path", "")
            file_format = request_data.get("file_format", "csv")
            json_data = request_data.get("json_data", "")
            source_agent = request_data.get("source_agent", "")
            description = request_data.get("description", "")
            help_requested = request_data.get("help", False)

            # Show help if requested
            if help_requested:
                agent = self.get_agent()
                help_text = agent.get_collaborative_workflow_help()
                return ActionResponse(message=help_text)

            # Log the request for debugging
            log.info("ml-pandas: Data loader action called with load_type: %s", load_type)
            if load_type == "json_data":
                log.info("ml-pandas: JSON data length: %d characters", len(json_data) if json_data else 0)
                log.info("ml-pandas: Source agent: %s", source_agent)
                log.info("ml-pandas: Description: %s", description)

            # Validate load_type
            valid_load_types = ["file", "agent_data", "json_data"]
            if load_type not in valid_load_types:
                return ActionResponse(
                    message=f"Invalid load_type '{load_type}'. Valid types are: {', '.join(valid_load_types)}",
                    error_info=ErrorInfo(f"Invalid load_type '{load_type}'. Valid types are: {', '.join(valid_load_types)}")
                )

            # Get the agent
            agent = self.get_agent()

            if load_type == "file":
                # Load data from file
                if not file_path:
                    return ActionResponse(
                        message="file_path is required when load_type is 'file'",
                        error_info=ErrorInfo("file_path is required when load_type is 'file'")
                    )

                # Check if this is an amfs:// URL (Agent Mesh File System)
                if file_path.startswith("amfs://"):
                    log.warning("ml-pandas: Received amfs:// URL: %s", file_path)
                    return ActionResponse(
                        message=f"File not found: {file_path}. This appears to be a reference to a file created by another agent that no longer exists. Please use the 'json_data' parameter to receive data directly from other agents, or ensure the file exists in the Agent Mesh File System.",
                        error_info=ErrorInfo(f"amfs:// file not found: {file_path}")
                    )

                # Validate that file_path is actually a file path, not data content
                if ',' in file_path and '\n' in file_path:
                    log.warning("ml-pandas: file_path appears to contain CSV data instead of a file path")
                    
                    # Try to convert the CSV data to JSON format
                    try:
                        csv_data = file_path.strip()
                        lines = csv_data.split('\n')
                        if len(lines) >= 2:
                            headers = lines[0].split(',')
                            records = []
                            for line in lines[1:]:
                                if line.strip():
                                    values = line.split(',')
                                    if len(values) == len(headers):
                                        record = dict(zip(headers, values))
                                        records.append(record)
                            
                            if records:
                                json_data = json.dumps(records)
                                log.info("ml-pandas: Converted CSV data to JSON format with %d records", len(records))
                                return ActionResponse(
                                    message=f"Detected CSV data in file_path parameter. Converted to JSON format with {len(records)} records. Please use load_type='json_data' and json_data parameter instead.",
                                    error_info=ErrorInfo("file_path contains CSV data - use json_data parameter")
                                )
                    except Exception as e:
                        log.debug("ml-pandas: Failed to convert CSV data to JSON: %s", str(e))
                    
                    return ActionResponse(
                        message="The file_path parameter appears to contain CSV data instead of a file path. If you have CSV data, use load_type='json_data' and pass the data as json_data parameter.",
                        error_info=ErrorInfo("file_path contains data content instead of file path")
                    )

                # Check if file exists
                if not os.path.exists(file_path):
                    return ActionResponse(
                        message=f"File not found: {file_path}. Please provide a valid file path or use load_type='json_data' to receive data directly from other agents.",
                        error_info=ErrorInfo(f"File not found: {file_path}")
                    )

                # Validate file format
                valid_formats = ["csv", "json", "excel", "parquet"]
                if file_format not in valid_formats:
                    return ActionResponse(
                        message=f"Invalid file_format '{file_format}'. Valid formats are: {', '.join(valid_formats)}",
                        error_info=ErrorInfo(f"Invalid file_format '{file_format}'. Valid formats are: {', '.join(valid_formats)}")
                    )

                log.info("ml-pandas: Loading data from file: %s (format: %s)", file_path, file_format)

                # Load the data
                data = agent.load_data_from_file(file_path, file_format)
                
                result = {
                    "load_type": "file",
                    "file_path": file_path,
                    "file_format": file_format,
                    "data_shape": data.shape,
                    "columns": data.columns.tolist(),
                    "data_types": data.dtypes.to_dict(),
                    "sample_data": data.head(5).to_dict('records')
                }

            elif load_type == "json_data":
                # Load data from JSON string
                if not json_data or not json_data.strip():
                    return ActionResponse(
                        message="json_data parameter is required and cannot be empty when load_type is 'json_data'",
                        error_info=ErrorInfo("json_data parameter is required and cannot be empty when load_type is 'json_data'")
                    )

                # Validate JSON format before processing
                try:
                    import json
                    json.loads(json_data)  # Test if it's valid JSON
                except json.JSONDecodeError as e:
                    return ActionResponse(
                        message=f"Invalid JSON format in json_data: {str(e)}",
                        error_info=ErrorInfo(f"Invalid JSON format in json_data: {str(e)}")
                    )

                # Load the data
                try:
                    data = agent.receive_data_from_json(json_data, source_agent, description)
                except ValueError as e:
                    return ActionResponse(
                        message=f"Failed to process JSON data: {str(e)}",
                        error_info=ErrorInfo(f"Failed to process JSON data: {str(e)}")
                    )
                
                result = {
                    "load_type": "json_data",
                    "source_agent": source_agent,
                    "description": description,
                    "data_shape": data.shape,
                    "columns": data.columns.tolist(),
                    "data_types": data.dtypes.to_dict(),
                    "sample_data": data.head(5).to_dict('records')
                }

            else:  # load_type == "agent_data"
                # This would be used when receiving DataFrame directly from another agent
                # For now, we'll require JSON data for agent-to-agent communication
                return ActionResponse(
                    message="agent_data load_type requires direct DataFrame transfer. Use 'json_data' for agent-to-agent communication.",
                    error_info=ErrorInfo("agent_data load_type requires direct DataFrame transfer. Use 'json_data' for agent-to-agent communication.")
                )

            # Clean result for JSON serialization
            clean_result = agent.get_data_service().clean_data_for_json(result)

            # Save results to file
            filename = f"data_loaded_{load_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            saved_path = agent.get_data_service().save_results(clean_result, filename)

            response_text = self._format_response(clean_result, saved_path)

            return ActionResponse(
                message=response_text
            )

        except Exception as e:
            log.error("ml-pandas: Error in data loader action: %s", str(e))
            return ActionResponse(
                message=f"Failed to load data: {str(e)}",
                error_info=ErrorInfo(f"Failed to load data: {str(e)}")
            )

    def _format_response(self, result: Dict[str, Any], saved_path: str) -> str:
        """Format the response text."""
        response_lines = [
            f"# Data Loaded Successfully",
            f"**Load Type:** {result['load_type'].title()}",
            ""
        ]

        if result["load_type"] == "file":
            response_lines.extend([
                f"**File Path:** {result['file_path']}",
                f"**File Format:** {result['file_format']}",
            ])
        elif result["load_type"] == "json_data":
            response_lines.extend([
                f"**Source Agent:** {result['source_agent'] or 'External source'}",
                f"**Description:** {result['description'] or 'No description provided'}",
            ])

        response_lines.extend([
            "",
            f"**Data Shape:** {result['data_shape'][0]:,} rows × {result['data_shape'][1]} columns",
            f"**Columns:** {len(result['columns'])}",
            ""
        ])

        # Show column names
        response_lines.append("### Available Columns:")
        for i, col in enumerate(result['columns'], 1):
            col_type = result['data_types'].get(col, 'unknown')
            response_lines.append(f"{i}. **{col}** ({col_type})")
        response_lines.append("")

        # Show sample data
        if "sample_data" in result and result["sample_data"]:
            response_lines.extend([
                "### Sample Data (First 5 rows):",
                ""
            ])
            
            for i, row in enumerate(result["sample_data"], 1):
                response_lines.append(f"**Row {i}:**")
                for key, value in row.items():
                    response_lines.append(f"  - {key}: {value}")
                response_lines.append("")

        response_lines.extend([
            "---",
            f"**Results saved to:** `{saved_path}`",
            "",
            "💡 **Next steps:** Use 'summarize_data' for quick summaries or 'data_analysis' for detailed analysis"
        ])

        return "\n".join(response_lines) 