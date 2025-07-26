"""Action for retrieving ML datasets."""

from typing import Dict, Any, Optional
import json
import yaml
import csv
import io
import datetime

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo
from solace_ai_connector.common.log import log


class GetDataset(Action):
    """Action for retrieving various ML datasets."""

    def __init__(self, **kwargs):
        """Initialize the action."""
        super().__init__(
            {
                "name": "get_dataset",
                "prompt_directive": (
                    "Retrieve ML datasets including sklearn datasets (iris, wine, breast_cancer, etc.), "
                    "seaborn datasets (tips, flights, titanic, etc.), and synthetic datasets. "
                    "Returns the dataset as a DataFrame with metadata. "
                    "Limited to 100 records by default for efficiency."
                ),
                "params": [
                    {
                        "name": "dataset_type",
                        "desc": "Type of dataset: 'sklearn', 'seaborn', or 'synthetic'",
                        "type": "string",
                        "required": True,
                    },
                    {
                        "name": "dataset_name",
                        "desc": "Name of the dataset to retrieve (e.g., 'iris', 'tips', 'classification')",
                        "type": "string",
                        "required": True,
                    },
                    {
                        "name": "max_records",
                        "desc": "Maximum number of records to return (default: 100)",
                        "type": "integer",
                        "required": False,
                        "default": 100,
                    },
                    {
                        "name": "response_format",
                        "desc": "Format of the response: 'json', 'yaml', or 'csv'",
                        "type": "string",
                        "required": False,
                        "default": "json",
                    },
                    {
                        "name": "include_metadata",
                        "desc": "Whether to include dataset metadata in the response",
                        "type": "boolean",
                        "required": False,
                        "default": True,
                    },
                    {
                        "name": "synthetic_params",
                        "desc": "Additional parameters for synthetic datasets (JSON string)",
                        "type": "string",
                        "required": False,
                    },
                ],
                "required_scopes": ["<agent_name>:get_dataset:execute"],
            },
            **kwargs,
        )

    def invoke(self, params: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        """Retrieve the requested ML dataset.

        Args:
            params: Action parameters
            meta: Optional metadata

        Returns:
            ActionResponse containing the dataset
        """
        try:
            dataset_type = params.get("dataset_type", "").lower()
            dataset_name = params.get("dataset_name", "")
            max_records_raw = params.get("max_records", 100)
            response_format = params.get("response_format", "json").lower()
            include_metadata_raw = params.get("include_metadata", True)
            synthetic_params_str = params.get("synthetic_params", "{}")

            if not dataset_type:
                raise ValueError("dataset_type is required")
            
            if not dataset_name:
                raise ValueError("dataset_name is required")

            # Convert max_records to integer
            try:
                max_records = int(max_records_raw)
                if max_records <= 0:
                    raise ValueError("max_records must be a positive integer")
            except (ValueError, TypeError):
                raise ValueError(f"max_records must be a positive integer, got: {max_records_raw}")

            # Convert include_metadata to boolean
            if isinstance(include_metadata_raw, str):
                include_metadata = include_metadata_raw.lower() in ['true', '1', 'yes', 'on']
            else:
                include_metadata = bool(include_metadata_raw)

            if response_format not in ["json", "yaml", "csv"]:
                raise ValueError("Invalid response format. Choose 'json', 'yaml', or 'csv'")

            # Parse synthetic parameters if provided
            synthetic_params = {}
            if synthetic_params_str:
                try:
                    synthetic_params = json.loads(synthetic_params_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in synthetic_params: {str(e)}")

            log.info("Retrieving %s dataset '%s' with max_records=%d", dataset_type, dataset_name, max_records)

            # Get the dataset service
            dataset_service = self.get_agent().dataset_service

            # Retrieve the dataset based on type
            if dataset_type == "sklearn":
                df, metadata = dataset_service.get_sklearn_dataset(dataset_name, max_records)
            elif dataset_type == "seaborn":
                df, metadata = dataset_service.get_seaborn_dataset(dataset_name, max_records)
            elif dataset_type == "synthetic":
                df, metadata = dataset_service.generate_synthetic_dataset(
                    dataset_name, max_records, **synthetic_params
                )
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")

            log.info("Successfully retrieved dataset with %d records and %d columns", len(df), len(df.columns))

            # Convert DataFrame to the requested format
            if response_format == "csv":
                content = df.to_csv(index=False)
                content_type = "text/csv"
                filename_ext = "csv"
            elif response_format == "yaml":
                data_dict = df.to_dict('records')
                content = yaml.dump(data_dict, default_flow_style=False, allow_unicode=True)
                content_type = "text/yaml"
                filename_ext = "yaml"
            else:  # json
                data_dict = df.to_dict('records')
                content = json.dumps(data_dict, indent=2, default=str)
                content_type = "application/json"
                filename_ext = "json"

            # Create filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{dataset_type}_{dataset_name}_{timestamp}.{filename_ext}"

            # Prepare response message
            response_message = f"**{dataset_type.title()} Dataset: {dataset_name}**\n\n"
            response_message += f"Records: {len(df)}\n"
            response_message += f"Features: {len(df.columns)}\n"
            response_message += f"Format: {response_format.upper()}\n"

            if include_metadata and metadata:
                response_message += "\n**Metadata:**\n"
                for key, value in metadata.items():
                    if key == 'description' and len(str(value)) > 200:
                        # Truncate long descriptions
                        response_message += f"- {key}: {str(value)[:200]}...\n"
                    else:
                        response_message += f"- {key}: {value}\n"

            response_message += f"\n**Dataset preview (first 5 rows):**\n"
            preview_df = df.head(5)
            if response_format == "csv":
                response_message += f"```csv\n{preview_df.to_csv(index=False)}\n```"
            elif response_format == "yaml":
                preview_data = preview_df.to_dict('records')
                preview_yaml = yaml.dump(preview_data, default_flow_style=False, allow_unicode=True)
                response_message += f"```yaml\n{preview_yaml}\n```"
            else:  # json
                preview_data = preview_df.to_dict('records')
                preview_json = json.dumps(preview_data, indent=2, default=str)
                response_message += f"```json\n{preview_json}\n```"

            # Create file response
            files = [{
                "filename": filename,
                "content": content,
                "content_type": content_type
            }]

            # Add metadata file if requested
            if include_metadata and metadata:
                metadata_content = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
                metadata_filename = f"{dataset_type}_{dataset_name}_{timestamp}_metadata.yaml"
                files.append({
                    "filename": metadata_filename,
                    "content": metadata_content,
                    "content_type": "text/yaml"
                })

            log.info("Successfully created response with %d files", len(files))

            return ActionResponse(
                message=response_message,
                files=files
            )

        except Exception as e:
            log.error("Error retrieving dataset: %s", str(e))
            return ActionResponse(
                message=f"Error retrieving dataset: {str(e)}",
                error_info=ErrorInfo(str(e)),
            )