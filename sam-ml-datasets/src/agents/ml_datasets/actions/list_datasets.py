"""Action for listing available ML datasets."""

from typing import Dict, Any
import json
import yaml
import datetime

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo
from solace_ai_connector.common.log import log


class ListDatasets(Action):
    """Action for listing all available ML datasets."""

    def __init__(self, **kwargs):
        """Initialize the action."""
        super().__init__(
            {
                "name": "list_datasets",
                "prompt_directive": (
                    "List all available ML datasets organized by type (sklearn, seaborn, synthetic). "
                    "Shows dataset names and brief descriptions for each type."
                ),
                "params": [
                    {
                        "name": "dataset_type",
                        "desc": "Filter by dataset type: 'sklearn', 'seaborn', 'synthetic', or 'all' (default)",
                        "type": "string",
                        "required": False,
                        "default": "all",
                    },
                    {
                        "name": "response_format",
                        "desc": "Format of the response: 'json' or 'yaml'",
                        "type": "string",
                        "required": False,
                        "default": "yaml",
                    },
                ],
                "required_scopes": ["<agent_name>:list_datasets:execute"],
            },
            **kwargs,
        )

    def invoke(self, params: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        """List available ML datasets.

        Args:
            params: Action parameters
            meta: Optional metadata

        Returns:
            ActionResponse containing the list of datasets
        """
        try:
            dataset_type = params.get("dataset_type", "all").lower()
            response_format = params.get("response_format", "yaml").lower()

            if response_format not in ["json", "yaml"]:
                raise ValueError("Invalid response format. Choose 'json' or 'yaml'")

            log.info("ml-datasets: Listing datasets with filter='%s'", dataset_type)

            # Get the dataset service
            dataset_service = self.get_agent().dataset_service

            # List available datasets
            available_datasets = dataset_service.list_available_datasets()

            # Filter datasets based on type
            if dataset_type == "all":
                datasets_to_show = available_datasets
            elif dataset_type in available_datasets:
                datasets_to_show = {dataset_type: available_datasets[dataset_type]}
            else:
                return ActionResponse(
                    message=f"Invalid dataset type '{dataset_type}'. Available types: {list(available_datasets.keys())}",
                    error_info=ErrorInfo(f"Invalid dataset type: {dataset_type}")
                )

            # Prepare response
            if response_format == "json":
                response_data = {}
                for dtype, datasets in datasets_to_show.items():
                    response_data[dtype] = {
                        "count": len(datasets),
                        "datasets": datasets
                    }
                content = json.dumps(response_data, indent=2)
                content_type = "application/json"
                filename_ext = "json"
            else:  # yaml
                response_data = {}
                for dtype, datasets in datasets_to_show.items():
                    response_data[dtype] = {
                        "count": len(datasets),
                        "datasets": datasets
                    }
                content = yaml.dump(response_data, default_flow_style=False, allow_unicode=True)
                content_type = "text/yaml"
                filename_ext = "yaml"

            # Create filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"available_datasets_{timestamp}.{filename_ext}"

            # Prepare response message
            response_message = f"**Available Datasets ({response_format.upper()} format)**\n\n"
            
            total_datasets = 0
            for dtype, datasets in datasets_to_show.items():
                response_message += f"**{dtype.upper()} Datasets ({len(datasets)}):**\n"
                for dataset in datasets:
                    response_message += f"- {dataset}\n"
                response_message += "\n"
                total_datasets += len(datasets)

            response_message += f"**Total: {total_datasets} datasets available**\n\n"
            response_message += f"Use `get_dataset` with `dataset_type` and `dataset_name` to retrieve specific datasets."

            # Create file response
            files = [{
                "filename": filename,
                "content": content,
                "content_type": content_type
            }]

            log.info("ml-datasets: Successfully listed %d datasets", total_datasets)

            return ActionResponse(
                message=response_message,
                files=files
            )

        except Exception as e:
            log.error("ml-datasets: Error listing datasets: %s", str(e))
            return ActionResponse(
                message=f"Error listing datasets: {str(e)}",
                error_info=ErrorInfo(str(e)),
            )

    def _get_type_description(self, dataset_type: str) -> str:
        """Get description for dataset type.
        
        Args:
            dataset_type: Type of dataset
            
        Returns:
            Description string
        """
        descriptions = {
            "sklearn": "Classic machine learning datasets from scikit-learn library, including iris, wine, breast cancer, and more. Great for learning and benchmarking ML algorithms.",
            "seaborn": "Real-world datasets used in statistical visualization examples. Includes tips, flights, titanic, and other datasets commonly used in data analysis tutorials.",
            "synthetic": "Programmatically generated datasets for specific ML tasks. Useful for testing algorithms with known ground truth or creating datasets with specific characteristics."
        }
        return descriptions.get(dataset_type, "Dataset collection")