"""Action for listing available ML datasets."""

from typing import Dict, Any
import json
import yaml

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

            log.info(f"ml-datasets: Listing datasets with filter='{dataset_type}'")

            # Get the dataset service
            dataset_service = self.get_agent().dataset_service
            available_datasets = dataset_service.list_available_datasets()

            # Filter by type if specified
            if dataset_type != "all" and dataset_type in available_datasets:
                filtered_datasets = {dataset_type: available_datasets[dataset_type]}
            elif dataset_type != "all":
                raise ValueError(f"Invalid dataset type: {dataset_type}. Available types: {list(available_datasets.keys())}")
            else:
                filtered_datasets = available_datasets

            # Add descriptions for better understanding
            dataset_info = {}
            for dtype, datasets in filtered_datasets.items():
                dataset_info[dtype] = {
                    "description": self._get_type_description(dtype),
                    "count": len(datasets),
                    "datasets": datasets
                }

            # Format response
            if response_format == "json":
                content = json.dumps(dataset_info, indent=2)
                content_type = "application/json"
            else:  # yaml
                content = yaml.dump(dataset_info, default_flow_style=False, allow_unicode=True)
                content_type = "text/yaml"

            # Create response message
            response_message = "**Available ML Datasets**\n\n"
            
            total_datasets = sum(info["count"] for info in dataset_info.values())
            response_message += f"Total datasets available: {total_datasets}\n\n"

            for dtype, info in dataset_info.items():
                response_message += f"**{dtype.upper()} Datasets ({info['count']} available):**\n"
                response_message += f"{info['description']}\n"
                
                # List first 10 datasets inline, rest in file
                datasets_to_show = info['datasets'][:10]
                response_message += f"Available: {', '.join(datasets_to_show)}"
                
                if len(info['datasets']) > 10:
                    response_message += f" (and {len(info['datasets']) - 10} more - see file for complete list)"
                
                response_message += "\n\n"

            response_message += "**Usage Examples:**\n"
            response_message += "- Get sklearn iris dataset: `get_dataset dataset_type=sklearn dataset_name=iris`\n"
            response_message += "- Get seaborn tips dataset: `get_dataset dataset_type=seaborn dataset_name=tips`\n"
            response_message += "- Generate synthetic classification data: `get_dataset dataset_type=synthetic dataset_name=classification`\n"
            response_message += "- Generate synthetic data with custom parameters: `get_dataset dataset_type=synthetic dataset_name=classification synthetic_params='{\"n_features\": 6, \"n_classes\": 3}'`"

            # Create file
            filename = f"available_datasets.{response_format}"
            files = [{
                "filename": filename,
                "content": content,
                "content_type": content_type
            }]

            log.info(f"ml-datasets: Successfully listed {total_datasets} datasets")

            return ActionResponse(
                message=response_message,
                files=files
            )

        except Exception as e:
            log.error(f"ml-datasets: Error listing datasets: {str(e)}")
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