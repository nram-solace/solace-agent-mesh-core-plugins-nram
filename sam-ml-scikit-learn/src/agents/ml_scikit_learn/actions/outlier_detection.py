"""Action for performing outlier detection."""

from typing import Dict, Any, List
import json
import yaml
from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import (
    ActionResponse,
    ErrorInfo,
    InlineFile,
)
from solace_ai_connector.common.log import log

class OutlierDetection(Action):
    """Action for performing outlier detection."""

    def __init__(self, **kwargs):
        super().__init__(
            {
                "name": "outlier_detection",
                "prompt_directive": (
                    "Detect outliers in the dataset using the specified method. "
                    "Supports isolation forest, local outlier factor, one-class SVM, and elliptic envelope. "
                    "Returns indices and statistics of detected outliers."
                ),
                "params": [
                    {"name": "data_source", "desc": "Type of data source (csv, excel, json, parquet)", "type": "string", "required": False, "default": "csv"},
                    {"name": "file_path", "desc": "Path to the data file", "type": "string", "required": True},
                    {"name": "feature_columns", "desc": "List of feature columns to use for outlier detection", "type": "list", "required": True},
                    {"name": "method", "desc": "Outlier detection method", "type": "string", "required": False, "default": "isolation_forest"},
                    {"name": "contamination", "desc": "Expected proportion of outliers", "type": "float", "required": False, "default": 0.1},
                    {"name": "response_format", "desc": "Format of the response (yaml, json)", "type": "string", "required": False, "default": "yaml"}
                ],
                "required_scopes": ["<agent_name>:outlier_detection:execute"],
            },
            **kwargs,
        )

    def invoke(self, params: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        try:
            data_source = params.get("data_source", "csv")
            file_path = params.get("file_path")
            feature_columns = params.get("feature_columns", [])
            method = params.get("method", "isolation_forest")
            contamination = params.get("contamination", 0.1)
            response_format = params.get("response_format", "yaml")

            if not file_path or not feature_columns:
                raise ValueError("file_path and feature_columns are required")

            agent = self.get_agent()
            df = agent.load_data(data_source, file_path)
            agent.validate_data(df, feature_columns=feature_columns)
            ml_service = agent.get_ml_service()
            X = df[feature_columns].values
            results = ml_service.detect_outliers(X, method, contamination)

            # Prepare response
            if response_format.lower() == "json":
                content = json.dumps(results, indent=2, default=str)
                file = InlineFile(filename="outlier_detection.json", content=content, content_type="application/json")
            else:
                content = yaml.dump(results, default_flow_style=False, allow_unicode=True)
                file = InlineFile(filename="outlier_detection.yaml", content=content, content_type="text/yaml")

            return ActionResponse(
                message=f"Outlier detection completed. Number of outliers: {results['n_outliers']}",
                files=[file],
                metadata={"method": method, "n_outliers": results["n_outliers"]}
            )
        except Exception as e:
            log.error("Error in outlier detection: %s", str(e))
            return ActionResponse(
                message=f"Error performing outlier detection: {str(e)}",
                error_info=ErrorInfo(str(e)),
            ) 