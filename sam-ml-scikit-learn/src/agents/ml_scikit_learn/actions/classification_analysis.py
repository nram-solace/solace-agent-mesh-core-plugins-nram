"""Action for performing classification analysis."""

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

class ClassificationAnalysis(Action):
    """Action for performing classification analysis."""

    def __init__(self, **kwargs):
        super().__init__(
            {
                "name": "classification_analysis",
                "prompt_directive": (
                    "Perform classification analysis using the specified model type. "
                    "Supports logistic regression, random forest, SVC, KNN, naive bayes, and more. "
                    "Returns model metrics, predictions, confusion matrix, and feature importance."
                ),
                "params": [
                    {"name": "data_source", "desc": "Type of data source (csv, excel, json, parquet)", "type": "string", "required": False, "default": "csv"},
                    {"name": "file_path", "desc": "Path to the data file", "type": "string", "required": True},
                    {"name": "target_column", "desc": "Target column for classification", "type": "string", "required": True},
                    {"name": "feature_columns", "desc": "List of feature columns", "type": "list", "required": True},
                    {"name": "model_type", "desc": "Type of classification model", "type": "string", "required": False, "default": "random_forest"},
                    {"name": "test_size", "desc": "Test set size (0-1)", "type": "float", "required": False, "default": 0.2},
                    {"name": "hyperparameter_tuning", "desc": "Enable hyperparameter tuning", "type": "boolean", "required": False, "default": False},
                    {"name": "response_format", "desc": "Format of the response (yaml, json)", "type": "string", "required": False, "default": "yaml"}
                ],
                "required_scopes": ["<agent_name>:classification_analysis:execute"],
            },
            **kwargs,
        )

    def invoke(self, params: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        try:
            data_source = params.get("data_source", "csv")
            file_path = params.get("file_path")
            target_column = params.get("target_column")
            feature_columns = params.get("feature_columns", [])
            model_type = params.get("model_type", "random_forest")
            test_size = params.get("test_size", 0.2)
            hyperparameter_tuning = params.get("hyperparameter_tuning", False)
            response_format = params.get("response_format", "yaml")

            if not file_path or not target_column or not feature_columns:
                raise ValueError("file_path, target_column, and feature_columns are required")

            agent = self.get_agent()
            df = agent.load_data(data_source, file_path)
            agent.validate_data(df, target_column, feature_columns)
            ml_service = agent.get_ml_service()
            X, y, preprocessing_info = ml_service.preprocess_data(df, target_column, feature_columns, task_type="classification")
            results = ml_service.train_classification_model(X, y, model_type, test_size, hyperparameter_tuning)

            # Prepare response
            if response_format.lower() == "json":
                content = json.dumps(results["metrics"], indent=2, default=str)
                file = InlineFile(filename="classification_metrics.json", content=content, content_type="application/json")
            else:
                content = yaml.dump(results["metrics"], default_flow_style=False, allow_unicode=True)
                file = InlineFile(filename="classification_metrics.yaml", content=content, content_type="text/yaml")

            return ActionResponse(
                message=f"Classification analysis completed. Test Accuracy: {results['metrics']['test_accuracy']:.3f}",
                files=[file],
                metadata={"model_type": model_type, "test_accuracy": results["metrics"]["test_accuracy"]}
            )
        except Exception as e:
            log.error("Error in classification analysis: %s", str(e))
            return ActionResponse(
                message=f"Error performing classification analysis: {str(e)}",
                error_info=ErrorInfo(str(e)),
            ) 